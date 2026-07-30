"""End-to-end tests against a real EMR percentile file from customer 11901, app 13052.

This is the actual production output of the Metica nightly job for a no-AppLovin
customer. The schema (`user.platform`, `user.adformat`, `count`, `total_revenue`,
`mean`, `min`, `max`, `revenue_per_traffic`, `cumulative_traffic`, `traffic_percentage`,
`pricing_strategy` alongside the percentile columns) is the contract the client
must consume robustly.

Note: the columns in this file are ``[p40, p70, p85, p94, p99]`` — the EMR
``select_equally_spaced_percentiles`` step already chose them based on the app's
ad-unit count. The pre-refactor client (hardcoded p10–p90) would have errored on
this file; these tests pin down that the refactored client handles it correctly.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "percentiles_11901_13052_android_reward.json"
S3_KEY = "bid-floor-optimisation/applovin/percentile/11901/13052/2026-05-11_android_reward.json"

pytestmark = pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason=(
        "Real-data fixture is local-only (not committed — see tests/fixtures/ in .gitignore). "
        "Drop a production percentile JSON at tests/fixtures/ to enable these tests."
    ),
)


def _fixture_payload():
    return FIXTURE_PATH.read_text()


def _build_s3_mock():
    mock_s3 = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Contents": [{"Key": S3_KEY, "LastModified": pd.Timestamp("2026-05-11")}]}
    ]
    mock_s3.get_paginator.return_value = paginator
    body = _fixture_payload().encode("utf-8")
    mock_s3.get_object.return_value = {"Body": SimpleNamespace(read=lambda: body)}
    return mock_s3


def _build_applovin_mock(ad_unit_count):
    instance = MagicMock()
    instance.get_ad_units.return_value = [
        {
            "id": f"au{i}",
            "name": f"metica_android_reward_{i}",
            "ad_format": "reward",
            "package_name": "com.app",
            "platform": "android",
        }
        for i in range(1, ad_unit_count + 2)
    ]
    return instance


def _argv():
    return [
        "prog",
        "--customer-id", "11901",
        "--app-id", "13052",
        "--ad-type", "reward",
        "--platform", "android",
        "--applovin-api-key", "k",
        "--aws-access-key-id", "ak",
        "--aws-secret-access-key", "sk",
        "--aws-region", "eu-west-1",
        "--s3-bucket", "com.metica.prod-eu.dplat.artifacts",
        "--package-name", "com.app",
    ]


def _bid_floors_by_ad_unit(applovin_mock):
    out = {}
    for call in applovin_mock.update_ad_unit.call_args_list:
        ad_unit_id = call.kwargs.get("ad_unit_id") or call.args[0]
        out[ad_unit_id] = call.kwargs["bid_floors"]
    return out


def _cpm_for(bid_floors, country):
    for bf in bid_floors:
        if country in bf["countries"]["values"]:
            return bf["cpm"]
    return None


def test_fixture_loads_and_columns_discovered():
    """Sanity check: the real EMR file parses and exposes the expected percentile columns."""
    from bid_optim_etl_py.helpers.data_helpers import discover_percentile_columns

    df = pd.read_json(FIXTURE_PATH, orient="records")
    assert discover_percentile_columns(df) == ["p40", "p70", "p85", "p94", "p99"]
    assert len(df) == 36
    assert "pricing_strategy" in df.columns
    assert "user.platform" in df.columns


@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_real_fixture_5_ad_units_1to1_mapping(mock_boto_sess, mock_client_cls):
    """5 ad units + 5 percentile columns -> ad unit i maps to column i.

    Specific assertions for the ``us`` row pin down exact post-CPM values.
    """
    from scripts.update_bid_floor_values import main

    mock_boto_sess.return_value.client.return_value = _build_s3_mock()
    applovin = _build_applovin_mock(ad_unit_count=5)
    mock_client_cls.return_value = applovin

    with patch("sys.argv", _argv()):
        main()

    assert applovin.update_ad_unit.call_count == 5
    floors = _bid_floors_by_ad_unit(applovin)
    assert sorted(floors.keys()) == ["au2", "au3", "au4", "au5", "au6"]

    # us row: p40=0.011883, p70=0.017631, p85=0.022887, p94=0.135490, p99=0.178210
    assert _cpm_for(floors["au2"], "us") == "11.88"
    assert _cpm_for(floors["au3"], "us") == "17.63"
    assert _cpm_for(floors["au4"], "us") == "22.89"
    assert _cpm_for(floors["au5"], "us") == "135.49"
    assert _cpm_for(floors["au6"], "us") == "178.21"

    assert _cpm_for(floors["au2"], "vn") == "0.02"
    assert _cpm_for(floors["au6"], "vn") == "1.18"
    assert _cpm_for(floors["au2"], "kz") == "165.31"
    assert _cpm_for(floors["au6"], "kz") == "165.31"
    assert _cpm_for(floors["au2"], "tr") == "0.01"
    assert _cpm_for(floors["au3"], "tr") == "3.21"

    fixture_countries = {row["user.country"] for row in json.loads(_fixture_payload())}
    for au_id in floors:
        configured = set()
        for bf in floors[au_id]:
            configured.update(bf["countries"]["values"])
        assert configured == fixture_countries, f"{au_id} missing countries: {fixture_countries - configured}"


@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_real_fixture_3_ad_units_picks_upper_percentiles(mock_boto_sess, mock_client_cls):
    """3 ad units + 5 percentiles -> indices [1, 3, 4] -> p70, p94, p99 (favouring higher floors)."""
    from scripts.update_bid_floor_values import main

    mock_boto_sess.return_value.client.return_value = _build_s3_mock()
    applovin = _build_applovin_mock(ad_unit_count=3)
    mock_client_cls.return_value = applovin

    with patch("sys.argv", _argv()):
        main()

    assert applovin.update_ad_unit.call_count == 3
    floors = _bid_floors_by_ad_unit(applovin)

    assert _cpm_for(floors["au2"], "us") == "17.63"
    assert _cpm_for(floors["au3"], "us") == "135.49"
    assert _cpm_for(floors["au4"], "us") == "178.21"


@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_real_fixture_metadata_columns_ignored(mock_boto_sess, mock_client_cls):
    """The fixture has 12 metadata columns alongside the 5 percentile columns.

    Only the percentile columns drive the bid-floor CPMs. Verify by confirming
    no ad unit gets an unreasonably-large CPM (which would indicate a stray
    ``count`` or ``cumulative_traffic`` column was multiplied by 1000).
    """
    from scripts.update_bid_floor_values import main

    mock_boto_sess.return_value.client.return_value = _build_s3_mock()
    applovin = _build_applovin_mock(ad_unit_count=5)
    mock_client_cls.return_value = applovin

    with patch("sys.argv", _argv()):
        main()

    floors = _bid_floors_by_ad_unit(applovin)
    for au_id, bid_floors in floors.items():
        for bf in bid_floors:
            assert float(bf["cpm"]) <= 499.0, (
                f"{au_id}: cpm {bf['cpm']} exceeds MAX_CPM-1 (=499). "
                "Suggests a non-percentile column leaked into the price computation."
            )
