import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


STANDARD_GRID = ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]
AGGRESSIVE_GRID = ["p40", "p50", "p60", "p70", "p80", "p83", "p86", "p90", "p94", "p97"]
ULTRA_AGGRESSIVE_GRID = ["p50", "p70", "p90", "p95", "p99"]


def _make_s3_list_response(keys):
    return {
        "Contents": [
            {"Key": k, "LastModified": pd.Timestamp("2025-01-01")}
            for k in keys
        ]
    }


def _percentiles_json(countries_to_values, percentile_columns):
    rows = []
    for country, values in countries_to_values.items():
        row = {"user.country": country}
        row.update(dict(zip(percentile_columns, values)))
        rows.append(row)
    return json.dumps(rows)


def _build_s3_mock(percentile_json, list_keys=None):
    mock_s3_client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        _make_s3_list_response(
            list_keys
            or [
                "bid-floor-optimisation/applovin/percentile/1/2/2025-09-10_android_reward.json",
                "bid-floor-optimisation/applovin/percentile/1/2/2025-10-01_android_reward.json",
            ]
        )
    ]
    mock_s3_client.get_paginator.return_value = paginator
    body_bytes = percentile_json.encode("utf-8")
    mock_s3_client.get_object.return_value = {"Body": SimpleNamespace(read=lambda: body_bytes)}
    return mock_s3_client


def _build_applovin_mock(ad_unit_count, prefix="metica_android_reward"):
    """Create AppLovin client mock returning ``ad_unit_count`` metica ad units."""
    client_instance = MagicMock()
    units = [
        {
            "id": f"au{i}",
            "name": f"{prefix}_{i}",
            "ad_format": "reward",
            "package_name": "com.app",
            "platform": "android",
        }
        for i in range(1, ad_unit_count + 2)
    ]
    client_instance.get_ad_units.return_value = units
    return client_instance


def _argv():
    return [
        "prog",
        "--customer-id", "1",
        "--app-id", "2",
        "--ad-type", "reward",
        "--platform", "android",
        "--applovin-api-key", "k",
        "--aws-access-key-id", "ak",
        "--aws-secret-access-key", "sk",
        "--aws-region", "eu-west-1",
        "--s3-bucket", "com.metica.prod-eu.dplat.artifacts",
        "--package-name", "com.app",
    ]


def _bid_floors_from_calls(client_instance):
    """Return dict of ad_unit_id -> bid_floors list passed to update_ad_unit."""
    out = {}
    for call in client_instance.update_ad_unit.call_args_list:
        ad_unit_id = call.kwargs.get("ad_unit_id") or call.args[0]
        bid_floors = call.kwargs["bid_floors"]
        out[ad_unit_id] = bid_floors
    return out


@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_main_happy_path(mock_boto_sess, mock_client_cls, monkeypatch):
    from scripts.update_bid_floor_values import main

    mock_s3_client = _build_s3_mock(
        _percentiles_json(
            {
                "us": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
                "gb": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            },
            STANDARD_GRID,
        )
    )
    mock_session = MagicMock()
    mock_session.client.return_value = mock_s3_client
    mock_boto_sess.return_value = mock_session

    client_instance = _build_applovin_mock(ad_unit_count=2)
    mock_client_cls.return_value = client_instance

    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    with patch("sys.argv", _argv()):
        main()

    assert mock_s3_client.get_paginator.called
    mock_s3_client.get_object.assert_called_once()
    mock_s3_client.put_object.assert_called_once()
    assert client_instance.update_ad_unit.call_count >= 1


@patch("scripts.update_bid_floor_values.boto3.Session")
def test_errors_when_no_percentiles_found(mock_boto_sess, monkeypatch):
    from scripts.update_bid_floor_values import main

    mock_s3_client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": []}]
    mock_s3_client.get_paginator.return_value = paginator
    mock_session = MagicMock()
    mock_session.client.return_value = mock_s3_client
    mock_boto_sess.return_value = mock_session

    with patch("sys.argv", _argv()):
        with pytest.raises(RuntimeError, match="No percentiles JSON found"):
            main()


@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_main_aggressive_grid_e2e(mock_boto_sess, mock_client_cls):
    from scripts.update_bid_floor_values import main

    mock_s3_client = _build_s3_mock(
        _percentiles_json(
            {"us": [0.4, 0.5, 0.6, 0.7, 0.8, 0.83, 0.86, 0.9, 0.94, 0.97]},
            AGGRESSIVE_GRID,
        )
    )
    mock_boto_sess.return_value.client.return_value = mock_s3_client
    client_instance = _build_applovin_mock(ad_unit_count=5)
    mock_client_cls.return_value = client_instance

    with patch("sys.argv", _argv()):
        main()

    assert client_instance.update_ad_unit.call_count == 5
    floors = _bid_floors_from_calls(client_instance)
    assert {"au2", "au3", "au4", "au5", "au6"} == set(floors.keys())


@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_main_ultra_aggressive_grid_e2e(mock_boto_sess, mock_client_cls):
    from scripts.update_bid_floor_values import main

    mock_s3_client = _build_s3_mock(
        _percentiles_json(
            {"us": [0.5, 0.7, 0.9, 0.95, 0.99]},
            ULTRA_AGGRESSIVE_GRID,
        )
    )
    mock_boto_sess.return_value.client.return_value = mock_s3_client
    client_instance = _build_applovin_mock(ad_unit_count=5)
    mock_client_cls.return_value = client_instance

    with patch("sys.argv", _argv()):
        main()

    assert client_instance.update_ad_unit.call_count == 5


@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_main_more_percentiles_than_ad_units_e2e(mock_boto_sess, mock_client_cls):
    """File has 20 percentile columns, AppLovin has only 10 metica ad units.

    Verifies the equal-distribution mapping: ad units land at indices
    [1, 3, 5, ..., 19] -> percentile values [0.2, 0.4, ..., 2.0] -> CPMs [200, 400, ..., 2000].
    The cap at MAX_CPM-1 (=499) means CPMs >500 all become "499.00".
    """
    from scripts.update_bid_floor_values import main

    pcols_20 = [f"p{n}" for n in range(5, 101, 5)]
    values = [round(0.1 * i, 2) for i in range(1, 21)]
    mock_s3_client = _build_s3_mock(_percentiles_json({"us": values}, pcols_20))
    mock_boto_sess.return_value.client.return_value = mock_s3_client
    client_instance = _build_applovin_mock(ad_unit_count=10)
    mock_client_cls.return_value = client_instance

    with patch("sys.argv", _argv()):
        main()

    assert client_instance.update_ad_unit.call_count == 10
    floors = _bid_floors_from_calls(client_instance)

    cpms_per_unit = {
        au_id: bf_list[0]["cpm"] for au_id, bf_list in floors.items()
    }
    expected_pre_cap = [200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]
    expected_capped = [v if v <= 500 else 499.0 for v in expected_pre_cap]
    for i, value in enumerate(expected_capped):
        assert cpms_per_unit[f"au{i + 2}"] == f"{value:.2f}"


@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_main_clips_low_percentiles_to_floor_so_cpm_never_formats_as_zero(mock_boto_sess, mock_client_cls):
    """AppLovin rejects bid floors where cpm <= 0 after formatting. CPMs format
    via f"{cpm:.2f}", so any raw value < 0.005 renders as "0.00" — including
    plain positives like 0.0005. Lower-clipping to 0.01 covers this whole range
    in one shot (zeros, negatives, and small positives that would round down).
    """
    from scripts.update_bid_floor_values import main

    pcols = STANDARD_GRID
    mock_s3_client = _build_s3_mock(
        _percentiles_json(
            {"us": [0.0, -0.001, 0.00001, 0.0001, 0.001, 0.01, 0.02, 0.03, 0.04]},
            pcols,
        )
    )
    mock_boto_sess.return_value.client.return_value = mock_s3_client
    mock_client_cls.return_value = _build_applovin_mock(ad_unit_count=2)

    with patch("sys.argv", _argv()):
        main()

    floors = _bid_floors_from_calls(mock_client_cls.return_value)
    for au_id, bid_floors in floors.items():
        for bf in bid_floors:
            assert float(bf["cpm"]) >= 0.01, f"{au_id}: cpm={bf['cpm']} below 0.01 floor"
            assert bf["cpm"] != "0.00", f"{au_id}: cpm formatted as 0.00 — AppLovin rejects"


@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_main_more_ad_units_than_percentiles_e2e(mock_boto_sess, mock_client_cls):
    from scripts.update_bid_floor_values import main

    pcols_5 = ["p10", "p30", "p50", "p70", "p90"]
    mock_s3_client = _build_s3_mock(
        _percentiles_json({"us": [0.01, 0.03, 0.05, 0.07, 0.09]}, pcols_5)
    )
    mock_boto_sess.return_value.client.return_value = mock_s3_client
    client_instance = _build_applovin_mock(ad_unit_count=10)
    mock_client_cls.return_value = client_instance

    with patch("sys.argv", _argv()):
        main()

    assert client_instance.update_ad_unit.call_count == 5
    floors = _bid_floors_from_calls(client_instance)
    assert set(floors.keys()) == {"au2", "au3", "au4", "au5", "au6"}
    assert "au7" not in floors
    assert "au11" not in floors


@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_main_no_percentile_columns_raises(mock_boto_sess, mock_client_cls):
    from scripts.update_bid_floor_values import main

    payload = json.dumps([{"user.country": "us", "count": 1000}])
    mock_s3_client = _build_s3_mock(payload)
    mock_boto_sess.return_value.client.return_value = mock_s3_client
    mock_client_cls.return_value = _build_applovin_mock(ad_unit_count=2)

    with patch("sys.argv", _argv()):
        with pytest.raises(ValueError, match="No percentile columns"):
            main()


@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_main_no_metica_ad_units_raises(mock_boto_sess, mock_client_cls):
    from scripts.update_bid_floor_values import main

    mock_s3_client = _build_s3_mock(
        _percentiles_json(
            {"us": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]},
            STANDARD_GRID,
        )
    )
    mock_boto_sess.return_value.client.return_value = mock_s3_client
    client_instance = MagicMock()
    client_instance.get_ad_units.return_value = [
        {
            "id": "x",
            "name": "house_android_reward_2",
            "ad_format": "reward",
            "package_name": "com.app",
            "platform": "android",
        },
    ]
    mock_client_cls.return_value = client_instance

    with patch("sys.argv", _argv()):
        with pytest.raises(RuntimeError, match="No metica ad units"):
            main()


@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_main_uploads_ad_unit_configurations_to_s3(mock_boto_sess, mock_client_cls):
    from scripts.update_bid_floor_values import main

    mock_s3_client = _build_s3_mock(
        _percentiles_json(
            {"us": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]},
            STANDARD_GRID,
        )
    )
    mock_boto_sess.return_value.client.return_value = mock_s3_client
    mock_client_cls.return_value = _build_applovin_mock(ad_unit_count=2)

    with patch("sys.argv", _argv()):
        main()

    put_call = mock_s3_client.put_object.call_args
    assert put_call.kwargs["Bucket"] == "com.metica.prod-eu.dplat.artifacts"
    assert put_call.kwargs["Key"] == (
        "bid-floor-optimisation/applovin/percentile/1/2/uploads/"
        "ad_unit_configurations_android_reward.json"
    )


@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_main_smart_pricing_country_gets_low_cpm_e2e(mock_boto_sess, mock_client_cls):
    from scripts.update_bid_floor_values import main

    pcols = STANDARD_GRID
    mock_s3_client = _build_s3_mock(
        _percentiles_json(
            {
                "us": [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13],
                "in": [0.00001] * 9,
            },
            pcols,
        )
    )
    mock_boto_sess.return_value.client.return_value = mock_s3_client
    mock_client_cls.return_value = _build_applovin_mock(ad_unit_count=3)
    with patch("sys.argv", _argv()):
        main()

    floors = _bid_floors_from_calls(mock_client_cls.return_value)
    for au_id, bid_floors in floors.items():
        country_cpms = {}
        for bf in bid_floors:
            for country in bf["countries"]["values"]:
                country_cpms[country] = bf["cpm"]
        assert country_cpms.get("in") == "0.01", f"{au_id}: expected in=0.01, got {country_cpms.get('in')}"


@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_main_picks_latest_modified_percentile_file(mock_boto_sess, mock_client_cls):
    """The S3 prefix may contain many dated files; the script must pick the latest one."""
    from scripts.update_bid_floor_values import main

    older = "bid-floor-optimisation/applovin/percentile/1/2/2025-09-10_android_reward.json"
    newer = "bid-floor-optimisation/applovin/percentile/1/2/2025-10-01_android_reward.json"

    mock_s3_client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": older, "LastModified": pd.Timestamp("2025-09-10")},
                {"Key": newer, "LastModified": pd.Timestamp("2025-10-01")},
            ]
        }
    ]
    mock_s3_client.get_paginator.return_value = paginator
    body = _percentiles_json(
        {"us": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]},
        STANDARD_GRID,
    ).encode("utf-8")
    mock_s3_client.get_object.return_value = {"Body": SimpleNamespace(read=lambda: body)}
    mock_boto_sess.return_value.client.return_value = mock_s3_client
    mock_client_cls.return_value = _build_applovin_mock(ad_unit_count=2)

    with patch("sys.argv", _argv()):
        main()

    get_call = mock_s3_client.get_object.call_args
    assert get_call.kwargs["Key"] == newer
