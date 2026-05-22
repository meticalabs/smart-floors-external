"""End-to-end tests for customer 11901/13052 without touching real AppLovin.

We don't have the customer's AppLovin keys, so these tests mock the AppLovin
client entirely. S3 is mocked too so the suite runs in CI without credentials.

Both tests deliberately recreate the customer's broken S3 listing — the
percentile file PLUS a stale upload file at /uploads/ with the newest mtime —
to lock in that the file-discovery fix correctly skips /uploads/ even when
the upload outpaces the latest percentile file.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests.test_update_bid_floor_values import (
    STANDARD_GRID,
    _argv,
    _bid_floors_from_calls,
    _build_applovin_mock,
    _percentiles_json,
)


PERCENTILE_KEY = (
    "bid-floor-optimisation/applovin/percentile/11901/13052/"
    "2026-05-11_android_reward.json"
)
UPLOAD_KEY = (
    "bid-floor-optimisation/applovin/percentile/11901/13052/"
    "uploads/ad_unit_configurations_android_reward.json"
)
FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "percentiles_11901_13052_android_reward.json"
)


def _s3_mock(body_text, listing):
    """S3 client mock with a configurable listing and get_object body."""
    client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": k, "LastModified": pd.Timestamp(d)} for k, d in listing
            ]
        }
    ]
    client.get_paginator.return_value = paginator
    body = body_text.encode("utf-8")
    client.get_object.return_value = {"Body": SimpleNamespace(read=lambda: body)}
    return client


def _customer_argv(package_name="com.app"):
    argv = _argv()
    argv[argv.index("--customer-id") + 1] = "11901"
    argv[argv.index("--app-id") + 1] = "13052"
    argv[argv.index("--package-name") + 1] = package_name
    return argv


@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_uploads_path_skipped_even_when_newest(mock_boto_sess, mock_client_cls):
    """Regression for the bug the customer hit: listing contains both a
    percentile file and a stale upload, the upload has the newest mtime.
    Pre-fix the script picked the upload and tried to parse it as percentile
    data, crashing on ENAMETOOLONG. Post-fix the percentile file wins.
    """
    from scripts.update_bid_floor_values import main

    body = _percentiles_json(
        {"us": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]},
        STANDARD_GRID,
    )
    mock_s3 = _s3_mock(
        body,
        listing=[
            (PERCENTILE_KEY, "2026-05-11"),
            (UPLOAD_KEY, "2026-05-22"),
        ],
    )
    mock_boto_sess.return_value.client.return_value = mock_s3
    mock_client_cls.return_value = _build_applovin_mock(ad_unit_count=3)

    with patch("sys.argv", _customer_argv()):
        main()

    assert mock_s3.get_object.call_args.kwargs["Key"] == PERCENTILE_KEY, (
        f"Discovery picked {mock_s3.get_object.call_args.kwargs['Key']!r}; "
        "the /uploads/ skip in discovery has regressed."
    )
    assert mock_client_cls.return_value.update_ad_unit.call_count == 3


@pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason=(
        "Customer percentile fixture not present at tests/fixtures/ (local-only). "
        "Download a production percentile JSON for 11901/13052 to enable this test."
    ),
)
@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_e2e_customer_11901_with_collision_in_listing(mock_boto_sess, mock_client_cls):
    """Full end-to-end run for 11901/13052: real production percentile body,
    listing that includes the stale upload at /uploads/ with newer mtime,
    AppLovin entirely mocked. Asserts the right file is read, every metica
    ad unit gets a bid-floor update, CPMs are in range, and the S3 upload
    targets the expected key.
    """
    from scripts.update_bid_floor_values import main

    body = FIXTURE_PATH.read_text()
    mock_s3 = _s3_mock(
        body,
        listing=[
            (PERCENTILE_KEY, "2026-05-11"),
            (UPLOAD_KEY, "2026-05-22"),
        ],
    )
    mock_boto_sess.return_value.client.return_value = mock_s3

    applovin = _build_applovin_mock(ad_unit_count=5, prefix="metica_android_reward")
    for unit in applovin.get_ad_units.return_value:
        unit["package_name"] = "com.merge.art.canvas"
    mock_client_cls.return_value = applovin

    with patch("sys.argv", _customer_argv(package_name="com.merge.art.canvas")):
        main()

    assert mock_s3.get_object.call_args.kwargs["Key"] == PERCENTILE_KEY
    assert applovin.update_ad_unit.call_count == 5

    floors = _bid_floors_from_calls(applovin)
    for au_id, bid_floors in floors.items():
        assert bid_floors, f"{au_id} has no bid floors"
        for bf in bid_floors:
            cpm = float(bf["cpm"])
            assert 0 <= cpm <= 500, f"{au_id} CPM out of range: {cpm}"
            assert bf["countries"]["values"], f"{au_id} bid floor with empty country list"

    upload_call = mock_s3.put_object.call_args
    assert upload_call.kwargs["Key"] == UPLOAD_KEY
    uploaded = json.loads(upload_call.kwargs["Body"])
    assert isinstance(uploaded, list) and len(uploaded) >= 5
