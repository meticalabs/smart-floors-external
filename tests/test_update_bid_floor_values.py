import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd


def _make_s3_list_response(keys):
    return {
        "Contents": [
            {"Key": k, "LastModified": pd.Timestamp("2025-01-01")}
            for k in keys
        ]
    }


def _percentiles_json():
    # minimal two countries with percentiles matching PERCENTILE_COLUMNS
    return json.dumps(
        [
            {"user.country": "us", "p10": 0.5, "p20": 0.6, "p30": 0.7, "p40": 0.8, "p50": 0.9, "p60": 1.0, "p70": 1.1, "p80": 1.2, "p90": 1.3},
            {"user.country": "gb", "p10": 0.4, "p20": 0.5, "p30": 0.6, "p40": 0.7, "p50": 0.8, "p60": 0.9, "p70": 1.0, "p80": 1.1, "p90": 1.2},
        ]
    )


@patch("scripts.update_bid_floor_values.ApplovinManagementApiClient")
@patch("scripts.update_bid_floor_values.boto3.Session")
def test_main_happy_path(mock_boto_sess, mock_client_cls, monkeypatch):
    from scripts.update_bid_floor_values import main

    # Arrange S3 mocks
    mock_s3_client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        _make_s3_list_response([
            "bid-floor-optimisation/applovin/percentile/1/2/2025-09-10/android_reward.json",
            "bid-floor-optimisation/applovin/percentile/1/2/2025-10-01/android_reward.json",
        ])
    ]
    mock_s3_client.get_paginator.return_value = paginator

    body_bytes = _percentiles_json().encode("utf-8")
    mock_s3_client.get_object.return_value = {"Body": SimpleNamespace(read=lambda: body_bytes)}

    mock_session = MagicMock()
    mock_session.client.return_value = mock_s3_client
    mock_boto_sess.return_value = mock_session

    # Arrange AppLovin client mocks
    client_instance = MagicMock()
    # Provide two metica ad units to be updated
    client_instance.get_ad_units.return_value = [
        {"id": "au1", "name": "metica_android_reward_1", "ad_format": "reward", "package_name": "com.app"},
        {"id": "au2", "name": "metica_android_reward_2", "ad_format": "reward", "package_name": "com.app"},
    ]
    mock_client_cls.return_value = client_instance

    # Argv
    argv = [
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

    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    with patch("sys.argv", argv):
        main()

    # Assert S3 interactions
    assert mock_s3_client.get_paginator.called
    mock_s3_client.get_object.assert_called_once()
    mock_s3_client.put_object.assert_called_once()

    # Assert AppLovin updates
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

    argv = [
        "prog",
        "--customer-id", "1",
        "--app-id", "2",
        "--ad-type", "reward",
        "--platform", "android",
        "--applovin-api-key", "k",
        "--aws-access-key-id", "ak",
        "--aws-secret-access-key", "sk",
        "--package-name", "com.app",
    ]

    with patch("sys.argv", argv):
        try:
            main()
            assert False, "Expected RuntimeError when no JSON present"
        except RuntimeError as e:
            assert "No percentiles JSON found" in str(e)


