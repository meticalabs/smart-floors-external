from unittest.mock import MagicMock

import pytest

from bid_optim_etl_py.utils.management_api import BidFloorManagementAPI, HttpClient, ETLConfig, ModelConfig


@pytest.fixture
def mock_http_client():
    """Fixture to create a mock HttpClient."""
    return MagicMock(spec=HttpClient)


@pytest.fixture
def api(mock_http_client):
    return BidFloorManagementAPI(http_client=mock_http_client)


def test_fetch_etl_config(api, mock_http_client):
    mock_response_data = {
        "context": [{"name": "example", "dataType": "string"}],
        "lookbackWindowInDays": 7,
    }
    mock_http_client.get.return_value.json.return_value = mock_response_data

    etl_config = api.fetch_etl_config(app_id=100)

    assert isinstance(etl_config, ETLConfig)
    assert etl_config.lookbackWindowInDays == 7
    assert etl_config.context[0].name == "example"
    assert etl_config.context[0].dataType == "string"
    mock_http_client.get.assert_called_once_with("/bidfloor/app/100/config/etl")


def test_fetch_model_config(api, mock_http_client):
    mock_response_data = {
        "modelId": "test_model",
        "parameters": {"param1": "value1", "param2": "value2"},
    }
    mock_http_client.get.return_value.json.return_value = mock_response_data

    model_config = api.fetch_model_config(app_id=100, model_id="test_model")

    assert isinstance(model_config, ModelConfig)
    assert model_config.modelId == "test_model"
    assert model_config.parameters["param1"] == "value1"
    assert model_config.parameters["param2"] == "value2"
    mock_http_client.get.assert_called_once_with("/bidfloor/app/100/model/test_model")
