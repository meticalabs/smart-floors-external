from unittest.mock import MagicMock

import pytest

from bid_optim_etl_py.utils import management_api
from bid_optim_etl_py.utils.management_api import BidFloorManagementAPI, HttpClient, ETLConfig, ModelConfig, Context


@pytest.fixture
def mock_http_client():
    """Fixture to create a mock HttpClient."""
    return MagicMock(spec=HttpClient)


@pytest.fixture
def api(mock_http_client):
    return BidFloorManagementAPI(http_client=mock_http_client)


def test_fetch_etl_config(api, mock_http_client):
    mock_response_data = {
        "context": [{"path": "example", "dataType": "string"}],
        "lookbackWindowInDays": 7,
    }
    mock_http_client.get.return_value.json.return_value = mock_response_data

    etl_config = api.fetch_etl_config(app_id=100)

    assert isinstance(etl_config, ETLConfig)
    assert etl_config.lookbackWindowInDays == 7
    assert etl_config.context[0].path == "example"
    assert etl_config.context[0].dataType == "string"
    mock_http_client.get.assert_called_once_with("/application/100/config/etl")


def test_fetch_model_config(api, mock_http_client):
    mock_response_data = {
        "parameters": {"param1": "value1", "param2": "value2"},
    }
    mock_http_client.get.return_value.json.return_value = mock_response_data

    model_config = api.fetch_model_config(app_id=100, model_id="test_model")

    assert isinstance(model_config, ModelConfig)
    assert model_config.parameters["param1"] == "value1"
    assert model_config.parameters["param2"] == "value2"
    mock_http_client.get.assert_called_once_with("/application/100/config/model/test_model")


def test_empty_etl_config_look_back():
    etl_config = ETLConfig(**{"context": [Context(path="default", dataType="string")]})
    assert etl_config.lookbackWindowInDays == 30
    assert len(etl_config.context) == 1


def test_all_empty_to_throw_error():
    with pytest.raises(ValueError, match="Context must contain at least one item."):
        ETLConfig(**{"context": []})


def test_empty_model_config():
    model_config = management_api.ModelConfig()
    assert isinstance(model_config, management_api.ModelConfig)
    assert model_config.parameters == {}
