from typing import List

import requests
from pydantic import BaseModel
from pydantic.dataclasses import dataclass


@dataclass
class Context:
    name: str
    dataType: str


@dataclass
class ETLConfig:
    context: List[Context]
    lookbackWindowInDays: int


@dataclass
class ModelConfig:
    modelId: str
    parameters: dict


@dataclass
class HttpClient:
    base_url: str

    def get(self, endpoint: str, params: dict = None) -> requests.Response:
        url = f"{self.base_url.rstrip('/')}{endpoint}"
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response


class BidFloorManagementAPI(BaseModel):
    http_client: HttpClient

    def fetch_etl_config(self, app_id: int) -> ETLConfig:
        response = self.http_client.get(f"/bidfloor/app/{app_id}/config/etl")
        return ETLConfig(**response.json())

    def fetch_model_config(self, app_id: int, model_id: str) -> ModelConfig:
        response = self.http_client.get(f"/bidfloor/app/{app_id}/model/{model_id}")
        return ModelConfig(**response.json())
