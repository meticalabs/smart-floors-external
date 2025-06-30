import dataclasses
from typing import List, Any

import requests
from pydantic import BaseModel, field_validator
from pydantic.dataclasses import dataclass


@dataclass
class Context:
    path: str
    dataType: str


@dataclass
class ETLConfig:
    context: List[Context]
    lookbackWindowInDays: int = dataclasses.field(default=30)

    @field_validator("context", mode="after")
    @classmethod
    def min_context_length(cls, context: Any) -> Any:
        if not context:
            raise ValueError("Context must contain at least one item.")
        return context


@dataclass
class ModelConfig:
    parameters: dict = dataclasses.field(default_factory=dict)


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
        response = self.http_client.get(f"/application/{app_id}/config/etl")
        return ETLConfig(**response.json())

    def fetch_model_config(self, app_id: int, model_id: str) -> ModelConfig:
        response = self.http_client.get(f"/application/{app_id}/config/model/{model_id}")
        return ModelConfig(**response.json())
