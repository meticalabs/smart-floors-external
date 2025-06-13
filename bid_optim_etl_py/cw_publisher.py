from dataclasses import dataclass

import boto3
from bid_optim_etl_py.cloudwatch_wrapper import CloudWatchWrapper


@dataclass
class CloudWatchAlerts:
    region: str

    def __post_init__(self):
        self.cw_wrapper = CloudWatchWrapper(boto3.resource("cloudwatch", region_name=self.region))
