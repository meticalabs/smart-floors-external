from dataclasses import dataclass

import boto3
from etl_py_commons.cloud_watch import CloudWatchWrapper


@dataclass
class CloudWatchAlerts:
    region: str

    def __post_init__(self):
        self.cw_wrapper = CloudWatchWrapper(boto3.resource("cloudwatch", region_name=self.region))
