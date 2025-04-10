import json
import logging
import random
import uuid
from datetime import datetime

import pandas as pd
import pytest
from pyspark.sql import Row

from bid_optim_etl_py.applovin_etl import Events, Schema


class TestApplovinETL:
    @pytest.fixture
    def sample_data(self):
        assignment_data = [
            Row(requestId="1", userId="U1", eventTime="2023-01-01T00:00:00Z", modelId="android_inter",
                placementTag="tag1", sessionId="session1", eventType="meticaBidFloorAssignment",
                customerId=123, appId=456, context="{}", cpmFloorAdUnitIds=["ad_unit_1", "ad_unit_2"],
                cpmFloorValues=[2.0, 1.0], propensity=2.5, inferenceData="{}", date=datetime(2023, 1, 1).date()),
            Row(requestId="2", userId="U2", eventTime="2023-01-01T00:00:00Z", modelId="android_inter",
                placementTag="tag2", sessionId="session2", eventType="meticaBidFloorAssignment",
                customerId=123, appId=456, context="{}", cpmFloorAdUnitIds=["ad_unit_3"],
                cpmFloorValues=[1.0], propensity=1.5, inferenceData="{}", date=datetime(2023, 1, 1).date()),
            Row(requestId="3", userId="U3", eventTime="2023-01-01T00:00:00Z", modelId="ios_inter",
                placementTag="tag3", sessionId="session3", eventType="meticaBidFloorAssignment",
                customerId=123, appId=456, context="{}", cpmFloorAdUnitIds=["ad_unit_4"],
                cpmFloorValues=[3.0], propensity=3.5, inferenceData="{}", date=datetime(2023, 1, 1).date()),
            Row(requestId="4", userId="U3", eventTime="2023-01-01T00:00:00Z", modelId="ios_inter",
                placementTag="tag3", sessionId="session3", eventType="meticaBidFloorAssignment",
                customerId=123, appId=456, context="{}", cpmFloorAdUnitIds=["ad_unit_4"],
                cpmFloorValues=[3.0], propensity=3.5, inferenceData="{}", date=datetime(2023, 1, 1).date())
        ]
        bid_sequence_data = [
            Row(requestId="1", userId="U1", isFilled=True, cpmFloorAdUnitId="ad_unit_1", cpmFloorValue=2.0,
                eventTime="2023-01-01T00:00:00Z", date=datetime(2023, 1, 1).date()),
            Row(requestId="2", userId="U2", isFilled=False, cpmFloorAdUnitId="ad_unit_3", cpmFloorValue=1.0,
                eventTime="2023-01-01T00:00:00Z", date=datetime(2023, 1, 1).date()),
            Row(requestId="3", userId="U3", isFilled=True, cpmFloorAdUnitId="ad_unit_4", cpmFloorValue=3.0,
                eventTime="2023-01-01T00:00:00Z", date=datetime(2023, 1, 1).date())
        ]
        ad_revenue_data = [
            Row(requestId="1", userId="U1", cpmFloorAdUnitId="ad_unit_1", totalAmount=2.0,
                eventTime="2023-01-01T00:00:00Z", date=datetime(2023, 1, 1).date()),
            Row(requestId="2", userId="U2", cpmFloorAdUnitId="ad_unit_3", totalAmount=1.0,
                eventTime="2023-01-01T00:00:00Z", date=datetime(2023, 1, 1).date()),
            Row(requestId="3", userId="U3", cpmFloorAdUnitId="ad_unit_4", totalAmount=3.0,
                eventTime="2023-01-01T00:00:00Z", date=datetime(2023, 1, 1).date())
        ]
        return assignment_data, bid_sequence_data, ad_revenue_data

    def test_join_all(self, spark, sample_data):
        assignment_data, bid_sequence_data, ad_revenue_data = sample_data

        assignment_df = spark.createDataFrame(assignment_data)
        bid_sequence_df = spark.createDataFrame(bid_sequence_data)
        ad_revenue_df = spark.createDataFrame(ad_revenue_data)

        events = Events(
            customer_id=123,
            app_id=456,
            s3_data_bucket="s3://dummy-bucket",
            date=datetime(2023, 1, 1),
            spark=spark,
            iceberg_catalog="dummy_catalog",
            region_name="us-east-1",
            logger=logging.getLogger("test_logger")
        )

        result_df = events.join_all(assignment_df, bid_sequence_df, ad_revenue_df)

        assert result_df.count() == 3
        assert all(col in result_df.columns for col in [
            Schema.REQUEST_ID, Schema.USER_ID, Schema.CPM_FLOOR_AD_UNIT_ID,
            Schema.TOTAL_AMOUNT, Schema.IS_FILLED, Schema.CPM_FLOOR_VALUE,
            Schema.MODEL_ID, Schema.CUSTOMER_ID, Schema.APP_ID, Schema.CONTEXT
        ])

    @pytest.fixture
    def assignment_data_with_complex_context(self, spark):
        data_size = 15

        context_data = spark.createDataFrame([
            Row(context=json.dumps({k: v for k, v in {
                "user.country": random.choice([None, "US", "IN", "UK"]),
                "user.languageCode": random.choice([None, "en", "fr", "es"]),
                "user.deviceType": random.choice([None, "mobile", "tablet"]),
                "user.osVersion": random.choice([None, "10.0", "11.0"]),
                "user.deviceModel": random.choice([None, "iPhone", "Samsung"]),
                "assignmentDayOfWeek": random.choice([None, 1, 2, 3, 4, 5, 6, 7]),
                "assignmentHourOfDay": random.choice(
                    [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),
                "user.minRevenueLast24Hours": random.choice([None, 0.0, 1.0, 2.0]),
                "user.avgRevenueLast24Hours": random.choice([None, 0.0, 1.0, 2.0]),
                "user.avgRevenueLast48Hours": random.choice([None, 0.0, 1.0, 2.0]),
                "user.avgRevenueLast72Hours": random.choice([None, 0.0, 1.0, 2.0]),
                "user.mostRecentAdSource": random.choice([None, 0.0, 1.0, 2.0]),
                "user.mostRecentAdRevenue": random.choice([None, 0.0, 1.0, 2.0]),
            }.items() if v is not None}))
            for _ in range(data_size)
        ])

        assignment_data = spark.createDataFrame([
            Row(requestId=str(uuid.uuid4()), userId=str(uuid.uuid4()),
                eventTime="2023-01-01T00:00:00Z",
                modelId=random.choice(["android_inter", "ios_inter"]),
                placementTag="tag1", sessionId="",
                eventType="meticaBidFloorAssignment",
                customerId=123, appId=456,
                cpmFloorAdUnitIds=random.choices(
                    ["ad_unit_1", "ad_unit_2", "ad_unit_3", "ad_unit_4",
                     "ad_unit_5"], k=3),
                cpmFloorValues=random.choices([5.0, 4.0, 3.0, 2.0, 1.0], k=3),
                propensity=0.5, inferenceData="{}",
                date=datetime(2023, 1, 1).date()) for _ in range(data_size)])

        return assignment_data.join(context_data)

    def test_denormalise_context_field(self, spark, assignment_data_with_complex_context):
        events = Events(
            customer_id=123,
            app_id=456,
            s3_data_bucket="s3://dummy-bucket",
            date=datetime(2023, 1, 1),
            spark=spark,
            iceberg_catalog="dev",
            region_name="us-east-1",
            logger=logging.getLogger("test_logger")
        )
        df = events.denormalise_context_field(assignment_data_with_complex_context)
        df.show(truncate=False)
        assert df.count() == assignment_data_with_complex_context.count()
        assert all(col.name in df.columns for col in events.context_schema.fields)
