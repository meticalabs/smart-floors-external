import json
import os
import random
import uuid
from datetime import datetime
from unittest import mock

import pandas as pd
import pytest
from etl_py_commons.cfg_parser import ConfigParser
from pyspark.sql import Row
from pyspark.sql.functions import date_format
from pyspark.sql.types import StructType, StructField, TimestampType, ArrayType, DoubleType, StringType, BooleanType

from bid_optim_etl_py.applovin_etl import Events, Schema
from bid_optim_etl_py.cfg_parser import Config
from bid_optim_etl_py.utils.management_api import ETLConfig


class TestApplovinETL:
    @pytest.fixture
    def sample_data(self):
        assignment_data = [
            Row(
                requestId="1",
                userId="U1",
                eventTime="2023-01-01T00:00:00Z",
                modelId="android_inter",
                placementTag="tag1",
                sessionId="session1",
                eventType="meticaBidFloorAssignment",
                customerId=123,
                appId=456,
                context="{}",
                cpmFloorAdUnitIds=["ad_unit_1", "ad_unit_2"],
                cpmFloorValues=[2.0, 1.0],
                propensity=2.5,
                inferenceData="{}",
                date=datetime(2023, 1, 1).date(),
            ),
            Row(
                requestId="2",
                userId="U2",
                eventTime="2023-01-01T00:00:00Z",
                modelId="android_inter",
                placementTag="tag2",
                sessionId="session2",
                eventType="meticaBidFloorAssignment",
                customerId=123,
                appId=456,
                context="{}",
                cpmFloorAdUnitIds=["ad_unit_3"],
                cpmFloorValues=[1.0],
                propensity=1.5,
                inferenceData="{}",
                date=datetime(2023, 1, 1).date(),
            ),
            Row(
                requestId="3",
                userId="U3",
                eventTime="2023-01-01T00:00:00Z",
                modelId="ios_inter",
                placementTag="tag3",
                sessionId="session3",
                eventType="meticaBidFloorAssignment",
                customerId=123,
                appId=456,
                context="{}",
                cpmFloorAdUnitIds=["ad_unit_4"],
                cpmFloorValues=[3.0],
                propensity=3.5,
                inferenceData="{}",
                date=datetime(2023, 1, 1).date(),
            ),
            Row(
                requestId="4",
                userId="U3",
                eventTime="2023-01-01T00:00:00Z",
                modelId="ios_inter",
                placementTag="tag3",
                sessionId="session3",
                eventType="meticaBidFloorAssignment",
                customerId=123,
                appId=456,
                context="{}",
                cpmFloorAdUnitIds=["ad_unit_4"],
                cpmFloorValues=[3.0],
                propensity=3.5,
                inferenceData="{}",
                date=datetime(2023, 1, 1).date(),
            ),
        ]
        bid_sequence_data = [
            Row(
                requestId="1",
                userId="U1",
                isFilled=True,
                cpmFloorAdUnitId="ad_unit_1",
                eventTime="2023-01-01T00:00:00Z",
                date=datetime(2023, 1, 1).date(),
            ),
            Row(
                requestId="2",
                userId="U2",
                isFilled=False,
                cpmFloorAdUnitId="ad_unit_3",
                eventTime="2023-01-01T00:00:00Z",
                date=datetime(2023, 1, 1).date(),
            ),
            Row(
                requestId="3",
                userId="U3",
                isFilled=True,
                cpmFloorAdUnitId="ad_unit_4",
                eventTime="2023-01-01T00:00:00Z",
                date=datetime(2023, 1, 1).date(),
            ),
        ]
        ad_revenue_data = [
            Row(
                requestId="1",
                userId="U1",
                cpmFloorAdUnitId="ad_unit_1",
                totalAmount=2.0,
                eventTime="2023-01-01T00:00:00Z",
                date=datetime(2023, 1, 1).date(),
            ),
            Row(
                requestId="2",
                userId="U2",
                cpmFloorAdUnitId="ad_unit_3",
                totalAmount=1.0,
                eventTime="2023-01-01T00:00:00Z",
                date=datetime(2023, 1, 1).date(),
            ),
            Row(
                requestId="3",
                userId="U3",
                cpmFloorAdUnitId="ad_unit_4",
                totalAmount=3.0,
                eventTime="2023-01-01T00:00:00Z",
                date=datetime(2023, 1, 1).date(),
            ),
        ]
        return assignment_data, bid_sequence_data, ad_revenue_data

    @pytest.fixture
    def config(self, request) -> Config:
        dir_path = os.path.dirname(request.module.__file__)
        return ConfigParser.parse(file_name="local", base_path=f"{dir_path}/resources/confs", config_clazz_type=Config)

    @pytest.fixture
    def events_instance(self, config, spark):
        mock_management_api = mock.Mock()
        mock_management_api.fetch_etl_config.return_value = ETLConfig(
            **{
                "context": [
                    {"name": "user.country", "dataType": "string"},
                    {"name": "user.languageCode", "dataType": "string"},
                    {"name": "user.deviceType", "dataType": "string"},
                    {"name": "user.osVersion", "dataType": "string"},
                    {"name": "user.deviceModel", "dataType": "string"},
                    {"name": "user.minRevenueLast24Hours", "dataType": "number"},
                    {"name": "user.avgRevenueLast24Hours", "dataType": "number"},
                    {"name": "user.avgRevenueLast48Hours", "dataType": "number"},
                    {"name": "user.avgRevenueLast72Hours", "dataType": "number"},
                    {"name": "user.mostRecentAdRevenue", "dataType": "number"},
                ],
                "lookbackWindowInDays": 30,
            }
        )

        return Events(
            customer_id=123,
            app_id=456,
            s3_data_bucket="s3://example-bucket",
            date=datetime(2023, 1, 1).date(),
            spark=spark,
            iceberg_catalog="example_catalog",
            region_name="us-east-1",
            logger=None,
            management_api=mock_management_api,
        )

    def test_join_all(self, config, spark, events_instance, sample_data):
        assignment_data, bid_sequence_data, ad_revenue_data = sample_data

        assignment_df = spark.createDataFrame(assignment_data)
        bid_sequence_df = spark.createDataFrame(bid_sequence_data)
        ad_revenue_df = spark.createDataFrame(ad_revenue_data)

        result_df = events_instance.join_all(assignment_df, bid_sequence_df, ad_revenue_df)

        assert result_df.count() == 3
        assert all(
            col in result_df.columns
            for col in [
                Schema.REQUEST_ID,
                Schema.USER_ID,
                Schema.CPM_FLOOR_AD_UNIT_ID,
                Schema.TOTAL_AMOUNT,
                Schema.IS_FILLED,
                Schema.MODEL_ID,
                Schema.CUSTOMER_ID,
                Schema.APP_ID,
                Schema.CONTEXT,
            ]
        )

    @pytest.fixture
    def assignment_data_with_complex_context(self, spark):
        data_size = 15

        # fmt:off
        context_data = spark.createDataFrame([
            Row(context=json.dumps({k: v for k, v in {
                "user.country": random.choice([None, "US", "IN", "UK"]),
                "user.languageCode": random.choice([None, "en", "fr", "es"]),
                "user.deviceType": random.choice([None, "mobile", "tablet"]),
                "user.osVersion": random.choice([None, "10.0", "11.0"]),
                "user.deviceModel": random.choice([None, "iPhone", "Samsung"]),
                "user.minRevenueLast24Hours": random.choice([None, 0.0, 1.0, 2.0]),
                "user.avgRevenueLast24Hours": random.choice([None, 0.0, 1.0, 2.0]),
                "user.avgRevenueLast48Hours": random.choice([None, 0.0, 1.0, 2.0]),
                "user.avgRevenueLast72Hours": random.choice([None, 0.0, 1.0, 2.0]),
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
                assignmentHourOfDay=random.choice([None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                   15, 16, 17, 18, 19, 20, 21, 22, 23]),
                assignmentDayOfWeek=random.choice([None, 1, 2, 3, 4, 5, 6, 7]),
                propensity=0.5, inferenceData="{}",
                date=datetime(2023, 1, 1).date()) for _ in range(data_size)])
        # fmt:on
        return assignment_data.join(context_data)

    def test_denormalise_context_field(self, config, spark, events_instance, assignment_data_with_complex_context):
        df = events_instance.denormalise_context_field(
            assignment_data_with_complex_context, events_instance.fetch_context_schema()
        )
        df.show(truncate=False)
        assert df.count() == assignment_data_with_complex_context.count()
        expected_schema = StructType(
            [
                StructField("user.country", StringType(), True),
                StructField("user.languageCode", StringType(), True),
                StructField("user.deviceType", StringType(), True),
                StructField("user.osVersion", StringType(), True),
                StructField("user.deviceModel", StringType(), True),
                StructField("user.minRevenueLast24Hours", DoubleType(), True),
                StructField("user.avgRevenueLast24Hours", DoubleType(), True),
                StructField("user.avgRevenueLast48Hours", DoubleType(), True),
                StructField("user.avgRevenueLast72Hours", DoubleType(), True),
                StructField("user.mostRecentAdRevenue", DoubleType(), True),
            ]
        )
        assert all(col.name in df.columns for col in expected_schema.fields)
        assert all(col.dataType == expected_schema[col.name].dataType for col in expected_schema.fields)

    def test_day_of_week(self, spark):
        from pyspark.sql import functions as F

        df = spark.createDataFrame(
            pd.DataFrame({"date": [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]})
        )

        assert df.withColumns({"day_of_week": F.expr("weekday(date)"), "day": date_format("date", "E")}).select(
            "day_of_week", "day"
        ).collect() == [Row(day_of_week=6, day="Sun"), Row(day_of_week=0, day="Mon"), Row(day_of_week=1, day="Tue")]

    @pytest.fixture
    def df_to_test_hardcoded_contexts(self, spark):
        schema = StructType(
            [
                StructField(Schema.EVENT_TIME, TimestampType(), True),
                StructField(Schema.CPM_FLOOR_VALUES, ArrayType(DoubleType()), True),
            ]
        )
        # fmt:off
        data = [
            Row(eventTime=datetime(2023,1,1,1,0,0), cpmFloorValues=[0.5, 0.3, 0.1]),
            Row(eventTime=datetime(2023,1,2,23,0,0), cpmFloorValues=[0.7, 0.4, 0.1]),
            Row(eventTime=datetime(2023,1,3,4,10,0), cpmFloorValues=[0.8, 0.2, 0.1]),
        ]
        # fmt: on
        return spark.createDataFrame(data, schema=schema)

    def test_add_hardcoded_contexts(self, df_to_test_hardcoded_contexts, events_instance):
        df = events_instance.add_hardcoded_contexts(df_to_test_hardcoded_contexts)
        df.show(truncate=False)
        expected_columns = [
            Schema.EVENT_TIME,
            Schema.CPM_FLOOR_VALUES,
            Schema.ASSIGNMENT_HOUR_OF_DAY,
            Schema.ASSIGNMENT_DAY_OF_WEEK,
            Schema.HIGHEST_BID_FLOOR_VALUE,
            Schema.MEDIUM_BID_FLOOR_VALUE,
        ]

        assert all(col in df.columns for col in expected_columns)
        result = df.collect()

        for row in result:
            assert isinstance(row[Schema.ASSIGNMENT_HOUR_OF_DAY], int)
            assert 0 <= row[Schema.ASSIGNMENT_HOUR_OF_DAY] <= 23
            assert isinstance(row[Schema.ASSIGNMENT_DAY_OF_WEEK], int)
            assert 0 <= row[Schema.ASSIGNMENT_DAY_OF_WEEK] <= 6
            assert isinstance(row[Schema.HIGHEST_BID_FLOOR_VALUE], float)
            assert isinstance(row[Schema.MEDIUM_BID_FLOOR_VALUE], float)

        assert all(row[Schema.HIGHEST_BID_FLOOR_VALUE] >= row[Schema.MEDIUM_BID_FLOOR_VALUE] for row in result)

        assert result == [
            Row(
                eventTime=datetime(2023, 1, 1, 1, 0, 0),
                cpmFloorValues=[0.5, 0.3, 0.1],
                assignmentHourOfDay=1,
                assignmentDayOfWeek=6,
                highestBidFloorValue=0.5,
                mediumBidFloorValue=0.3,
            ),
            Row(
                eventTime=datetime(2023, 1, 2, 23, 0, 0),
                cpmFloorValues=[0.7, 0.4, 0.1],
                assignmentHourOfDay=23,
                assignmentDayOfWeek=0,
                highestBidFloorValue=0.7,
                mediumBidFloorValue=0.4,
            ),
            Row(
                eventTime=datetime(2023, 1, 3, 4, 10, 0),
                cpmFloorValues=[0.8, 0.2, 0.1],
                assignmentHourOfDay=4,
                assignmentDayOfWeek=1,
                highestBidFloorValue=0.8,
                mediumBidFloorValue=0.2,
            ),
        ]

    def test_fetch_assignment_events_filter_conditions(self, events_instance, spark):
        schema = StructType(
            [
                StructField("context", StringType(), True),
                StructField("cpmFloorValues", ArrayType(DoubleType()), True),
                StructField("date", StringType(), True),
            ]
        )
        data = [
            Row(context="{}", cpmFloorValues=[1.0, 2.0, 3.0], date="2022-12-31"),  # Valid
            Row(context=None, cpmFloorValues=[1.0, 2.0, 3.0], date="2022-12-31"),  # Invalid: Context is null
            Row(context="{}", cpmFloorValues=[1.0, 2.0], date="2022-12-31"),  # Invalid: Less than 3 values
            Row(context="{}", cpmFloorValues=[1.0, 2.0, 3.0], date="2023-01-02"),  # Invalid: Date out of range
        ]
        df = spark.createDataFrame(data, schema)

        filtered_df = df.filter(
            (df.date <= events_instance.date.isoformat())
            & events_instance.valid_context_values()
            & events_instance.has_valid_bid_floor_values()
        )
        assert filtered_df.count() == 1

    def test_fetch_revenue_events_filter_conditions(self, events_instance, spark):
        schema = StructType(
            [
                StructField("totalAmount", DoubleType(), True),
                StructField("date", StringType(), True),
            ]
        )
        data = [
            Row(totalAmount=5.0, date="2022-12-31"),  # Valid
            Row(totalAmount=None, date="2022-12-31"),  # Invalid: Total amount is null
            Row(totalAmount=-1.0, date="2022-12-31"),  # Invalid: Negative total amount
            Row(totalAmount=10.0, date="2023-01-02"),  # Invalid: Date out of range
        ]
        df = spark.createDataFrame(data, schema)

        filtered_df = df.filter((df.date <= events_instance.date.isoformat()) & events_instance.valid_revenue_rows())
        assert filtered_df.count() == 1

    def test_fetch_bid_sequence_events_filter_conditions(self, events_instance, spark):
        schema = StructType(
            [
                StructField("isFilled", BooleanType(), True),
                StructField("date", StringType(), True),
            ]
        )
        data = [
            Row(isFilled=True, date="2022-12-31"),  # Valid
            Row(isFilled=False, date="2022-12-31"),  # Invalid: isFilled is false
            Row(isFilled=True, date="2023-01-02"),  # Invalid: Date out of range
        ]
        df = spark.createDataFrame(data, schema)

        filtered_df = df.filter(df.isFilled & (df.date <= events_instance.date.isoformat()))
        assert filtered_df.count() == 1
