import json
import logging
import os
import random
import uuid
from datetime import datetime, timedelta
from unittest import mock

import pandas as pd
import pytest
from etl_py_commons.cfg_parser import ConfigParser
from pyspark.sql import Row
from pyspark.sql.functions import date_format, col
from pyspark.sql.types import StructType, StructField, TimestampType, ArrayType, DoubleType, StringType, BooleanType

from bid_optim_etl_py.applovin_etl import Events, Schema, fill_with_cached_context
from bid_optim_etl_py.cfg_parser import ConfigFile
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
    def config(self, request) -> ConfigFile:
        dir_path = os.path.dirname(request.module.__file__)
        return ConfigParser.parse(
            file_name="local", base_path=f"{dir_path}/resources/confs", config_clazz_type=ConfigFile
        )

    @pytest.fixture
    def events_instance(self, config, spark):
        mock_management_api = mock.Mock()
        mock_management_api.fetch_etl_config.return_value = ETLConfig(
            **{
                "context": [
                    {"path": "user.country", "dataType": "string"},
                    {"path": "user.languageCode", "dataType": "string"},
                    {"path": "user.deviceType", "dataType": "string"},
                    {"path": "user.osVersion", "dataType": "string"},
                    {"path": "user.deviceModel", "dataType": "string"},
                    {"path": "user.minRevenueLast24Hours", "dataType": "number"},
                    {"path": "user.avgRevenueLast24Hours", "dataType": "number"},
                    {"path": "user.avgRevenueLast48Hours", "dataType": "number"},
                    {"path": "user.avgRevenueLast72Hours", "dataType": "number"},
                    {"path": "user.mostRecentAdRevenue", "dataType": "number"},
                ],
                "lookbackWindowInDays": 30,
                "maxAdUnits": 3,
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
            logger=logging.getLogger("bid_optim_etl_py.applovin_etl"),
            management_api=mock_management_api,
        )

    @pytest.fixture
    def events_instance_with_max_ad_units(self, config, spark, max_ad_units_value):
        mock_management_api = mock.Mock()
        mock_management_api.fetch_etl_config.return_value = ETLConfig(
            **{
                "context": [
                    {"path": "user.country", "dataType": "string"},
                    {"path": "user.languageCode", "dataType": "string"},
                    {"path": "user.deviceType", "dataType": "string"},
                    {"path": "user.osVersion", "dataType": "string"},
                    {"path": "user.deviceModel", "dataType": "string"},
                    {"path": "user.minRevenueLast24Hours", "dataType": "number"},
                    {"path": "user.avgRevenueLast24Hours", "dataType": "number"},
                    {"path": "user.avgRevenueLast48Hours", "dataType": "number"},
                    {"path": "user.avgRevenueLast72Hours", "dataType": "number"},
                    {"path": "user.mostRecentAdRevenue", "dataType": "number"},
                ],
                "lookbackWindowInDays": 30,
                "maxAdUnits": max_ad_units_value,
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
            logger=logging.getLogger("bid_optim_etl_py.applovin_etl"),
            management_api=mock_management_api,
        )

    def test_join_all(self, config, spark, events_instance, sample_data):
        assignment_data, bid_sequence_data, ad_revenue_data = sample_data

        assignment_df = spark.createDataFrame(assignment_data)
        ad_revenue_df = spark.createDataFrame(ad_revenue_data)

        result_df = events_instance.join_assignment_and_revenue(assignment_df, ad_revenue_df)

        assert (
            result_df.select(Schema.REQUEST_ID).distinct().count()
            == assignment_df.select(Schema.REQUEST_ID).distinct().count()
        )

        assert result_df.count() == 4
        assert all(
            col in result_df.columns
            for col in [
                Schema.REQUEST_ID,
                Schema.USER_ID,
                Schema.TOTAL_AMOUNT,
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
            Row(id=i, context=json.dumps({k: v for k, v in {
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
            for i in range(data_size)
        ])

        assignment_data = spark.createDataFrame([
            Row(id=i, requestId=str(uuid.uuid4()), userId=str(uuid.uuid4()),
                eventTime=datetime(2023, 1, 1, 0, 0, 0) +
                          timedelta(hours=random.randint(0, 23),
                                    minutes=random.randint(0, 59),
                                    seconds=random.randint(0, 59)),
                modelId=random.choice(["android_inter", "ios_inter"]),
                placementTag="tag1", sessionId="",
                eventType="meticaBidFloorAssignment",
                customerId=123, appId=456,
                inferenceData=json.dumps(random.choice([{"inferenceEndpoint":"value1","model":"model_file.tar.gz"}
                                                           , {}])),
                cpmFloorAdUnitIds=random.choices(
                    ["ad_unit_1", "ad_unit_2", "ad_unit_3", "ad_unit_4",
                     "ad_unit_5"], k=3),
                cpmFloorValues=random.choices([5.0, 4.0, 3.0, 2.0, 1.0], k=3),
                assignmentHourOfDay=random.choice([None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                   15, 16, 17, 18, 19, 20, 21, 22, 23]),
                assignmentDayOfWeek=random.choice([None, 1, 2, 3, 4, 5, 6, 7]),
                propensity=0.5,
                date=datetime(2023, 1, 1).date()) for i in range(data_size)])
        # fmt:on
        return assignment_data.join(context_data, on="id").drop("id")

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
            # Test case 1: Array with 3 elements
            Row(eventTime=datetime(2023,1,1,1,0,0), cpmFloorValues=[0.5, 0.3, 0.1]),
            # Test case 2: Array with 2 elements
            Row(eventTime=datetime(2023,1,2,23,0,0), cpmFloorValues=[0.7, 0.4]),
            # Test case 3: Array with 1 element
            Row(eventTime=datetime(2023,1,3,4,10,0), cpmFloorValues=[0.8]),
            # Test case 4: Empty array
            Row(eventTime=datetime(2023,1,4,5,0,0), cpmFloorValues=[]),
            # Test case 5: Array with more than 3 elements
            Row(eventTime=datetime(2023,1,5,6,0,0), cpmFloorValues=[0.9, 0.8, 0.7, 0.6, 0.5]),
        ]
        # fmt: on
        return spark.createDataFrame(data, schema=schema)

    def test_add_hardcoded_contexts(self, df_to_test_hardcoded_contexts, events_instance):
        df = events_instance.add_hardcoded_contexts(df_to_test_hardcoded_contexts)
        expected_columns = [
            Schema.EVENT_TIME,
            Schema.CPM_FLOOR_VALUES,
            Schema.ASSIGNMENT_HOUR_OF_DAY,
            Schema.ASSIGNMENT_DAY_OF_WEEK,
            Schema.HIGHEST_BID_FLOOR_VALUE,
            Schema.MEDIUM_BID_FLOOR_VALUE,
        ]

        assert all(col in df.columns for col in expected_columns)

        result = df.orderBy(Schema.EVENT_TIME).collect()

        expected_data = [
            Row(
                eventTime=datetime(2023, 1, 1, 1, 0),
                cpmFloorValues=[0.5, 0.3, 0.1],
                assignmentHourOfDay=1,
                assignmentDayOfWeek=6,
                highestBidFloorValue=0.3,
                mediumBidFloorValue=0.5,
            ),
            Row(
                eventTime=datetime(2023, 1, 2, 23, 0),
                cpmFloorValues=[0.7, 0.4],
                assignmentHourOfDay=23,
                assignmentDayOfWeek=0,
                highestBidFloorValue=0.7,
                mediumBidFloorValue=None,
            ),
            Row(
                eventTime=datetime(2023, 1, 3, 4, 10),
                cpmFloorValues=[0.8],
                assignmentHourOfDay=4,
                assignmentDayOfWeek=1,
                highestBidFloorValue=None,
                mediumBidFloorValue=None,
            ),
            Row(
                eventTime=datetime(2023, 1, 4, 5, 0),
                cpmFloorValues=[],
                assignmentHourOfDay=5,
                assignmentDayOfWeek=2,
                highestBidFloorValue=None,
                mediumBidFloorValue=None,
            ),
            Row(
                eventTime=datetime(2023, 1, 5, 6, 0),
                cpmFloorValues=[0.9, 0.8, 0.7, 0.6, 0.5],
                assignmentHourOfDay=6,
                assignmentDayOfWeek=3,
                highestBidFloorValue=0.6,
                mediumBidFloorValue=0.7,
            ),
        ]

        # Convert to list of dictionaries for easier comparison
        result_dicts = [row.asDict(recursive=True) for row in result]
        expected_dicts = [row.asDict(recursive=True) for row in expected_data]

        assert result_dicts == expected_dicts

    @pytest.mark.parametrize("max_ad_units_value, expected_count", [(3, 1), (2, 2), (1, 3), (None, 1)])
    def test_fetch_assignment_events_filter_conditions(
        self, events_instance_with_max_ad_units, spark, max_ad_units_value, expected_count
    ):
        schema = StructType(
            [
                StructField("context", StringType(), True),
                StructField("cpmFloorValues", ArrayType(DoubleType()), True),
                StructField("date", StringType(), True),
            ]
        )
        data = [
            Row(context="{}", cpmFloorValues=[1.0, 2.0, 3.0], date="2022-12-31"),  # Valid for maxAdUnits <= 3
            Row(context=None, cpmFloorValues=[1.0, 2.0, 3.0], date="2022-12-31"),  # Invalid: Context is null
            Row(context="", cpmFloorValues=[1.0, 2.0, 3.0], date="2022-12-31"),  # Invalid: Context is empty string
            Row(context="{}", cpmFloorValues=[1.0, 2.0], date="2022-12-31"),  # Valid for maxAdUnits <= 2
            Row(context="{}", cpmFloorValues=[1.0], date="2022-12-31"),  # Valid for maxAdUnits <= 1
            Row(context="{}", cpmFloorValues=[], date="2022-12-31"),  # Valid for maxAdUnits <= 0 (or any)
            Row(context="{}", cpmFloorValues=None, date="2022-12-31"),  # Invalid: cpmFloorValues is null
            Row(context="{}", cpmFloorValues=[1.0, 2.0, 3.0], date="2023-01-02"),  # Invalid: Date out of range
        ]
        df = spark.createDataFrame(data, schema)

        filtered_df = df.filter(
            (df.date <= events_instance_with_max_ad_units.date.isoformat())
            & events_instance_with_max_ad_units.valid_context_values()
            & events_instance_with_max_ad_units.has_valid_bid_floor_values()
        )
        assert filtered_df.count() == expected_count

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

    def test_fill_with_cached_context(self, assignment_data_with_complex_context):
        filled_context_data = fill_with_cached_context(assignment_data_with_complex_context)
        filled_context_data.show(100, truncate=False)

    def test_fill_with_cached_context_handles_empty_dataframe(self, spark):
        empty_df = spark.createDataFrame([], schema=StructType([]))
        result_df = fill_with_cached_context(empty_df)
        assert result_df.isEmpty()

    def test_fill_with_cached_context_handles_single_row(self, spark):
        schema = StructType(
            [
                StructField(Schema.USER_ID, StringType(), True),
                StructField(Schema.CPM_FLOOR_AD_UNIT_IDS, ArrayType(StringType()), True),
                StructField(Schema.CPM_FLOOR_VALUES, ArrayType(DoubleType()), True),
                StructField(Schema.EVENT_TIME, StringType(), True),
                StructField(Schema.INFERENCE_DATA, StringType(), True),
                StructField(Schema.CONTEXT, StringType(), True),
            ]
        )
        data = [
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_1"],
                cpmFloorValues=[1.0],
                eventTime="2023-01-01T00:00:00Z",
                context="{}",
                inferenceData="{}",
            )
        ]
        df = spark.createDataFrame(data, schema=schema)
        result_df = fill_with_cached_context(df)
        assert result_df.count() == 1
        assert result_df.select(Schema.CONTEXT).collect()[0][0] == "{}"

    def test_fill_with_cached_context_handles_multiple_rows_with_same_group(self, spark):
        schema = StructType(
            [
                StructField(Schema.USER_ID, StringType(), True),
                StructField(Schema.CPM_FLOOR_AD_UNIT_IDS, ArrayType(StringType()), True),
                StructField(Schema.CPM_FLOOR_VALUES, ArrayType(DoubleType()), True),
                StructField(Schema.EVENT_TIME, StringType(), True),
                StructField(Schema.INFERENCE_DATA, StringType(), True),
                StructField(Schema.CONTEXT, StringType(), True),
            ]
        )
        data = [
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_1"],
                cpmFloorValues=[1.0],
                eventTime="2023-01-01T00:00:00Z",
                inferenceData=None,
                context="{}",
            ),
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_1"],
                cpmFloorValues=[1.0],
                eventTime="2023-01-01T01:00:00Z",
                inferenceData=None,
                context=None,
            ),
        ]
        df = spark.createDataFrame(data, schema=schema)
        result_df = fill_with_cached_context(df)
        assert result_df.count() == 2
        assert all(row[Schema.CONTEXT] == "{}" for row in result_df.collect())

    def test_fill_with_cached_context_handles_multiple_rows_with_different_groups(self, spark):
        schema = StructType(
            [
                StructField(Schema.USER_ID, StringType(), True),
                StructField(Schema.CPM_FLOOR_AD_UNIT_IDS, ArrayType(StringType()), True),
                StructField(Schema.CPM_FLOOR_VALUES, ArrayType(DoubleType()), True),
                StructField(Schema.EVENT_TIME, StringType(), True),
                StructField(Schema.INFERENCE_DATA, StringType(), True),
                StructField(Schema.CONTEXT, StringType(), True),
            ]
        )
        data = [
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_1"],
                cpmFloorValues=[1.0],
                eventTime="2023-01-01T00:00:00Z",
                inferenceData=None,
                context="{}",
            ),
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_2"],
                cpmFloorValues=[2.0],
                eventTime="2023-01-01T01:00:00Z",
                inferenceData=None,
                context=None,
            ),
        ]
        df = spark.createDataFrame(data, schema=schema)
        result_df = fill_with_cached_context(df)
        assert result_df.count() == 2
        assert (
            result_df.filter(col(Schema.CPM_FLOOR_AD_UNIT_IDS).getItem(0) == "ad_unit_1")
            .select(Schema.CONTEXT)
            .collect()[0][0]
            == "{}"
        )
        assert (
            result_df.filter(col(Schema.CPM_FLOOR_AD_UNIT_IDS).getItem(0) == "ad_unit_2")
            .select(Schema.CONTEXT)
            .collect()[0][0]
            is None
        )

    def test_fill_with_cached_context_handles_unique_context_per_group(self, spark):
        schema = StructType(
            [
                StructField(Schema.USER_ID, StringType(), True),
                StructField(Schema.CPM_FLOOR_AD_UNIT_IDS, ArrayType(StringType()), True),
                StructField(Schema.CPM_FLOOR_VALUES, ArrayType(DoubleType()), True),
                StructField(Schema.EVENT_TIME, StringType(), True),
                StructField(Schema.INFERENCE_DATA, StringType(), True),
                StructField(Schema.CONTEXT, StringType(), True),
            ]
        )
        data = [
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_1"],
                cpmFloorValues=[1.0],
                eventTime="2023-01-01T00:00:00Z",
                inferenceData="{'endpoint':'value1'}",
                context=json.dumps({"key": "value1"}),
            ),
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_1"],
                cpmFloorValues=[1.0],
                eventTime="2023-01-01T01:00:00Z",
                inferenceData=None,
                context=json.dumps({"key": "value2"}),
            ),
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_2"],
                cpmFloorValues=[2.0],
                eventTime="2023-01-01T02:00:00Z",
                inferenceData="{'endpoint':'value1'}",
                context=json.dumps({"key": "value3"}),
            ),
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_2"],
                cpmFloorValues=[2.0],
                eventTime="2023-01-01T03:00:00Z",
                inferenceData=None,
                context=json.dumps({"key": "value4"}),
            ),
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_3"],
                cpmFloorValues=[3.0],
                eventTime="2023-01-01T04:00:00Z",
                inferenceData="{'endpoint':'value1'}",
                context=json.dumps({"key": "value5"}),
            ),
            Row(
                userId="U2",
                cpmFloorAdUnitIds=["ad_unit_4"],
                cpmFloorValues=[4.0],
                eventTime="2023-01-02T00:00:00Z",
                inferenceData="{'endpoint':'value1'}",
                context=json.dumps({"key": "value6"}),
            ),
            Row(
                userId="U2",
                cpmFloorAdUnitIds=["ad_unit_4"],
                cpmFloorValues=[4.0],
                eventTime="2023-01-02T01:00:00Z",
                inferenceData=None,
                context=json.dumps({"key": "value7"}),
            ),
            Row(
                userId="U2",
                cpmFloorAdUnitIds=["ad_unit_5"],
                cpmFloorValues=[5.0],
                eventTime="2023-01-02T02:00:00Z",
                inferenceData="{'endpoint':'value1'}",
                context=json.dumps({"key": "value8"}),
            ),
            Row(
                userId="U2",
                cpmFloorAdUnitIds=["ad_unit_5"],
                cpmFloorValues=[5.0],
                eventTime="2023-01-02T03:00:00Z",
                inferenceData=None,
                context=json.dumps({"key": "value9"}),
            ),
            Row(
                userId="U2",
                cpmFloorAdUnitIds=["ad_unit_6"],
                cpmFloorValues=[6.0],
                eventTime="2023-01-02T04:00:00Z",
                inferenceData="{'endpoint':'value1'}",
                context=json.dumps({"key": "value10"}),
            ),
            Row(
                userId="U3",
                cpmFloorAdUnitIds=["ad_unit_7"],
                cpmFloorValues=[7.0],
                eventTime="2023-01-03T00:00:00Z",
                inferenceData="{'endpoint':'value1'}",
                context=json.dumps({"key": "value11"}),
            ),
            Row(
                userId="U3",
                cpmFloorAdUnitIds=["ad_unit_7"],
                cpmFloorValues=[7.0],
                eventTime="2023-01-03T01:00:00Z",
                inferenceData=None,
                context=json.dumps({"key": "value12"}),
            ),
            Row(
                userId="U3",
                cpmFloorAdUnitIds=["ad_unit_8"],
                cpmFloorValues=[8.0],
                eventTime="2023-01-03T02:00:00Z",
                inferenceData="{'endpoint':'value1'}",
                context=json.dumps({"key": "value13"}),
            ),
            Row(
                userId="U3",
                cpmFloorAdUnitIds=["ad_unit_8"],
                cpmFloorValues=[8.0],
                eventTime="2023-01-03T03:00:00Z",
                inferenceData=None,
                context=json.dumps({"key": "value14"}),
            ),
            Row(
                userId="U3",
                cpmFloorAdUnitIds=["ad_unit_9"],
                cpmFloorValues=[9.0],
                eventTime="2023-01-03T04:00:00Z",
                inferenceData="{'endpoint':'value1'}",
                context=json.dumps({"key": "value15"}),
            ),
            Row(
                userId="U4",
                cpmFloorAdUnitIds=["ad_unit_10"],
                cpmFloorValues=[10.0],
                eventTime="2023-01-04T00:00:00Z",
                inferenceData="{'endpoint':'value1'}",
                context=json.dumps({"key": "value16"}),
            ),
            Row(
                userId="U4",
                cpmFloorAdUnitIds=["ad_unit_10"],
                cpmFloorValues=[10.0],
                eventTime="2023-01-04T01:00:00Z",
                inferenceData=None,
                context=json.dumps({"key": "value17"}),
            ),
            Row(
                userId="U4",
                cpmFloorAdUnitIds=["ad_unit_11"],
                cpmFloorValues=[11.0],
                eventTime="2023-01-04T02:00:00Z",
                inferenceData="{'endpoint':'value1'}",
                context=json.dumps({"key": "value18"}),
            ),
            Row(
                userId="U4",
                cpmFloorAdUnitIds=["ad_unit_11"],
                cpmFloorValues=[11.0],
                eventTime="2023-01-04T03:00:00Z",
                inferenceData=None,
                context=json.dumps({"key": "value19"}),
            ),
            Row(
                userId="U4",
                cpmFloorAdUnitIds=["ad_unit_12"],
                cpmFloorValues=[12.0],
                eventTime="2023-01-04T04:00:00Z",
                inferenceData="{'endpoint':'value1'}",
                context=json.dumps({"key": "value20"}),
            ),
            Row(
                userId="U5",
                cpmFloorAdUnitIds=["ad_unit_13"],
                cpmFloorValues=[13.0],
                eventTime="2023-01-05T00:00:00Z",
                inferenceData="{'endpoint':'value1'}",
                context=json.dumps({"key": "value21"}),
            ),
            Row(
                userId="U5",
                cpmFloorAdUnitIds=["ad_unit_13"],
                cpmFloorValues=[13.0],
                eventTime="2023-01-05T01:00:00Z",
                inferenceData=None,
                context=json.dumps({"key": "value22"}),
            ),
            Row(
                userId="U5",
                cpmFloorAdUnitIds=["ad_unit_14"],
                cpmFloorValues=[14.0],
                eventTime="2023-01-05T02:00:00Z",
                inferenceData="{'endpoint':'value1'}",
                context=json.dumps({"key": "value23"}),
            ),
            Row(
                userId="U5",
                cpmFloorAdUnitIds=["ad_unit_14"],
                cpmFloorValues=[14.0],
                eventTime="2023-01-05T03:00:00Z",
                inferenceData=None,
                context=json.dumps({"key": "value24"}),
            ),
            Row(
                userId="U5",
                cpmFloorAdUnitIds=["ad_unit_15"],
                cpmFloorValues=[15.0],
                eventTime="2023-01-05T04:00:00Z",
                inferenceData=None,
                context=json.dumps({"key": "value25"}),
            ),
        ]
        df = spark.createDataFrame(data, schema=schema)
        result_df = fill_with_cached_context(df)
        result_df.show(truncate=False)
        assert result_df.count() == 25
        assert len(result_df.select(Schema.CONTEXT).distinct().collect()) == 15
        assert len(result_df.select(Schema.LIVE_CONTEXT).distinct().collect()) == 25

    def test_fill_with_cached_context_adds_inference_data_column_when_missing(self, spark):
        """Test that INFERENCE_DATA column is added when missing from the DataFrame."""
        schema = StructType(
            [
                StructField(Schema.USER_ID, StringType(), True),
                StructField(Schema.CPM_FLOOR_AD_UNIT_IDS, ArrayType(StringType()), True),
                StructField(Schema.CPM_FLOOR_VALUES, ArrayType(DoubleType()), True),
                StructField(Schema.EVENT_TIME, StringType(), True),
                StructField(Schema.CONTEXT, StringType(), True),
                # Note: INFERENCE_DATA column is intentionally missing
            ]
        )
        data = [
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_1"],
                cpmFloorValues=[1.0],
                eventTime="2023-01-01T00:00:00Z",
                context="{}",
            ),
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_1"],
                cpmFloorValues=[1.0],
                eventTime="2023-01-01T01:00:00Z",
                context="{}",
            ),
        ]
        df = spark.createDataFrame(data, schema=schema)

        # Verify INFERENCE_DATA column is missing initially
        assert Schema.INFERENCE_DATA not in df.columns

        result_df = fill_with_cached_context(df)

        # Verify INFERENCE_DATA column is now present
        assert Schema.INFERENCE_DATA in result_df.columns
        assert result_df.count() == 2

        # Verify the column has the correct data type (StringType)
        inference_data_col = result_df.schema[Schema.INFERENCE_DATA]
        assert isinstance(inference_data_col.dataType, StringType)

        # Verify all values are null as expected
        null_count = result_df.filter(col(Schema.INFERENCE_DATA).isNull()).count()
        assert null_count == 2

    def test_fill_with_cached_context_preserves_existing_inference_data_column(self, spark):
        """Test that existing INFERENCE_DATA column is preserved when already present."""
        schema = StructType(
            [
                StructField(Schema.USER_ID, StringType(), True),
                StructField(Schema.CPM_FLOOR_AD_UNIT_IDS, ArrayType(StringType()), True),
                StructField(Schema.CPM_FLOOR_VALUES, ArrayType(DoubleType()), True),
                StructField(Schema.EVENT_TIME, StringType(), True),
                StructField(Schema.INFERENCE_DATA, StringType(), True),
                StructField(Schema.CONTEXT, StringType(), True),
            ]
        )
        data = [
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_1"],
                cpmFloorValues=[1.0],
                eventTime="2023-01-01T00:00:00Z",
                inferenceData="{'endpoint':'test'}",
                context="{}",
            ),
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_1"],
                cpmFloorValues=[1.0],
                eventTime="2023-01-01T01:00:00Z",
                inferenceData=None,
                context="{}",
            ),
        ]
        df = spark.createDataFrame(data, schema=schema)

        # Verify INFERENCE_DATA column exists initially
        assert Schema.INFERENCE_DATA in df.columns

        result_df = fill_with_cached_context(df)

        # Verify INFERENCE_DATA column is still present
        assert Schema.INFERENCE_DATA in result_df.columns
        assert result_df.count() == 2

        # Verify the original values are preserved
        rows = result_df.orderBy(Schema.EVENT_TIME).collect()
        assert rows[0][Schema.INFERENCE_DATA] == "{'endpoint':'test'}"
        assert rows[1][Schema.INFERENCE_DATA] is None

    def test_fill_with_cached_context_handles_none_input(self, spark):
        """Test that the function raises ValueError when input DataFrame is None."""
        with pytest.raises(ValueError, match="assignment_df cannot be None"):
            fill_with_cached_context(None)

    def test_fill_with_cached_context_with_different_inference_data_types(self, spark):
        """Test that the function handles different types of inference data values correctly."""
        schema = StructType(
            [
                StructField(Schema.USER_ID, StringType(), True),
                StructField(Schema.CPM_FLOOR_AD_UNIT_IDS, ArrayType(StringType()), True),
                StructField(Schema.CPM_FLOOR_VALUES, ArrayType(DoubleType()), True),
                StructField(Schema.EVENT_TIME, StringType(), True),
                StructField(Schema.INFERENCE_DATA, StringType(), True),
                StructField(Schema.CONTEXT, StringType(), True),
            ]
        )
        data = [
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_1"],
                cpmFloorValues=[1.0],
                eventTime="2023-01-01T00:00:00Z",
                inferenceData="{'endpoint':'value1', 'model':'model1'}",
                context="{}",
            ),
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_1"],
                cpmFloorValues=[1.0],
                eventTime="2023-01-01T01:00:00Z",
                inferenceData="null",  # String "null"
                context="{}",
            ),
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_1"],
                cpmFloorValues=[1.0],
                eventTime="2023-01-01T02:00:00Z",
                inferenceData="",  # Empty string
                context="{}",
            ),
            Row(
                userId="U1",
                cpmFloorAdUnitIds=["ad_unit_1"],
                cpmFloorValues=[1.0],
                eventTime="2023-01-01T03:00:00Z",
                inferenceData=None,  # Actual null
                context="{}",
            ),
        ]
        df = spark.createDataFrame(data, schema=schema)
        result_df = fill_with_cached_context(df)

        assert result_df.count() == 4
        assert Schema.INFERENCE_DATA in result_df.columns

        # Verify the function processes all rows correctly
        rows = result_df.orderBy(Schema.EVENT_TIME).collect()
        assert rows[0][Schema.INFERENCE_DATA] == "{'endpoint':'value1', 'model':'model1'}"
        assert rows[1][Schema.INFERENCE_DATA] == "null"
        assert rows[2][Schema.INFERENCE_DATA] == ""
        assert rows[3][Schema.INFERENCE_DATA] is None
