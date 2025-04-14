import datetime
import logging
from dataclasses import dataclass
from typing import Optional

from pyspark.sql import SparkSession, DataFrame, Column, functions as F
from pyspark.sql.functions import col, current_timestamp, from_json, hour
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

from bid_optim_etl_py.spark.iceberg import IcebergIO, Maintenance, TableConfig


class Schema:
    REQUEST_ID = "requestId"
    USER_ID = "userId"
    TOTAL_AMOUNT = "totalAmount"
    EVENT_TIME = "eventTime"
    CONTEXT = "context"
    IS_FILLED = "isFilled"
    CPM_FLOOR_AD_UNIT_ID = "cpmFloorAdUnitId"
    CPM_FLOOR_VALUE = "cpmFloorValue"
    CPM_FLOOR_VALUES = "cpmFloorValues"
    DATE = "date"
    LAST_UPDATE_TIME = "lastUpdateTime"
    CUSTOMER_ID = "customerId"
    APP_ID = "appId"
    MODEL_ID = "modelId"
    ASSIGNMENT_HOUR_OF_DAY = "assignmentHourOfDay"
    ASSIGNMENT_DAY_OF_WEEK = "assignmentDayOfWeek"
    HIGHEST_BID_FLOOR_VALUE = "highestBidFloorValue"
    MEDIUM_BID_FLOOR_VALUE = "mediumBidFloorValue"


class ApplovinETLException(Exception):
    pass


def spark_log4j_logger(spark_session: SparkSession, logger_name: str):
    log4jLogger = spark_session.sparkContext._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger(logger_name)
    logger.setLevel(log4jLogger.Level.DEBUG)
    return logger


def arg_parser(args):
    import argparse

    parser = argparse.ArgumentParser(description="Run the applovin bid floor training")
    parser.add_argument("--customerId", type=int, help="Customer ID")
    parser.add_argument("--appId", type=int, help="App ID")
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    parser.add_argument(
        "--region", required=True, type=str, help="Region to load config (str). Ex: us-east-1, eu-west-1"
    )
    parser.add_argument("--s3DataBucket", help="S3 bucket name for all events and bid floor training data")
    parser.add_argument(
        "--icebergCatalog", required=True, type=str, help="Iceberg catalog name (str). Ex: iceberg_catalog."
    )
    return parser.parse_args(args)


def sanitise_path(path):
    return path.strip("/")


@dataclass
class Events:
    customer_id: int
    app_id: int
    s3_data_bucket: str
    date: datetime.date
    spark: SparkSession
    iceberg_catalog: str
    region_name: str
    logger: logging.Logger

    def __post_init__(self):
        self.iceberg_io = IcebergIO(
            spark=self.spark, region_name=self.region_name, iceberg_catalog=self.iceberg_catalog
        )
        self.maintenance = Maintenance(spark=self.spark, iceberg_catalog=self.iceberg_catalog)
        self.training_data_table_config = TableConfig(
            format="iceberg", database_name="applovin", table_name="bid_floor_training_data"
        )
        self.training_data_db_table_name = self.training_data_table_config.db_table_name(
            catalog_name=self.iceberg_catalog
        )
        self.date_iso = self.date.isoformat()

        self.ad_revenue_columns = [
            Schema.REQUEST_ID,
            Schema.USER_ID,
            Schema.CPM_FLOOR_AD_UNIT_ID,
            Schema.TOTAL_AMOUNT,
            Schema.DATE,
        ]

        self.bid_sequence_columns = [
            Schema.REQUEST_ID,
            Schema.USER_ID,
            Schema.IS_FILLED,
            Schema.CPM_FLOOR_AD_UNIT_ID,
            Schema.CPM_FLOOR_VALUE,
            Schema.DATE,
        ]

        self.context_schema = StructType(
            [
                StructField("user.country", StringType(), True),
                StructField("user.languageCode", StringType(), True),
                StructField("user.deviceType", StringType(), True),
                StructField("user.osVersion", StringType(), True),
                StructField("user.deviceModel", StringType(), True),
                StructField("assignmentDayOfWeek", IntegerType(), True),
                StructField("assignmentHourOfDay", IntegerType(), True),
                StructField("user.minRevenueLast24Hours", DoubleType(), True),
                StructField("user.avgRevenueLast24Hours", DoubleType(), True),
                StructField("user.avgRevenueLast48Hours", DoubleType(), True),
                StructField("user.avgRevenueLast72Hours", DoubleType(), True),
                StructField("user.mostRecentAdSource", DoubleType(), True),
                StructField("user.mostRecentAdRevenue", DoubleType(), True),
            ]
        )

    def read_events_parquet(self, event_name, columns: Optional[list[str]] = None):
        path = (
            f"{sanitise_path(self.s3_data_bucket)}/ingested-events-parquet/"
            f"customerId={self.customer_id}/appId={self.app_id}/{event_name}/"
        )
        try:
            dataset = self.spark.read.option("mergeSchema", "true").parquet(path)
            return dataset.select(columns) if columns else dataset
        except Exception as e:
            raise ApplovinETLException(f"Error reading parquet file from {path}: {e}")

    def has_valid_bid_floor_values(self) -> Column:
        return col(Schema.CPM_FLOOR_VALUES).isNotNull().__and__(F.size(col(Schema.CPM_FLOOR_VALUES)) >= 3)

    def valid_context_values(self) -> Column:
        return col(Schema.CONTEXT).isNotNull().__and__(col(Schema.CONTEXT).__ne__(""))

    def valid_revenue_rows(self) -> Column:
        return col(Schema.TOTAL_AMOUNT).isNotNull().__and__(col(Schema.TOTAL_AMOUNT).__ge__(0))

    def add_hardcoded_contexts(self, df: DataFrame):
        return df.withColumns(
            {
                Schema.ASSIGNMENT_HOUR_OF_DAY: hour(col(Schema.EVENT_TIME).cast("timestamp")),
                Schema.ASSIGNMENT_DAY_OF_WEEK: F.expr(f"weekday({Schema.EVENT_TIME})"),  # 0-6, Monday-Sunday
                Schema.HIGHEST_BID_FLOOR_VALUE: col(Schema.CPM_FLOOR_VALUES).getItem(0),
                Schema.MEDIUM_BID_FLOOR_VALUE: col(Schema.CPM_FLOOR_VALUES).getItem(1),
            }
        )

    def fetch_assignment_events(self):
        return self.add_hardcoded_contexts(
            self.read_events_parquet("metica_bid_floor_assignment").filter(
                (col(Schema.DATE) <= self.date_iso) & self.valid_context_values() & self.has_valid_bid_floor_values()
            )
        )

    def fetch_revenue_events(self):
        return self.read_events_parquet("estimated_ad_revenue", self.ad_revenue_columns).filter(
            (col(Schema.DATE) <= self.date_iso) & self.valid_revenue_rows()
        )

    def fetch_bid_sequence_events(self):
        return self.read_events_parquet("applovin_bid_floor", self.bid_sequence_columns).filter(
            col(Schema.IS_FILLED) & (col(Schema.DATE) <= self.date_iso)
        )

    def join_all(self, assignment_data: DataFrame, bid_sequence_data: DataFrame, ad_revenue_data: DataFrame):
        merged_df = assignment_data.join(
            ad_revenue_data.select(self.ad_revenue_columns).drop(Schema.DATE),
            on=[Schema.REQUEST_ID, Schema.USER_ID],
            how="inner",
        )
        return merged_df.join(
            bid_sequence_data.select(self.bid_sequence_columns).drop(Schema.DATE),
            on=[Schema.REQUEST_ID, Schema.USER_ID, Schema.CPM_FLOOR_AD_UNIT_ID],
            how="inner",
        ).withColumns(
            {
                Schema.EVENT_TIME: col(Schema.EVENT_TIME).cast("timestamp"),
                Schema.DATE: col(Schema.EVENT_TIME).cast("date"),
                Schema.LAST_UPDATE_TIME: current_timestamp(),
            }
        )

    def denormalise_context_field(self, df: DataFrame):
        # Parse the JSON string in the 'context' column
        df = df.withColumn("context_json", from_json(col("context"), self.context_schema))

        context_fields = {}
        for field in self.context_schema.fields:
            context_fields.update({field.name: col("context_json").getItem(field.name)})

        if context_fields:
            df = df.withColumns(context_fields)

        # Drop the intermediate JSON column
        df = df.drop("context_json")

        return df

    def save_as_iceberg(self, df: DataFrame):
        self.logger.info(f"Saving DataFrame to Iceberg table: {self.training_data_db_table_name}")
        self.iceberg_io.write(
            spark_df=df,
            table_config=self.training_data_table_config,
            partition_by=[Schema.CUSTOMER_ID, Schema.APP_ID, Schema.MODEL_ID],
            create_database=True,
            overwrite_partitions=True,
        )

    def perform_maintenance(self):
        older_than_7_days = datetime.datetime.combine(self.date, datetime.datetime.min.time()) - datetime.timedelta(
            days=7
        )
        self.maintenance.expire_snapshots(self.training_data_db_table_name, expire_older_than=older_than_7_days)
        self.maintenance.remove_orphan_files(self.training_data_db_table_name)
        self.maintenance.rewrite_data_files(self.training_data_db_table_name)


def _log_start(logger: logging.Logger, args: [str]):
    logger.info(f"Started Applovin ETL Runner PySpark job with args: {args}")


def _log_complete(logger: logging.Logger, args: [str]):
    logger.info(f"Completed Applovin ETL Runner PySpark job with args: {args}")


def extract_events(spark: SparkSession, logger: logging.Logger, args: [str]):
    parsed_args_obj = arg_parser(args)
    events = Events(
        customer_id=parsed_args_obj.customerId,
        app_id=parsed_args_obj.appId,
        s3_data_bucket=parsed_args_obj.s3DataBucket,
        date=datetime.date.fromisoformat(parsed_args_obj.date),
        spark=spark,
        iceberg_catalog=parsed_args_obj.icebergCatalog,
        region_name=parsed_args_obj.region,
        logger=logger,
    )
    assignments = events.fetch_assignment_events()
    bid_sequence_df = events.fetch_bid_sequence_events()
    ad_revenue_df = events.fetch_revenue_events()
    assignment_contexts_denormalised = events.denormalise_context_field(assignments)
    final_df = events.join_all(assignment_contexts_denormalised, bid_sequence_df, ad_revenue_df)
    events.save_as_iceberg(final_df)
    events.perform_maintenance()


def run(spark: SparkSession, args: [str]):
    try:
        logger = spark_log4j_logger(spark, __name__)
        _log_start(logger=logger, args=args)
        extract_events(spark=spark, logger=logger, args=args)
        _log_complete(logger=logger, args=args)
    except Exception as exp:
        logging.exception("Error while running Applovin ETL")
        raise exp
