import datetime
import logging
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from etl_py_commons.job_initialiser import Initialisation
from pyspark.sql import SparkSession, DataFrame, Column, functions as F, Window
from pyspark.sql.functions import col, current_timestamp, from_json, hour, coalesce
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

from bid_optim_etl_py.cfg_parser import ConfigFile
from bid_optim_etl_py.command_line_args import ApplovinETLConfigParser
from bid_optim_etl_py.spark.iceberg import IcebergIO, Maintenance, TableConfig
from bid_optim_etl_py.utils.management_api import BidFloorManagementAPI, HttpClient


class Schema:
    REQUEST_ID = "requestId"
    USER_ID = "userId"
    TOTAL_AMOUNT = "totalAmount"
    EVENT_TIME = "eventTime"
    CONTEXT = "context"
    LIVE_CONTEXT = "liveContext"
    IS_FILLED = "isFilled"
    CPM_FLOOR_AD_UNIT_ID = "cpmFloorAdUnitId"
    CPM_FLOOR_AD_UNIT_IDS = "cpmFloorAdUnitIds"
    CPM_FLOOR_VALUE = "cpmFloorValue"
    CPM_FLOOR_VALUES = "cpmFloorValues"
    PROPENSITY = "propensity"
    DATE = "date"
    LAST_UPDATE_TIME = "lastUpdateTime"
    CUSTOMER_ID = "customerId"
    APP_ID = "appId"
    MODEL_ID = "modelId"
    ASSIGNMENT_HOUR_OF_DAY = "assignmentHourOfDay"
    ASSIGNMENT_DAY_OF_WEEK = "assignmentDayOfWeek"
    HIGHEST_BID_FLOOR_VALUE = "highestBidFloorValue"
    MEDIUM_BID_FLOOR_VALUE = "mediumBidFloorValue"
    INFERENCE_DATA = "inferenceData"


class ApplovinETLException(Exception):
    pass


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
    management_api: BidFloorManagementAPI
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
            Schema.DATE,
        ]

        self.etl_config = self.management_api.fetch_etl_config(app_id=self.app_id)
        self.max_ad_units = self.etl_config.maxAdUnits if self.etl_config.maxAdUnits is not None else 3
        self.logger.info(f"Fetched ETL config for app {self.app_id}: {self.etl_config}")

    def fetch_context_schema(self):
        context_schema = StructType([])
        if self.etl_config.context:
            context_schema = StructType(
                [
                    StructField(
                        context.path, StringType() if context.dataType.lower() == "string" else DoubleType(), True
                    )
                    for context in self.etl_config.context
                ]
            )
            self.logger.info(f"Context schema for app {self.app_id}: {context_schema}")
        else:
            self.logger.warning(f"No context fields found for app {self.app_id}. Using default schema.")
        return context_schema

    def read_events_parquet(self, event_name, columns: Optional[list[str]] = None):
        path = (
            f"{sanitise_path(self.s3_data_bucket)}/ingested-events-parquet/"
            f"customerId={self.customer_id}/appId={self.app_id}/{event_name}/"
        )
        try:
            dataset = self.spark.read.option("mergeSchema", "true").parquet(path)
            return dataset.select(columns) if columns else dataset
        except Exception as e:
            if "Path does not exist" in str(e):
                self.logger.info(f"No data found for {event_name} in {path}.")
                return self.spark.createDataFrame([], schema=StructType([]))
            else:
                raise ApplovinETLException(f"Error reading parquet file from {path}: {e}")
        except Exception as e:
            raise ApplovinETLException(f"Error reading parquet file from {path}: {e}")

    def has_valid_bid_floor_values(self) -> Column:
        return (
            col(Schema.CPM_FLOOR_VALUES).isNotNull().__and__(F.size(col(Schema.CPM_FLOOR_VALUES)) >= self.max_ad_units)
        )

    def valid_context_values(self) -> Column:
        return col(Schema.CONTEXT).isNotNull().__and__(col(Schema.CONTEXT).__ne__(""))

    def valid_revenue_rows(self) -> Column:
        return col(Schema.TOTAL_AMOUNT).isNotNull().__and__(col(Schema.TOTAL_AMOUNT).__ge__(0))

    def valid_propensity_values(self) -> Column:
        return (
            col(Schema.PROPENSITY)
            .isNotNull()
            .__and__(col(Schema.PROPENSITY).__gt__(0.0))
            .__and__(col(Schema.PROPENSITY).__le__(1.0))
        )

    def add_hardcoded_contexts(self, df: DataFrame):
        return df.withColumns(
            {
                Schema.ASSIGNMENT_HOUR_OF_DAY: hour(col(Schema.EVENT_TIME).cast("timestamp")),
                Schema.ASSIGNMENT_DAY_OF_WEEK: F.expr(f"weekday({Schema.EVENT_TIME})"),  # 0-6, Monday-Sunday
                Schema.HIGHEST_BID_FLOOR_VALUE: F.element_at(col(Schema.CPM_FLOOR_VALUES), -2),
                Schema.MEDIUM_BID_FLOOR_VALUE: F.element_at(col(Schema.CPM_FLOOR_VALUES), -3),
            }
        )

    def fetch_assignment_events(self):
        assignment_event = self.read_events_parquet("metica_bid_floor_assignment")
        if assignment_event.isEmpty():
            self.logger.info(f"No assignment events found for date {self.date_iso}.")
            return assignment_event
        return self.add_hardcoded_contexts(
            assignment_event.filter(
                (col(Schema.DATE) <= self.date_iso)
                & self.valid_context_values()
                & self.has_valid_bid_floor_values()
                & self.valid_propensity_values()
            )
        )

    def fetch_revenue_events(self):
        ad_revenue_event = self.read_events_parquet("estimated_ad_revenue", self.ad_revenue_columns)
        if ad_revenue_event.isEmpty():
            self.logger.info(f"No ad revenue events found for date {self.date_iso}.")
            return ad_revenue_event
        return ad_revenue_event.filter((col(Schema.DATE) <= self.date_iso) & self.valid_revenue_rows())

    def fetch_bid_sequence_events(self):
        bid_sequence_event = self.read_events_parquet("applovin_bid_floor", self.bid_sequence_columns)
        if bid_sequence_event.isEmpty():
            self.logger.info(f"No bid sequence events found for date {self.date_iso}.")
            return bid_sequence_event
        return bid_sequence_event.filter(col(Schema.IS_FILLED) & (col(Schema.DATE) <= self.date_iso))

    def join_assignment_and_revenue(self, assignment_data: DataFrame, ad_revenue_data: DataFrame):
        return assignment_data.join(
            ad_revenue_data.select(self.ad_revenue_columns).drop(Schema.DATE),
            on=[Schema.REQUEST_ID, Schema.USER_ID],
            how="left",
        ).withColumns(
            {
                Schema.EVENT_TIME: col(Schema.EVENT_TIME).cast("timestamp"),
                Schema.DATE: col(Schema.EVENT_TIME).cast("date"),
                Schema.LAST_UPDATE_TIME: current_timestamp(),
                Schema.TOTAL_AMOUNT: coalesce(col(Schema.TOTAL_AMOUNT), F.lit(0.0)),
            }
        )

    def denormalise_context_field(self, df: DataFrame, context_schema: StructType):
        # Parse the JSON string in the 'context' column
        df = df.withColumn("context_json", from_json(col("context"), context_schema))
        context_fields = {}

        for field in context_schema.fields:
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


def fill_with_cached_context(assignment_df: DataFrame) -> DataFrame:
    """
    Fills the assignment DataFrame with cached context values based on the user ID and ad unit IDs.
    :param assignment_df: DataFrame containing assignment events with user ID, ad unit IDs, and context.
    :return: DataFrame with filled context values.
    """
    if assignment_df.isEmpty():
        return assignment_df
    window_spec = Window.partitionBy(Schema.USER_ID, Schema.CPM_FLOOR_AD_UNIT_IDS).orderBy(Schema.EVENT_TIME)
    is_new_group = "is_new_group"
    group_id = "group_id"

    assignment_df = assignment_df.withColumn(
        is_new_group,
        F.when(
            (col(Schema.INFERENCE_DATA).isNotNull() & ~(col(Schema.INFERENCE_DATA).eqNullSafe("null")))
            | (F.lag(Schema.USER_ID, 1).over(window_spec) != col(Schema.USER_ID))
            | (F.lag(Schema.CPM_FLOOR_AD_UNIT_IDS, 1).over(window_spec) != col(Schema.CPM_FLOOR_AD_UNIT_IDS))
            | (F.lag(Schema.CPM_FLOOR_VALUES, 1).over(window_spec) != col(Schema.CPM_FLOOR_VALUES)),
            1,
        ).otherwise(0),
    )

    assignment_df = assignment_df.withColumn(group_id, F.sum(is_new_group).over(window_spec)).withColumnsRenamed(
        {Schema.CONTEXT: Schema.LIVE_CONTEXT}
    )

    assignment_df = assignment_df.withColumn(
        Schema.CONTEXT,
        F.first(Schema.LIVE_CONTEXT, ignorenulls=True).over(
            Window.partitionBy(Schema.USER_ID, Schema.CPM_FLOOR_AD_UNIT_IDS, group_id)
            .orderBy(Schema.EVENT_TIME)
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        ),
    )

    return assignment_df.drop(is_new_group, group_id)


def extract_events(spark: SparkSession, logger: logging.Logger, parsed_args_obj: Namespace, config_file: ConfigFile):
    events = Events(
        customer_id=parsed_args_obj.customerId,
        app_id=parsed_args_obj.appId,
        s3_data_bucket=parsed_args_obj.s3DataBucket,
        date=datetime.date.fromisoformat(parsed_args_obj.date),
        spark=spark,
        iceberg_catalog=parsed_args_obj.icebergCatalog,
        region_name=parsed_args_obj.region,
        logger=logger,
        management_api=BidFloorManagementAPI(
            http_client=HttpClient(base_url=config_file.managementApiBaseUrl),
        ),
    )

    assignments = fill_with_cached_context(events.fetch_assignment_events())
    ad_revenue_df = events.fetch_revenue_events()

    if assignments.isEmpty() or ad_revenue_df.isEmpty():
        logger.info(f"No data found for the given date {events.date_iso}. Exiting.")
        return

    context_schema = events.fetch_context_schema()
    assignment_contexts_denormalised = events.denormalise_context_field(assignments, context_schema)
    final_df = events.join_assignment_and_revenue(assignment_contexts_denormalised, ad_revenue_df)
    events.save_as_iceberg(final_df)
    events.perform_maintenance()


def run(spark: SparkSession, args: [str]):
    try:
        parsed_args_obj = Initialisation.parse_args(args=args, parser_obj=ApplovinETLConfigParser())
        logger = Initialisation.fetch_logger(spark, __name__)
        _log_start(logger=logger, args=args)
        config_file = parsed_args_obj.read_config(Path(__file__).parent.joinpath("confs"), ConfigFile)
        extract_events(spark=spark, logger=logger, parsed_args_obj=parsed_args_obj.parsed_args, config_file=config_file)
        _log_complete(logger=logger, args=args)
    except Exception as exp:
        logging.exception("Error while running Applovin ETL")
        raise exp
