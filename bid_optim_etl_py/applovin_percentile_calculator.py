import datetime
import logging
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from etl_py_commons.job_initialiser import Initialisation
from pyspark.sql import SparkSession, DataFrame, Column, functions as F
from pyspark.sql.functions import col, from_json, coalesce, lit
from pyspark.sql.types import StructType, StructField, StringType

from bid_optim_etl_py.cfg_parser import ConfigFile
from bid_optim_etl_py.command_line_args import ApplovinPercentileCalculatorArgsParser
from bid_optim_etl_py.utils.management_api import BidFloorManagementAPI, HttpClient
from bid_optim_etl_py.helpers.aws_helpers import S3Helper
from bid_optim_etl_py.helpers.data_helpers import format_s3_key

from bid_optim_etl_py.constants import BID_FLOOR_PERCENTILES_PREFIX, S3_ARTIFACTS_BUCKET


class Schema:
    REQUEST_ID = "requestId"
    USER_ID = "userId"
    TOTAL_AMOUNT = "totalAmount"
    EVENT_TIME = "eventTime"
    CONTEXT = "context"
    DATE = "date"
    CUSTOMER_ID = "customerId"
    APP_ID = "appId"
    CPM_FLOOR_AD_UNIT_ID = "cpmFloorAdUnitId"
    CPM_FLOOR_VALUES = "cpmFloorValues"
    USER_COUNTRY = "user.country"
    USER_PLATFORM = "user.platform"
    USER_ADFORMAT = "user.adformat"


class ApplovinPercentileException(Exception):
    pass


def sanitise_path(path):
    return path.strip("/")


@dataclass
class PercentileCalculator:
    customer_id: int
    app_id: int
    s3_data_bucket: str
    date: datetime.date
    spark: SparkSession
    region_name: str
    management_api: BidFloorManagementAPI
    logger: logging.Logger
    cut_off_days: int = 7

    def __post_init__(self):
        self.date_iso = self.date.isoformat()

        # Calculate cutoff date for event time filtering
        self.cutoff_date = self.date - datetime.timedelta(days=self.cut_off_days)
        self.cutoff_date_iso = self.cutoff_date.isoformat()

        self.logger.info(
            f"Using cutoff date: {self.cutoff_date_iso} (looking back {self.cut_off_days} days from {self.date_iso})"
        )

        # Columns needed for revenue events
        self.ad_revenue_columns = [
            Schema.REQUEST_ID,
            Schema.USER_ID,
            Schema.CPM_FLOOR_AD_UNIT_ID,
            Schema.TOTAL_AMOUNT,
            Schema.DATE,
            Schema.EVENT_TIME,
        ]

        # Columns needed for assignment events (to get context/country)
        self.assignment_columns = [
            Schema.REQUEST_ID,
            Schema.USER_ID,
            Schema.CONTEXT,
            Schema.DATE,
            Schema.EVENT_TIME,
        ]

        self.etl_config = self.management_api.fetch_etl_config(app_id=self.app_id)
        self.logger.info(f"Fetched ETL config for app {self.app_id}: {self.etl_config}")

    def fetch_context_schema(self):
        """Create a context schema that includes user.country, user.platform, and user.adformat"""
        context_schema = StructType(
            [
                StructField(Schema.USER_COUNTRY, StringType(), True),
                StructField(Schema.USER_PLATFORM, StringType(), True),
                StructField(Schema.USER_ADFORMAT, StringType(), True),
            ]
        )
        self.logger.info(f"Using context schema for country, platform, and adformat extraction: {context_schema}")
        return context_schema

    def read_events_parquet(self, event_name, columns: Optional[list[str]] = None):
        """Read parquet events from S3"""
        path = (
            f"{sanitise_path(self.s3_data_bucket)}/ingested-events-parquet/"
            f"customerId={self.customer_id}/appId={self.app_id}/{event_name}/"
        )
        # Convert s3:// to s3a:// for Spark compatibility
        if path.startswith("s3://"):
            path = path.replace("s3://", "s3a://", 1)

        try:
            dataset = self.spark.read.option("mergeSchema", "true").parquet(path)
            return dataset.select(columns) if columns else dataset
        except Exception as e:
            if "Path does not exist" in str(e):
                self.logger.info(f"No data found for {event_name} in {path}.")
                return self.spark.createDataFrame([], schema=StructType([]))
            else:
                raise ApplovinPercentileException(f"Error reading parquet file from {path}: {e}")

    def valid_revenue_rows(self) -> Column:
        """Filter for valid revenue data"""
        return col(Schema.TOTAL_AMOUNT).isNotNull().__and__(col(Schema.TOTAL_AMOUNT).__ge__(0))

    def valid_context_values(self) -> Column:
        """Filter for valid context data"""
        return col(Schema.CONTEXT).isNotNull().__and__(col(Schema.CONTEXT).__ne__(""))

    def fetch_revenue_events(self):
        """Fetch revenue events with total amount data"""
        ad_revenue_event = self.read_events_parquet("estimated_ad_revenue", self.ad_revenue_columns)
        if ad_revenue_event.isEmpty():
            self.logger.info(f"No ad revenue events found for date {self.date_iso}.")
            return ad_revenue_event

        # Filter by event time within cutoff days
        return ad_revenue_event.filter(
            (col(Schema.EVENT_TIME).cast("timestamp").cast("date") >= self.cutoff_date_iso)
            & (col(Schema.EVENT_TIME).cast("timestamp").cast("date") <= self.date_iso)
            & self.valid_revenue_rows()
        )

    def fetch_assignment_events(self):
        """Fetch assignment events to get context/country information"""
        assignment_event = self.read_events_parquet("metica_bid_floor_assignment", self.assignment_columns)
        if assignment_event.isEmpty():
            self.logger.info(f"No assignment events found for date {self.date_iso}.")
            return assignment_event

        # Filter by event time within cutoff days
        return assignment_event.filter(
            (col(Schema.EVENT_TIME).cast("timestamp").cast("date") >= self.cutoff_date_iso)
            & (col(Schema.EVENT_TIME).cast("timestamp").cast("date") <= self.date_iso)
            & self.valid_context_values()
        )

    def extract_context_fields(self, df: DataFrame, context_schema: StructType):
        """Extract country, platform, and adformat information from context JSON"""
        # Parse the JSON string in the 'context' column
        df = df.withColumn("context_json", from_json(col("context"), context_schema))

        # Extract context fields
        df = df.withColumn(Schema.USER_COUNTRY, col("context_json").getItem(Schema.USER_COUNTRY))
        df = df.withColumn(Schema.USER_PLATFORM, col("context_json").getItem(Schema.USER_PLATFORM))
        df = df.withColumn(Schema.USER_ADFORMAT, col("context_json").getItem(Schema.USER_ADFORMAT))

        # Drop the intermediate JSON column
        df = df.drop("context_json")

        return df

    def join_assignment_and_revenue(self, assignment_data: DataFrame, ad_revenue_data: DataFrame):
        """Join assignment data (for country) with revenue data (for total amount)"""
        return assignment_data.join(
            ad_revenue_data.select(self.ad_revenue_columns).drop(Schema.DATE),
            on=[Schema.REQUEST_ID, Schema.USER_ID],
            how="inner",  # Only include records that have both assignment and revenue data
        ).withColumns(
            {
                Schema.TOTAL_AMOUNT: coalesce(col(Schema.TOTAL_AMOUNT), lit(0.0)),
            }
        )

    def calculate_percentiles_by_country_platform_adformat(self, df: DataFrame):
        """Calculate percentiles of total amount grouped by country, platform, and adformat"""
        if df.isEmpty():
            self.logger.warning("No data available for percentile calculation")
            return df

        # Filter out null values for required fields
        df_filtered = df.filter(
            col(f"`{Schema.USER_COUNTRY}`").isNotNull()
            & col(f"`{Schema.USER_PLATFORM}`").isNotNull()
            & col(f"`{Schema.USER_ADFORMAT}`").isNotNull()
        )

        # Remove duplicates with the same requestId, keeping the first occurrence
        df_filtered = df_filtered.dropDuplicates([Schema.REQUEST_ID])
        if df_filtered.isEmpty():
            self.logger.warning("No data with valid country, platform, and adformat information")
            return df_filtered

        # Calculate percentiles and statistics by country, platform, and adformat
        percentile_exprs = {
            "count": F.count("*").alias("count"),
            "p10": F.expr("percentile_approx(totalAmount, 0.10)").alias("p10"),
            "p20": F.expr("percentile_approx(totalAmount, 0.20)").alias("p20"),
            "p30": F.expr("percentile_approx(totalAmount, 0.30)").alias("p30"),
            "p40": F.expr("percentile_approx(totalAmount, 0.40)").alias("p40"),
            "p50": F.expr("percentile_approx(totalAmount, 0.50)").alias("p50"),
            "p60": F.expr("percentile_approx(totalAmount, 0.60)").alias("p60"),
            "p70": F.expr("percentile_approx(totalAmount, 0.70)").alias("p70"),
            "p80": F.expr("percentile_approx(totalAmount, 0.80)").alias("p80"),
            "p85": F.expr("percentile_approx(totalAmount, 0.85)").alias("p85"),
            "p90": F.expr("percentile_approx(totalAmount, 0.90)").alias("p90"),
            "mean": F.avg(col(Schema.TOTAL_AMOUNT)).alias("mean"),
            "min": F.min(col(Schema.TOTAL_AMOUNT)).alias("min"),
            "max": F.max(col(Schema.TOTAL_AMOUNT)).alias("max"),
        }

        result_df = df_filtered.groupBy(
            col(f"`{Schema.USER_COUNTRY}`"), col(f"`{Schema.USER_PLATFORM}`"), col(f"`{Schema.USER_ADFORMAT}`")
        ).agg(*percentile_exprs.values())
        # Rename columns to match expected output formtry, platform, and adformat
        result_df = result_df.orderBy(
            f"`{Schema.USER_COUNTRY}`", f"`{Schema.USER_PLATFORM}`", f"`{Schema.USER_ADFORMAT}`"
        )

        return result_df

    def save_to_s3(self, df: DataFrame, output_path: str):
        """Save the percentile results to S3 as CSV"""
        self.logger.info(f"Saving percentile results to S3: {output_path}")

        # Convert to Pandas for CSV output (since we expect small result set)
        pandas_df = df.toPandas()

        # Save to S3 using boto3
        import boto3
        import io

        s3_client = boto3.client("s3", region_name=self.region_name)

        # Convert DataFrame to CSV string
        csv_buffer = io.StringIO()
        pandas_df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        # Extract bucket and key from path
        if output_path.startswith("s3://"):
            path_parts = output_path[5:].split("/", 1)
            bucket = path_parts[0]
            key = path_parts[1] if len(path_parts) > 1 else ""
        else:
            raise ApplovinPercentileException(f"Invalid S3 path format: {output_path}")

        # Upload to S3
        s3_client.put_object(Bucket=bucket, Key=key, Body=csv_content, ContentType="text/csv")

        self.logger.info(f"Successfully saved percentile results to s3://{bucket}/{key}")


def save_percentiles_to_s3(
    percentiles_df: DataFrame, customer_id: int, app_id: int, platform: str, ad_type: str, logger: logging.Logger
) -> str:
    """Save percentile results to S3 in the specified bucket structure."""
    try:
        s3_helper = S3Helper()
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        s3_key = format_s3_key(BID_FLOOR_PERCENTILES_PREFIX, customer_id, app_id, today, platform, ad_type)

        # Convert DataFrame to Pandas for JSON conversion
        pandas_df = percentiles_df.toPandas()

        percentiles_json = pandas_df.to_json(orient="records")

        pandas_df.to_csv("tmp.csv", index=False)

        s3_path = s3_helper.write_json(S3_ARTIFACTS_BUCKET, s3_key, percentiles_json)
        logger.info(f"Successfully saved percentiles to S3: {s3_path}")
        return s3_path

    except Exception as e:
        logger.error(f"Error saving percentiles to S3: {e}")
        raise


def _log_start(logger: logging.Logger, args: [str]):
    logger.info(f"Started Applovin Percentile Calculator PySpark job with args: {args}")


def _log_complete(logger: logging.Logger, args: [str]):
    logger.info(f"Completed Applovin Percentile Calculator PySpark job with args: {args}")


def calculate_percentiles(
    spark: SparkSession, logger: logging.Logger, parsed_args_obj: Namespace, config_file: ConfigFile
):
    """Main function to calculate percentiles by country"""
    calculator = PercentileCalculator(
        customer_id=parsed_args_obj.customerId,
        app_id=parsed_args_obj.appId,
        s3_data_bucket=parsed_args_obj.s3DataBucket,
        date=datetime.date.fromisoformat(parsed_args_obj.date),
        spark=spark,
        region_name=parsed_args_obj.region,
        logger=logger,
        management_api=BidFloorManagementAPI(
            http_client=HttpClient(base_url=config_file.managementApiBaseUrl),
        ),
        cut_off_days=parsed_args_obj.cutOffDays,
    )

    # Fetch assignment events (for country context)
    assignments = calculator.fetch_assignment_events()

    # Fetch revenue events (for total amount)
    ad_revenue_df = calculator.fetch_revenue_events()

    if assignments.isEmpty() or ad_revenue_df.isEmpty():
        logger.info(f"No data found for the given date {calculator.date_iso}. Exiting.")
        return

    # Extract context fields (country, platform, adformat) from context
    context_schema = calculator.fetch_context_schema()
    assignments_with_context = calculator.extract_context_fields(assignments, context_schema)

    # Join assignment and revenue data
    joined_data = calculator.join_assignment_and_revenue(assignments_with_context, ad_revenue_df)

    if joined_data.isEmpty():
        logger.info("No data after joining assignment and revenue events. Exiting.")
        return

    # Calculate percentiles by country, platform, and adformat
    percentile_results = calculator.calculate_percentiles_by_country_platform_adformat(joined_data)

    if percentile_results.isEmpty():
        logger.info("No percentile results generated. Exiting.")
        return

    # Get unique platform and adformat combinations
    platform_adformat_combinations = (
        percentile_results.select(f"`{Schema.USER_PLATFORM}`", f"`{Schema.USER_ADFORMAT}`").distinct().collect()
    )

    logger.info(f"Found {platform_adformat_combinations} platform/adformat combinations")

    # Save separate files for each platform/adformat combination
    for row in platform_adformat_combinations:
        platform = row[f"{Schema.USER_PLATFORM}"]
        adformat = row[f"{Schema.USER_ADFORMAT}"]

        # Filter data for this specific platform/adformat combination
        filtered_data = percentile_results.filter(
            (col(f"`{Schema.USER_PLATFORM}`") == platform) & (col(f"`{Schema.USER_ADFORMAT}`") == adformat)
        )

        if not filtered_data.isEmpty():
            # Save to S3 using the new function
            s3_path = save_percentiles_to_s3(
                filtered_data, calculator.customer_id, calculator.app_id, platform, adformat, logger
            )
            logger.info(f"Saved percentiles for {platform}/{adformat} to: {s3_path}")

    logger.info(
        f"Percentile calculation completed successfully for {len(platform_adformat_combinations)} combinations."
    )


def run(spark: SparkSession, args: [str]):
    """Main entry point for the percentile calculator"""
    try:
        parsed_args_obj = Initialisation.parse_args(args=args, parser_obj=ApplovinPercentileCalculatorArgsParser())
        logger = Initialisation.fetch_logger(spark, __name__)
        _log_start(logger=logger, args=args)
        config_file = parsed_args_obj.read_config(Path(__file__).parent.joinpath("confs"), ConfigFile)
        calculate_percentiles(
            spark=spark, logger=logger, parsed_args_obj=parsed_args_obj.parsed_args, config_file=config_file
        )
        _log_complete(logger=logger, args=args)
    except Exception as exp:
        logging.exception("Error while running Applovin Percentile Calculator")
        raise exp
