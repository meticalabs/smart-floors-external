import json
import random
import uuid
from datetime import datetime, timedelta

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, ArrayType, BooleanType

# Define schemas
assignment_schema = StructType(
    [
        StructField("requestId", StringType(), False),
        StructField("placementTag", StringType(), True),
        StructField("sessionId", StringType(), False),
        StructField("eventType", StringType(), False),
        StructField("customerId", IntegerType(), False),
        StructField("appId", IntegerType(), False),
        StructField("modelId", StringType(), False),
        StructField("eventTime", StringType(), False),
        StructField("userId", StringType(), False),
        StructField("context", StringType(), False),
        StructField("cpmFloorAdUnitIds", ArrayType(StringType()), False),
        StructField("cpmFloorValues", ArrayType(FloatType()), False),
        StructField("propensity", FloatType(), False),
        StructField("inferenceData", StringType(), False),
    ]
)

bid_schema = StructType(
    [
        StructField("requestId", StringType(), False),
        StructField("placementTag", StringType(), True),
        StructField("sessionId", StringType(), False),
        StructField("eventType", StringType(), False),
        StructField("customerId", IntegerType(), False),
        StructField("appId", IntegerType(), False),
        StructField("userId", StringType(), False),
        StructField("cpmFloorAdUnitId", StringType(), False),
        StructField("isFilled", BooleanType(), False),
        StructField("winningBidValue", FloatType(), True),
        StructField("eventTime", StringType(), False),
    ]
)

revenue_schema = StructType(
    [
        StructField("requestId", StringType(), False),
        StructField("placementTag", StringType(), True),
        StructField("sessionId", StringType(), False),
        StructField("eventType", StringType(), False),
        StructField("cpmFloorAdUnitId", StringType(), False),
        StructField("adFormat", StringType(), False),
        StructField("adSource", StringType(), False),
        StructField("customerId", IntegerType(), False),
        StructField("appId", IntegerType(), False),
        StructField("eventTime", StringType(), False),
        StructField("userId", StringType(), False),
        StructField("totalAmount", FloatType(), False),
        StructField("currencyCode", StringType(), False),
    ]
)


def generate_random_context():
    countries = ["US", "CA", "UK", "IN", "AU", None]
    language_codes = ["en", "fr", "es", "hi", None]
    device_types = ["mobile", "tablet", None]
    os_versions = ["10.0", "11.0", "12.0", "13.0", None]
    device_models = ["iPhone12", "GalaxyS21", "Pixel6", None]
    ad_sources = ["Applovin", "Google Ads", "Unity Ads", None]

    context = {
        "user.country": random.choice(countries),
        "user.languageCode": random.choice(language_codes),
        "user.deviceType": random.choice(device_types),
        "user.osVersion": random.choice(os_versions),
        "user.deviceModel": random.choice(device_models),
        "user.minRevenueLast24Hours": round(random.uniform(0.0, 10.0), 2) if random.choice([True, False]) else None,
        "user.avgRevenueLast24Hours": round(random.uniform(0.0, 5.0), 2) if random.choice([True, False]) else None,
        "user.avgRevenueLast48Hours": round(random.uniform(0.0, 5.0), 2) if random.choice([True, False]) else None,
        "user.avgRevenueLast72Hours": round(random.uniform(0.0, 5.0), 2) if random.choice([True, False]) else None,
        "user.mostRecentAdSource": random.choice(ad_sources),
        "user.mostRecentAdRevenue": round(random.uniform(0.0, 10.0), 2) if random.choice([True, False]) else None,
        "assignmentDayOfWeek": random.randint(0, 6) if random.choice([True, False]) else None,
        "assignmentHourOfDay": random.randint(0, 23) if random.choice([True, False]) else None,
    }
    return json.dumps(context)


class EventNames:
    METICA_BID_FLOOR_ASSIGNMENT = "metica_bid_floor_assignment"
    APPLOVIN_BID_FLOOR = "applovin_bid_floor"
    ESTIMATED_AD_REVENUE = "estimated_ad_revenue"

ad_units = ["ad_unit_1", "ad_unit_2", "ad_unit_3", "ad_unit_4", "ad_unit_5"]

def generate_random_data(
    customer_id, app_id, num_records, event_type, request_id_user_time_map, full_set=False, bid_only=False
):
    data = []
    ad_formats = ["INTER", "BANNER", "REWARDED"]
    ad_sources = ["Applovin", "Google Ads", "Unity Ads"]

    for i in range(num_records):
        request_id = list(request_id_user_time_map.keys())[i]
        user_id = request_id_user_time_map[request_id]["userId"]
        session_id = str(uuid.uuid4())
        placement_tag = f"tag_{random.randint(1, 10)}" if random.choice([True, False]) else None

        if event_type.lower() == EventNames.METICA_BID_FLOOR_ASSIGNMENT.lower():
            event_time = request_id_user_time_map[request_id]["assignmentTime"]
            context = generate_random_context()
            inference_data = json.dumps({"endpoint": "sagemaker"})
            cpm_floor_ad_unit_ids = request_id_user_time_map[request_id]["adUnits"]
            cpm_floor_values = sorted([round(random.uniform(0.0, 5.0), 2) for _ in cpm_floor_ad_unit_ids], reverse=True)
            record = (
                request_id,
                placement_tag,
                session_id,
                event_type,
                customer_id,
                app_id,
                "android_inter",
                event_time,
                user_id,
                context,
                cpm_floor_ad_unit_ids,
                cpm_floor_values,
                round(random.uniform(1.0, 5.0), 2),
                inference_data,
            )
            data.append(record)

        elif event_type.lower() == EventNames.APPLOVIN_BID_FLOOR.lower() and (full_set or bid_only):
            event_time = request_id_user_time_map[request_id]["bidTime"]
            cpm_floor_ad_unit_id = request_id_user_time_map[request_id]["adUnitId"]
            is_filled = random.choice([True, False])
            cpm_floor_value = round(random.uniform(0.0, 5.0), 2)
            winning_bid_value = round(random.uniform(cpm_floor_value, cpm_floor_value + 2.0), 2) if is_filled else None
            record = (
                request_id,
                placement_tag,
                session_id,
                event_type,
                customer_id,
                app_id,
                user_id,
                cpm_floor_ad_unit_id,
                is_filled,
                winning_bid_value,
                event_time,
            )
            data.append(record)

        elif event_type.lower() == EventNames.ESTIMATED_AD_REVENUE.lower() and full_set:
            event_time = request_id_user_time_map[request_id]["revenueTime"]
            cpm_floor_ad_unit_id = request_id_user_time_map[request_id]["adUnitId"]
            record = (
                request_id,
                placement_tag,
                session_id,
                event_type,
                cpm_floor_ad_unit_id,
                random.choice(ad_formats),
                random.choice(ad_sources),
                customer_id,
                app_id,
                event_time,
                user_id,
                round(random.uniform(0.5, 10.0), 2),
                "USD",
            )
            data.append(record)

    return data


def arg_parser(args):
    import argparse

    parser = argparse.ArgumentParser(description="Run the applovin bid floor training")
    parser.add_argument("--customerId", type=int, help="Customer ID")
    parser.add_argument("--appId", type=int, help="App ID")
    parser.add_argument("--numRecords", type=int, help="Number of records to generate")
    parser.add_argument("--s3BasePath", type=str, help="S3 base path for the data")
    parser.add_argument("--startDate", type=str, help="Start date for the data generation")
    parser.add_argument("--endDate", type=str, help="End date for the data generation")
    return parser.parse_args(args)


def run(spark: SparkSession, args: [str]):
    try:

        parsed_args = arg_parser(args)
        customer_id = parsed_args.customerId
        app_id = parsed_args.appId
        total_records = parsed_args.numRecords
        s3_base_path = parsed_args.s3BasePath
        start_date = datetime.fromisoformat(parsed_args.startDate)
        end_date = datetime.fromisoformat(parsed_args.endDate)
        full_set_count = int(total_records * 0.75)  # 75% with all events
        bid_only_count = int(total_records * 0.10)  # 10% with assignment and bid only
        assignment_only_count = int(total_records * 0.15)  # 15% with assignment only

        num_users = 100
        user_ids = [f"U{i}" for i in range(1, num_users + 1)]

        request_id_user_time_map = {}
        all_request_ids = [str(uuid.uuid4()) for _ in range(total_records)]
        date_diffs = [diff for diff in range((end_date - start_date).days + 1)]
        for i, req_id in enumerate(all_request_ids):
            user_id = random.choice(user_ids)
            base_time = (
                timedelta(days=random.choice(date_diffs))
                + start_date
                + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
            )
            assignment_time = base_time.isoformat()
            bid_time = (
                (base_time + timedelta(seconds=random.randint(1, 60))).isoformat()
                if i < full_set_count + bid_only_count
                else None
            )
            revenue_time = (
                (base_time + timedelta(minutes=random.randint(1, 60 * 32))).isoformat() if i < full_set_count else None
            )

            rnd_units = sorted(random.sample(ad_units, 3), reverse=True)

            request_id_user_time_map[req_id] = {
                "userId": user_id,
                "assignmentTime": assignment_time,
                "bidTime": bid_time,
                "revenueTime": revenue_time,
                "adUnits": rnd_units,
                "adUnitId": random.choice(rnd_units),
            }

        request_ids_full = all_request_ids[:full_set_count]
        request_ids_bid_only = all_request_ids[full_set_count : full_set_count + bid_only_count]
        request_ids_assignment_only = all_request_ids[full_set_count + bid_only_count :]

        request_id_user_map_full = {k: request_id_user_time_map[k] for k in request_ids_full}
        request_id_user_map_bid_only = {k: request_id_user_time_map[k] for k in request_ids_bid_only}
        request_id_user_map_assignment_only = {k: request_id_user_time_map[k] for k in request_ids_assignment_only}

        assignment_data_full = generate_random_data(
            customer_id,
            app_id,
            full_set_count,
            EventNames.METICA_BID_FLOOR_ASSIGNMENT,
            request_id_user_map_full,
            full_set=True,
        )
        bid_data_full = generate_random_data(
            customer_id, app_id, full_set_count, EventNames.APPLOVIN_BID_FLOOR, request_id_user_map_full, full_set=True
        )
        revenue_data_full = generate_random_data(
            customer_id,
            app_id,
            full_set_count,
            EventNames.ESTIMATED_AD_REVENUE,
            request_id_user_map_full,
            full_set=True,
        )

        assignment_data_bid_only = generate_random_data(
            customer_id,
            app_id,
            bid_only_count,
            EventNames.METICA_BID_FLOOR_ASSIGNMENT,
            request_id_user_map_bid_only,
            bid_only=True,
        )
        bid_data_bid_only = generate_random_data(
            customer_id,
            app_id,
            bid_only_count,
            EventNames.APPLOVIN_BID_FLOOR,
            request_id_user_map_bid_only,
            bid_only=True,
        )

        assignment_data_assignment_only = generate_random_data(
            customer_id,
            app_id,
            assignment_only_count,
            EventNames.METICA_BID_FLOOR_ASSIGNMENT,
            request_id_user_map_assignment_only,
        )

        assignment_df = spark.createDataFrame(
            assignment_data_full + assignment_data_bid_only + assignment_data_assignment_only, schema=assignment_schema
        )
        bid_df = spark.createDataFrame(bid_data_full + bid_data_bid_only, schema=bid_schema)
        revenue_df = spark.createDataFrame(revenue_data_full, schema=revenue_schema)

        assignment_df.withColumns(
            {"date": col("eventTime").cast("date"), "hour": hour(col("eventTime").cast("timestamp"))}
        ).write.partitionBy("date", "hour").mode("overwrite").format("parquet").save(
            f"{s3_base_path}/customerId={customer_id}/appId={app_id}/{EventNames.METICA_BID_FLOOR_ASSIGNMENT}/"
        )

        bid_df.withColumns(
            {"date": col("eventTime").cast("date"), "hour": hour(col("eventTime").cast("timestamp"))}
        ).write.partitionBy("date", "hour").mode("overwrite").format("parquet").save(
            f"{s3_base_path}/customerId={customer_id}/appId={app_id}/{EventNames.APPLOVIN_BID_FLOOR}/"
        )

        revenue_df.withColumns(
            {"date": col("eventTime").cast("date"), "hour": hour(col("eventTime").cast("timestamp"))}
        ).write.partitionBy("date", "hour").mode("overwrite").format("parquet").save(
            f"{s3_base_path}/customerId={customer_id}/appId={app_id}/{EventNames.ESTIMATED_AD_REVENUE}/"
        )

    except Exception as exp:
        print(f"Error occurred: {exp}")
        raise exp


if __name__ == "__main__":
    spark_conf = (
        SparkConf()
        .set("spark.driver.host", "127.0.0.1")
        .set("spark.driver.port", "7077")
        .set("spark.sql.execution.arrow.pyspark.enabled", "true")
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .set("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .set("spark.sql.catalog.dev", "org.apache.iceberg.spark.SparkCatalog")
        .set("spark.sql.catalog.dev.type", "hadoop")
        .set("spark.sql.catalog.dev.warehouse", "../target/warehouse")
        .set("spark.jars.packages", "org.apache.iceberg:iceberg-spark-runtime-3.4_2.12:1.7.0")
        .set("spark.sql.shuffle.partitions", "2")
        .set("spark.default.parallelism", "2")
    )
    spark = SparkSession.builder.config(conf=spark_conf).master("local[*]").appName("Local Test Runner").getOrCreate()
    spark.sparkContext.setLogLevel("INFO")

    # Example arguments
    args = [
        "--customerId",
        "123",
        "--appId",
        "456",
        "--numRecords",
        "1000",
        "--s3BasePath",
        "../target",
        "--startDate",
        "2023-01-01T00:00:00",
        "--endDate",
        "2023-01-02T00:00:00",
    ]

    run(spark, args)
