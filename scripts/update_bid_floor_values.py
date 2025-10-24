import argparse
import json
import logging
from datetime import datetime
from typing import List, Dict

import boto3
import pandas as pd

from bid_optim_etl_py.constants import (
    APPLOVIN_API_BASE_URL,
    S3_ARTIFACTS_BUCKET,
    BID_FLOOR_PERCENTILES_PREFIX,
    PERCENTILE_COLUMNS,
    CPM_MULTIPLIER,
    MAX_CPM,
)
from bid_optim_etl_py.helpers.applovin_management_api_client import ApplovinManagementApiClient
from bid_optim_etl_py.helpers.data_helpers import (
    convert_to_cpm,
    create_price_points_by_country,
    group_countries_by_cpm,
    create_bid_floor_entry,
    filter_metica_ad_units,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_percentiles_prefix(customer_id: int, app_id: int) -> str:
    return f"{BID_FLOOR_PERCENTILES_PREFIX}/{customer_id}/{app_id}/"


def read_percentiles_from_s3(s3_client, bucket: str, key: str) -> pd.DataFrame:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read().decode("utf-8")
    percentiles_df = pd.read_json(data, orient="records")
    percentiles_df = convert_to_cpm(percentiles_df, PERCENTILE_COLUMNS, CPM_MULTIPLIER)
    for col in PERCENTILE_COLUMNS:
        if col in percentiles_df.columns:
            percentiles_df.loc[percentiles_df[col] > MAX_CPM, col] = MAX_CPM -100 
    if "user.country" in percentiles_df.columns:
        percentiles_df = percentiles_df[percentiles_df["user.country"].notnull()]
        percentiles_df = percentiles_df[percentiles_df["user.country"].astype(str).str.strip() != ""]
    return percentiles_df


def get_metica_ad_units(client: ApplovinManagementApiClient, app_id: int, ad_type: str, package_name: str) -> List[Dict]:
    fields = ["ad_network_settings", "frequency_capping_settings", "bid_floors"]
    ad_units = client.get_ad_units(fields=fields)
    metica_ad_units = filter_metica_ad_units(ad_units, package_name, ad_type)
    return metica_ad_units


def create_bid_floor_configurations(metica_ad_units: List[Dict], percentiles_df: pd.DataFrame) -> List[Dict]:
    price_points_by_country = create_price_points_by_country(percentiles_df, PERCENTILE_COLUMNS)
    ad_unit_configurations = []
    for i, ad_unit in enumerate(metica_ad_units):
        country_cpm_pairs = []
        for country, prices in price_points_by_country.items():
            if i < len(prices):
                price_point = prices[i]
                country_cpm_pairs.append((country, price_point))
        cpm_to_countries = group_countries_by_cpm(country_cpm_pairs)
        bid_floors = []
        for cpm_str, countries in cpm_to_countries.items():
            country_group_name = sorted(countries)[0].upper()
            bid_floors.append(create_bid_floor_entry(country_group_name, cpm_str, countries))
        if bid_floors:
            ad_unit_configurations.append(
                {
                    "ad_unit_id": ad_unit["id"],
                    "ad_unit_name": ad_unit["name"],
                    "bid_floors": bid_floors,
                }
            )
    return ad_unit_configurations


def update_bid_floors_applovin(client: ApplovinManagementApiClient, configurations: List[Dict], metica_ad_units: List[Dict]) -> None:
    for config in configurations:
        ad_unit_id = config["ad_unit_id"]
        logger.info(f"Updating bid floors for ad unit id: {ad_unit_id} ")
        bid_floors = config["bid_floors"]
        original_ad_unit = next(unit for unit in metica_ad_units if unit["id"] == ad_unit_id)
        client.update_ad_unit(ad_unit_id=ad_unit_id, ad_unit_data=original_ad_unit, bid_floors=bid_floors)


def main():
    parser = argparse.ArgumentParser(description="Client-side updater for AppLovin bid floors and S3 upload of configurations")
    parser.add_argument("--customer-id", type=int, required=True)
    parser.add_argument("--app-id", type=int, required=True)
    parser.add_argument("--ad-type", type=str, default="reward")
    parser.add_argument("--platform", type=str, default="android")
    parser.add_argument("--applovin-api-key", type=str, required=True)
    parser.add_argument("--aws-access-key-id", type=str, required=True)
    parser.add_argument("--aws-secret-access-key", type=str, required=True)
    parser.add_argument("--aws-region", type=str, default="eu-west-1")
    parser.add_argument("--s3-bucket", type=str, default=S3_ARTIFACTS_BUCKET)
    parser.add_argument("--package-name", type=str, required=True)

    args = parser.parse_args()

    # Configure AWS client with provided credentials
    session = boto3.Session(
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        region_name=args.aws_region,
    )
    s3_client = session.client("s3")

    # Find latest JSON under the expected prefix
    prefix = build_percentiles_prefix(args.customer_id, args.app_id)
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=args.s3_bucket, Prefix=prefix)
    latest_obj = None
    for page in page_iterator:
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if key.endswith(f"{args.platform}_{args.ad_type}.json"):
                if latest_obj is None or obj["LastModified"] > latest_obj["LastModified"]:
                    latest_obj = obj
    if latest_obj is None:
        raise RuntimeError(
            "No percentiles JSON found in S3 for the specified platform and ad_type"
        )
    percentiles_key = latest_obj["Key"]
    logger.info(f"Reading percentiles from s3://{args.s3_bucket}/{percentiles_key}")
    percentiles_df = read_percentiles_from_s3(s3_client, args.s3_bucket, percentiles_key)

    applovin_client = ApplovinManagementApiClient(api_key=args.applovin_api_key, base_url=APPLOVIN_API_BASE_URL)


    metica_ad_units = get_metica_ad_units(applovin_client, args.app_id, args.ad_type, args.package_name)
    if not metica_ad_units:
        raise RuntimeError("No metica ad units found to update")

    configurations = create_bid_floor_configurations(metica_ad_units, percentiles_df)
    if not configurations:
        raise RuntimeError("No bid floor configurations were created")

    logger.info("Updating AppLovin bid floors...")
    update_bid_floors_applovin(applovin_client, configurations, metica_ad_units)
    logger.info("AppLovin update complete")

    updated_ad_unit_configurations = get_metica_ad_units(applovin_client, args.app_id, args.ad_type, args.package_name)
    upload_key = f"{BID_FLOOR_PERCENTILES_PREFIX}/{args.customer_id}/{args.app_id}/uploads/ad_unit_configurations_{args.platform}_{args.ad_type}.json"
    body = json.dumps(updated_ad_unit_configurations)
    s3_client.put_object(Bucket=args.s3_bucket, Key=upload_key, Body=body, ContentType="application/json")
    logger.info(f"Uploaded configurations to s3://{args.s3_bucket}/{upload_key}")


if __name__ == "__main__":
    main()


