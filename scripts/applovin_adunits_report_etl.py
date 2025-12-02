
import argparse
from struct import pack
from typing import Dict, Optional
from io import StringIO
import requests
import pandas as pd
import datetime as dt
import boto3
from io import StringIO

def get_id_to_name(management_api_key) -> Dict[str, str]:
    """
    Fetches ad unit IDs and names from the AppLovin Management API.

    Args:
        management_api_key (str): The API key for AppLovin management API.

    Returns:
        Dict[str, str]: A dictionary mapping ad unit IDs to their names.

    Raises:
        requests.HTTPError: If the API request fails.
    """
    r = requests.get(
        "https://o.applovin.com/mediation/v1/ad_units",
        headers={"Api-Key": management_api_key},
        timeout=30,
    )
    r.raise_for_status()
    ad_units = r.json()
    return {au["id"]: au["name"] for au in ad_units}


def produce_df_adunit(
    region, management_api_key, reporting_api_key, applovin_id) -> pd.DataFrame:
    """
    Produce a DataFrame of ad unit data for a given app and customer using the AppLovin and Metica APIs.

    Args:
        region (str, optional): AWS region for secrets. Defaults to "eu-west-1".
        customer_id (int, optional): Customer ID. Defaults to 10851.
        app_id (int, optional): Application ID. Defaults to 12101.

    Returns:
        pd.DataFrame: DataFrame containing ad unit data with ad unit names.
    """
    columns = [
        "package_name",
        "platform",
        "day",
        "network",
        "ad_format",
        "max_ad_unit_id",
        "ecpm",
        "fill_rate",
        "estimated_revenue",
        "responses",
        "impressions",
        "attempts",
    ]
    df_adunit = max_revenue_by_adunit(reporting_api_key=reporting_api_key, package_name=applovin_id, columns=columns)
    id_to_name = get_id_to_name(management_api_key=management_api_key)
    
    df_adunit["name"] = df_adunit["Max_ad_unit_id"].map(lambda x: id_to_name.get(x, ""))

    return df_adunit


def max_revenue_by_adunit(
    reporting_api_key: str,
    package_name: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    columns: Optional[list] = None,
    not_zero: bool = True,
    days_of_data: int = 10,
) -> pd.DataFrame:
    """
    Fetch the maximum revenue by ad unit from the AppLovin reporting API.

    Args:
        reporting_api_key (str): AppLovin reporting API key.
        package_name (str): App package name.
        start (str, optional): Start date (YYYY-MM-DD). Defaults to days_of_data days ago.
        end (str, optional): End date (YYYY-MM-DD). Defaults to today.
        columns (List[str], optional): Columns to request. Defaults to a standard set.
        not_zero (bool, optional): Whether to filter for non-zero rows. Defaults to True.
        days_of_data (int, optional): Number of days of data to fetch. Defaults to 40.

    Returns:
        pd.DataFrame: DataFrame containing the report data.
    """

    if columns is None:
        columns = [
            "day",
            "package_name",
            "platform",
            "ad_format",
            "max_ad_unit_id",
            "impressions",
            "ecpm",
            "requests",
        ]
    if start is None:
        start = (dt.date.today() - dt.timedelta(days=days_of_data)).isoformat()
    if end is None:
        end = dt.date.today().isoformat()
    params = {
        "api_key": reporting_api_key,
        "start": start,
        "end": end,
        "format": "csv",
        "columns": ",".join(columns),
        "filter_package_name": package_name,
    }
    if not_zero:
        params["not_zero"] = 1
    try:
        r = requests.get("https://r.applovin.com/maxReport", params=params, timeout=60)
        r.raise_for_status()
    except requests.HTTPError as e:
        print("Error fetching report from AppLovin API.")
        print("This may be due to an incorrect API key or AppLovin ID.")
        print(f"HTTP Status: {r.status_code} - {r.text}")
        raise
    df = pd.read_csv(StringIO(r.text))
    return df


def get_cli_args():
    parser = argparse.ArgumentParser(description="Customer script for AppLovin management.")
    parser.add_argument("--applovin_id", required=True, help="AppLovin ID")
    parser.add_argument("--management_api_key", required=True, help="Management API Key")
    parser.add_argument("--reporting_api_key", required=True, help="Reporting API Key")
    parser.add_argument("--s3_output", required=True, help="S3 output location for the DataFrame (e.g. s3://bucket/path/output.csv)")
    parser.add_argument("--aws_access_key_id", type=str, required=True)
    parser.add_argument("--aws_secret_access_key", type=str, required=True)
    parser.add_argument("--aws_region", type=str, default="eu-west-1")
    
    return parser.parse_args()

if __name__ == "__main__":
    print("Starting AppLovin ad unit reporting script...")
    args = get_cli_args()
    applovin_id = args.applovin_id
    management_api_key = args.management_api_key
    reporting_api_key = args.reporting_api_key
    s3_output = args.s3_output
    print(f"Using AppLovin ID: {applovin_id}")
    print(f"Output will be saved to: {s3_output}")
    print("Fetching ad unit data from AppLovin APIs...")
    df = produce_df_adunit(region=args.aws_region, management_api_key=management_api_key, reporting_api_key=reporting_api_key, applovin_id=applovin_id)
    print(f"Fetched {len(df)} rows of ad unit data.")
    print("Uploading data to S3...")

    s3_client = boto3.client(
        's3',
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        region_name=args.aws_region
    )
    # Parse bucket and key from s3_output
    if not s3_output.startswith("s3://"):
        raise ValueError("s3_output must start with 's3://'")
    s3_path = s3_output[5:]
    bucket, key = s3_path.split('/', 1)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
    print("Done! Data saved successfully to S3.")
    