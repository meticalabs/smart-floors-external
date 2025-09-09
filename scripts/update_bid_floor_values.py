import pandas as pd
from datetime import datetime
import logging
import sys
from typing import List, Dict

from etl_py_commons.job_initialiser import Initialisation
from bid_optim_etl_py.command_line_args import PercentileCalculationArgsParser
from bid_optim_etl_py.helpers.applovin_management_api_client import ApplovinManagementApiClient
from bid_optim_etl_py.helpers.metica_management_api_client import MeticaManagementApiClient
from bid_optim_etl_py.constants import (
    APPLOVIN_API_BASE_URL,
    S3_ARTIFACTS_BUCKET,
    BID_FLOOR_PERCENTILES_PREFIX,
    APP_ID_TO_APPLOVIN_ID,
    DEFAULT_AWS_REGION,
    PERCENTILE_COLUMNS,
    CPM_MULTIPLIER,
    APPLOVIN_MANAGEMENT_API_KEYS,
)
from bid_optim_etl_py.helpers.aws_helpers import S3Helper, SecretsManagerHelper
from bid_optim_etl_py.helpers.data_helpers import (
    convert_to_cpm,
    create_price_points_by_country,
    group_countries_by_cpm,
    create_bid_floor_entry,
    filter_metica_ad_units,
    format_s3_key,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_secret(secret_name: str, region_name: str = DEFAULT_AWS_REGION) -> str:
    """Retrieve secret from AWS Secrets Manager."""
    try:
        secrets_helper = SecretsManagerHelper(region_name)
        return secrets_helper.get_secret(secret_name)
    except Exception as e:
        logger.error(f"Error retrieving secret {secret_name}: {e}")
        raise


def update_bid_floor_metica_platform(client: MeticaManagementApiClient, ad_unit_configurations: List[Dict]):
    """Update metica platform ad units."""
    try:
        old_ad_units = client.get_ad_units()
        for ad_unit in old_ad_units:
            # Find the matching configuration for this ad unit
            matching_config = next(
                (unit for unit in ad_unit_configurations
                if unit["ad_unit_id"] == ad_unit.get("id")),
                None
            )
            if not matching_config:
                logger.warning(f"No configuration found for ad unit {ad_unit.get('name')}, skipping.")
                continue
            bid_floors = matching_config.get("bid_floors")
            if bid_floors:
                try:
                    client.update_ad_unit(ad_unit_id=ad_unit["id"], ad_unit_data=ad_unit, bid_floors=bid_floors)
                    logger.info(f"Updated bid floor for ad unit {ad_unit['id']}")
                except Exception as update_exc:
                    logger.error(f"  ❌ Error updating ad unit {ad_unit['id']}: {str(update_exc)}")
            else:
                logger.info(f"No bid floors to update for ad unit {ad_unit['id']}")
    except Exception as e:
        logger.error(f"Error updating metica platform ad units: {e}")
        raise

def get_bid_floor_percentiles(app_id: int, customer_id: int, ad_type: str, platform: str = "android") -> pd.DataFrame:
    """Fetch bid floor percentiles from S3 for the given app, customer, ad_type, and platform."""
    try:
        s3_helper = S3Helper()
        today = datetime.now().strftime("%Y-%m-%d")
        s3_key = format_s3_key(BID_FLOOR_PERCENTILES_PREFIX, customer_id, app_id, today, platform, ad_type)

        data = s3_helper.read_json(S3_ARTIFACTS_BUCKET, s3_key)
        percentiles_df = pd.read_json(data, orient="records")

        # Convert to CPM
        percentiles_df = convert_to_cpm(percentiles_df, PERCENTILE_COLUMNS, CPM_MULTIPLIER)

        # Remove rows where 'user.country' is None, empty, or only whitespace
        if "user.country" in percentiles_df.columns:
            percentiles_df = percentiles_df[percentiles_df["user.country"].notnull()]
            percentiles_df = percentiles_df[percentiles_df["user.country"].astype(str).str.strip() != ""]

        logger.info(f"Successfully fetched percentiles for app {app_id}, shape: {percentiles_df.shape}")
        return percentiles_df
    except Exception as e:
        logger.error(f"Error fetching percentiles from S3: {e}")
        raise


def get_metica_ad_units(client: ApplovinManagementApiClient, app_id: int, ad_type: str) -> List[Dict]:
    """Get metica ad units for the specified app and ad type."""
    try:
        logger.info("Fetching ad units from AppLovin API...")
        fields = ["ad_network_settings", "frequency_capping_settings", "bid_floors"]
        ad_units = client.get_ad_units(fields=fields)

        metica_ad_units = filter_metica_ad_units(ad_units, APP_ID_TO_APPLOVIN_ID[app_id], ad_type)

        app_ad_units = [unit for unit in ad_units if unit.get("package_name") == APP_ID_TO_APPLOVIN_ID[app_id]]
        metica_ad_units_complete = [
            unit
            for unit in app_ad_units
            if "metica" in unit.get("name", "").lower() and unit.get("ad_format", "").lower() == ad_type.lower()
        ]

        logger.info(f"Found {len(app_ad_units)} ad units for app {app_id}")
        logger.info(f"Found {len(metica_ad_units_complete)} metica {ad_type} ad units")
        logger.info(f"Processing {len(metica_ad_units)} metica ad units (excluding _1 units)")

        return metica_ad_units
    except Exception as e:
        logger.error(f"Error fetching metica ad units: {e}")
        raise


def create_bid_floor_configurations(metica_ad_units: List[Dict], percentiles_df: pd.DataFrame) -> List[Dict]:
    """Create bid floor configurations for each ad unit based on percentiles."""
    try:
        # Group price points by country and sort by price_point (ascending)
        price_points_by_country = create_price_points_by_country(percentiles_df, PERCENTILE_COLUMNS)

        logger.info("Price points by country (sorted ascending):")
        for country, prices in price_points_by_country.items():
            logger.info(f"  {country}: {prices}")

        ad_unit_configurations = []

        for i, ad_unit in enumerate(metica_ad_units):
            # Collect all countries and their CPMs for this ad unit
            country_cpm_pairs = []

            for country, prices in price_points_by_country.items():
                if i < len(prices):  # Only if there are enough price points for this ad unit
                    price_point = prices[i]
                    country_cpm_pairs.append((country, price_point))

            # Group countries by CPM value to avoid API deduplication issues
            cpm_to_countries = group_countries_by_cpm(country_cpm_pairs)

            # Create bid floors for each unique CPM
            bid_floors = []
            for cpm_str, countries in cpm_to_countries.items():
                # Use the first country (alphabetically) as the group name for API compatibility
                country_group_name = sorted(countries)[0].upper()
                bid_floors.append(create_bid_floor_entry(country_group_name, cpm_str, countries))

            if bid_floors:  # Only add configuration if there are bid floors to set
                ad_unit_configurations.append(
                    {
                        "ad_unit_id": ad_unit["id"],
                        "ad_unit_name": ad_unit["name"],
                        "bid_floors": bid_floors,
                    }
                )

        logger.info(f"Created {len(ad_unit_configurations)} ad unit configurations")
        return ad_unit_configurations
    except Exception as e:
        logger.error(f"Error creating bid floor configurations: {e}")
        raise


def update_bid_floors(
    client: ApplovinManagementApiClient, ad_unit_configurations: List[Dict], metica_ad_units: List[Dict]
) -> List[Dict]:
    """Update bid floors for all configured ad units."""
    try:
        logger.info("Updating bid floors for all metica ad units...")
        logger.info("=" * 60)

        update_results = []

        for config in ad_unit_configurations:
            ad_unit_id = config["ad_unit_id"]
            ad_unit_name = config["ad_unit_name"]
            bid_floors = config["bid_floors"]

            logger.info(f"Updating {ad_unit_name} (ID: {ad_unit_id})...")
            logger.info(f"  Bid floors to set: {len(bid_floors)}")
            for bf in bid_floors:
                countries_str = ", ".join(bf["countries"]["values"])
                logger.info(f"    - Group {bf['country_group_name']}: ${bf['cpm']} ({countries_str})")

            try:
                # Find the original ad unit data
                original_ad_unit = next(unit for unit in metica_ad_units if unit["id"] == ad_unit_id)

                # Update the ad unit with new bid floors
                response = client.update_ad_unit(
                    ad_unit_id=ad_unit_id, ad_unit_data=original_ad_unit, bid_floors=bid_floors
                )

                if response.get("success", False) or "id" in response:
                    logger.info("  ✅ Successfully updated bid floors")

                    update_results.append(
                        {
                            "ad_unit_id": ad_unit_id,
                            "ad_unit_name": ad_unit_name,
                            "status": "success",
                            "bid_floors_set": len(bid_floors),
                        }
                    )
                else:
                    logger.error(f"  ❌ Failed to update: {response}")
                    update_results.append(
                        {
                            "ad_unit_id": ad_unit_id,
                            "ad_unit_name": ad_unit_name,
                            "status": "failed",
                            "error": str(response),
                        }
                    )

            except Exception as e:
                logger.error(f"  ❌ Error updating ad unit: {str(e)}")
                update_results.append(
                    {
                        "ad_unit_id": ad_unit_id,
                        "ad_unit_name": ad_unit_name,
                        "status": "error",
                        "error": str(e),
                    }
                )

        return update_results
    except Exception as e:
        logger.error(f"Error updating bid floors: {e}")
        raise


def print_summary(update_results: List[Dict]) -> None:
    """Print summary of bid floor updates."""
    logger.info("\n" + "=" * 60)
    logger.info("UPDATE SUMMARY")
    logger.info("=" * 60)

    successful_updates = [r for r in update_results if r["status"] == "success"]
    failed_updates = [r for r in update_results if r["status"] in ["failed", "error"]]

    logger.info(f"✅ Successful updates: {len(successful_updates)}")
    logger.info(f"❌ Failed updates: {len(failed_updates)}")

    if failed_updates:
        logger.info("\nFailed updates:")
        for result in failed_updates:
            logger.info(f"  - {result['ad_unit_name']}: {result.get('error', 'Unknown error')}")

    logger.info(f"\nTotal ad units processed: {len(update_results)}")


def run(customer_id: int, app_id: int, ad_type: str = "reward", platform: str = "android", cutoff_days: int = 7):
    """Main function to process a specific configuration and update bid floors."""
    try:
        logger.info(f"Processing bid floor updates for customer_id={customer_id}, app_id={app_id}")

        # Validate app_id is supported
        if app_id not in APP_ID_TO_APPLOVIN_ID:
            raise ValueError(f"App ID {app_id} is not supported. Supported IDs: {list(APP_ID_TO_APPLOVIN_ID.keys())}")

        # Get AppLovin API client

        api_key = APPLOVIN_MANAGEMENT_API_KEYS[APP_ID_TO_APPLOVIN_ID[app_id]]
        client = ApplovinManagementApiClient(api_key=api_key, base_url=APPLOVIN_API_BASE_URL)
        metica_client = MeticaManagementApiClient(customer_id=customer_id, app_id=app_id)

        # Get bid floor percentiles
        percentiles_df = get_bid_floor_percentiles(app_id, customer_id, ad_type, platform)
        percentiles_df = percentiles_df.rename(columns={"user_x2Ecountry": "user.country"})
        # Get metica ad units
        metica_ad_units = get_metica_ad_units(client, app_id, ad_type)

        if not metica_ad_units:
            logger.warning(f"No metica ad units found for app {app_id}")
            return {}

        # Create bid floor configurations
        ad_unit_configurations = create_bid_floor_configurations(metica_ad_units, percentiles_df)

        if not ad_unit_configurations:
            logger.warning(f"No bid floor configurations created for app {app_id}")
            return {}

        # Update bid floors
        update_results = update_bid_floors(client, ad_unit_configurations, metica_ad_units)
        # Update bid floors for metica platform
        update_bid_floor_metica_platform(metica_client, ad_unit_configurations)
        # Print summary
        print_summary(update_results)

        logger.info(f"Completed bid floor updates for customer_id={customer_id}, app_id={app_id}")
        return update_results

    except Exception as e:
        import traceback

        logger.error(f"Error in run function: {e}\n{traceback.format_exc()}")
        raise


def main():
    """Main entry point for the script."""
    argvs = sys.argv[1:] if len(sys.argv) > 1 else []
    parsed_args_obj = Initialisation.parse_args(args=argvs, parser_obj=PercentileCalculationArgsParser())
    cmd_line_args = parsed_args_obj.parsed_args

    try:
        results = run(
            customer_id=cmd_line_args.customerId,
            app_id=cmd_line_args.appId,
            ad_type=cmd_line_args.adType,
            platform=cmd_line_args.platform,
        )
        print("Bid floor updates completed successfully.")
        print("Results:")
        print(f"Customer ID: {cmd_line_args.customerId}")
        print(f"App ID: {cmd_line_args.appId}")
        print(f"Update results: {results}")
    except Exception as e:
        print(f"Error during bid floor updates: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
