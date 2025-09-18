# this version of the update_bid_floor_values script is intended for use by externals (customers)
# the workflow is to send them a file with the exact configurations they need to upload
# this is instead of them to calculate the bid floors themselves

import logging
import sys
from typing import List, Dict
import json
import os

from bid_optim_etl_py.helpers.applovin_management_api_client import ApplovinManagementApiClient
from bid_optim_etl_py.constants import (
    APPLOVIN_API_BASE_URL,
)
from bid_optim_etl_py.helpers.data_helpers import (
    filter_metica_ad_units,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_metica_ad_units(client: ApplovinManagementApiClient, package_name: str, ad_type: str) -> List[Dict]:
    """Get metica ad units for the specified app and ad type."""
    try:
        logger.info("Fetching ad units from AppLovin API...")
        fields = ["ad_network_settings", "frequency_capping_settings", "bid_floors"]
        ad_units = client.get_ad_units(fields=fields)

        metica_ad_units = filter_metica_ad_units(ad_units, package_name, ad_type)

        app_ad_units = [unit for unit in ad_units if unit.get("package_name") == package_name]
        metica_ad_units_complete = [
            unit
            for unit in app_ad_units
            if "metica" in unit.get("name", "").lower() and unit.get("ad_format", "").lower() == ad_type.lower()
        ]

        logger.info(f"Found {len(app_ad_units)} ad units for app {package_name}")
        logger.info(f"Found {len(metica_ad_units_complete)} metica {ad_type} ad units")
        logger.info(f"Processing {len(metica_ad_units)} metica ad units (excluding _1 units)")

        return metica_ad_units
    except Exception as e:
        logger.error(f"Error fetching metica ad units: {e}")
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


def run(package_name: str, ad_unit_config_folder: str, ad_type: str = "reward", platform: str = "android", api_key: str = None):
    """Main function to process a specific configuration and update bid floors."""
    try:
        logger.info(f"Processing bid floor updates for package_name={package_name}")
        
        # Get AppLovin API client
        if api_key is None:
            raise ValueError("API key must be provided either as an argument or via environment/configuration.")
        client = ApplovinManagementApiClient(api_key=api_key, base_url=APPLOVIN_API_BASE_URL)

        # Get metica ad units
        metica_ad_units = get_metica_ad_units(client, package_name, ad_type)

        if not metica_ad_units:
            logger.warning(f"No metica ad units found for app {package_name}")
            return {}

        # Read all JSON files from the ad_unit_config_folder and concatenate as a list        
        ad_unit_configurations = []
        filenames = os.listdir(ad_unit_config_folder)
        if not filenames:
            raise ValueError(f"The configuration folder '{ad_unit_config_folder}' is empty. Please provide at least one configuration file.")
        for filename in filenames:
            if filename.endswith(".json"):
                file_path = os.path.join(ad_unit_config_folder, filename)
                with open(file_path, "r") as f:
                    config = json.load(f)
                    if isinstance(config, list):
                        ad_unit_configurations.extend(config)
                    else:
                        ad_unit_configurations.append(config)

        if not ad_unit_configurations:
            logger.warning(f"No bid floor configurations created for app {package_name}")
            return {}

        # Update bid floors
        update_results = update_bid_floors(client, ad_unit_configurations, metica_ad_units)
        
        # Print summary
        print_summary(update_results)

        logger.info(f"Completed bid floor updates for package_name={package_name}")
        return update_results

    except Exception as e:
        import traceback

        logger.error(f"Error in run function: {e}\n{traceback.format_exc()}")
        raise


def main():
    """Main entry point for the script."""

    import argparse

    parser = argparse.ArgumentParser(description="Update bid floor values for AppLovin metica ad units")
    parser.add_argument("--adType", type=str, default="reward", help="Ad type (reward/inter)")
    parser.add_argument("--platform", type=str, default="android", help="Platform (android/ios)")
    parser.add_argument("--packageName", type=str, required=True, help="Package name")
    parser.add_argument("--adUnitConfigFolder", type=str, required=True, help="Folder containing ad unit configuration JSON files")
    parser.add_argument("--apiKey", type=str, required=True, help="AppLovin API key")
    parsed_args_obj = argparse.Namespace(parsed_args=parser.parse_args(sys.argv[1:]))
    cmd_line_args = parsed_args_obj.parsed_args

    try:
        results = run(
            package_name=cmd_line_args.packageName,
            ad_type=cmd_line_args.adType,
            platform=cmd_line_args.platform,
            ad_unit_config_folder=cmd_line_args.adUnitConfigFolder,
            api_key=cmd_line_args.apiKey,
        )
        print("Bid floor updates completed successfully.")
        print("Results:")
        print(f"Package Name: {cmd_line_args.packageName}")
        print(f"Update results: {results}")
    except Exception as e:
        print(f"Error during bid floor updates: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
