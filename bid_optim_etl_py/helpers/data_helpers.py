import pandas as pd
import re
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def extract_numeric_suffix(ad_unit_name: str) -> int:
    """Extract numeric suffix from ad unit names like 'metica_android_inter_ad_unit_10'."""
    match = re.search(r"_(\d+)$", ad_unit_name)
    return int(match.group(1)) if match else 0


def convert_to_cpm(df: pd.DataFrame, columns: List[str], multiplier: float = 1000.0) -> pd.DataFrame:
    """Convert specified columns to CPM by multiplying by multiplier."""
    df_copy = df.copy()
    df_copy[columns] = df_copy[columns] * multiplier
    return df_copy


def create_price_points_by_country(
    percentiles_df: pd.DataFrame, percentile_columns: List[str]
) -> Dict[str, List[float]]:
    """Group price points by country and sort by price point (ascending)."""
    price_points_by_country = {}

    logger.info(f"Percentiles df columns: {percentiles_df.columns}")

    for country in percentiles_df["user.country"].unique():
        country_prices = (
            percentiles_df[percentiles_df["user.country"] == country][percentile_columns]
            .sort_values(by=percentile_columns)
            .stack()
            .rename("cpm")
            .reset_index(drop=True)
            .to_list()
        )
        price_points_by_country[country] = country_prices

    return price_points_by_country


def group_countries_by_cpm(country_cpm_pairs: List[Tuple[str, float]]) -> Dict[str, List[str]]:
    """Group countries by CPM value to avoid API deduplication issues."""
    cpm_to_countries = {}

    for country, cpm in country_cpm_pairs:
        cpm_str = f"{cpm:.2f}"
        if cpm_str not in cpm_to_countries:
            cpm_to_countries[cpm_str] = []
        cpm_to_countries[cpm_str].append(country)

    return cpm_to_countries


def create_bid_floor_entry(country_group_name: str, cpm: str, countries: List[str]) -> Dict:
    """Create a single bid floor entry."""
    return {
        "country_group_name": country_group_name,
        "cpm": cpm,
        "countries": {
            "type": "INCLUDE",
            "values": [c.lower() for c in sorted(countries)],
        },
    }


def filter_metica_ad_units(ad_units: List[Dict], app_id: str, ad_type: str, exclude_suffix: str = "_1") -> List[Dict]:
    """Filter metica ad units for specific app and ad type, excluding specified suffix."""
    app_ad_units = [unit for unit in ad_units if unit.get("package_name") == app_id]

    metica_ad_units = [
        unit
        for unit in app_ad_units
        if "metica" in unit.get("name", "").lower()
        and unit.get("ad_format", "").lower() == ad_type.lower()
        and not unit["name"].endswith(exclude_suffix)
    ]

    return sorted(metica_ad_units, key=lambda x: extract_numeric_suffix(x["name"]))


# Removed unused S3 formatting helpers to keep the public surface minimal
