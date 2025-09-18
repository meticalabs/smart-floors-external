import re
from typing import List, Dict

def extract_numeric_suffix(ad_unit_name: str) -> int:
    """Extract numeric suffix from ad unit names like 'metica_android_inter_ad_unit_10'."""
    match = re.search(r"_(\d+)$", ad_unit_name)
    return int(match.group(1)) if match else 0


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
