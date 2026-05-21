import pandas as pd
import pytest

from bid_optim_etl_py.helpers.data_helpers import (
    convert_to_cpm,
    create_bid_floor_entry,
    create_price_points_by_country,
    discover_percentile_columns,
    extract_numeric_suffix,
    filter_metica_ad_units,
    group_countries_by_cpm,
)


STANDARD_GRID = ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]
AGGRESSIVE_GRID = ["p40", "p50", "p60", "p70", "p80", "p83", "p86", "p90", "p94", "p97"]
ULTRA_AGGRESSIVE_GRID = ["p50", "p70", "p90", "p95", "p99"]


def _df_with_cols(cols):
    return pd.DataFrame([dict.fromkeys(cols, 1.0)])


def test_discover_percentile_columns_standard_grid():
    df = _df_with_cols(["user.country", *reversed(STANDARD_GRID)])
    assert discover_percentile_columns(df) == STANDARD_GRID


def test_discover_percentile_columns_aggressive_grid():
    df = _df_with_cols(["user.country", *AGGRESSIVE_GRID])
    assert discover_percentile_columns(df) == AGGRESSIVE_GRID


def test_discover_percentile_columns_ultra_aggressive_grid():
    df = _df_with_cols(["user.country", *ULTRA_AGGRESSIVE_GRID])
    assert discover_percentile_columns(df) == ULTRA_AGGRESSIVE_GRID


def test_discover_percentile_columns_ignores_metadata_cols():
    df = _df_with_cols(
        [
            "user.country",
            "user.platform",
            "user.adformat",
            "count",
            "total_revenue",
            "mean",
            "min",
            "max",
            "revenue_per_traffic",
            "pricing_strategy",
            "cumulative_traffic",
            "traffic_percentage",
            "p10",
            "p50",
            "p90",
        ]
    )
    assert discover_percentile_columns(df) == ["p10", "p50", "p90"]


def test_discover_percentile_columns_sorts_by_numeric_value():
    df = _df_with_cols(["p99", "p10", "p83", "p9"])
    assert discover_percentile_columns(df) == ["p9", "p10", "p83", "p99"]


def test_discover_percentile_columns_raises_on_empty():
    df = _df_with_cols(["user.country", "count"])
    with pytest.raises(ValueError, match="No percentile columns"):
        discover_percentile_columns(df)


def test_convert_to_cpm_multiplies_by_1000():
    df = pd.DataFrame([{"p10": 0.5, "p90": 1.3}])
    out = convert_to_cpm(df, ["p10", "p90"], 1000)
    assert out["p10"].iloc[0] == pytest.approx(500.0)
    assert out["p90"].iloc[0] == pytest.approx(1300.0)


def test_convert_to_cpm_skips_missing_columns():
    df = pd.DataFrame([{"p10": 0.5}])
    out = convert_to_cpm(df, ["p10", "p50", "p90"], 1000)
    assert out["p10"].iloc[0] == pytest.approx(500.0)
    assert list(out.columns) == ["p10"]


def test_convert_to_cpm_does_not_mutate_input():
    df = pd.DataFrame([{"p10": 0.5}])
    convert_to_cpm(df, ["p10"], 1000)
    assert df["p10"].iloc[0] == pytest.approx(0.5)


def test_create_price_points_by_country_per_country():
    df = pd.DataFrame(
        [
            {"user.country": "us", "p10": 100.0, "p50": 200.0, "p90": 300.0},
            {"user.country": "gb", "p10": 150.0, "p50": 250.0, "p90": 350.0},
        ]
    )
    out = create_price_points_by_country(df, ["p10", "p50", "p90"])
    assert out["us"] == [100.0, 200.0, 300.0]
    assert out["gb"] == [150.0, 250.0, 350.0]


def test_group_countries_by_cpm_merges_same_cpm():
    pairs = [("us", 1.50), ("gb", 1.50), ("de", 2.00)]
    out = group_countries_by_cpm(pairs)
    assert out["1.50"] == ["us", "gb"]
    assert out["2.00"] == ["de"]


def test_create_bid_floor_entry_shape():
    entry = create_bid_floor_entry("US", "1.50", ["us", "gb"])
    assert entry == {
        "country_group_name": "US",
        "cpm": "1.50",
        "countries": {"type": "INCLUDE", "values": ["gb", "us"]},
    }


def test_create_bid_floor_entry_lowercases_country_codes():
    entry = create_bid_floor_entry("US", "1.50", ["US", "GB"])
    assert entry["countries"]["values"] == ["gb", "us"]


def test_filter_metica_ad_units_excludes_underscore_1():
    units = [
        {"id": "a", "name": "metica_android_reward_1", "ad_format": "reward", "package_name": "com.app"},
        {"id": "b", "name": "metica_android_reward_2", "ad_format": "reward", "package_name": "com.app"},
        {"id": "c", "name": "metica_android_reward_3", "ad_format": "reward", "package_name": "com.app"},
    ]
    out = filter_metica_ad_units(units, "com.app", "reward")
    assert [u["id"] for u in out] == ["b", "c"]


def test_filter_metica_ad_units_filters_by_package():
    units = [
        {"id": "a", "name": "metica_android_reward_2", "ad_format": "reward", "package_name": "com.app"},
        {"id": "b", "name": "metica_android_reward_3", "ad_format": "reward", "package_name": "com.other"},
    ]
    out = filter_metica_ad_units(units, "com.app", "reward")
    assert [u["id"] for u in out] == ["a"]


def test_filter_metica_ad_units_filters_by_ad_format():
    units = [
        {"id": "a", "name": "metica_android_reward_2", "ad_format": "reward", "package_name": "com.app"},
        {"id": "b", "name": "metica_android_inter_2", "ad_format": "inter", "package_name": "com.app"},
    ]
    out = filter_metica_ad_units(units, "com.app", "reward")
    assert [u["id"] for u in out] == ["a"]


def test_filter_metica_ad_units_excludes_non_metica_names():
    units = [
        {"id": "a", "name": "metica_android_reward_2", "ad_format": "reward", "package_name": "com.app"},
        {"id": "b", "name": "house_android_reward_2", "ad_format": "reward", "package_name": "com.app"},
    ]
    out = filter_metica_ad_units(units, "com.app", "reward")
    assert [u["id"] for u in out] == ["a"]


def test_filter_metica_ad_units_sorts_by_numeric_suffix():
    units = [
        {"id": "ten", "name": "metica_android_reward_10", "ad_format": "reward", "package_name": "com.app"},
        {"id": "two", "name": "metica_android_reward_2", "ad_format": "reward", "package_name": "com.app"},
        {"id": "five", "name": "metica_android_reward_5", "ad_format": "reward", "package_name": "com.app"},
    ]
    out = filter_metica_ad_units(units, "com.app", "reward")
    assert [u["id"] for u in out] == ["two", "five", "ten"]


def test_extract_numeric_suffix_returns_int():
    assert extract_numeric_suffix("metica_android_reward_42") == 42


def test_extract_numeric_suffix_no_match_returns_zero():
    assert extract_numeric_suffix("no_number_here") == 0
