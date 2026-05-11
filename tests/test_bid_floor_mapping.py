"""Lock-in tests for the ad-unit -> percentile equal-distribution mapping.

These tests pin down the exact numpy.linspace + numpy.ceil behaviour so the
mapping can't silently regress. The percentile-to-ad-unit mapping is the most
fragile part of the script and the contract that clients depend on.
"""
import pandas as pd
import pytest

from scripts.update_bid_floor_values import create_bid_floor_configurations


def _au(i):
    return {"id": f"au{i}", "name": f"metica_android_reward_{i}"}


def _df(country_to_values, percentile_columns):
    rows = []
    for country, values in country_to_values.items():
        row = {"user.country": country}
        row.update(dict(zip(percentile_columns, values)))
        rows.append(row)
    return pd.DataFrame(rows)


def _cpms_per_ad_unit(configurations, country="us"):
    """Return the cpm string assigned to ``country`` for each configured ad unit, keyed by ad_unit_id."""
    result = {}
    for cfg in configurations:
        for bf in cfg["bid_floors"]:
            if country in bf["countries"]["values"]:
                result[cfg["ad_unit_id"]] = bf["cpm"]
    return result


def test_equal_count_5_each():
    pcols = ["p10", "p20", "p30", "p40", "p50"]
    df = _df({"us": [100.0, 200.0, 300.0, 400.0, 500.0]}, pcols)
    ad_units = [_au(i) for i in range(2, 7)]

    cfgs = create_bid_floor_configurations(ad_units, df, pcols)

    assert len(cfgs) == 5
    cpms = _cpms_per_ad_unit(cfgs)
    assert cpms == {
        "au2": "100.00",
        "au3": "200.00",
        "au4": "300.00",
        "au5": "400.00",
        "au6": "500.00",
    }


def test_fewer_ad_units_2au_5p():
    pcols = ["p10", "p30", "p50", "p70", "p90"]
    df = _df({"us": [10.0, 30.0, 50.0, 70.0, 90.0]}, pcols)
    ad_units = [_au(2), _au(3)]

    cfgs = create_bid_floor_configurations(ad_units, df, pcols)

    cpms = _cpms_per_ad_unit(cfgs)
    assert cpms == {"au2": "50.00", "au3": "90.00"}


def test_fewer_ad_units_5au_10p():
    pcols = [f"p{n}" for n in range(10, 101, 10)]
    df = _df({"us": [float(i * 10) for i in range(1, 11)]}, pcols)
    ad_units = [_au(i) for i in range(2, 7)]

    cfgs = create_bid_floor_configurations(ad_units, df, pcols)

    cpms = _cpms_per_ad_unit(cfgs)
    assert cpms == {
        "au2": "20.00",
        "au3": "40.00",
        "au4": "60.00",
        "au5": "80.00",
        "au6": "100.00",
    }


def test_fewer_ad_units_10au_20p():
    pcols = [f"p{n}" for n in range(5, 101, 5)]
    df = _df({"us": [float(i) for i in range(1, 21)]}, pcols)
    ad_units = [_au(i) for i in range(2, 12)]

    cfgs = create_bid_floor_configurations(ad_units, df, pcols)

    cpms = _cpms_per_ad_unit(cfgs)
    expected_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    assert cpms == {f"au{i+2}": f"{v:.2f}" for i, v in enumerate(expected_values)}


def test_fewer_ad_units_7au_10p_uneven():
    pcols = [f"p{n}" for n in range(10, 101, 10)]
    df = _df({"us": [float(i * 10) for i in range(1, 11)]}, pcols)
    ad_units = [_au(i) for i in range(2, 9)]

    cfgs = create_bid_floor_configurations(ad_units, df, pcols)

    cpms = _cpms_per_ad_unit(cfgs)
    expected_indices = [1, 2, 4, 5, 7, 8, 9]
    expected_values = [(i + 1) * 10 for i in expected_indices]
    assert cpms == {f"au{i+2}": f"{v:.2f}" for i, v in enumerate(expected_values)}

    cpm_floats = [float(cpms[f"au{i+2}"]) for i in range(7)]
    assert cpm_floats == sorted(cpm_floats)


def test_more_ad_units_10au_5p():
    pcols = ["p10", "p30", "p50", "p70", "p90"]
    df = _df({"us": [10.0, 30.0, 50.0, 70.0, 90.0]}, pcols)
    ad_units = [_au(i) for i in range(2, 12)]

    cfgs = create_bid_floor_configurations(ad_units, df, pcols)

    assert len(cfgs) == 5
    cpms = _cpms_per_ad_unit(cfgs)
    assert cpms == {
        "au2": "10.00",
        "au3": "30.00",
        "au4": "50.00",
        "au5": "70.00",
        "au6": "90.00",
    }
    assert "au7" not in cpms
    assert "au11" not in cpms


def test_more_ad_units_3au_2p():
    pcols = ["p50", "p90"]
    df = _df({"us": [50.0, 90.0]}, pcols)
    ad_units = [_au(2), _au(3), _au(4)]

    cfgs = create_bid_floor_configurations(ad_units, df, pcols)

    cpms = _cpms_per_ad_unit(cfgs)
    assert cpms == {"au2": "50.00", "au3": "90.00"}


def test_zero_ad_units_returns_empty():
    pcols = ["p10", "p50", "p90"]
    df = _df({"us": [10.0, 50.0, 90.0]}, pcols)
    cfgs = create_bid_floor_configurations([], df, pcols)
    assert cfgs == []


def test_single_ad_unit_single_percentile():
    pcols = ["p50"]
    df = _df({"us": [50.0]}, pcols)
    cfgs = create_bid_floor_configurations([_au(2)], df, pcols)

    assert len(cfgs) == 1
    assert cfgs[0]["ad_unit_id"] == "au2"
    assert cfgs[0]["bid_floors"][0]["cpm"] == "50.00"


def test_aggressive_grid_5_ad_units():
    pcols = ["p40", "p70", "p86", "p97", "p99"]
    df = _df({"us": [400.0, 700.0, 860.0, 970.0, 990.0]}, pcols)
    ad_units = [_au(i) for i in range(2, 7)]

    cfgs = create_bid_floor_configurations(ad_units, df, pcols)

    cpms = _cpms_per_ad_unit(cfgs)
    assert cpms == {
        "au2": "400.00",
        "au3": "700.00",
        "au4": "860.00",
        "au5": "970.00",
        "au6": "990.00",
    }


def test_smart_pricing_country_gets_low_floor_at_every_ad_unit():
    pcols = ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]
    df = _df(
        {
            "us": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            "in": [0.01] * 9,
        },
        pcols,
    )
    ad_units = [_au(2), _au(3), _au(4)]

    cfgs = create_bid_floor_configurations(ad_units, df, pcols)

    assert len(cfgs) == 3
    us_cpms = _cpms_per_ad_unit(cfgs, country="us")
    in_cpms = _cpms_per_ad_unit(cfgs, country="in")
    assert us_cpms == {"au2": "30.00", "au3": "60.00", "au4": "90.00"}
    assert in_cpms == {"au2": "0.01", "au3": "0.01", "au4": "0.01"}


def test_countries_with_same_cpm_merged_into_one_bid_floor():
    pcols = ["p10", "p50", "p90"]
    df = _df(
        {
            "us": [10.0, 50.0, 90.0],
            "gb": [10.0, 50.0, 90.0],
            "de": [20.0, 60.0, 100.0],
        },
        pcols,
    )
    ad_units = [_au(2), _au(3), _au(4)]
    cfgs = create_bid_floor_configurations(ad_units, df, pcols)

    by_cpm_middle = {bf["cpm"]: bf for bf in cfgs[1]["bid_floors"]}
    assert sorted(by_cpm_middle["50.00"]["countries"]["values"]) == ["gb", "us"]
    assert by_cpm_middle["50.00"]["country_group_name"] == "GB"
    assert sorted(by_cpm_middle["60.00"]["countries"]["values"]) == ["de"]


def test_skipped_ad_units_logged(caplog):
    pcols = ["p50"]
    df = _df({"us": [50.0]}, pcols)
    ad_units = [_au(2), _au(3)]

    with caplog.at_level("WARNING"):
        cfgs = create_bid_floor_configurations(ad_units, df, pcols)

    assert len(cfgs) == 1
    assert any("More ad units (2) than percentiles (1)" in rec.message for rec in caplog.records)


def test_ad_units_ordering_preserved_in_output():
    pcols = ["p10", "p50", "p90"]
    df = _df({"us": [10.0, 50.0, 90.0]}, pcols)
    ad_units = [_au(2), _au(3), _au(4)]
    cfgs = create_bid_floor_configurations(ad_units, df, pcols)

    assert [c["ad_unit_id"] for c in cfgs] == ["au2", "au3", "au4"]
    assert [c["ad_unit_name"] for c in cfgs] == [
        "metica_android_reward_2",
        "metica_android_reward_3",
        "metica_android_reward_4",
    ]
