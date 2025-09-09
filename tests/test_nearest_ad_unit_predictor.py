import pytest
import pandas as pd
from bid_optim_etl_py.nearest_ad_unit_predictor import NearestAdUnitPredictor
import numpy as np


@pytest.fixture
def predictor():
    return NearestAdUnitPredictor()


def test_sort_by_name_postfix_desc(predictor):
    ad_units = [
        {"name": "ad_unit_10", "id": "10", "bidFloor": 10.0},
        {"name": "ad_unit_2", "id": "2", "bidFloor": 2.0},
        {"name": "ad_unit_5", "id": "5", "bidFloor": 5.0},
    ]
    sorted_units = predictor.sort_by_name_postfix_desc(ad_units)
    assert [u["id"] for u in sorted_units] == ["10", "5", "2"]


def test_split_based_on_name(predictor):
    ad_units = [
        {"name": "ad_unit_3", "id": "3", "bidFloor": 3.0},
        {"name": "ad_unit_1", "id": "1", "bidFloor": 1.0},
        {"name": "ad_unit_2", "id": "2", "bidFloor": 2.0},
    ]
    rest, lowest = predictor.split_based_on_name(ad_units)
    assert isinstance(rest, list)
    assert isinstance(lowest, dict)
    assert lowest["id"] == "1"


def test_form_response(predictor):
    assignments = [
        {"id": "A", "name": "ad_unit_A", "bidFloor": 10.0},
        {"id": "B", "name": "ad_unit_B", "bidFloor": 20.0},
    ]
    lowest_bid_floor = {"id": "C", "name": "ad_unit_C", "bidFloor": 5.0}
    propensity = 0.75
    response = predictor.form_response(assignments, lowest_bid_floor, propensity)
    assert response["cpmFloorAdUnitIds"] == ["A", "B", "C"]
    assert response["cpmFloorValues"] == [10.0, 20.0, 5.0]
    assert response["propensity"] == 0.75


def test_predict_max_ad_units_1(predictor):
    context = pd.Series({"user.avgInterRevenueLast72Hours": 2.0})
    floors = [
        {"name": "ad_unit_1", "id": "1", "bidFloor": 1.0},
        {"name": "ad_unit_2", "id": "2", "bidFloor": 2.0},
    ]
    result = predictor.predict(context, floors, max_ad_units=1)
    assert result["cpmFloorAdUnitIds"] == ["1"] or result["cpmFloorAdUnitIds"] == ["2"]
    assert result["propensity"] == 1.0


def test_predict_invalid_max_ad_units(predictor):
    context = pd.Series({"user.avgInterRevenueLast72Hours": 2.0})
    floors = [
        {"name": "ad_unit_1", "id": "1", "bidFloor": 1.0},
        {"name": "ad_unit_2", "id": "2", "bidFloor": 2.0},
    ]
    with pytest.raises(ValueError):
        predictor.predict(context, floors, max_ad_units=3)


def test_predict_closest_ad_unit_different_context(predictor):
    # Different context value
    context = pd.Series({"user.avgInterRevenueLast72Hours": 1.2})
    # Ad units with various bid floors
    floors = [
        {"name": "ad_unit_1", "id": "1", "bidFloor": 0.5},
        {"name": "ad_unit_2", "id": "2", "bidFloor": 1.0},
        {"name": "ad_unit_3", "id": "3", "bidFloor": 1.5},
        {"name": "ad_unit_4", "id": "4", "bidFloor": 2.0},
    ]
    # For max_ad_units=2, the logic will try to find the ad unit whose bidFloor
    #  * HIGH_MULTIPLIER is closest to 1.2 * 1.5 = 1.8
    result = predictor.predict(context, floors, max_ad_units=2)
    returned_ids = set(result["cpmFloorAdUnitIds"])
    lowest = min(floors, key=lambda x: x["bidFloor"])
    returned_ids.discard(lowest["id"])
    assert len(returned_ids) == 1
    returned_id = returned_ids.pop()
    returned_bidfloor = next(f["bidFloor"] for f in floors if f["id"] == returned_id)
    target = context["user.avgInterRevenueLast72Hours"] * predictor.HIGH_MULTIPLIER
    closest = min(
        [f for f in floors if f["id"] != lowest["id"]],
        key=lambda x: abs(x["bidFloor"] * predictor.HIGH_MULTIPLIER - target),
    )
    assert abs(returned_bidfloor * predictor.HIGH_MULTIPLIER - target) == pytest.approx(
        abs(closest["bidFloor"] * predictor.HIGH_MULTIPLIER - target), abs=1e-6
    )

    # Context value to match
    context = pd.Series({"user.avgInterRevenueLast72Hours": 2.7})
    # Ad units with various bid floors
    floors = [
        {"name": "ad_unit_1", "id": "1", "bidFloor": 1.0},
        {"name": "ad_unit_2", "id": "2", "bidFloor": 2.5},
        {"name": "ad_unit_3", "id": "3", "bidFloor": 3.1},
        {"name": "ad_unit_4", "id": "4", "bidFloor": 4.0},
    ]
    # max_ad_units=1 should select the ad unit with bidFloor closest to 2.7 * HIGH_MULTIPLER (1.5)
    # But for max_ad_units=1, the logic just returns the lowest, so test for max_ad_units=2
    # For max_ad_units=2, the logic will try to find the ad unit whose bidFloor * HIGH_MULTIPLER is 
    # closest to 2.7 * 1.5 = 4.05
    # So ad_unit_3 (3.1*1.5=4.65) and ad_unit_4 (4.0*1.5=6.0) are possible, but 3.1 is closer
    result = predictor.predict(context, floors, max_ad_units=2)
    # The returned ad unit should be the one whose bidFloor*HIGH_MULTIPLER is closest to 4.05
    # Find which ad unit was returned (should be 3.1 or 2.5, but 3.1 is closer)
    returned_ids = set(result["cpmFloorAdUnitIds"])
    # The lowest bid floor is always included, so remove it
    lowest = min(floors, key=lambda x: x["bidFloor"])
    returned_ids.discard(lowest["id"])
    # Only one ad unit should remain
    assert len(returned_ids) == 1
    returned_id = returned_ids.pop()
    # Find the bidFloor of the returned ad unit
    returned_bidfloor = next(f["bidFloor"] for f in floors if f["id"] == returned_id)
    # Compute which is closest
    target = context["user.avgInterRevenueLast72Hours"] * predictor.HIGH_MULTIPLIER
    closest = min(
        [f for f in floors if f["id"] != lowest["id"]],
        key=lambda x: abs(x["bidFloor"] * predictor.HIGH_MULTIPLIER - target),
    )
    assert abs(returned_bidfloor * predictor.HIGH_MULTIPLIER - target) == pytest.approx(
        abs(closest["bidFloor"] * predictor.HIGH_MULTIPLIER - target), abs=1e-6
    )


def test_update_rng_compatibility():
    predictor = NearestAdUnitPredictor()
    # Save original RNG state
    before = predictor.rng_exploration.random()
    # Simulate model_handler.py update_rng logic
    base_seed = 12345
    ss = np.random.SeedSequence(base_seed)
    rngs = [np.random.default_rng(s) for s in ss.spawn(2)]
    predictor.rng_exploration, predictor.rng_shuffle = rngs
    after = predictor.rng_exploration.random()
    # The new state should be deterministic and different from before