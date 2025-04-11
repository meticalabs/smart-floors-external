import datetime
from unittest.mock import Mock
import pandas as pd
import pytest
from bid_optim_etl_py.applovin_train_runner import Predictor, Features, Field, ValueReplacer


@pytest.fixture
def mock_xgboost_model():
    mock_model = Mock()
    mock_model.predict = Mock(return_value=[1.0])
    return mock_model


@pytest.mark.parametrize(
    "ad_unit_list, expected_rest, expected_lowest",
    [
        (
                [
                    {"name": "metica_ad_unit_3", "id": "3", "bidFloor": 3.0},
                    {"name": "metica_ad_unit_1", "id": "1", "bidFloor": 1.0},
                    {"name": "metica_ad_unit_2", "id": "2", "bidFloor": 2.0},
                ],
                [
                    {"name": "metica_ad_unit_3", "id": "3", "bidFloor": 3.0},
                    {"name": "metica_ad_unit_2", "id": "2", "bidFloor": 2.0},
                ],
                {"name": "metica_ad_unit_1", "id": "1", "bidFloor": 1.0},
        ),
        (
                [
                    {"name": "metica_ad_unit_10", "id": "10", "bidFloor": 10.0},
                    {"name": "metica_ad_unit_2", "id": "2", "bidFloor": 2.0},
                    {"name": "metica_ad_unit_5", "id": "5", "bidFloor": 5.0},
                ],
                [
                    {"name": "metica_ad_unit_10", "id": "10", "bidFloor": 10.0},
                    {"name": "metica_ad_unit_5", "id": "5", "bidFloor": 5.0},
                ],
                {"name": "metica_ad_unit_2", "id": "2", "bidFloor": 2.0},
        ),
    ],
)
def test_split_based_on_name(ad_unit_list, expected_rest, expected_lowest):
    predictor = Predictor(epsilon=0.5)
    rest, lowest = predictor.split_based_on_name(ad_unit_list)
    assert rest == expected_rest
    assert lowest == expected_lowest


@pytest.mark.parametrize(
    "assignments, expected_sorted",
    [
        (
                [
                    {"name": "metica_ad_unit_3", "id": "3", "bidFloor": 3.0},
                    {"name": "metica_ad_unit_1", "id": "1", "bidFloor": 1.0},
                    {"name": "metica_ad_unit_2", "id": "2", "bidFloor": 2.0},
                ],
                [
                    {"name": "metica_ad_unit_3", "id": "3", "bidFloor": 3.0},
                    {"name": "metica_ad_unit_2", "id": "2", "bidFloor": 2.0},
                    {"name": "metica_ad_unit_1", "id": "1", "bidFloor": 1.0},
                ],
        ),
        (
                [
                    {"name": "metica_ad_unit_10", "id": "3", "bidFloor": 3.0},
                    {"name": "metica_ad_unit_01", "id": "1", "bidFloor": 1.0},
                    {"name": "metica_ad_unit_30", "id": "2", "bidFloor": 2.0},
                ],
                [
                    {"name": "metica_ad_unit_30", "id": "2", "bidFloor": 2.0},
                    {"name": "metica_ad_unit_10", "id": "3", "bidFloor": 3.0},
                    {"name": "metica_ad_unit_01", "id": "1", "bidFloor": 1.0},
                ],
        ),
    ],
)
def test_sort_assignment_by_ad_unit_name(assignments, expected_sorted):
    predictor = Predictor(epsilon=0.5)
    sorted_assignments = predictor.sort_by_name_postfix_desc(assignments)
    assert sorted_assignments == expected_sorted


def test_predicts_random_assignment_when_model_is_not_trained():
    predictor = Predictor(epsilon=0.5)
    context = pd.Series({"user.country": "US"})
    floors = [
        {"name": "metica_ad_unit_1", "id": "1", "bidFloor": 1.0},
        {"name": "metica_ad_unit_2", "id": "2", "bidFloor": 2.0},
        {"name": "metica_ad_unit_3", "id": "3", "bidFloor": 3.0},
    ]
    result = predictor.predict(context, floors)
    assert result == {
        "cpmFloorAdUnitIds": ["3", "2", "1"],
        "cpmFloorValues": [3.0, 2.0, 1.0],
        "propensity": 0.5,
    }


def test_empty_floors_throws_exception():
    predictor = Predictor(epsilon=0.5)
    context = pd.Series({"user.country": "US"})
    with pytest.raises(Exception):
        predictor.predict(context, [])


def test_handles_missing_context_fields(mock_xgboost_model):
    predictor = Predictor(
        epsilon=0.1,
        rng=Mock(uniform=Mock(return_value=0.5)),
        clf=mock_xgboost_model,
        features=Features(fields=[
            Field(name="user.country", dtype="category"),
            Field(name="highestBidFloorValue", dtype="float32"),
            Field(name="mediumBidFloorValue", dtype="float32"),
        ]),
        value_replacer=ValueReplacer(
            valid_values={"user.country": ["US"]}, default_value="other"
        ),
    )
    result = predictor.predict(pd.Series({}), [
        {"name": "metica_ad_unit_1", "id": "1", "bidFloor": 1.0},
        {"name": "metica_ad_unit_2", "id": "2", "bidFloor": 2.0},
        {"name": "metica_ad_unit_3", "id": "3", "bidFloor": 3.0},
    ])
    mock_xgboost_model.predict.assert_called_once()
    assert "cpmFloorAdUnitIds" in result
    assert "cpmFloorValues" in result
    assert "propensity" in result


def test_adds_hardcoded_context_values():
    predictor = Predictor(epsilon=0.5)
    context = pd.Series({"user.country": "US"})
    bid_floor_adunit = [{"bidFloor": 5.0}, {"bidFloor": 3.0}]
    result = predictor.add_hardcoded_contexts(context, bid_floor_adunit)
    now = datetime.datetime.now(datetime.timezone.utc)
    pd.testing.assert_series_equal(result, pd.Series({
        "user.country": "US",
        "assignmentDayOfWeek": now.weekday(),
        "assignmentHourOfDay": now.hour,
        "highestBidFloorValue": 5.0,
        "mediumBidFloorValue": 3.0,
    }))


def test_handles_empty_context_series():
    predictor = Predictor(epsilon=0.5)
    result = predictor.add_hardcoded_contexts(pd.Series({}), [
        {"bidFloor": 5.0}, {"bidFloor": 3.0},
    ])
    now = datetime.datetime.now(datetime.timezone.utc)
    pd.testing.assert_series_equal(result, pd.Series({
        "assignmentDayOfWeek": now.weekday(),
        "assignmentHourOfDay": now.hour,
        "highestBidFloorValue": 5.0,
        "mediumBidFloorValue": 3.0,
    }))


def test_handles_empty_bid_floor_adunit():
    predictor = Predictor(epsilon=0.5)
    with pytest.raises(IndexError):
        predictor.add_hardcoded_contexts(pd.Series({"user.country": "US"}), [])


def test_forms_response_with_valid_assignments_and_lowest_bid_floor():
    predictor = Predictor(epsilon=0.5)
    result = predictor.form_response(
        [{"id": "1", "bidFloor": 2.0}, {"id": "2", "bidFloor": 3.0}],
        {"id": "3", "bidFloor": 1.0},
        0.8,
    )
    assert result == {
        "cpmFloorAdUnitIds": ["1", "2", "3"],
        "cpmFloorValues": [2.0, 3.0, 1.0],
        "propensity": 0.8,
    }


def test_handles_empty_assignments_list_fill_with_lowest():
    predictor = Predictor(epsilon=0.5)
    result = predictor.form_response([], {"id": "1", "bidFloor": 1.0}, 0.5)
    assert result == {
        "cpmFloorAdUnitIds": ["1"] * 3,
        "cpmFloorValues": [1.0] * 3,
        "propensity": 0.5,
    }


def test_handles_missing_lowest_bid_floor():
    predictor = Predictor(epsilon=0.5)
    with pytest.raises(Exception):
        predictor.form_response([{"id": "1", "bidFloor": 2.0}], {}, 0.7)


def test_handles_empty_assignments_and_missing_lowest_bid_floor():
    predictor = Predictor(epsilon=0.5)
    with pytest.raises(Exception):
        predictor.form_response([], {}, 0.3)
