import datetime
from unittest.mock import Mock, patch
import pandas as pd
import pytest
from bid_optim_etl_py.applovin_train_runner import Predictor, Features, Field, ValueReplacer


floors = [
    {"name": "metica_ad_unit_4", "id": "4", "bidFloor": 4.0},
    {"name": "metica_ad_unit_3", "id": "3", "bidFloor": 3.0},
    {"name": "metica_ad_unit_2", "id": "2", "bidFloor": 2.0},
    {"name": "metica_ad_unit_1", "id": "1", "bidFloor": 1.0},
]

features = Features(
    fields=[
        Field(name="user.country", dtype="category"),
        Field(name="assignmentDayOfWeek", dtype="Int64"),
        Field(name="assignmentHourOfDay", dtype="Int64"),
        Field(name="highestBidFloorValue", dtype="float32"),
        Field(name="mediumBidFloorValue", dtype="float32"),
    ]
)


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


def test_predict_exploration_path():
    # Test when clf is None, ensuring random assignment and correct propensity
    predictor = Predictor(epsilon=0.5)
    context = pd.Series({"user.country": "US"})
    floors_data = [
        {"name": "metica_ad_unit_1", "id": "1", "bidFloor": 1.0},
        {"name": "metica_ad_unit_2", "id": "2", "bidFloor": 2.0},
        {"name": "metica_ad_unit_3", "id": "3", "bidFloor": 3.0},
        {"name": "metica_ad_unit_4", "id": "4", "bidFloor": 4.0},
    ]
    # Mock rng_exploration to return a predictable random choice
    predictor.rng_exploration = Mock()
    predictor.rng_exploration.choice.return_value = [[floors_data[0], floors_data[1]]]
    result = predictor.predict(context, floors_data)

    # There are 3 possible combinations of 2 from 3 (excluding the lowest): (4,3), (4,2), (3,2)
    # The lowest bid floor is 1.0, so we consider 4,3,2. Combinations are (4,3), (4,2), (3,2)
    # If we choose [1,2] from the original floors_data, it means we chose the first two elements
    # after sorting by postfix desc.
    # The combinations are (4,3), (4,2), (3,2)
    # If we choose [floors_data[0], floors_data[1]] which are 4 and 3, then the propensity should be 1/3.
    assert result["cpmFloorAdUnitIds"] == ["1", "2", "1"]
    assert result["cpmFloorValues"] == [1.0, 2.0, 1.0]
    assert pytest.approx(result["propensity"], abs=1e-6) == 1 / 3.0
    assert result["estimates"][0]["predictedBidFloor"] == -1.0

def test_add_hardcoded_contexts():
    predictor = Predictor(epsilon=0.5)
    context = pd.Series({"existing_key": "existing_value"})
    bid_floor_adunit = [{"bidFloor": 10.0}, {"bidFloor": 5.0}]
    
    # Mock datetime.datetime.now to ensure consistent test results
    with patch('bid_optim_etl_py.applovin_train_runner.datetime') as mock_datetime:
        mock_datetime.datetime.now.return_value = datetime.datetime(2025, 7, 1, 10, 30, 0, tzinfo=datetime.timezone.utc)

        updated_context = predictor.add_hardcoded_contexts(context, bid_floor_adunit)

        assert updated_context["existing_key"] == "existing_value"
        assert updated_context["assignmentDayOfWeek"] == 1 # Tuesday
        assert updated_context["assignmentHourOfDay"] == 10
        assert updated_context["highestBidFloorValue"] == 10.0
        assert updated_context["mediumBidFloorValue"] == 5.0

def test_split_based_on_name():
    predictor = Predictor(epsilon=0.5)
    ad_unit_list = [
        {"name": "metica_ad_unit_3", "id": "3", "bidFloor": 3.0},
        {"name": "metica_ad_unit_1", "id": "1", "bidFloor": 1.0},
        {"name": "metica_ad_unit_2", "id": "2", "bidFloor": 2.0},
    ]
    rest, lowest = predictor.split_based_on_name(ad_unit_list)
    assert lowest == {"name": "metica_ad_unit_1", "id": "1", "bidFloor": 1.0}
    assert len(rest) == 2
    assert {"name": "metica_ad_unit_3", "id": "3", "bidFloor": 3.0} in rest
    assert {"name": "metica_ad_unit_2", "id": "2", "bidFloor": 2.0} in rest

def test_form_response():
    predictor = Predictor(epsilon=0.5)
    assignments = [
        {"id": "A", "bidFloor": 10.0},
        {"id": "B", "bidFloor": 20.0},
    ]
    lowest_bid_floor = {"id": "C", "bidFloor": 5.0}
    propensity = 0.75
    prediction_estimates = [{"adUnitIds": ["A", "B"], "predictedBidFloor": 15.0}]

    response = predictor.form_response(assignments, lowest_bid_floor, propensity, prediction_estimates)

    assert response["cpmFloorAdUnitIds"] == ["A", "B", "C"]
    assert response["cpmFloorValues"] == [10.0, 20.0, 5.0]
    assert response["propensity"] == propensity
    assert response["estimates"] == prediction_estimates


def test_empty_floors_throws_exception():
    predictor = Predictor(epsilon=0.5)
    context = pd.Series({"user.country": "US"})
    with pytest.raises(Exception):
        predictor.predict(context, [])


def test_handles_missing_context_fields(mock_xgboost_model):
    predictor = Predictor(
        epsilon=0.1,
        rng_exploration=Mock(uniform=Mock(return_value=0.5)),
        clf=mock_xgboost_model,
        features=Features(
            fields=[
                Field(name="user.country", dtype="category"),
                Field(name="highestBidFloorValue", dtype="float32"),
                Field(name="mediumBidFloorValue", dtype="float32"),
            ]
        ),
        value_replacer=ValueReplacer(valid_values={"user.country": ["US"]}, default_value="other"),
    )
    result = predictor.predict(
        pd.Series({}),
        [
            {"name": "metica_ad_unit_1", "id": "1", "bidFloor": 1.0},
            {"name": "metica_ad_unit_2", "id": "2", "bidFloor": 2.0},
            {"name": "metica_ad_unit_3", "id": "3", "bidFloor": 3.0},
        ],
    )
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
    pd.testing.assert_series_equal(
        result,
        pd.Series(
            {
                "user.country": "US",
                "assignmentDayOfWeek": now.weekday(),
                "assignmentHourOfDay": now.hour,
                "highestBidFloorValue": 5.0,
                "mediumBidFloorValue": 3.0,
            }
        ),
    )


def test_handles_empty_context_series():
    predictor = Predictor(epsilon=0.5)
    result = predictor.add_hardcoded_contexts(
        pd.Series({}),
        [
            {"bidFloor": 5.0},
            {"bidFloor": 3.0},
        ],
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    pd.testing.assert_series_equal(
        result,
        pd.Series(
            {
                "assignmentDayOfWeek": now.weekday(),
                "assignmentHourOfDay": now.hour,
                "highestBidFloorValue": 5.0,
                "mediumBidFloorValue": 3.0,
            }
        ),
    )


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
        [{"adUnitIds": ["1", "2"], "predictedBidFloor": 2.0}],
    )
    assert result == {
        "cpmFloorAdUnitIds": ["1", "2", "3"],
        "cpmFloorValues": [2.0, 3.0, 1.0],
        "propensity": 0.8,
        "estimates": [{"adUnitIds": ["1", "2"], "predictedBidFloor": 2.0}],
    }


def test_handles_empty_assignments_list_fill_with_lowest():
    predictor = Predictor(epsilon=0.5)
    result = predictor.form_response(
        [], {"id": "1", "bidFloor": 1.0}, 0.5, [{"adUnitIds": ["1"], "predictedBidFloor": 1.0}]
    )
    assert result == {
        "cpmFloorAdUnitIds": ["1"] * 3,
        "cpmFloorValues": [1.0] * 3,
        "propensity": 0.5,
        "estimates": [{"adUnitIds": ["1"], "predictedBidFloor": 1.0}],
    }


def test_handles_missing_lowest_bid_floor():
    predictor = Predictor(epsilon=0.5)
    with pytest.raises(Exception):
        predictor.form_response(
            [{"id": "1", "bidFloor": 2.0}], {}, 0.7, [{"adUnitIds": ["1"], "predictedBidFloor": 2.0}]
        )


def test_handles_empty_assignments_and_missing_lowest_bid_floor():
    predictor = Predictor(epsilon=0.5)
    with pytest.raises(Exception):
        predictor.form_response([], {}, 0.3, [{"adUnitIds": ["1"], "predictedBidFloor": 1.0}])


def test_predict_model_present_random_not_triggered():
    mock_rng = Mock()
    mock_rng.uniform.return_value = 0.2  # >= 0.1, no random choice
    mock_model = Mock()
    mock_model.predict.return_value = [10.0, 8.0, 9.0]  # Best: [4,3]
    predictor = Predictor(
        epsilon=0.1,
        rng_exploration=mock_rng,
        rng_shuffle=mock_rng,
        clf=mock_model,
        features=features,
        value_replacer=None,
    )
    context = pd.Series({"user.country": "US"})
    result = predictor.predict(context, floors)
    expected_response = {
        "cpmFloorAdUnitIds": ["4", "3", "1"],
        "cpmFloorValues": [4.0, 3.0, 1.0],
        "propensity": pytest.approx(0.9333, abs=1e-4),
        "estimates": [
            {
                "adUnitIds": [
                    {"bidFloor": 4.0, "id": "4", "name": "metica_ad_unit_4"},
                    {"bidFloor": 3.0, "id": "3", "name": "metica_ad_unit_3"},
                ],
                "predictedBidFloor": 10.0,
            },
            {
                "adUnitIds": [
                    {"bidFloor": 4.0, "id": "4", "name": "metica_ad_unit_4"},
                    {"bidFloor": 2.0, "id": "2", "name": "metica_ad_unit_2"},
                ],
                "predictedBidFloor": 8.0,
            },
            {
                "adUnitIds": [
                    {"bidFloor": 3.0, "id": "3", "name": "metica_ad_unit_3"},
                    {"bidFloor": 2.0, "id": "2", "name": "metica_ad_unit_2"},
                ],
                "predictedBidFloor": 9.0,
            },
        ],
    }
    assert result == expected_response


def test_predict_model_present_random_triggered_not_best():
    mock_rng = Mock()
    mock_rng.uniform.return_value = 0.05  # < 0.1, triggers random choice
    mock_rng.choice.return_value = [[floors[0], floors[2]]]  # Chooses [4,2]
    mock_model = Mock()
    mock_model.predict.return_value = [10.0, 8.0, 9.0]  # Best: [4,3]
    predictor = Predictor(
        epsilon=0.1,
        rng_exploration=mock_rng,
        rng_shuffle=mock_rng,
        clf=mock_model,
        features=features,
        value_replacer=None,
    )
    context = pd.Series({"user.country": "US"})
    result = predictor.predict(context, floors)
    expected_response = {
        "cpmFloorAdUnitIds": ["4", "2", "1"],
        "cpmFloorValues": [4.0, 2.0, 1.0],
        "propensity": pytest.approx(0.0333, abs=1e-4),
        "estimates": [
            {
                "adUnitIds": [
                    {"bidFloor": 4.0, "id": "4", "name": "metica_ad_unit_4"},
                    {"bidFloor": 3.0, "id": "3", "name": "metica_ad_unit_3"},
                ],
                "predictedBidFloor": 10.0,
            },
            {
                "adUnitIds": [
                    {"bidFloor": 4.0, "id": "4", "name": "metica_ad_unit_4"},
                    {"bidFloor": 2.0, "id": "2", "name": "metica_ad_unit_2"},
                ],
                "predictedBidFloor": 8.0,
            },
            {
                "adUnitIds": [
                    {"bidFloor": 3.0, "id": "3", "name": "metica_ad_unit_3"},
                    {"bidFloor": 2.0, "id": "2", "name": "metica_ad_unit_2"},
                ],
                "predictedBidFloor": 9.0,
            },
        ],
    }
    assert result == expected_response


def test_predict_model_present_random_triggered_is_best():
    mock_rng = Mock()
    mock_rng.uniform.return_value = 0.05  # < 0.1, triggers	random choice
    mock_rng.choice.return_value = [[floors[0], floors[1]]]  # Chooses [4,3]
    mock_model = Mock()
    mock_model.predict.return_value = [10.0, 8.0, 9.0]  # Best: [4,3]
    predictor = Predictor(
        epsilon=0.1,
        rng_exploration=mock_rng,
        rng_shuffle=mock_rng,
        clf=mock_model,
        features=features,
        value_replacer=None,
    )
    context = pd.Series({"user.country": "US"})
    result = predictor.predict(context, floors)
    print(result)
    expected_response = {
        "cpmFloorAdUnitIds": ["4", "3", "1"],
        "cpmFloorValues": [4.0, 3.0, 1.0],
        "propensity": 0.9333333333333333,
        "estimates": [
            {
                "adUnitIds": [
                    {"name": "metica_ad_unit_4", "id": "4", "bidFloor": 4.0},
                    {"name": "metica_ad_unit_3", "id": "3", "bidFloor": 3.0},
                ],
                "predictedBidFloor": 10.0,
            },
            {
                "adUnitIds": [
                    {"name": "metica_ad_unit_4", "id": "4", "bidFloor": 4.0},
                    {"name": "metica_ad_unit_2", "id": "2", "bidFloor": 2.0},
                ],
                "predictedBidFloor": 8.0,
            },
            {
                "adUnitIds": [
                    {"name": "metica_ad_unit_3", "id": "3", "bidFloor": 3.0},
                    {"name": "metica_ad_unit_2", "id": "2", "bidFloor": 2.0},
                ],
                "predictedBidFloor": 9.0,
            },
        ],
    }
    assert result == expected_response


def test_predict_model_with_series_ad_units():
    mock_rng = Mock()
    mock_rng.uniform.return_value = 0.05  # < 0.1, triggers	random choice
    mock_rng.choice.return_value = [[floors[0], floors[1]]]  # Chooses [4,3]
    mock_model = Mock()
    mock_model.predict.return_value = [10.0, 8.0, 9.0]  # Best: [4,3]
    predictor = Predictor(
        epsilon=0.1,
        rng_exploration=mock_rng,
        rng_shuffle=mock_rng,
        clf=mock_model,
        features=features,
        value_replacer=None,
    )
    context = pd.Series({"user.country": "US"})
    floors_series = [pd.Series(floor) for floor in floors]
    result = predictor.predict(context, floors_series)
    expected_response = {
        "cpmFloorAdUnitIds": ["4", "3", "1"],
        "cpmFloorValues": [4.0, 3.0, 1.0],
        "propensity": pytest.approx(0.9333, abs=1e-4),
        "estimates": [
            {
                "adUnitIds": [
                    {"bidFloor": 4.0, "id": "4", "name": "metica_ad_unit_4"},
                    {"bidFloor": 3.0, "id": "3", "name": "metica_ad_unit_3"},
                ],
                "predictedBidFloor": 10.0,
            },
            {
                "adUnitIds": [
                    {"bidFloor": 4.0, "id": "4", "name": "metica_ad_unit_4"},
                    {"bidFloor": 2.0, "id": "2", "name": "metica_ad_unit_2"},
                ],
                "predictedBidFloor": 8.0,
            },
            {
                "adUnitIds": [
                    {"bidFloor": 3.0, "id": "3", "name": "metica_ad_unit_3"},
                    {"bidFloor": 2.0, "id": "2", "name": "metica_ad_unit_2"},
                ],
                "predictedBidFloor": 9.0,
            },
        ],
    }
    assert result == expected_response
