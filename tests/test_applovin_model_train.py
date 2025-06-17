import datetime

import numpy as np
import pandas as pd
import pytest
import ray
from ray.train.xgboost import RayTrainReportCallback

from bid_optim_etl_py.applovin_train_runner import ModelTrainer, ValueReplacer, ModelFeatures, ModelConfig, Features
from bid_optim_etl_py.utils.management_api import ETLConfig


@pytest.fixture(scope="session", autouse=True)
def ray_cluster():
    ray.init(num_cpus=4)
    yield
    ray.shutdown()


@pytest.fixture
def model_object():
    model_features = ModelFeatures(
        etl_config=ETLConfig(
            **{
                "context": [
                    {"name": "user.country", "dataType": "string"},
                    {"name": "user.languageCode", "dataType": "string"},
                    {"name": "user.deviceType", "dataType": "string"},
                    {"name": "user.osVersion", "dataType": "string"},
                    {"name": "user.deviceModel", "dataType": "string"},
                    {"name": "user.minRevenueLast24Hours", "dataType": "number"},
                    {"name": "user.avgRevenueLast24Hours", "dataType": "number"},
                    {"name": "user.avgRevenueLast48Hours", "dataType": "number"},
                    {"name": "user.avgRevenueLast72Hours", "dataType": "number"},
                    {"name": "user.mostRecentAdRevenue", "dataType": "number"},
                ],
                "lookbackWindowInDays": 30,
            }
        )
    ).as_ray_schema()
    model_config = ModelConfig(
        **{
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "learning_rate": 0.1,
            "max_depth": 4,
            "eval_metric": ["rmse", "mae"],
            "num_boost_rounds": 10,
        }
    )
    return ModelTrainer(
        customer_id=1,
        app_id=1,
        model_id="test_model",
        date=datetime.datetime.now(),
        num_boost_round=10,
        features=model_features,
        model_config=model_config,
    )


class TestImpressionCount:

    @pytest.fixture
    def sample_dataset(self):
        data = [
            {"category": "A", "adUnitId": "a", "device": "iPhone", "value": 1, "NoneCol": None},
            {"category": "B", "adUnitId": "b", "device": "iPhone", "value": 1, "NoneCol": None},
            {"category": "A", "adUnitId": "a", "device": "iPhone", "value": 1, "NoneCol": None},
            {"category": "C", "adUnitId": "a", "device": "iPhone", "value": 1, "NoneCol": None},
            {"category": "B", "adUnitId": "a", "device": "iPhone", "value": 1, "NoneCol": None},
            {"category": "A", "adUnitId": "b", "device": "iPhone", "value": 1, "NoneCol": None},
            {"category": "C", "adUnitId": "a", "device": "iPhone", "value": 1, "NoneCol": None},
            {"category": "C", "adUnitId": "a", "device": "iPhone", "value": 1, "NoneCol": None},
            {"category": "E", "adUnitId": "c", "device": "iPhone", "value": 1, "NoneCol": None},
            {"category": "D", "adUnitId": "a", "device": "Samsung", "value": 1, "NoneCol": None},
        ]
        return ray.data.from_items(data)

    @pytest.fixture
    def model_features(self) -> Features:
        return ModelFeatures(
            etl_config=ETLConfig(
                **{
                    "context": [
                        {"name": "user.country", "dataType": "string"},
                        {"name": "user.languageCode", "dataType": "string"},
                        {"name": "user.deviceType", "dataType": "string"},
                        {"name": "user.osVersion", "dataType": "string"},
                        {"name": "user.deviceModel", "dataType": "string"},
                        {"name": "user.minRevenueLast24Hours", "dataType": "number"},
                        {"name": "user.avgRevenueLast24Hours", "dataType": "number"},
                        {"name": "user.avgRevenueLast48Hours", "dataType": "number"},
                        {"name": "user.avgRevenueLast72Hours", "dataType": "number"},
                        {"name": "user.mostRecentAdRevenue", "dataType": "number"},
                    ],
                    "lookbackWindowInDays": 30,
                }
            )
        ).as_ray_schema()

    def test_calculate_impression_count(self, model_features, ray_cluster, sample_dataset):
        trainer = ModelTrainer(
            customer_id=1,
            app_id=1,
            model_id="test_model",
            date=datetime.datetime.now(),
            features=model_features,
            model_config=ModelConfig(
                **{
                    "tree_method": "hist",
                    "objective": "reg:squarederror",
                    "learning_rate": 0.1,
                    "max_depth": 4,
                    "eval_metric": ["rmse", "mae"],
                    "num_boost_rounds": 10,
                }
            ),
        )
        result = trainer.value_replacer_based_on_impressions(
            sample_dataset, columns=["category"], min_impressions=2, default_category="other"
        )
        assert sorted(result.valid_values["category"]) == sorted(["A", "B", "C"])

    def test_calculate_impression_count_for_multiple_columns(self, model_features, ray_cluster, sample_dataset):
        trainer = ModelTrainer(
            customer_id=1,
            app_id=1,
            model_id="test_model",
            date=datetime.datetime.now(),
            features=model_features,
            model_config=ModelConfig(
                **{
                    "tree_method": "hist",
                    "objective": "reg:squarederror",
                    "learning_rate": 0.1,
                    "max_depth": 4,
                    "eval_metric": ["rmse", "mae"],
                    "num_boost_rounds": 10,
                }
            ),
        )
        result = trainer.value_replacer_based_on_impressions(
            sample_dataset, columns=["category", "adUnitId", "device"], min_impressions=2, default_category="default"
        )
        assert result.valid_values == {"category": ["A", "B", "C"], "adUnitId": ["a", "b"], "device": ["iPhone"]}
        assert result.default_value == "default"

    @pytest.mark.parametrize(
        "value_replacer, input_data, expected_output",
        [
            (
                ValueReplacer(valid_values={"category": ["A", "B"]}, default_value="other"),
                {"category": "A"},
                {"category": "A"},
            ),
            (
                ValueReplacer(valid_values={"category": ["A"]}, default_value="other"),
                {"category": "B"},
                {"category": "other"},
            ),
            (
                ValueReplacer(valid_values={"category": ["A", "B"]}, default_value="other"),
                {"category": None},
                {"category": "other"},
            ),
            (
                ValueReplacer(valid_values={"category": ["A", "B", None]}, default_value="other"),
                {"category": None},
                {"category": None},
            ),
            (
                ValueReplacer(valid_values={"category": ["A", "B"]}, default_value="other"),
                {"category": "C"},
                {"category": "other"},
            ),
            (
                ValueReplacer(valid_values={"category": ["A", "B"], "device": ["iPhone"]}, default_value="other"),
                {"category": "A", "device": "iPhone"},
                {"category": "A", "device": "iPhone"},
            ),
            (
                ValueReplacer(valid_values={"category": ["A", "B"], "device": ["iPhone"]}, default_value="other"),
                {"category": "C", "device": "iPhone"},
                {"category": "other", "device": "iPhone"},
            ),
        ],
    )
    def test_value_replacer_transform(self, value_replacer, input_data, expected_output):
        output = value_replacer.transform_series(pd.Series(input_data))
        assert output.to_dict() == expected_output

    def test_map_batches(self, ray_cluster, sample_dataset):
        trainer = ModelTrainer(
            customer_id=1,
            app_id=1,
            model_id="test_model",
            date=datetime.datetime.now(),
            features=Features(),
            model_config=ModelConfig(
                **{
                    "tree_method": "hist",
                    "objective": "reg:squarederror",
                    "learning_rate": 0.1,
                    "max_depth": 4,
                    "eval_metric": ["rmse", "mae"],
                    "num_boost_rounds": 10,
                }
            ),
        )
        result = trainer.value_replacer_based_on_impressions(
            sample_dataset,
            columns=["category", "adUnitId", "device", "not_present", "NoneCol"],
            min_impressions=2,
            default_category="other",
        )
        transformed_ds = sample_dataset.map_batches(result.transform, batch_format="pandas")
        transformed_data = transformed_ds.to_pandas()

        # Validate columns
        expected_columns = ["category", "adUnitId", "device", "NoneCol"]
        assert all(col in transformed_data.columns for col in expected_columns)
        assert "not_present" not in transformed_data.columns

        # Validate row count
        expected_data = pd.DataFrame(
            {
                "category": ["A", "B", "A", "C", "B", "A", "C", "C", "other", "other"],
                "adUnitId": ["a", "b", "a", "a", "a", "b", "a", "a", "other", "a"],
                "device": ["iPhone"] * 9 + ["other"],
                "NoneCol": [None] * 10,
            }
        )
        assert len(transformed_data) == len(expected_data)

        # Sort both DataFrames
        transformed_data = transformed_data[expected_columns].sort_values(by=expected_columns).reset_index(drop=True)
        expected_data = expected_data.sort_values(by=expected_columns).reset_index(drop=True)

        # Validate each column
        for col in expected_columns:
            pd.testing.assert_series_equal(transformed_data[col], expected_data[col], check_dtype=False)


class TestFeatures:
    def test_feature_assembler(self, model_object):
        df = model_object.features.assemble_fields_from_series(pd.Series({"user.country": "US"}))
        assert df is not None
        assert df.shape == (1, 15)

    def test_feature_assembler_with_none(self, model_object):
        df = model_object.features.assemble_fields_from_series(pd.Series({"user.country": None}))
        assert df is not None
        assert df.shape == (1, 15)

    def test_feature_assembler_with_empty(self, model_object):
        df = model_object.features.assemble_fields_from_series(pd.Series({}))
        assert df is not None
        assert df.shape == (1, 15)

    def test_feature_assembler_with_empty_df(self, model_object):
        df = model_object.features.assemble_fields_from_df(pd.DataFrame({}))
        assert df is not None
        assert df.shape == (1, 15)


class TestModelTrainingRun:
    @pytest.fixture
    def training_data(self):
        return ray.data.from_pandas(
            pd.DataFrame(
                {
                    "user.country": ["US", "CA", "GB"],
                    "user.deviceModel": ["iPhone 16", "Galaxy 1", "Pixel 1"],
                    "user.deviceType": ["iPhone", "Samsung", "Google"],
                    "user.languageCode": ["en", "en", "en"],
                    "user.osVersion": [None, "16.5", None],
                    "user.mostRecentAdRevenue": [0.1, 0.2, 0.3],
                    "user.avgRevenueLast24Hours": [0.1, 0.2, 0.3],
                    "user.avgRevenueLast48Hours": [0.1, 0.2, 0.3],
                    "user.avgRevenueLast72Hours": [0.01, 0.02, 0.4],
                    "user.minRevenueLast24Hours": [0.1, 0.2, 0.3],
                    "assignmentDayOfWeek": [1, 2, 1],
                    "assignmentHourOfDay": [1, 2, 3],
                    "highestBidFloorValue": [1.0, 0.2, 0.5],
                    "mediumBidFloorValue": [0.5, 0.2, 0.3],
                    "totalAmount": [0.1, 0.2, 0.3],
                    "propensity": [0.1, 0.2, 0.3],
                }
            )
        )

    def generic_model_training_test(
        self, model_object, training_data, use_validation, drop_columns=None, expected_values=None
    ):
        # Drop specified columns if any
        if drop_columns:
            training_data = training_data.drop_columns(drop_columns)

        # Run the training process
        result, value_replacer, features = model_object.run(
            training_data, target_column="totalAmount", use_validation_set=use_validation
        )

        # Extract the trained model
        booster = RayTrainReportCallback.get_model(result.checkpoint)

        # Generate predictions
        predicted_value = booster.predict(
            features.fields_to_dmatrix_from_df(
                value_replacer.transform(training_data.to_pandas()), prediction_phase=True
            )
        )

        # Debug output
        print(predicted_value)

        # Assertions
        assert predicted_value is not None
        assert len(predicted_value) == training_data.count()
        np.testing.assert_allclose(predicted_value, np.array(expected_values), rtol=1e-3, atol=2e-2)

    def test_train_without_validation(self, model_object, training_data):
        self.generic_model_training_test(
            model_object, training_data, use_validation=False, expected_values=[0.124535, 0.184767, 0.238754]
        )

    def test_train_with_validation(self, model_object, training_data):
        self.generic_model_training_test(
            model_object, training_data, use_validation=True, expected_values=[0.129937, 0.170063, 0.129937]
        )

    def test_train_with_validation_no_propensities(self, model_object, training_data):
        self.generic_model_training_test(
            model_object,
            training_data,
            use_validation=True,
            drop_columns=["propensity"],
            expected_values=[0.129937, 0.170063, 0.129937],
        )

    def test_train_without_validation_no_propensities(self, model_object, training_data):
        self.generic_model_training_test(
            model_object,
            training_data,
            use_validation=False,
            drop_columns=["propensity"],
            expected_values=[0.159874, 0.2, 0.240126],
        )
