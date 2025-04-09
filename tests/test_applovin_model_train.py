import datetime

import numpy as np
import pandas as pd
import pytest
import ray
from ray.train.xgboost import RayTrainReportCallback

from bid_optim_etl_py.applovin_train_runner import ModelTrainer  # Replace with the actual import path


@pytest.fixture(scope="module")
def ray_cluster():
    ray.init()
    yield
    ray.shutdown()


@pytest.fixture
def model_object():
    return ModelTrainer(customer_id=1, app_id=1, model_id="test_model", date=datetime.datetime.now())


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
            {"category": "D", "adUnitId": "a", "device": "Samsung", "value": 1, "NoneCol": None}
        ]
        return ray.data.from_items(data)

    def test_calculate_impression_count(self, ray_cluster, sample_dataset):
        trainer = ModelTrainer(customer_id=1, app_id=1, model_id="test_model", date=datetime.datetime.now())
        result = trainer.value_replacer_based_on_impressions(sample_dataset, columns=["category"],
                                                             min_impressions=2, default_category="other")
        assert sorted(result.valid_values["category"]) == sorted(["A", "B", "C"])

    def test_calculate_impression_count_for_multiple_columns(self, ray_cluster, sample_dataset):
        trainer = ModelTrainer(customer_id=1, app_id=1, model_id="test_model", date=datetime.datetime.now())
        result = trainer.value_replacer_based_on_impressions(sample_dataset, columns=["category", "adUnitId", "device"],
                                                             min_impressions=2, default_category="default")
        assert result.valid_values == {
            "category": ["A", "B", "C"],
            "adUnitId": ["a", "b"],
            "device": ["iPhone"]
        }
        assert result.default_value == "default"

    def test_map_batches(self, ray_cluster, sample_dataset):
        trainer = ModelTrainer(customer_id=1, app_id=1, model_id="test_model", date=datetime.datetime.now())
        result = trainer.value_replacer_based_on_impressions(sample_dataset,
                                                             columns=["category", "adUnitId", "device", "not_present",
                                                                      "NoneCol"],
                                                             min_impressions=2, default_category="other")
        transformed_ds = sample_dataset.map_batches(result.transform, batch_format="pandas")
        transformed_data = transformed_ds.to_pandas()
        assert all(col in transformed_data.columns for col in ["category", "adUnitId", "device"])
        assert "not_present" not in transformed_data.columns
        pd.testing.assert_frame_equal(
            transformed_data[["category", "adUnitId", "device", "NoneCol"]].sort_values(by=["category", "adUnitId"]),
            pd.DataFrame({
                "category": ["A", "B", "A", "C", "B", "A", "C", "C", "other", "other"],
                "adUnitId": ["a", "b", "a", "a", "a", "b", "a", "a", "other", "a"],
                "device": ["iPhone"] * 9 + ["other"],
                "NoneCol": [None] * 10
            }).sort_values(by=["category", "adUnitId"]),
            check_dtype=False,
            check_exact=False,
            check_like=True,
        )


class TestFeatures:
    def test_feature_assembler(self, model_object):
        df = model_object.features.assemble_fields_from_series(pd.Series({"user.country": "US"}))
        assert df is not None
        assert df.shape == (1, 16)

    def test_feature_assembler_with_none(self, model_object):
        df = model_object.features.assemble_fields_from_series(pd.Series({"user.country": None}))
        assert df is not None
        assert df.shape == (1, 16)

    def test_feature_assembler_with_empty(self, model_object):
        df = model_object.features.assemble_fields_from_series(pd.Series({}))
        assert df is not None
        assert df.shape == (1, 16)

    def test_feature_assembler_with_empty_df(self, model_object):
        df = model_object.features.assemble_fields_from_df(pd.DataFrame({}))
        assert df is not None
        assert df.shape == (1, 16)


class TestModelTrainingRun:
    @pytest.fixture
    def training_data(self):
        return ray.data.from_pandas(pd.DataFrame({
            "user.country": ["US", "CA", "GB"],
            "user.deviceModel": ["iPhone 16", "Galaxy 1", "Pixel 1"],
            "user.deviceType": ["iPhone", "Samsung", "Google"],
            "user.languageCode": ["en", "en", "en"],
            "user.osVersion": [None, "16.5", None],
            "user.mostRecentAdSource": ["source1", "source2", "source3"],
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
            "propensities": [0.1, 0.2, 0.3],
        }))

    def generic_model_training_test(self, model_object, training_data, cross_validation, drop_columns=None,
                                    expected_values=None):
        # Drop specified columns if any
        if drop_columns:
            training_data = training_data.drop_columns(drop_columns)

        # Run the training process
        result, value_replacer, features = model_object.run(
            training_data,
            target_column="totalAmount",
            cross_validation=cross_validation
        )

        # Extract the trained model
        booster = RayTrainReportCallback.get_model(result.checkpoint)

        # Generate predictions
        predicted_value = booster.predict(
            features.fields_to_dmatrix_from_df(
                value_replacer.transform(training_data.to_pandas()),
                prediction_phase=True
            )
        )

        # Debug output
        print(predicted_value)

        # Assertions
        assert predicted_value is not None
        assert len(predicted_value) == training_data.count()
        np.testing.assert_allclose(predicted_value, np.array(expected_values), rtol=1e-3, atol=2e-2)

    def test_train_without_cross_validation(self, model_object, training_data):
        self.generic_model_training_test(
            model_object,
            training_data,
            cross_validation=False,
            expected_values=[0.100221, 0.199911, 0.299453]
        )

    def test_train_with_cross_validation(self, model_object, training_data):
        self.generic_model_training_test(
            model_object,
            training_data,
            cross_validation=True,
            expected_values=[0.10495479, 0.18830131, 0.10495479]
        )

    def test_train_with_cross_validation_no_propensities(self, model_object, training_data):
        self.generic_model_training_test(
            model_object,
            training_data,
            cross_validation=True,
            drop_columns=["propensities"],
            expected_values=[0.10495479, 0.18830131, 0.10495479]
        )

    def test_train_without_cross_validation_no_propensities(self, model_object, training_data):
        self.generic_model_training_test(
            model_object,
            training_data,
            cross_validation=False,
            drop_columns=["propensities"],
            expected_values=[0.101092, 0.200001, 0.298907]
        )
