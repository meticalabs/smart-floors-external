import datetime

import numpy as np
import pandas as pd
import pytest
import ray
from ray.train.xgboost import RayTrainReportCallback

from bid_optim_etl_py.applovin_train_runner import (
    ModelTrainer,
    ValueReplacer,
    ModelFeatures,
    ModelConfig,
    Features,
    cast_types_numpy,
)
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
                    {"path": "user.country", "dataType": "string"},
                    {"path": "user.languageCode", "dataType": "string"},
                    {"path": "user.deviceType", "dataType": "string"},
                    {"path": "user.osVersion", "dataType": "string"},
                    {"path": "user.deviceModel", "dataType": "string"},
                    {"path": "user.minRevenueLast24Hours", "dataType": "number"},
                    {"path": "user.avgRevenueLast24Hours", "dataType": "number"},
                    {"path": "user.avgRevenueLast48Hours", "dataType": "number"},
                    {"path": "user.avgRevenueLast72Hours", "dataType": "number"},
                    {"path": "user.mostRecentAdRevenue", "dataType": "number"},
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
                        {"path": "user.country", "dataType": "string"},
                        {"path": "user.languageCode", "dataType": "string"},
                        {"path": "user.deviceType", "dataType": "string"},
                        {"path": "user.osVersion", "dataType": "string"},
                        {"path": "user.deviceModel", "dataType": "string"},
                        {"path": "user.minRevenueLast24Hours", "dataType": "number"},
                        {"path": "user.avgRevenueLast24Hours", "dataType": "number"},
                        {"path": "user.avgRevenueLast48Hours", "dataType": "number"},
                        {"path": "user.avgRevenueLast72Hours", "dataType": "number"},
                        {"path": "user.mostRecentAdRevenue", "dataType": "number"},
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

    def test_calculate_impression_count_for_empty_ds(self, model_features, ray_cluster, sample_dataset):
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
        empty_dataset = ray.data.from_items([])
        result = trainer.value_replacer_based_on_impressions(
            empty_dataset, columns=["category"], min_impressions=2, default_category="default"
        )
        assert result.valid_values == {}
        assert result.default_value == "default"

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

    def test_calculate_impression_count_min_impressions_one(self, model_features, ray_cluster, sample_dataset):
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
            sample_dataset, columns=["category"], min_impressions=1, default_category="other"
        )
        assert sorted(result.valid_values["category"]) == sorted(["A", "B", "C", "D", "E"])

    def test_calculate_impression_count_empty_columns_list(self, model_features, ray_cluster, sample_dataset):
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
            sample_dataset, columns=[], min_impressions=2, default_category="other"
        )
        assert result.valid_values == {}

    def test_calculate_impression_count_non_existent_columns(self, model_features, ray_cluster, sample_dataset):
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
            sample_dataset,
            columns=["non_existent_col1", "non_existent_col2"],
            min_impressions=2,
            default_category="other",
        )
        assert result.valid_values == {}

    def test_calculate_impression_count_single_row_dataset(self, model_features, ray_cluster):
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
        single_row_data = [{"category": "A", "adUnitId": "a", "device": "iPhone", "value": 1, "NoneCol": None}]
        single_row_dataset = ray.data.from_items(single_row_data)
        result = trainer.value_replacer_based_on_impressions(
            single_row_dataset, columns=["category", "adUnitId"], min_impressions=1, default_category="other"
        )
        assert result.valid_values == {"category": ["A"], "adUnitId": ["a"]}

    def test_calculate_impression_count_all_below_min_impressions(self, model_features, ray_cluster):
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
        data = [
            {"category": "A"},
            {"category": "B"},
            {"category": "C"},
        ]
        dataset = ray.data.from_items(data)
        result = trainer.value_replacer_based_on_impressions(
            dataset, columns=["category"], min_impressions=2, default_category="other"
        )
        assert result.valid_values == {}

    def test_calculate_impression_count_with_none_values(self, model_features, ray_cluster):
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
        data = [
            {"category": "A"},
            {"category": None},
            {"category": "A"},
            {"category": None},
            {"category": "B"},
        ]
        dataset = ray.data.from_items(data)
        result = trainer.value_replacer_based_on_impressions(
            dataset, columns=["category"], min_impressions=2, default_category="other"
        )
        assert sorted(result.valid_values["category"], key=lambda x: (x is None, "" if x is None else x)) == sorted(
            ["A", None], key=lambda x: (x is None, "" if x is None else x)
        )

    def test_value_replacer_transform_with_none_and_new_category(self, ray_cluster):
        # Create a sample DataFrame with a categorical column containing None
        data = {"category": ["A", "B", None, "A"], "value": [1, 2, 3, 4]}
        df = pd.DataFrame(data)
        df["category"] = df["category"].astype("category")

        # Define a ValueReplacer where 'new_category' is not in the initial categories
        value_replacer = ValueReplacer(valid_values={"category": ["A", "B"]}, default_value="new_category")

        # Apply the transform method
        transformed_df = value_replacer.transform(df.copy())

        # Assertions
        # Check if 'new_category' is added to categories
        assert "new_category" in transformed_df["category"].cat.categories

        # Check if None values are replaced with 'new_category'
        expected_category = pd.Series(
            pd.Categorical(["A", "B", "new_category", "A"], categories=["A", "B", "new_category"]),
            dtype="category",
            name="category",
        )
        pd.testing.assert_series_equal(transformed_df["category"], expected_category)

        # Test with a column that doesn't exist in valid_values but exists in df
        data_with_extra_col = {"category": ["A", None], "other_col": [10, 20]}
        df_extra = pd.DataFrame(data_with_extra_col)
        df_extra["category"] = df_extra["category"].astype("category")

        value_replacer_extra = ValueReplacer(valid_values={"category": ["A"]}, default_value="default_val")
        transformed_df_extra = value_replacer_extra.transform(df_extra.copy())

        assert "default_val" in transformed_df_extra["category"].cat.categories
        expected_category_extra = pd.Series(
            pd.Categorical(["A", "default_val"], categories=["A", "default_val"]),
            dtype="category",
            name="category",
        )
        pd.testing.assert_series_equal(transformed_df_extra["category"], expected_category_extra)
        assert "other_col" in transformed_df_extra.columns

        # Test with a column that is not in df but is in valid_values
        data_missing_col = {"category": ["A", "B"]}
        df_missing = pd.DataFrame(data_missing_col)
        df_missing["category"] = df_missing["category"].astype("category")

        value_replacer_missing = ValueReplacer(valid_values={"category": ["A"], "new_col": [1, 2]},
                                               default_value="missing_val")
        transformed_df_missing = value_replacer_missing.transform(df_missing.copy())

        assert "new_col" in transformed_df_missing.columns
        assert all(transformed_df_missing["new_col"] == "missing_val")


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


class TestModelTrainerUtilities:
    @pytest.fixture
    def trainer_instance(self):
        # Create a dummy ModelTrainer instance for testing get_weights
        return ModelTrainer(
            customer_id=1,
            app_id=1,
            model_id="test_model",
            date=datetime.datetime.now(),
            features=Features(),  # Dummy Features instance
            model_config=ModelConfig(),  # Dummy ModelConfig instance
        )

    def test_get_weights_with_valid_data(self, trainer_instance):
        data = {"propensity": [0.5, 0.2, 0.8], "feature1": [1, 2, 3]}
        dataset = ray.data.from_pandas(pd.DataFrame(data))
        train_weights, valid_weights = trainer_instance.get_weights(dataset, dataset)

        expected_weights = [2.0, 5.0, 1.25]

        # Check train_weights
        train_weights_list = train_weights.to_pandas()["propensity"].tolist()
        np.testing.assert_allclose(train_weights_list, expected_weights)

        # Check valid_weights
        valid_weights_list = valid_weights.to_pandas()["propensity"].tolist()
        np.testing.assert_allclose(valid_weights_list, expected_weights)

    def test_get_weights_with_empty_dataset(self, trainer_instance):
        empty_dataset = ray.data.from_items([])
        train_weights, valid_weights = trainer_instance.get_weights(empty_dataset, empty_dataset)

        assert train_weights.count() == 0
        assert valid_weights.count() == 0

    def test_get_weights_without_propensity_column(self, trainer_instance):
        data = {"feature1": [1, 2, 3], "feature2": ["A", "B", "C"]}
        dataset = ray.data.from_pandas(pd.DataFrame(data))
        train_weights, valid_weights = trainer_instance.get_weights(dataset, dataset)

        assert train_weights.count() == 0
        assert valid_weights.count() == 0


class TestCastTypes:
    def test_cast_types_numpy(self):
        # Sample dictionary of numpy arrays
        batch = {
            "col1": np.array([1, 4, 7], dtype=object),
            "col2": np.array(["2", "5", "8"], dtype=object),
            "col3": np.array([3.0, 6.0, 9.0], dtype=object),
        }

        # Schema for casting (column names as keys)
        feature_schema = {"col1": "int32", "col2": "str", "col3": "float64"}

        # Cast types
        casted_batch = cast_types_numpy(batch.copy(), feature_schema)

        # Assertions
        assert isinstance(casted_batch["col1"][0], (int, np.integer))
        assert isinstance(casted_batch["col2"][0], (str, np.str_))
        assert isinstance(casted_batch["col3"][0], (float, np.floating))

        assert isinstance(casted_batch["col1"][1], (int, np.integer))
        assert isinstance(casted_batch["col2"][1], (str, np.str_))
        assert isinstance(casted_batch["col3"][1], (float, np.floating))

        assert isinstance(casted_batch["col1"][2], (int, np.integer))
        assert isinstance(casted_batch["col2"][2], (str, np.str_))
        assert isinstance(casted_batch["col3"][2], (float, np.floating))

        # Test with different dtypes
        batch_2 = {
            "colA": np.array(["10", "40"], dtype=object),
            "colB": np.array([20, 50], dtype=object),
            "colC": np.array([30.5, 60.5], dtype=object),
        }

        feature_schema_2 = {"colA": "str", "colB": "float32", "colC": "int64"}

        casted_batch_2 = cast_types_numpy(batch_2.copy(), feature_schema_2)

        assert isinstance(casted_batch_2["colA"][0], (str, np.str_))
        assert isinstance(casted_batch_2["colB"][0], (float, np.floating))
        assert isinstance(casted_batch_2["colC"][0], (int, np.integer))

        assert isinstance(casted_batch_2["colA"][1], (str, np.str_))
        assert isinstance(casted_batch_2["colB"][1], (float, np.floating))
        assert isinstance(casted_batch_2["colC"][1], (int, np.integer))


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

    def test_run_with_empty_training_data(self, model_object):
        empty_dataset = ray.data.from_items([])
        result, value_replacer, features = model_object.run(
            empty_dataset, target_column="totalAmount", use_validation_set=True
        )
        assert result is None
        assert value_replacer.valid_values == {}
        assert features == model_object.features

    def test_run_with_single_record_training_data(self, model_object):
        single_record_data = ray.data.from_items([{"user.country": "US", "totalAmount": 0.1}])
        result, value_replacer, features = model_object.run(
            single_record_data, target_column="totalAmount", use_validation_set=True
        )
        # When only one record, it should all go to training, and validation should be empty
        train_dataset, valid_dataset, test_dataset = model_object.prepare_data(single_record_data, "totalAmount")
        assert train_dataset.count() == 1
        assert valid_dataset.count() == 0
        assert test_dataset.count() == 0  # test_dataset is derived from valid_dataset, so it should also be empty

    def test_prepare_data_with_empty_dataset(self, model_object):
        empty_dataset = ray.data.from_items([])
        with pytest.raises(ValueError, match="Input dataset is empty. Cannot perform train-test split."):
            model_object.prepare_data(empty_dataset, "totalAmount")

    def test_dynamic_num_workers_edge_cases(self, model_object):
        # Set a smaller min_rows_per_worker for faster testing
        min_rows_per_worker_test = 10

        # Test with a very small dataset (less than min_rows_per_worker)
        small_data = [{"col1": 1}]
        small_dataset = ray.data.from_items(small_data)
        num_workers_small = model_object.dynamic_num_workers(
            small_dataset, min_rows_per_worker=min_rows_per_worker_test
        )
        assert num_workers_small == 1  # Should always be at least 1 worker

        # Test with a dataset slightly larger than min_rows_per_worker
        medium_data = [{"col1": i} for i in range(min_rows_per_worker_test + 1)]
        medium_dataset = ray.data.from_items(medium_data)
        num_workers_medium = model_object.dynamic_num_workers(
            medium_dataset, min_rows_per_worker=min_rows_per_worker_test
        )
        assert num_workers_medium == 1  # Should be 1 worker for (min_rows_per_worker_test + 1) rows

        # Test with a dataset large enough for multiple workers (e.g., 2x min_rows_per_worker)
        large_data = [{"col1": i} for i in range(min_rows_per_worker_test * 2)]
        large_dataset = ray.data.from_items(large_data)
        # Mock ray.cluster_resources() to control available CPUs for predictable testing
        original_cluster_resources = ray.cluster_resources
        ray.cluster_resources = lambda: {"CPU": 4}
        num_workers_large = model_object.dynamic_num_workers(
            large_dataset, min_rows_per_worker=min_rows_per_worker_test
        )
        assert num_workers_large == 2  # Should be 2 workers for 2x min_rows_per_worker_test rows with 4 CPUs
        ray.cluster_resources = original_cluster_resources  # Restore original

        # Test with a dataset that would result in more workers than available CPUs
        very_large_data = [{"col1": i} for i in range(min_rows_per_worker_test * 5)]
        very_large_dataset = ray.data.from_items(very_large_data)
        ray.cluster_resources = lambda: {"CPU": 4}
        num_workers_very_large = model_object.dynamic_num_workers(
            very_large_dataset, min_rows_per_worker=min_rows_per_worker_test
        )
        assert num_workers_very_large == 4  # Should be capped by available CPUs
        ray.cluster_resources = original_cluster_resources  # Restore original

    def test_get_weights_with_null_propensity(self, model_object):
        """Test that get_weights handles null propensity values correctly."""
        # Create test data with various propensity cases
        test_data = [
            {"propensity": 0.5, "feature1": 1},      # Valid -> weight = 1/0.5 = 2.0
            {"propensity": None, "feature1": 2},     # Null propensity -> weight = 1.0 (default)
            {"propensity": 1.2, "feature1": 3},      # Valid -> weight = 1/1.2 ≈ 0.833
            {"propensity": 1.0, "feature1": 4},      # Valid -> weight = 1/1.0 = 1.0
            {"propensity": 0.019444444444444445, "feature1": 5}  # Valid -> weight = 1/0.019... ≈ 51.43
        ]
        
        # Create train and valid datasets
        train_dataset = ray.data.from_items(test_data)
        valid_dataset = ray.data.from_items(test_data[:3])  # Just first 3 rows for validation
        
        # Get weights using the model's get_weights method
        train_weights, valid_weights = model_object.get_weights(train_dataset, valid_dataset)
        
        # Verify that weights were calculated
        assert train_weights.count() == 5
        assert valid_weights.count() == 3
        
        # Check the calculated weights
        train_weights_list = train_weights.to_pandas()["propensity"].tolist()
        valid_weights_list = valid_weights.to_pandas()["propensity"].tolist()
        
        # Expected weights based on simplified logic (only null gets default weight 1.0)
        expected_train_weights = [
            2.0,  # 1/0.5
            1.0,  # null -> default
            1/1.2,  # 1/1.2 ≈ 0.833
            1.0,  # 1/1.0
            1/0.019444444444444445  # ≈ 51.43
        ]
        expected_valid_weights = [
            2.0,  # 1/0.5
            1.0,  # null -> default  
            1/1.2  # 1/1.2 ≈ 0.833
        ]
        
        # Assert weights are positive
        assert all(w > 0 for w in train_weights_list), "All weights should be positive"
        assert all(w > 0 for w in valid_weights_list), "All validation weights should be positive"
        
        # Check specific values (with tolerance for floating point)
        assert abs(train_weights_list[0] - 2.0) < 0.001
        assert abs(train_weights_list[1] - 1.0) < 0.001  # null propensity -> default weight 1.0
        assert abs(train_weights_list[2] - (1/1.2)) < 0.001
        assert abs(train_weights_list[3] - 1.0) < 0.001
        assert abs(train_weights_list[4] - (1/0.019444444444444445)) < 0.1
        
        assert abs(valid_weights_list[0] - 2.0) < 0.001
        assert abs(valid_weights_list[1] - 1.0) < 0.001  # null propensity -> default weight 1.0
        assert abs(valid_weights_list[2] - (1/1.2)) < 0.001
