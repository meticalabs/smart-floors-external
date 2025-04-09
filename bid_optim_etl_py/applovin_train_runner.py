import datetime
from dataclasses import dataclass
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd
import ray
import xgboost
from pyiceberg.catalog import load_catalog
from pyiceberg.expressions import EqualTo
from ray.data import Dataset
from ray.train import CheckpointConfig, RunConfig, ScalingConfig, Result
from ray.train.xgboost import RayTrainReportCallback
from ray.train.xgboost import XGBoostTrainer
from xgboost import DMatrix


@dataclass
class ValueReplacer:
    valid_values: dict
    default_value: any

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df.update({
            column: [None] * len(df) if column not in df.columns else df[column].apply(
                lambda x: x if x is None or x in valid_vals else self.default_value)
            for column, valid_vals in self.valid_values.items()
        })
        return df


def sanitise_path(path):
    return path.strip("/")


class Schema:
    REQUEST_ID: str = "requestId"
    USER_ID: str = "userId"
    TOTAL_AMOUNT: str = "totalAmount"
    EVENT_TIME: str = "eventTime"
    CONTEXT: str = "context"
    IS_FILLED: str = "isFilled"
    CPM_FLOOR_AD_UNIT_ID: str = "cpmFloorAdUnitId"
    CPM_FLOOR_VALUE: str = "cpmFloorValue"
    DATE = "date"
    LAST_UPDATE_TIME = "lastUpdateTime"
    CUSTOMER_ID = "customerId"
    APP_ID = "appId"
    MODEL_ID = "modelId"


@dataclass(frozen=True)
class Field:
    name: str
    dtype: str
    feature_column: bool = True
    target_column: bool = False


@dataclass
class Features:
    fields: list[Field]

    def fields_sorted(self):
        return sorted(filter(lambda x: isinstance(x, Field), self.fields), key=lambda f: f.name)

    def fields_to_dmatrix_from_series(self, series: pd.Series, prediction_phase: bool = False) -> DMatrix:
        df = self.assemble_fields_from_series(series, prediction_phase)
        dmatrix = DMatrix(df, enable_categorical=True)
        return dmatrix

    def fields_to_dmatrix_from_df(self, df: pd.DataFrame, prediction_phase: bool = False) -> DMatrix:
        df = self.assemble_fields_from_df(df, prediction_phase)
        dmatrix = DMatrix(df, enable_categorical=True)
        return dmatrix

    def assemble_fields_from_series(self, series: pd.Series, prediction_phase: bool = False) -> pd.DataFrame:
        return self.assemble_fields_from_df(pd.DataFrame([series] if series is not None and len(series) != 0 else {}),
                                            prediction_phase)

    def assemble_fields_from_df(self, df: pd.DataFrame, prediction_phase: bool = False) -> pd.DataFrame:
        if df is None:
            df = pd.DataFrame()

        for field in self.fields_sorted():
            if field.target_column and prediction_phase:
                continue
            if df.empty or field.name not in df.columns:
                df[field.name] = [np.nan]

            if field.dtype == "category":
                df[field.name] = df[field.name].replace({None: np.nan}).astype(field.dtype)

        if prediction_phase:
            ordered_features = [f.name for f in self.fields_sorted() if f.feature_column]
        else:
            ordered_features = [f.name for f in self.fields_sorted()]

        return df[ordered_features]

    def assemble_fields_from_ds(self, ds: ray.data.Dataset) -> ray.data.Dataset:
        ordered_features = [f.name for f in self.fields_sorted()]
        return ds.map_batches(
            lambda df: self.assemble_fields_from_df(df),
            batch_format="pandas",
        ).select_columns(ordered_features)


def init_ray_cluster():
    ray.init()


@dataclass
class ModelTrainer:
    customer_id: int
    app_id: int
    model_id: str
    date: datetime.datetime
    s3_checkpoint_path: Optional[str] = None

    def __post_init__(self):
        self.features = Features([
            Field(name="user.country", dtype="category"),
            Field(name="user.languageCode", dtype="category"),
            Field(name="user.deviceType", dtype="category"),
            Field(name="user.osVersion", dtype="category"),
            Field(name="user.deviceModel", dtype="category"),
            Field(name="assignmentDayOfWeek", dtype="int"),
            Field(name="assignmentHourOfDay", dtype="int"),
            Field(name="user.minRevenueLast24Hours", dtype="float"),
            Field(name="user.avgRevenueLast24Hours", dtype="float"),
            Field(name="user.avgRevenueLast48Hours", dtype="float"),
            Field(name="user.avgRevenueLast72Hours", dtype="float"),
            Field(name="user.mostRecentAdSource", dtype="category"),
            Field(name="user.mostRecentAdRevenue", dtype="float"),
            Field(name="highestBidFloorValue", dtype="float"),
            Field(name="mediumBidFloorValue", dtype="float"),
            Field(name="totalAmount", dtype="float", target_column=True, feature_column=False)
        ])

        self.weight_column = "propensities"

    def _run_config(self, storage_path: Optional[str] = None) -> RunConfig:
        return RunConfig(
            name=f"applovin_train_{self.app_id}_{self.model_id}",
            checkpoint_config=CheckpointConfig(
                # Checkpoint every 10 iterations.
                checkpoint_frequency=10,
                # Only keep the latest checkpoint and delete the others.
                num_to_keep=1,
            ),
            storage_path=storage_path or f"~/ray_results_{self.customer_id}_{self.app_id}_{self.model_id}",
        )

    def prepare_data(self, input_data: ray.data.Dataset, target_column: str) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and split the dataset into train, validation, and test sets."""
        train_dataset, valid_dataset = input_data.train_test_split(test_size=0.3)
        test_dataset = valid_dataset.drop_columns([target_column])
        return train_dataset, valid_dataset, test_dataset

    def preprocess_data(self, train_dataset: ray.data.Dataset, valid_dataset: Optional[ray.data.Dataset] = None,
                        test_dataset: Optional[ray.data.Dataset] = None) -> Tuple[
        ray.data.Dataset, ray.data.Dataset, ray.data.Dataset]:
        """Preprocess the dataset by applying feature engineering and transformations."""
        return (
            self.features.assemble_fields_from_ds(train_dataset),
            self.features.assemble_fields_from_ds(valid_dataset) if valid_dataset else ray.data.from_items([]),
            self.features.assemble_fields_from_ds(test_dataset) if test_dataset else ray.data.from_items([]),
        )

    def save_model(self, result: Result, model_save_path: str):
        """Save the trained model to the specified path."""
        checkpoint = result.checkpoint
        # Save the model as .xgb file
        booster = RayTrainReportCallback.get_model(checkpoint)
        booster.save_model(model_save_path)

    def value_replacer_based_on_impressions(self, dataset: ray.data.Dataset, columns: list[str],
                                            min_impressions: int = 10000,
                                            default_category: str = "other") -> ValueReplacer:
        """
        Replaces values in specified columns of a dataset based on the minimum number of
        impressions, ensuring that only frequently occurring values in the dataset are
        retained, while infrequent values are replaced with a default category.

        The function takes a dataset, a list of columns to analyze, a threshold for
        minimum impressions, and a default category to replace infrequent values. The
        processed data is encapsulated within a ValueReplacer instance which contains
        validated values for replacement.

        :param dataset: ray.data.Dataset
            The input dataset to analyze.
        :param columns: list[str]
            A list of column names from the dataset to be processed.
        :param min_impressions: int, optional
            The minimum number of impressions that a value must meet to be retained.
            Defaults to 10000.
        :param default_category: str, optional
            The category used to replace any values not meeting the minimum impressions
            threshold or those from columns not found in the dataset. Defaults to "other".
        :return: ValueReplacer
            An object that contains the validated values for the specified columns
            based on the minimum impressions and the default replacement category.
        """
        features_with_min_impressions = {}
        for col in columns:
            if col not in dataset.columns():
                continue
            ds = dataset.groupby(col).count().rename_columns({"count()": "count"}).filter(
                lambda x: x["count"] >= min_impressions)
            if ds.count() == 0:
                continue
            values_with_min_impressions = ds.select_columns([col]).to_pandas()[col].tolist()
            features_with_min_impressions[col] = sorted(list(set(values_with_min_impressions)))

        return ValueReplacer(
            valid_values=features_with_min_impressions,
            default_value=default_category
        )

    def _train_fn_per_worker(self, config: dict):
        """
        Trains an XGBoost model on worker-specific data shards and evaluates it on validation data if available.

        Detailed Steps:
        - Extracts training and evaluation datasets from Ray Train's dataset shards and converts them to pandas
        DataFrames.
        - Ensures compatibility of data, handling edge cases such as empty datasets or Series format, and prepares
          the data for XGBoost training and validation.
        - Configures relevant training parameters, including model objective, tree method, learning rate, etc.
        - Handles empty training datasets gracefully by creating a simulated dataset with no rows.
        - Initiates the model training process using XGBoost. Evaluation datasets are passed if available.

        :param config: A dictionary containing training configurations. Must include:
                       - "target_column" (str): The target column in the training data.
                       - "feature_columns" (List[Field]): Metadata for feature columns, including name and dtype.
                       - "num_boost_round" (int): Number of boosting rounds for the training.
        :type config: dict
        :return: None
        """
        # Get dataset shards
        train_ds = ray.train.get_dataset_shard("train").materialize().to_pandas()
        eval_ds = ray.train.get_dataset_shard("valid").materialize().to_pandas()
        train_weights_ds = ray.train.get_dataset_shard("train_weights").materialize().to_pandas()
        eval_weights_ds = ray.train.get_dataset_shard("valid_weights").materialize().to_pandas()

        # Ensure dataframes are valid
        train_df = pd.DataFrame([train_ds]) if isinstance(train_ds, pd.Series) else train_ds
        eval_df = pd.DataFrame([eval_ds]) if isinstance(eval_ds, pd.Series) else eval_ds
        train_weights_df = pd.DataFrame([train_weights_ds]) if isinstance(train_weights_ds,
                                                                          pd.Series) else train_weights_ds
        eval_weights_df = pd.DataFrame([eval_weights_ds]) if isinstance(eval_weights_ds, pd.Series) else eval_weights_ds

        target_column = config["target_column"]
        num_boost_round = config["num_boost_round"]

        # Handle empty training data
        if train_df.empty:
            features = config["feature_columns"]
            # Create a dummy DataFrame with the same columns and dtypes as the features with all np.nan
            dummy_X = pd.DataFrame(columns=[f.name for f in features]).astype({f.name: f.dtype for f in features})
            # Fill the dummy DataFrame with NaN values
            for col in dummy_X.columns:
                dummy_X[col] = np.nan
            dtrain = xgboost.DMatrix(dummy_X.drop(columns=[target_column]), label=pd.Series([], dtype=float),
                                     enable_categorical=True)
            deval = None
        else:
            train_X, train_y = train_df.drop(columns=[target_column]), train_df[target_column]
            eval_X, eval_y = (eval_df.drop(columns=[target_column]), eval_df[target_column]) if not eval_df.empty else (
                None, None)
            train_weights_df = None if train_weights_df.empty else train_weights_df
            eval_weights_df = None if eval_weights_df.empty else eval_weights_df
            dtrain = xgboost.DMatrix(train_X, label=train_y, enable_categorical=True, weight=train_weights_df)
            deval = (xgboost.DMatrix(eval_X, label=eval_y, enable_categorical=True,
                                     weight=eval_weights_df) if eval_X is not None else None)

        # Training parameters
        params = {
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42,
            "eval_metric": ["rmse"],
            "missing": np.nan
        }

        # Train the model
        xgboost.train(
            params,
            dtrain=dtrain,
            evals=[(deval, "validation")] if deval else None,
            num_boost_round=num_boost_round,
            callbacks=[RayTrainReportCallback()],
        )

    def get_weights(self, train_dataset: ray.data.Dataset, valid_dataset: ray.data.Dataset):
        """
        Retrieves the weights for the training and validation datasets. If the weight column
        is not present in the datasets, it returns empty datasets.
        """
        is_train_data_empty = train_dataset is None or train_dataset.limit(1).count() == 0
        is_valid_data_empty = valid_dataset is None or valid_dataset.limit(1).count() == 0

        train_weights = train_dataset.select_columns([
            self.weight_column]) if not is_train_data_empty and self.weight_column in train_dataset.columns() \
            else ray.data.from_items([])
        valid_weights = (valid_dataset.select_columns([
            self.weight_column]) if not is_valid_data_empty and self.weight_column in valid_dataset.columns()
                         else ray.data.from_items([]))

        # Perform 1 / self.weight_column to get the inverse of the weights
        def inverse_propensity(df: pd.DataFrame) -> pd.DataFrame:
            df[self.weight_column] = 1 / df[self.weight_column]
            return df

        if train_weights.limit(1).count() > 0:
            train_weights = train_weights.map_batches(inverse_propensity, batch_format="pandas")
        if valid_weights.limit(1).count() > 0:
            valid_weights = valid_weights.map_batches(inverse_propensity, batch_format="pandas")

        return train_weights, valid_weights

    def dynamic_num_workers(self, train_dataset: ray.data.Dataset, min_rows_per_worker: int = 1000) -> int:
        """
        Dynamically calculates the number of workers based on the number of rows in the dataset
        :param train_dataset: The training dataset
        :param min_rows_per_worker: Minimum number of rows per worker
        :return: Number of workers to be used
        """
        num_rows = train_dataset.count()
        available_cores = ray.cluster_resources().get("CPU", 1)
        num_workers = min(int(num_rows / min_rows_per_worker), int(available_cores))
        num_workers = max(1, num_workers)  # Ensure at least 1 worker
        return num_workers

    def run(self, assignments_with_ad_revenue: ray.data.Dataset, target_column: str,
            *, cross_validation: bool = False, num_workers: int = 4) -> Tuple[Result, ValueReplacer, Features]:
        """
        Executes the process of training an XGBoost model on the given dataset with
        options for cross-validation and preprocessing. The function handles data
        transformation, preprocessing, splitting, and configuration of the trainer.

        :param assignments_with_ad_revenue: The input dataset to be processed and
            trained on. It is an instance of the `ray.data.Dataset` which contains
            assignment data combined with associated ad revenue information.
        :param target_column: Specifies the column in the dataset that serves as the
            target variable for prediction.
        :param cross_validation: A boolean flag that indicates whether cross-validation
            should be applied during dataset preparation. Defaults to False. If True,
            the dataset will be split into training and validation subsets.

        :return: A tuple containing the following:
            - `Result`: The result object from the XGBoost training process, which
              contains training metrics and other output information.
            - `ValueReplacer`: An instance of the value replacer used for handling
              missing or infrequent categorical values in the dataset.
            - `Features`: An instance representing the feature definitions used for
              training the model.
        """
        # Configure checkpointing
        run_config = self._run_config(storage_path=self.s3_checkpoint_path)

        # Create a value replacer for categorical columns
        value_replacer = self.value_replacer_based_on_impressions(
            assignments_with_ad_revenue,
            columns=[f.name for f in self.features.fields_sorted() if f.dtype == "category"],
            min_impressions=10000,
            default_category="other"
        )

        # Transform the dataset
        transformed_ds = assignments_with_ad_revenue.map_batches(value_replacer.transform, batch_format="pandas")

        # Split the dataset
        train_dataset, valid_dataset, _ = self.prepare_data(transformed_ds, target_column) if cross_validation else (
            transformed_ds, ray.data.from_items([]), None)

        # Get the propensities for each record
        train_weights, valid_weights = self.get_weights(train_dataset, valid_dataset)

        # Dynamic worker calculation
        num_workers = self.dynamic_num_workers(train_dataset, min_rows_per_worker=1000)

        # Preprocess the dataset
        preprocessed_train_dataset, preprocessed_valid_dataset, _ = self.preprocess_data(train_dataset, valid_dataset)

        # Prepare datasets for training
        datasets = {"train": preprocessed_train_dataset, "train_weights": train_weights}
        if valid_dataset:
            datasets["valid"] = preprocessed_valid_dataset
            datasets["valid_weights"] = valid_weights

        # Configure and run the XGBoost trainer
        trainer = XGBoostTrainer(
            train_loop_per_worker=self._train_fn_per_worker,
            train_loop_config={
                "target_column": target_column,
                "num_boost_round": 100,
                "feature_columns": self.features.fields_sorted(),
            },
            scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=False),
            datasets=datasets,
            run_config=run_config,
        )
        result = trainer.fit()
        return result, value_replacer, self.features


def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Run the applovin bid floor training')
    parser.add_argument('--customerId', type=int, help='Customer ID')
    parser.add_argument('--appId', type=int, help='App ID')
    parser.add_argument('--modelId', type=str, help='Model ID')
    parser.add_argument('--icebergTrainDataTable', help='Iceberg db table name for training data')
    return parser.parse_args()


def read_training_data(iceberg_train_data: str, customer_id: int, app_id: int, model_id: str,
                       num_blocks: Optional[int] = None) -> ray.data.Dataset:
    catalog = load_catalog(name="default", type="glue")
    table = catalog.load_table(iceberg_train_data)
    table_data = (
        table.scan(row_filter=EqualTo(Schema.CUSTOMER_ID, customer_id) &
                              EqualTo(Schema.APP_ID, app_id) &
                              EqualTo(Schema.MODEL_ID, model_id)
                   ).to_ray()
    )
    if num_blocks:
        return table_data.repartition(num_blocks)
    return table_data


def run():
    args = arg_parser()
    init_ray_cluster()
    training_data = read_training_data(args.icebergTrainDataTable, args.customerId, args.appId, args.modelId)
    trainer = ModelTrainer(customer_id=1, app_id=1, model_id="test_model", date=datetime.datetime.now())
    result, value_replacer, features = trainer.run(
        assignments_with_ad_revenue=training_data,
        target_column=Schema.TOTAL_AMOUNT,
        cross_validation=True
    )
    print(result.metrics)


if __name__ == '__main__':
    run()
