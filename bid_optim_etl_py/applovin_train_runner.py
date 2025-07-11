import dataclasses
import datetime
import itertools
import logging
import os
import sys
from typing import Tuple, Optional

import boto3
import joblib
import numpy as np
import pandas as pd
import ray
import xgboost
from etl_py_commons.job_initialiser import Initialisation
from pydantic import ConfigDict, BaseModel, SkipValidation
from pydantic.dataclasses import dataclass
from pyiceberg.catalog import load_catalog
from pyiceberg.expressions import EqualTo, LessThanOrEqual, GreaterThanOrEqual
from ray.data import Dataset
from ray.train import CheckpointConfig, RunConfig, ScalingConfig, Result
from ray.train.xgboost import RayTrainReportCallback
from ray.train.xgboost import XGBoostTrainer
from xgboost import DMatrix

from bid_optim_etl_py.cfg_parser import ConfigFile
from bid_optim_etl_py.command_line_args import ApplovinModelTrainingArgsParser
from bid_optim_etl_py.cw_publisher import CloudWatchAlerts
from bid_optim_etl_py.utils.management_api import BidFloorManagementAPI, ETLConfig, HttpClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)


@dataclass
class ValueReplacer:
    valid_values: dict
    default_value: str | int | float

    def transform_series(self, series: pd.Series) -> pd.Series:
        for column, valid_vals in self.valid_values.items():
            if column not in series.index or (series[column] not in valid_vals):
                series[column] = self.default_value
        return series

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for column, valid_vals in self.valid_values.items():
            if column in df.columns:
                # Ensure categorical dtype can hold default_value if it's not already there
                if isinstance(df[column].dtype, pd.CategoricalDtype):
                    if self.default_value not in df[column].cat.categories:
                        df[column] = df[column].cat.add_categories([self.default_value])

                # Replace values not in valid_vals with default_value
                df.loc[~df[column].isin(valid_vals), column] = self.default_value
            else:
                # If column doesn't exist, create it with default_value
                df[column] = self.default_value
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


@dataclass
class Field:
    name: str
    dtype: str
    feature_column: bool = True
    target_column: bool = False


@dataclass
class Features:
    fields: list[Field] = dataclasses.field(default_factory=list)

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
        return self.assemble_fields_from_df(
            pd.DataFrame([series] if series is not None and len(series) != 0 else {}), prediction_phase
        )

    def assemble_fields_from_df(self, df: pd.DataFrame, prediction_phase: bool = False) -> pd.DataFrame:
        if df is None:
            df = pd.DataFrame()

        for field in self.fields_sorted():
            if field.target_column and prediction_phase:
                continue
            if df.empty or field.name not in df.columns:
                df[field.name] = [np.nan] * max(len(df), 1)

            if field.dtype == "category":
                df[field.name] = df[field.name].replace({None: np.nan}).astype(field.dtype)
            else:
                df[field.name] = df[field.name].astype(field.dtype)

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
class ModelFeatures:
    etl_config: ETLConfig

    def as_ray_schema(self):
        mandatory_context_fields = [
            Field(name="assignmentDayOfWeek", dtype="Int64"),
            Field(name="assignmentHourOfDay", dtype="Int64"),
            Field(name="highestBidFloorValue", dtype="float32"),
            Field(name="mediumBidFloorValue", dtype="float32"),
            Field(name="totalAmount", dtype="float32", target_column=True, feature_column=False),
        ]

        features = Features(
            [
                Field(name=context.path, dtype=("category" if context.dataType.lower() == "string" else "float32"))
                for context in self.etl_config.context
            ]
            + mandatory_context_fields
        )
        return features


@dataclass
class ModelConfig:
    num_boost_rounds: int = 100
    tree_method: str = "hist"
    objective: str = "reg:squarederror"
    learning_rate: float = 0.1
    max_depth: int = 4
    eval_metric: list[str] = dataclasses.field(default_factory=lambda: ["rmse", "mae"])


@dataclass
class ModelTrainer:
    customer_id: int
    app_id: int
    model_id: str
    date: datetime.datetime
    features: Features
    model_config: ModelConfig
    s3_checkpoint_path: Optional[str] = None
    num_boost_round: int = 100

    def __post_init__(self):
        self.weight_column = "propensity"

    def _run_config(self, storage_path: Optional[str] = None) -> RunConfig:
        batch_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        return RunConfig(
            name=f"applovin_train_{self.app_id}_{self.model_id}",
            checkpoint_config=CheckpointConfig(
                # Checkpoint every 10 iterations.
                checkpoint_frequency=10,
                # Only keep the latest checkpoint and delete the others.
                num_to_keep=1,
            ),
            storage_path=storage_path or f"~/ray_results_{self.customer_id}_{self.app_id}_{self.model_id}_{batch_time}",
        )

    def prepare_data(self, input_data: ray.data.Dataset, target_column: str) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and split the dataset into train, validation, and test sets."""
        if input_data.count() == 0:
            raise ValueError("Input dataset is empty. Cannot perform train-test split.")
        elif input_data.count() == 1:
            train_dataset = input_data
            valid_dataset = input_data.limit(0)  # Empty dataset
        else:
            train_dataset, valid_dataset = input_data.train_test_split(test_size=0.3)
        test_dataset = valid_dataset.drop_columns([target_column])
        return train_dataset, valid_dataset, test_dataset

    def preprocess_data(
        self,
        train_dataset: ray.data.Dataset,
        valid_dataset: Optional[ray.data.Dataset] = None,
        test_dataset: Optional[ray.data.Dataset] = None,
    ) -> Tuple[ray.data.Dataset, ray.data.Dataset, ray.data.Dataset]:
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

    def value_replacer_based_on_impressions(
        self,
        dataset: ray.data.Dataset,
        columns: list[str],
        min_impressions: int = 10000,
        default_category: str = "other",
    ) -> ValueReplacer:
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
        if dataset.count() == 0:
            return ValueReplacer(valid_values={}, default_value=default_category)

        # Create a dataset of (column_name, value) pairs for all relevant columns
        # Replace None to "metica-None" str and later replace it to None
        metica_none = "metica-None"

        long_ds = dataset.map_batches(
            lambda df: (
                df[[col for col in columns if col in df.columns]]
                .apply(
                    lambda series: (
                        series.cat.add_categories([metica_none])
                        if isinstance(series.dtype, pd.CategoricalDtype) and metica_none not in series.cat.categories
                        else series
                    )
                )
                .fillna(metica_none)
                .melt(var_name="column_name", value_name="value")
            ),
            batch_format="pandas",
        )

        # Group by column_name and value, then count
        counts_ds = long_ds.groupby(["column_name", "value"]).count()

        # Filter based on min_impressions
        filtered_counts_ds = counts_ds.filter(lambda x: x["count()"] >= min_impressions)

        # Collect the results
        all_counts = filtered_counts_ds.to_pandas()

        if not all_counts.empty:
            for col in columns:
                # Filter for the current column
                col_values = [
                    None if value == metica_none else value
                    for value in all_counts[all_counts["column_name"] == col]["value"].tolist()
                ]
                if col_values:
                    features_with_min_impressions[col] = sorted(list(set(col_values)), key=lambda x: (x is None, x))

        return ValueReplacer(valid_values=features_with_min_impressions, default_value=default_category)

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
                       - "model_config" (ModelConfig): Configuration for the XGBoost model, including parameters like
        :type config: dict
        :return: None
        """
        # Get dataset shards
        train_ds = ray.train.get_dataset_shard("train").materialize().to_pandas()
        eval_ds = ray.train.get_dataset_shard("valid").materialize().to_pandas()

        train_ds = cast_types(train_ds, {f.name: f.dtype for f in config["feature_columns"]})
        eval_ds = cast_types(eval_ds, {f.name: f.dtype for f in config["feature_columns"]})

        train_weights_ds = ray.train.get_dataset_shard("train_weights").materialize().to_pandas()
        eval_weights_ds = ray.train.get_dataset_shard("valid_weights").materialize().to_pandas()

        # Convert to DataFrames if necessary
        train_df = pd.DataFrame([train_ds]) if isinstance(train_ds, pd.Series) else train_ds
        eval_df = pd.DataFrame([eval_ds]) if isinstance(eval_ds, pd.Series) else eval_ds
        train_weights_df = None if train_weights_ds.empty else train_weights_ds
        eval_weights_df = None if eval_weights_ds.empty else eval_weights_ds

        # Extract target and features
        target_column = config["target_column"]
        train_X, train_y = train_df.drop(columns=[target_column]), train_df[target_column]
        eval_X, eval_y = (
            (eval_df.drop(columns=[target_column]), eval_df[target_column]) if not eval_df.empty else (None, None)
        )

        # Create DMatrix for XGBoost
        dtrain = xgboost.DMatrix(train_X, label=train_y, enable_categorical=True, weight=train_weights_df)
        deval = (
            xgboost.DMatrix(eval_X, label=eval_y, enable_categorical=True, weight=eval_weights_df)
            if eval_X is not None
            else None
        )

        model_config = config["model_config"]

        # Training parameters
        params = {
            "tree_method": model_config.tree_method,
            "objective": model_config.objective,
            "learning_rate": model_config.learning_rate,
            "max_depth": model_config.max_depth,
            "eval_metric": model_config.eval_metric,
        }

        # Train the model
        xgboost.train(
            params,
            dtrain=dtrain,
            evals=[(dtrain, "train")] + [(deval, "validation")] if deval else None,
            num_boost_round=model_config.num_boost_rounds,
            callbacks=[RayTrainReportCallback()],
        )

    def get_weights(self, train_dataset: ray.data.Dataset, valid_dataset: ray.data.Dataset):
        """
        Retrieves the weights for the training and validation datasets. If the weight column
        is not present in the datasets, it returns empty datasets.
        """
        is_train_data_empty = train_dataset is None or train_dataset.limit(1).count() == 0
        is_valid_data_empty = valid_dataset is None or valid_dataset.limit(1).count() == 0

        train_weights = (
            train_dataset.select_columns([self.weight_column])
            if not is_train_data_empty and self.weight_column in train_dataset.columns()
            else ray.data.from_items([])
        )
        valid_weights = (
            valid_dataset.select_columns([self.weight_column])
            if not is_valid_data_empty and self.weight_column in valid_dataset.columns()
            else ray.data.from_items([])
        )

        # Perform 1 / self.weight_column to get the inverse of the weights
        if train_weights.count() > 0:
            train_weights = train_weights.map(lambda x: {self.weight_column: 1 / x[self.weight_column]})
        if valid_weights.count() > 0:
            valid_weights = valid_weights.map(lambda x: {self.weight_column: 1 / x[self.weight_column]})

        return train_weights, valid_weights

    def dynamic_num_workers(self, train_dataset: ray.data.Dataset, min_rows_per_worker: int = 10_000_000) -> int:
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

    def update_array_fields_to_list(self, batch: dict) -> dict:
        # This function converts numpy arrays to lists for specific fields in the batch
        # as arrow conversion fails for numpy arrays with dtype=object
        if "cpmFloorAdUnitIds" in batch:
            batch["cpmFloorAdUnitIds"] = np.array([arr.tolist() for arr in batch["cpmFloorAdUnitIds"]], dtype=object)
        return batch

    def run(
        self, assignments_with_ad_revenue: ray.data.Dataset, target_column: str, *, use_validation_set: bool = False
    ) -> Tuple[Result, ValueReplacer, Features]:
        """
        Executes the process of training an XGBoost model on the given dataset with
        options for cross-validation and preprocessing. The function handles data
        transformation, preprocessing, splitting, and configuration of the trainer.

        :param assignments_with_ad_revenue: The input dataset to be processed and
            trained on. It is an instance of the `ray.data.Dataset` which contains
            assignment data combined with associated ad revenue information.
        :param target_column: Specifies the column in the dataset that serves as the
            target variable for prediction.
        :param use_validation_set: A boolean flag that indicates whether to use a
            validation set for model evaluation during training. If set to True, the
            dataset will be split into training and validation sets.

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
            default_category="other",
        )

        # If the dataset is empty, return early
        if assignments_with_ad_revenue.count() == 0:
            return None, value_replacer, self.features

        # Transform the dataset
        transformed_ds = assignments_with_ad_revenue.map_batches(value_replacer.transform, batch_format="pandas")

        # Split the dataset
        train_dataset, valid_dataset, _ = (
            self.prepare_data(transformed_ds, target_column)
            if use_validation_set
            else (transformed_ds, ray.data.from_items([]), None)
        )

        # Get the propensities for each record
        train_weights, valid_weights = self.get_weights(train_dataset, valid_dataset)

        # Dynamic worker calculation
        num_workers = self.dynamic_num_workers(train_dataset, min_rows_per_worker=10_000_000)

        logging.info(f"Number of workers: {num_workers}")

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
                "feature_columns": self.features.fields_sorted(),
                "model_config": self.model_config,
            },
            scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=False),
            datasets=datasets,
            run_config=run_config,
        )
        result = trainer.fit()
        return result, value_replacer, self.features


def arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run the applovin bid floor training")
    parser.add_argument("--region", type=str, help="AWS region name, e.g., us-east-1", required=True)
    parser.add_argument("--customerId", type=int, help="Customer ID")
    parser.add_argument("--appId", type=int, help="App ID")
    parser.add_argument("--modelId", type=str, help="Model ID")
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    parser.add_argument("--icebergTrainDataTable", help="Iceberg db table name for training data")
    parser.add_argument("--s3ModelArtifactBucket", help="S3 bucket name for model artifact")
    return parser.parse_args()


def cast_types(batch: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """
    Casts the types of the columns in the batch DataFrame according to the provided schema.
    """
    for column, dtype in schema.items():
        if column in batch.columns:
            batch[column] = batch[column].astype(dtype)
    return batch


def cast_types_numpy(batch, feature_schema):
    """
    Function to cast a batch of data (NumPy array) to the specified schema types.
    `batch` is a dictionary of NumPy arrays, where keys are column names.
    `feature_schema` is a dictionary where keys are feature names and values are dtypes.
    """
    for feature_name, feature_type in feature_schema.items():
        if feature_name in batch:
            batch[feature_name] = batch[feature_name].astype(feature_type)
    return batch


def read_training_data(
    region: str,
    iceberg_train_data: str,
    customer_id: int,
    app_id: int,
    model_id: str,
    date: datetime.date,
    lookback_window_in_days: int,
    num_blocks: Optional[int] = None,
) -> ray.data.Dataset:
    cw_alert = CloudWatchAlerts(region=region)
    try:
        catalog = load_catalog(name="default", **{"type": "glue", "client.region": region})
        table = catalog.load_table(iceberg_train_data)
        schema = table.schema()
        all_columns = [field.name for field in schema.fields]
        columns_to_exclude = ["estimates", "cpmFloorAdUnitIds"]
        selected_columns = [col for col in all_columns if col not in columns_to_exclude]
        row_filter = (
            EqualTo(Schema.CUSTOMER_ID, customer_id)
            & EqualTo(Schema.APP_ID, app_id)
            & EqualTo(Schema.MODEL_ID, model_id)
            & LessThanOrEqual(Schema.DATE, date.strftime("%Y-%m-%d"))
            & GreaterThanOrEqual(
                Schema.DATE, (date - datetime.timedelta(days=lookback_window_in_days)).strftime("%Y-%m-%d")
            )
        )
        table_data = ray.data.read_iceberg(
            table_identifier=iceberg_train_data,
            catalog_kwargs={"name": "default", "type": "glue", "client.region": region},
            row_filter=row_filter,
            selected_fields=tuple(selected_columns),
            override_num_blocks=num_blocks,
        )
    except Exception:
        cw_alert.cw_wrapper.put_metric_data(
            namespace="SmartBidPipeline",
            name="TrainingDataReadError",
            value=1,
            unit="Count",
            dimensions={
                "Job": os.path.basename(__file__),
                "CustomerId": str(customer_id),
                "AppId": str(app_id),
                "ModelId": model_id,
                "SnapshotDate": date,
            },
        )

        logging.exception("Error reading training data from Iceberg table")
        return ray.data.from_items([])
    return table_data


class Predictor(BaseModel):
    epsilon: float
    rng_exploration: SkipValidation[np.random.Generator] = dataclasses.field(
        default_factory=lambda: np.random.default_rng()
    )
    rng_shuffle: SkipValidation[np.random.Generator] = dataclasses.field(
        default_factory=lambda: np.random.default_rng()
    )
    clf: SkipValidation[xgboost.Booster] | None = None
    value_replacer: ValueReplacer | None = None
    features: Features | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def sort_by_name_postfix_desc(self, assignments: list[dict]) -> list[dict]:
        """
        Sorts a list of assignments in descending order based on the numerical postfix
        in their "name" field. The postfix is determined by splitting the "name" field
        on underscore ('_') and converting the last part into an integer. The sorting
        order prioritizes assignments with higher numerical postfix values.

        :param assignments: A list of dictionaries, each representing an assignment.
        :return: A sorted list of assignments in descending order based on the numerical postfix.
        """
        return sorted(assignments, key=lambda x: int(x["name"].split("_")[-1]), reverse=True)

    def add_hardcoded_contexts(
        self,
        context: pd.Series,
        highest_bid_floor_value: Optional[float],
        medium_bid_floor_value: Optional[float],
    ) -> pd.Series:
        """
        Adds hardcoded context values to the provided context series.
        :param context: The context series to be modified.
        :param highest_bid_floor_value: The highest bid floor value to add to context.
        :param medium_bid_floor_value: The medium bid floor value to add to context.
        :return: The modified context series with hardcoded values added.
        """
        nw = datetime.datetime.now(datetime.timezone.utc)
        context["assignmentDayOfWeek"] = nw.weekday()  # 0-6, Monday-Sunday
        context["assignmentHourOfDay"] = nw.hour
        context["highestBidFloorValue"] = highest_bid_floor_value
        context["mediumBidFloorValue"] = medium_bid_floor_value
        return context

    def split_based_on_name(self, ad_unit_list: list[dict]) -> Tuple[list[dict], dict]:
        """
        Splits the ad unit list into two parts: one with the lowest bid floor and the other with the rest.
        :param ad_unit_list: List of ad units to be split.
        :return: Tuple containing the lowest bid floor and the rest of the ad units.
        """
        sorted_by_ad_unit_name = self.sort_by_name_postfix_desc(ad_unit_list)
        return sorted_by_ad_unit_name[:-1], sorted_by_ad_unit_name[-1:][0]

    def form_response(
        self,
        assignments: list[dict],
        lowest_bid_floor: dict,
        propensity: float,
        prediction_estimates: list[dict],
    ) -> dict:
        """
        Forms the response dictionary with the predicted bid floor and other details.
        :param assignments: List of assignments.
        :param lowest_bid_floor: The lowest bid floor value.
        :param propensity: The propensity value.
        :param prediction_estimates: List of prediction estimates for the ad units.
        :return: Response dictionary.
        """
        assignments = assignments + [lowest_bid_floor]

        cpm_floor_ad_unit_ids = list(map(lambda x: x["id"], assignments))
        cpm_floor_values = list(map(lambda x: x["bidFloor"], assignments))

        response = {
            "cpmFloorAdUnitIds": cpm_floor_ad_unit_ids,
            "cpmFloorValues": cpm_floor_values,
            "propensity": propensity,
            "estimates": prediction_estimates,
        }
        return response

    def predict(self, context: pd.Series, floors: list[pd.Series | dict], max_ad_units: Optional[int] = None) -> dict:
        floors = [floor.to_dict() if isinstance(floor, pd.Series) else floor for floor in floors]
        lowest_bid_floor = self.sort_by_name_postfix_desc(floors)[-1]
        floors_to_predict = [f for f in floors if f != lowest_bid_floor]

        if max_ad_units is None or max_ad_units >= 3:
            # Current logic applies for 3 or more
            ad_unit_combinations = [
                self.sort_by_name_postfix_desc(list(pair)) for pair in itertools.combinations(floors_to_predict, 2)
            ]
        elif max_ad_units == 2:
            # If exactly 2, choose 1 from combination, mark as medium bid floor, highest as None
            ad_unit_combinations = [list(pair) for pair in itertools.combinations(floors_to_predict, 1)]
        else:
            return self.form_response(
                [],
                lowest_bid_floor,
                1.0,  # Propensity is 1.0 since we are choosing the only available ad unit
                [{"adUnitIds": [lowest_bid_floor["id"]], "predictedBidFloor": -1.0}],
            )

        # Shuffle the ad unit combinations to ensure randomness in selection if estimates are same
        self.rng_shuffle.shuffle(ad_unit_combinations)

        # If the model is not trained or if the random number is less than epsilon, return a random assignment
        if self.clf is None:
            assignments = self.rng_exploration.choice(ad_unit_combinations, size=1)
            propensity = 1 / len(ad_unit_combinations)

            return self.form_response(
                list(assignments[0]),
                lowest_bid_floor,
                propensity,
                [{"adUnitIds": list(assignments[0]), "predictedBidFloor": -1.0}],
            )

        transformed = []
        for ad_unit_list in ad_unit_combinations:
            if len(ad_unit_list) >= 2:
                # Choose the last 2 elements as highest and medium bid floors
                highest_bid_floor_value = ad_unit_list[-1]["bidFloor"]
                medium_bid_floor_value = ad_unit_list[-2]["bidFloor"]
            else:
                highest_bid_floor_value = None
                medium_bid_floor_value = ad_unit_list[-1]["bidFloor"]
            with_hardcoded_context = self.add_hardcoded_contexts(
                context, highest_bid_floor_value, medium_bid_floor_value
            )
            transformed.append(with_hardcoded_context.to_dict())

        value_replaced = (
            self.value_replacer.transform(pd.DataFrame(transformed))
            if self.value_replacer
            else pd.DataFrame(transformed)
        )

        feature_dmatrix = self.features.fields_to_dmatrix_from_df(value_replaced, prediction_phase=True)
        predictions_array = self.clf.predict(feature_dmatrix)
        prediction_estimates = [
            {
                "adUnitIds": list(ad_unit_list),
                "predictedBidFloor": float(pred),
            }
            for ad_unit_list, pred in zip(ad_unit_combinations, predictions_array)
        ]
        best_bid_floor_combo = max(prediction_estimates, key=lambda x: x["predictedBidFloor"])
        propensity = (1 - self.epsilon) + self.epsilon / len(ad_unit_combinations)

        if self.rng_exploration.uniform() < self.epsilon:
            assignments = self.rng_exploration.choice(ad_unit_combinations, size=1)
            if best_bid_floor_combo["adUnitIds"] != list(assignments[0]):
                propensity = self.epsilon / len(ad_unit_combinations)

            return self.form_response(list(assignments[0]), lowest_bid_floor, propensity, prediction_estimates)

        return self.form_response(best_bid_floor_combo["adUnitIds"], lowest_bid_floor, propensity, prediction_estimates)


@dataclass
class S3ModelArtifactInfo:
    bucket: str
    key: str
    file_name: str
    file_name_wo_ext: str


def upload_model_file_to_s3(local_model_base_path, model_artifact_path: S3ModelArtifactInfo):
    boto3_client = boto3.client("s3")
    boto3_client.upload_file(
        os.path.join(local_model_base_path, model_artifact_path.file_name),
        model_artifact_path.bucket,
        model_artifact_path.key,
    )


def save_predictor_model_to_s3(predictor: Predictor, app_id: str, model_artifact_path: S3ModelArtifactInfo):
    """
    Save the predictor model to S3.
    """

    logging.info(f"Saving model to S3: {model_artifact_path.bucket}/{model_artifact_path.key}")

    # Save the model locally
    local_model_base_path = os.path.join("/tmp", str(app_id))

    if not os.path.exists(local_model_base_path):
        os.makedirs(local_model_base_path)

    joblib.dump(predictor, os.path.join(local_model_base_path, model_artifact_path.file_name))

    # Upload to S3
    upload_model_file_to_s3(local_model_base_path, model_artifact_path)


def save_predictor_object(predictor: Predictor, args):
    predictor_file_name = f"{args.customerId}_{args.appId}_{args.modelId}"
    predictor_final_name_with_ext = f"{predictor_file_name}.joblib"

    save_predictor_model_to_s3(
        predictor,
        args.appId,
        S3ModelArtifactInfo(
            bucket=args.s3ModelArtifactBucket,
            key=f"bid_floor_models/{args.date}/{predictor_final_name_with_ext}",
            file_name=predictor_final_name_with_ext,
            file_name_wo_ext=predictor_file_name,
        ),
    )


def run():
    argvs = sys.argv[1:] if len(sys.argv) > 1 else []
    parsed_args_obj = Initialisation.parse_args(args=argvs, parser_obj=ApplovinModelTrainingArgsParser())
    cmd_line_args = parsed_args_obj.parsed_args
    config_file = parsed_args_obj.read_config(
        confs_dir_path=os.path.join(os.path.dirname(__file__), "confs"), config_clazz_type=ConfigFile
    )
    logging.info(
        f"Starting model training for customer {cmd_line_args.customerId}, app {cmd_line_args.appId}, "
        f"model {cmd_line_args.modelId} on date {cmd_line_args.date}"
    )

    bid_floor_management_api = BidFloorManagementAPI(http_client=HttpClient(base_url=config_file.managementApiBaseUrl))
    etl_config = bid_floor_management_api.fetch_etl_config(cmd_line_args.appId)
    model_config = bid_floor_management_api.fetch_model_config(cmd_line_args.appId, cmd_line_args.modelId)

    logging.info(f"ETL Config: {etl_config}")
    logging.info(f"Model Config: {model_config}")

    epsilon = 0.1
    create_empty_model = True
    if model_config and model_config.parameters:
        epsilon = model_config.parameters.get("epsilon", 0.1)
        create_empty_model = model_config.parameters.get("create_empty_model", True)

    init_ray_cluster()

    training_data = read_training_data(
        cmd_line_args.region,
        cmd_line_args.icebergTrainDataTable,
        cmd_line_args.customerId,
        cmd_line_args.appId,
        cmd_line_args.modelId,
        datetime.date.fromisoformat(cmd_line_args.date),
        etl_config.lookbackWindowInDays,
    )

    if create_empty_model:
        logging.info("Creating empty model as it is requested in the args")
        save_predictor_object(
            Predictor(
                epsilon=epsilon,
                clf=None,
                value_replacer=ValueReplacer(valid_values={}, default_value="other"),
                features=Features([]),
            ),
            cmd_line_args,
        )
        return

    if training_data.limit(1).count() != 0:
        logging.info("Training data is not empty, proceeding with model training")

        model_features = ModelFeatures(etl_config=etl_config).as_ray_schema()

        trainer = ModelTrainer(
            customer_id=cmd_line_args.customerId,
            app_id=cmd_line_args.appId,
            model_id=cmd_line_args.modelId,
            date=datetime.datetime.fromisoformat(cmd_line_args.date),
            features=model_features,
            model_config=ModelConfig(**model_config.parameters),
            s3_checkpoint_path=f"s3://{cmd_line_args.s3ModelArtifactBucket}/bid_floor_checkpoints/"
            f"{cmd_line_args.date}/{cmd_line_args.customerId}/{cmd_line_args.appId}/",
        )

        feature_schema = {(field.name, field.dtype) for field in model_features.fields}

        training_data = training_data.map_batches(
            lambda batch: cast_types(batch, dict(feature_schema)),
            batch_format="pandas",
        )

        result, value_replacer, features = trainer.run(
            assignments_with_ad_revenue=training_data, target_column=Schema.TOTAL_AMOUNT, use_validation_set=True
        )

        save_predictor_object(
            Predictor(
                epsilon=epsilon,
                clf=RayTrainReportCallback.get_model(result.checkpoint),
                value_replacer=value_replacer,
                features=features,
            ),
            cmd_line_args,
        )
        logging.info("Model training completed successfully")
    else:
        logging.warning("Training data is empty, hence skipping model training, creating empty model.")
        save_predictor_object(
            Predictor(
                epsilon=epsilon,
                clf=None,
                value_replacer=ValueReplacer(valid_values={}, default_value="other"),
                features=Features([]),
            ),
            cmd_line_args,
        )


if __name__ == "__main__":
    run()
