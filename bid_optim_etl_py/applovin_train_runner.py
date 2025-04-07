import datetime
from dataclasses import dataclass
from typing import Tuple, Optional

import ray
from pyiceberg.catalog import load_catalog
from pyiceberg.expressions import EqualTo
from ray.data import Dataset
from ray.data.preprocessors import StandardScaler
from ray.train import CheckpointConfig, RunConfig, ScalingConfig, Result
from ray.train.xgboost import RayTrainReportCallback
from ray.train.xgboost import XGBoostTrainer


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


def init_ray_cluster():
    ray.init()


@dataclass
class ModelTrainer:
    customer_id: int
    app_id: int
    model_id: str
    date: datetime.datetime
    s3_checkpoint_path: Optional[str] = None

    def _run_config(self, storage_path: Optional[str] = None) -> RunConfig:
        return RunConfig(
            checkpoint_config=CheckpointConfig(
                # Checkpoint every 10 iterations.
                checkpoint_frequency=10,
                # Only keep the latest checkpoint and delete the others.
                num_to_keep=1,
            ),
            storage_path=storage_path or f"~/ray_results_{self.customer_id}_{self.app_id}_{self.model_id}",
            ## If running in a multi-node cluster, this is where you
            ## should configure the run's persistent storage that is accessible
            ## across all worker nodes with `storage_path="s3://..."`
        )

    def prepare_data(self, input_data: ray.data.Dataset, target_column: str) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and split the dataset into train, validation, and test sets."""
        train_dataset, valid_dataset = input_data.train_test_split(test_size=0.3)
        test_dataset = valid_dataset.drop_columns([target_column])
        return train_dataset, valid_dataset, test_dataset

    def preprocess_data(self, train_dataset: ray.data.Dataset, valid_dataset: Optional[ray.data.Dataset] = None,
                        test_dataset: Optional[ray.data.Dataset] = None) -> Tuple[
        ray.data.Dataset, ray.data.Dataset, ray.data.Dataset]:
        """Preprocess the dataset by scaling specific columns."""
        # Define the columns to scale
        columns_to_scale = ["mean radius", "mean texture"]

        # Initialize the preprocessor
        preprocessor = StandardScaler(columns=columns_to_scale)

        return (
            preprocessor.fit_transform(train_dataset),
            preprocessor.fit_transform(valid_dataset) if valid_dataset else None,
            preprocessor.fit_transform(test_dataset) if test_dataset else None
        )

    def save_model(self, result: Result, model_save_path: str):
        checkpoint = result.checkpoint
        # Save the model as .xgb file
        booster = RayTrainReportCallback.get_model(checkpoint)
        booster.save_model(model_save_path)

    def run(self, assignments_with_ad_revenue: ray.data.Dataset, target_column: str,
            cross_validation: bool = False) -> Result:
        # Configure checkpointing to save progress during training
        run_config = self._run_config(storage_path=self.s3_checkpoint_path)

        # Load and split the dataset
        if cross_validation:
            train_dataset, valid_dataset, _test_dataset = self.prepare_data(assignments_with_ad_revenue, target_column)
        else:
            train_dataset, valid_dataset, _test_dataset = (assignments_with_ad_revenue, None, None)

        # Preprocess the dataset
        train_dataset, valid_dataset, _test_dataset = self.preprocess_data(
            train_dataset, valid_dataset, _test_dataset
        )

        datasets = {"train": train_dataset}

        if valid_dataset:
            datasets["valid"] = valid_dataset

        # Set up the XGBoost trainer with the specified configuration
        trainer = XGBoostTrainer(
            # see "How to scale out training?" for more details
            scaling_config=ScalingConfig(
                # Number of workers to use for data parallelism.
                num_workers=2,
                # Whether to use GPU acceleration. Set to True to schedule GPU workers.
                use_gpu=False,
            ),
            label_column=target_column,
            num_boost_round=20,
            # XGBoost specific params (see the `xgboost.train` API reference)
            params={
                "objective": "binary:logistic",
                # uncomment this and set `use_gpu=True` to use GPU for training
                # "tree_method": "gpu_hist",
                "eval_metric": ["logloss", "error"],
            },
            datasets=datasets,
            run_config=run_config,
        )
        result = trainer.fit()
        return result


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
    result = trainer.run(
        assignments_with_ad_revenue=training_data,
        target_column="target",
    )
    print(result.metrics)


if __name__ == '__main__':
    run()
