import datetime
import json
import logging
import os
import shutil
import tarfile
from dataclasses import dataclass

import boto3
import joblib
import requests

from bid_optim_etl_py.applovin_train_runner import Predictor, ValueReplacer, Features, Field  # noqa


class ApplovinETLException(Exception):
    pass


def arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run the applovin bid floor training")
    parser.add_argument("--customerId", type=int, help="Customer ID")
    parser.add_argument("--appId", type=int, help="App ID")
    parser.add_argument("--modelIds", nargs="+", type=str, help="Model IDs")
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    parser.add_argument("--s3ModelArtifactBucket", help="S3 bucket name for model artifact")
    parser.add_argument("--bidFloorVersion", help="Bid floor version")
    parser.add_argument("--allocatorServiceUri", help="Allocator service URI")
    return parser.parse_args()


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


def file_exists_in_s3(boto3_client, bucket: str, key: str) -> bool:
    try:
        boto3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def download_model_artifact_from_s3(
    customer_id, app_id, model_id, model_artifact_path: S3ModelArtifactInfo, postfix="", error_if_empty=True
):
    local_model_file_path = f"/tmp/{customer_id}_{app_id}_{model_id}_{model_artifact_path.file_name}{postfix}"
    boto3_client = boto3.client("s3")
    if file_exists_in_s3(boto3_client, model_artifact_path.bucket, model_artifact_path.key):
        print(f"Model artifact {model_artifact_path} found in s3. Loading predictor object.")
        boto3_client.download_file(model_artifact_path.bucket, model_artifact_path.key, local_model_file_path)
        return joblib.load(local_model_file_path)
    else:
        if error_if_empty:
            raise ApplovinETLException(f"Model artifact {model_artifact_path} not found in s3.")
        else:
            print(f"Model artifact {model_artifact_path} not found in s3. Returning None.")
            return None


def create_tar_archive(source_dir, output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def create_tar_and_upload_to_s3(
    resultant_sagemaker_artifact_name: str, sagemaker_inference_bucket_path: S3ModelArtifactInfo
):
    s3_client = boto3.client("s3")
    staging_folder_location = f"./{resultant_sagemaker_artifact_name}"

    # Tar the staging folder location
    sagemaker_model_tar_file = f"{resultant_sagemaker_artifact_name}.tar.gz"
    create_tar_archive(staging_folder_location, sagemaker_model_tar_file)

    # Uploading the tar file to sagemaker inference path
    s3_client.upload_file(
        f"./{sagemaker_model_tar_file}",
        sagemaker_inference_bucket_path.bucket,
        sagemaker_inference_bucket_path.key,
    )

    return sagemaker_model_tar_file


def publish_model_artifact(customer_id, app_id, model_ids: [str], date, s3_model_artifact_bucket, bid_floor_version):
    sagemaker_tar_content = {}

    for model_id in model_ids:
        model_artifact_path = S3ModelArtifactInfo(
            bucket=s3_model_artifact_bucket,
            key=f"bid_floor_models/{date}/{customer_id}_{app_id}_{model_id}.joblib",
            file_name=f"{customer_id}_{app_id}_{model_id}.joblib",
            file_name_wo_ext=f"{customer_id}_{app_id}_{model_id}",
        )

        model_obj = download_model_artifact_from_s3(
            customer_id, app_id, model_id, model_artifact_path, error_if_empty=True
        )

        sagemaker_tar_content[model_id] = model_obj

    if not sagemaker_tar_content:
        raise ApplovinETLException("No model artifacts found to publish.")

    local_staging_folder = os.path.join("/tmp", f"bid_floor_model_{customer_id}_{app_id}")

    if os.path.exists(local_staging_folder):
        logging.info(f"Deleting existing staging folder {local_staging_folder}")
        shutil.rmtree(local_staging_folder)

    os.makedirs(local_staging_folder, exist_ok=True)

    joblib.dump(sagemaker_tar_content, os.path.join(local_staging_folder, "predictor.joblib"))

    local_tar_gz_artifact_name = f"bid_cb_{app_id}_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}"

    local_tar_file = os.path.join("/tmp", f"{local_tar_gz_artifact_name}.tar.gz")

    create_tar_archive(local_staging_folder, local_tar_file)

    sagemaker_model_artifact_path = S3ModelArtifactInfo(
        bucket=s3_model_artifact_bucket,
        key=f"sagemaker_inference/bid-floor-{bid_floor_version}/{local_tar_gz_artifact_name}.tar.gz",
        file_name=f"{local_tar_gz_artifact_name}.tar.gz",
        file_name_wo_ext=local_tar_gz_artifact_name,
    )

    s3_client = boto3.client("s3")

    s3_client.upload_file(
        local_tar_file,
        sagemaker_model_artifact_path.bucket,
        sagemaker_model_artifact_path.key,
    )


def publish_artifacts():
    parsed_args_obj = arg_parser()

    publish_model_artifact(
        customer_id=parsed_args_obj.customerId,
        app_id=parsed_args_obj.appId,
        model_ids=parsed_args_obj.modelIds,
        date=parsed_args_obj.date,
        s3_model_artifact_bucket=parsed_args_obj.s3ModelArtifactBucket,
        bid_floor_version=parsed_args_obj.bidFloorVersion,
    )

    for model_id in parsed_args_obj.modelIds:
        call_allocator_service(
            allocator_service_uri=parsed_args_obj.allocatorServiceUri,
            reference=parsed_args_obj.appId,
            endpoint_name=f"bid-floor-{parsed_args_obj.appId}-{parsed_args_obj.bidFloorVersion.replace('.', '-')}",
            model_name=model_id,
        )


def call_allocator_service(allocator_service_uri: str, reference: str, endpoint_name: str, model_name: str):
    logging.info(
        f"Updating allocator service for application: {reference} "
        f"with endpoint: {endpoint_name} and model_name: {model_name}"
    )
    headers = {"Content-Type": "application/json"}
    payload = {
        "reference": str(reference),
        "endpointName": endpoint_name,
        "modelName": model_name,
    }

    response = requests.put(
        allocator_service_uri,
        json.dumps(payload),
        headers=headers,
    )

    if response.status_code != 200:
        logging.exception(
            f"Failed to update allocator service for experiment: {reference} "
            f"with status code: {response.status_code}"
        )
        raise ApplovinETLException(
            f"Failed to update allocator service for experiment: {reference} "
            f"with status code: {response.status_code}"
        )


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)
        logging.info("Starting Applovin ETL publish job")
        publish_artifacts()
        logging.info("Completed Applovin ETL publish job")
    except Exception as exp:
        logging.exception("Error while running Applovin ETL publish job")
        raise exp
