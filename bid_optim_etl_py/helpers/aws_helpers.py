import boto3
import logging
from typing import List, Dict, Any
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class S3Helper:
    """Helper class for S3 operations."""
    
    def __init__(self, region_name: str = "eu-west-1"):
        self.s3_client = boto3.client("s3", region_name=region_name)
    
    def list_parquet_files(self, bucket: str, prefix: str, cutoff_date=None) -> List[str]:
        """List parquet files in S3 bucket with optional date filtering."""
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            parquet_files = []
            for page in page_iterator:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        if obj["Key"].endswith(".parquet"):
                            s3_path = f"s3://{bucket}/{obj['Key']}"
                            parquet_files.append(s3_path)
            
            logger.info(f"Found {len(parquet_files)} parquet files in {bucket}/{prefix}")
            return parquet_files
            
        except ClientError as e:
            logger.error(f"Error listing S3 objects: {e}")
            raise
    
    def read_json(self, bucket: str, key: str) -> str:
        """Read JSON file from S3."""
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read().decode("utf-8")
        except ClientError as e:
            logger.error(f"Error reading S3 object {bucket}/{key}: {e}")
            raise
    
    def write_json(self, bucket: str, key: str, data: str, content_type: str = "application/json") -> str:
        """Write JSON data to S3."""
        try:
            self.s3_client.put_object(
                Bucket=bucket, 
                Key=key, 
                Body=data, 
                ContentType=content_type
            )
            s3_path = f"s3://{bucket}/{key}"
            logger.info(f"Successfully wrote to {s3_path}")
            return s3_path
        except ClientError as e:
            logger.error(f"Error writing to S3 {bucket}/{key}: {e}")
            raise


class SecretsManagerHelper:
    """Helper class for AWS Secrets Manager operations."""
    
    def __init__(self, region_name: str = "eu-west-1"):
        self.client = boto3.client("secretsmanager", region_name=region_name)
    
    def get_secret(self, secret_name: str) -> str:
        """Retrieve secret from AWS Secrets Manager."""
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            return response['SecretString']
        except ClientError as e:
            logger.error(f"Error retrieving secret {secret_name}: {e}")
            raise
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
