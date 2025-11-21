#!/usr/bin/env python3
"""
Script to check which S3 files can be accessed with the provided AWS credentials.
"""

import argparse
import boto3
import logging
from botocore.exceptions import ClientError, NoCredentialsError
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_credentials(access_key_id: str, secret_access_key: str, region: str = 'eu-west-1') -> bool:
    """
    Test if the provided AWS credentials are valid.
    
    Args:
        access_key_id: AWS access key ID
        secret_access_key: AWS secret access key
        region: AWS region
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    try:
        session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region
        )
        
        # Test credentials by calling STS GetCallerIdentity
        sts_client = session.client('sts')
        response = sts_client.get_caller_identity()
        logger.info(f"Credentials are valid. Account ID: {response.get('Account')}")
        logger.info(f"User ARN: {response.get('Arn')}")
        return True
        
    except NoCredentialsError:
        logger.error("No credentials found")
        return False
    except ClientError as e:
        logger.error(f"Invalid credentials: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error testing credentials: {e}")
        return False


def check_bucket_access(s3_client, bucket_name: str) -> bool:
    """
    Check if we can access the specified S3 bucket.
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: Name of the S3 bucket
        
    Returns:
        bool: True if bucket is accessible, False otherwise
    """
    try:
        # Try to get bucket location
        response = s3_client.get_bucket_location(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' is accessible")
        logger.info(f"Bucket region: {response.get('LocationConstraint', 'us-east-1')}")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            logger.error(f"Bucket '{bucket_name}' does not exist")
        elif error_code == 'AccessDenied':
            logger.error(f"Access denied to bucket '{bucket_name}'")
        else:
            logger.error(f"Error accessing bucket '{bucket_name}': {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking bucket access: {e}")
        return False


def list_objects_in_bucket(s3_client, bucket_name: str, prefix: str = '', max_keys: int = 1000) -> List[Dict]:
    """
    List objects in the S3 bucket that the credentials can access.
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: Name of the S3 bucket
        prefix: Prefix to filter objects (optional)
        max_keys: Maximum number of objects to return
        
    Returns:
        List of object metadata dictionaries
    """
    objects = []
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=bucket_name,
            Prefix=prefix,
            PaginationConfig={'MaxItems': max_keys}
        )
        
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    objects.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'],
                        'storage_class': obj.get('StorageClass', 'STANDARD')
                    })
        
        logger.info(f"Found {len(objects)} objects in bucket '{bucket_name}' with prefix '{prefix}'")
        return objects
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDenied':
            logger.error(f"Access denied when listing objects in bucket '{bucket_name}'")
        else:
            logger.error(f"Error listing objects in bucket '{bucket_name}': {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error listing objects: {e}")
        return []


def test_object_access(s3_client, bucket_name: str, object_key: str) -> bool:
    """
    Test if we can access a specific object in the S3 bucket.
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: Name of the S3 bucket
        object_key: Key of the object to test
        
    Returns:
        bool: True if object is accessible, False otherwise
    """
    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        logger.info(f"Object '{object_key}' is accessible (Size: {response['ContentLength']} bytes)")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logger.warning(f"Object '{object_key}' does not exist")
        elif error_code == 'AccessDenied':
            logger.warning(f"Access denied to object '{object_key}'")
        else:
            logger.warning(f"Error accessing object '{object_key}': {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error testing object access: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check S3 file access with provided AWS credentials")
    parser.add_argument("--access-key-id", type=str, required=True, help="AWS Access Key ID")
    parser.add_argument("--secret-access-key", type=str, required=True, help="AWS Secret Access Key")
    parser.add_argument("--bucket", type=str, default="com.metica.prod-eu.dplat.artifacts", help="S3 bucket name")
    parser.add_argument("--region", type=str, default="eu-west-1", help="AWS region")
    parser.add_argument("--prefix", type=str, default="", help="Prefix to filter objects (optional)")
    parser.add_argument("--max-objects", type=int, default=1000, help="Maximum number of objects to list")
    parser.add_argument("--test-specific-object", type=str, help="Test access to a specific object key")
    
    args = parser.parse_args()
    
    logger.info("Starting S3 access check...")
    logger.info(f"Bucket: {args.bucket}")
    logger.info(f"Region: {args.region}")
    if args.prefix:
        logger.info(f"Prefix: {args.prefix}")
    
    # Test credentials
    logger.info("Testing AWS credentials...")
    if not test_credentials(args.access_key_id, args.secret_access_key, args.region):
        logger.error("Credential test failed. Exiting.")
        return 1
    
    # Create S3 client
    session = boto3.Session(
        aws_access_key_id=args.access_key_id,
        aws_secret_access_key=args.secret_access_key,
        region_name=args.region
    )
    s3_client = session.client('s3')
    
    # Check bucket access
    logger.info("Checking bucket access...")
    if not check_bucket_access(s3_client, args.bucket):
        logger.error("Bucket access check failed. Exiting.")
        return 1
    
    # Test specific object if provided
    if args.test_specific_object:
        logger.info(f"Testing access to specific object: {args.test_specific_object}")
        test_object_access(s3_client, args.bucket, args.test_specific_object)
    
    # List objects in bucket
    logger.info("Listing accessible objects...")
    objects = list_objects_in_bucket(s3_client, args.bucket, args.prefix, args.max_objects)
    
    if objects:
        logger.info("\n=== ACCESSIBLE OBJECTS ===")
        for obj in objects[:50]:  # Show first 50 objects
            logger.info(f"Key: {obj['key']}")
            logger.info(f"  Size: {obj['size']} bytes")
            logger.info(f"  Last Modified: {obj['last_modified']}")
            logger.info(f"  Storage Class: {obj['storage_class']}")
            logger.info("")
        
        if len(objects) > 50:
            logger.info(f"... and {len(objects) - 50} more objects")
        
        logger.info(f"\nTotal accessible objects: {len(objects)}")
    else:
        logger.warning("No objects found or accessible in the bucket")
    
    logger.info("S3 access check completed.")
    return 0


if __name__ == "__main__":
    exit(main())



