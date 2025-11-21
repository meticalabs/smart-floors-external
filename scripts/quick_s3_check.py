#!/usr/bin/env python3
"""
Quick script to check S3 access with provided AWS credentials.
"""

import boto3
import logging
from botocore.exceptions import ClientError, NoCredentialsError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # AWS credentials - should be set via environment variables
    import os
    access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    bucket_name = "com.metica.prod-eu.dplat.artifacts"
    region = os.getenv("AWS_REGION", "eu-west-1")
    
    logger.info("Testing AWS credentials...")
    
    try:
        # Create session and test credentials
        session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region
        )
        
        # Test credentials
        sts_client = session.client('sts')
        response = sts_client.get_caller_identity()
        logger.info(f"✅ Credentials are valid!")
        logger.info(f"   Account ID: {response.get('Account')}")
        logger.info(f"   User ARN: {response.get('Arn')}")
        
        # Test bucket access
        s3_client = session.client('s3')
        logger.info(f"Testing access to bucket: {bucket_name}")
        
        # Check if bucket exists and is accessible
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"✅ Bucket '{bucket_name}' is accessible")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                logger.error(f"❌ Bucket '{bucket_name}' does not exist")
            elif error_code == 'AccessDenied':
                logger.error(f"❌ Access denied to bucket '{bucket_name}'")
            else:
                logger.error(f"❌ Error accessing bucket: {e}")
            return
        
        # List first 10 objects to test listing permissions
        logger.info("Testing object listing permissions...")
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                MaxKeys=10
            )
            
            if 'Contents' in response:
                logger.info(f"✅ Can list objects! Found {len(response['Contents'])} objects (showing first 10):")
                for obj in response['Contents']:
                    logger.info(f"   📄 {obj['Key']} ({obj['Size']} bytes)")
            else:
                logger.info("✅ Can list objects, but bucket appears to be empty")
                
        except ClientError as e:
            logger.error(f"❌ Cannot list objects: {e}")
        
        # Test with a specific prefix if needed
        logger.info("Testing with common prefixes...")
        common_prefixes = ['bid-floor-percentiles/', 'uploads/', 'data/', 'logs/']
        
        for prefix in common_prefixes:
            try:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=prefix,
                    MaxKeys=5
                )
                if 'Contents' in response:
                    logger.info(f"✅ Found {len(response['Contents'])} objects with prefix '{prefix}':")
                    for obj in response['Contents'][:3]:  # Show first 3
                        logger.info(f"   📄 {obj['Key']}")
                else:
                    logger.info(f"ℹ️  No objects found with prefix '{prefix}'")
            except ClientError as e:
                logger.warning(f"⚠️  Cannot access prefix '{prefix}': {e}")
        
        logger.info("✅ S3 access check completed successfully!")
        
    except NoCredentialsError:
        logger.error("❌ No credentials found")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()



