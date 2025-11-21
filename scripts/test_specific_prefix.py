#!/usr/bin/env python3
"""
Test S3 access with the specific prefix allowed by the IAM policy.
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
    
    # The specific prefix allowed by the IAM policy
    allowed_prefix = "bid-floor-optimisation/applovin/percentile/11601/"
    
    logger.info("Testing AWS credentials...")
    
    try:
        # Create session
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
        
        # Create S3 client
        s3_client = session.client('s3')
        
        # Test bucket access with the specific prefix
        logger.info(f"Testing access to bucket: {bucket_name}")
        logger.info(f"Using allowed prefix: {allowed_prefix}")
        
        # List objects with the specific prefix
        logger.info("Listing objects in the allowed prefix...")
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=allowed_prefix,
                MaxKeys=50
            )
            
            if 'Contents' in response:
                logger.info(f"✅ Successfully accessed prefix! Found {len(response['Contents'])} objects:")
                for obj in response['Contents']:
                    logger.info(f"   📄 {obj['Key']}")
                    logger.info(f"      Size: {obj['Size']} bytes")
                    logger.info(f"      Last Modified: {obj['LastModified']}")
                    logger.info("")
            else:
                logger.info("✅ Can access the prefix, but no objects found")
                
        except ClientError as e:
            logger.error(f"❌ Cannot list objects in prefix: {e}")
            return
        
        # Test reading a specific object if any exist
        if 'Contents' in response and response['Contents']:
            test_object = response['Contents'][0]['Key']
            logger.info(f"Testing read access to object: {test_object}")
            try:
                obj_response = s3_client.get_object(Bucket=bucket_name, Key=test_object)
                logger.info(f"✅ Successfully read object! Content length: {obj_response['ContentLength']} bytes")
                logger.info(f"   Content type: {obj_response.get('ContentType', 'Unknown')}")
            except ClientError as e:
                logger.error(f"❌ Cannot read object: {e}")
        
        # Test writing a small test file
        test_key = f"{allowed_prefix}test_access.txt"
        test_content = "This is a test file to verify write access."
        logger.info(f"Testing write access with key: {test_key}")
        try:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=test_key,
                Body=test_content,
                ContentType='text/plain'
            )
            logger.info("✅ Successfully wrote test file!")
            
            # Clean up the test file
            s3_client.delete_object(Bucket=bucket_name, Key=test_key)
            logger.info("✅ Successfully deleted test file")
            
        except ClientError as e:
            logger.error(f"❌ Cannot write object: {e}")
        
        # Test with the exact prefix from the policy (without trailing slash)
        exact_prefix = "bid-floor-optimisation/applovin/percentile/11601"
        logger.info(f"Testing with exact prefix from policy: {exact_prefix}")
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=exact_prefix,
                MaxKeys=10
            )
            
            if 'Contents' in response:
                logger.info(f"✅ Found {len(response['Contents'])} objects with exact prefix")
                for obj in response['Contents'][:5]:  # Show first 5
                    logger.info(f"   📄 {obj['Key']}")
            else:
                logger.info("ℹ️  No objects found with exact prefix")
                
        except ClientError as e:
            logger.error(f"❌ Cannot access exact prefix: {e}")
        
        logger.info("✅ S3 access test completed!")
        
    except NoCredentialsError:
        logger.error("❌ No credentials found")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()



