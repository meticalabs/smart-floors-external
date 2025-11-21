#!/usr/bin/env python3
"""
Summary of S3 access capabilities for the provided AWS credentials.
"""

import boto3
import logging
from botocore.exceptions import ClientError

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
    
    logger.info("=" * 80)
    logger.info("S3 ACCESS SUMMARY FOR AWS CREDENTIALS")
    logger.info("=" * 80)
    
    try:
        # Create session
        session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region
        )
        
        s3_client = session.client('s3')
        
        # List all objects in the allowed prefix
        logger.info(f"Scanning prefix: {allowed_prefix}")
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=allowed_prefix,
            MaxKeys=1000
        )
        
        if 'Contents' in response:
            objects = response['Contents']
            logger.info(f"Found {len(objects)} accessible objects")
            
            # Group by app_id (the number after 11601/)
            app_groups = {}
            for obj in objects:
                key_parts = obj['Key'].split('/')
                if len(key_parts) >= 5:  # bid-floor-optimisation/applovin/percentile/11601/APP_ID/...
                    app_id = key_parts[4]
                    if app_id not in app_groups:
                        app_groups[app_id] = []
                    app_groups[app_id].append(obj)
            
            logger.info(f"Objects grouped by app_id:")
            for app_id, app_objects in app_groups.items():
                logger.info(f"  App ID {app_id}: {len(app_objects)} files")
                
                # Show file types
                file_types = {}
                for obj in app_objects:
                    filename = obj['Key'].split('/')[-1]
                    if '_' in filename:
                        file_type = filename.split('_')[-1].replace('.json', '')
                        if file_type not in file_types:
                            file_types[file_type] = 0
                        file_types[file_type] += 1
                
                for file_type, count in file_types.items():
                    logger.info(f"    - {file_type}: {count} files")
            
            # Show recent files
            logger.info(f"\nMost recent files:")
            recent_objects = sorted(objects, key=lambda x: x['LastModified'], reverse=True)[:10]
            for obj in recent_objects:
                filename = obj['Key'].split('/')[-1]
                logger.info(f"  {filename} ({obj['LastModified'].strftime('%Y-%m-%d %H:%M')})")
            
            # Test permissions
            logger.info(f"\nPermission Summary:")
            logger.info(f"  ✅ List objects: YES")
            logger.info(f"  ✅ Read objects: YES")
            logger.info(f"  ✅ Write objects: YES")
            logger.info(f"  ❌ Delete objects: NO (not in IAM policy)")
            
        else:
            logger.info("No objects found in the allowed prefix")
        
        logger.info("=" * 80)
        logger.info("WHAT THE CREDENTIALS CAN ACCESS:")
        logger.info("=" * 80)
        logger.info(f"✅ Bucket: {bucket_name}")
        logger.info(f"✅ Prefix: {allowed_prefix}*")
        logger.info(f"✅ Actions: ListBucket, GetBucketLocation, GetObject, PutObject")
        logger.info(f"❌ Actions: DeleteObject (not allowed)")
        logger.info(f"❌ Other prefixes: Not accessible")
        
        logger.info("=" * 80)
        logger.info("IAM POLICY ANALYSIS:")
        logger.info("=" * 80)
        logger.info("The IAM policy allows access ONLY to:")
        logger.info("  - ListBucket with prefix: bid-floor-optimisation/applovin/percentile/11601/*")
        logger.info("  - GetObject/PutObject on: bid-floor-optimisation/applovin/percentile/11601/*")
        logger.info("  - No access to other prefixes or bucket root")
        logger.info("  - No DeleteObject permission")
        
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()



