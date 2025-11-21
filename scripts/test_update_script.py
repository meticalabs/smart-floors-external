#!/usr/bin/env python3
"""
Test script to demonstrate the correct usage of update_bid_floor_values.py
with the provided AWS credentials.
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_correct_usage():
    """Test with the correct customer_id (11601) that's allowed by the IAM policy."""
    
    # AWS credentials - should be set via environment variables
    import os
    access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    
    # Parameters that should work with the IAM policy
    cmd = [
        "python", "scripts/update_bid_floor_values.py",
        "--customer-id", "11601",  # This is the only customer_id allowed by the IAM policy
        "--app-id", "12751",       # This app_id exists in the S3 bucket
        "--ad-type", "reward",     # This ad type exists in the S3 bucket
        "--platform", "android",   # This platform exists in the S3 bucket
        "--applovin-api-key", "YOUR_APPLOVIN_API_KEY",  # You'll need to provide this
        "--aws-access-key-id", access_key_id,
        "--aws-secret-access-key", secret_access_key,
        "--aws-region", "eu-west-1",
        "--package-name", "com.example.app"  # You'll need to provide the actual package name
    ]
    
    logger.info("Testing with correct parameters (customer_id=11601):")
    logger.info("Command: " + " ".join(cmd))
    logger.info("Note: You'll need to provide a valid AppLovin API key and package name")
    
    return cmd


def test_incorrect_usage():
    """Test with an incorrect customer_id to show the error."""
    
    # AWS credentials - should be set via environment variables
    import os
    access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    
    # Parameters that should fail due to IAM policy restrictions
    cmd = [
        "python", "scripts/update_bid_floor_values.py",
        "--customer-id", "99999",  # This customer_id is NOT allowed by the IAM policy
        "--app-id", "12751",
        "--ad-type", "reward",
        "--platform", "android",
        "--applovin-api-key", "YOUR_APPLOVIN_API_KEY",
        "--aws-access-key-id", access_key_id,
        "--aws-secret-access-key", secret_access_key,
        "--aws-region", "eu-west-1",
        "--package-name", "com.example.app"
    ]
    
    logger.info("\nTesting with incorrect parameters (customer_id=99999):")
    logger.info("Command: " + " ".join(cmd))
    logger.info("This should fail with a clear error message about customer_id restrictions")
    
    return cmd


def main():
    logger.info("=" * 80)
    logger.info("TESTING UPDATE_BID_FLOOR_VALUES.PY SCRIPT")
    logger.info("=" * 80)
    
    # Show the correct usage
    correct_cmd = test_correct_usage()
    
    # Show what would happen with incorrect usage
    incorrect_cmd = test_incorrect_usage()
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info("✅ The script has been fixed to:")
    logger.info("   1. Validate that customer_id is 11601 (the only one allowed)")
    logger.info("   2. Provide clear error messages when access is denied")
    logger.info("   3. Log the prefix being searched for debugging")
    logger.info("")
    logger.info("🔑 Key points:")
    logger.info("   - Only customer_id 11601 is allowed by the IAM policy")
    logger.info("   - The script will now fail fast with a clear error if wrong customer_id is used")
    logger.info("   - You need to provide a valid AppLovin API key and package name")
    logger.info("")
    logger.info("📁 Available files in S3:")
    logger.info("   - App ID 12751 has both 'inter' and 'reward' ad types")
    logger.info("   - Files are available for Android platform")
    logger.info("   - Most recent files are from 2025-10-15")


if __name__ == "__main__":
    main()



