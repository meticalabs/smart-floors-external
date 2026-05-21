Applovin Ad Units Report ETL
=======================
This script enables sharing AppLovin ad unit performance with Metica without sharing the management API key or reporting API key.

## Features
- Generates reports for ad units
- Does not require sharing management or reporting API keys

## Usage
1. Install dependencies:
   ```zsh
   poetry install
   ```
2. Run the script using Python, providing all required AWS and API credentials:
   ```zsh
   python scripts/applovin_adunits_report_etl.py \
     --applovin_id YOUR_APPLOVIN_ID \
     --management_api_key YOUR_MANAGEMENT_API_KEY \
     --reporting_api_key YOUR_REPORTING_API_KEY \
     --s3_output s3://your-bucket/path/to/output.csv \
     --aws-access-key-id YOUR_AWS_ACCESS_KEY_ID \
     --aws-secret-access-key YOUR_AWS_SECRET_ACCESS_KEY \
     --aws-region YOUR_AWS_REGION
   ```

## Requirements
- Python 3.x
- Any additional dependencies should be listed in your project's requirements file.

## Contact
For questions or support, please contact rowan@metica.com.
