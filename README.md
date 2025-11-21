## AppLovin Bid Floor Updater (Client)

This repository provides a robust client-side tool to update AppLovin bid floors based on the latest bid floor percentiles stored in S3. It also uploads the computed ad unit configurations back to S3 for observability and audit.

### What it does
- Reads the latest bid floor percentiles JSON from S3.
- Computes bid floor configurations per Metica ad unit.
- Updates AppLovin ad units using the client-provided API key.
- Uploads the resulting `ad_unit_configurations.json` to S3.

### Requirements
- Python 3.10–3.11
- AppLovin Management API key
- AWS access key/secret with read and write access to artifacts bucket

### Installation
```bash
pip install -e .
```

### Expected S3 layout
```
s3://com.metica.prod-eu.dplat.artifacts/
  bid-floor-optimisation/applovin/percentile/
    <customer_id>/<app_id>/<YYYY-MM-DD>/<platform>/<ad_type>.json
    <customer_id>/<app_id>/uploads/ad_unit_configurations.json
```

### Usage
```bash
python scripts/update_bid_floor_values.py \
  --customer-id <METICA_CUSTOMER_ID> \
  --app-id <METICA_APP_ID> \
  --ad-type reward \
  --platform android \
  --applovin-api-key "<CLIENT_APPLOVIN_API_KEY>" \
  --aws-access-key-id "<AWS_ACCESS_KEY_ID>" \
  --aws-secret-access-key "<AWS_SECRET_ACCESS_KEY>" \
  --aws-region eu-west-1 \
  --s3-bucket com.metica.prod-eu.dplat.artifacts \
  --package-name <APPLOVIN_PACKAGE_NAME>
```

### Exit behavior
- Fails with a clear error if no latest percentiles JSON is found for the given `platform` and `ad_type`.
- Logs a success message after updating AppLovin and uploading configurations.

### Development
- Run tests: `pytest -q`
- Lint: use your preferred linter/formatter; keep code readable and typed hints explicit where helpful.