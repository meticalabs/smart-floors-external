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
    <customer_id>/<app_id>/<YYYY-MM-DD>_<platform>_<ad_type>.json
    <customer_id>/<app_id>/uploads/ad_unit_configurations_<platform>_<ad_type>.json
```

The script lists the prefix `<customer_id>/<app_id>/` and picks the most recently
modified file whose key ends with `_<platform>_<ad_type>.json`.

### Percentile file contract

The percentile JSON is a flat array of records produced by Metica's nightly
pipeline. Each row corresponds to one country (or country/platform/ad-format
combination) and contains:

- `user.country` — ISO-2 country code (lowercase). Required.
- One or more **percentile columns** — any column whose name matches the regex
  `^p\d+$` (e.g. `p10`, `p40`, `p83`, `p99`). The script auto-discovers them and
  sorts ascending by percentile number.
- Optional metadata columns such as `count`, `total_revenue`, `mean`,
  `pricing_strategy`, etc. The script ignores these.

**Clients do not need to update this script when Metica changes the percentile
grid (standard / aggressive / ultra-aggressive), the lookback window, the
traffic-threshold smart-pricing config, or any future percentile-calculator
option.** Whichever `pNN` columns appear in the file are used as-is.

### Ad-unit ↔ percentile mapping

The script fetches all Metica ad units from AppLovin for the given
package/ad-format, excludes the `..._1` control unit, sorts the remainder by the
numeric suffix in the name, and assigns each one a percentile column via equal
distribution (`numpy.linspace` + `numpy.ceil`):

- **Equal count** (`N` ad units, `N` percentiles) — ad unit *i* gets percentile
  column *i*.
- **Fewer ad units than percentiles** (`N` ad units, `M` percentiles with `N<M`)
  — ad units are spread across the upper end of the grid; the lowest percentile
  is intentionally skipped.
- **More ad units than percentiles** (`N` ad units, `M` percentiles with `N>M`)
  — only the first `M` ad units receive bid-floor updates; the remainder are
  logged as skipped.
- **Zero ad units** — no updates are made.

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