# AppLovin Bid Floor Update Tool

This tool allows external customers to update bid floor values for their AppLovin ad units using pre-configured JSON files.

## Installation

```bash
pip install requests
```

## Usage

```bash
python scripts/update_bid_floor_values.py \
    --packageName "your.app.package" \
    --adUnitConfigFolder "path/to/config/folder" \
    --apiKey "your_applovin_api_key" \
    --adType "reward" \
    --platform "android"
```

### Parameters

- `--packageName`: Your app's package name (e.g., "com.example.myapp")
- `--adUnitConfigFolder`: Path to folder containing JSON configuration files
- `--apiKey`: Your AppLovin Management API key
- `--adType`: Ad type - "reward" or "inter" (default: "reward")
- `--platform`: Platform - "android" or "ios" (default: "android")

### Configuration File Format

Place JSON files in the configuration folder with the following format:

```json
[
    {
        "ad_unit_id": "your_ad_unit_id",
        "ad_unit_name": "your_ad_unit_name",
        "bid_floors": [
            {
                "country_group_name": "tier1",
                "cpm": "2.50",
                "countries": {
                    "type": "INCLUDE",
                    "values": ["us", "ca", "gb"]
                }
            }
        ]
    }
]
```

## Requirements

- Python 3.10+
- Valid AppLovin Management API key
- Pre-configured bid floor JSON files provided by Metica