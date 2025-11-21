# Bid Floor ETL Constants

# AppLovin API Configuration
APPLOVIN_API_BASE_URL = "https://o.applovin.com/mediation/v1"

# AWS S3 Buckets
S3_ARTIFACTS_BUCKET = "com.metica.prod-eu.dplat.artifacts"
BID_FLOOR_PERCENTILES_PREFIX = "bid-floor-optimisation/applovin/percentile"

# Percentile Columns
PERCENTILE_COLUMNS = ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]

# Data Processing
CPM_MULTIPLIER = 1000  # Convert to CPM

# Max CPM value
MAX_CPM = 500