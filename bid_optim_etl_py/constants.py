# Bid Floor ETL Constants

import json
from bid_optim_etl_py.helpers.aws_helpers import SecretsManagerHelper

# AppLovin API Configuration
APPLOVIN_API_BASE_URL = "https://o.applovin.com/mediation/v1"

# AWS S3 Buckets
S3_DATA_BUCKET = "com.metica.prod-eu.dplat.data"
S3_ARTIFACTS_BUCKET = "com.metica.prod-eu.dplat.artifacts"
BID_FLOOR_PERCENTILES_PREFIX = "bid-floor-optimisation/applovin/percentile"

APP_ID_TO_APPLOVIN_ID = {
    12101: "com.gamepark.vehicle.simulator.driving.games",
    11802: "com.fp.fashionbattle.catwalkshow.dressupgame",
    12102: "com.gtsy.passengerexpress",
    12053: "com.sh.twoplayergame.mini.challenge",
    12751: "com.playoneer.boxjam",
    12051: "com.sport.cornhole",
    12001: "com.playspare.dinorace",
    12501: "com.deadpixel.colorblockjam",
    12251: "com.braindom",
    12052: "com.tapped.dunkworld",
    10050: "games.starberry.idlevillage",
    12651: "com.dna.solitaireapp",
    12551: "com.noxgames.hex.polis.civilization.empire",
}

# AWS Default Region
DEFAULT_AWS_REGION = "eu-west-1"

# Percentile Columns
PERCENTILE_COLUMNS = ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]

# Data Processing
CPM_MULTIPLIER = 1000  # Convert to CPM
MIN_BID_AMOUNT = 0.01

APPLOVIN_MANAGEMENT_API_KEYS = json.loads(
    json.loads(SecretsManagerHelper(DEFAULT_AWS_REGION).get_secret("prod/appLovinManagmentkeys"))[
        "applovin-management-api-keys"
    ]
)
