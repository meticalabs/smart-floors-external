#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Fetch required arguments
# -----------------------------------------------------------------------------
PYTHON_RUNNER=$1
CUSTOMER_ID=$2
APP_ID=$3
MODEL_IDS=$4
BID_FLOOR_VERSION=$5
ENV_TAR_FILE=$6
S3_MODEL_ARTIFACT_BUCKET=$7
DATE=$8

# -----------------------------------------------------------------------------
# Activate the conda environment
# -----------------------------------------------------------------------------
echo "Activating conda environment"
source $HOME/.bashrc
conda init
source $HOME/.bashrc
conda activate py-env

# -----------------------------------------------------------------------------
# Install the dependencies
# -----------------------------------------------------------------------------
aws s3 cp "${ENV_TAR_FILE}" /tmp/publisher_env.tar.gz
pip install -U /tmp/publisher_env.tar.gz

# -----------------------------------------------------------------------------
# Run the Publisher job
# -----------------------------------------------------------------------------
echo "Running the Publisher job"
aws s3 cp "${PYTHON_RUNNER}" /tmp/runner.py
python3 /tmp/runner.py --customerId "${CUSTOMER_ID}" --appId "${APP_ID}" --modelIds "${MODEL_IDS//,/ }" --bidFloorVersion "${BID_FLOOR_VERSION}" --s3ModelArtifactBucket "${S3_MODEL_ARTIFACT_BUCKET}" --date "${DATE}"

