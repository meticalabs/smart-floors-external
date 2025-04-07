#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Fetch required arguments
# -----------------------------------------------------------------------------
PYTHON_RUNNER=$1
CUSTOMER_ID=$2
APP_ID=$3
MODEL_ID=$4
ICEBERG_TRAIN_TABLE=$5
ENV_TAR_FILE=$6

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
aws s3 cp "${ENV_TAR_FILE}" /tmp/env.tar.gz
pip install -U /tmp/env.tar.gz

# -----------------------------------------------------------------------------
# Run the Ray job
# -----------------------------------------------------------------------------
echo "Running the Ray job"
aws s3 cp "${PYTHON_RUNNER}" /tmp/runner.py
python3 /tmp/runner.py --customerId "${CUSTOMER_ID}" --appId "${APP_ID}" --modelId "${MODEL_ID}" --icebergTrainDataTable "${ICEBERG_TRAIN_TABLE}"

