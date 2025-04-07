#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Fetch required arguments
# -----------------------------------------------------------------------------
PYTHON_RUNNER=$1
CUSTOMER_ID=$2
APP_ID=$3
EVENTS_BASE_PATH=$4
ENV_TAR_FILE=$5

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
python3 /tmp/runner.py --customer_id "${CUSTOMER_ID}" --app_id "${APP_ID}" --app_reqs_s3_tar "s3://com.metica.dev.dplat.artifacts/bid_optim_etl_py-0.1.0-py3-none-any.whl" --events_base_path_s3 "${EVENTS_BASE_PATH}"

