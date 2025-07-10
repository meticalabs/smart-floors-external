#!/bin/bash

# This script generates an empty model artifact, uploads it to a specified S3 location,
# and optionally updates the allocator service via a Lambda invocation.

set -e

# Default values
UPDATE_ALLOCATOR=false
MODEL_VERSION=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --s3-path)
            S3_PATH="$2"
            shift # past argument
            shift # past value
            ;;
        --update-allocator)
            UPDATE_ALLOCATOR=true
            shift # past argument
            ;;
        --model-version)
            MODEL_VERSION="$2"
            shift # past argument
            shift # past value
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for required arguments
if [ -z "$S3_PATH" ]; then
  echo "Error: S3 path argument is missing."
  echo "Usage: $0 --s3-path s3://your-bucket/your-prefix/"
  exit 1
fi

# Determine the project root directory based on the script's location
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Generate a unique filename for the artifact
TIMESTAMP=$(date +%Y%m%d%H%M%S)
ARTIFACT_NAME="empty_model_${TIMESTAMP}.tar.gz"
LOCAL_TAR_PATH="/tmp/${ARTIFACT_NAME}"
PYTHON_SCRIPT_PATH="${PROJECT_ROOT}/bid_optim_etl_py/create_empty_model.py"

# Run the Python script to create the empty model
python "$PYTHON_SCRIPT_PATH" --output-file "$LOCAL_TAR_PATH"

# Check if the tar file exists
if [ ! -f "$LOCAL_TAR_PATH" ]; then
    echo "Error: Model artifact not found at $LOCAL_TAR_PATH"
    exit 1
fi

# Upload the tar file to the specified S3 path
S3_FULL_PATH="${S3_PATH%/}/${ARTIFACT_NAME}"
echo "Uploading $LOCAL_TAR_PATH to $S3_FULL_PATH"
aws s3 cp "$LOCAL_TAR_PATH" "$S3_FULL_PATH"

# Update the allocator service if the flag is set
if [ "$UPDATE_ALLOCATOR" = true ]; then
    if [ -z "$MODEL_VERSION" ]; then
        echo "Error: --model-version is required when using --update-allocator"
        exit 1
    fi

    LAMBDA_NAME="bid-floor-model-update-lambda"
    ENDPOINT_NAME="bid-floor-${MODEL_VERSION//./-}"
    REFERENCE="default_bid_floor"

    echo "Invoking Lambda function: $LAMBDA_NAME"
    echo "Payload: reference=$REFERENCE, endpointName=$ENDPOINT_NAME, modelName=$ARTIFACT_NAME"

    PAYLOAD=$(jq -n \
                --arg ref "${REFERENCE}" \
                --arg endpoint "${ENDPOINT_NAME}" \
                --arg model "${ARTIFACT_NAME}" \
                '{reference: $ref, endpointName: $endpoint, modelName: $model}')

    aws lambda invoke \
        --function-name "$LAMBDA_NAME" \
        --invocation-type RequestResponse \
        --payload "$PAYLOAD" \
        /dev/null

    echo "Lambda invocation complete."
fi

# Clean up the local file
rm "$LOCAL_TAR_PATH"

echo "Upload complete."
