#!/bin/bash

# This script generates an empty model artifact, uploads it to a specified S3 location,
# and optionally updates the allocator service.

set -e

# Default values
UPDATE_ALLOCATOR=false
ALLOCATOR_URI=""
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
        --allocator-uri)
            ALLOCATOR_URI="$2"
            shift # past argument
            shift # past value
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
    if [ -z "$ALLOCATOR_URI" ] || [ -z "$MODEL_VERSION" ]; then
        echo "Error: --allocator-uri and --model-version are required when using --update-allocator"
        exit 1
    fi

    ENDPOINT_NAME="bid-floor-${MODEL_VERSION//./-}"
    REFERENCE="default_bid_floor"

    echo "Updating allocator service at $ALLOCATOR_URI"
    echo "Endpoint: $ENDPOINT_NAME, Model: $ARTIFACT_NAME, Reference: $REFERENCE"

    PAYLOAD=$(cat <<-END
    {
        "reference": "${REFERENCE}",
        "endpointName": "${ENDPOINT_NAME}",
        "modelName": "${ARTIFACT_NAME}"
    }
	END
	)

    curl -X PUT -H "Content-Type: application/json" -d "$PAYLOAD" "$ALLOCATOR_URI"
fi

# Clean up the local file
rm "$LOCAL_TAR_PATH"

echo "Upload complete."
