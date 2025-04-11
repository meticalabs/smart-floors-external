FROM openjdk:slim

# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Basic dependencies
RUN apt-get install -y curl build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget libbz2-dev


FROM python:3.10.14-slim
COPY --from=openjdk:8-jre-slim /usr/local/openjdk-8 /usr/local/openjdk-8

# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Install necessary dependencies for MMS and SageMaker Inference Toolkit
RUN apt-get clean && apt-get update \
     && apt-get install -y \
     build-essential \
     curl \
     vim \
     git-all


# Install MMS, sagemaker inference and other libraries
RUN pip --no-cache-dir install multi-model-server \
                                sagemaker-inference \
                                retrying \
                                joblib \
                                mlflow


ARG APP_WHEEL_FILE

COPY $APP_WHEEL_FILE /home/$APP_WHEEL_FILE

RUN pip install /home/$APP_WHEEL_FILE

# Copy entrypoint script to the image
COPY dockerd-entrypoint.py /usr/local/bin/dockerd-entrypoint.py
RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

RUN mkdir -p /home/model-server/

# Copy the default custom service file to handle incoming data and inference requests
COPY model_handler.py /home/model-server/model_handler.py
COPY config.properties /home/model-server/config.properties
COPY log4j.xml /home/model-server/log4j.xml

ENV JAVA_HOME=/usr/local/openjdk-8

# Add Java to the PATH
ENV PATH=$JAVA_HOME/bin:$PATH

# Define an entrypoint script for the docker image
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]

# Define command to be passed to the entrypoint
CMD ["serve"]
