#!/bin/sh
#
# Set environment variables.

export IMAGE_NAME=experiment_template
export CONTAINER_NAME=ExperimentTemplate
export MLFLOW_HOST_PORT=5003
export MLFLOW_CONTAINER_PORT=5003

if [ -e docker/env_dev.sh ]; then
  . docker/env_dev.sh
fi
