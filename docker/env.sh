#!/bin/sh
#
# Set environment variables.

export IMAGE_NAME=video_prediction
export CONTAINER_NAME=VideoPrediction

if [ -e docker/env_dev.sh ]; then
  . docker/env_dev.sh
fi
