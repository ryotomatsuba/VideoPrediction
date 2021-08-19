#!/bin/sh
#
# Run the docker container.

. docker/env.sh
docker run \
  -dit \
  --gpus all \
  -v $PWD:/workspace \
  -v /home/data/ryoto/:/data \
  --name $CONTAINER_NAME \
  --rm \
  --shm-size=2g \
  $IMAGE_NAME
