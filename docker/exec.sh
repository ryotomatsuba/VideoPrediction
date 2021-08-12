#!/bin/sh
#
# Run zsj shell in the docker container.

. docker/env.sh
docker exec \
  -it \
  $CONTAINER_NAME sudo zsh