#!/bin/bash

# Build a Docker image containing the software stack build and runtime
# dependencies.
#
# Author: Helio Perroni Filho

SCRIPT_PATH=$(readlink -f "$0")
export SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

cd "$SCRIPT_DIR"
docker build \
  --build-arg uid="$(id -u)" \
  --build-arg gid="$(id -g)" \
  -t elderlab/cstrack:latest .
