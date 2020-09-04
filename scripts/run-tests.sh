#!/usr/bin/env bash

# Run from project root.
SCRIPT_DIR=$(dirname $0)
pushd ${SCRIPT_DIR}/..

# Run unit tests.
IMAGE="spherical-hashing"
docker build -t ${IMAGE} .
docker run -it --rm \
    -v $(pwd):/workspace \
    ${IMAGE} \
    bash -c "pytest"
