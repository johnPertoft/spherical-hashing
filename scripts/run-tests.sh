#!/usr/bin/env bash

IMAGE="spherical-hashing"

docker build -t ${IMAGE} .
docker run -it --rm \
    -v $(pwd):/workspace \
    ${IMAGE} \
    bash -c "pytest"
