#!/usr/bin/env bash
set -e

if [ -z "${PIP_USER}" ]; then
    echo "Environment variable PIP_USER is not set."
    exit 1
fi

if [ -z "${PIP_PWD}" ]; then
    echo "Environment variable PIP_PWD is not set."
    exit 1
fi

# Run from project root.
SCRIPT_DIR=$(dirname $0)
pushd ${SCRIPT_DIR}/..

# Move only relevant files to temp dir.
TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)
cp Dockerfile ${TMPDIR}/Dockerfile
cp LICENSE ${TMPDIR}/LICENSE
cp README.md ${TMPDIR}/README.md
cp setup.py ${TMPDIR}/setup.py
rsync -am --exclude='*_test.py' --exclude='__pycache__' spherical_hashing ${TMPDIR}/

# Build and publish pip package.
pushd ${TMPDIR}
IMAGE="spherical-hashing"
docker build -t ${IMAGE} .
docker run -it --rm \
    -v $(pwd):/workspace \
    -e PIP_USER="${PIP_USER}" \
    -e PIP_PWD="${PIP_PWD}" \
    ${IMAGE} \
    bash -c "
        python setup.py sdist bdist_wheel && 
        twine upload -u \"\${PIP_USER}\" -p \"\${PIP_PWD}\" --repository-url https://upload.pypi.org/legacy/ dist/*
    "
