#!/usr/bin/env bash
set -e

if [ -z "${PYPI_USER}" ]; then
    echo "Environment variable PYPI_USER is not set."
    exit 1
fi

if [ -z "${PYPI_PWD}" ]; then
    echo "Environment variable PYPI_PWD is not set."
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
    -e PYPI_USER="${PYPI_USER}" \
    -e PYPI_PWD="${PYPI_PWD}" \
    ${IMAGE} \
    bash -c "
        python setup.py sdist bdist_wheel && 
        twine upload -u \"\${PYPI_USER}\" -p \"\${PYPI_PWD}\" --repository-url https://upload.pypi.org/legacy/ dist/*
    "
