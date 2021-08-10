#!/bin/bash

echo "running configure_worker.sh"

set -xeo pipefail

# Check out the requested git version
echo "checking out $THOR_GIT_REF"
cd /opt/thor
git fetch
git fetch fork
git checkout $THOR_GIT_REF

# Update dependencies, if requested
if [ "${UPDATE_DEPS}" = "true" ]; then
    echo "updating dependencies"
    /opt/miniconda3/bin/conda run -n thor_py38 --live-stream pip install -e .
fi

# Update thor queue setting
echo "setting thor queue"
sed -i "s/THOR_QUEUE=.*/THOR_QUEUE=${THOR_QUEUE}/g" /etc/thor/env

# Enable THOR on system boot
echo "enabling THOR on boot"
systemctl enable thor-worker.service
