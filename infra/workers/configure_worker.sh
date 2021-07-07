#!/bin/bash

set -euo pipefail

# Check out the requested git version
cd /opt/thor
git fetch
git checkout $THOR_GIT_REF

# Update dependencies, if requested
if [ "${UPDATE_DEPS}" = "true" ]; then
    /opt/miniconda3/bin/conda run -n thor_py38 --live-stream pip install -e .
fi

# Update thor queue setting
sed -i "s/THOR_QUEUE=.*/THOR_QUEUE=${THOR_QUEUE}/g" /etc/thor/env

# Enable THOR on system boot
systemctl enable thor-worker
