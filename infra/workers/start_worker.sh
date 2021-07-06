#!/bin/bash

set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate thor_py38

source /etc/thor/env

python /opt/thor/runTHORWorker.py \
       --idle-shutdown-timeout=60 \
       ${THOR_QUEUE}
