#!/bin/bash

set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate thor_py38

source /etc/thor/env

thorctl autoscale ${THOR_AUTOSCALED_QUEUES}
