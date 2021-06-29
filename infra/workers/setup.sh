#!/bin/bash

##
## Setup script for building a disk image for the taskqueue worker.
##

set -eo pipefail

## Install system dependencies
add-apt-repository universe  # Required to install jq in Ubuntu 18.04.1
apt-get update -y
apt-get install -y git curl systemd jq

## Install Conda
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > /tmp/miniconda_install.sh
bash /tmp/miniconda_install.sh -b -p /opt/miniconda3

## Clone THOR
git clone https://github.com/moeyensj/thor /opt/thor
cd /opt/thor

##### FIXME: Temporary while unmerged
git remote add fork https://github.com/spenczar/thor
git fetch fork
git checkout taskqueues
##### FIXME: End

## Install THOR conda environment
/opt/miniconda3/bin/conda create \
      --yes \
      --name thor_py38 \
      --channel defaults \
      --channel astropy \
      --channel moeyensj \
      --channel conda-forge \
      --file requirements.txt \
      python=3.8
/opt/miniconda3/bin/conda run -n thor_py38 --live-stream pip install -e .

## Install environment file for systemd service
mkdir -p /etc/thor
touch /etc/thor/env

echo "THOR_QUEUE=thor-tasks" >> /etc/thor/env

### Fetch RabbitMQ password
RABBIT_PASSWORD=$(gcloud secrets versions access latest \
                         --secret rabbitmq-credentials | jq '.password' -r)
echo "RABBIT_PASSWORD=${RABBIT_PASSWORD}" >> /etc/thor/env
unset RABBIT_PASSWORD

mv /tmp/thor-worker.service /etc/systemd/system/thor-worker.service
systemctl daemon-reload
service thor-worker start
