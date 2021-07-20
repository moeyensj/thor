#!/bin/bash

##
## Setup script for building a disk image for the taskqueue worker.
##

set -xeo pipefail

## Install system dependencies
add-apt-repository universe  # Required to install jq in Ubuntu 18.04.1
apt-get update -y
apt-get install -y git curl systemd

curl -L https://github.com/stedolan/jq/releases/download/jq-1.5/jq-linux64 > /usr/bin/jq
chmod +x /usr/bin/jq

## Install Conda
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > /tmp/miniconda_install.sh
bash /tmp/miniconda_install.sh -b -p /opt/miniconda3

## Clone THOR
git clone https://github.com/moeyensj/thor /opt/thor
cd /opt/thor

##### FIXME: Temporary while unmerged
git remote add fork https://github.com/spenczar/thor
git fetch fork
##### FIXME: End

## Check out requested git ref
git checkout $THOR_GIT_REF

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

echo "THOR_QUEUE=unset" >> /etc/thor/env

### Fetch RabbitMQ password
RABBIT_PASSWORD=$(gcloud secrets versions access latest \
                         --secret rabbitmq-credentials | jq '.password' -r)
echo "RABBIT_PASSWORD=${RABBIT_PASSWORD}" >> /etc/thor/env
unset RABBIT_PASSWORD

## Install start script
mv /tmp/start_worker.sh /etc/thor/start_worker.sh
chmod +x /etc/thor/start_worker.sh

# Put service definition in place - but don't enable it yet, because we don't
# have an actual queue name.
mv /tmp/thor-worker.service /etc/systemd/system/thor-worker.service
systemctl daemon-reload
