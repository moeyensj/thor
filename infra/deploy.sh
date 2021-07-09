#!/bin/bash

set -euo pipefail

green=$(tput setaf 2)
normal=$(tput sgr0)

say_green() {
    printf "  ${green}${1}${normal}\n"
}

if [[ $# -gt 0 ]]; then
    VERSION=$1
else
    echo "no version provided, inferring it..."
    VERSION=$(git describe HEAD)
fi

say_green "Releasing version ${VERSION}: Does this version look correct? (yes/no)"
read response

if [[ $response != "yes" ]]; then
    echo "aborting"
    exit 1
fi

pushd workers

say_green "building base worker image..."
packer build \
       -var git_ref="${VERSION}" \
       base-image.pkr.hcl

say_green "building base worker image..."
packer build \
       -var queue=production-tasks \
       -var git_ref="${VERSION}" \
       image.pkr.hcl

popd

pushd autoscaler
say_green "building autoscaler image..."
packer build \
       -var git_ref="${VERSION}" \
       image.pkr.hcl

say_green "updating autoscaler..."
./launch.sh

popd
