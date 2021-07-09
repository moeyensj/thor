#!/bin/bash

echo "Bringing down existing autoscaler..."
gcloud compute instances delete thor-autoscaler
echo "Bringing up new one..."
gcloud compute instances create thor-autoscaler \
       --create-disk="image-family=thor-autoscaler,boot=yes,size=100GB,auto-delete=yes" \
       --service-account="thor-autoscaler@moeyens-thor-dev.iam.gserviceaccount.com" \
       --scopes=cloud-platform \
       --zone=us-west1-a \
       --machine-type=e2-small
