#!/bin/bash

gcloud compute instances create thor-autoscaler \
       --create-disk="image-family=thor-autoscaler,boot=yes,size=100GB,auto-delete=yes" \
       --service-account="thor-autoscaler" \
       --scopes=cloud-platform \
       --zone=us-west1-a \
