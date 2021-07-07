# This is the configuration for a base image, used for full setup including
# installing all dependencies.

variable "disk_size_gb" {
  type    = number
  default = 100
}

variable "git_ref" {
  type = string
  default = "origin/main"
}

packer {
  required_plugins {
    googlecompute = {
      version = ">= 0.0.1"
      source  = "github.com/hashicorp/googlecompute"
    }
  }
}

local "timestamp" {
  expression = "${formatdate("YYYY-MM-DD't'hh-mm-ss", timestamp())}"
}

source "googlecompute" "thor-worker-base" {
  project_id            = "moeyens-thor-dev"
  image_name            = "thor-worker-base-${local.timestamp}"
  image_family          = "thor-worker-base"

  source_image_family   = "ubuntu-2004-lts"
  ssh_username          = "packer"
  zone                  = "us-west1-b"
  disk_size             = var.disk_size_gb
  image_description     = "Worker base image for THOR"
  instance_name         = "thor-worker-packerbuild-base"
  service_account_email = "thor-worker@moeyens-thor-dev.iam.gserviceaccount.com"
  scopes = [
    "https://www.googleapis.com/auth/cloud-platform",
  ]
  machine_type          = "e2-standard-8"
}

build {
  sources = ["sources.googlecompute.thor-worker-base"]

  provisioner "file" {
    source      = "thor-worker.service"
    destination = "/tmp/thor-worker.service"
  }
  provisioner "shell" {
    # Execute command with sudo
    execute_command = "chmod +x {{ .Path }}; sudo sh -c '{{ .Vars }} {{ .Path }}'"
    environment_vars = [
      "THOR_GIT_REF=${var.git_ref}"
    ]
    script          = "setup_base_image.sh"
  }
}
