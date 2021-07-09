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

variable "autoscaled_queues" {
  type = list(string)
  default = [
    "production-tasks"
  ]
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

source "googlecompute" "thor-autoscaler" {
  project_id            = "moeyens-thor-dev"
  image_name            = "thor-autoscaler-${local.timestamp}"
  image_family          = "thor-autoscaler"

  source_image_family   = "ubuntu-2004-lts"
  ssh_username          = "packer"
  zone                  = "us-west1-a"
  disk_size             = var.disk_size_gb
  image_description     = "Autoscaler base image for THOR"
  instance_name         = "thor-autoscaler-packerbuild"
  service_account_email = "thor-autoscaler@moeyens-thor-dev.iam.gserviceaccount.com"
  scopes = [
    "https://www.googleapis.com/auth/cloud-platform",
  ]
  machine_type          = "e2-medium"
}

build {
  sources = ["sources.googlecompute.thor-autoscaler"]

  provisioner "file" {
    source      = "thor-autoscaler.service"
    destination = "/tmp/thor-autoscaler.service"
  }
  provisioner "file" {
    source      = "start_autoscaler.sh"
    destination = "/tmp/start_autoscaler.sh"
  }
  provisioner "shell" {
    # Execute command with sudo
    execute_command = "chmod +x {{ .Path }}; sudo sh -x -c '{{ .Vars }} {{ .Path }}'"
    environment_vars = [
      "THOR_GIT_REF=${var.git_ref}",
      "THOR_AUTOSCALED_QUEUES=${join(" ", var.autoscaled_queues)}",
    ]
    script          = "install.sh"
  }
}
