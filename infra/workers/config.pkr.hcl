variable "machine_type" {
  type    = string
  default = "e2-standard-8"
}

variable "disk_size_gb" {
  type    = number
  default = 100
}

packer {
  required_plugins {
    googlecompute = {
      version = ">= 0.0.1"
      source  = "github.com/hashicorp/googlecompute"
    }
  }
}

source "googlecompute" "thor-worker" {
  project_id            = "moeyens-thor-dev"
  source_image_family   = "ubuntu-1804-lts"
  ssh_username          = "packer"
  zone                  = "us-west1-b"
  disk_size             = var.disk_size_gb
  image_name            = "thor-worker-taskqueues"
  image_description     = "Worker base image for THOR"
  instance_name         = "thor-worker-packerbuild-${uuidv4()}"
  service_account_email = "thor-worker@moeyens-thor-dev.iam.gserviceaccount.com"
  scopes = [
    "https://www.googleapis.com/auth/cloud-platform",
  ]
  machine_type          = var.machine_type
}

build {
  sources = ["sources.googlecompute.thor-worker"]

  provisioner "file" {
    source      = "thor-worker.service"
    destination = "/tmp/thor-worker.service"
  }
  provisioner "shell" {
    environment_vars = [
      "GIT_REF=${var.git_ref}"
    ]
    # Execute command with sudo
    execute_command = "chmod +x {{ .Path }}; sudo sh -c '{{ .Vars }} {{ .Path }}'"
    script          = "setup.sh"
  }
}
