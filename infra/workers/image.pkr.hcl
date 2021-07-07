variable "queue" {
  # The name of the RabbitMQ Queue to listen to.
  type = string
  default = "production-tasks"
}

variable "git_ref" {
  type = string
  default = "origin/main"
}

variable "update_deps" {
  type = bool
  default = false
}

local "timestamp" {
  expression = "${formatdate("YYYY-MM-DD't'hh-mm-ss", timestamp())}"
}

source "googlecompute" "thor-worker" {
  project_id = "moeyens-thor-dev"
  image_family = "thor-worker-${var.queue}"
  image_name   = "thor-worker-${var.queue}-${local.timestamp}"
  source_image_family = "thor-worker-base"

  ssh_username          = "packer"
  zone                  = "us-west1-b"
  disk_size             = "100"
  image_description     = "Worker base image for THOR"
  instance_name         = "thor-worker-packerbuild-${var.queue}"
  service_account_email = "thor-worker@moeyens-thor-dev.iam.gserviceaccount.com"
  scopes = [
    "https://www.googleapis.com/auth/cloud-platform",
  ]
  machine_type          = "e2-standard-8"
}

build {
  sources = ["sources.googlecompute.thor-worker"]

  provisioner "shell" {
    environment_vars = [
      "UPDATE_DEPS=${var.update_deps}",
      "THOR_QUEUE=${var.queue}",
      "THOR_GIT_REF=${var.git_ref}",
    ]
    # Execute command with sudo
    execute_command = "chmod +x {{ .Path }}; sudo sh -c '{{ .Vars }} {{ .Path }}'"
    script = "configure_worker.sh"
  }
}
