# workers #

This directory contains a [Packer](https://www.packer.io) configuration for
building the disk image for a THOR task queue worker.

## Updating the worker image

Install Packer, authenticate to Google (`gcloud auth application-default
login`), and run it thusly:

```sh
packer init .
packer build image.pkr.hcl
```

This will make a new disk image for the '`thor-workers-prod-tasks`' family, so
instances can be launched from it.

### Updating to a specific git reference:

You can update to a specific git reference that has been pushed - anything that
you can 'git checkout' on a clean machine. For example, you can specify a branch
name:

```sh
packer build -var "git_ref=cool_feature_branch image.pkr.hcl
```

Or a tag:

```sh
packer build -var "git_ref=v1.1.0" image.pkr.hcl
```

Or a SHA:

```sh
packer build -var "git_ref=b47e25aedaff6f15abf048fc7ff0b895f4176a9f" image.pkr.hcl
```

Or an abbreviated SHA:

```sh
packer build -var "git_ref=b47e25" image.pkr.hcl
```

## Updating dependencies

The worker image configuration is designed to be fast. It lives on top of a
"base image" which installs all the THOR dependencies. This takes a lot longer,
so you only need to do it when dependencies are updated.

To do it, specify the base-image configuration:

```sh
packer build base-image.pkr.hcl
```


## Notes on the disk image ##

Conda is installed in /opt/miniconda3.

THOR is installed in /opt/thor.

A Conda environment named `thor_py38` has THOR (and all its dependencies)
already installed.

The THOR worker runs as a systemd service, which is defined in
[./thor-worker.service](./thor-worker.service).

The Queue that the worker listens to is set with an environment file in /etc/thor/env. The file looks like this by default:
```
THOR_QUEUE=unset
RABBIT_PASSWORD=<...>
```

The password is loaded by calling out to Google Cloud SecretManager, looking up
the `rabbitmq-credentials` secret.

The THOR_QUEUE line gets overwritten by the `./configure_worker.sh` script.
