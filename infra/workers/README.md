# workers #

This directory contains a [Packer](https://www.packer.io) configuration for
building the disk image for a THOR task queue worker.

Install Packer, authenticate to Google (`gcloud auth application-default
login`), and run it thusly:

```sh
packer init .
packer build .
```

This will create a disk image named `thor-worker-taskqueues`.

## Notes on the disk image ##

Conda is installed in /opt/miniconda3.

THOR is installed in /opt/thor.

A Conda environment named `thor_py38` has THOR (and all its dependencies)
already installed.

The THOR worker runs as a systemd service, which is defined in
[./thor-worker.service](./thor-worker.service).

The Queue that the worker listens to is set with an environment file in /etc/thor/env. The file looks like this by default:
```
THOR_QUEUE=thor-tasks
RABBIT_PASSWORD=<...>
```

The password is loaded by calling out to Google Cloud SecretManager, looking up
the `rabbitmq-credentials` secret.

To change the Queue, edit that file, and then run `service thor-worker restart`.
