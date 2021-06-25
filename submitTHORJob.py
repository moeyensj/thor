import logging
import argparse
import pandas as pd
import os

import pika
from google.cloud.storage import Client as GCSClient
import google.cloud.exceptions

from thor.taskqueue.client import Client as TaskQueueClient
from thor.taskqueue.queue import TaskQueueConnection
from thor.orbits import Orbits
from thor.config import Config


logger = logging.getLogger("thor")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Tracklet-less Heliocentric Orbit Recovery through a queue"
    )
    parser.add_argument(
        "preprocessed_observations", type=str, help="Preprocessed observations."
    )
    parser.add_argument("test_orbits", type=str, help="Path to test orbits.")
    parser.add_argument(
        "bucket",
        type=str,
        help="name of the Google Cloud Storage bucket to use to hold inputs and outputs",
    )
    parser.add_argument(
        "queue", type=str, help="name of the queue to submit the job to"
    )
    parser.add_argument(
        "--config", type=str, default=None,
    )
    parser.add_argument(
        "--create-bucket",
        type=bool,
        default=False,
        help="create the bucket if it does not exist already",
    )
    parser.add_argument(
        "--rabbit-host",
        type=str,
        default="rabbit.c.moeyens-thor-dev.internal",
        help="hostname of the rabbit broker",
    )
    parser.add_argument(
        "--rabbit-port", type=int, default=5672, help="port of the rabbit broker"
    )
    parser.add_argument(
        "--rabbit-username",
        type=str,
        default="thor",
        help="username to connect with to the rabbit broker",
    )
    parser.add_argument(
        "--rabbit-password",
        type=str,
        default="$RABBIT_PASSWORD env var",
        help="password to connect with to the rabbit broker",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="time in seconds between checking whether there are more tasks available",
    )
    args = parser.parse_args()
    if args.rabbit_password == "$RABBIT_PASSWORD env var":
        args.rabbit_password = os.environ["RABBIT_PASSWORD"]

    return args


def main():
    args = parse_args()
    if not isinstance(args.config, str):
        config = Config
    else:
        config = Config.fromYaml(args.config)

    # Read observations
    preprocessed_observations = pd.read_csv(
        args.preprocessed_observations, index_col=False, dtype={"obs_id": str}
    )

    # Read test orbits
    test_orbits = Orbits.from_csv(args.test_orbits)

    # Connect to Rabbit
    queue = TaskQueueConnection(
        pika.ConnectionParameters(
            host=args.rabbit_host,
            port=args.rabbit_port,
            credentials=pika.PlainCredentials(
                username=args.rabbit_username, password=args.rabbit_password,
            ),
        ),
        args.queue,
    )

    # Connect to GCS bucket
    gcs = GCSClient()
    if args.create_bucket:
        try:
            gcs.create_bucket(args.bucket)
        except google.cloud.exceptions.Conflict:
            # Bucket already exists.
            pass
    bucket = gcs.bucket(args.bucket)
    taskqueue_client = TaskQueueClient(bucket, queue)

    manifest = taskqueue_client.launch_job(
        config, preprocessed_observations, test_orbits
    )
    taskqueue_client.monitor_job_status(manifest.job_id)


if __name__ == "__main__":
    main()
