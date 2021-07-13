import logging
import argparse
import pandas as pd
import os

import pika
from google.cloud.storage import Client as GCSClient
from google.cloud.pubsub_v1 import PublisherClient
import google.cloud.exceptions

logger = logging.getLogger("thor")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Tracklet-less Heliocentric Orbit Recovery through a queue",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    parser.add_argument("out_dir", type=str, help="destination path for results")
    parser.add_argument(
        "--config", type=str, default=None,
    )
    parser.add_argument(
        "--create-bucket",
        action="store_true",
        help="create the bucket if it does not exist already",
    )
    parser.add_argument(
        "--pubsub_topic",
        type=str,
        default=None,
        help="""
        (optional) The name of a pubsub topic (in 'projects/{proj}/topics/{topic}' format).
        When the job is complete, a message will be published to this topic. The
        message's contents will be JSON serialization of a
        thor.taskqueue.jobs.JobManifest object.
        """,
    )
    parser.add_argument(
        "--create-pubsub-topic",
        action="store_true",
        help="create the pubsub topic if it does not exist already"
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

    # Imports of thor modules are deferred until after argument parsing to avoid
    # numba JIT time if the arguments are invalid or the user asked for --help.
    import thor.utils.logging

    thor.utils.logging.setupLogger("thor")

    from thor.taskqueue.client import Client as TaskQueueClient
    from thor.taskqueue.queue import TaskQueueConnection
    from thor.orbits import Orbits
    from thor.config import Config

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
    queue.connect()

    # Connect to GCS bucket
    gcs = GCSClient()
    if args.create_bucket:
        try:
            gcs.create_bucket(args.bucket)
        except google.cloud.exceptions.Conflict:
            # Bucket already exists.
            pass
    bucket = gcs.bucket(args.bucket)

    # Set up PubSub topic
    if args.pubsub_topic is not None:
        # Validate pubsub topic
        split_topic = args.pubsub_topic.split("/")
        if len(split_topic) != 4 or split_topic[0] != "projects" or split_topic[2] != "topics":
            raise ValueError(
                "--pubsub-topic must match pattern 'projects/{project}/topics/{topic}'"
            )

        if args.create_pubsub_topic:
            try:
                pubsub_client = PublisherClient()
                pubsub_client.create_topic(name=args.pubsub_topic)
            except google.cloud.exceptions.Conflict:
                # Topic already exists.
                pass

    taskqueue_client = TaskQueueClient(bucket, queue)

    manifest = taskqueue_client.launch_job(
        config=config,
        observations=preprocessed_observations,
        orbits=test_orbits
        job_completion_pubsub_topic=args.pubsub_topic,
    )
    taskqueue_client.monitor_job_status(manifest.job_id)
    taskqueue_client.download_results(manifest, args.out_dir)


if __name__ == "__main__":
    main()
