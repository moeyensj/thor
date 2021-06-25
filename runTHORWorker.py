import argparse
import os

import pika
from google.cloud.storage.client import Client as GCSClient

from thor.taskqueue.client import Worker
from thor.taskqueue.queue import TaskQueueConnection


def parse_args():
    parser = argparse.ArgumentParser(description="Worker process to run THOR tasks")
    parser.add_argument("queue", type=str, help="name of the queue to listen to")
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
    gcs = GCSClient()
    worker = Worker(gcs, queue)
    worker.run_worker_loop(args.poll_interval)


if __name__ == "__main__":
    main()
