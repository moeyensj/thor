import argparse
import os

import pika
from google.cloud.storage.client import Client as GCSClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Worker process to run THOR tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    parser.add_argument(
        "--idle-shutdown-timeout",
        type=int,
        default=60,
        help="""maximum idle time in seconds. If negative, continue forever. If this time
        elapses, the program exits, and on Google Compute Engine it also
        terminates the running instance.""",
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

    from thor.taskqueue.client import Worker
    from thor.taskqueue.queue import TaskQueueConnection

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
    gcs = GCSClient()
    worker = Worker(gcs, queue)
    worker.run_worker_loop(args.poll_interval, args.idle_shutdown_timeout)


if __name__ == "__main__":
    main()
