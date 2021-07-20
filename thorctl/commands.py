from typing import List
import os
import logging
import argparse
import pika

from .worker_pool import WorkerPoolManager
from .sshconn import WorkerPoolSSHConnection
from .autoscaler import Autoscaler

logger = logging.getLogger("thorctl")


def dispatch(parser, args):
    if args.command == "size":
        check_size(args.queue)
    elif args.command == "scale-up":
        scale_up(args.queue, args.n_workers, args.machine_type)
    elif args.command == "destroy":
        destroy(args.queue)
    elif args.command == "logs":
        logs(args.queue)
    elif args.command == "autoscale":
        autoscale(
            args.queue,
            args.max_size,
            args.machine_type,
            args.poll_interval,
            args.rabbit_host,
            args.rabbit_port,
            args.rabbit_username,
            args.rabbit_password,
        )
    elif args.command is None:
        parser.print_usage()
    else:
        raise parser.error("unknown command %s" % args.command)


def parse_args():
    parser = argparse.ArgumentParser("thorctl")

    subparsers = parser.add_subparsers(dest="command")

    scale_up = subparsers.add_parser("scale-up", help="add more workers")
    scale_up.add_argument("--n-workers", type=int, help="end size to arrive at")
    scale_up.add_argument(
        "--machine-type",
        type=str,
        default="e2-standard-8",
        help="Compute Engine machine type",
    )
    scale_up.add_argument(
        "queue", type=str, help="name of the queue that workers will listen to"
    )

    check_size = subparsers.add_parser(
        "size", help="look up the current number of workers"
    )
    check_size.add_argument(
        "queue", type=str, help="name of the queue that workers are listening to",
    )

    destroy = subparsers.add_parser(
        "destroy", help="destroy all workers on a queue, even if they are doing work"
    )
    destroy.add_argument(
        "queue", type=str, help="name of the queue that workers are listening to"
    )

    logs = subparsers.add_parser("logs", help="stream logs from the workers")
    logs.add_argument("queue", type=str, help="name of the queue")

    autoscale = subparsers.add_parser(
        "autoscale", help="monitor a queue and automatically scale it up to handle load"
    )
    autoscale.add_argument(
        "queue",
        action="append",
        help="name of queue to monitor. Can be specified multiple times.",
    )
    autoscale.add_argument(
        "--max-size",
        type=int,
        default=16,
        help="maximum number of workers to run at once, per queue",
    )
    autoscale.add_argument(
        "--machine-type",
        type=str,
        default="e2-standard-8",
        help="Compute Engine machine type",
    )
    autoscale.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="time interval in seconds to check sizes",
    )
    autoscale.add_argument(
        "--rabbit-host",
        type=str,
        default="rabbit.c.moeyens-thor-dev.internal",
        help="hostname of the rabbit broker",
    )
    autoscale.add_argument(
        "--rabbit-port", type=int, default=5672, help="port of the rabbit broker"
    )
    autoscale.add_argument(
        "--rabbit-username",
        type=str,
        default="thor",
        help="username to connect with to the rabbit broker",
    )
    autoscale.add_argument(
        "--rabbit-password",
        type=str,
        default="$RABBIT_PASSWORD env var",
        help="password to connect with to the rabbit broker",
    )

    return parser, parser.parse_args()


def scale_up(queue_name: str, n_workers: int, instance_type: str):
    manager = WorkerPoolManager(queue_name)
    current_num = manager.current_num_workers()
    n_to_add = n_workers - current_num
    if n_to_add > 0:
        logger.info(
            "%s has %d workers, scaling up to %d", queue_name, current_num, n_workers
        )
        manager.launch_workers(n_to_add, instance_type)
    else:
        logger.warning(
            "%s has %d workers already, doing nothing", queue_name, current_num
        )


def check_size(queue_name: str):
    manager = WorkerPoolManager(queue_name)
    print(manager.current_num_workers())


def destroy(queue_name: str):
    manager = WorkerPoolManager(queue_name)
    current_num = manager.current_num_workers()
    if current_num == 0:
        logger.warning("queue has no workers")
        return
    response = input(
        f"{queue_name} has {current_num} workers. Really destroy them? (yes/no)"
    )
    if response != "yes":
        print("not proceeding, since you didn't say yes")
    manager.terminate_all_workers()


def logs(queue_name: str):
    manager = WorkerPoolManager(queue_name)
    current_num = manager.current_num_workers()
    if current_num == 0:
        logger.warning("queue has no workers currently")
    conn = WorkerPoolSSHConnection(manager)
    conn.stream_logs()


def autoscale(
    queues: List[str],
    max_size: int,
    machine_type: str,
    poll_interval: int,
    rabbit_host: str,
    rabbit_port: int,
    rabbit_username: str,
    rabbit_password: str,
):

    if rabbit_password == "$RABBIT_PASSWORD env var":
        rabbit_password = os.environ["RABBIT_PASSWORD"]

    rabbit_params = pika.ConnectionParameters(
        host=rabbit_host,
        port=rabbit_port,
        credentials=pika.PlainCredentials(
            username=rabbit_username,
            password=rabbit_password,
        )
    )
    scaler = Autoscaler(rabbit_params, queues, max_size, machine_type)
    scaler.run(poll_interval)
