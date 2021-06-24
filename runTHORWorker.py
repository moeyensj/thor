import os

import pika
from google.cloud.storage.client import Client as GCSClient

from thor.taskqueue.client import Client
from thor.taskqueue.queue import TaskQueueConnection


def main():
    queue = TaskQueueConnection(
        pika.ConnectionParameters(
            host="rabbit.c.moeyens-thor-dev.internal",
            port=5672,
            credentials=pika.PlainCredentials(
                username="thor", password=os.environ["RABBIT_PASSWORD"],
            ),
        ),
        "thor-tasks",
    )
    bucket = GCSClient().bucket("thor_tasks_v1")
    client = Client(bucket, queue)
    client.run_worker_loop()


if __name__ == "__main__":
    main()
