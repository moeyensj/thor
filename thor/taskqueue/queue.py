import pika
from typing import Optional

from . import tasks


class TaskQueueConnection:
    def __init__(self, host, port, username, password, queue_name: str):
        self.connection = None
        self.connection_params = pika.ConnectionParameters(
            host=host,
            port=port,
            credentials=pika.PlainCredentials(
                username=username, password=password,
            ),
        )
        self.channel = None
        self.queue_name = queue_name

    def connect(self):
        self.connection = pika.BlockingConnection(self.connection_params)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue_name, durable=True)
        self.channel.basic_qos(prefetch_count=1)

    def close(self):
        self.channel.close()
        self.connection.close()

    def publish(self, task: tasks.Task):
        self.channel.basic_publish(
            exchange="",
            routing_key=self.queue_name,
            body=task.to_bytes(),
        )

    def receive(self) -> Optional[tasks.Task]:
        method, properties, body = self.channel.basic_get(
            queue=self.queue_name,
            auto_ack=False,
        )
        if method is None:
            return None
        return tasks.Task.from_msg(self.channel, method, properties, body)
