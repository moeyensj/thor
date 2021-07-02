import pika
from typing import Optional

from thor.taskqueue.tasks import Task


class TaskQueueConnection:
    def __init__(self, conn_params: pika.ConnectionParameters, queue_name: str):
        """Create a logical connection to a task queue.

        The connection to the task queue is stateful, and represents a
        connection to a RabbitMQ queue of tasks to be done. Callers should use
        connect() to actually establish a connection, and close() to tear that
        connection down.

        Parameters
        ----------
        conn_params : pika.ConnectionParameters
            Parameters which are used to establish the RabbitMQ connection.
        queue_name : str
            Name of a queue to connect to. The queue will be created if it does
            not already exist.

        Examples
        --------
        >>> import pika
        >>> conn = TaskQueueConnection(
        ...     pika.ConnectionParameters(
        ...        host="localhost",
        ...        port=5762,
        ...        credentials=pika.PlainCredentials(
        ...            username="thor",
        ...            password="supersecret",
        ...        ),
        ...    ),
        ...    "task-queue",
        ...)
        >>> conn.connect()
        >>> task = conn.receve()
        >>> if task is not None:
        ...    handleTask(task)
        >>> conn.close()
        """

        self.connection = None
        self.connection_params = conn_params
        self.channel = None
        self.queue_name = queue_name

    def connect(self):
        """Establish a persistent connection to the RabbitMQ backend. """

        self.connection = pika.BlockingConnection(self.connection_params)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue_name, durable=True)
        self.channel.basic_qos(prefetch_count=1)

    def close(self):
        """
        Destroy an existing connection to the RabbitMQ backend, freeing server and
        client resources.
        """

        self.channel.close()
        self.connection.close()

    def publish(self, task: Task):
        """Submit a Task for execution by a worker.

        The Task will be added to the TaskQueueConnection's queue, and will be
        accepted by the first available worker process listening to that queue.

        Parameters
        ----------
        task : Task
            The task to be exeucted.
        """

        self.channel.basic_publish(
            exchange="",
            routing_key=self.queue_name,
            body=task.to_bytes(),
        )

    def receive(self) -> Optional[Task]:
        """Poll the queue for a task to be done.

        If a task is not available, receive will return None.

        Returns
        -------
        Optional[Task]
            A Task to be done, if one is available. If there is no work to be
            done, receive will return None.
        """

        method, properties, body = self.channel.basic_get(
            queue=self.queue_name,
            auto_ack=False,
        )
        if method is None:
            return None
        return Task.from_msg(self.channel, method, properties, body)
