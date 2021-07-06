from typing import Sequence
import time
import pika
import logging
from .worker_pool import WorkerPoolManager
from thor.taskqueue.queue import TaskQueueConnection


logger = logging.getLogger("thorctl")


class Autoscaler:
    def __init__(
        self,
        rabbit_params: pika.ConnectionParameters,
        queues: Sequence[str],
        max_size: int,
        machine_type: str,
    ):
        self._managers = []
        self._queue_connections = []
        for q in queues:
            self._managers.append(WorkerPoolManager(q))
            q_conn = TaskQueueConnection(rabbit_params, q)
            q_conn.connect()
            self._queue_connections.append(q_conn)

        self.max_size = max_size
        self.machine_type = machine_type

    def run(self, poll_interval: int):
        while True:
            for i, mgr in enumerate(self._managers):
                worker_pool_size = mgr.current_num_workers()
                if worker_pool_size >= self.max_size:
                    logger.info("scaling %s: already at max size", mgr.queue_name)
                    continue

                queue_conn = self._queue_connections[i]
                queue_size = queue_conn.size()
                if queue_size <= worker_pool_size:
                    logger.info(
                        "scaling %s: %d messages, %d workers, so doing nothing",
                        mgr.queue_name,
                        queue_size,
                        worker_pool_size,
                    )
                    continue

                scale_to = min(self.max_size, queue_size)
                logger.info(
                    "scaling %s: %d messages, scaling up to %d workers",
                    mgr.queue_name,
                    queue_size,
                    scale_to,
                )
                mgr.launch_workers(scale_to - worker_pool_size, self.machine_type)

            time.sleep(poll_interval)
