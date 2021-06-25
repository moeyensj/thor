import logging
import uuid
from typing import Sequence
import time
import tempfile

import pandas as pd

from google.cloud.storage import Bucket
from google.cloud.storage import Client as GCSClient

from thor.main import runTHOROrbit
from thor.config import Configuration
from thor.orbits import Orbits
from thor.taskqueue.queue import TaskQueueConnection
from thor.taskqueue.tasks import Task, upload_job_inputs, get_task_status
from thor.taskqueue.jobs import (
    JobManifest,
    upload_job_manifest,
    download_job_manifest,
    get_job_statuses,
)


logger = logging.getLogger("thor")


class Client:
    def __init__(
        self, bucket: Bucket, queue: TaskQueueConnection,
    ):
        self.bucket = bucket
        self.queue = queue

    def launch_job(
        self, config: Configuration, observations: pd.DataFrame, orbits: Orbits,
    ) -> Sequence[Task]:

        logger.info("launching new job")

        job_id = str(uuid.uuid1())
        logger.info("generated job ID: %s", job_id)

        logger.info("uploading job inputs")
        upload_job_inputs(self.bucket, job_id, config, observations)

        manifest = JobManifest.create(job_id)
        for orbit in orbits.split(1):
            task = Task.create(job_id, self.bucket, orbit)
            logger.info("created task (id=%s)", task.task_id)

            self.queue.publish(task)
            manifest.append(orbit, task)

        logger.info("uploading job manifest")
        upload_job_manifest(self.bucket, manifest)

        return manifest

    def monitor_job_status(self, job_id: str, poll_interval=10):
        logger.debug("downloading job manifest")
        manifest = download_job_manifest(self.bucket, job_id)
        logger.info("monitoring status of %d tasks", len(manifest.task_ids))

        tasks_pending = True
        while tasks_pending:
            tasks_pending = False
            statuses = get_job_statuses(self.bucket, manifest)
            for i, task_id in enumerate(manifest.task_ids):
                orbit_ids = manifest.orbit_ids[i]
                status = statuses[task_id]
                state = status["state"]
                worker = status["worker"]
                line = "\t".join((str(orbit_ids), task_id, status, worker))
                logger.info(line)

                if state not in ("succeeded", "failed"):
                    tasks_pending = True
            time.sleep(poll_interval)

    def get_job_manifest(self, job_id):
        return download_job_manifest(self.bucket, job_id)

    def get_job_statuses(self, manifest):
        return get_job_statuses(self.bucket, manifest)


class Worker:
    def __init__(self, gcs: GCSClient, queue: TaskQueueConnection):
        self.gcs = gcs
        self.queue = queue

    def run_worker_loop(self, poll_interval: float):
        for task in self.poll_for_tasks():
            self.handle_task(task)

    def poll_for_tasks(self, poll_interval: float = 5.0, limit: int = -1):
        i = 0
        while True:
            task = self.queue.receive()
            if task is None:
                logger.debug("no tasks in queue")
                time.sleep(poll_interval)
            else:
                bucket = self.gcs.bucket(task.bucket)
                task.mark_in_progress(bucket)
                logger.info(
                    "received task job_id=%s task_id=%s", task.job_id, task.task_id
                )
                yield task
            i += 1
            if limit >= 0:
                if i >= limit:
                    break

    def handle_task(self, task: Task):
        try:
            bucket = self.gcs.bucket(task.bucket)
            config, observations, orbit = task.download_inputs(bucket)
            logger.debug("downloaded inputs")

            out_dir = tempfile.TemporaryDirectory(
                prefix=f"thor_{task.job_id}_{task.task_id}",
            ).name

            logger.info(
                "beginning execution for job %s, task %s", task.job_id, task.task_id
            )

            runTHOROrbit(
                observations,
                orbit,
                range_shift_config=config.RANGE_SHIFT_CONFIG,
                cluster_link_config=config.CLUSTER_LINK_CONFIG,
                iod_config=config.IOD_CONFIG,
                od_config=config.OD_CONFIG,
                odp_config=config.ODP_CONFIG,
                out_dir=out_dir,
                if_exists="erase",
                logging_level=logging.INFO,
            )
            task.mark_success(bucket, out_dir)
            return out_dir
        except Exception as e:
            task.mark_failure(bucket, out_dir, e)
