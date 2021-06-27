import logging
import sys
import uuid
from typing import Sequence, Mapping, Iterator
import time
import tempfile

import pandas as pd

from google.cloud.storage import Bucket
from google.cloud.storage import Client as GCSClient

from thor.main import runTHOROrbit
from thor.config import Configuration
from thor.orbits import Orbits
from thor.taskqueue.queue import TaskQueueConnection
from thor.taskqueue.tasks import (
    Task,
    TaskStatus,
    TaskState,
    upload_job_inputs,
    get_task_status,
    get_task_statuses,
    download_task_outputs,
)
from thor.taskqueue.jobs import (
    JobManifest,
    upload_job_manifest,
    download_job_manifest,
)


logger = logging.getLogger("thor")


class Client:
    """
    Client is an API client for submitting jobs to a THOR task queue.
    """
    def __init__(
        self, bucket: Bucket, queue: TaskQueueConnection,
    ):
        """Create a new client.

        Parameters
        ----------
        bucket : Bucket
            The Google Storage Bucket which will host job inputs, outputs, and
            status. The bucket should already exist.
        queue : TaskQueueConnection
            The queue where new jobs should be placed. The queue connection should
            already be connected.
        """
        self.bucket = bucket
        self.queue = queue

    def launch_job(
        self, config: Configuration, observations: pd.DataFrame, orbits: Orbits,
    ) -> JobManifest:
        """Submit a new job for execution.

        The job will be handled by the first available worker which is listening
        on the same queue that the Client submitted to.

        Parameters
        ----------
        config : Configuration
            The THOR configuration that task executors should load.
        observations : pd.DataFrame
            A dataframe of preprocessed observations in the format expected by
            thor.main.runTHOR.
        orbits : Orbits
            The test orbits to be used by task executors.

        Returns
        -------
        JobManifest
            A manifest document which lists task IDs and the job ID. These
            persistent identifiers can be used to retrieve statuses and results.

        """
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

    def monitor_job_status(self, job_id: str, poll_interval: float = 10):
        """Poll for task status updates and log them until all tasks are complete.

        This monitoring loop runs continuously until all tasks have status
        "succeeded" or "failed". It logs outputs using the thor logger at info
        level.

        Parameters
        ----------
        job_id : str
            The ID of the job to monitor.
        poll_interval : float
            The time between status checks in seconds.
        """

        logger.debug("downloading job manifest")
        manifest = download_job_manifest(self.bucket, job_id)
        logger.info("monitoring status of %d tasks", len(manifest.task_ids))

        tasks_pending = True
        while tasks_pending:
            tasks_pending = False
            statuses = self.get_job_statuses(manifest)
            for i, task_id in enumerate(manifest.task_ids):
                status = statuses[task_id]
                line = "\t".join(
                    ("task=" + task_id, "state=" + status.state, "worker=" + status.worker)
                )
                logger.info(line)

                if not status.state.completed():
                    tasks_pending = True
            if tasks_pending:
                time.sleep(poll_interval)

    def get_job_manifest(self, job_id: str) -> JobManifest:
        """
        Fetch the JobManifest for a specific job.

        Parameters
        ----------
        job_id : str
            The ID of the job to get.

        Returns
        -------
        JobManifest
            The JobManifest of the job.
        """
        return download_job_manifest(self.bucket, job_id)

    def get_task_statuses(self, manifest: JobManifest) -> Mapping[str, TaskStatus]:
        """Download the status of all tasks in the manifest.

        Parameters
        ----------
        manifest : JobManifest
            A Manifest holding identifiers for all tasks in the job. Can be
            retrieved with get_job_manifest.

        Returns
        -------
        Mapping[str, TaskStatus]
            A mapping from task IDs to their status.
        """

        return get_task_statuses(self.bucket, manifest)

    def download_results(self, manifest: JobManifest, path: str):
        """
        Download all the results from the job.

        Results are downloaded for all tasks. They are placed in a directory
        relative to path.

        Each task's outputs are placed in a separate subdirectory, in
        {path}/tasks/task-{task_id}/outputs. It is up to the caller to combine
        task outputs as they see fit.

        Parameters
        ----------
        manifest : JobManifest
            A manifest holding identifiers for all tasks in the job. Can be
            retrieved with get_job_manifest.
        path : str
            A local directory where outputs should be placed.
        """

        logger.info("downloading results to %s", path)
        for task_id in manifest.task_ids:
            task_status = get_task_status(self.bucket, manifest.job_id, task_id)
            if task_status.state.completed():
                logger.info("downloading results for task=%s", task_id)
                download_task_outputs(path, self.bucket, manifest.job_id, task_id)
            else:
                logger.info(
                    "not downloading results for task=%s because it has state=%s",
                    task_id,
                    task_status.state,
                )


class Worker:
    """
    Represents a handler for tasks in a THOR task queue, executing THOR and
    putting results into a GCS bucket.
    """
    def __init__(self, gcs: GCSClient, queue: TaskQueueConnection):
        """Creates a new worker.

        The worker's receive loop is not set up automatically. The caller should
        call run_worker_loop when they are ready to handle tasks.

        Parameters
        ----------
        gcs : GCSClient
            A Google Cloud Storage client.
        queue : TaskQueueConnection
            A queue to listen to. The queue should be connected before calling
            run_worker_loop.

        """

        self.gcs = gcs
        self.queue = queue

    def run_worker_loop(self, poll_interval: float):
        """Block forever, handling new tasks in a loop.

        Parameters
        ----------
        poll_interval : float
            The time, in seconds, between successive checks for work to be done.
        """

        for task in self.poll_for_tasks(poll_interval=poll_interval):
            self.handle_task(task)

    def poll_for_tasks(self, poll_interval: float = 5.0, limit: int = -1) -> Iterator[Task]:
        """
        Blocks forever, checking for new tasks to be done in a loop and yielding
        them whenever they are available.

        Each task should be handled fully, and then marked either as a success
        or failure, before retrieving the next value from the iterator.

        Tasks are automatically marked as "in_progress" when yielded from the
        iterator.

        Parameters
        ----------
        poll_interval : float
            The time, in seconds, between successive checks for work to be done.
        limit : int
            The maximum number of times to check for work. If negative, check
            forever.

        Returns
        -------
        Iterator[Task]
            An infinite iterator of tasks to be done.
        """

        logger.info("starting to poll for tasks")
        i = 0
        while True:
            task = self.queue.receive()
            if task is None:
                logger.info("no tasks in queue")
                time.sleep(poll_interval)
            else:
                logger.info(
                    "received task job_id=%s task_id=%s", task.job_id, task.task_id
                )
                bucket = self.gcs.bucket(task.bucket)
                task.mark_in_progress(bucket)
                yield task
            i += 1
            if limit >= 0:
                if i >= limit:
                    break

    def handle_task(self, task: Task):
        """
        Run THOR on a single Task.

        Downloads all inputs into a temporary directory and uploads them when
        done.

        Blocks until the THOR execution completes. If it errors, the exception
        is caught and uploaded in the result set.

        Parameters
        ----------
        task : Task
            The task to execute.

        """
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
            logger.error("task %s failed", task_id, exc_info=sys.exc_info)
            task.mark_failure(bucket, out_dir, e)
