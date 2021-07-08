import logging
import sys
import uuid
from typing import Mapping, Iterator, Optional
import time
import tempfile

import pandas as pd

from google.cloud.storage import Bucket
from google.cloud.storage import Client as GCSClient
from google.cloud.pubsub_v1 import PublisherClient

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
    set_task_status,
    get_task_statuses,
    download_task_outputs,
)
from thor.taskqueue.jobs import (
    JobManifest,
    upload_job_manifest,
    download_job_manifest,
    mark_task_done_in_manifest,
    announce_job_done,
)
from thor.taskqueue import compute_engine


logger = logging.getLogger("thor")


class Client:
    """
    Client is an API client for submitting jobs to a THOR task queue.
    """

    def __init__(self, bucket: Bucket, queue: TaskQueueConnection):
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
        self,
        config: Configuration,
        observations: pd.DataFrame,
        orbits: Orbits,
        job_completion_pubsub_topic: Optional[str] = None,
    ) -> JobManifest:
        """
        Submit a new job for execution.

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
        job_completion_pubsub_topic : Optional[str]
            The name of a pubsub topic (in the canonical
            projects/{project}/topics/{topic} format) which should get an
            announcement when a launched job completed.

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

        manifest = JobManifest.create(job_id, job_completion_pubsub_topic)
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
            statuses = self.get_task_statuses(manifest)
            for i, task_id in enumerate(manifest.task_ids):
                status = statuses[task_id]
                line = "\t".join(
                    (
                        "task=" + task_id,
                        "state=" + str(status.state),
                        "worker=" + str(status.worker),
                    )
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

    def __init__(
        self, gcs: GCSClient, pubsub_client: PublisherClient, queue: TaskQueueConnection
    ):
        """Creates a new worker.

        The worker's receive loop is not set up automatically. The caller should
        call run_worker_loop when they are ready to handle tasks.

        Parameters
        ----------
        gcs : GCSClient
            A Google Cloud Storage client.
        pubsub_client : google.cloud.pubsub_v1.PublisherClient
            A Google Cloud PubSub Publisher client.
        queue : TaskQueueConnection
            A queue to listen to. The queue should be connected before calling
            run_worker_loop.

        """

        self.gcs = gcs
        self.pubsub_client = pubsub_client
        self.queue = queue

    def run_worker_loop(self, poll_interval: float, idle_shutdown_timeout: int):
        """
        Block forever, handling new tasks in a loop.

        Parameters
        ----------
        poll_interval : float
            The time, in seconds, between successive checks for work to be done.

        idle_shutdown_timeout : int
            If non-negative, terminate the worker and the entire instance if
            there are no tasks to be done for this many seconds. On Google
            Compute, this kills the VM. Otherwise, this returns, ending the loop
            (and probably the process)
        """

        for task in self.poll_for_tasks(
            poll_interval=poll_interval, idle_shutdown_timeout=idle_shutdown_timeout
        ):
            self.handle_task(task)

    def poll_for_tasks(
        self,
        poll_interval: float = 5.0,
        limit: int = -1,
        idle_shutdown_timeout: int = -1,
    ) -> Iterator[Task]:
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
        idle_shutdown_timeout : int
            If non-negative, terminate the worker and the entire instance if
            there are no tasks to be done for this many seconds. On Google
            Compute, this kills the VM. Otherwise, this returns, ending the loop
            (and probably the process)

        Returns
        -------
        Iterator[Task]
            An infinite iterator of tasks to be done.
        """

        logger.info("starting to poll for tasks")
        i = 0
        last_task_time = time.time()
        while True:
            task = self.queue.receive()
            if task is None:
                since_last_task = time.time() - last_task_time
                logger.info("no tasks in queue (%s since last task)", since_last_task)
                if 0 <= idle_shutdown_timeout < since_last_task:
                    logger.info("idle shutdown timeout has elapsed")
                    self.terminate()
                    return

                time.sleep(poll_interval)
            else:
                logger.info(
                    "received task job_id=%s task_id=%s", task.job_id, task.task_id
                )
                bucket = self.gcs.bucket(task.bucket)
                self.mark_in_progress(task, bucket)
                yield task
                last_task_time = time.time()
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
            self.mark_task_succeeded(task, bucket, out_dir)
            return out_dir
        except Exception as e:
            logger.error("task %s failed", task.task_id, exc_info=sys.exc_info)
            self.mark_task_failed(task, bucket, out_dir, e)
        finally:
            updated_manifest = mark_task_done_in_manifest(
                bucket, task.job_id, task.task_id
            )

            # If our update reduced the manifest down to zero tasks, then we
            # were the last task. Announce everything is done.
            all_tasks_done = len(updated_manifest.incomplete_tasks) == 0
            if all_tasks_done:
                announce_job_done(self.pubsub_client, updated_manifest)

    def mark_task_succeeded(self, task: Task, bucket: Bucket, result_dir: str):
        """
        Mark a task as succesfully completed, handled by the current process, and
        upload its results.

        Parameters
        ----------
        task : Task
            The Task which succeeded.
        bucket : Bucket
            The bucket hosting the job.
        result_directory : str
            A local directory holding all the results of the execution which
            should be uploaded to the bucket.
        """

        logger.info("marking task %s as a success", task.task_id)
        # Store the results for the publisher
        task._upload_results(bucket, result_dir)
        # Mark the message as successfully handled
        self.queue.channel.basic_ack(delivery_tag=task._delivery_tag)
        set_task_status(bucket, task.job_id, task.task_id, TaskState.SUCCEEDED)

    def mark_task_failed(
        self, task: Task, bucket: Bucket, result_directory: str, exception: Exception
    ):
        """Mark a task as having failed.

        The task's status is updated in the bucket. Any intermediate results in
        result_directory are uploaded to the bucket, as well as an error trace.

        Parameters
        ----------
        task : Task
            The Task which failed.
        bucket : Bucket
            The bucket hosting the job.
        result_directory : str
            A local directory holding all the results of the execution.
        exception : Exception
            The exception that triggered the failure.
        """

        logger.error(
            "marking task %s as a failure, reason: %s", task.task_id, exception
        )
        # store the failed results
        task._upload_failure(bucket, result_directory, exception)
        # Mark the message as unsuccessfully attempted
        self.queue.channel.basic_nack(delivery_tag=task._delivery_tag, requeue=False)
        set_task_status(bucket, task.job_id, task.task_id, TaskState.FAILED)

    def mark_in_progress(self, task: Task, bucket: Bucket):
        """Mark the task as in-progress, handled by the current process.

        Parameters
        ----------
        bucket : Bucket
            The bucket hosting the job.
        """
        set_task_status(bucket, task.job_id, task.task_id, TaskState.IN_PROGRESS)

    def terminate(self):
        # Determine whether we are running on a Google Compute VM.
        on_google_compute_engine = compute_engine.discover_running_on_compute_engine()

        if on_google_compute_engine:
            logger.info("detected that process is running on Google Compute Engine")

            compute_engine.terminate_self()
        else:
            logger.info(
                "Google Compute Engine not detected, nothing special to do for termination"
            )
