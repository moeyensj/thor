from typing import Optional, Tuple, Mapping

import io
import os
import json
import logging
import uuid
import socket
import posixpath
import traceback
import enum

import pika
import pandas as pd

from google.cloud.storage.bucket import Bucket

from thor.config import Configuration
from thor.orbits import Orbits


logger = logging.getLogger("thor")


class Task:
    def __init__(
        self,
        job_id: str,
        task_id: str,
        bucket: str,
        channel: pika.channel.Channel,
        delivery_tag: int,
    ):
        """
        Low-level constructor for a new task. Callers should prefer the create or
        from_msg methods.
        """
        self.job_id = job_id
        self.task_id = task_id
        self.bucket = bucket

        self._channel = channel
        self._delivery_tag = delivery_tag

    @classmethod
    def create(cls, job_id: str, bucket: Bucket, orbits: Orbits) -> "Task":
        """
        Create a new Task to handle a given orbit under a particular job.

        Parameters
        ----------
        job_id: str, an identifier for the job
        bucket: google.cloud.storage.bucket.Bucket, a bucket that will hold task
            inputs and outputs
        channel: pika.channel.Channel where the Task will be sent.
        orbits: thor.orbits.Orbits, the orbits to analyze

        Returns
        -------
        Task:
            The newly created Task.
        """

        tp = cls(
            job_id=job_id,
            task_id=new_task_id(orbits),
            bucket=bucket.name,
            channel=None,
            delivery_tag=-1,
        )
        upload_task_inputs(bucket, tp, orbits)
        set_task_status(bucket, job_id, tp.task_id, TaskState.REQUESTED, worker=None)
        return tp

    @classmethod
    def from_msg(
        cls,
        channel: pika.channel.Channel,
        method: pika.amqp_object.Method,
        properties: pika.amqp_object.Properties,
        body: bytes,
    ) -> "Task":
        """
        Construct a Task from a message off of a task queue.

        Parameters
        ----------
        cls :
        channel : pika.channel.Channel
            An opened channel to a RabbitMQ task queue.
        method : pika.amqp_object.Method
            RabbitMQ Method metadata associated with a message.
        properties : pika.amqp_object.Properties
            RabbitMQ Properties metadata associated with an object.
        body : bytes
            The payload of the RabbitMQ message.

        Returns
        -------
        Task
            A deserialized Task sourced from the message.
        """
        data = json.loads(body.decode("utf8"))
        return Task(
            job_id=data["job_id"],
            task_id=data["task_id"],
            bucket=data["bucket"],
            channel=channel,
            delivery_tag=method.delivery_tag,
        )

    def to_bytes(self) -> bytes:
        """Serialize the Task into a payload suitable for the task queue.

        Returns
        -------
        bytes
            JSON serialization of the task.
        """

        data = {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "bucket": self.bucket,
        }
        return json.dumps(data).encode("utf8")

    def download_inputs(
        self, bucket: Bucket
    ) -> Tuple[Configuration, pd.DataFrame, Orbits]:
        """Download the input data to execute a Task.

        Data are downloaded into memory, not to disk.

        Parameters
        ----------
        bucket : Bucket
            The Google Cloud Storage bucket hosting the job.

        Returns
        -------
        Tuple[Configuration, pd.DataFrame, Orbits]
            All the inputs required to handle a Task.
        """

        return download_task_inputs(bucket, self)

    def mark_in_progress(self, bucket: Bucket):
        """Mark the task as in-progress, handled by the current process.

        Parameters
        ----------
        bucket : Bucket
            The bucket hosting the job.
        """

        set_task_status(bucket, self.job_id, self.task_id, TaskState.IN_PROGRESS)

    def mark_success(self, bucket: Bucket, result_directory: str):
        """
        Mark the task as succesfully completed, handled by the current process, and
        upload its results.

        Parameters
        ----------
        bucket : Bucket
            The bucket hosting the job.
        result_directory : str
            A local directory holding all the results of the execution which
            should be uploaded to the bucket.
        """

        logger.info("marking task %s as a success", self.task_id)
        # Store the results for the publisher
        self._upload_results(bucket, result_directory)
        # Mark the message as successfully handled
        self._channel.basic_ack(delivery_tag=self._delivery_tag)
        set_task_status(bucket, self.job_id, self.task_id, TaskState.SUCCEEDED)

    def mark_failure(self, bucket: Bucket, result_directory: str, exception: Exception):
        """Mark the task as having failed.

        The task's status is updated in the bucket. Any intermediate results in
        result_directory are uploaded to the bucket, as well as an error trace.

        Parameters
        ----------
        bucket : Bucket
            The bucket hosting the job.
        result_directory : str
            A local directory holding all the results of the execution.
        exception : Exception
            The exception that triggered the failure.
        """

        logger.error(
            "marking task %s as a failure, reason: %s", self.task_id, exception
        )
        # store the failed results
        self._upload_failure(bucket, result_directory, exception)
        # Mark the message as unsuccessfully attempted
        self._channel.basic_nack(
            delivery_tag=self._delivery_tag,
            requeue=False,
        )
        set_task_status(bucket, self.job_id, self.task_id, TaskState.FAILED)

    def _upload_results(self, bucket: Bucket, result_directory: str):
        # Task-wide directory in the bucket where results go
        output_blobdir = _task_output_path(self.job_id, self.task_id)
        for (dirpath, _, filenames) in os.walk(result_directory):
            # Trim off the result_directory prefix from dirpath.
            relative_dir = os.path.relpath(dirpath, result_directory)
            for filename in filenames:
                # filepath is the path of the file locally
                filepath = os.path.join(dirpath, filename)
                # blobpath is the path that we want to use remotely.
                blobpath = posixpath.join(
                    output_blobdir,
                    relative_dir,
                    filename,
                )
                # Normalize any components like './' or '../'
                blobpath = posixpath.normpath(blobpath)
                logger.debug("uploading %s to %s", filepath, blobpath)
                bucket.blob(blobpath).upload_from_filename(filepath)

    def _upload_failure(
        self, bucket: Bucket, result_directory: str, exception: Exception
    ):
        output_blobdir = _task_output_path(self.job_id, self.task_id)
        exception_string = traceback.format_exception(
            etype=type(exception),
            value=exception,
            tb=exception.__traceback__,
        )
        blobpath = posixpath.join(output_blobdir, "error_message.txt")
        logger.error("uploading exception trace to %s", blobpath)
        bucket.blob(blobpath).upload_from_string(exception_string)
        self._upload_results(self.bucket, result_directory)

        raise NotImplementedError()


# Generated randomly:
_task_id_namespace = uuid.UUID("b3f9427b-8ee5-4c79-a3a1-875c6947777b")


def new_task_id(orbits: Orbits) -> str:
    """Generate a new ID for a task which handles the given orbits.

    Parameters
    ----------
    orbits : Orbits
        The orbits being handled in the Task.

    Returns
    -------
    str
        The generated ID.

    """
    if len(orbits) == 1:
        return orbits.ids[0]

    combined_ids = ".".join(orbits.ids)
    return str(uuid.uuid3(_task_id_namespace, combined_ids))


def download_task_inputs(
    bucket: Bucket, task: Task
) -> Tuple[Configuration, pd.DataFrame, Orbits]:
    """Download the data required to process this task.

    All data are downloaded into memory, not onto disk anywhere.

    Parameters
    ----------
    bucket : Bucket
        The bucket hosting the task.
    task : Task
        The Task to be performed.

    Returns
    -------
    Tuple[Configuration, pd.DataFrame, Orbits]
        The configuration, observations, and orbits that form the inputs to a
        runTHOR task.

    """

    cfg_path = _job_input_path(task.job_id, "config.yml")
    logger.info("downloading task input %s", cfg_path)
    cfg_bytes = bucket.blob(cfg_path).download_as_string()
    config = Configuration().fromYamlString(cfg_bytes.decode("utf8"))

    obs_path = _job_input_path(task.job_id, "observations.csv")
    logger.info("downloading task input %s", obs_path)
    obs_bytes = bucket.blob(obs_path).download_as_string()
    observations = pd.read_csv(
        io.BytesIO(obs_bytes),
        index_col=False,
        dtype={"obs_id": str},
    )

    orbit_path = _task_input_path(task.job_id, task.task_id, "orbit.csv")
    logger.info("downloading task input %s", orbit_path)
    orbit_bytes = bucket.blob(orbit_path).download_as_string()
    orbit = Orbits.from_csv(io.BytesIO(orbit_bytes))

    return (config, observations, orbit)


def upload_job_inputs(bucket: Bucket, job_id: str, config: Configuration, observations: pd.DataFrame):
    """Upload all the inputs required to execute a task.

    These inputs are uploaded into the given bucket. This function uploads the
    inputs that are common to all tasks in a job: the configuration and the
    observations. The related method upload_task_inputs uploads the inputs that
    are specific to a single task, namely the orbits.

    Parameters
    ----------
    bucket : Bucket
        The bucket hosting the job.
    job_id : str
        The ID of the job.
    config : Configuration
        A THOR configuration which the Task executors should use.
    observations : pd.DataFrame
        The preprocessed observations which should be used by task executors.

    """

    # Upload configuration file
    cfg_bytes = config.toYamlString()
    cfg_path = _job_input_path(job_id, "config.yml")
    logger.info("uploading job input %s", cfg_path)
    bucket.blob(cfg_path).upload_from_string(cfg_bytes)

    # Upload observations
    observations_buf = io.BytesIO()
    observations.to_csv(observations_buf, index=False)
    observations_bytes = observations_buf.getvalue()

    observations_path = _job_input_path(job_id, "observations.csv")
    logger.info("uploading job input %s", observations_path)
    bucket.blob(observations_path).upload_from_string(observations_bytes)


def upload_task_inputs(bucket: Bucket, task: Task, orbit: Orbits):
    """Uploads the inputs required to execute a specific task.

    These inputs are uploaded into the given bucket. The related method
    upload_job_inputs should also be executed, but just once for all tasks in a
    job; it uploads the observations and configuration.

    Parameters
    ----------
    bucket : Bucket
        The bucket hosting the task's job.
    task : Task
        The Task to be executed.
    orbit : Orbits
        The test orbits to use in a THOR run.

    """


    # Upload orbit
    orbit_buf = io.BytesIO()
    orbit.to_csv(orbit_buf)
    orbit_bytes = orbit_buf.getvalue()

    orbit_path = _task_input_path(task.job_id, task.task_id, "orbit.csv")
    logger.info("uploading task input %s", orbit_path)
    bucket.blob(orbit_path).upload_from_string(orbit_bytes)


def download_task_outputs(
    root_directory: str, bucket: Bucket, job_id: str, task_id: str
):
    """
    Download all the results from a task execution.

    The task may have succeeded or failed; either is fine. Results are
    downloaded into a directory relative to root_directory. They are placed in a
    subpath tasks/task-{task_id}/outputs. The actual files within that
    subdirectory come directly from the thor.main.runTHOR function.

    Parameters
    ----------
    root_directory : str
        A local directory where outputs should be placed.
    bucket : Bucket
        The bucket hosting the job with outputs.
    job_id : str
        The ID of the job.
    task_id : str
        The ID of the task to get outputs from.
    """

    job_prefix = _job_path(job_id) + "/"
    task_prefix = _task_output_path(job_id, task_id) + "/"
    blobs = bucket.list_blobs(prefix=task_prefix)
    for b in blobs:
        relative_path = b.name[len(job_prefix) :]
        local_path = os.path.join(root_directory, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        logger.info("downloading %s", local_path)
        b.download_to_filename(local_path)


def _job_path(job_id: str) -> str:
    return f"thor_jobs/v1/job-{job_id}"


def _job_input_path(job_id: str, name: str):
    return f"thor_jobs/v1/job-{job_id}/inputs/{name}"


def _task_input_path(job_id: str, task_id: str, name: str):
    return f"thor_jobs/v1/job-{job_id}/tasks/task-{task_id}/inputs/{name}"


def _task_output_path(job_id: str, task_id: str):
    return f"thor_jobs/v1/job-{job_id}/tasks/task-{task_id}/outputs"


def _task_status_path(job_id: str, task_id: str):
    return f"thor_jobs/v1/job-{job_id}/tasks/task-{task_id}/status"


_LOCAL_FQDN = socket.getfqdn()


class TaskState(enum.Enum):
    """
    Represents the present state of a single Task. Tasks go from being requested
    (unhandled by any worker) to in progress (while THOR is running) to
    succeeded (if execution terminates without error) or failed (if any
    exception occurred while running THOR).
    """
    REQUESTED = "requested"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    def completed(self) -> bool:
        """Returns true if the TaskState represents a completed task.

        Returns
        -------
        bool
            Whether the task has finished execution.
        """

        return self in (TaskState.SUCCEEDED, TaskState.FAILED)


class TaskStatus:
    """
    Represents all status information about a single task.

    Attributes
    ----------
    state : TaskState
        The current state of the task.
    worker : Optional[str]
        An identifier for the process which was assigned the task, or None if
        the task has not been assigned.
    """

    state: TaskState
    worker: Optional[str]

    def __init__(self, state: TaskState, worker: Optional[str] = None):
        """Create a new TaskStatus.

        Parameters
        ----------
        state : TaskState
            The current state of the task.
        worker : Optional[str]
            A string identifier for the worker, or None if no worker is handling
            the task.
        """

        self.state = state
        self.worker = worker

    def to_bytes(self) -> bytes:
        """Encode the TaskStatus as bytes.

        The TaskStatus gets serialized as UTF8-encoded JSON.

        Returns
        -------
        bytes
            The serialized status.
        """

        data = {
            "state": str(self.state.value),
            "worker": self.worker,
        }
        return json.dumps(data).encode("utf8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "TaskStatus":
        """
        Construct a new TaskStatus from bytes.

        Parameters
        ----------
        data : bytes
            The raw serialized bytes of a TaskStatus.

        Returns
        -------
        TaskStatus
            The deserialized TaskStatus.
        """
        data = json.loads(data)
        state = TaskState(data["state"])
        worker = data["worker"]
        if worker == "None":
            worker = None
        return TaskStatus(state, worker)


def set_task_status(
    bucket: Bucket,
    job_id: str,
    task_id: str,
    state: TaskState,
    worker: Optional[str] = _LOCAL_FQDN,
):
    """Set the status of a task.

    Uploads the JSON serialization of a TaskStatus into a bucket, recording its
    present state.

    Parameters
    ----------
    bucket : Bucket
        The Google Cloud Storage bucket that hosts the given job and task.
    job_id : str
        The ID of the job.
    task_id : str
        The ID of the task.
    state : TaskState
        The state of the task.
    worker : Optional[str]
        An identifier for the worker reporting the state of the task (or None if
        no worker is handling the task).
    """

    if state == TaskState.REQUESTED:
        worker = None
    status = TaskStatus(state, worker)
    blob_path = _task_status_path(job_id, task_id)
    bucket.blob(blob_path).upload_from_string(status.to_bytes())


def get_task_status(bucket: Bucket, job_id: str, task_id: str) -> TaskStatus:
    """Get the status of a task.

    Parameters
    ----------
    bucket : Bucket
        The Google Cloud Storage bucket that hosts th egiven job and task.
    job_id : str
        The ID of the job.
    task_id : str
        The ID of the task.

    Returns
    -------
    TaskStatus
        The status of the Task.
    """

    blob_path = _task_status_path(job_id, task_id)
    status_str = bucket.blob(blob_path).download_as_string()
    return TaskStatus.from_bytes(status_str)


def get_task_statuses(bucket: Bucket, manifest: "JobManifest") -> Mapping[str, TaskStatus]:
    """Retrieve the status of all tasks in the manifest.

    Task statuses are stored in a bucket, so this does a sequence of remote
    calls - one for each task in the job - to retrieve status objects.

    Parameters
    ----------
    bucket : Bucket
        The bucket associated with the job.
    manifest : JobManifest
        The manifest for the job, listing all tasks.

    Returns
    -------
    Mapping[str, TaskStatus]
        The status of each Task, indexed by Task ID.

    Examples
    --------
    FIXME: Add docs.

    """
    statuses = {}
    for task_id in manifest.task_ids:
        statuses[task_id] = get_task_status(bucket, manifest.job_id, task_id)
    return statuses
