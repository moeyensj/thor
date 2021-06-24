import io
import os
import json
import logging
import uuid
import socket
import posixpath
import traceback

import pika
import pandas as pd

from google.cloud.storage.bucket import Bucket, Blob
from google.cloud.storage.client import Client as GCSClient

from thor.config import Config
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

        self.job_id = job_id
        self.task_id = task_id
        self.bucket = bucket

        self._channel = channel
        self._delivery_tag = delivery_tag

    @classmethod
    def create(cls, job_id, bucket, orbits):
        """
        Create a new Task to handle a given orbit under a particular job.

        Parameters
        ----------
        job_id: str, an identifier for the job
        bucket: google.cloud.storage.bucket.Bucket, a bucket that will hold task
            inputs and outputs
        channel: pika.channel.Channel where the Task will be sent.
        orbits: thor.orbits.Orbits, the orbits to analyze
        """

        tp = cls(
            job_id=job_id,
            task_id=new_task_id(orbits),
            bucket=bucket.name,
            channel=None,
            delivery_tag=-1,
        )
        upload_task_inputs(bucket, tp, orbits)
        set_task_status(bucket, job_id, tp.task_id, "requested")
        return tp

    @classmethod
    def from_msg(
        cls,
        channel: pika.channel.Channel,
        method: pika.amqp_object.Method,
        properties: pika.amqp_object.Properties,
        body: bytes,
    ) -> "Task":
        data = json.loads(body.decode("utf8"))
        return Task(
            job_id=data["job_id"],
            task_id=data["task_id"],
            bucket=data["bucket"],
            channel=channel,
            delivery_tag=method.delivery_tag,
        )

    def to_bytes(self) -> bytes:
        data = {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "bucket": self.bucket,
        }
        return json.dumps(data).encode("utf8")

    def download_inputs(self, bucket: Bucket):
        return download_task_inputs(bucket, self)

    def mark_in_progress(self, bucket: Bucket):
        set_task_status(bucket, self.job_id, self.task_id, "in_progress")

    def mark_success(self, bucket: Bucket, result_directory):
        logger.info("marking task %s as a success", self.task_id)
        # Store the results for the publisher
        self._upload_results(bucket, result_directory)
        # Mark the message as successfully handled
        self._channel.basic_ack(delivery_tag=self._delivery_tag)
        set_task_status(bucket, self.job_id, self.task_id, "succeeded")

    def mark_failure(self, bucket: Bucket, result_directory, exception):
        logger.error(
            "marking task %s as a failure, reason: %s", self.task_id, exception
        )
        # store the failed results
        self._upload_failure(bucket, result_directory, exception)
        # Mark the message as unsuccessfully attempted
        self._channel.basic_nack(
            delivery_tag=self._delivery_tag, requeue=False,
        )
        set_task_status(bucket, self.job_id, self.task_id, "failed")

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
                blobpath = posixpath.join(output_blobdir, relative_dir, filename,)
                logger.debug("uploading %s to %s", filepath, blobpath)
                bucket.blob(blobpath).upload_from_filename(filepath)

    def _upload_failure(
        self, bucket: Bucket, result_directory: str, exception: Exception
    ):
        output_blobdir = _task_output_path(self.job_id, self.task_id)
        exception_string = traceback.format_exception(
            etype=type(exception), value=exception, tb=exception.__traceback__,
        )
        blobpath = posixpath.join(output_blobdir, "error_message.txt")
        logger.error("uploading exception trace to %s", blobpath)
        bucket.blob(blobpath).upload_from_string(exception_string)
        self._upload_results(self.bucket, result_directory)

        raise NotImplementedError()


# Generated randomly:
_task_id_namespace = uuid.UUID("b3f9427b-8ee5-4c79-a3a1-875c6947777b")


def new_task_id(orbits: Orbits) -> str:
    """
    Generate an identifier for a task to handle given orbits.
    """

    if len(orbits) == 1:
        return orbits.ids[0]

    combined_ids = ".".join(orbits.ids)
    return str(uuid.uuid3(_task_id_namespace, combined_ids))


def download_task_inputs(bucket: Bucket, task: Task):
    cfg_path = _job_input_path(task.job_id, "config.yml")
    logger.info("downloading task input %s", cfg_path)
    cfg_bytes = bucket.blob(cfg_path).download_as_string()
    config = Config.fromYamlString(cfg_bytes.decode("utf8"))

    obs_path = _job_input_path(task.job_id, "observations.csv")
    logger.info("downloading task input %s", obs_path)
    obs_bytes = bucket.blob(obs_path).download_as_string()
    observations = pd.read_csv(
        io.BytesIO(obs_bytes), index_col=False, dtype={"obs_id": str},
    )

    orbit_path = _task_input_path(task.job_id, task.task_id, "orbit.csv")
    logger.info("downloading task input %s", orbit_path)
    orbit_bytes = bucket.blob(orbit_path).download_as_string()
    orbit = Orbits.from_csv(io.BytesIO(orbit_bytes))

    return (config, observations, orbit)


def upload_job_inputs(bucket, job_id, config, observations):
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


def upload_task_inputs(bucket: Bucket, task: Task, orbit):
    # Upload orbit
    orbit_buf = io.BytesIO()
    orbit.to_csv(orbit_buf)
    orbit_bytes = orbit_buf.getvalue()

    orbit_path = _task_input_path(task.job_id, task.task_id, "orbit.csv")
    logger.info("uploading task input %s", orbit_path)
    bucket.blob(orbit_path).upload_from_string(orbit_bytes)


def _job_input_path(job_id: str, name: str):
    return f"thor_jobs/v1/job-{job_id}/inputs/{name}"


def _task_input_path(job_id: str, task_id: str, name: str):
    return f"thor_jobs/v1/job-{job_id}/tasks/task-{task_id}/inputs/{name}"


def _task_output_path(job_id: str, task_id: str):
    return f"thor_jobs/v1/job-{job_id}/tasks/task-{task_id}/outputs"


def _task_status_path(job_id: str, task_id: str):
    return f"thor_jobs/v1/job-{job_id}/tasks/task-{task_id}/status"


_LOCAL_FQDN = socket.getfqdn()


def set_task_status(bucket, job_id, task_id, status_msg, worker=_LOCAL_FQDN):
    if status_msg == "requested":
        worker = "None"
    status_obj = {
        "state": status_msg,
        "worker": worker,
    }
    status_str = json.dumps(status_obj)
    blob_path = _task_status_path(job_id, task_id)
    bucket.blob(blob_path).upload_from_string(status_str)


def get_task_status(bucket, job_id, task_id):
    blob_path = _task_status_path(job_id, task_id)
    status_str = bucket.blob(blob_path).download_as_string()
    return json.loads(status_str)
