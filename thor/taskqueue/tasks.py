import io
import json
import logging
import uuid

import pika
import pandas as pd

from google.cloud.storage.bucket import Bucket
from google.cloud.storage.client import Client as GCSClient

from thor.config import Config
from thor.orbits import Orbits


logger = logging.getLogger("thor")


class Task:
    def __init__(self,
                 job_id: str,
                 task_id: str,
                 bucket: str,
                 channel: pika.channel.Channel,
                 delivery_tag: int):

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
            task_id=new_task_id(),
            bucket=bucket.name,
            channel=None,
            delivery_tag=-1
        )
        upload_task_inputs(bucket, tp, orbits)
        return tp

    @classmethod
    def from_msg(cls,
                 channel: pika.channel.Channel,
                 method: pika.amqp_object.Method,
                 properties: pika.amqp_object.Properties,
                 body: bytes) -> "Task":
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

    def download_inputs(self, client: GCSClient):
        bucket = client.bucket(self.bucket)
        return download_task_inputs(bucket, self)

    def success(self, result_directory):
        # Store the results for the publisher
        self._upload_results(result_directory)
        # Mark the message as successfully handled
        self._channel.basic_ack(delivery_tag=self._delivery_tag)

    def failure(self, result_directory, exception):
        # store the failed results
        self._upload_failure(result_directory, exception)
        # Mark the message as unsuccessfully attempted
        self._channel.basic_nack(
            delivery_tag=self._delivery_tag,
            requeue=False,
        )

    def _upload_results(self, result_directory):
        raise NotImplementedError()

    def _upload_failure(self, result_directory):
        raise NotImplementedError()


def new_task_id():
    return str(uuid.uuid4())


def download_task_inputs(bucket: Bucket, task: Task):
    cfg_path = _job_input_path(task.job_id, "config.yml")
    logger.info("downloading task input %s", cfg_path)
    cfg_bytes = bucket.blob(cfg_path).download_as_string()
    config = Config.fromYamlString(cfg_bytes.decode("utf8"))

    obs_path = _job_input_path(task.job_id,  "observations.csv")
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
