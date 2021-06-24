import json
import uuid
import io

import pika
import pandas as pd

from google.cloud.storage.bucket import Bucket
from google.cloud.storage.client import Client as GCSClient

from thor.config import Config
from thor.orbits import Orbits


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
    def create(cls, job_id, bucket, config, observations, orbits):
        """
        Create a new Task, uploading the inputs to a bucket.

        Parameters
        ----------
        job_id: str, an identifier for the job
        bucket: google.cloud.storage.bucket.Bucket, a bucket that will hold task
            inputs and outputs
        channel: pika.channel.Channel where the Task will be sent.
        config: thor.config.Config, a configuration object defining the THOR run
        observations: pandas.DataFrame, preprocessed observations
        orbits: thor.orbits.Orbits, the orbits to analyze
        """
        tp = cls(
            job_id=job_id,
            task_id=new_task_id(),
            bucket=bucket.name,
            channel=None,
            delivery_tag=-1
        )
        upload_task_inputs(bucket, tp, config, observations, orbits)
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
    config_bytes = _download_input(task, bucket, "config.yaml")
    config = Config.fromYamlString(config_bytes.decode("utf8"))

    observations_bytes = _download_input(task, bucket, "observations.csv")
    observations = pd.read_csv(
        io.BytesIO(observations_bytes),
        index_col=False,
        dtype={"obs_id": str},
    )

    orbits_bytes = _download_input(task, bucket, "orbits.csv")
    orbits = Orbits.from_csv(io.BytesIO(orbits_bytes))

    return (config, observations, orbits)


def upload_task_inputs(bucket: Bucket, task: Task, config, observations, orbits):
    # Upload Config
    _upload_input(task, bucket, config.toYamlString(), "config.yaml")

    # Upload observations
    observations_buf = io.BytesIO()
    observations.to_csv(observations_buf, index=False)
    observations_bytes = observations_buf.getvalue()
    _upload_input(task, bucket, observations_bytes, "observations.csv")

    # Upload orbits
    orbits_buf = io.BytesIO()
    orbits.to_csv(orbits_buf)
    orbits_bytes = orbits_buf.getvalue()
    _upload_input(task, bucket, orbits_bytes, "orbits.csv")


def _input_path_prefix(job_id: str, task_id: str):
    return f"thor_jobs/v1/job-{job_id}/task-{task_id}/inputs"


def _upload_input(task, bucket, contents, dest):
    prefix = _input_path_prefix(task.job_id, task.task_id)
    path = prefix + "/" + dest
    blob = bucket.blob(path)
    blob.upload_from_string(contents)


def _download_input(task, bucket, src):
    prefix = _input_path_prefix(task.job_id, task.task_id)
    path = prefix + "/" + src
    blob = bucket.blob(path)
    return blob.download_as_string()
