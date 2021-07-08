from typing import AnyStr, List, Mapping, Optional

import datetime
import json
import logging
import re

from google.cloud.storage import Bucket
import google.api_core.exceptions
from google.cloud.pubsub_v1 import PublisherClient
import google.cloud.pubsub

from thor.orbits import Orbits
from thor.taskqueue.tasks import Task

logger = logging.getLogger("thor")


class JobManifest:
    """
    A manifest which lists all the orbit IDs and task IDs associated with a
    particular job. This class is serialized into a string and stored in the
    Google Cloud Storage bucket for a job. It can be retrieved to iterate over
    tasks.

    Attributes
    ----------
    creation_time : datetime.datetime
        The time that this JobManifest was created, including time zone.
    update_time : datetime.datetime
        The most recent time that this JobManifest was modified, including time zone.
    job_id : str
        A string identifier for the job.
    task_ids : List[str]
        A list of the IDs of tasks in the job.
    orbit_ids : List[str]
        A list of string IDs of orbits associated with the job's tasks. Each
        Task is assigned a single Orbit.
    incomplete_tasks : List[str]
        A list of the IDs of tasks that have not completed for this job.
    pubsub_topic : Optional[str]
        The name of a pubsub topic to publish to when all tasks in the Job are
        done. Note that pubsub topic names look like URL paths, and include a
        project name; they follow the pattern
        projects/{project-id}/topic/{topic}.
    """

    def __init__(
        self,
        creation_time: datetime.datetime,
        update_time: datetime.datetime,
        job_id: str,
        orbit_ids: List[str],
        task_ids: List[str],
        incomplete_tasks: List[str],
        pubsub_topic: Optional[str] = None,
    ):
        """
        Low-level constructor for a JobManifest.

        Use the create classmethod instead.
        """
        self.creation_time = creation_time
        self.update_time = update_time
        self.job_id = job_id
        self.orbit_ids = orbit_ids
        self.task_ids = task_ids
        self.incomplete_tasks = incomplete_tasks

        if pubsub_topic is not None:
            topic_parts = pubsub_topic.split("/")
            if len(topic_parts) != 4 or topic_parts[0] != "projects" or topic_parts[2] != "topics":
                raise ValueError(
                    "pubsub topic must match pattern 'projects/{project}/topics/{topic}'"
                )
            self.pubsub_topic = pubsub_topic
        else:
            self.pubsub_topic = None


    @classmethod
    def create(cls, job_id: str, pubsub_topic: Optional[str] = None) -> "JobManifest":
        """
        Create a new empty JobManifest with the given job ID.

        Parameters
        ----------
        job_id : str
            Identifier for the job.
        job_completion_pubsub_topic : Optional[str]
            The name of a pubsub topic (in the canonical
            projects/{project}/topics/{topic} format) which should get an
            announcement when a launched job completed.

        Returns
        -------
        JobManifest
            The newly created JobManifest.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        return JobManifest(
            creation_time=now,
            update_time=now,
            job_id=job_id,
            orbit_ids=[],
            task_ids=[],
            incomplete_tasks=[],
            pubsub_topic=pubsub_topic,
        )

    def append(self, orbit: Orbits, task: Task):
        """Add an Orbit and Task to the Manifest.

        Parameters
        ----------
        orbit : Orbits
            The orbit that will be handled as part of the task.
        task : Task
            The task to be done as part of this job.
        """

        assert len(orbit) == 1, "There should be exactly one Orbit per task."
        self.orbit_ids.append(orbit.ids[0])
        self.task_ids.append(task.task_id)
        self.incomplete_tasks.append(task.task_id)

    def to_str(self) -> str:
        """Serialize the JobManifest as a string.

        Returns
        -------
        str
            JSON serialization of the JobManifest.
        """

        data = {
            "job_id": self.job_id,
            "creation_time": self.creation_time.isoformat(),
            "update_time": self.update_time.isoformat(),
            "orbit_ids": self.orbit_ids,
            "task_ids": self.task_ids,
            "incomplete_tasks": self.incomplete_tasks,
        }
        if self.pubsub_topic is not None:
            data["pubsub_topic"] = self.pubsub_topic
        return json.dumps(data)

    @classmethod
    def from_str(cls, data: AnyStr) -> "JobManifest":
        """
        Deserialize data into a JobManifest.

        Parameters
        ----------
        cls :
        data : AnyStr
            Serialized JobManifest data, as created with to_str

        Returns
        -------
        JobManifest
            The deserialized JobManifest.
        """
        as_dict = json.loads(data)
        as_dict["creation_time"] = datetime.datetime.fromisoformat(
            as_dict["creation_time"]
        )
        as_dict["update_time"] = datetime.datetime.fromisoformat(as_dict["update_time"])
        return cls(**as_dict)


def upload_job_manifest(bucket: Bucket, manifest: JobManifest):
    """
    Upload a JobManifest to a bucket. It gets stored in
    BUCKET/thor_jobs/v1/job-{job_id}/manifest.json

    Parameters
    ----------
    bucket : google.cloud.storage.Bucket
        The GCS bucket where job data is stored.
    manifest : thor.taskqueue.JobManifest
        The manifest to upload.
    """
    path = f"thor_jobs/v1/job-{manifest.job_id}/manifest.json"
    bucket.blob(path).upload_from_string(manifest.to_str())


def mark_task_done_in_manifest(
    bucket: Bucket, job_id: str, task_id: str
) -> JobManifest:
    """
    """
    path = f"thor_jobs/v1/job-{job_id}/manifest.json"

    i = 0
    max_retries = 32
    while i < max_retries:
        i += 1
        try:
            # Download the old version. Note the generation.
            old_manifest_blob = bucket.blob(path)
            old_manifest_blob.reload()
            generation = old_manifest_blob.generation
            assert generation is not None

            as_str = old_manifest_blob.download_as_string(
                if_generation_match=generation,
            )

            manifest = JobManifest.from_str(as_str)

            # Remove the task from the manifest.
            manifest.incomplete_tasks.remove(task_id)
            # Update the timestamp
            manifest.update_time = datetime.datetime.now(datetime.timezone.utc)

            # Update the new version - as long as the generation hasn't changed.
            bucket.blob(path).upload_from_string(
                manifest.to_str(), if_generation_match=generation,
            )
            logger.debug(f"updated manifest generation=%s", generation)
            return manifest
        except google.api_core.exceptions.PreconditionFailed:
            # The generation changed out from under us. Try again.
            logger.debug(
                "lost a race to update manifest (tried generation=%s)", generation
            )
        except google.api_core.exceptions.NotFound:
            # Sometimes this appears if we ask for a very recent generation
            logger.debug(
                "got a 404 when asking to update manifest (tried generation=%s)",
                generation,
            )


def download_job_manifest(bucket: Bucket, job_id: str) -> JobManifest:
    """
    Download the JobManifest associated with job_id in given bucket.

    Parameters
    ----------
    bucket : google.cloud.storage.Bucket
        The GCS bucket where job data is stored.
    job_id : str
        The ID of the job.

    Returns
    -------
    JobManifest
    """
    path = f"thor_jobs/v1/job-{job_id}/manifest.json"
    as_str = bucket.blob(path).download_as_string()
    return JobManifest.from_str(as_str)


def announce_job_done(client: PublisherClient, manifest: JobManifest):
    logger.info("announcing that job %s is complete", manifest.job_id)
    if manifest.pubsub_topic is None:
        logger.info("no pubsub topic set for job %s", manifest.job_id)
        return
    announcement = manifest.to_str().encode()
    future = client.publish(manifest.pubsub_topic, announcement)
    future.result()
