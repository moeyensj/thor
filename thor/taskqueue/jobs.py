from typing import AnyStr, Sequence, Mapping

import datetime
import json

from google.cloud.storage import Bucket

from thor.orbits import Orbits
from thor.taskqueue.tasks import Task, get_task_status


class JobManifest:
    """
    A manifest which lists all the orbit IDs and task IDs associated with a
    particular job. This class is serialized into a string and stored in the
    Google Cloud Storage bucket for a job. It can be retrieved to iterate over
    tasks.
    """

    def __init__(self, creation_time, job_id, orbit_ids, task_ids):
        """
        Low-level constructor for a JobManifest.

        Use the create classmethod instead.
        """
        self.creation_time = creation_time
        self.job_id = job_id
        self.orbit_ids = orbit_ids
        self.task_ids = task_ids

    @classmethod
    def create(cls, job_id: str):
        """
        Create a new empty JobManifest with the given job ID.

        Parameters
        ----------
        job_id : str
            Identifier for the job.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        return JobManifest(creation_time=now, job_id=job_id, orbit_ids=[], task_ids=[])

    def append(self, orbit: Orbits, task: Task):
        """
        Add an orbit and task to the manifest.

        The orbits should be the Orbits being handled in the given task.
        """
        if len(orbit) == 1:
            self.orbit_ids.append(orbit.ids[0])
        else:
            self.orbit_ids.append(list(orbit.ids))
        self.task_ids.append(task.task_id)

    def to_str(self) -> str:
        """
        JSON-serializes the JobManifest as a string.
        """
        return json.dumps(
            {
                "job_id": self.job_id,
                "creation_time": self.creation_time.isoformat(),
                "orbit_ids": self.orbit_ids,
                "task_ids": self.task_ids,
            }
        )

    @classmethod
    def from_str(cls, data: AnyStr) -> "JobManifest":
        """
        Creates a JobManifest from a string JSON representation, as created by
        JobManifest.to_str.
        """
        as_dict = json.loads(data)
        as_dict["creation_time"] = datetime.datetime.fromisoformat(
            as_dict["creation_time"]
        )
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


def get_job_statuses(
    bucket: Bucket, manifest: JobManifest
) -> Mapping[str, Mapping[str, str]]:
    """
    Look up the status of all tasks in the manifest by checking in bucket for
    task status marker objects.
    """
    statuses = {}
    for task_id in manifest.task_ids:
        statuses[task_id] = get_task_status(bucket, manifest.job_id, task_id)
    return statuses
