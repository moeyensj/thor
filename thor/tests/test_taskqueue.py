"""
This file contains integration tests, which may be slow and require setup.
"""

import os
import pytest
import pandas as pd
import pandas.testing as pd_testing
import uuid

from google.cloud.storage import Client as GCSClient

from thor import config
from thor.orbits import Orbits
from thor.taskqueue import tasks
from thor.taskqueue import queue
from thor.testing import integration_test

RUN_INTEGRATION_TESTS = "THOR_INTEGRATION_TEST" in os.environ

_RABBIT_HOST = os.environ.get("RABBIT_HOST", "localhost")
_RABBIT_PORT = os.environ.get("RABBIT_PORT", 5672)
_RABBIT_USER = os.environ.get("RABBIT_USER", "thor")
_RABBIT_PASSWORD = os.environ.get("RABBIT_PASSWORD", None)


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "testing", "data",
)


@pytest.fixture()
def queue_connection(request):
    if _RABBIT_PASSWORD is None:
        raise ValueError(
            "you must set RABBIT_PASSWORD env variable for integration tests",
        )
    queue_name = request.function.__name__
    conn = queue.TaskQueueConnection(
        _RABBIT_HOST,
        _RABBIT_PORT,
        _RABBIT_USER,
        _RABBIT_PASSWORD,
        queue_name,
    )
    conn.connect()
    yield conn
    conn.channel.queue_delete(queue_name)
    conn.close()


@pytest.fixture()
def google_storage_bucket(request):
    client = GCSClient()
    bucket_name = f"test_bucket__{request.function.__name__}"
    bucket = client.create_bucket(bucket_name)
    yield bucket
    bucket.delete(force=True, client=client)


@pytest.fixture()
def observations():
    yield pd.read_csv(os.path.join(DATA_DIR, "observations.csv"), index_col=False, dtype={"obs_id": str})


@pytest.fixture()
def orbits():
    yield Orbits.from_csv(os.path.join(DATA_DIR, "orbits.csv"))


@integration_test
def test_queue_roundtrip(queue_connection, google_storage_bucket, observations, orbits):

    test_config = config.Configuration()
    job_id = str(uuid.uuid1())
    tasks.upload_job_inputs(google_storage_bucket, job_id, test_config, observations)
    task = tasks.Task.create(
        job_id=job_id,
        bucket=google_storage_bucket,
        orbits=orbits,
    )
    queue_connection.publish(task)
    have = queue_connection.receive()
    assert have.job_id == task.job_id
    assert have.bucket == task.bucket
    assert have.task_id == task.task_id

    have_config, have_obs, have_orb = tasks.download_task_inputs(
        google_storage_bucket, have,
    )

    assert have_config == test_config
    pd_testing.assert_frame_equal(have_obs, observations)
    assert have_orb == orbits

    # have.mark_success()
