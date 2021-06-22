"""
This file contains integration tests, which may be slow and require setup.
"""

import os
import pytest

import thorctl.tasks
import thorctl.queue

RUN_INTEGRATION_TESTS = "THOR_INTEGRATION_TEST" in os.environ

_RABBIT_HOST = os.environ.get("RABBIT_HOST", "localhost")
_RABBIT_PORT = os.environ.get("RABBIT_PORT", 5672)
_RABBIT_USER = os.environ.get("RABBIT_USER", "thor")
_RABBIT_PASSWORD = os.environ.get("RABBIT_PASSWORD", None)


@pytest.fixture()
def queue_connection(request):
    if _RABBIT_PASSWORD is None:
        raise ValueError(
            "you must set RABBIT_PASSWORD env variable for integration tests",
        )
    queue_name = request.function.__name__
    conn = thorctl.queue.TaskQueueConnection(
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


@pytest.mark.skipif(not RUN_INTEGRATION_TESTS,
                    reason="integration_tests not enabled")
def test_queue_roundtrip(queue_connection):
    task_payload = thorctl.tasks.TaskPayload(
        job_id="job_id",
        task_id="task_id",
        input_data_uri="some_source",
        result_uri="some_dest",
    )
    queue_connection.publish(task_payload)
    have = queue_connection.receive()
    assert have.payload == task_payload

    have.mark_success()
