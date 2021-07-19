import tempfile
import os

from thor.taskqueue import tasks


class MockBucket():
    def __init__(self):
        self.blobs = {}

    def blob(self, name):
        if name in self.blobs:
            return self.blobs[name]
        blob = MockBlob()
        self.blobs[name] = blob
        return blob


class MockBlob():
    def __init__(self):
        content = b""

    def upload_from_string(self, content):
        self.content = content.encode()

    def upload_from_filename(self, filename):
        with open(filename, "rb") as f:
            self.content = f.read()

class MockChannel():
    def __init__(self):
        pass


def test_upload_failure():
    bucket = MockBucket()
    channel = MockChannel()
    task = tasks.Task("job-id", "task-id", bucket, channel, 1)

    try:
        raise ValueError("a genuine exception!")
    except ValueError as e:
        exception = e

    with tempfile.TemporaryDirectory() as tempdir:
        with open(os.path.join(tempdir, "data-example"), "wb") as temp_data_file:
            temp_data_file.write(b"test data upload")
        task._upload_failure(bucket, tempdir, exception)

    # Should have uploaded an error message
    output_root = tasks._task_output_path("job-id", "task-id" )
    error_blob = bucket.blobs[output_root + "/error_message.txt"]
    assert b"a genuine exception!" in error_blob.content

    data_blob = bucket.blobs[output_root + "/./data-example"]
    assert data_blob.content == b"test data upload"


def test_upload_failure_multiple_errors():
    bucket = MockBucket()
    channel = MockChannel()
    task = tasks.Task("job-id", "task-id", bucket, channel, 1)

    try:
        raise ValueError("a genuine exception!")
    except ValueError as e:
        try:
            raise AssertionError("an internal one") from e
        except AssertionError as e2:
            exception = e2

    with tempfile.TemporaryDirectory() as tempdir:
        with open(os.path.join(tempdir, "data-example"), "wb") as temp_data_file:
            temp_data_file.write(b"test data upload")
        task._upload_failure(bucket, tempdir, exception)

    # Should have uploaded an error message
    output_root = tasks._task_output_path("job-id", "task-id" )
    error_blob = bucket.blobs[output_root + "/error_message.txt"]
    assert b"Multiple errors:" in error_blob.content
    assert b"a genuine exception!" in error_blob.content
    assert b"an internal one" in error_blob.content

    data_blob = bucket.blobs[output_root + "/./data-example"]
    assert data_blob.content == b"test data upload"
