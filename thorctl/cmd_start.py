import uuid

from .main import cli
from . import tasks


@cli.command()
def start():
    print("starting a new job")
    job_id = str(uuid.uuid1())
    print(f"Job ID: {job_id}")
    task_id = tasks.new_task_id()
    print(f"Task ID: {task_id}")
    task = tasks.Task(
        job_id=job_id,
        task_id=task_id,
        input_data_uri="nowhere",
        result_uri="nowhere",
    )
    print(task.to_str())
