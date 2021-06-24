import logging
import argparse
import pandas as pd
import uuid
import time

from google.cloud.storage import Client as GCSClient

from thor.taskqueue import tasks, queue
from thor.orbits import Orbits
from thor.config import Config


logger = logging.getLogger("thor")


def main():
    parser = argparse.ArgumentParser(
        description="Run Tracklet-less Heliocentric Orbit Recovery"
    )
    parser.add_argument(
        "preprocessed_observations",
        type=str,
        help="Preprocessed observations."
    )
    parser.add_argument(
        "test_orbits",
        type=str,
        help="Path to test orbits."
    )
    parser.add_argument(
        "out_dir",
        type=str,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if not isinstance(args.config, str):
        config = Config
    else:
        config = Config.fromYaml(args.config)

    # Read observations
    preprocessed_observations = pd.read_csv(
        args.preprocessed_observations,
        index_col=False,
        dtype={
            "obs_id": str
        }
    )

    # Read test orbits
    test_orbits = Orbits.from_csv(args.test_orbits)

    # Connect to Rabbit
    q = queue.TaskQueueConnection(
        config.TASKQUEUE_CONFIG.hostname,
        config.TASKQUEUE_CONFIG.port,
        config.TASKQUEUE_CONFIG.username,
        config.TASKQUEUE_CONFIG.password,
        config.TASKQUEUE_CONFIG.queue
    )
    q.connect()

    # Connect to Google Cloud Storage
    gcs_client = GCSClient()
    bucket = gcs_client.bucket(config.TASKQUEUE_CONFIG.bucket)

    job_id = str(uuid.uuid1())
    logger.info("Generated Job ID: %s", job_id)
    tasks.upload_job_inputs(bucket, job_id, config, preprocessed_observations)

    launched_tasks = []
    task_id_to_orbit_id = {}
    for orbit in test_orbits.split(1):
        task = tasks.Task.create(job_id, config, preprocessed_observations, orbit)
        logger.info("Created Task (id=%s) for orbit %s", task.task_id, orbit.ids[0])

        q.publish(task)

        launched_tasks.append(task)
        task_id_to_orbit_id[task.task_id] = orbit.ids[0]

    logger.info("Launched %d tasks", len(launched_tasks))
    q.close()

    while True:
        counts_by_state = {
            "launched": 0,
            "in_progress": 0,
            "succeeded": 0,
            "failed": 0,
        }
        for t in launched_tasks:
            status = tasks.get_status(bucket, job_id, t.task_id)
            orbit_id = task_id_to_orbit_id[t.task_id]
            state = status["state"]
            worker = status["worker"]
            print(f"{orbit_id}\t{state}\t{worker}")

            counts_by_state[state] += 1
            time.sleep(0.05)

        print("counts:")
        print(f"  launched:  {counts_by_state['launched']}")
        print(f"  in_progress:  {counts_by_state['in_progress']}")
        print(f"  succeeded:  {counts_by_state['succeeded']}")
        print(f"  failed:  {counts_by_state['failed']}")

        if counts_by_state["launched"] == 0 and counts_by_state["in_progress"] == 0:
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
