import logging
import argparse
import pandas as pd
import os

import pika
from google.cloud.storage import Client as GCSClient

from thor.taskqueue.client import Client as TaskQueueClient
from thor.taskqueue.queue import TaskQueueConnection
from thor.orbits import Orbits
from thor.config import Config


logger = logging.getLogger("thor")


def main():
    parser = argparse.ArgumentParser(
        description="Run Tracklet-less Heliocentric Orbit Recovery"
    )
    parser.add_argument(
        "preprocessed_observations", type=str, help="Preprocessed observations."
    )
    parser.add_argument("test_orbits", type=str, help="Path to test orbits.")
    parser.add_argument(
        "out_dir", type=str,
    )
    parser.add_argument(
        "--config", type=str, default=None,
    )
    args = parser.parse_args()

    if not isinstance(args.config, str):
        config = Config
    else:
        config = Config.fromYaml(args.config)

    # Read observations
    preprocessed_observations = pd.read_csv(
        args.preprocessed_observations, index_col=False, dtype={"obs_id": str}
    )

    # Read test orbits
    test_orbits = Orbits.from_csv(args.test_orbits)

    # Connect to Rabbit
    queue = TaskQueueConnection(
        pika.ConnectionParameters(
            host="rabbit.c.moeyens-thor-dev.internal",
            port=5672,
            credentials=pika.PlainCredentials(
                username="thor", password=os.environ["RABBIT_PASSWORD"],
            ),
        ),
        "thor-tasks",
    )
    taskqueue_client = TaskQueueClient(GCSClient(), queue)

    manifest = taskqueue_client.launch_job(
        config, preprocessed_observations, test_orbits
    )
    taskqueue_client.monitor_job_status(manifest.job_id)


if __name__ == "__main__":
    main()
