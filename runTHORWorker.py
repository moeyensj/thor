import logging
import os
import tempfile
import time

from google.cloud.storage.client import Client as GCSClient

from thor import runTHOR
from thor.taskqueue import queue


def main():
    q = queue.TaskQueueConnection(
        "rabbit.c.moeyens-thor-dev.internal",
        5762,
        "thor",
        os.environ["RABBIT_PASSWORD"],
    )
    q.connect()

    gcs = GCSClient()

    while True:
        task = q.receive()
        if task is None:
            # No work to do: retry in a second
            time.sleep(1)
            continue

        try:
            config, preprocessed_observations, test_orbits = task.download_inputs(gcs)

            out_dir = tempfile.TemporaryDirectory(
                prefix=f"thor_{task.payload.job_id}_{task.payload.task_id}",
            )

            test_orbits_, recovered_orbits, recovered_orbit_members = runTHOR(
                preprocessed_observations,
                test_orbits,
                range_shift_config=config["RANGE_SHIFT_CONFIG"],
                cluster_link_config=config["CLUSTER_LINK_CONFIG"],
                iod_config=config["IOD_CONFIG"],
                od_config=config["OD_CONFIG"],
                odp_config=config["ODP_CONFIG"],
                out_dir=out_dir,
                if_exists="continue",
                logging_level=logging.INFO,
            )

            task.success(out_dir)
        except Exception as e:
            task.failure(out_dir, e)


if __name__ == "__main__":
    main()
