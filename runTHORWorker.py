import logging
import os
import tempfile
import time

from google.cloud.storage.client import Client as GCSClient

from thor import runTHOROrbit
from thor.taskqueue import queue

logger = logging.getLogger("thor")

def main():
    q = queue.TaskQueueConnection(
        "rabbit.c.moeyens-thor-dev.internal",
        5762,
        "thor",
        os.environ["RABBIT_PASSWORD"],
        "thor-tasks",
    )
    q.connect()

    gcs = GCSClient()

    i = 1
    while True:
        task = q.receive()
        if task is None:
            # No work to do: retry in a second
            if i % 100 == 0:
                logger.info("polling for tasks: none found (repeated 100 times)")
            else:
                logger.debug("polling for tasks: none found")
            i += 1
            time.sleep(1)
            continue
        try:
            i = 1
            logger.info(
                "polling for tasks: found task job_id=%s task_id=%s",
                task.job_id, task.task_id,
            )
            config, preprocessed_observations, test_orbit = task.download_inputs(gcs)

            logger.debug("downloaded inputs")

            out_dir = tempfile.TemporaryDirectory(
                prefix=f"thor_{task.job_id}_{task.task_id}",
            )

            logger.info("starting run on test orbit %s", test_orbit)
            runTHOROrbit(
                preprocessed_observations,
                test_orbit,
                range_shift_config=config.RANGE_SHIFT_CONFIG,
                cluster_link_config=config.CLUSTER_LINK_CONFIG,
                iod_config=config.IOD_CONFIG,
                od_config=config.OD_CONFIG,
                odp_config=config.ODP_CONFIG,
                out_dir=out_dir,
                if_exists="erase",
                logging_level=logging.INFO,
            )
            task.success(out_dir)

        except Exception as e:
            task.failure(out_dir, e)


if __name__ == "__main__":
    main()
