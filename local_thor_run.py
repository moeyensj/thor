from adam_core.ray_cluster import initialize_use_ray

from thor.config import Config
from thor.main import link_test_orbit
from thor.observations import Observations
from thor.orbit import TestOrbits

WORKING_DIR = "/opt/volumes/"
TEST_ORBIT_ID = "1115577"

# Load the configuration file
config = Config.parse_file(WORKING_DIR + "inputs/config.json")

# Load the test orbit
test_orbit = TestOrbits.from_parquet(WORKING_DIR + "inputs/test_orbits.parquet", filters=[("orbit_id", "=", TEST_ORBIT_ID)])

# Load the observations
observations = Observations.from_feather(WORKING_DIR + "inputs/observations.feather")

initialize_use_ray(num_cpus=8, object_store_bytes=8000000000)

# Run the link test orbit
for result in link_test_orbit(test_orbit, observations, working_dir=WORKING_DIR, config=config):
    print(result)