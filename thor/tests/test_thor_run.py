# from adam_core.ray_cluster import initialize_use_ray

# from thor.config import Config
# from thor.main import link_test_orbit
# from thor.observations import Observations
# from thor.orbit import TestOrbits

# test_orbit = TestOrbits.from_parquet("/opt/volumes/inputs/test_orbits.parquet", filters=[("orbit_id", "=", "932960")])

# observations = Observations.from_feather("/opt/volumes/inputs/observations.feather")
# config = Config.parse_file("/opt/volumes/inputs/config.json")

# initialize_use_ray(num_cpus=8, object_store_bytes=8000000000)
# link_gen = link_test_orbit(test_orbit, observations, working_dir="/opt/volumes/", config=config)

# for result in link_gen:
#     print(result)
    