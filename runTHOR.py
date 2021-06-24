import logging
import argparse
import pandas as pd

if __name__ == "__main__":

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

    from thor import runTHOR
    from thor.orbits import Orbits
    from thor.config import Config

    if not isinstance(args.config, str):
        config = Config
    else:
        config = Config.fromYaml(args.config)

    # Read observations
    preprocessed_observations = pd.read_csv(
        args.preprocessed_observations,
        index_col=False,
        dtype={
            "obs_id" : str
        }
    )

    # Read test orbits
    test_orbits = Orbits.from_csv(
        args.test_orbits
    )

    # Run THOR
    test_orbits_, recovered_orbits, recovered_orbit_members = runTHOR(
        preprocessed_observations,
        test_orbits,
        range_shift_config=config.RANGE_SHIFT_CONFIG,
        cluster_link_config=config.CLUSTER_LINK_CONFIG,
        iod_config=config.IOD_CONFIG,
        od_config=config.OD_CONFIG,
        odp_config=config.ODP_CONFIG,
        out_dir=args.out_dir,
        if_exists="continue",
        logging_level=logging.INFO
    )



