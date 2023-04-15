import os

import numpy as np
import pandas as pd

from ..main import clusterAndLink, rangeAndShift
from ..orbits import Orbits

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../testing/data")


def test_rangeAndShift():
    """
    Read the test observations and orbits, use the orbits as test orbits and make sure their observations
    are all recovered near 0,0 on the gnomonic tangent plane.
    """
    preprocessed_observations = pd.read_csv(
        os.path.join(DATA_DIR, "observations.csv"), index_col=False
    )
    preprocessed_associations = pd.read_csv(
        os.path.join(DATA_DIR, "associations.csv"), index_col=False
    )
    orbits = Orbits.from_csv(os.path.join(DATA_DIR, "orbits.csv"))

    # Remove noise observations
    noise_ids = preprocessed_associations[
        preprocessed_associations["obj_id"].str.contains("^u[0-9]{8}", regex=True)
    ]["obs_id"].values
    preprocessed_observations = preprocessed_observations[
        ~preprocessed_observations["obs_id"].isin(noise_ids)
    ]
    preprocessed_associations = preprocessed_associations[
        ~preprocessed_associations["obs_id"].isin(noise_ids)
    ]

    for i in range(len(orbits)):
        # Select test orbit
        orbit = orbits[i : i + 1]

        # Range and shift the observations using the test orbit
        projected_observations = rangeAndShift(
            preprocessed_observations,
            orbit,
            cell_area=0.1,
            backend="PYOORB",
            num_jobs=1,
            parallel_backend="mp",
        )
        analyis_projected_observations = projected_observations.merge(
            preprocessed_associations, how="left", on="obs_id"
        )
        observations_mask = analyis_projected_observations["obj_id"].isin(
            [orbit.ids[0]]
        )

        # Test that the gnomonic coordinates for this object all lie within 1 arcsecond of 0.0
        gnomonic_offset = np.linalg.norm(
            analyis_projected_observations[observations_mask][
                ["theta_x_deg", "theta_y_deg"]
            ].values,
            axis=1,
        )
        assert np.all(gnomonic_offset <= 1 / 3600)

        # Test that all this object's observations were recovered with range and shift
        obs_ids = preprocessed_associations[
            preprocessed_associations["obj_id"].isin([orbit.ids[0]])
        ]["obs_id"].values
        assert np.all(
            analyis_projected_observations[observations_mask]["obs_id"].values
            == obs_ids
        )

    return


def test_clusterAndLink():
    """
    Read the test observations and orbits, use the orbits as test orbits and make sure their observations
    are recovered as a single cluster.
    """
    preprocessed_observations = pd.read_csv(
        os.path.join(DATA_DIR, "observations.csv"), index_col=False
    )
    preprocessed_associations = pd.read_csv(
        os.path.join(DATA_DIR, "associations.csv"), index_col=False
    )
    orbits = Orbits.from_csv(os.path.join(DATA_DIR, "orbits.csv"))

    # Remove noise observations
    noise_ids = preprocessed_associations[
        preprocessed_associations["obj_id"].str.contains("^u[0-9]{8}", regex=True)
    ]["obs_id"].values
    preprocessed_observations = preprocessed_observations[
        ~preprocessed_observations["obs_id"].isin(noise_ids)
    ]
    preprocessed_associations = preprocessed_associations[
        ~preprocessed_associations["obs_id"].isin(noise_ids)
    ]

    for i in range(len(orbits)):
        # Select test orbit
        orbit = orbits[i : i + 1]

        # Range and shift the observations using the test orbit
        projected_observations = rangeAndShift(
            preprocessed_observations,
            orbit,
            cell_area=0.1,
            backend="PYOORB",
            num_jobs=1,
            parallel_backend="mp",
        )
        analyis_projected_observations = projected_observations.merge(
            preprocessed_associations, how="left", on="obs_id"
        )
        observations_mask = analyis_projected_observations["obj_id"].isin(
            [orbit.ids[0]]
        )

        # Cluster the observations with a single 0,0 velocity combination
        clusters, cluster_members = clusterAndLink(
            projected_observations,
            vx_values=[0],
            vy_values=[0],
            eps=1 / 3600,
            min_obs=5,
            min_arc_length=1.0,
            num_jobs=1,
        )
        analyis_cluster_members = cluster_members.merge(
            preprocessed_associations, how="left", on="obs_id"
        )

        # Test that all this object's observations were recovered in a single cluster
        obs_ids = preprocessed_associations[
            preprocessed_associations["obj_id"].isin([orbit.ids[0]])
        ]["obs_id"].values
        assert np.all(np.in1d(obs_ids, analyis_cluster_members["obs_id"].values))

    return
