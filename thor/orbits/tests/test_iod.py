import numpy as np
import pandas as pd
from astropy.time import Time

from ...data_processing import preprocessObservations
from ..orbits import Orbits
from ..ephemeris import generateEphemeris
from ..iod import initialOrbitDetermination

TARGETS = [
    "Ivezic",
    "Eros",
    "Amor",
    "Duende",
    "Eugenia"
]
EPOCH = 54000.00
DT = np.arange(0, 15, 2)
T0 = Time(
    [EPOCH],
    format="mjd",
    scale="tdb",
)
T1 = Time(
    EPOCH + DT,
    format="mjd",
    scale="tdb"
)
OBSERVERS = {"I11" : T1}
ORBITS = Orbits.fromHorizons(
    TARGETS, T0
    )

def createOrbitDeterminationDataset(
        orbits,
        observers,
        astrometric_errors=[0.1/3600, 0.1/3600]
    ):

    # Generate simulated observations for each orbit
    observations = generateEphemeris(
        orbits,
        observers,
        backend="PYOORB",
        num_jobs=1
    )
    # Mark observations of "real" orbits as obsXXXXXXXXX
    observations["obs_id"] = ["obs{:08d}".format(i) for i in range(len(observations))]
    observations.sort_values(
        by=["mjd_utc"],
        inplace=True
    )
    observations.reset_index(
        inplace=True,
        drop=True
    )

    # Make a copy of the observations and offset the astrometry to create a set of
    # noise observations. Identify these as noiseXXXXXXXXX.
    observations_noise = observations.copy()
    observations_noise["orbit_id"] = ["u{:08d}".format(i) for i in range(len(observations_noise))]
    observations_noise.loc[:, ["RA_deg", "Dec_deg"]] += np.random.normal(loc=10/3600, scale=1/3600)
    observations_noise["obs_id"] = ["noise{:08d}".format(i) for i in range(len(observations_noise))]

    # Combine both types of observations into a single dataframe and
    # produce a preprocessed dataset
    observations = pd.concat([observations, observations_noise])
    observations.sort_values(
        by=["mjd_utc", "obs_id"],
        inplace=True
    )
    observations.reset_index(
        inplace=True,
        drop=True
    )

    column_mapping = {
        "obs_id" : "obs_id",
        "mjd" : "mjd_utc",
        "RA_deg" : "RA_deg",
        "Dec_deg" : "Dec_deg",
        "RA_sigma_deg" : None,
        "Dec_sigma_deg" : None,
        "observatory_code" : "observatory_code",
        "obj_id" : "orbit_id",
    }
    preprocessed_observations, preprocessed_associations = preprocessObservations(
        observations,
        column_mapping,
        astrometric_errors=astrometric_errors,
        mjd_scale="utc"
    )
    preprocessed_observations = preprocessed_observations.merge(
        observations[["obs_id", "obs_x", "obs_y", "obs_z"]],
    )

    # Now for each target create a pure and a partial linkage
    analysis_observations = preprocessed_observations.merge(preprocessed_associations, on="obs_id")
    findable_obj = analysis_observations[analysis_observations["obs_id"].str.contains("obs")]["obj_id"].unique()

    linkage_members = []
    linkage_num = 0
    for obj in findable_obj:
        obs_ids = preprocessed_associations[preprocessed_associations["obj_id"].isin([obj])]["obs_id"].values

        linkage_members_i = preprocessed_associations[preprocessed_associations["obs_id"].isin(obs_ids)][["obs_id"]].copy()
        linkage_members_i["linkage_id"] = ["pure{:02d}".format(linkage_num) for _ in range(len(linkage_members_i))]
        linkage_members.append(linkage_members_i)
        linkage_num += 1

        linkage_members_i = preprocessed_associations[preprocessed_associations["obs_id"].isin(obs_ids)][["obs_id"]].copy()
        linkage_members_i["linkage_id"] = ["partial{:02d}".format(linkage_num) for _ in range(len(linkage_members_i))]
        random_obs_ids = np.random.choice(
            obs_ids,
            size=np.random.choice([1, 2]),
            replace=False
        )
        linkage_members_i.loc[linkage_members_i["obs_id"].isin(random_obs_ids), "obs_id"] = linkage_members_i[linkage_members_i["obs_id"].isin(random_obs_ids)]["obs_id"].str.replace("obs", "noise")

        linkage_members.append(linkage_members_i)
        linkage_num += 1

    linkage_members = pd.concat(linkage_members)
    linkage_members = linkage_members[["linkage_id", "obs_id"]]
    linkage_members.reset_index(
        inplace=True,
        drop=True
    )

    return preprocessed_observations, preprocessed_associations, linkage_members

def test_initialOrbitDetermination_outlier_rejection():
    np.random.seed(42)

    preprocessed_observations, preprocessed_associations, linkage_members = createOrbitDeterminationDataset(
        ORBITS,
        OBSERVERS
    )

    iod_orbits, iod_orbit_members = initialOrbitDetermination(
        preprocessed_observations,
        linkage_members,
        observation_selection_method='combinations',
        min_obs=6,
        rchi2_threshold=1000,
        contamination_percentage=20.0,
        iterate=False,
        light_time=True,
        linkage_id_col='linkage_id',
        identify_subsets=False,
        num_jobs=1,
        backend='PYOORB',
        backend_kwargs={}
    )

    # Make sure all outlier observations were appropriate identified
    assert np.all(iod_orbit_members[iod_orbit_members["obs_id"].str.contains("noise")]["outlier"] == 1)

    # Make sure no "real" observations were marked as outliers
    assert np.all(iod_orbit_members[iod_orbit_members["obs_id"].str.contains("obs")]["outlier"] == 0)

    return
