import os
import logging
import cProfile
import pandas as pd
from astropy.time import Time
from difi import analyzeObservations
from contextlib import contextmanager

from thor import (
    preprocessObservations,
    rangeAndShift,
    clusterAndLink,
)
from thor.orbits import (
    Orbits,
    initialOrbitDetermination,
    differentialCorrection,
    mergeAndExtendOrbits,
)

from thor import __version__

THREADS = 1


# load observations from a CSV
def load_observations(data_dir):
    observations = pd.read_csv(
        os.path.join(data_dir, "ztf_observations_610_624.csv"),
        sep=" ",
        index_col=False,
        low_memory=False,
    )
    observations.sort_values(by="jd", inplace=True)

    observations["observatory_code"] = ["I41"] * len(observations)
    observations["mjd_utc"] = Time(
        observations["jd"],
        scale="utc",
        format="jd"
    ).utc.mjd

    column_mapping = {
        "obs_id": "candid",
        "mjd": "mjd_utc",
        "RA_deg": "ra",
        "Dec_deg": "decl",
        "RA_sigma_deg": None,
        "Dec_sigma_deg": None,
        "observatory_code": "observatory_code",
        "obj_id": "ssnamenr",
    }
    mjd_scale = "utc"
    astrometric_errors = {
        "I41": [
            0.1/3600,
            0.1/3600
        ]
    }
    preprocessed_observations, preprocessed_associations = preprocessObservations(
        observations,
        column_mapping,
        mjd_scale=mjd_scale,
        astrometric_errors=astrometric_errors
    )
    return preprocessed_observations, preprocessed_associations


def assess_discoverability(observations, associations, min_obs=5):
    analysis_observations = observations.merge(
        associations,
        on="obs_id",
        how="left"
    )

    n_unknown = len(analysis_observations[analysis_observations["obj_id"] == "None"])
    unknown_labels = ["unknown{:d}".format(i) for i in range(n_unknown)]
    analysis_observations.loc[analysis_observations["obj_id"] == "None", "obj_id"] = unknown_labels

    column_mapping = {
        "obs_id" : "obs_id",
        "truth" : "obj_id",
        "linkage_id" : "cluster_id"
    }

    all_truths, findable_observations, summary = analyzeObservations(
        analysis_observations,
        column_mapping=column_mapping,
        min_obs=min_obs
    )
    return all_truths, findable_observations, summary


def define_test_orbit(observations):
    t0 = Time([
        observations["mjd_utc"].min()],
              scale="utc",
              format="mjd"
    )
    test_orbit = Orbits.fromHorizons(["2010 EG43"], t0)
    return test_orbit


def range_and_shift(observations, test_orbit):
    projected_observations = rangeAndShift(
        observations,
        test_orbit,
        cell_area=1000,
        backend="PYOORB",
        backend_kwargs={},
        threads=THREADS,
    )
    return projected_observations


def cluster_and_link(proj_observations, min_obs=5, min_arc_length=1.0):
    clusters, cluster_members = clusterAndLink(
        proj_observations,
        vx_range=[-0.1, 0.1],
        vy_range=[-0.1, 0.1],
        vx_bins=100,  # normally 300 x 300
        vy_bins=100,
        eps=0.005,
        min_samples=min_obs,
        min_arc_length=min_arc_length,
        threads=THREADS,
        identify_subsets=False
    )
    return clusters, cluster_members


def determine_orbits(
        proj_observations,
        cluster_members,
        min_obs=5,
        min_arc_length=1.0,
        contamination_percentage=20.0):
    iod_orbits, iod_orbit_members = initialOrbitDetermination(
        proj_observations,
        cluster_members,
        rchi2_threshold=100000,
        min_obs=min_obs,
        min_arc_length=min_arc_length,
        contamination_percentage=contamination_percentage,
        backend="PYOORB",
        threads=THREADS,
        iterate=False,
        identify_subsets=False
    )

    # Remove outliers
    iod_orbit_members = iod_orbit_members[(iod_orbit_members["outlier"] == 0)]

    for df in [iod_orbits, iod_orbit_members]:
        df.reset_index(
            inplace=True,
            drop=True
        )
    return iod_orbits, iod_orbit_members


def apply_differential_correction(
        iod_orbits,
        iod_orbit_members,
        proj_observations,
        min_obs=5,
        min_arc_length=1.0,
        contamination_percentage=20.0):
    od_orbits, od_orbit_members = differentialCorrection(
        iod_orbits,
        iod_orbit_members,
        proj_observations,
        rchi2_threshold=10.0,
        min_obs=min_obs,
        min_arc_length=min_arc_length,
        contamination_percentage=contamination_percentage,
        delta=1e-6,
        method="central",
        max_iter=10,
        threads=THREADS,
        fit_epoch=False,
        backend="PYOORB"
    )
    return od_orbits, od_orbit_members


def merge_and_extend(od_orbits, od_orbit_members, proj_observations,
                     min_obs=5):
    odp_orbits, odp_orbit_members = mergeAndExtendOrbits(
        od_orbits,
        od_orbit_members[(od_orbit_members["outlier"] == 0)],
        proj_observations,
        min_obs=min_obs,
        eps=1/3600,
        rchi2_threshold=10.0,
        contamination_percentage=0.0,
        delta=1e-8,
        max_iter=5,
        method="central",
        fit_epoch=False,
        orbits_chunk_size=100,
        observations_chunk_size=100000,
        threads=THREADS,
        backend="PYOORB",
    )
    return odp_orbits, odp_orbit_members


def init_debug_logging():
    logger = logging.getLogger("thor")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


@contextmanager
def cprofile(label):
    profile = cProfile.Profile()
    try:
        profile.enable()
        yield
    finally:
        profile.disable()
        profile.dump_stats(label)
        profile.print_stats()


def main():
    print(f"THOR: {__version__}")
    DATA_DIR = "../thor_data/2021_05_13"

    with cprofile("load_observations.profile"):
        observations, associations = load_observations(DATA_DIR)
    print(f"{len(observations)} observations loaded")
    print(f"{len(associations)} associations loaded")

    with cprofile("define_test_orbit.profile"):
        test_orbit = define_test_orbit(observations)
    print("test orbit defined")

    with cprofile("range_and_shift.profile"):
        projected_observations = range_and_shift(observations, test_orbit)
    print("range and shift complete")

    with cprofile("cluster_and_link.profile"):
        clusters, cluster_members = cluster_and_link(projected_observations)
    print("cluster and link complete")

    with cprofile("iod.profile"):
        iod_orbits, iod_orbit_members = determine_orbits(
            projected_observations, cluster_members,
        )
    print("iod complete")

    with cprofile("differential_correction.profile"):
        od_orbits, od_orbit_members = apply_differential_correction(
            iod_orbits, iod_orbit_members, projected_observations,
        )
    print("differential correction complete")

    with cprofile("merge_and_extend.profile"):
        odp_orbits, odp_orbit_members = merge_and_extend(
            od_orbits, od_orbit_members, projected_observations,
        )
    print("orbit extension complete")
    print("all done")


if __name__ == "__main__":
    main()
