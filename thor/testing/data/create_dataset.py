import os

import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplsbdb import SBDB

from thor import preprocessObservations
from thor.orbits import Orbits, generateEphemeris
from thor.utils import (
    getHorizonsElements,
    getHorizonsEphemeris,
    getHorizonsObserverState,
    getHorizonsVectors,
)

TARGETS = [
    # Atira
    "2020 AV2",
    "163693",
    # Aten
    "2010 TK7",
    "3753",
    # Apollo
    "54509",
    "2063",
    # Amor
    "1221",
    "433",
    "3908",
    # IMB
    "434",
    "1876",
    "2001",
    # MBA
    "2",
    "6",
    "6522",
    "202930",
    # Jupiter Trojans
    "911",
    "1143",
    "1172",
    "3317",
    # Centaur
    "5145",
    "5335",
    "49036",
    # Trans-Neptunian Objects
    "15760",
    "15788",
    "15789",
    # ISOs
    "A/2017 U1",
]
T0 = Time([58000], scale="utc", format="mjd")
DTS = np.arange(-30, 30, 1)
OBSERVATORY_CODES = ["500", "I11", "I41", "F51", "703"]
CARTESIAN_COLS = ["x", "y", "z", "vx", "vy", "vz"]
FLOAT_FORMAT = "%.16E"
NOISE_FACTOR = 0.5
RANDOM_SEED = 1719
DATA_DIR = os.path.abspath(os.path.dirname(__file__))


def getSBDBClass(obj_ids):
    data = {"targetname": [], "orbit_class": []}
    for obj_id in obj_ids:
        results = SBDB.query(obj_id)
        targetname = results["object"]["fullname"]
        orbit_class = results["object"]["orbit_class"]["name"]
        data["targetname"].append(targetname)
        data["orbit_class"].append(orbit_class)

    return pd.DataFrame(data)


def createTestDataset(
    targets=TARGETS,
    t0=T0,
    observatory_codes=OBSERVATORY_CODES,
    dts=DTS,
    out_dir=DATA_DIR,
):
    """
    Creates a test data set using data products from JPL Horizons and SBDB.
    The following files are created:
        vectors.csv : heliocentric cartesian state vectors in units of au and au per day
            for each target at t0.
        vectors_barycentric.csv : barycentric cartesian state vectors in units of au and au per day
            for each target at t0.
        elements.csv : keplerian elements in units of au and degrees for each target at t0.
        ephemeris.csv : ephemerides for each target as observed by each observer at t0 + dts.
        observer_states.csv : heliocentric cartesian state vectors in units of au and au per day
            for each observer at t1.
        observer_states_barycentric.csv : heliocentric cartesian state vectors in units
            of au and au per day for each observer at t1.
        observations.csv : Preprocessed observations for the elliptical orbits.
        associations.csv : Labels ('truths') for the preprocessed observations.
        orbits.csv : Elliptical orbits saved as THOR orbit class.

    Parameters
    ----------
    targets : list
        Names of targets for which to great data set.
    t0 : `~astropy.time.core.Time`
        Initial epoch at which to get state vectors and elements for
        each target.
    observatory_codes : list
        MPC observatory codes of observatories for which ephemerides should be generated.
    dts : `~numpy.ndarray` (N)
        Array of delta times (in units of days) relative to t0 with which ephemerides should be generated.
    out_dir : str
        Location to save data set.

    Returns
    -------
    None
    """
    # Set t1 and the observers dictionary
    t1 = t0 + dts
    observers = {code: t1 for code in observatory_codes}

    # Query JPL's SBDB for orbit class of each target
    orbit_classes = getSBDBClass(targets)

    # Get heliocentric state vectors for each target
    vectors = getHorizonsVectors(targets, t0, location="@sun")
    vectors = vectors.join(orbit_classes[["orbit_class"]])
    vectors["mjd_tdb"] = Time(
        vectors["datetime_jd"].values, scale="tdb", format="jd"
    ).tdb.mjd
    vectors = vectors[["targetname", "mjd_tdb"] + CARTESIAN_COLS + ["orbit_class"]]
    if out_dir is not None:
        vectors.to_csv(
            os.path.join(out_dir, "vectors.csv"), index=False, float_format=FLOAT_FORMAT
        )

    # Get barycentric state vectors for each target
    vectors_barycentric = getHorizonsVectors(targets, t0, location="@ssb")
    vectors_barycentric = vectors_barycentric.join(orbit_classes[["orbit_class"]])
    vectors_barycentric["mjd_tdb"] = Time(
        vectors_barycentric["datetime_jd"].values, scale="tdb", format="jd"
    ).tdb.mjd
    vectors_barycentric = vectors_barycentric[
        ["targetname", "mjd_tdb"] + CARTESIAN_COLS + ["orbit_class"]
    ]
    if out_dir is not None:
        vectors_barycentric.to_csv(
            os.path.join(out_dir, "vectors_barycentric.csv"),
            index=False,
            float_format=FLOAT_FORMAT,
        )

    # Get heliocentric elements for each target
    elements = getHorizonsElements(targets, t0)
    elements = elements.join(orbit_classes[["orbit_class"]])
    elements["mjd_tdb"] = Time(
        elements["datetime_jd"].values, scale="tdb", format="jd"
    ).tdb.mjd
    elements = elements[
        [
            "targetname",
            "mjd_tdb",
            "e",
            "q",
            "incl",
            "Omega",
            "w",
            "Tp_jd",
            "n",
            "M",
            "nu",
            "a",
            "Q",
            "P",
            "orbit_class",
        ]
    ]
    if out_dir is not None:
        elements.to_csv(
            os.path.join(out_dir, "elements.csv"),
            index=False,
            float_format=FLOAT_FORMAT,
        )

    # Get heliocentric observer states for each observatory
    observer_states = getHorizonsObserverState(
        observers.keys(), t1, origin="heliocenter"
    )
    observer_states["mjd_utc"] = Time(
        observer_states["datetime_jd"].values, scale="tdb", format="jd"
    ).utc.mjd
    observer_states = observer_states[["observatory_code", "mjd_utc"] + CARTESIAN_COLS]
    observer_states.sort_values(
        by=["observatory_code", "mjd_utc"], inplace=True, ignore_index=True
    )

    if out_dir is not None:
        observer_states.to_csv(
            os.path.join(out_dir, "observer_states.csv"),
            index=False,
            float_format=FLOAT_FORMAT,
        )

    # Get barycentric observer states for each observatory
    observer_states_barycentric = getHorizonsObserverState(
        observers.keys(), t1, origin="barycenter"
    )
    observer_states_barycentric["mjd_utc"] = Time(
        observer_states_barycentric["datetime_jd"].values, scale="tdb", format="jd"
    ).utc.mjd
    observer_states_barycentric = observer_states_barycentric[
        ["observatory_code", "mjd_utc"] + CARTESIAN_COLS
    ]
    observer_states_barycentric.sort_values(
        by=["observatory_code", "mjd_utc"], inplace=True, ignore_index=True
    )
    if out_dir is not None:
        observer_states_barycentric.to_csv(
            os.path.join(out_dir, "observer_states_barycentric.csv"),
            index=False,
            float_format=FLOAT_FORMAT,
        )

    # Get aberrated topocentric vectors for each target
    vectors_topo_aberrated_dfs = []
    for observatory in observatory_codes:
        vectors_topo_aberrated = getHorizonsVectors(
            targets, t1, location=observatory, aberrations="apparent"
        )
        vectors_topo_aberrated["observatory_code"] = observatory
        vectors_topo_aberrated["mjd_utc"] = Time(
            vectors_topo_aberrated["datetime_jd"].values, scale="tdb", format="jd"
        ).utc.mjd

        vectors_topo_aberrated.sort_values(
            by=["targetname", "observatory_code", "mjd_utc"],
            inplace=True,
            ignore_index=True,
        )

        # Make vectors heliocentric (but now the heliocentric vectors include the aberrations)
        for target in vectors_topo_aberrated["targetname"].unique():
            vectors_topo_aberrated.loc[
                vectors_topo_aberrated["targetname"].isin([target]), CARTESIAN_COLS
            ] += observer_states[
                observer_states["observatory_code"].isin([observatory])
            ][
                CARTESIAN_COLS
            ].values

        vectors_topo_aberrated_dfs.append(vectors_topo_aberrated)

    vectors_topo_aberrated = pd.concat(vectors_topo_aberrated_dfs, ignore_index=True)
    vectors_topo_aberrated.sort_values(
        by=["targetname", "observatory_code", "mjd_utc"],
        inplace=True,
        ignore_index=True,
    )

    # Get ephemerides for each target as observed by the observers
    ephemeris = getHorizonsEphemeris(targets, observers)
    ephemeris = ephemeris[["targetname", "observatory_code", "mjd_utc", "RA", "DEC"]]
    ephemeris.sort_values(
        by=["targetname", "observatory_code", "mjd_utc"],
        inplace=True,
        ignore_index=True,
    )
    ephemeris = ephemeris.join(vectors_topo_aberrated[CARTESIAN_COLS])
    if out_dir is not None:
        ephemeris.to_csv(
            os.path.join(out_dir, "ephemeris.csv"),
            index=False,
            float_format=FLOAT_FORMAT,
        )

    # Limit vectors and ephemerides to elliptical orbits only
    elliptical_vectors = (~vectors["orbit_class"].str.contains("Hyperbolic")) & (
        ~vectors["orbit_class"].str.contains("Parabolic")
    )
    vectors = vectors[elliptical_vectors]

    elliptical_ephemeris = ephemeris["targetname"].isin(vectors["targetname"].unique())
    ephemeris_df = ephemeris[elliptical_ephemeris]

    # Get the target names
    targets = vectors["targetname"].unique()

    # Get the initial epochs
    t0 = Time(vectors["mjd_tdb"].values, scale="tdb", format="mjd")

    # Pull state vectors
    vectors = vectors[["x", "y", "z", "vx", "vy", "vz"]].values

    # Create orbits class
    orbits = Orbits(vectors, t0, ids=targets)
    orbits.to_csv(os.path.join(out_dir, "orbits.csv"))

    # Construct observers' dictionary
    observers = {}
    for observatory_code in ["I41"]:
        observers[observatory_code] = Time(
            ephemeris_df[ephemeris_df["observatory_code"].isin([observatory_code])][
                "mjd_utc"
            ].unique(),
            scale="utc",
            format="mjd",
        )

    # Generate ephemerides with PYOORB
    ephemeris = generateEphemeris(orbits, observers, backend="PYOORB", num_jobs=1)
    observations = ephemeris.sort_values(
        by=["mjd_utc", "observatory_code"],
        ignore_index=True,
    )

    # Add noise observations
    np.random.seed(RANDOM_SEED)
    inds = np.arange(0, len(observations))
    inds_selected = np.random.choice(
        inds, int(NOISE_FACTOR * len(observations)), replace=False
    )
    observations_noise = observations.iloc[sorted(inds_selected)].copy()
    observations_noise["orbit_id"] = [
        "u{:08d}".format(i) for i in range(len(observations_noise))
    ]
    observations_noise.loc[:, ["RA_deg", "Dec_deg"]] = observations_noise[
        ["RA_deg", "Dec_deg"]
    ].values + np.random.normal(
        loc=1 / 3600, scale=30 / 3600, size=(len(observations_noise), 2)
    )

    observations = pd.concat([observations, observations_noise])
    observations = observations.sort_values(
        by=["mjd_utc", "observatory_code"],
        ignore_index=True,
    )

    column_mapping = {
        "obs_id": None,
        "mjd": "mjd_utc",
        "RA_deg": "RA_deg",
        "Dec_deg": "Dec_deg",
        "RA_sigma_deg": None,
        "Dec_sigma_deg": None,
        "observatory_code": "observatory_code",
        "obj_id": "orbit_id",
    }
    preprocessed_observations, preprocessed_associations = preprocessObservations(
        observations,
        column_mapping,
        astrometric_errors=[0.1 / 3600, 0.1 / 3600],
        mjd_scale="utc",
    )

    if out_dir is not None:
        preprocessed_observations.to_csv(
            os.path.join(out_dir, "observations.csv"),
            index=False,
            float_format=FLOAT_FORMAT,
        )
        preprocessed_associations.to_csv(
            os.path.join(out_dir, "associations.csv"),
            index=False,
            float_format=FLOAT_FORMAT,
        )

    return


if __name__ == "__main__":

    createTestDataset(out_dir=os.path.dirname(__file__))
