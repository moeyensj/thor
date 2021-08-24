import numpy as np
import pandas as pd
from astropy.time import Time

from .orbits import Orbits
from .utils import assignPatchesSquare

__all__ = [
    "findAverageOrbits",
    "findTestOrbitsPatch",
    "selectTestOrbits",
]

def findAverageOrbits(
        ephemeris: pd.DataFrame,
        orbits: pd.DataFrame,
        d_values: list = None,
        element_type: str = "keplerian",
    ) -> pd.DataFrame:
    """
    Find the object with observations that represents
    the most average in terms of cartesian velocity and the
    heliocentric distance. Assumes that a subset of the designations in the orbits
    dataframe are identical to at least some of the designations in the observations
    dataframe. No propagation is done, so the orbits need to be defined at an epoch near
    the time of observations, for example like the midpoint or start of a two-week window.

    Parameters
    ----------
    ephemeris : `~pandas.DataFrame`
        DataFrame containing simulated ephemerides.
    orbits : `~pandas.DataFrame`
        DataFrame containing orbits for each unique object in observations.
    d_values : {list (N>=2), None}, optional
        If None, will find average orbit in all of observations. If a list, will find an
        average orbit between each value in the list. For example, passing d_values = [1.0, 2.0, 4.0] will
        mean an average orbit will be found in the following bins: (1.0 <= d < 2.0), (2.0 <= d < 4.0).
    element_type : {'keplerian', 'cartesian'}, optional
        Find average orbits using which elements. If 'keplerian' will use a-e-i for average,
        if 'cartesian' will use r, v.
        [Default = 'keplerian']

    Returns
    -------
    average_orbits : `~pandas.DataFrame`
        Orbits selected from
    """
    if element_type == "keplerian":
        d_col = "a"
    elif element_type == "cartesian":
        d_col = "r_au"
    else:
        err = (
            "element_type should be one of {'keplerian', 'cartesian'}"
        )
        raise ValueError(err)

    dataframe = pd.merge(orbits, ephemeris, on="orbit_id", how="right").copy()

    d_bins = []
    if d_values != None:
        for d_i, d_f in zip(d_values[:-1], d_values[1:]):
            d_bins.append(dataframe[(dataframe[d_col] >= d_i) & (dataframe[d_col] < d_f)])
    else:
        d_bins.append(dataframe)

    average_orbits = []

    for i, obs in enumerate(d_bins):
        if len(obs) == 0:
            continue

        if element_type == "cartesian":
            rv = obs[["vx", "vy", "vz", d_col]].values
            median = np.median(rv, axis=0)
            percent_diff = np.abs((rv - median) / median)

        else:
            aie = obs[["a", "i", "e"]].values
            median = np.median(aie, axis=0)
            percent_diff = np.abs((aie - median) / median)


        # Sum the percent differences
        summed_diff = np.sum(percent_diff, axis=1)

        # Find the minimum summed percent difference and call that
        # the average object
        index = np.where(summed_diff == np.min(summed_diff))[0][0]
        orbit_id = obs["orbit_id"].values[index]
        average_orbits.append(orbit_id)

    average_orbits = orbits[orbits["orbit_id"].isin(average_orbits)]
    average_orbits.reset_index(
        inplace=True,
        drop=True
    )

    return average_orbits

def findTestOrbitsPatch(ephemeris: pd.DataFrame) -> pd.DataFrame:
    """
    Find test orbits for a patch of ephemerides.

    Parameters
    ----------
    ephemeris : `~pandas.DataFrame`
        DataFrame containing predicted ephemerides (including aberrated cartesian state
        vectors) of an input catalog of orbits for a patch or small region of the
        sky.

    Returns
    -------
    test_orbits : `~pandas.DataFrame` (<=9)
        Up to 9 test orbits for the given patch of ephemerides.
    """
    observation_times = Time(
        ephemeris["mjd_utc"].values,
        scale="utc",
        format="mjd"
    )
    orbits = Orbits(
        ephemeris[["obj_x", "obj_y", "obj_z", "obj_vx", "obj_vy", "obj_vz"]].values,
        observation_times,
        ids=ephemeris["orbit_id"],
        orbit_type="cartesian"
    )
    orbits_df = orbits.to_df(
        include_units=False,
        include_keplerian=True
    )

    patch_id = ephemeris["patch_id"].unique()[0]

    test_orbits_hun1_patch = findAverageOrbits(
        ephemeris,
        orbits_df[(orbits_df["a"] < 2.06) & (orbits_df["a"] >= 1.7) & (orbits_df["e"] <= 0.1)],
        element_type="keplerian",
        d_values=[1.7, 2.06]
    )
    test_orbits_hun2_patch = findAverageOrbits(
        ephemeris,
        orbits_df[(orbits_df["a"] < 2.06) & (orbits_df["a"] >= 1.7) & (orbits_df["e"] > 0.1) & (orbits_df["e"] <= 0.2)],
        element_type="keplerian",
        d_values=[1.7, 2.06]
    )
    test_orbits_hun3_patch = findAverageOrbits(
        ephemeris,
        orbits_df[(orbits_df["a"] < 2.06) & (orbits_df["a"] >= 1.7) & (orbits_df["e"] > 0.2) & (orbits_df["e"] <= 0.4)],
        element_type="keplerian",
        d_values=[1.7, 2.06]
    )

    test_orbits_patch = findAverageOrbits(
        ephemeris,
        orbits_df[(orbits_df["e"] < 0.5)],
        element_type="keplerian",
        d_values=[2.06, 2.5, 2.82, 2.95, 3.27, 5.0, 50.0],
    )
    test_orbits_patch = pd.concat(
        [
            test_orbits_hun1_patch,
            test_orbits_hun2_patch,
            test_orbits_hun3_patch,
            test_orbits_patch
        ],
        ignore_index=True
    )
    test_orbits_patch.insert(0, "patch_id", patch_id)
    test_orbits_patch["r"] = np.linalg.norm(test_orbits_patch[["x", "y", "z"]].values, axis=1)
    test_orbits_patch.sort_values(
        by=["r"],
        inplace=True
    )

    return test_orbits_patch

def findTestOrbits_worker(ephemeris_list: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Find test orbits for a given list of patches of ephemerides.

    Parameters
    ----------
    ephemeris_list : list[`~pandas.DataFrame`]
        Small patches of ephemerides for which to find test orbits.

    Returns
    -------
    test_orbits : `~pandas.DataFrame`
        Test orbits for the given ephemerides.
    """
    test_orbits_list = []
    for ephemeris in ephemeris_list:

        test_orbits_patch = findTestOrbitsPatch(ephemeris)
        test_orbits_list.append(test_orbits_patch)

    test_orbits = pd.concat(test_orbits_list, ignore_index=True)
    return test_orbits

def selectTestOrbits(
        observations: pd.DataFrame,
        ephemeris: pd.DataFrame,
        ra_width: int = 15,
        dec_width: int = 15
    ) -> pd.DataFrame:
    """
    Select test orbits from ephemerides. Both the observations and the ephemerides
    are divided into patches, for each patch up to 9 test orbits are selected in bins
    of semi-major axis from the ephemerides. These orbits should represent
    the average in semi-major axis, eccentricity, and inclination for each bin.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame of observations containing at least the observation time ('mjd_utc'),
        and the astrometry ('RA_deg', 'Dec_deg').
    ephemeris : `~pandas.DataFrame`
        DataFrame of ephemerides containing at least the observation time ('mjd_utc'),
        and the astrometry ('RA_deg', 'Dec_deg'), and the cartesian state vector of the
        orbit at the time of the observation (must be correctly aberrated)
        ('obj_x', 'obj_y', 'obj_z', 'obj_vx', 'obj_vy', 'obj_vz').
    ra_width : int
        Width of patches in RA in degrees.
    dec_width : int
        Width of patches in Dec in degrees.

    Returns
    -------
    test_orbits : `~pandas.DataFrame`
        DataFrame containing test orbits. Can be read into an Orbits class using
        Orbits.from_df(test_orbits).
    """
    observations_ = observations[["mjd_utc", "RA_deg", "Dec_deg"]].copy()
    ephemeris_ = ephemeris.copy()

    # Divide the observations into patches of size ra_width x dec_width
    patch_ids = assignPatchesSquare(
        observations_["RA_deg"].values,
        observations_["Dec_deg"].values,
        ra_width=ra_width,
        dec_width=dec_width,
    )
    observations_["patch_id"] = patch_ids
    observations_.sort_values(
        by=["patch_id", "mjd_utc"],
        inplace=True
    )

    # Divide the ephemrides into patches of size ra_width x dec_width
    ephemeris_["patch_id"] = assignPatchesSquare(
        ephemeris_["RA_deg"].values,
        ephemeris_["Dec_deg"].values,
        ra_width=ra_width,
        dec_width=dec_width,
    )

    # Keep only ephemerides within the same patches as the observations
    ephemeris_ = ephemeris_[ephemeris_["patch_id"].isin(patch_ids)]
    ephemeris_.sort_values(
        by=["patch_id", "mjd_utc"],
        inplace=True,
    )

    grouped_ephemeris = ephemeris_.groupby(by=["patch_id"])
    ephemeris_split = [grouped_ephemeris.get_group(g).reset_index(drop=True) for g in grouped_ephemeris.groups]

    test_orbits = findTestOrbits_worker(ephemeris_split)

    return test_orbits