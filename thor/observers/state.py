import numpy as np
import pandas as pd
import spiceypy as sp
from astropy.time import Time
from typing import (
    List,
)

from ..constants import Constants as c
from ..utils.astropy import _check_times
from ..utils.spice import setup_SPICE
from ..utils.spice import get_perturber_state
from ..utils.mpc import read_MPC_observatory_codes

__all__ = ["get_observer_state"]

R_EARTH = c.R_EARTH
OMEGA_EARTH = 2 * np.pi / 0.997269675925926
Z_AXIS = np.array([0., 0., 1.])

def get_observer_state(
        observatory_codes: np.ndarray,
        observation_times: Time,
        frame: str = "ecliptic",
        origin: str = "heliocenter"
    ):
    """
    Find the heliocentric or barycentric ecliptic or equatorial J2000 state vectors for different observers or observatories at
    the desired epochs. Currently only supports ground-based observers.

    The Earth body-fixed frame used for calculations is the standard ITRF93, which takes into account:
        - precession (IAU-1976)
        - nutation (IAU-1980 with IERS corrections)
        - polar motion
    This frame is retrieved through SPICE.

    Parameters
    ----------
    observatory_codes : list or `~numpy.ndarray` (N)
        MPC observatory codes.
    observation_times : `~astropy.time.core.Time` (N)
        Epochs for which to find the observatory locations. This array can
        contain non-unique epochs.
    frame : {'equatorial', 'ecliptic'}
        Return observer state in the equatorial or ecliptic J2000 frames.
    origin : {'barycenter', 'heliocenter'}
        Return observer state with heliocentric or barycentric origin.

    Returns
    -------
    observer_states : `~numpy.ndarray` (N, 7)
        Array with a column of MJDs (in UTC) and
        colums containing the observatories' states at the desired times.
    """
    if not isinstance(observatory_codes, (np.ndarray)):
        err = (
            "observatory_codes should be a `~numpy.ndarray`."
        )
        raise TypeError(err)

    if frame == "ecliptic":
        frame_spice = "ECLIPJ2000"
    elif frame == "equatorial":
        frame_spice = "J2000"
    else:
        err = (
            "frame should be one of {'equatorial', 'ecliptic'}"
        )
        raise ValueError(err)

    if len(observatory_codes) != len(observation_times):
        err = (
            "observatory_codes and observation_times should have the same length."
        )
        raise ValueError(err)

    setup_SPICE()

    # Check that times is an astropy time object
    _check_times(observation_times, "observation_times")

    observatories = read_MPC_observatory_codes()
    observation_times_utc_jd = observation_times.utc.jd
    observation_times_utc_mjd = observation_times.utc.mjd
    unique_observation_times_jd = pd.unique(observation_times_utc_jd)
    unique_observation_times_mjd = pd.unique(observation_times_utc_mjd)
    unique_observation_times_jd = Time(
        unique_observation_times_jd,
        scale="utc",
        format="jd",
    ).tdb.jd
    unique_observatory_codes = pd.unique(observatory_codes)

    # Grab Earth state vectors
    # This function already optimizes the SPICE queries by only querying for
    # state vector at unique times
    geocenter_state = get_perturber_state(
        "earth",
        observation_times,
        frame=frame,
        origin=origin
    )

    # Initialize topocentric offset vector
    topocentric_offsets = np.zeros((len(unique_observatory_codes), 3), dtype=np.float64)

    # Grab unique topocentric offset vectors
    for i, code in enumerate(unique_observatory_codes):
        if (len(observatories[observatories["code"] == code]) == 0):
            err = (
                "Observatory code ('{}') could not be found in the MPC observatory code file. The MPC observatory code\n"
                "file may be missing this particular observatory code or the MPC observatory code is not valid."
            )
            raise ValueError(err.format(code))

        geodetics = np.array([*observatories[observatories["code"] == code][["longitude_deg", "cos", "sin"]][0]])
        if np.any(np.isnan(geodetics)):
            err = (
                "Observatory code ('{}') is missing information on Earth-based geodetic coordinates. The MPC observatory code\n"
                "file may be missing this information and/or the observatory might be space-based."
            )
            raise ValueError(err.format(code))

        # Get observer location on Earth
        longitude = geodetics[0]
        cos_phi = geodetics[1]
        sin_phi = geodetics[2]
        sin_longitude = np.sin(np.radians(longitude))
        cos_longitude = np.cos(np.radians(longitude))

        # Calculate pointing vector from geocenter to observatory
        o_hat_ITRF93 = np.array([
            cos_longitude * cos_phi,
            sin_longitude * cos_phi,
            sin_phi
        ])

        # Multiply pointing vector with Earth radius to get actual vector
        o_vec_ITRF93 = np.dot(R_EARTH, o_hat_ITRF93)

        # Add topocentric offset vector to array
        topocentric_offsets[i] = o_vec_ITRF93

    # Grab unique rotation matrices for the Earth's surface
    rotation_matrices = np.zeros((len(unique_observation_times_jd), 3, 3), dtype=np.float64)

    # Convert MJD epoch in TDB to ET in TDB
    epochs_et = np.array([sp.str2et(f'JD {i:.16f} TDB') for i in unique_observation_times_jd])
    for i, time in enumerate(epochs_et):
        # Grab rotaton matrices from ITRF93 to desired frame
        # The ITRF93 high accuracy Earth rotation model takes into account:
        # Precession:  1976 IAU model from Lieske.
        # Nutation:  1980 IAU model, with IERS corrections due to Herring et al.
        # True sidereal time using accurate values of TAI-UT1
        # Polar motion
        rotation_matrices[i] = sp.pxform('ITRF93', frame_spice, time)

    df_topocentric_offsets = pd.DataFrame({
        "observatory_code" : unique_observatory_codes,
        "topo_x" : topocentric_offsets[:, 0],
        "topo_y" : topocentric_offsets[:, 1],
        "topo_z" : topocentric_offsets[:, 2],
    })

    df_rotation_matrices = pd.DataFrame({
        "mjd_utc" : unique_observation_times_mjd,
        "rm_00" : rotation_matrices[:, 0, 0],
        "rm_01" : rotation_matrices[:, 0, 1],
        "rm_02" : rotation_matrices[:, 0, 2],
        "rm_10" : rotation_matrices[:, 1, 0],
        "rm_11" : rotation_matrices[:, 1, 1],
        "rm_12" : rotation_matrices[:, 1, 2],
        "rm_20" : rotation_matrices[:, 2, 0],
        "rm_21" : rotation_matrices[:, 2, 1],
        "rm_22" : rotation_matrices[:, 2, 2],
    })

    df_observatory_states = pd.DataFrame({
        "mjd_utc" : observation_times.utc.mjd,
        "observatory_code": observatory_codes,
    })
    #df_observatory_states["geocenter_states"] = geocenter_state

    df_observatory_states = df_observatory_states.merge(
        df_rotation_matrices,
        on="mjd_utc",
        how="left"
    )
    df_observatory_states = df_observatory_states.merge(
        df_topocentric_offsets,
        on="observatory_code",
        how="left"
    )

    rotation_matrices = np.zeros((len(df_observatory_states), 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            rotation_matrices[:, i, j] = df_observatory_states[f"rm_{i}{j}"].values

    topocentric_offsets = np.zeros((len(df_observatory_states), 3), dtype=np.float64)
    topocentric_offsets[:, 0] = df_observatory_states["topo_x"].values
    topocentric_offsets[:, 1] = df_observatory_states["topo_y"].values
    topocentric_offsets[:, 2] = df_observatory_states["topo_z"].values

    # Add o_vec + r_geo to get r_obs
    r_obs = geocenter_state[:, 0:3] + np.einsum("ijk,ik->ij", rotation_matrices, topocentric_offsets)

    # Calculate velocity
    v_obs = geocenter_state[:, 3:6] + np.einsum("ijk,ik->ij", rotation_matrices, -OMEGA_EARTH * np.cross(topocentric_offsets, Z_AXIS))

    observer_states = np.zeros((len(df_observatory_states), 7),  dtype=np.float64)
    observer_states[:, 0] = observation_times.utc.mjd
    observer_states[:, 1] = r_obs[:, 0]
    observer_states[:, 2] = r_obs[:, 1]
    observer_states[:, 3] = r_obs[:, 2]
    observer_states[:, 4] = v_obs[:, 0]
    observer_states[:, 5] = v_obs[:, 1]
    observer_states[:, 6] = v_obs[:, 2]

    return observer_states