import numpy as np
import pandas as pd
import spiceypy as sp

from ..constants import Constants as c
from ..utils.astropy import _check_times
from ..utils.spice import setup_SPICE
from ..utils.spice import get_perturber_state
from ..utils.mpc import read_MPC_observatory_codes

__all__ = ["get_observer_state"]

R_EARTH = c.R_EARTH
OMEGA_EARTH = 2 * np.pi / 0.997269675925926

def get_observer_state(observatory_codes, observation_times, frame="ecliptic", origin="heliocenter"):
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
    observation_times : `~astropy.time.core.Time` (M)
        Epochs for which to find the observatory locations.
    frame : {'equatorial', 'ecliptic'}
        Return observer state in the equatorial or ecliptic J2000 frames.
    origin : {'barycenter', 'heliocenter'}
        Return observer state with heliocentric or barycentric origin.

    Returns
    -------
    observer_codes : `~numpy.ndarray` (N * M)
        Array containing the observatory code for each state
    observer_states : `~numpy.ndarray` (N * M, 7)
        Array with a column of MJDs (in UTC) and
        colums containing the observatories' states at the desired times.
    """
    if not isinstance(observatory_codes, (list, np.ndarray)):
        err = (
            "observatory_codes should be a list or `~numpy.ndarray`."
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

    setup_SPICE()

    # Check that times is an astropy time object
    _check_times(observation_times, "observation_times")

    observatories = read_MPC_observatory_codes()
    num_times = len(observation_times)
    N = len(observatory_codes) * num_times
    observer_codes = np.zeros(N, dtype=object)
    observer_states = np.zeros((N, 7), dtype=np.float64)

    for i, code in enumerate(observatory_codes):
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

        # Grab earth state vector
        state = get_perturber_state("earth", observation_times, frame=frame, origin=origin)

        # Convert MJD epochs in TDB to ET in TDB
        epochs_tdb = observation_times.tdb
        epochs_et = np.array([sp.str2et('JD {:.16f} TDB'.format(i)) for i in epochs_tdb.jd])

        # Grab rotaton matrices from ITRF93 to desired frame
        # The ITRF93 high accuracy Earth rotation model takes into account:
        # Precession:  1976 IAU model from Lieske.
        # Nutation:  1980 IAU model, with IERS corrections due to Herring et al.
        # True sidereal time using accurate values of TAI-UT1
        # Polar motion
        rotation_matrices = np.array([sp.pxform('ITRF93', frame_spice, i) for i in epochs_et])

        # Add o_vec + r_geo to get r_obs
        r_obs = np.array([rg + rm @ o_vec_ITRF93 for rg, rm in zip(state[:, :3], rotation_matrices)])

        # Calculate velocity
        v_obs = np.array([vg + rm @ (- OMEGA_EARTH * R_EARTH * np.cross(o_hat_ITRF93, np.array([0, 0, 1]))) for vg, rm in zip(state[:, 3:], rotation_matrices)])

        # Insert states into structured array
        observer_codes[i * num_times : (i + 1) * num_times] = code
        observer_states[i * num_times : (i + 1) * num_times, 0] = observation_times.utc.mjd
        observer_states[i * num_times : (i + 1) * num_times, 1] = r_obs[:, 0]
        observer_states[i * num_times : (i + 1) * num_times, 2] = r_obs[:, 1]
        observer_states[i * num_times : (i + 1) * num_times, 3] = r_obs[:, 2]
        observer_states[i * num_times : (i + 1) * num_times, 4] = v_obs[:, 0]
        observer_states[i * num_times : (i + 1) * num_times, 5] = v_obs[:, 1]
        observer_states[i * num_times : (i + 1) * num_times, 6] = v_obs[:, 2]

    return observer_codes, observer_states