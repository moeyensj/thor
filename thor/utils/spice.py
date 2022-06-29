import os
import logging
import numpy as np
import spiceypy as sp

from ..constants import KM_P_AU
from ..constants import S_P_DAY
from .astropy import _check_times
from .file_manager import FileManager

__all__ = [
    "NAIF_MAPPING",
    "KERNEL_URLS",
    "KERNELS_DE430",
    "KERNELS_DE440",
    "get_SPICE_kernels",
    "setup_SPICE",
    "use_DE430",
    "use_DE440",
    "use_default_DEXXX",
    "get_perturber_state",
    "shift_states_origin"
]

logger = logging.getLogger(__name__)

NAIF_MAPPING = {
    "solar system barycenter" : 0,
    "mercury barycenter" : 1,
    "venus barycenter" : 2,
    "earth barycenter" : 3,
    "mars barycenter" : 4,
    "jupiter barycenter" : 5,
    "saturn barycenter" : 6,
    "uranus barycenter" : 7,
    "neptune barycenter" : 8,
    "pluto barycenter" : 9,
    "sun" : 10,
    "mercury" : 199,
    "venus" : 299,
    "earth" : 399,
    "moon" : 301
}

KERNEL_URLS = {
    # Internal Name :  URL
    "latest_leapseconds.tls" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/latest_leapseconds.tls",
    "pck00010.tpc" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc",
    "earth_latest_high_prec.bpc" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc",
    "earth_200101_990628_predict.bpc" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_200101_990628_predict.bpc",
    "earth_assoc_itrf93.tf" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/planets/earth_assoc_itrf93.tf",
    "de430.bsp" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp",
    "de440.bsp" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp",
}

BASEKERNELS = [
    "latest_leapseconds.tls",
    "pck00010.tpc",
    "earth_200101_990628_predict.bpc",
    "earth_latest_high_prec.bpc",
]
KERNELS_DE430 = BASEKERNELS + ["de430.bsp"]
KERNELS_DE440 = BASEKERNELS + ["de440.bsp"]

def get_SPICE_kernels(
        kernels=KERNELS_DE440
    ):
    """
    Download SPICE kernels. If any already exist, check if they have been updated. If so, replace the
    outdated file with the latest version.

    SPICE kernels used by THOR:
    "latest_leapseconds.tls": latest_leapseconds.tls downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk,
    "earth_latest_high_prec.bpc": earth_latest_high_prec.bpc downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck,
    "earth_200101_990628_predict.bpc": earth_070425_370426_predict.bpc downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/
    "de430.bsp": de430.bsp downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/

    Only the leapsecond and Earth planetary constants kernels are checked for updates since these files are rather small (< 10 MB). The
    planetary ephemerides file is over 1.5 GB and is not checked for an update (these files are not updated regularly and are often released as
    different version with different physical assumptions)

    Parameters
    ----------
    kernels : list, optional
        Names of the kernels to download. By default, all kernels required by THOR are downloaded.
        Possible options are:
            "latest_leapseconds.tls"
            "earth_latest_high_prec.bpc"
            "earth_720101_070426.bpc"
            "earth_200101_990628_predict.bpc"
            "de430.bsp" or "de440.bsp"

    Returns
    -------
    None
    """
    file_manager = FileManager()
    for kernel in kernels:
        logger.info("Checking for {} kernel...".format(kernel))
        file_manager.download(KERNEL_URLS[kernel], sub_directory="spice")
    return

def setup_SPICE(
        kernels=KERNELS_DE440,
        force=False
    ):
    """
    Loads the leapsecond, the Earth planetary constants and the planetary ephemerides kernels into SPICE.

    Parameters
    ----------
    kernels : list, optional
        Names of the kernels to load. By default, all kernels required by THOR are loaded.
        Possible options are:
            "latest_leapseconds.tls"
            "earth_latest_high_prec.bpc"
            "earth_720101_070426.bpc"
            "earth_200101_990628_predict.bpc"
            "de430.bsp" or "de440.bsp"
    force : bool, optional
        Force spiceypy to set up kernels regardless of if SPICE is already set up.

    Returns
    -------
    None
    """
    pid = os.getpid()
    var_name = f"THOR_SPICE_pid{pid}"

    is_setup = var_name in os.environ.keys()
    is_ephemeris_correct = False
    if is_setup:
        is_ephemeris_correct = os.environ[var_name] in kernels

    if (is_setup or is_ephemeris_correct) and not force:
        logger.debug("SPICE is already enabled.")
    else:
        logger.debug("Enabling SPICE...")
        file_manager = FileManager()

        ephemeris_file = ""
        for kernel in kernels:
            file_name = os.path.basename(KERNEL_URLS[kernel])

            # Check if the current file is an ephemeris file
            if os.path.splitext(file_name)[1] == ".bsp":
                ephemeris_file = file_name

            if file_name not in file_manager.log.keys():
                err = ("{} not found. Please run thor.utils.get_SPICE_kernels to download SPICE kernels.")
                raise FileNotFoundError(err.format(file_name))
            sp.furnsh(file_manager.log[file_name]["location"])

        if ephemeris_file == "":
            err = (
                "SPICE has not recieved a planetary ephemeris file.\n" \
                "Please provide either de430.bsp, de440.bsp, or similar."
            )
            raise ValueError(err)

        os.environ[var_name] = ephemeris_file
        logger.debug("SPICE enabled.")
    return

def use_DE430(func):
    """
    Decorator: Configures SPICE (via spiceypy) to
    use the DE430 planetary ephemerides.
    """
    get_SPICE_kernels(KERNELS_DE430)
    setup_SPICE(KERNELS_DE430, force=True)

    def wrap(*args, **kwargs):
        return func(*args, **kwargs)

    return wrap

def use_DE440(func):
    """
    Decorator: Configures SPICE (via spiceypy) to
    use the DE440 planetary ephemerides.
    """
    get_SPICE_kernels(KERNELS_DE440)
    setup_SPICE(KERNELS_DE440, force=True)

    def wrap(*args, **kwargs):
        return func(*args, **kwargs)

    return wrap

# Set default to DE430
use_default_DEXXX = use_DE430

def get_perturber_state(
        body_name,
        times,
        frame="ecliptic",
        origin="heliocenter"
    ):
    """
    Query the JPL ephemeris files loaded in SPICE for the state vectors of desired perturbers.

    Major bodies and dynamical centers available:
        'solar system barycenter', 'sun',
        'mercury', 'venus', 'earth',
        'mars', 'jupiter', 'saturn',
        'uranus', 'neptune'

    Parameters
    ----------
    body_name : str
        Name of major body.
    times : `~astropy.time.core.Time` (N)
        Times at which to get state vectors.
    frame : {'equatorial', 'ecliptic'}
        Return perturber state in the equatorial or ecliptic J2000 frames.
    origin : {'barycenter', 'heliocenter'}
        Return perturber state with heliocentric or barycentric origin.

    Returns
    -------
    states : `~numpy.ndarray` (N, 6)
        Heliocentric ecliptic J2000 state vector with postion in AU
        and velocity in AU per day.
    """
    if origin == "barycenter":
        center = 0 # Solar System Barycenter
    elif origin == "heliocenter":
        center = 10 # Heliocenter
    else:
        err = ("origin should be one of 'heliocenter' or 'barycenter'")
        raise ValueError(err)

    if frame == "ecliptic":
        frame_spice = "ECLIPJ2000"
    elif frame == "equatorial":
        frame_spice = "J2000"
    else:
        err = (
            "frame should be one of {'equatorial', 'ecliptic'}"
        )
        raise ValueError(err)

    # Make sure SPICE is ready to roll
    setup_SPICE()

    # Check that times is an astropy time object
    _check_times(times, "times")

    # Convert MJD epochs in TDB to ET in TDB
    epochs_tdb = times.tdb
    epochs_et = np.array([sp.str2et('JD {:.16f} TDB'.format(i)) for i in epochs_tdb.jd])

    # Get position of the body in heliocentric ecliptic J2000 coordinates
    states = []
    for epoch in epochs_et:
        state, lt = sp.spkez(
            NAIF_MAPPING[body_name.lower()],
            epoch,
            frame_spice,
            'NONE',
            center
        )
        states.append(state)
    states = np.vstack(states)

    # Convert to AU and AU per day
    states = states / KM_P_AU
    states[:, 3:] = states[:, 3:] * S_P_DAY
    return states

def shift_states_origin(states, t0, origin_in="heliocenter", origin_out="barycenter"):
    """
    Shift the origin of the given Cartesian states. States should be expressed in
    ecliptic J2000 cartesian coordinates.

    Parameters
    ----------
    states : `~numpy.ndarray` (N, 6)
        states to shift to a different coordinate frame.
    t0 : `~astropy.time.core.Time` (N)
        Epoch at which states are defined.
    origin_in : {'heliocenter', 'barycenter'}
        Origin of the input states.
    origin_out : {'heliocenter', 'barycenter'}
        Desired origin of the output states.

    Returns
    -------
    states_shifted : `~numpy.ndarray` (N, 6)
        states shifted to the desired output origin.
    """
    _check_times(t0, "t0")

    states_shifted = states.copy()
    bary_to_helio = get_perturber_state("sun", t0, origin="barycenter")
    helio_to_bary = get_perturber_state("solar system barycenter", t0, origin="heliocenter")

    if origin_in == origin_out:
        return states_shifted
    elif origin_in == "heliocenter" and origin_out == "barycenter":
        states_shifted += bary_to_helio
    elif origin_in == "barycenter" and origin_out == "heliocenter":
        states_shifted += helio_to_bary
    else:
        err = (
            "states_in and states_out should be one of {'heliocenter', 'barycenter'}"
        )
        raise ValueError(err)

    return states_shifted