import os
import logging
import spiceypy as sp

from .io import _downloadFile
from .io import _readFileLog

logger = logging.getLogger(__name__)

__all__ = [
    "KERNEL_URLS",
    "KERNELS_DE430",
    "KERNELS_DE440",
    "getSPICEKernels",
    "setupSPICE",
    "useDE430",
    "useDE440",
    "useDefaultDEXXX"
]

KERNEL_URLS = {
    # Internal Name :  URL
    "latest_leapseconds.tls" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/latest_leapseconds.tls",
    "pck00010.tpc" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc",
    "earth_latest_high_prec.bpc" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc",
    "earth_720101_070426.bpc" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_720101_070426.bpc",
    "earth_200101_990628_predict.bpc" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_200101_990628_predict.bpc",
    "earth_assoc_itrf93.tf" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/planets/earth_assoc_itrf93.tf",
    "de430.bsp" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp",
    "de440.bsp" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp",
}

BASEKERNELS = [
    "latest_leapseconds.tls",
    "pck00010.tpc",
    "earth_200101_990628_predict.bpc",
    "earth_720101_070426.bpc",
    "earth_latest_high_prec.bpc",
]
KERNELS_DE430 = BASEKERNELS + ["de430.bsp"]
KERNELS_DE440 = BASEKERNELS + ["de440.bsp"]

def getSPICEKernels(
        kernels=KERNELS_DE430
    ):
    """
    Download SPICE kernels. If any already exist, check if they have been updated. If so, replace the
    outdated file with the latest version.

    SPICE kernels used by THOR:
    "latest_leapseconds.tls": latest_leapseconds.tls downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk,
    "earth_latest_high_prec.bpc": earth_latest_high_prec.bpc downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck,
    "earth_720101_070426.bpc": earth_720101_070426.bpc downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck,
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
    for kernel in kernels:
        logger.info("Checking for {} kernel...".format(kernel))
        url = KERNEL_URLS[kernel]
        _downloadFile(os.path.join(os.path.dirname(__file__), "..", "data"), url)
    return

def setupSPICE(
        kernels=KERNELS_DE430,
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
        logger.info("SPICE is already enabled.")
    else:
        logger.info("Enabling SPICE...")
        log = _readFileLog(os.path.join(os.path.dirname(__file__), "..", "data/log.yaml"))

        ephemeris_file = ""
        for kernel in kernels:
            file_name = os.path.basename(KERNEL_URLS[kernel])

            # Check if the current file is an ephemeris file
            if os.path.splitext(file_name)[1] == ".bsp":
                ephemeris_file = file_name

            if file_name not in log.keys():
                err = ("{} not found. Please run thor.utils.getSPICEKernels to download SPICE kernels.")
                raise FileNotFoundError(err.format(file_name))
            sp.furnsh(log[file_name]["location"])

        if ephemeris_file == "":
            err = (
                "SPICE has not recieved a planetary ephemeris file.\n" \
                "Please provide either de430.bsp, de440.bsp, or similar."
            )
            raise ValueError(err)

        os.environ[var_name] = ephemeris_file
        logger.info("SPICE enabled.")
    return

def useDE430(func):
    """
    Decorator: Configures SPICE (via spiceypy) to
    use the DE430 planetary ephemerides.
    """
    getSPICEKernels(KERNELS_DE430)
    setupSPICE(KERNELS_DE430, force=True)

    def wrap(*args, **kwargs):
        return func(*args, **kwargs)

    return wrap

def useDE440(func):
    """
    Decorator: Configures SPICE (via spiceypy) to
    use the DE440 planetary ephemerides.
    """
    getSPICEKernels(KERNELS_DE440)
    setupSPICE(KERNELS_DE440, force=True)

    def wrap(*args, **kwargs):
        return func(*args, **kwargs)

    return wrap

# Set default to DE430
useDefaultDEXXX = useDE430