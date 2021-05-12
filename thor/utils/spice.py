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
    "setupSPICE"
]

KERNEL_URLS = {
    # Internal Name :  URL
    "LSK - Latest" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/latest_leapseconds.tls",
    "Planetary Constants" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc",
    "Earth PCK - Latest High Accuracy" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc",
    "Earth PCK - Historical High Accuracy" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_720101_070426.bpc",
    "Earth PCK - Long Term Predict Low Accuracy" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_200101_990628_predict.bpc",
    "Earth FK" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/planets/earth_assoc_itrf93.tf",
    "Planetary SPK - DE430" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp",
    "Planetary SPK - DE440" : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp",
}

BASEKERNELS = [
    "LSK - Latest",  
    "Planetary Constants", 
    "Earth PCK - Long Term Predict Low Accuracy",
    "Earth PCK - Historical High Accuracy", 
    "Earth PCK - Latest High Accuracy", 
]
KERNELS_DE430 = BASEKERNELS + ["Planetary SPK - DE430"]
KERNELS_DE440 = BASEKERNELS + ["Planetary SPK - DE440"]

def getSPICEKernels(
        kernels=KERNELS_DE430
    ):
    """
    Download SPICE kernels. If any already exist, check if they have been updated. If so, replace the 
    outdated file with the latest version. 
    
    SPICE kernels used by THOR: 
    "LSK - Latest": latest_leapseconds.tls downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk,
    "Earth PCK - Latest High Accuracy": earth_latest_high_prec.bpc downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck,
    "Earth PCK - Historical High Accuracy": earth_720101_070426.bpc downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck,
    "Earth PCK - Long Term Predict Low Accuracy": earth_070425_370426_predict.bpc downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/
    "Planetary SPK - DE430": de430.bsp downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
    
    Only the leapsecond and Earth planetary constants kernels are checked for updates since these files are rather small (< 10 MB). The 
    planetary ephemerides file is over 1.5 GB and is not checked for an update (these files are not updated regularly and are often released as 
    different version with different physical assumptions)
    
    Parameters
    ----------
    kernels : list, optional
        Names of the kernels to download. By default, all kernels required by THOR are downloaded. 
        Possible options are:
            "Planetary Constants" 
            "Earth PCK - Latest High Accuracy"
            "Earth PCK - Historical High Accuracy"
            "Earth PCK - Long Term Predict Low Accuracy"
            "Planetary SPK - DE430"
    
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
        kernels=KERNELS_DE430
    ):
    """
    Loads the leapsecond, the Earth planetary constants and the planetary ephemerides kernels into SPICE. 
    
    Parameters
    ----------
    kernels : list, optional
        Names of the kernels to load. By default, all kernels required by THOR are loaded. 
        Possible options are:
            "Planetary Constants" 
            "Earth PCK - Latest High Accuracy"
            "Earth PCK - Historical High Accuracy"
            "Earth PCK - Long Term Predict Low Accuracy"
            "Planetary SPK"

    Returns
    -------
    None
    """
    pid = os.getpid()
    var_name = f"THOR_SPICE_pid{pid}"
    if var_name in os.environ.keys() and os.environ[var_name] == "True":
        logger.info("SPICE is already enabled.")
    else:
        logger.info("Enabling SPICE...")
        log = _readFileLog(os.path.join(os.path.dirname(__file__), "..", "data/log.yaml"))
        for kernel in kernels:
            file_name = os.path.basename(KERNEL_URLS[kernel])
            if file_name not in log.keys():
                err = ("{} not found. Please run thor.utils.getSPICEKernels to download SPICE kernels.")
                raise FileNotFoundError(err.format(file_name))
            sp.furnsh(log[file_name]["location"])
        os.environ[var_name] = "True"
        logger.info("SPICE enabled.")
    return