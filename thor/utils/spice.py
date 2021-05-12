import os
import logging
import spiceypy as sp

from .io import _downloadFile
from .io import _readFileLog

logger = logging.getLogger(__name__)

__all__ = [
    "getSPICEKernels",
    "setupSPICE"
]

KERNELS = {
        # Internal Name : [File, URL]
        "LSK - Latest" : [
            "latest_leapseconds.tls", 
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/latest_leapseconds.tls"
        ],
        "Planetary Constants" : [
            "pck00010.tpc",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc"
        ],
        "Earth PCK - Latest High Accuracy" : [
            "earth_latest_high_prec.bpc",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc"
        ],
        "Earth PCK - Historical High Accuracy" : [
            "earth_720101_070426.bpc",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_720101_070426.bpc"
        ],
        "Earth PCK - Long Term Predict Low Accuracy" : [
            "earth_200101_990628_predict.bpc",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_200101_990628_predict.bpc"
        ],
        "Earth FK" : [
            "earth_assoc_itrf93.tf",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/planets/earth_assoc_itrf93.tf"
        ],
        "Planetary SPK" : [
            "de430.bsp", 
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp" 
        ]
    }

def getSPICEKernels(kernels=["LSK - Latest",  
                             "Planetary Constants", 
                             "Earth PCK - Long Term Predict Low Accuracy",
                             "Earth PCK - Historical High Accuracy", 
                             "Earth PCK - Latest High Accuracy", 
                             "Planetary SPK"]):
    """
    Download SPICE kernels. If any already exist, check if they have been updated. If so, replace the 
    outdated file with the latest version. 
    
    SPICE kernels used by THOR: 
    "LSK - Latest": latest_leapseconds.tls downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk,
    "Earth PCK - Latest High Accuracy": earth_latest_high_prec.bpc downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck,
    "Earth PCK - Historical High Accuracy": earth_720101_070426.bpc downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck,
    "Earth PCK - Long Term Predict Low Accuracy": earth_070425_370426_predict.bpc downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/
    "Planetary SPK": de430.bsp downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
    
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
            "Planetary SPK"
    
    Returns
    -------
    None
    """
    for kernel in kernels:
        logger.info("Checking for {} kernel...".format(kernel))
        _downloadFile(os.path.join(os.path.dirname(__file__), "..", "data"), KERNELS[kernel][1])
    return

def setupSPICE(kernels=["LSK - Latest",  
                        "Planetary Constants", 
                        "Earth PCK - Long Term Predict Low Accuracy",
                        "Earth PCK - Historical High Accuracy", 
                        "Earth PCK - Latest High Accuracy", 
                        "Planetary SPK"]
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
    if "THOR_SPICE" in os.environ.keys() and os.environ["THOR_SPICE"] == "True":
        logger.info("SPICE is already enabled.")
    else:
        logger.info("Enabling SPICE...")
        log = _readFileLog(os.path.join(os.path.dirname(__file__), "..", "data/log.yaml"))
        for kernel in kernels:
            file_name = KERNELS[kernel][0]
            if file_name not in log.keys():
                err = ("{} not found. Please run thor.utils.getSPICEKernels to download SPICE kernels.")
                raise FileNotFoundError(err.format(file_name))
            sp.furnsh(log[file_name]["location"])
        os.environ["THOR_SPICE"] = "True"
        logger.info("SPICE enabled.")
    return