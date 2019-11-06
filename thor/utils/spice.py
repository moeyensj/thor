import os
import spiceypy as sp

from .io import _downloadFile

__all__ = [
    "getSPICEKernels",
    "setupSPICE"
]

KERNELS = {
        # Internal Name : [File, URL, Update]
        "leapsecond" : [
            "naif0011.tls", 
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0011.tls",
            True
        ],
        "earth planetary constants" : [
            "earth_latest_high_prec.bpc",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc",
            True
        ],
        "planetary ephemerides" : [
            "de430.bsp", 
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp", 
            False
        ]
    }

def getSPICEKernels(kernels=["leapsecond", "earth planetary constants", "planetary ephemerides"]):
    """
    Download SPICE kernels. If any already exist, check if they have been updated. If so, replace the 
    outdated file with the latest version. 
    
    SPICE kernels used by THOR: 
    "leapsecond": naif0011.tls downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk,
    "earth planetary constants": earth_latest_high_prec.bpc downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck,
    "planetary ephemerides": de430.bsp downloaded from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
    
    Only the leapsecond and Earth planetary constants kernels are checked for updates since these files are rather small (< 10 MB). The 
    planetary ephemerides file is over 1.5 GB and is not checked for an update (these files are not updated regularly and are often released as 
    different version with different physical assumptions)
    
    Parameters
    ----------
    kernels : list, optional
        Names of the kernels to download. By default, all kernels required by THOR are downloaded. 
        Possible options are "leapsecond", "earth planetary constants", "planetary ephemerides". 
    
    Returns
    -------
    None
    """
    for kernel, info in KERNELS.items():
        print("Checking for {} kernel...".format(kernel))
        _downloadFile(os.path.join(os.path.dirname(__file__), "data", info[0]), info[1], update=info[2])
        print("")
    return

def setupSPICE(kernels=["leapsecond", "earth planetary constants", "planetary ephemerides"]):
    """
    Loads the leapsecond, the Earth planetary constants and the planetary ephemerides kernels into SPICE. 
    
    Parameters
    ----------
    kernels : list, optional
        Names of the kernels to load. By default, all kernels required by THOR are loaded. 
        Possible options are "leapsecond", "earth planetary constants", "planetary ephemerides". 
    
    Returns
    -------
    None
    """
    if "THOR_SPICE" in os.environ.keys() and os.environ["THOR_SPICE"] == "True":
        print("SPICE is already enabled.")
    else:
        print("Enabling SPICE...")
        for kernel in kernels:
            sp.furnsh(os.path.join(os.path.dirname(__file__), "data", KERNELS[kernel][0]))
        os.environ["THOR_SPICE"] = "True"
        print("Done.")
    return