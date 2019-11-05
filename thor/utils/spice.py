import os

from .io import _downloadFile

__all__ = [
    "getSPICEKernels"
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
        "ephemerides" : [
            "de430.bsp", 
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp", 
            False
        ]
    }

def getSPICEKernels():
    """
    Download SPICE kernels. If any already exist, check if they have been updated. If so, replace the 
    outdated file with the latest version. 
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    """
    for kernel, info in KERNELS.items():
        print("Checking for {} kernel...".format(kernel))
        _downloadFile(os.path.join(os.path.dirname(__file__), "data", info[0]), info[1], update=info[2])
        print("")
    return
    