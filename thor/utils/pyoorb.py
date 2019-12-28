import os
import warnings
import pyoorb as oo

__all__ = [
    "setupPYOORB"
]

def setupPYOORB(ephemeris_file="de430.dat"):
    """
    Initialize PyOORB with the designatied JPL ephemeris file.  
    
    Parameters
    ----------
    ephemeris_file : str, optional
        Which ephemeris file to load with PyOORB.
    
    Returns
    -------
    None
    """
    if "THOR_PYOORB" in os.environ.keys() and os.environ["THOR_PYOORB"] == "True":
        print("PYOORB is already enabled.")
    else:
        print("Enabling PYOORB...")
        if os.environ.get("OORB_DATA") == None:
            os.environ["OORB_DATA"] = os.path.join(os.environ["CONDA_PREFIX"], "share/openorb")
        # Prepare pyoorb
        ephfile = os.path.join(os.getenv('OORB_DATA'), ephemeris_file)
        err = oo.pyoorb.oorb_init(ephfile)
        if err == 0:
            os.environ["THOR_PYOORB"] = "True"
            print("Done.")
        else:
            warnings.warn("PYOORB returned error code: {}".format(err))
            
    return