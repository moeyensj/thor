import os

from ..pyoorb import setupPYOORB
        
def test_setupPYOORB():
    
    # Make sure that THOR PYOORB is unset when not set up
    assert os.environ.get("THOR_PYOORB") == None
    
    setupPYOORB(ephemeris_file="de430.dat")
    
    # Test that THOR PYOORB has been set up 
    assert os.environ.get("THOR_PYOORB") == "True"
    
