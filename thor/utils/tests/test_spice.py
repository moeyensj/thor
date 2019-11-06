import os

from ..spice import getSPICEKernels
from ..spice import setupSPICE

def test_getSPICEKernels():

    # Define some file paths, we expect only obscodes and obscodes_old to be affected 
    # by any function call to getMPCObsCodeFile
    lsk = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/naif0011.tls"))
    
    # Assert that the file doesn't exist yet
    assert os.path.isfile(lsk) == False
    
    getSPICEKernels(kernels=["leapsecond"])
    
    # Lets check the file downloaded
    assert os.path.isfile(lsk) == True
        
def test_setupSPICE():
    
    # Make sure that THOR_SPICE is unset when not set up
    assert os.environ.get("THOR_SPICE") == None
    
    lsk = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/naif0011.tls"))
    setupSPICE(kernels=["leapsecond"])
    
    # Test that THOR spice has been set up 
    assert os.environ.get("THOR_SPICE") == "True"
    
    # Clean up files
    os.remove(lsk)