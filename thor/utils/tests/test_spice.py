import os

from ..spice import getSPICEKernels
from ..spice import setupSPICE

def test_getSPICEKernels():

    # Define some file paths
    lsk = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/latest_leapseconds.tls"))
    
    # Assert that the file doesn't exist yet
    assert os.path.isfile(lsk) == False
    
    getSPICEKernels(kernels=["LSK - Latest"])
    
    # Lets check the file downloaded
    assert os.path.isfile(lsk) == True

    # Clean up files
    os.remove(lsk)
        
def test_setupSPICE():
    
    # Make sure that THOR_SPICE is unset when not set up
    assert os.environ.get("THOR_SPICE") == None
    
    lsk = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/latest_leapseconds.tls"))
    getSPICEKernels(kernels=["LSK - Latest"])
    setupSPICE(kernels=["LSK - Latest"])
    
    # Test that THOR spice has been set up 
    assert os.environ.get("THOR_SPICE") == "True"

    # Clean up files
    os.remove(lsk)