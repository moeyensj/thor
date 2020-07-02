import numpy as np
import spiceypy as sp
from astropy import units as u
from astropy.time import Time

from ....constants import Constants as c
from ....utils import getHorizonsVectors
from ..universal import propagateUniversal

MU = c.G * c.M_SUN

TARGETS = [
    "Amor",
    "Eros", 
    "Eugenia",
    "C/2019 Q4" # Borisov
] 
EPOCHS = [57257.0]

def test_propagateUniversal():
    """
    Using a selection of 4 asteroids, this function queries Horizons for an initial state vector at one epoch, then propagates
    that state to 1000 different times and compares each propagation to the SPICE 2-body propagator. 
    """
    dts = -np.linspace(0.01, 500, num=1000)
    
    for target in TARGETS: 
        for epoch in EPOCHS:
            times = Time(epoch + dts, scale="utc", format="mjd")

            # Grab vectors from Horizons at epoch
            vectors = getHorizonsVectors(target, times[:1])
            vectors = vectors[["x", "y", "z", "vx", "vy", "vz"]].values
            
            # Propagate vector to each new epoch (epoch + dt)
            spice_elements = []
            for dt in dts:
                spice_elements.append(sp.prop2b(MU, list(vectors[0, :]), dt))
            spice_elements = np.array(spice_elements)
            
            # Repeat but now using THOR's universal propagator
            vectors_new = propagateUniversal(
                vectors[0:1, :], 
                np.array([epoch]), 
                times.utc.mjd,  
                mu=MU, 
                max_iter=1000, 
                tol=1e-15
            )
               
            orbit_id = vectors_new[:, 0]
            new_epochs = vectors_new[:, 1]
            
            # Make sure the first column is a bunch of 0s since only one orbit was passed
            np.testing.assert_allclose(orbit_id, np.zeros(len(dts)))

            # Make sure the second column has all the new epochs
            np.testing.assert_allclose(new_epochs, dts + epoch)
            
            r_diff = np.linalg.norm(vectors_new[:, 2:5] - spice_elements[:, :3], axis=1)  * u.AU.to(u.cm)
            v_diff = np.linalg.norm(vectors_new[:, 5:] - spice_elements[:, 3:], axis=1) * (u.AU / u.d).to(u.mm / u.s)

            # Test position to within a 10 cm and velocity to within a mm/s
            np.testing.assert_allclose(r_diff, np.zeros(len(dts)), atol=10, rtol=0)
            np.testing.assert_allclose(v_diff, np.zeros(len(dts)), atol=1, rtol=0)