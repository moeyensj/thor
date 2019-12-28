import numpy as np
import spiceypy as sp
from astroquery.jplhorizons import Horizons

from ....constants import Constants as c
from ..universal import propagateUniversal

MU = c.G * c.M_SUN

def test_propagateUniversal():
    """
    Using a selection of 4 asteroids, this function queries Horizons for an initial state vector at one epoch, then propagates
    that state to 1000 different times and compares each propagation to the SPICE 2-body propagator. 
    """
    targets = [
        "Amor",
        "Eros", 
        "Eugenia",
        "C/2019 Q4" #Borisov
    ] 
    
    epochs = [57257.0]
    dts = np.linspace(0.01, 500, num=1000)
    
    for name in targets: 
        for epoch in epochs:
            # Grab vectors from Horizons at epoch
            target = Horizons(id=name, epochs=epoch, location="@sun")
            vectors = target.vectors()
            vectors = np.array(vectors["x", "y", "z", "vx", "vy", "vz"]).view("float64")
            vectors = vectors.reshape(-1, 6)
            
            # Propagate vector to each new epoch (epoch + dt)
            spice_elements = []
            for dt in dts:
                spice_elements.append(sp.prop2b(MU, list(vectors[0, :]), dt))
            spice_elements = np.array(spice_elements)
            
            # Repeat but now using THOR's universal propagator
            vectors_new = propagateUniversal(vectors[0:1, :], np.array(epochs), dts + epochs[0],  mu=MU, max_iter=1000, tol=1e-15)
               
            orbit_id = vectors_new[:, 0]
            new_epochs = vectors_new[:, 1]
            
            # Make sure the first column is a bunch of 0s since only one orbit was passed
            np.testing.assert_allclose(orbit_id, np.zeros(len(dts)))
            # Make sure the second column has all the new epochs
            np.testing.assert_allclose(new_epochs, dts + epochs[0])
            
            # Extract position and velocity components and compare them
            r = vectors_new[:, 2:5]
            v = vectors_new[:, 5:]
            
            r_mag = np.sqrt(np.sum(r**2, axis=1))
            v_mag = np.sqrt(np.sum(v**2, axis=1))
            
            r_spice_mag = np.sqrt(np.sum(spice_elements[:, :3]**2, axis=1))
            v_spice_mag = np.sqrt(np.sum(spice_elements[:, 3:]**2, axis=1))

            r_diff = (r_mag - r_spice_mag) / r_spice_mag
            v_diff = (v_mag - v_spice_mag) / v_spice_mag

            # Test position to within a meter and velocity to within a mm/s
            np.testing.assert_allclose(r_diff, np.zeros(len(dts)), atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(v_diff, np.zeros(len(dts)), atol=1e-10, rtol=1e-10)