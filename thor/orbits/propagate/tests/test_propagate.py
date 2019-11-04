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
    
    epochs = [57257]
    dts = np.linspace(0.01, 500, num=1000)
    
    for name in targets: 
        for epoch in epochs:
            target = Horizons(id=name, epochs=epoch, location="@sun")
            for dt in dts:
                #print("Target: {}, Epoch: {}, dt: {}".format(name, epoch, dt))
                vectors = target.vectors()
                vectors = np.array(vectors["x", "y", "z", "vx", "vy", "vz"]).view("float64")
                elements = sp.prop2b(MU, list(vectors), dt)
                
                r, v = propagateUniversal(vectors[:3], vectors[3:], dt, mu=MU, maxIterations=1000, tol=1e-15)
                
                r_mag = np.sqrt(np.sum(r**2))
                r_spice_mag = np.sqrt(np.sum(elements[:3]**2))
                v_mag = np.sqrt(np.sum(v**2))
                v_spice_mag = np.sqrt(np.sum(elements[3:]**2))
                
                r_diff = (r_mag - r_spice_mag) / r_spice_mag
                v_diff = (v_mag - v_spice_mag) / v_spice_mag

                # Test position to within a meter and velocity to within a mm/s
                np.testing.assert_allclose(r_diff, np.zeros(3), atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(v_diff, np.zeros(3), atol=1e-10, rtol=1e-10)
    