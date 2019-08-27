import numpy as np

from astropy.time import Time
from astropy import units as u
from astroquery.jplhorizons import Horizons

from .. import convertCartesianToKeplerian

def test_convertCartesianToKeplerian():
    # Grab orbital elements and state vectors of Ceres and Eros at the defined epoch 
    start_epoch = Time("1993-02-02T00:00:00.000", format="isot", scale="utc")
    
    ceres_horizons = Horizons(id="Ceres", epochs=start_epoch.mjd)
    ceres_elements = ceres_horizons.elements()
    ceres_elements = np.array(ceres_elements["a", "e", "incl", "Omega", "w", "M"]).view("float64")
    ceres_vectors  = ceres_horizons.vectors()
    ceres_vectors  = np.array(ceres_vectors["x", "y", "z", "vx", "vy", "vz"]).view("float64")
    
    eros_horizons = Horizons(id="Eros", epochs=start_epoch.mjd)
    eros_elements = eros_horizons.elements()
    eros_elements = np.array(eros_elements["a", "e", "incl", "Omega", "w", "M"]).view("float64")
    eros_vectors  = eros_horizons.vectors()
    eros_vectors = np.array(eros_vectors["x", "y", "z", "vx", "vy", "vz"]).view("float64")
    
    # Convert cartesian state vector to Keplerian elements 
    np.testing.assert_allclose(convertCartesianToKeplerian(eros_vectors), eros_elements)