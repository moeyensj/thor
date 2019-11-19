import numpy as np
from astropy.time import Time
from astropy import units as u
from astroquery.jplhorizons import Horizons

from ...constants import Constants as c
from ..kepler import convertCartesianToKeplerian

MU = c.G * c.M_SUN

def test_convertCartesianToKeplerian():
    epochs = Time(["{}-02-02T00:00:00.000".format(i) for i in range(1993, 2050)], format="isot", scale="tdb")
    
    targets = [
        "Amor",
        "Eros", 
        "Eugenia",
        "Ceres",
        "C/2019 Q4" #Borisov
    ] 

    for name in targets:
        target = Horizons(id=name, epochs=epochs.mjd)
        vectors = target.vectors()
        vectors = np.array(vectors["x", "y", "z", "vx", "vy", "vz"])
        vectors = vectors.view("float64").reshape(vectors.shape + (-1,))    
       
        elements = target.elements()
        elements = np.array(elements["a", "q", "e", "incl", "Omega", "w", "M", "nu"])
        elements = elements.view("float64").reshape(elements.shape + (-1,))    
     
        for v, e in zip(vectors, elements):
            np.testing.assert_allclose(convertCartesianToKeplerian(v, mu=MU), e)