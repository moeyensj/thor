import numpy as np
import warnings
from astropy.time import Time
from astropy import units as u
from astroquery.jplhorizons import Horizons

from ...constants import Constants as c
from ..kepler import convertOrbitalElements
from ..kepler import _convertCartesianToKeplerian
from ..kepler import _convertKeplerianToCartesian

MU = c.G * c.M_SUN

def test_convertOrbitalElements():
    epochs = Time(["{}-02-02T00:00:00.000".format(i) for i in range(1993, 2050)], format="isot", scale="tdb")
    targets = [
        "Amor",
        "Eros", 
        "Eugenia",
        "Ceres"
    ] 
    for name in targets:
        target = Horizons(id=name, epochs=epochs.mjd)
        vectors = target.vectors()
        vectors = np.array(vectors["x", "y", "z", "vx", "vy", "vz"])
        vectors = vectors.view("float64").reshape(vectors.shape + (-1,))    
       
        elements = target.elements()
        elements = np.array(elements["a", "e", "incl", "Omega", "w", "M", "nu"])
        elements = elements.view("float64").reshape(elements.shape + (-1,))    
        
        np.testing.assert_allclose(elements[:, :6], convertOrbitalElements(vectors, "cartesian", "keplerian"))
        np.testing.assert_allclose(vectors, convertOrbitalElements(elements[:, :6], "keplerian", "cartesian"))

def test_convertCartesianToKeplerian_elliptical():
    epochs = Time(["{}-02-02T00:00:00.000".format(i) for i in range(1993, 2050)], format="isot", scale="tdb")
    targets = [
        "Amor",
        "Eros", 
        "Eugenia",
        "Ceres",
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
            np.testing.assert_allclose(_convertCartesianToKeplerian(v.reshape(1, -1), mu=MU), e.reshape(1, -1))

def test_convertCartesianToKeplerian_parabolic():
    warnings.warn("Need to implement and test parabolic conversions!!!")
    
def test_convertCartesianToKeplerian_hyperbolic():
    epochs = Time(["{}-02-02T00:00:00.000".format(i) for i in range(2017, 2023)], format="isot", scale="tdb")       
    iso_targets = [
        "1I/2017 U1", #Oumuamua  
        "C/2019 Q4" #Borisov
    ]
        
    for name in iso_targets:
        target = Horizons(id=name, epochs=epochs.mjd)
        vectors = target.vectors()
        vectors = np.array(vectors["x", "y", "z", "vx", "vy", "vz"])
        vectors = vectors.view("float64").reshape(vectors.shape + (-1,))    
       
        elements = target.elements()
        elements = np.array(elements["a", "q", "e", "incl", "Omega", "w", "M", "nu"])
        elements = elements.view("float64").reshape(elements.shape + (-1,))    
       
        for v, e in zip(vectors, elements):
            np.testing.assert_allclose(_convertCartesianToKeplerian(v.reshape(1, -1), mu=MU), e.reshape(1, -1))
        
def test_convertKeplerianToCartesian_elliptical():
    epochs = Time(["{}-02-02T00:00:00.000".format(i) for i in range(1993, 2050)], format="isot", scale="tdb")
    targets = [
        "Amor",
        "Eros", 
        "Eugenia",
        "Ceres"
    ] 

    for name in targets:
        target = Horizons(id=name, epochs=epochs.mjd)
        vectors = target.vectors()
        vectors = np.array(vectors["x", "y", "z", "vx", "vy", "vz"])
        vectors = vectors.view("float64").reshape(vectors.shape + (-1,))    
       
        elements = target.elements()
        elements = np.array(elements["a", "e", "incl", "Omega", "w", "M", "nu"])
        elements = elements.view("float64").reshape(elements.shape + (-1,))    
       
        for v, e in zip(vectors, elements):
            np.testing.assert_allclose(_convertKeplerianToCartesian(e.reshape(1, -1), mu=MU, maxIterations=100, tol=1e-15), v.reshape(1, -1))
            
def test_convertKeplerianToCartesian_parabolic():
    warnings.warn("Need to implement and test parabolic conversions!!!")

def test_convertKeplerianToCartesian_hyperbolic():
    epochs = Time(["{}-02-02T00:00:00.000".format(i) for i in range(2017, 2023)], format="isot", scale="tdb")       
    iso_targets = [
        "1I/2017 U1", #Oumuamua  
        "C/2019 Q4" #Borisov
    ]
        
    for name in iso_targets:
        target = Horizons(id=name, epochs=epochs.mjd)
        vectors = target.vectors()
        vectors = np.array(vectors["x", "y", "z", "vx", "vy", "vz"])
        vectors = vectors.view("float64").reshape(vectors.shape + (-1,))    
       
        elements = target.elements()
        elements = np.array(elements["a", "e", "incl", "Omega", "w", "M"])
        elements = elements.view("float64").reshape(elements.shape + (-1,))    
       
        for v, e in zip(vectors, elements):
            np.testing.assert_allclose(_convertKeplerianToCartesian(e.reshape(1, -1), mu=MU, maxIterations=100, tol=1e-15), v.reshape(1, -1))