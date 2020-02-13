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
TOL = 1e-15
MAX_ITER = 100

EPOCHS = Time(
    ["{}-02-02T00:00:00.000".format(i) for i in range(1993, 2050)], 
    format="isot", 
    scale="tdb"
)
ISO_EPOCHS = Time(
    ["{}-02-02T00:00:00.000".format(i) for i in range(2017, 2022)], 
    format="isot", 
    scale="tdb"
)
TARGETS = [
    "Amor",
    "Eros", 
    "Eugenia",
    "Ceres",
] 
ISO_TARGETS = [
    "1I/2017 U1", # Oumuamua  
    "C/2019 Q4" # Borisov
]
        

def test_convertOrbitalElements():
    for name in TARGETS:
        target = Horizons(id=name, epochs=EPOCHS.mjd)
        
        vectors = target.vectors().to_pandas()
        vectors = vectors[["x", "y", "z", "vx", "vy", "vz"]].values
        
        elements = target.elements().to_pandas()
        elements = elements[["a", "e", "incl", "Omega", "w", "M", "nu"]].values
        
        np.testing.assert_allclose(
            elements[:, :6], 
            convertOrbitalElements(
                vectors, 
                "cartesian", 
                "keplerian", 
                max_iter=MAX_ITER, 
                tol=TOL,
            )
        )
        
        np.testing.assert_allclose(
            vectors, 
            convertOrbitalElements(
                elements[:, :6], 
                "keplerian", 
                "cartesian",
                max_iter=MAX_ITER, 
                tol=TOL,
            )
        )
    return

def test_convertCartesianToKeplerian_elliptical():
    for name in TARGETS:
        target = Horizons(id=name, epochs=EPOCHS.mjd)
        vectors = target.vectors().to_pandas()
        vectors = vectors[["x", "y", "z", "vx", "vy", "vz"]].values
        
        elements = target.elements().to_pandas()
        elements = elements[["a", "q", "e", "incl", "Omega", "w", "M", "nu"]].values
        
        for v, e in zip(vectors, elements):
            np.testing.assert_allclose(
                _convertCartesianToKeplerian(
                    v.reshape(1, -1), 
                    mu=MU,
                ), 
                e.reshape(1, -1)
            )
    return

def test_convertCartesianToKeplerian_parabolic():
    warnings.warn("Need to implement and test parabolic conversions!!!")
    return
    
def test_convertCartesianToKeplerian_hyperbolic():
    for name in ISO_TARGETS:
        target = Horizons(id=name, epochs=ISO_EPOCHS.mjd)
        vectors = target.vectors().to_pandas()
        vectors = vectors[["x", "y", "z", "vx", "vy", "vz"]].values
        
        elements = target.elements().to_pandas()
        elements = elements[["a", "q", "e", "incl", "Omega", "w", "M", "nu"]].values
        
        for v, e in zip(vectors, elements):
            np.testing.assert_allclose(
                _convertCartesianToKeplerian(
                    v.reshape(1, -1),
                    mu=MU,
                ), 
                e.reshape(1, -1)
            )     
    return
        
def test_convertKeplerianToCartesian_elliptical():
    for name in TARGETS:
        target = Horizons(id=name, epochs=EPOCHS.mjd)
        vectors = target.vectors().to_pandas()
        vectors = vectors[["x", "y", "z", "vx", "vy", "vz"]].values   
       
        elements = target.elements().to_pandas()
        elements = elements[["a", "e", "incl", "Omega", "w", "M", "nu"]].values
       
        for v, e in zip(vectors, elements):
            np.testing.assert_allclose(
                _convertKeplerianToCartesian(
                    e.reshape(1, -1), 
                    mu=MU, 
                    max_iter=MAX_ITER,
                    tol=TOL,
                ), 
                v.reshape(1, -1)
            )
    return
            
def test_convertKeplerianToCartesian_parabolic():
    warnings.warn("Need to implement and test parabolic conversions!!!")

def test_convertKeplerianToCartesian_hyperbolic():
    for name in ISO_TARGETS:
        target = Horizons(id=name, epochs=ISO_EPOCHS.mjd)
        vectors = target.vectors().to_pandas()
        vectors = vectors[["x", "y", "z", "vx", "vy", "vz"]].values 
       
        elements = target.elements().to_pandas()
        elements = elements[["a", "e", "incl", "Omega", "w", "M", "nu"]].values
       
        for v, e in zip(vectors, elements):
            np.testing.assert_allclose(
                _convertKeplerianToCartesian(
                    e.reshape(1, -1), 
                    mu=MU,
                    max_iter=MAX_ITER, 
                    tol=TOL,
                ), 
                v.reshape(1, -1)
            )
    return