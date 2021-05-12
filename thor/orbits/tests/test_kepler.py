import numpy as np
import warnings
from astropy.time import Time
from astropy import units as u

from ...constants import Constants as c
from ...utils import getHorizonsVectors
from ...utils import getHorizonsElements
from ...testing import testOrbits
from ..kepler import convertOrbitalElements
from ..kepler import _convertCartesianToKeplerian
from ..kepler import _convertKeplerianToCartesian


T0 = Time(
    ["{}-02-02T00:00:00.000".format(i) for i in range(1993, 2050)], 
    format="isot", 
    scale="tdb"
)
T0_ISO = Time(
    ["{}-02-02T00:00:00.000".format(i) for i in range(2017, 2022)], 
    format="isot", 
    scale="tdb"
)
TARGETS = [
    "Amor",
    "Eros", 
    "Eugenia",
    "Duende",
    "Ceres",
] 
TARGETS_ISO = [
    "1I/2017 U1", # Oumuamua  
    "C/2019 Q4" # Borisov
]

MU = c.MU       

def test_convertOrbitalElements_elliptical():
    """
    Query Horizons (via astroquery) for cartesian and keplerian states for each elliptical orbit target at each T0. 
    Using THOR convert the cartesian states to keplerian states, and convert the keplerian states
    to cartesian states. Then compare how well the converted states agree to the ones pulled from 
    Horizons.
    """
    # Query Horizons for cartesian states of each target at each T0
    orbits_cartesian_horizons = getHorizonsVectors(
        TARGETS, 
        T0,
        location="@sun",
        aberrations="geometric"
    )
    orbits_cartesian_horizons = orbits_cartesian_horizons[["x", "y", "z", "vx", "vy", "vz"]].values

    # Query Horizons for keplerian states of each target at each T0
    orbits_keplerian_horizons = getHorizonsElements(
        TARGETS, 
        T0,
        location="@sun",
    )
    orbits_keplerian_horizons = orbits_keplerian_horizons[["a", "e", "incl", "Omega", "w", "M"]].values
    
    # Convert the keplerian states to cartesian states using THOR
    orbits_cartesian_converted = convertOrbitalElements(
        orbits_keplerian_horizons, 
        "keplerian", 
        "cartesian",
        mu=MU
    )

    # Convert the cartesian states to keplerian states using THOR
    orbits_keplerian_converted = convertOrbitalElements(
        orbits_cartesian_horizons, 
        "cartesian",
        "keplerian", 
        mu=MU
    )

    # Conversion of cartesian orbits to keplerian orbits
    # is within this tolerance of Horizons
    testOrbits(
        orbits_keplerian_converted,
        orbits_keplerian_horizons,
        orbit_type="keplerian",
        position_tol=(1*u.cm),
        angle_tol=(1*u.microarcsecond),
        unitless_tol=(1e-10*u.dimensionless_unscaled)
    )

    # Conversion of keplerian orbits to cartesian orbits
    # is within this tolerance of Horizons
    testOrbits(
        orbits_cartesian_converted,
        orbits_cartesian_horizons,
        orbit_type="cartesian",
        position_tol=(1*u.cm),
        velocity_tol=(1*u.mm/u.s),
        magnitude=True
    )

    return

def test_convertOrbitalElements_parabolilic():
    warnings.warn("Need to implement and test parabolic conversions!!!")
    return

def test_convertOrbitalElements_hyperbolic():
    """
    Query Horizons (via astroquery) for cartesian and keplerian states for each hyperbolic orbit target at each T0. 
    Using THOR convert the cartesian states to keplerian states, and convert the keplerian states
    to cartesian states. Then compare how well the converted states agree to the ones pulled from 
    Horizons.
    """
    # Query Horizons for cartesian states of each target at each T0
    orbits_cartesian_horizons = getHorizonsVectors(
        TARGETS_ISO, 
        T0_ISO,
        location="@sun",
        aberrations="geometric"
    )
    orbits_cartesian_horizons = orbits_cartesian_horizons[["x", "y", "z", "vx", "vy", "vz"]].values

    # Query Horizons for keplerian states of each target at each T0
    orbits_keplerian_horizons = getHorizonsElements(
        TARGETS_ISO, 
        T0_ISO,
        location="@sun",
    )
    orbits_keplerian_horizons = orbits_keplerian_horizons[["a", "e", "incl", "Omega", "w", "M"]].values
    
    # Convert the keplerian states to cartesian states using THOR
    orbits_cartesian_converted = convertOrbitalElements(
        orbits_keplerian_horizons, 
        "keplerian", 
        "cartesian",
        mu=MU
    )

    # Convert the cartesian states to keplerian states using THOR
    orbits_keplerian_converted = convertOrbitalElements(
        orbits_cartesian_horizons, 
        "cartesian",
        "keplerian", 
        mu=MU
    )

    # Conversion of cartesian orbits to keplerian orbits
    # is within this tolerance of Horizons
    testOrbits(
        orbits_keplerian_converted,
        orbits_keplerian_horizons,
        orbit_type="keplerian",
        position_tol=(5*u.cm),
        angle_tol=(1*u.microarcsecond),
        unitless_tol=(1e-10*u.dimensionless_unscaled)
    )

    # Conversion of keplerian orbits to cartesian orbits
    # is within this tolerance of Horizons
    testOrbits(
        orbits_cartesian_converted,
        orbits_cartesian_horizons,
        orbit_type="cartesian",
        position_tol=(5*u.cm),
        velocity_tol=(1*u.mm/u.s),
        magnitude=True
    )
    return

