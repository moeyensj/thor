import os
import warnings
import pandas as pd
from astropy import units as u

from ...constants import DE44X as c
from ...utils import KERNELS_DE440
from ...utils import getSPICEKernels
from ...utils import setupSPICE
from ...testing import testOrbits
from ..kepler import convertOrbitalElements

MU = c.MU
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../testing/data"
)

def test_convertOrbitalElements_elliptical():
    """
    Read the test dataset for cartesian and keplerian states for each elliptical orbit target at each T0.
    Using THOR convert the cartesian states to keplerian states, and convert the keplerian states
    to cartesian states. Then compare how well the converted states agree to the ones pulled from
    Horizons.
    """
    getSPICEKernels(KERNELS_DE440)
    setupSPICE(KERNELS_DE440, force=True)

    # Read vectors and elements from test data set
    vectors_df = pd.read_csv(
        os.path.join(DATA_DIR, "vectors.csv")
    )
    elements_df = pd.read_csv(
        os.path.join(DATA_DIR, "elements.csv")
    )

    # Limit vectors and elements to elliptical orbits only
    elliptical_vectors = (
        (~vectors_df["orbit_class"].str.contains("Hyperbolic"))
        & (~vectors_df["orbit_class"].str.contains("Parabolic"))
    )
    elliptical_elements = (
        (~elements_df["orbit_class"].str.contains("Hyperbolic"))
        & (~elements_df["orbit_class"].str.contains("Parabolic"))
    )
    vectors_df = vectors_df[elliptical_vectors]
    elements_df = elements_df[elliptical_elements]

    # Pull state vectors and elements
    vectors = vectors_df[["x", "y", "z", "vx", "vy", "vz"]].values
    elements = elements_df[["a", "e", "incl", "Omega", "w", "M"]].values

    # Convert the keplerian states to cartesian states using THOR
    orbits_cartesian_converted = convertOrbitalElements(
        elements,
        "keplerian",
        "cartesian",
        mu=MU
    )

    # Convert the cartesian states to keplerian states using THOR
    orbits_keplerian_converted = convertOrbitalElements(
        vectors,
        "cartesian",
        "keplerian",
        mu=MU
    )

    # Conversion of cartesian orbits to keplerian orbits
    # is within this tolerance of Horizons
    testOrbits(
        orbits_keplerian_converted,
        elements,
        orbit_type="keplerian",
        position_tol=(1*u.cm),
        angle_tol=(10*u.nanoarcsecond),
        unitless_tol=(1e-12*u.dimensionless_unscaled)
    )

    # Conversion of keplerian orbits to cartesian orbits
    # is within this tolerance of Horizons
    testOrbits(
        orbits_cartesian_converted,
        vectors,
        orbit_type="cartesian",
        position_tol=(1*u.cm),
        velocity_tol=(1*u.mm/u.s),
        magnitude=True
    )

    return

def test_convertOrbitalElements_parabolilic():
    getSPICEKernels(KERNELS_DE440)
    setupSPICE(KERNELS_DE440, force=True)
    warnings.warn("Need to implement and test parabolic conversions!!!")
    return

def test_convertOrbitalElements_hyperbolic():
    """
    Read the test dataset for cartesian and keplerian states for each hyperbolic orbit target at each T0.
    Using THOR convert the cartesian states to keplerian states, and convert the keplerian states
    to cartesian states. Then compare how well the converted states agree to the ones pulled from
    Horizons.
    """
    getSPICEKernels(KERNELS_DE440)
    setupSPICE(KERNELS_DE440, force=True)

    # Read vectors and elements from test data set
    vectors_df = pd.read_csv(
        os.path.join(DATA_DIR, "vectors.csv")
    )
    elements_df = pd.read_csv(
        os.path.join(DATA_DIR, "elements.csv")
    )

    # Limit vectors and elements to elliptical orbits only
    hyperbolic_vectors = vectors_df["orbit_class"].str.contains("Hyperbolic")
    hyperbolic_elements = elements_df["orbit_class"].str.contains("Hyperbolic")
    vectors_df = vectors_df[hyperbolic_vectors]
    elements_df = elements_df[hyperbolic_elements]

    # Pull state vectors and elements
    vectors = vectors_df[["x", "y", "z", "vx", "vy", "vz"]].values
    elements = elements_df[["a", "e", "incl", "Omega", "w", "M"]].values

    # Convert the keplerian states to cartesian states using THOR
    orbits_cartesian_converted = convertOrbitalElements(
        elements,
        "keplerian",
        "cartesian",
        mu=MU
    )

    # Convert the cartesian states to keplerian states using THOR
    orbits_keplerian_converted = convertOrbitalElements(
        vectors,
        "cartesian",
        "keplerian",
        mu=MU
    )

    # Conversion of cartesian orbits to keplerian orbits
    # is within this tolerance of Horizons
    testOrbits(
        orbits_keplerian_converted,
        elements,
        orbit_type="keplerian",
        position_tol=(1*u.cm),
        angle_tol=(10*u.nanoarcsecond),
        unitless_tol=(1e-12*u.dimensionless_unscaled)
    )

    # Conversion of keplerian orbits to cartesian orbits
    # is within this tolerance of Horizons
    testOrbits(
        orbits_cartesian_converted,
        vectors,
        orbit_type="cartesian",
        position_tol=(1*u.cm),
        velocity_tol=(1*u.mm/u.s),
        magnitude=True
    )
    return
