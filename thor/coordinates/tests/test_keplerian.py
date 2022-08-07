import pytest
import numpy as np
import numpy.testing as npt
import spiceypy as sp


from ...constants import Constants as c
from ..keplerian import (
    _cartesian_to_keplerian,
    _keplerian_to_cartesian
)

MU = c.MU
RELATIVE_TOLERANCE = 0.
ABSOLUTE_TOLERANCE = 1e-14

### Tests last updated: 2022-08-06

def test__cartesian_to_keplerian_circular_prograde():
    ### Test Cartesian to Keplerian conversions for different
    ### circular (e = 0) orbits.

    # Circular orbit with 0 degree inclination.
    r = 1.0
    v = np.sqrt(MU / r)
    t0 = 59000.0
    cartesian = np.array([r, 0.0, 0.0, 0.0, v, 0.0])
    keplerian = _cartesian_to_keplerian(cartesian, t0, mu=MU)
    (a_actual, p_actual, q_actual, Q_actual,
        e_actual, i_actual, raan_actual, ap_actual,
        M_actual, nu_actual, n_actual, P_actual, tp_actual) = keplerian

    a_desired = p_desired = q_desired = Q_desired = 1.
    e_desired = 0.
    i_desired = 0.
    raan_desired = 0.
    ap_desired = 0.
    M_desired = 180.
    nu_desired = 180.
    # Period, mean motion and time of periapse passage are not declared to
    # high precision here so we won't test them to the same tolerance
    P_desired = 365.25
    n_desired = 360.0 / P_desired
    tp_desired = t0 - P_desired / 2

    actual = (
        a_actual, p_actual, q_actual, Q_actual, i_actual, e_actual,
        raan_actual, ap_actual, M_actual, nu_actual
    )
    desired = (
        a_desired, p_desired, q_desired, Q_desired, i_desired, e_desired,
        raan_desired, ap_desired, M_desired, nu_desired
    )
    npt.assert_allclose(
        actual,
        desired,
        atol=ABSOLUTE_TOLERANCE,
        rtol=RELATIVE_TOLERANCE
    )

    npt.assert_allclose(
        (P_actual, n_actual, tp_actual),
        (P_desired, n_desired, tp_desired),
        atol=0.001,
        rtol=1e-4
    )

    # Now let's test the spice conversion for the same orbit and
    # compare the result to the one calculated by THOR
    # We will ignore argument of periapse since the behavior in SPICE for orbits
    # with eccentricity near 0 is not well constrained. See Exception 5 and 6
    # in https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/oscelt.html
    # for more details. Copied below:
    #
    # 5)  If the eccentricity is determined to be zero, the argument of
    #     periapse is set to zero.
    #
    # 6)  If the eccentricity of the orbit is very close to but not
    #     equal to zero, the argument of periapse may not be accurately
    #     determined.
    #
    spice_keplerian = sp.oscelt(cartesian, t0, MU)
    spice_keplerian[2:6] = np.degrees(spice_keplerian[2:6])
    q_spice, e_spice, i_spice, raan_spice, ap_spice, M_spice = spice_keplerian[:6]

    actual = (q_actual, i_actual, e_actual, raan_actual, M_actual)
    desired = (q_spice, i_spice, e_spice, raan_spice, M_spice)

    npt.assert_allclose(
        actual,
        desired,
        atol=ABSOLUTE_TOLERANCE,
        rtol=RELATIVE_TOLERANCE
    )

def test__cartesian_to_keplerian_circular_retrograde():
    ### Test Cartesian to Keplerian conversions for different
    ### circular (e = 0) orbits.

    # Circular orbit with 180 degree inclination.
    r = 1.0
    v = np.sqrt(MU / r)
    t0 = 59000.0
    cartesian = np.array([r, 0.0, 0.0, 0.0, -v, 0.0])
    keplerian = _cartesian_to_keplerian(cartesian, t0, mu=MU)
    (a_actual, p_actual, q_actual, Q_actual,
        e_actual, i_actual, raan_actual, ap_actual,
        M_actual, nu_actual, n_actual, P_actual, tp_actual) = keplerian

    a_desired = p_desired = q_desired = Q_desired = 1.
    e_desired = 0.
    i_desired = 180.
    raan_desired = 0.
    ap_desired = 0.
    M_desired = 180.
    nu_desired = 180.
    # Period, mean motion and time of periapse passage are not declared to
    # high precision here so we won't test them to the same tolerance
    P_desired = 365.25
    n_desired = 360.0 / P_desired
    tp_desired = t0 - P_desired / 2

    actual = (
        a_actual, p_actual, q_actual, Q_actual, i_actual, e_actual,
        raan_actual, ap_actual, M_actual, nu_actual
    )
    desired = (
        a_desired, p_desired, q_desired, Q_desired, i_desired, e_desired,
        raan_desired, ap_desired, M_desired, nu_desired
    )
    npt.assert_allclose(
        actual,
        desired,
        atol=ABSOLUTE_TOLERANCE,
        rtol=RELATIVE_TOLERANCE
    )

    npt.assert_allclose(
        (P_actual, n_actual, tp_actual),
        (P_desired, n_desired, tp_desired),
        atol=0.001,
        rtol=1e-4
    )

    # Now let's test the spice conversion for the same orbit and
    # compare the result to the one calculated by THOR
    # We will ignore argument of periapse since the behavior in SPICE for orbits
    # with eccentricity near 0 is not well constrained. See Exception 5 and 6
    # in https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/oscelt.html
    # for more details. Copied below:
    #
    # 5)  If the eccentricity is determined to be zero, the argument of
    #     periapse is set to zero.
    #
    # 6)  If the eccentricity of the orbit is very close to but not
    #     equal to zero, the argument of periapse may not be accurately
    #     determined.
    #
    spice_keplerian = sp.oscelt(cartesian, t0, MU)
    spice_keplerian[2:6] = np.degrees(spice_keplerian[2:6])
    q_spice, e_spice, i_spice, raan_spice, ap_spice, M_spice = spice_keplerian[:6]

    actual = (q_actual, i_actual, e_actual, raan_actual, M_actual)
    desired = (q_spice, i_spice, e_spice, raan_spice, M_spice)

    npt.assert_allclose(
        actual,
        desired,
        atol=ABSOLUTE_TOLERANCE,
        rtol=RELATIVE_TOLERANCE
    )

