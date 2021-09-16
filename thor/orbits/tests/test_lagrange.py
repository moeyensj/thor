import numpy as np

from ...constants import Constants as c
from ..lagrange import calcLagrangeCoeffs
from ..lagrange import applyLagrangeCoeffs


MU = c.MU

def test_calcLangrangeCoeffs_zerodt():
    r = np.array([1., 0., 0.])
    v = np.array([0.0002, 0.0002, 0.])
    dt = 0.0

    lagrange_coeffs, stumpff_coeffs, chi = calcLagrangeCoeffs(r, v, dt, mu=MU, max_iter=100, tol=1e-16)
    f, g, f_dot, g_dot = lagrange_coeffs
    assert f == 1.0
    assert g == 0.0
    assert f_dot == 0.0
    assert g_dot == 1.0
    return

def test_applyLagrangeCoeffs_zerodt():
    r0 = np.array([1., 0., 0.])
    v0 = np.array([0.0002, 0.0002, 0.])

    f, g, f_dot, g_dot = (1.0, 0.0, 0.0, 1.0)

    r1, v1 = applyLagrangeCoeffs(r0, v0, f, g, f_dot, g_dot)
    np.testing.assert_array_equal(r0, r1)
    np.testing.assert_array_equal(v0, v1)
    return