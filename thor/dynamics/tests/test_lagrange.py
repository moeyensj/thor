import pytest
import numpy as np
import numpy.testing as npt

from ...constants import Constants as c
from ..lagrange import (
    calc_lagrange_coefficients,
    apply_lagrange_coefficients
)

### Tests last updated: 2022-08-25

MU = c.MU

def test_calc_lagrange_coefficients_zerodt():
    r = np.array([1., 0., 0.])
    v = np.array([0.0002, 0.0002, 0.])
    dt = 0.0

    lagrange_coeffs, stumpff_coeffs, chi = calc_lagrange_coefficients(r, v, dt, mu=MU, max_iter=100, tol=1e-16)
    f, g, f_dot, g_dot = lagrange_coeffs
    assert f == 1.0
    assert g == 0.0
    assert f_dot == 0.0
    assert g_dot == 1.0
    return

def test_apply_lagrange_coefficients_zerodt():
    r0 = np.array([1., 0., 0.])
    v0 = np.array([0.0002, 0.0002, 0.])

    f, g, f_dot, g_dot = (1.0, 0.0, 0.0, 1.0)

    r1, v1 = apply_lagrange_coefficients(r0, v0, f, g, f_dot, g_dot)
    npt.assert_array_equal(r0, r1)
    npt.assert_array_equal(v0, v1)
    return