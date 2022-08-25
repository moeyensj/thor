import pytest
import numpy as np
import numpy.testing as npt

from ..tisserand import calc_tisserand_parameter

### Tests last updated: 2022-08-25

# From JPL's SBDB (2021-07-23) https://ssd.jpl.nasa.gov/sbdb.cgi
# a, e, i
DAMOCLES = (11.85176279503047, 0.8658394058630634, 61.58514385448225)
CERES = (2.765655253487926, 0.07839201989374402, 10.58819557618916)
BIELA = (3.53465808340135, 0.751299, 13.2164)

def test_calcTisserandParameter_damocloids():

    Tp = calc_tisserand_parameter(*DAMOCLES, third_body="jupiter")

    # Damocles has a published T_jupiter of 1.158
    npt.assert_allclose(np.round(Tp, 3), 1.158)
    return

def test_calcTisserandParameter_asteroids():

    Tp = calc_tisserand_parameter(*CERES, third_body="jupiter")

    # Ceres has a published T_jupiter of 3.310
    npt.assert_allclose(np.round(Tp, 3), 3.310)
    return

def test_calcTisserandParameter_jupiterfamilycomets():

    Tp = calc_tisserand_parameter(*BIELA, third_body="jupiter")

    # 3D/Biela has a published T_jupiter of 2.531
    npt.assert_allclose(np.round(Tp, 3), 2.531)
    return

def test_calcTisserandParameter_raise():
    # Not a valid planet name
    with pytest.raises(ValueError):
        Tp = calc_tisserand_parameter(*CERES, third_body="")
