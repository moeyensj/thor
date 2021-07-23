import pytest
import numpy as np

from ..tisserand import calcTisserandParameter

# From JPL's SBDB (2021-07-23) https://ssd.jpl.nasa.gov/sbdb.cgi
DAMOCLES = (11.85176279503047, 0.8658394058630634, 61.58514385448225)
CERES = (2.765655253487926, 0.07839201989374402, 10.58819557618916)
BIELA = (3.53465808340135, 0.751299, 13.2164)

def test_calcTisserandParameter_damocloids():

    Tp = calcTisserandParameter(*DAMOCLES, third_body="jupiter")

    # Damocles has a published T_jupiter of 1.158
    np.testing.assert_allclose(np.round(Tp, 3), 1.158)
    return

def test_calcTisserandParameter_asteroids():

    Tp = calcTisserandParameter(*CERES, third_body="jupiter")

    # Ceres has a published T_jupiter of 3.310
    np.testing.assert_allclose(np.round(Tp, 3), 3.310)
    return

def test_calcTisserandParameter_jupiterfamilycomets():

    Tp = calcTisserandParameter(*BIELA, third_body="jupiter")

    # 3D/Biela has a published T_jupiter of 2.531
    np.testing.assert_allclose(np.round(Tp, 3), 2.531)
    return

def test_calcTisserandParameter_raise():
    # Not a valid planet name
    with pytest.raises(ValueError):
        Tp = calcTisserandParameter(*CERES, third_body="")
