import numpy as np

from ..coordinate_transforms import _angularToCartesian
from ..coordinate_transforms import eclipticAngularToCartesian
from ..coordinate_transforms import equatorialAngularToCartesian
from ..coordinate_transforms import calcNae

def __cardinality(func, params):
    # Define cardinal axes
    x_axis = np.array([[1, 0, 0]])
    y_axis = np.array([[0, 1, 0]]) 
    z_axis = np.array([[0, 0, 1]]) 
    
    # Test x-axis
    np.testing.assert_almost_equal(func(np.array([[0, 0]]), *params), x_axis)
    np.testing.assert_almost_equal(func(np.array([[np.pi, 0]]), *params), -x_axis)
    
    # Test y-axis
    np.testing.assert_almost_equal(func(np.array([[np.pi/2, 0]]), *params), y_axis)
    np.testing.assert_almost_equal(func(np.array([[3*np.pi/2, 0]]), *params), -y_axis)
    
    # Test z-axis
    np.testing.assert_almost_equal(func(np.array([[0, np.pi/2]]), *params), z_axis)
    np.testing.assert_almost_equal(func(np.array([[0, 3*np.pi/2]]), *params), -z_axis)
    
def test__angularToCartesian():
    __cardinality(_angularToCartesian, [np.array([1])])
    
def test_eclipticAngularToCartesian():
    __cardinality(eclipticAngularToCartesian, [np.array([1])])
    
def test_equatorialAngularToCartesian():
    __cardinality(equatorialAngularToCartesian, [np.array([1])])
    
def test_calcNae():
    __cardinality(calcNae, [])
    