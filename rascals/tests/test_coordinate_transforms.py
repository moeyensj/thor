import numpy as np

from ..coordinate_transforms import equatorialToEclipticCartesian
from ..coordinate_transforms import eclipticToEquatorialCartesian
from ..coordinate_transforms import equatorialAngularToCartesian
from ..coordinate_transforms import eclipticAngularToCartesian
from ..coordinate_transforms import equatorialCartesianToAngular
from ..coordinate_transforms import eclipticCartesianToAngular
from ..coordinate_transforms import equatorialToEclipticAngular
from ..coordinate_transforms import eclipticToEquatorialAngular

OBLIQUITY = np.radians(23.43928)
TRANSFORM_EQ2EC = np.matrix([[1, 0 , 0],
                             [0,  np.cos(OBLIQUITY), np.sin(OBLIQUITY)],
                             [0, -np.sin(OBLIQUITY), np.cos(OBLIQUITY)]])
TRANSFORM_EC2EQ = np.matrix([[1, 0 , 0],
                             [0, np.cos(OBLIQUITY), -np.sin(OBLIQUITY)],
                             [0, np.sin(OBLIQUITY),  np.cos(OBLIQUITY)]])

# Define cardinal axes in cartesian coordinates
x_axis = np.array([[1, 0, 0]])
y_axis = np.array([[0, 1, 0]])
z_axis = np.array([[0, 0, 1]])
axes = np.vstack([x_axis, y_axis, z_axis])

# Define cardinal axes in angular coordinates
x_axis_ang = np.array([[0, 0, 1]])
neg_x_axis_ang = np.array([[np.pi, 0, 1]])
y_axis_ang = np.array([[np.pi/2, 0, 1]]) 
neg_y_axis_ang = np.array([[3*np.pi/2, 0, 1]]) 
z_axis_ang = np.array([[0, np.pi/2, 1]]) 
neg_z_axis_ang = np.array([[0, -np.pi/2, 1]]) 

# Cartesian to Cartesian

def test_eclipticEquatorialCartesianSymmetry():
    # Test transformation matrices
    np.testing.assert_array_almost_equal(TRANSFORM_EC2EQ.T, eclipticToEquatorialCartesian(axes))
    np.testing.assert_array_almost_equal(TRANSFORM_EQ2EC.T, equatorialToEclipticCartesian(axes))
    
def test_eclipticEquatorialX():
    # Check that the x-axis for ecliptic and equatorial is the same
    np.testing.assert_array_almost_equal(equatorialToEclipticCartesian(axes[:,0]), eclipticToEquatorialCartesian(axes[:,0]))
    
# Angular to Cartesian

def __cardinalityCart(func, args=[], kwargs={}):
    # Test x-axis
    np.testing.assert_array_almost_equal(func(x_axis_ang, *args, **kwargs), x_axis)
    np.testing.assert_array_almost_equal(func(neg_x_axis_ang, *args, **kwargs), -x_axis)
    
    # Test y-axis
    np.testing.assert_array_almost_equal(func(y_axis_ang, *args, **kwargs), y_axis)
    np.testing.assert_array_almost_equal(func(neg_y_axis_ang, *args, **kwargs), -y_axis)
    
    # Test z-axis
    np.testing.assert_array_almost_equal(func(z_axis_ang, *args, **kwargs), z_axis)
    np.testing.assert_array_almost_equal(func(neg_z_axis_ang, *args, **kwargs), -z_axis)

def test_eclipticAngularToCartesian():
    __cardinalityCart(eclipticAngularToCartesian, [np.array([1])])
    
def test_equatorialAngularToCartesian():
    __cardinalityCart(equatorialAngularToCartesian, [np.array([1])])
    
# Cartesian to Angular
    
def __cardinalityAng(func, args=[], kwargs={}):
    # Test x-axis
    np.testing.assert_array_almost_equal(func(x_axis, *args, **kwargs), x_axis_ang)
    np.testing.assert_array_almost_equal(func(-x_axis, *args, **kwargs), neg_x_axis_ang)
    
    # Test y-axis
    np.testing.assert_array_almost_equal(func(y_axis, *args, **kwargs), y_axis_ang)
    np.testing.assert_array_almost_equal(func(-y_axis, *args, **kwargs), neg_y_axis_ang)
    
    # Test z-axis
    np.testing.assert_array_almost_equal(func(z_axis, *args, **kwargs), z_axis_ang)
    np.testing.assert_array_almost_equal(func(-z_axis, *args, **kwargs), neg_z_axis_ang)

def test_eclipticCartesianToAngular():
    __cardinalityAng(eclipticCartesianToAngular, [])
    
def test_equatorialCartesianToAngular():
    __cardinalityAng(equatorialCartesianToAngular, [])
    
# Angular to Angular
    
def test_eclipticEquatorialAngular(): 
    # Test round-about converstion
    # RA, Dec -> Lon, Lat -> RA, Dec
    num = 1000
    ra = np.random.uniform(0, 2*np.pi, num)
    dec  = np.random.uniform(-np.pi/2, np.pi/2, num)
    rr, dd = np.meshgrid(ra, dec)
    rr = rr.flatten()
    dd = dd.flatten()

    coords_eq_ang = np.zeros([len(rr), 2])
    coords_eq_ang[:,0] = rr
    coords_eq_ang[:,1] = dd
    coords_ec_ang = equatorialToEclipticAngular(coords_eq_ang)
    np.testing.assert_allclose(coords_eq_ang,   eclipticToEquatorialAngular(coords_ec_ang[:,0:2])[:,0:2])
    

    