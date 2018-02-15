import numpy as np

__all__ = ["_angularToCartesian",
           "eclipticAngularToCartesian",
           "equatorialAngularToCartesian",
           "calcNae"]

def _angularToCartesian(coords, dist):
    """
    Converts angular coordinates to cartesian given
    an angle or angles of longitude (alpha), an angle or angles of latitude (beta),
    and distance. 
    
    Input array should have shape (N, 2):
    np.array([[lon1, lat1],
              [lon2, lat2],
              [lon3, lat3],
                  .....
              [lonN, latN]])
    
    Parameters
    ----------
    coords : `~np.ndarray` (N, 2)
        Longitude in radians, latitude in radians.
    dist : float or `~np.ndarray` (N)
        Distance in arbitrary units.
        
    Returns
    -------
    coords_cart : `~np.ndarray` (N, 3)
        Cartesian x, y, z coordinates.
    """
    x = dist * np.cos(coords[:,1]) * np.cos(coords[:,0])
    y = dist * np.cos(coords[:,1]) * np.sin(coords[:,0]) 
    z = dist * np.sin(coords[:,1]) 
    return np.array([x, y, z]).T

def eclipticAngularToCartesian(coords_ec_ang, dist):
    """
    Convert angular ecliptic coordinates to 
    cartesian coordinates.
    
    Input coordinate array should have shape (N, 2):
    np.array([[lon1, lat1],
              [lon2, lat2],
              [lon3, lat3],
                  .....
              [lonN, latN]])
    
    Parameters
    ----------
    coords_ec_ang : `~np.ndarray` (N, 2)
        Ecliptic longitude and latitude in radians.
    dist : float or `~np.ndarray` (N)
        Distance in arbitrary units.
        
    Returns
    -------
    coords_ec_cart : `~np.ndarray` (N, 3)
        Cartesian ecliptic x, y, z coordinates
        in same units as dist.
        
    See Also
    --------
    equatorialAngularToCartesian : Convert angular equatorial
    coordinates to equatorial cartesian.
    """
    return _angularToCartesian(coords_ec_ang, dist)

def equatorialAngularToCartesian(coords_eq_ang, dist):
    """
    Convert angular equatorial coordinates to 
    cartesian coordinates.
    
    Input coordinate array should have shape (N, 2):
    np.array([[ra1, dec1],
              [ra2, dec2],
              [ra3, dec3],
                  .....
              [raN, decN]])
    
    Parameters
    ----------
    coords_eq_ang : `~np.ndarray` (N, 2)
        Right Ascension and Declination in radians.
    dist : float or `~np.ndarray` (N)
        Distance in arbitrary units.
        
    Returns
    -------
    coords_eq_cart : `~np.ndarray` (N, 3)
        Cartesian equatorial x, y, z coordinates
        in same units as dist.
        
    See Also
    --------
    eclipticAngularToCartesian : Convert angular ecliptic
    coordinates to ecliptic cartesian.
    """
    return _angularToCartesian(coords_eq_ang, dist)

def calcNae(coords_ec_ang):
    """
    Convert angular ecliptic coordinates to 
    to a cartesian unit vector. 
    
    Input coordinate array should have shape (N, 2):
    np.array([[lon1, lat1],
              [lon2, lat2],
              [lon3, lat3],
                  .....
              [lonN, latN]])
    
    Parameters
    ----------
    coords_ec_ang : `~np.ndarray` (N, 2)
        Ecliptic longitude and latitude in radians.
        
    Returns
    -------
    coords_eq_cart : `~np.ndarray` (N, 3)
        Cartesian unit vector in direction of provided
        angular coordinates.
    """
    return _angularToCartesian(coords_ec_ang, 1)
