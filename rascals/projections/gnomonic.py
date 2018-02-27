import numpy as np

__all__ = ["angularToGnomonic",
           "cartesianToGnomonic"]

def angularToGnomonic(coords_ang, coords_ang_center=np.array([0, 0])):
    """
    Project angular spherical coordinates onto a gnomonic tangent plane. 
    
    Parameters
    ----------
    coords_ang : `~np.ndarray` (N, 2)
        Longitude (between 0 and 2 pi) and latitude (between -pi/2 and pi/2)
        in radians.
    coords_ang_center : `~np.ndarray` (2), optional
        Longitude (between 0 and 2 pi) and latitude (between -pi/2 and pi/2)
        in radians about which to center projection.
        [Default = np.array([0, 0])]
        
    Returns
    -------
    coords_gnomonic : `~np.ndarray` (N, 2)
        Gnomonic longitude and latitude in radians.
    """
    lon = coords_ang[:, 0] 
    lon = np.where(lon > np.pi, lon - 2*np.pi, lon)
    lat = coords_ang[:, 1]
    lon_0, lat_0 = coords_ang_center
    
    c = np.sin(lat_0) * np.sin(lat) + np.cos(lat_0) * np.cos(lat) * np.cos(lon - lon_0)
    u = np.cos(lat) * np.sin(lon - lon_0) / c
    v = (np.cos(lat_0) * np.sin(lat) - np.sin(lat_0) * np.cos(lat) * np.cos(lon - lon_0)) / c
    return np.array([u, v]).T

def cartesianToGnomonic(coords_cart):
    """
    Project cartesian coordinates onto a gnomonic tangent plane centered about
    the x-axis. 
    
    Parameters
    ----------
    coords_cart : `~np.ndarray` (N, 3)
        Cartesian equatorial x, y, z coordinates.
            
    Returns
    -------
    coords_gnomonic : `~np.ndarray` (N, 2)
        Gnomonic longitude and latitude in radians.
    """
    u = coords_cart[:,1] / coords_cart[:,0]
    v = coords_cart[:,2] / coords_cart[:,0]
    return np.array([u, v]).T