import numpy as np

__all__ = ["findObsInCell"]

def findObsInCell(obsIds, coords, coord_center, radius):
    """
    Find the observation IDs in a circular / spherical region 
    about a central point.
    
    Parameters
    ----------
    obsIds : `~np.ndarray` (N, 1)
        Array of observation IDs corresponding to the coords.
    coords : `~np.ndarray` (N, D)
        Array of coordinates of N rows for each observation
        and D dimensions. 
    coord_center : `~np.ndarray` (1, D)
        Array containing coordinates in d dimensions about which
        to search. 
    radius : float
        Search radius.
        
    Returns
    -------
    `~np.ndarray`
        Array of observation IDs that fall within the search radius.
    """
    distances = np.sqrt(np.sum((coords - coord_center)**2, axis=1))
    return obsIds[np.where(distances <= radius)[0]]