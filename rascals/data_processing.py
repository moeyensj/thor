import numpy as np

__all__ = ["findObsInCell"]

def findObsInCell(obsIds, coords, coordsCenter, fieldArea=10, fieldShape="square"):
    """
    Find the observation IDs in a circular / spherical region 
    about a central point.
    
    Parameters
    ----------
    obsIds : `~numpy.ndarray` (N, 1)
        Array of observation IDs corresponding to the coords.
    coords : `~numpy.ndarray` (N, D)
        Array of coordinates of N rows for each observation
        and D dimensions. 
    coordsCenter : `~numpy.ndarray` (1, D)
        Array containing coordinates in D dimensions about which
        to search. 
    fieldArea : float, optional
        Field area in square degrees. 
        [Default = 10]
    fieldShape : str, optional
        Field's geometric shape: one of 'square' or 'circle'.
        [Default = 'square']
    
    Returns
    -------
    `~np.ndarray`
        Array of observation IDs that fall within cell.
    """
    if fieldShape == "square":
        half_side = np.sqrt(fieldArea) / 2
        return obsIds[np.all(np.abs(coords - coordsCenter) <= half_side, axis=1)]
    elif fieldShape == "circle":
        radius = np.sqrt(fieldArea / np.pi)
        distances = np.sqrt(np.sum((coords - coordsCenter)**2, axis=1))
        return obsIds[np.where(distances <= radius)[0]]
    else:
        raise ValueError("fieldType should be one of 'square' or 'circle'")
    return
   