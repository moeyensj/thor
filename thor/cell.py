import numpy as np
import pandas as pd

from .config import Config

__all__ = ["_findObsInCell",
           "Cell"]

def _findObsInCell(obsIds, 
                  coords,
                  coordsCenter, 
                  fieldArea=10, 
                  fieldShape="square"):
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

class Cell:
    """
    Cell: Construct used to find detections around a test particle. As the 
    the test particle is propagated forward or backward in time, the cell's shape is used to find 
    observations that lie around that particle. The particle's transformation matrix is designed to be 
    applied to this class.
    
    Parameters
    ----------
    center : `~numpy.ndarray` (2)
        Location of cell center. Typically, the center is defined in RA and Dec. 
    mjd : float
        Time in units of MJD.
    shape : {'square', 'circle'}, optional
        Cell's shape can be square or circle. Combined with the area parameter, will set the search 
        area when looking for observations contained within the defined cell. 
        [Default = 'square']
    area : float, optional
        Cell's area in units of square degrees. 
        [Default = 10]
    dataframe : {None, `~pandas.DataFrame`}, required**
        Pandas DataFrame from which to look for observations contained within the cell.
        [Default = `~pandas.DataFrame]
        **This parameter should either be replaced with a required argument or some other form
        of data structure.
        
    Returns
    -------
    None
    """
    def __init__(self, center, mjd, dataframe, shape="square", area=10):
        self.center = center
        self.area = area
        self.shape = shape
        self.mjd = mjd
        # Effectively a placeholder for a better data structure like maybe a database connection 
        # or something similar
        self.dataframe = dataframe
        self.observations = None
        return
        
    def getObservations(self, columnMapping=Config.columnMapping):
        """
        Get the observations that lie within the cell. Populates 
        self.observations with the observations from self.dataframe 
        that lie within the cell.
        
        Note: Looks for observations with +- 0.00001 MJD of self.mjd
        
        Parameters
        ----------
        columnMapping : dict, optional
            Column name mapping of observations to internally used column names. 
            [Default = `~thor.Config.columnMapping`]
            
        Returns
        -------
        None
        """
        if self.dataframe is None:
            # Effectively a placeholder for a better data structure like maybe a database connection 
            # or something similar
            raise ValueError("Cell has no associated dataframe. Please redefine cell and add observations.")
        observations = self.dataframe
        exp_observations = observations[(observations[columnMapping["exp_mjd"]] <= self.mjd + 0.00001) 
                                        & (observations[columnMapping["exp_mjd"]] >= self.mjd - 0.00001)]
        keep = _findObsInCell(exp_observations[columnMapping["obs_id"]].values,
                             exp_observations[[columnMapping["RA_deg"], columnMapping["Dec_deg"]]].values,
                             self.center,
                             fieldArea=self.area,
                             fieldShape=self.shape)
        self.observations = exp_observations[exp_observations[columnMapping["obs_id"]].isin(keep)].copy()
        return
 