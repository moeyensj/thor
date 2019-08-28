import numpy as np

from ...config import Config

__all__ = ["selectObservations"]

def selectObservations(observations, method="first+middle+last", columnMapping=Config.columnMapping):
    """
    Selects which three observations to use for IOD depending on the method. 
    
    Methods:
        'first+middle+last' : Grab the first, middle and last observations in time. 
        'thirds' : Grab the middle observation in the first third, second third, and final third. 
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame containing observations with at least a column of observation IDs and a column
        of exposure times. 
    method : {'first+middle+last', 'thirds}, optional
        Which method to use to select observations. 
        [Default = `first+middle+last`]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
        
    Returns
    -------
    obs_id : `~numpy.ndarray' (3 or 0)
        An array of selected observation IDs. If three unique observations could 
        not be selected than returns an empty array. 
    """
    if method == "first+middle+last":
        selected = np.percentile(observations[columnMapping["exp_mjd"]].values, [0, 50, 100], interpolation="nearest")
    elif method == "thirds":
        selected = np.percentile(observations[columnMapping["exp_mjd"]].values, [1/6 * 100, 50, 5/6*100], interpolation="nearest")
    else:
        raise ValueError("method should be one of {'first+middle+last', 'thirds'}")
        
    if len(np.unique(selected)) != 3:
        print("Could not find three observations that satisfy the criteria.")
        return np.array([])
    
    index = np.intersect1d(observations[columnMapping["exp_mjd"]].values, selected, return_indices=True)[1]
    return observations[columnMapping["obs_id"]].values[index]
        