import numpy as np
from itertools import combinations

from ...config import Config

__all__ = ["selectObservations"]


def selectObservations(observations, method="combinations", columnMapping=Config.columnMapping):
    """
    Selects which three observations to use for IOD depending on the method. 
    
    Methods:
        'first+middle+last' : Grab the first, middle and last observations in time. 
        'thirds' : Grab the middle observation in the first third, second third, and final third. 
        'combinations' : Return the observation IDs corresponding to every possible combination of three observations with
            non-coinciding observation times.
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame containing observations with at least a column of observation IDs and a column
        of exposure times. 
    method : {'first+middle+last', 'thirds', 'combinations'}, optional
        Which method to use to select observations. 
        [Default = 'combinations']
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
        
    Returns
    -------
    obs_id : `~numpy.ndarray' (N, 3 or 0)
        An array of selected observation IDs. If three unique observations could 
        not be selected then returns an empty array. 
    """
    obs_ids = observations[columnMapping["obs_id"]].values
    indexes = np.arange(0, len(obs_ids))
    times = observations[columnMapping["exp_mjd"]].values
    selected = np.array([])

    if method == "first+middle+last":
        selected_times = np.percentile(times, 
                                 [0, 50, 100], 
                                 interpolation="nearest")
        selected_index = np.intersect1d(times, selected_times, return_indices=True)[1]
        selected_index = np.array([selected_index])
        
    elif method == "thirds":
        selected_times = np.percentile(times, 
                                 [1/6*100, 50, 5/6*100], 
                                 interpolation="nearest")
        selected_index = np.intersect1d(times, selected_times, return_indices=True)[1]
        selected_index = np.array([selected_index])
        
    elif method == "combinations":
        # Make all possible combinations of 3 observations
        selected_index = np.array([np.array(index) for index in combinations(indexes, 3)])
        
    else:
        raise ValueError("method should be one of {'first+middle+last', 'thirds'}")
    
    # Make sure each returned combination of observation ids have at least 3 unique
    # times 
    keep = []
    for i, comb in enumerate(times[selected_index]):
        if len(np.unique(comb)) == 3:
            keep.append(i)
    keep = np.array(keep)
    
    # Return an empty array if no observations satisfy the criteria
    if len(keep) == 0:
        return np.array([])
    
    return obs_ids[selected_index[keep, :]]
        
   