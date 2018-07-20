import numpy as np

from .config import Config

__all__ = ["findObsInCell",
           "findExpTimes",
           "findAverageObject"]

def findObsInCell(obsIds, 
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
   
def findExpTimes(observations, 
                 r, 
                 v, 
                 mjdStart, 
                 nights, 
                 vMax=3.0, 
                 verbose=True,
                 columnMapping=Config.columnMapping):
    
    """
    Finds the unique exposure times of all detections that fall within
    a maximum search radius (set by a maximum angular velocity) for a 
    test particle defined by r and v at mjdStart. 
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing observations.
    r : float
        Heliocentric distance in AU.
    v : `~numpy.ndarray` (1, 3)
        Velocity vector in AU per day (ecliptic).  
    mjdStart : float
        Epoch at which ecliptic coordinates and velocity are measured in MJD.
    nights : `~numpy.ndarray` (N)
        List of nights at which to calculate exposure times. Should ideally not include
        the night containing mjdStart. 
    vMax : float, optional
        Maximum angular velocity (in RA and Dec) permitted when searching for exposure times
        in degrees per day. 
        [Default = 3.0]
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~rascals.Config.columnMapping`] 

    Returns
    -------
    mjds : `~numpy.ndarray`
        Sorted unique exposure times. 
    """
    possible_obs_ids = np.empty(0)
    ri = r
    vi = v
    mjdStarti = mjdStart
    
    for night in nights: 
        
        # Find exposure time in the middle of the night  
        mjdEnd = np.median(observations[observations["night"] == night]["exp_mjd"].unique())
        
        # Propagate particle
        ephemeris = propagateTestParticle(ri, 
                                          vi,
                                          mjdStarti,
                                          mjdEnd,
                                          verbose=verbose)
        
        # Find all observation IDs within some maximum angular velocity from the location of the particle
        obs_ids = observations[observations[columnMapping["night"]] == night][columnMapping["obs_id"]].values
        dRA = observations[observations[columnMapping["night"]] == night][columnMapping["RA_deg"]].values - ephemeris["RA_deg"].values
        dDec = observations[observations[columnMapping["night"]] == night][columnMapping["Dec_deg"]].values - ephemeris["Dec_deg"].values
        dec = ephemeris["Dec_deg"].values + dDec/2
        dt = observations[observations[columnMapping["night"]] == night][columnMapping["exp_mjd"]].values - ephemeris["mjd_utc"].values
        v = np.abs(np.sqrt((dRA * np.cos(np.radians(dec)))**2  + dDec**2) / dt)
        index = np.where(v <= vMax)[0]
        possible_obs_ids = np.concatenate([possible_obs_ids, obs_ids[index]])
        
        # Set new r, v and time to particle's current location
        ri = ephemeris[['HEclObj_X_au', 'HEclObj_Y_au', 'HEclObj_Z_au']].values[0]
        vi = ephemeris[['HEclObj_dX/dt_au_p_day', 'HEclObj_dY/dt_au_p_day', 'HEclObj_dZ/dt_au_p_day']].values[0]
        mjdStarti = mjdEnd
        
    mjds = observations[observations[columnMapping["obs_id"]].isin(possible_obs_ids)][columnMapping["exp_mjd"]].unique()
    mjds.sort()
    return mjds

def findAverageObject(observations, 
                      columnMapping=Config.columnMapping):
    """
    Find the object with observations that represents 
    the most average in terms of cartesian velocity and the
    heliocentric distance.
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing observations.
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~rascals.Config.columnMapping`] 
    
    Returns
    -------
    name : str
        The name of the object
    """
    objects = observations[observations[columnMapping["name"]] != "NS"]
    
    if len(objects) == 0:
        # No real objects
        return -1
    
    rv = objects[[
        columnMapping["obj_dx/dt_au_p_day"],
        columnMapping["obj_dy/dt_au_p_day"],
        columnMapping["obj_dz/dt_au_p_day"],
        columnMapping["r_au"]
    ]].values
    
    # Calculate the percent difference between the median of each velocity element
    # and the heliocentric distance
    percent_diff = np.abs((rv - np.median(rv, axis=0)) / np.median(rv, axis=0))
    
    # Sum the percent differences
    summed_diff = np.sum(percent_diff, axis=1)
    
    # Find the minimum summed percent difference and call that 
    # the average object
    index = np.where(summed_diff == np.min(summed_diff))[0][0]
    return objects[columnMapping["name"]].values[index]
