import time
import numpy as np
import pandas as pd

from .config import Config
from .cell import Cell
from .oorb import propagateTestParticle

__all__ = ["findExposureTimes",
           "findAverageObject",
           "buildCellForVisit"]
   
def findExposureTimes(observations, 
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
        [Default = `~thor.Config.columnMapping`] 

    Returns
    -------
    mjds : `~numpy.ndarray`
        Sorted unique exposure times. 
    """
    
    if verbose == True:
        print("THOR: findExposureTimes")
        print("-------------------------")
        print("Generating particle ephemeris for the middle of every night.")
        print("Finding optimal exposure times (maximum angular velocity: {})...".format(vMax))
        print("")
    
    time_start = time.time()
    possible_obs_ids = np.empty(0)
    ri = r
    vi = v
    mjdStarti = mjdStart
    
    for i, night in enumerate(nights): 
        
        # Find exposure time in the middle of the night  
        mjdEnd = np.median(observations[observations["night"] == night]["exp_mjd"].unique())
        
        # Propagate particle (set verbose to false here)
        if verbose == True:
            print("Night: {} ({}/{})".format(night, i + 1, len(nights)))
            print("Propagating particle to {}...".format(mjdEnd))
        ephemeris = propagateTestParticle(ri, 
                                          vi,
                                          mjdStarti,
                                          mjdEnd,
                                          verbose=False)
        if verbose == True:
            print("Looking for observations...")
        
        # Find all observation IDs within some maximum angular velocity from the location of the particle
        obs_ids = observations[observations[columnMapping["night"]] == night][columnMapping["obs_id"]].values
        dRA = observations[observations[columnMapping["night"]] == night][columnMapping["RA_deg"]].values - ephemeris["RA_deg"].values
        dDec = observations[observations[columnMapping["night"]] == night][columnMapping["Dec_deg"]].values - ephemeris["Dec_deg"].values
        dec = ephemeris["Dec_deg"].values + dDec/2
        dt = observations[observations[columnMapping["night"]] == night][columnMapping["exp_mjd"]].values - ephemeris["mjd_utc"].values
        v = np.abs(np.sqrt((dRA * np.cos(np.radians(dec)))**2  + dDec**2) / dt)
        index = np.where(v <= vMax)[0]
        possible_obs_ids = np.concatenate([possible_obs_ids, obs_ids[index]])
        
        if verbose == True:
            print("Found {} observations within search area.".format(len(index)))
            print("")
        
        # Set new r, v and time to particle's current location
        ri = ephemeris[['HEclObj_X_au', 'HEclObj_Y_au', 'HEclObj_Z_au']].values[0]
        vi = ephemeris[['HEclObj_dX/dt_au_p_day', 'HEclObj_dY/dt_au_p_day', 'HEclObj_dZ/dt_au_p_day']].values[0]
        mjdStarti = mjdEnd
        
    mjds = observations[observations[columnMapping["obs_id"]].isin(possible_obs_ids)][columnMapping["exp_mjd"]].unique()
    mjds.sort()
    time_end = time.time()
    
    if verbose == True:
        print("Done. Found {} unique exposure times.".format(len(mjds)))
        print("Total time in seconds: {}".format(time_end - time_start))
        print("-------------------------")
        print("")
  
    return mjds

def findAverageObject(observations, 
                      verbose=True,
                      columnMapping=Config.columnMapping):
    """
    Find the object with observations that represents 
    the most average in terms of cartesian velocity and the
    heliocentric distance.
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing observations.
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`] 
    
    Returns
    -------
    name : {str, -1}
        The name of the object, if there are no real objects returns -1
    """
    if verbose == True:
        print("THOR: findAverageObject")
        print("-------------------------")
    objects = observations[observations[columnMapping["name"]] != "NS"]
    
    if len(objects) == 0:
        # No real objects
        if verbose == True:
            print("No real objects found.")
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
    name = objects[columnMapping["name"]].values[index]
    if verbose == True:
        print("{} is the most average object.".format(name))
        print("-------------------------")
        print("")
        
    return name
    

def buildCellForVisit(observations, 
                      visitId, 
                      shape="square", 
                      area=10, 
                      columnMapping=Config.columnMapping):
    """
    Builds a cell for a unique visit. Populates cell with observations. 
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing observations.
    visitId : int
        Visit ID.
    shape : {'square', 'circle'}, optional
        Cell's shape can be square or circle. Combined with the area parameter, will set the search 
        area when looking for observations contained within the defined cell. 
        [Default = 'square']
    area : float, optional
        Cell's area in units of square degrees. 
        [Default = 10]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
        
    Returns
    -------
    cell : `~thor.Cell`
        Cell with observations populated.
    """
    visit = observations[observations[columnMapping["visit_id"]] == visitId]
    center = visit[[columnMapping["field_RA_deg"], columnMapping["field_Dec_deg"]]].values[0]
    mjd = visit[columnMapping["exp_mjd"]].values[0]
    cell = Cell(center, mjd, observations, shape=shape, area=area)
    cell.getObservations()
    return cell
    