import time
import numpy as np
import pandas as pd

from .config import Config
from .cell import Cell
from .pyoorb import propagateTestParticle

__all__ = ["findExposureTimes",
           "findAverageObject",
           "buildCellForVisit"]
   
def findExposureTimes(observations, 
                      r, 
                      v, 
                      mjd, 
                      numNights=14, 
                      dMax=20.0, 
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
    r : `~numpy.ndarray` (1, 3)
        Heliocentric distance in AU.
    v : `~numpy.ndarray` (1, 3)
        Velocity vector in AU per day (ecliptic).  
    mjd : float
        Epoch at which ecliptic coordinates and velocity are measured in MJD. 
    numNights : int, optional
        List of nights at which to calculate exposure times. 
        [Default = 14]
    dMax : float, optional
        Maximum angular distance (in RA and Dec) permitted when searching for exposure times
        in degrees. 
        [Default = 20.0]
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
        print("Finding optimal exposure times (with maximum angular distance of {} degrees)...".format(dMax))
        print("")
    
    time_start = time.time()
    # Grab exposure times and night values, insure they are sorted
    # then grab the night corresponding to exposure time closest to
    # mjd start, use that value to calculate the appropriate array
    # of nights to use and their exposure times
    times_nights = observations[observations[columnMapping["exp_mjd"]] > mjd][[columnMapping["exp_mjd"], columnMapping["night"]]]
    times_nights.sort_values(by=columnMapping["exp_mjd"], inplace=True)
    nightStart = times_nights[times_nights[columnMapping["exp_mjd"]] >= mjd][columnMapping["night"]].values[0]
    times = np.unique(times_nights[(times_nights[columnMapping["night"]] >= nightStart) 
                         & (times_nights[columnMapping["night"]] <= nightStart + numNights)][columnMapping["exp_mjd"]].values)
    
    eph = propagateTestParticle([*r, *v], mjd, times)
    eph.rename(columns={"RA_deg": "RA_deg_orbit", "Dec_deg": "Dec_deg_orbit"}, inplace=True)
    
    df = pd.merge(observations[[columnMapping["obs_id"],
                                columnMapping["exp_mjd"], 
                                columnMapping["RA_deg"],
                                columnMapping["Dec_deg"]]],
                  eph[["mjd", "RA_deg_orbit", "Dec_deg_orbit"]], 
                  how='inner', 
                  left_on=columnMapping["exp_mjd"], 
                  right_on="mjd")

    dRA = df[columnMapping["RA_deg"]] - df["RA_deg_orbit"]
    dDec = df[columnMapping["Dec_deg"]] - df["Dec_deg_orbit"]
    dec = np.mean(df[[columnMapping["Dec_deg"], "Dec_deg_orbit"]].values, axis=1)
    d = np.sqrt((dRA * np.cos(np.radians(dec)))**2 + dDec**2)
    indexes = np.where(d <= dMax)[0]
    mjds = np.unique(df["exp_mjd"].values[indexes])
    
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
    cell.getObservations(columnMapping=columnMapping)
    return cell
    