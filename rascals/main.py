import numpy as np
import pandas as pd

from .config import Config
from .cell import Cell
from .particle import TestParticle
from .oorb import propagateTestParticle
from .data_processing import findExpTimes

__all__ = ["rascalizeCell"]

def rascalizeCell(observations,
                  cell, 
                  r, 
                  v,
                  mjds="auto",
                  vMax=3.0,
                  includeEquatorialProjection=True, 
                  verbose=True, 
                  columnMapping=Config.columnMapping):
    """
    Run RaSCaLS on a cell. 
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing observations.
    cell : `~rascals.Cell`
        RaSCaLS cell. 
    r : float
        Heliocentric distance in AU.
    v : `~numpy.ndarray` (1, 3)
        Velocity vector in AU per day (ecliptic). 
    mjds : {'auto', `~numpy.ndarray` (N)}
        If mjds is 'auto', will propagate the particle to middle of each unique night in 
        obervations and look for all detections within an angular search radius (defined by a 
        maximum allowable angular speed) and extract the exposure time. Alternatively, an array
        of exposure times may be passed. 
    vMax : float, optional
        Maximum angular velocity (in RA and Dec) permitted when searching for exposure times
        in degrees per day. 
        [Default = 3.0]
    includeEquatorialProjection : bool, optional
        Include naive shifting in equatorial coordinates without properly projecting
        to the plane of the orbit. This is useful if performance comparisons want to be made.
        [Default = True]
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~rascals.Config.columnMapping`]
        
    Returns
    -------
    `~pandas.DataFrame`
        Observations dataframe (from cell.observations) with columns containing
        projected coordinates. 
    """
    # If initial doesn't have observations loaded,
    # get them
    if cell.observations is None:
        cell.getObservations()
        
    x_e = cell.observations[[columnMapping["obs_x_au"], columnMapping["obs_y_au"], columnMapping["obs_z_au"]]].values[0]
   
    # Instantiate particle
    particle = TestParticle(cell.center, r, v, x_e, cell.mjd)
    
    # Prepare transformation matrices
    particle.prepare(verbose=verbose)
    
    if mjds == "auto":
        if verbose == True:
            print("Generating particle ephemeris for the middle of every night.")
            print("Finding optimal exposure times (maximum angular velocity: {})...".format(vMax))
            print("")
            
        nights = observations[columnMapping["night"]].unique() 
        cell_night = cell.observations["night"].unique()[0]
        nights.sort()
        nights = nights[nights > cell_night]
        mjds = findExpTimes(observations, particle.x_a, v, cell.mjd, nights, verbose=verbose)
        
        if verbose == True:
            print("Done. Found {} exposure times.".format(len(mjds)))
            print("")
    
    # Apply tranformations to observations
    particle.apply(cell, verbose=verbose)
    
    # Add initial cell and particle to lists
    cells = [cell]
    particles = [particle]
    
    if includeEquatorialProjection is True:
            cell.observations["theta_x_eq_deg"] = cell.observations[columnMapping["RA_deg"]] - particle.coords_eq_ang[0]
            cell.observations["theta_y_eq_deg"] = cell.observations[columnMapping["RA_deg"]] - particle.coords_eq_ang[1]
    
    # Initialize final dataframe and add observations
    final_df = pd.DataFrame()
    final_df = pd.concat([cell.observations, final_df])

    for mjd_f in mjds:
        oldCell = cells[-1]
        oldParticle = particles[-1]
        
        # Propagate particle to new mjd
        propagated = propagateTestParticle(oldParticle.x_a,
                                           oldParticle.v,
                                           oldParticle.mjd,
                                           mjd_f,
                                           verbose=verbose)
        # Get new equatorial coordinates
        new_coords_eq_ang = propagated[["RA_deg", "Dec_deg"]].values[0]
        
        # Get new barycentric distance
        new_r = propagated["r_au"].values[0]
        
        # Get new velocity in ecliptic cartesian coordinates
        new_v = propagated[["HEclObj_dX/dt_au_p_day",
                            "HEclObj_dY/dt_au_p_day",
                            "HEclObj_dZ/dt_au_p_day"]].values[0]
        
        # Get new location of observer
        new_x_e = propagated[["HEclObsy_X_au",
                              "HEclObsy_Y_au",
                              "HEclObsy_Z_au"]].values[0]
        
        # Get new mjd (same as mjd_f)
        new_mjd = propagated["mjd_utc"].values[0]

        # Define new cell at new coordinates
        newCell = Cell(new_coords_eq_ang,
                       new_mjd,
                       area=oldCell.area,
                       shape=oldCell.shape,
                       dataframe=oldCell.dataframe)
        
        # Get the observations in that cell
        newCell.getObservations()
        
        # Define new particle at new coordinates
        newParticle = TestParticle(new_coords_eq_ang,
                                   new_r,
                                   new_v,
                                   new_x_e,
                                   new_mjd)
        
        # Prepare transformation matrices
        newParticle.prepare(verbose=verbose)
       
        # Apply tranformations to new observations
        newParticle.apply(newCell, verbose=verbose)
        
        if includeEquatorialProjection is True:
            newCell.observations["theta_x_eq_deg"] = newCell.observations[columnMapping["RA_deg"]] - newParticle.coords_eq_ang[0]
            newCell.observations["theta_y_eq_deg"] = newCell.observations[columnMapping["Dec_deg"]] - newParticle.coords_eq_ang[1]
        
        # Add observations to final dataframe
        final_df = pd.concat([newCell.observations, final_df])
    
        # Append new cell and particle to lists
        cells.append(newCell)
        particles.append(newParticle)
        
    final_df.sort_values(by=columnMapping["exp_mjd"], inplace=True)
        
    return final_df
