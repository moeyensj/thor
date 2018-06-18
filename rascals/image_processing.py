import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from sklearn import metrics
from sklearn.cluster import DBSCAN

from .cell import Cell
from .particle import TestParticle
from .oorb import propagateTestParticle

__all__ = ["rascalize",
           "clusterVelocity",
           "findClusters"]

def rascalize(cell, particle, mjds, mjd_range=0.5, includeEquatorialProjection=True, verbose=True):
    # If initial doesn't have observations loaded,
    # get them
    if cell.observations is None:
        cell.getObservations()
    
    # Prepare transformation matrices
    particle.prepare(verbose=verbose)
    
    # Apply tranformations to observations
    particle.apply(cell, verbose=verbose)
    
    # Add initial cell and particle to lists
    cells = [cell]
    particles = [particle]
    
    if includeEquatorialProjection is True:
            cell.observations["theta_x_eq_deg"] = cell.observations["RA_deg"] - particle.coords_eq_ang[0]
            cell.observations["theta_y_eq_deg"] = cell.observations["Dec_deg"] - particle.coords_eq_ang[1]
    
    # Initialize final dataframe and add observations
    final_df = pd.DataFrame()
    final_df = pd.concat([cell.observations, final_df])
    
    for mjd_f in mjds[1:]:
        oldCell = cells[-1]
        oldParticle = particles[-1]
        
        # Propagate particle to new mjd
        propagated = propagateTestParticle(oldParticle.x_a,
                                           oldParticle.velocity_ec_cart,
                                           oldParticle.mjd,
                                           mjd_f,
                                           verbose=verbose)
        # Get new equatorial coordinates
        new_coords_eq_ang = propagated[["RA_deg", "Dec_deg"]].values[0]
        
        # Get new barycentric distance
        new_r = propagated["r_au"].values[0]
        
        # Get new velocity in ecliptic cartesian coordinates
        new_velocity_ec_cart = propagated[["HEclObj_dX/dt_au_p_day",
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
                                   new_velocity_ec_cart,
                                   new_x_e,
                                   new_mjd)
        
        # Prepare transformation matrices
        newParticle.prepare(verbose=verbose)
       
        # Apply tranformations to new observations
        newParticle.apply(newCell, verbose=verbose)
        
        if includeEquatorialProjection is True:
            newCell.observations["theta_x_eq_deg"] = newCell.observations["RA_deg"] - newParticle.coords_eq_ang[0]
            newCell.observations["theta_y_eq_deg"] = newCell.observations["Dec_deg"] - newParticle.coords_eq_ang[1]
        
        # Add observations to final dataframe
        final_df = pd.concat([newCell.observations, final_df])
    
        # Append new cell and particle to lists
        cells.append(newCell)
        particles.append(newParticle)
        
    final_df.sort_values(by="exp_mjd", inplace=True)
        
    return final_df, cells, particles

def clusterVelocity(obs_ids, theta_x, theta_y, dt, vx, vy, eps=0.005, min_samples=3):
    xx = theta_x - vx * dt
    yy = theta_y - vy * dt
    X = np.vstack([xx, yy]).T  
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    
    clusters = db.labels_[np.where(db.labels_ != -1)[0]]
    cluster_ids = []
    
    if len(clusters) != 0:
        for cluster in np.unique(clusters):
            cluster_ids.append(obs_ids[np.where(db.labels_ == cluster)[0]])
    else:
        cluster_ids = -1
    
    del db
    return cluster_ids
           
def _clusterVelocity(vx, vy,
                     obs_ids=None,
                     theta_x=None,
                     theta_y=None,
                     dt=None,
                     eps=None,
                     min_samples=None):
    return clusterVelocity(obs_ids,
                           theta_x,
                           theta_y,
                           dt,
                           vx,
                           vy,
                           eps=eps,
                           min_samples=min_samples) 

def findClusters(detections, gridpoints=10000, threads=10, eps=0.005, min_samples=3, verbose=True):
    
    # Extract useful quantities
    obs_ids = detections["obs_id"].values
    theta_x = detections["theta_x_deg"].values
    theta_y = detections["theta_y_deg"].values
    mjd = detections["exp_mjd"].values
    truth = detections["name"].values

    # Select detections in first exposure
    first = np.where(mjd == mjd.min())[0]
    theta_x0 = theta_x[first]
    theta_y0 = theta_y[first]
    mjd0 = mjd[first][0]
    dt = mjd - mjd0

    # Grab remaining detections
    remaining = np.where(mjd != mjd.min())[0]
    
    # Create min, max velocity variables
    vx_max = 0
    vx_min = 0
    vy_max = 0
    vy_min = 0
    
    # Calculate the velocity from every detection in the first exposure
    # to every detection in the following exposures. Keep the max and min
    # for both x and y.
    if verbose:
        print("Calculating velocity ranges...")
        
    for i in first:
        vx = (theta_x[remaining] - theta_x0[i]) / dt[remaining]
        vy = (theta_y[remaining] - theta_y0[i]) / dt[remaining]
        if np.min(vx) < vx_min:
            vx_min = np.min(vx)
        if np.max(vx) > vx_max:
            vx_max = np.max(vx)
        if np.min(vy) < vy_min:
            vy_min = np.min(vy)
        if np.max(vy) > vy_max:
            vy_max = np.max(vy)
            
    if verbose:
        print("Maximum possible x velocity range: {:.4e} to {:.4e}".format(vx_min, vx_max))
        print("Maximum possible y velocity range: {:.4e} to {:.4e}".format(vy_min, vy_max))
        
    # Define velocity grid
    possible_vx = np.linspace(vx_min, vx_max, num=int(np.sqrt(gridpoints)))
    possible_vy = np.linspace(vy_min, vy_max, num=int(np.sqrt(gridpoints)))
    vxx, vyy = np.meshgrid(possible_vx, possible_vy)
    vxx = vxx.flatten()
    vyy = vyy.flatten()
    
    if verbose:
        print("Running velocity space clustering.")
    
    possible_clusters = []
    if threads > 1:
        if verbose:
            print("Using {} threads".format(threads))
        p = mp.Pool(threads)
        possible_clusters = p.starmap(partial(_clusterVelocity, 
                                              obs_ids=obs_ids,
                                              theta_x=theta_x,
                                              theta_y=theta_y,
                                              dt=dt,
                                              eps=eps,
                                              min_samples=min_samples),
                                              zip(vxx.T, vyy.T))
        p.close()
    else:
        possible_clusters = []
        for vxi, vyi in zip(vxx, vyy):
            possible_clusters.append(_clusterVelocity(vxi, vyi))
    
    # Clean up returned arrays and remove empty cases
    populated_clusters = []
    populated_cluster_velocities = []
    for cluster, vxi, vyi in zip(possible_clusters, vxx, vyy):
        if type(cluster) == int:
            continue
        else:
            populated_clusters.append(cluster)
            populated_cluster_velocities.append([vxi, vyi])
            
    if verbose:
        print("Found {} cluster groups.".format(len(populated_clusters)))

    return populated_clusters, populated_cluster_velocities, dt