import os
import time
import uuid
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from astropy.time import Time

from .config import Config
from .cell import Cell
from .orbit import TestOrbit
from .orbits import Orbits
from .orbits import generateEphemeris
from .orbits import initialOrbitDetermination
from .orbits import differentialCorrection
from .orbits import mergeAndExtendOrbits
from .backend import _init_worker
from .utils import identifySubsetLinkages


USE_RAY = Config.USE_RAY
USE_GPU = Config.USE_GPU
USE_GPU = False
NUM_THREADS = Config.NUM_THREADS

if USE_GPU:
    import cudf
    from cuml.cluster import DBSCAN
else:
    from sklearn.cluster import DBSCAN

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

__all__ = [
    "rangeAndShift_worker",
    "rangeAndShift",
    "clusterVelocity",
    "clusterVelocity_worker",
    "clusterAndLink",
    "runTHOROrbit"
]

def rangeAndShift_worker(observations, ephemeris, cell_area=10):
    
    assert len(observations["mjd_utc"].unique()) == 1
    assert len(ephemeris["mjd_utc"].unique()) == 1
    assert observations["mjd_utc"].unique()[0] == ephemeris["mjd_utc"].unique()[0]
    observation_time = observations["mjd_utc"].unique()[0]
    
    # Create Cell centered on the sky-plane location of the
    # test orbit
    cell = Cell(
        ephemeris[["RA_deg", "Dec_deg"]].values[0],
        observation_time,
        area=cell_area,
    )

    # Grab observations within cell
    cell.getObservations(observations)

    if len(cell.observations) != 0:

        # Create test orbit with state of orbit at visit time
        test_orbit = TestOrbit(
            ephemeris[["obj_x", "obj_y", "obj_z", "obj_vx", "obj_vy", "obj_vz"]].values[0],
            observation_time
        )

        # Prepare rotation matrices 
        test_orbit.prepare(verbose=False)

        # Apply rotation matrices and transform observations into the orbit's
        # frame of motion. 
        test_orbit.applyToObservations(cell.observations, verbose=False)
        
        projected_observations = cell.observations
        
    else:
        
        projected_observations = pd.DataFrame()
        
    return projected_observations

def clusterVelocity(
        obs_ids,
        x, 
        y, 
        dt, 
        vx, 
        vy, 
        eps=0.005, 
        min_samples=5,
    ):
    """
    Clusters THOR projection with different velocities
    in the projection plane using `~scipy.cluster.DBSCAN`.
    
    Parameters
    ----------
    obs_ids : `~numpy.ndarray' (N)
        Observation IDs.
    x : `~numpy.ndarray' (N)
        Projection space x coordinate in degrees or radians.
    y : `~numpy.ndarray' (N)
        Projection space y coordinate in degrees or radians.    
    dt : `~numpy.ndarray' (N)
        Change in time from 0th exposure in units of MJD.
    vx : `~numpy.ndarray' (N)
        Projection space x velocity in units of degrees or radians per day in MJD. 
    vy : `~numpy.ndarray' (N)
        Projection space y velocity in units of degrees or radians per day in MJD. 
    eps : float, optional
        The maximum distance between two samples for them to be considered 
        as in the same neighborhood. 
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
        [Default = 0.005]
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a 
        point to be considered as a core point. This includes the point itself.
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
        [Default = 5]
        
    Returns
    -------
    list
        If clusters are found, will return a list of numpy arrays containing the 
        observation IDs for each cluster. If no clusters are found, will return np.NaN.
    """ 

    xx = x - vx * dt
    yy = y - vy * dt
    if USE_GPU:
        kwargs = {}
    else:
        kwargs = {"n_jobs" : 1}

    X = np.vstack([xx, yy]).T  
    
    db = DBSCAN(
        eps=eps, 
        min_samples=min_samples,
        **kwargs
    )
    db.fit(X)

    clusters = db.labels_[np.where(db.labels_ != -1)[0]]
    cluster_ids = []
    
    if len(clusters) != 0:
        for cluster in np.unique(clusters):
            cluster_mask = np.where(db.labels_ == cluster)[0]
            
            dt_in_cluster = dt[cluster_mask]
            num_obs = len(dt_in_cluster)
            if num_obs == len(np.unique(dt_in_cluster)) and num_obs >= min_samples:
                cluster_ids.append(obs_ids[cluster_mask])

    else:
        cluster_ids = np.NaN
                
    del db
    return cluster_ids

def clusterVelocity_worker(
        vx,
        vy,
        obs_ids=None,
        x=None,
        y=None,
        dt=None,
        eps=None,
        min_samples=None,
    ):
    """
    Helper function to multiprocess clustering.
    
    """
    cluster_ids = clusterVelocity(
        obs_ids,
        x,
        y,
        dt,
        vx,
        vy,
        eps=eps,
        min_samples=min_samples,
    )
    return cluster_ids

if USE_RAY:
    import ray
    rangeAndShift_worker = ray.remote(rangeAndShift_worker)
    clusterVelocity_worker = ray.remote(clusterVelocity_worker)

def rangeAndShift(
        observations, 
        orbit, 
        cell_area=10, 
        threads=NUM_THREADS,
        backend="PYOORB",
        backend_kwargs={},
        verbose=True
    ):
    """
    Propagate the orbit to all observation times in observations. At each epoch gather a circular region of observations of size cell_area
    centered about the location of the orbit on the sky-plane. Transform and project each of the gathered observations into
    the frame of motion of the test orbit. 
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing preprocessed observations.    
            Should contain the following columns:
                obs_id : observation IDs
                RA_deg : Right Ascension in degrees. 
                Dec_deg : Declination in degrees.
                RA_sigma_deg : 1-sigma uncertainty for Right Ascension in degrees.
                Dec_sigma_deg : 1-sigma uncertainty for Declination in degrees.
                observatory_code : MPC observatory code
    orbit : `~numpy.ndarray` (6)
        Orbit to propagate. If backend is 'THOR', then these orbits must be expressed
        as heliocentric ecliptic cartesian elements. If backend is 'PYOORB' orbits may be 
        expressed in keplerian, cometary or cartesian elements.
    cell_area : float, optional
        Cell's area in units of square degrees. 
        [Default = 10]
    backend : {'THOR', 'PYOORB'}, optional
        Which backend to use. 
    backend_kwargs : dict, optional
        Settings and additional parameters to pass to selected 
        backend.
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
        
    Returns
    -------
    projected_observations : {`~pandas.DataFrame`, -1}
        Observations dataframe (from cell.observations) with columns containing
        projected coordinates. 
    """
    time_start = time.time()
    if verbose == True:
        print("THOR: rangeAndShift")
        print("-------------------------------")
        print("Running range and shift...")
        print("Assuming r = {} AU".format(orbit.cartesian[0, :3]))
        print("Assuming v = {} AU per day".format(orbit.cartesian[0, 3:]))

    # Build observers dictionary: keys are observatory codes with exposure times (as astropy.time objects)
    # as values
    observers = {}
    for code in observations["observatory_code"].unique():
        observers[code] = Time(
            observations[observations["observatory_code"].isin([code])]["mjd_utc"].unique(), 
            format="mjd", 
            scale="utc"
        )

    # Propagate test orbit to all times in observations
    ephemeris = generateEphemeris(
        orbit, 
        observers, 
        backend=backend,     
        backend_kwargs=backend_kwargs
    )
    if backend == "FINDORB":
        
        observer_states = []
        for observatory_code, observation_times in observers.items():
            observer_states.append(
                getObserverState(
                    [observatory_code], 
                    observation_times,
                    frame='ecliptic',
                    origin='heliocenter',
                )
            )
        
        observer_states = pd.concat(observer_states)
        observer_states.reset_index(
            inplace=True,
            drop=True
        )
        ephemeris = ephemeris.join(observer_states[["obs_x", "obs_y", "obs_z", "obs_vx", "obs_vy", "obs_vz"]])
        
    velocity_cols = []
    if backend != "PYOORB":
        velocity_cols = ["obs_vx", "obs_vy", "obs_vz"]
                
    observations = observations.merge(
        ephemeris[["mjd_utc", "observatory_code", "obs_x", "obs_y", "obs_z"] + velocity_cols], 
        left_on=["mjd_utc", "observatory_code"], 
        right_on=["mjd_utc", "observatory_code"]
    )
    
    # Split the observations into a single dataframe per unique observatory code and observation time
    # Basically split the observations into groups of unique exposures
    observations_grouped = observations.groupby(by=["observatory_code", "mjd_utc"])
    observations_split = [observations_grouped.get_group(g) for g in observations_grouped.groups]
    
    # Do the same for the test orbit's ephemerides
    ephemeris_grouped = ephemeris.groupby(by=["observatory_code", "mjd_utc"])
    ephemeris_split = [ephemeris_grouped.get_group(g) for g in ephemeris_grouped.groups]
    
    if verbose:
        print("Using {} threads...".format(threads))
    if threads > 1:

        if USE_RAY:
            shutdown = False
            if not ray.is_initialized():
                ray.init(num_cpus=threads)
                shutdown = True

            p = []
            for observations_i, ephemeris_i in zip(observations_split, ephemeris_split):
                p.append(
                    rangeAndShift_worker.remote(
                        observations_i,
                        ephemeris_i, 
                        cell_area=cell_area
                    )
                )
            projected_dfs = ray.get(p)

            if shutdown:
                ray.shutdown()
        else:
            p = mp.Pool(
                processes=threads,
                initializer=_init_worker,
            ) 

            projected_dfs = p.starmap(
                partial(
                    rangeAndShift_worker, 
                    cell_area=cell_area
                ),
                zip(
                    observations_split, 
                    ephemeris_split, 
                ) 
            )
            p.close()  

    else:
        projected_dfs = []
        for observations_i, ephemeris_i in zip(observations_split, ephemeris_split):
            projected_df = rangeAndShift_worker(
                observations_i,
                ephemeris_i,  
                cell_area=cell_area
            )
            projected_dfs.append(projected_df)

    projected_observations = pd.concat(projected_dfs)   
    if len(projected_observations) > 0:
        projected_observations.sort_values(by=["mjd_utc", "observatory_code"], inplace=True)
        projected_observations.reset_index(inplace=True, drop=True)
    else:
        projected_observations = pd.DataFrame(
            columns=[
                'obs_id', 'mjd_utc', 'RA_deg', 'Dec_deg', 'RA_sigma_deg',
                'Dec_sigma_deg', 'observatory_code', 'obs_x', 'obs_y', 'obs_z', 'obj_x',
                'obj_y', 'obj_z', 'theta_x_deg', 'theta_y_deg'
            ]
        )
    
    time_end = time.time()
    if verbose == True:
        print("Done. Final DataFrame has {} observations.".format(len(projected_observations)))
        print("Total time in seconds: {}".format(time_end - time_start))  
        print("-------------------------------")
        print("")
        
    return projected_observations

def clusterAndLink(
        observations, 
        vx_range=[-0.1, 0.1], 
        vy_range=[-0.1, 0.1],
        vx_bins=100, 
        vy_bins=100,
        vx_values=None,
        vy_values=None,
        eps=0.005, 
        min_samples=5,
        identify_subsets=False,
        threads=NUM_THREADS, 
        verbose=True
    ):
    """
    Cluster and link correctly projected (after ranging and shifting)
    detections.
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing post-range and shift observations.
    vx_range : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in x. 
        Will not be used if vx_values are specified. 
        [Default = [-0.1, 0.1]]
    vy_range : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in y. 
        Will not be used if vy_values are specified. 
        [Default = [-0.1, 0.1]]
    vx_bins : int, optional
        Length of x-velocity grid between vx_range[0] 
        and vx_range[-1]. Will not be used if vx_values are 
        specified. 
        [Default = 100]
    vy_bins: int, optional
        Length of y-velocity grid between vy_range[0] 
        and vy_range[-1]. Will not be used if vy_values are 
        specified. 
        [Default = 100]
    vx_values : {None, `~numpy.ndarray`}, optional
        Values of velocities in x at which to cluster
        and link. 
        [Default = None]
    vy_values : {None, `~numpy.ndarray`}, optional
        Values of velocities in y at which to cluster
        and link. 
        [Default = None]
    threads : int, optional
        Number of threads to use. 
        [Default = 12]
    eps : float, optional
        The maximum distance between two samples for them to be considered 
        as in the same neighborhood. 
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
        [Default = 0.005]
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a 
        point to be considered as a core point. This includes the point itself.
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
        [Default = 5]
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
        
    Returns
    -------
    all_clusters : `~pandas.DataFrame`
        DataFrame with the cluster ID, the number of observations, and the x and y velocity. 
    cluster_members : `~pandas.DataFrame`
        DataFrame containing the cluster ID and the observation IDs of its members. 
    """ 
    # Extract useful quantities
    obs_ids = observations["obs_id"].values
    theta_x = observations["theta_x_deg"].values
    theta_y = observations["theta_y_deg"].values
    mjd = observations["mjd_utc"].values

    # Select detections in first exposure
    first = np.where(mjd == mjd.min())[0]
    theta_x0 = theta_x[first]
    theta_y0 = theta_y[first]
    mjd0 = mjd[first][0]
    dt = mjd - mjd0

    if vx_values is None and vx_range is not None:
        vx = np.linspace(*vx_range, num=vx_bins)
    elif vx_values is None and vx_range is None:
        raise ValueError("Both vx_values and vx_range cannot be None.")
    else:
        vx = vx_values
        vx_range = [vx_values[0], vx_values[-1]]
        vx_bins = len(vx)
     
    if vy_values is None and vy_range is not None:
        vy = np.linspace(*vy_range, num=vy_bins)
    elif vy_values is None and vy_range is None:
        raise ValueError("Both vy_values and vy_range cannot be None.")
    else:
        vy = vy_values
        vy_range = [vy_values[0], vy_values[-1]]
        vy_bins = len(vy)
        
    if vx_values is None and vy_values is None:
        vxx, vyy = np.meshgrid(vx, vy)    
        vxx = vxx.flatten()
        vyy = vyy.flatten()
    elif vx_values is not None and vy_values is not None:
        vxx = vx
        vyy = vy
    else:
        raise ValueError("")
        
    time_start_cluster = time.time()
    if verbose == True:
        print("THOR: clusterAndLink")
        print("-------------------------------")
        print("Running velocity space clustering...")
        print("X velocity range: {}".format(vx_range))
        
        if vx_values is not None:
            print("X velocity values: {}".format(vx_bins))
        else:
            print("X velocity bins: {}".format(vx_bins))
            
        print("Y velocity range: {}".format(vy_range))
        if vy_values is not None:
            print("Y velocity values: {}".format(vy_bins))
        else:
            print("Y velocity bins: {}".format(vy_bins))
        if vx_values is not None:
            print("User defined x velocity values: True")
        else: 
            print("User defined x velocity values: False")
        if vy_values is not None:
            print("User defined y velocity values: True")
        else:
            print("User defined y velocity values: False")
            
        if vx_values is None and vy_values is None:
            print("Velocity grid size: {}".format(vx_bins * vy_bins))
        else: 
            print("Velocity grid size: {}".format(vx_bins))
        print("Max sample distance: {}".format(eps))
        print("Minimum samples: {}".format(min_samples))
    
    possible_clusters = []
    if threads > 1 and not USE_GPU:
        if verbose:
            print("Using {} threads...".format(threads))
            
        if USE_RAY:
            shutdown = False
            if not ray.is_initialized():
                ray.init(num_cpus=threads)
                shutdown = True

            p = []
            for vxi, vyi in zip(vxx, vyy):
                p.append(
                    clusterVelocity_worker.remote(
                        obs_ids,
                        theta_x, 
                        theta_y, 
                        dt, 
                        vxi, 
                        vyi, 
                        eps=eps, 
                        min_samples=min_samples
                    )
                )
            possible_clusters = ray.get(p)

            if shutdown:
                ray.shutdown()
                
        else:
        
            p = mp.Pool(threads, _init_worker)
            try:
                possible_clusters = p.starmap(
                    partial(
                        clusterVelocity_worker, 
                        obs_ids=obs_ids,
                        x=theta_x,
                        y=theta_y,
                        dt=dt,
                        eps=eps,
                        min_samples=min_samples,
                    ),
                    zip(vxx, vyy)
                )

            except KeyboardInterrupt:
                p.terminate()

            p.close()
    else:
        possible_clusters = []
        for vxi, vyi in zip(vxx, vyy):
            possible_clusters.append(
                clusterVelocity(
                    obs_ids,
                    theta_x, 
                    theta_y, 
                    dt, 
                    vxi, 
                    vyi, 
                    eps=eps, 
                    min_samples=min_samples
                )
            )
    time_end_cluster = time.time()    
    
    if verbose == True:
        print("Done. Completed in {} seconds.".format(time_end_cluster - time_start_cluster))
        print("")
        
    if verbose == True:
        print("Restructuring clusters...")
    time_start_restr = time.time()
    
    possible_clusters = pd.DataFrame({"clusters": possible_clusters})
    
    # Remove empty clusters
    possible_clusters = possible_clusters[~possible_clusters["clusters"].isna()]

    if len(possible_clusters) != 0:
        ### The following code is a little messy, its a lot of pandas dataframe manipulation.
        ### I have tried doing an overhaul wherein the clusters and cluster_members dataframe are created per
        ### velocity combination in the clusterVelocity function. However, this adds an overhead in that function
        ### of ~ 1ms. So clustering 90,000 velocities takes 90 seconds longer which on small datasets is problematic.
        ### On large datasets, the effect is not as pronounced because the below code takes a while to run due to 
        ### in-memory pandas dataframe restructuring.

        # Make DataFrame with cluster velocities so we can figure out which 
        # velocities yielded clusters, add names to index so we can enable the join
        cluster_velocities = pd.DataFrame({"vtheta_x": vxx, "vtheta_y": vyy})
        cluster_velocities.index.set_names("velocity_id", inplace=True)

        # Split lists of cluster ids into one column per cluster for each different velocity
        # then stack the result
        possible_clusters = pd.DataFrame(
            possible_clusters["clusters"].values.tolist(), 
            index=possible_clusters.index
        )
        possible_clusters = pd.DataFrame(possible_clusters.stack())
        possible_clusters.rename(
            columns={0: "obs_ids"}, 
            inplace=True
        )
        possible_clusters = pd.DataFrame(possible_clusters["obs_ids"].values.tolist(), index=possible_clusters.index)

        # Drop duplicate clusters
        possible_clusters.drop_duplicates(inplace=True)

        # Set index names
        possible_clusters.index.set_names(["velocity_id", "cluster_id"], inplace=True)

        # Reset index
        possible_clusters.reset_index(
            "cluster_id", 
            drop=True, 
            inplace=True
        )
        possible_clusters["cluster_id"] = [str(uuid.uuid4().hex) for i in range(len(possible_clusters))]

        # Make all_clusters DataFrame
        all_clusters = possible_clusters.join(cluster_velocities)
        all_clusters.reset_index(drop=True, inplace=True)
        all_clusters = all_clusters[["cluster_id", "vtheta_x", "vtheta_y"]]

        # Make cluster_members DataFrame
        cluster_members = possible_clusters.reset_index(drop=True).copy()
        cluster_members.index = cluster_members["cluster_id"]
        cluster_members.drop("cluster_id", axis=1, inplace=True)
        cluster_members = pd.DataFrame(cluster_members.stack())
        cluster_members.rename(columns={0: "obs_id"}, inplace=True)
        cluster_members.reset_index(inplace=True)
        cluster_members.drop("level_1", axis=1, inplace=True)        

        # Calculate arc length and add it to the all_clusters dataframe
        cluster_members_time = cluster_members.merge(
            observations[["obs_id", "mjd_utc"]], 
            on="obs_id", 
            how="left"
        )
        all_clusters_time = cluster_members_time.groupby(
            by=["cluster_id"])["mjd_utc"].apply(lambda x: x.max() - x.min()).to_frame()
        all_clusters_time.reset_index(
            inplace=True
        )
        all_clusters_time.rename(
            columns={"mjd_utc" : "arc_length"}, 
            inplace=True
        )
        all_clusters = all_clusters.merge(
            all_clusters_time[["cluster_id", "arc_length"]],
            on="cluster_id",
            how="left",
        )
       
    else: 
        cluster_members = pd.DataFrame(columns=["cluster_id", "obs_id"])
        all_clusters = pd.DataFrame(columns=["cluster_id", "vtheta_x", "vtheta_y", "arc_length"])
        
        
    time_end_restr = time.time() 
    if verbose == True:
        print("Done. Completed in {} seconds.".format(time_end_restr - time_start_restr))
        print("")
        
    if identify_subsets == True:
        if verbose == True:
            print("Identifying subsets...")
        all_clusters, cluster_members = identifySubsetLinkages(
            all_clusters, 
            cluster_members,
            linkage_id_col="cluster_id"
        )
        if verbose == True:
            print("Done. {} subset clusters identified.".format(len(all_clusters[~all_clusters["subset_of"].isna()])))

   
    if verbose == True:
        print("")
        print("Found {} clusters.".format(len(all_clusters)))
        print("Total time in seconds: {}".format(time_end_restr - time_start_cluster))   
        print("-------------------------------")
        print("")
        
    return all_clusters, cluster_members

def runTHOROrbit(
        preprocessed_observations, 
        orbit,
        range_shift_config=Config.RANGE_SHIFT_CONFIG,
        cluster_link_config=Config.CLUSTER_LINK_CONFIG,
        iod_config=Config.IOD_CONFIG,
        od_config=Config.OD_CONFIG,
        odp_config=Config.ODP_CONFIG,
        out_dir=None,
        verbose=True
    ):
    
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
    
    # Run range and shift and get the projected observations
    print("Running range and shift...")
    projected_observations = rangeAndShift(
        preprocessed_observations, 
        orbit, 
        verbose=False,
        **range_shift_config
    )
    if out_dir is not None:
        projected_observations.to_csv(
            os.path.join(out_dir, "projected_observations.csv"),
            index=False
            
        )
    print("Done. Found {} observations.".format(len(projected_observations)))
    

    print("Running clustering...")
    clusters, cluster_members = clusterAndLink(
        projected_observations, 
        verbose=False,
        **cluster_link_config
    )
    if out_dir is not None:
        clusters.to_csv(
            os.path.join(out_dir, "clusters.csv"),
            index=False
        )
        cluster_members.to_csv(
            os.path.join(out_dir, "cluster_members.csv"),
            index=False
        )
    print("Done. Found {} clusters.".format(len(clusters)))

    print("Running initial orbit determination...")
    iod_orbits, iod_orbit_members = initialOrbitDetermination(
        projected_observations, 
        cluster_members, 
        verbose=False,
        **iod_config
    )
    if out_dir is not None:
        Orbits.from_df(iod_orbits).to_csv(
            os.path.join(out_dir, "iod_orbits.csv")
        )
        iod_orbit_members.to_csv(
            os.path.join(out_dir, "iod_orbit_members.csv"),
            index=False
        )
    print("Done. Found {} initial orbits.".format(len(iod_orbits)))
        
    iod_orbits = iod_orbits[["orbit_id", "epoch", "x", "y", "z", "vx", "vy", "vz"]]
    iod_orbit_members = iod_orbit_members[iod_orbit_members["outlier"] == 0][["orbit_id", "obs_id"]]

    print("Running differential correction...")
    od_orbits, od_orbit_members = differentialCorrection(
        iod_orbits,
        iod_orbit_members,
        projected_observations,
        verbose=False,
        **od_config
    )
    if out_dir is not None:
        Orbits.from_df(od_orbits).to_csv(
            os.path.join(out_dir, "od_orbits.csv")
        )
        od_orbit_members.to_csv(
            os.path.join(out_dir, "od_orbit_members.csv"),
            index=False
        )
    print("Done. Corrected {} initial orbits.".format(len(od_orbits)))
        
    od_orbits = od_orbits[["orbit_id", "epoch", "x", "y", "z", "vx", "vy", "vz", "covariance"]]
    od_orbit_members = od_orbit_members[od_orbit_members["outlier"] == 0][["orbit_id", "obs_id"]]

    print("Running orbit merging and extending...")
    odp_orbits, odp_orbit_members = mergeAndExtendOrbits(
        od_orbits, 
        od_orbit_members, 
        projected_observations, 
        verbose=False,
        **odp_config
    )
    print("Done. Merged/extended {} orbits.".format(len(od_orbits)))

    if out_dir is not None:
        Orbits.from_df(odp_orbits).to_csv(
            os.path.join(out_dir, "od+_orbits.csv")
        )
        odp_orbit_members.to_csv(
            os.path.join(out_dir, "od+_orbit_members.csv"),
            index=False
        )
        
    recovered_orbit_members = odp_orbit_members[odp_orbit_members["outlier"] == 0]
    recovered_orbit_members.drop(columns="outlier", inplace=True)
    recovered_orbits = odp_orbits[odp_orbits["orbit_id"].isin(odp_orbit_members["orbit_id"].unique())]
    for df in [recovered_orbits, recovered_orbit_members]:
        df.reset_index(
            inplace=True,
            drop=True
        )
        
    if out_dir is not None:
        Orbits.from_df(recovered_orbits).to_csv(
            os.path.join(out_dir, "recovered_orbits.csv")
        )
        recovered_orbit_members.to_csv(
            os.path.join(out_dir, "recovered_orbit_members.csv"),
            index=False
        )

    return recovered_orbits, recovered_orbit_members
