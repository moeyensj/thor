import logging
import time
from itertools import combinations
from typing import Literal, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.coordinates.residuals import Residuals
from adam_core.propagator import PYOORB, Propagator
from adam_core.propagator.utils import _iterate_chunks

from ..clusters import ClusterMembers
from ..observations.observations import Observations
from ..orbit_determination.fitted_orbits import FittedOrbitMembers, FittedOrbits
from .gauss import gaussIOD

logger = logging.getLogger(__name__)

__all__ = ["initial_orbit_determination"]


def select_observations(
    observations: Observations,
    method: Literal["combinations", "first+middle+last", "thirds"] = "combinations",
) -> npt.NDArray[np.str_]:
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

    Returns
    -------
    obs_id : `~numpy.ndarray' (N, 3 or 0)
        An array of selected observation IDs. If three unique observations could
        not be selected then returns an empty array.
    """
    obs_ids = observations.id.to_numpy(zero_copy_only=False)
    if len(obs_ids) < 3:
        return np.array([])

    indexes = np.arange(0, len(obs_ids))
    times = observations.coordinates.time.mjd().to_numpy(zero_copy_only=False)

    if method == "first+middle+last":
        selected_times = np.percentile(times, [0, 50, 100], interpolation="nearest")
        selected_index = np.intersect1d(times, selected_times, return_indices=True)[1]
        selected_index = np.array([selected_index])

    elif method == "thirds":
        selected_times = np.percentile(
            times, [1 / 6 * 100, 50, 5 / 6 * 100], interpolation="nearest"
        )
        selected_index = np.intersect1d(times, selected_times, return_indices=True)[1]
        selected_index = np.array([selected_index])

    elif method == "combinations":
        # Make all possible combinations of 3 observations
        selected_index = np.array(
            [np.array(index) for index in combinations(indexes, 3)]
        )

        # Calculate arc length
        arc_length = times[selected_index][:, 2] - times[selected_index][:, 0]

        # Calculate distance of second observation from middle point (last + first) / 2
        time_from_mid = np.abs(
            (times[selected_index][:, 2] + times[selected_index][:, 0]) / 2
            - times[selected_index][:, 1]
        )

        # Sort by descending arc length and ascending time from midpoint
        sort = np.lexsort((time_from_mid, -arc_length))
        selected_index = selected_index[sort]

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
    else:
        selected_index = selected_index[keep, :]

    return obs_ids[selected_index]


def iod_worker(
    linkage_ids: npt.NDArray[np.str_],
    observations: Union[Observations, ray.ObjectRef],
    linkage_members: Union[ClusterMembers, FittedOrbitMembers, ray.ObjectRef],
    min_obs: int = 6,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 0.0,
    rchi2_threshold: float = 200,
    observation_selection_method: Literal[
        "combinations", "first+middle+last", "thirds"
    ] = "combinations",
    linkage_id_col: str = "cluster_id",
    iterate: bool = False,
    light_time: bool = True,
    propagator: Type[Propagator] = PYOORB,
    propagator_kwargs: dict = {},
) -> Tuple[FittedOrbits, FittedOrbitMembers]:

    prop = propagator(**propagator_kwargs)

    iod_orbits_list = []
    iod_orbit_members_list = []
    for linkage_id in linkage_ids:

        time_start = time.time()
        logger.debug(f"Finding initial orbit for linkage {linkage_id}...")

        obs_ids = linkage_members.apply_mask(
            pc.equal(linkage_members.column(linkage_id_col), linkage_id)
        ).obs_id
        observations_linkage = observations.apply_mask(
            pc.is_in(observations.id, obs_ids)
        )

        iod_orbit, iod_orbit_members = iod(
            observations_linkage,
            min_obs=min_obs,
            min_arc_length=min_arc_length,
            rchi2_threshold=rchi2_threshold,
            contamination_percentage=contamination_percentage,
            observation_selection_method=observation_selection_method,
            iterate=iterate,
            light_time=light_time,
            propagator=prop,
        )
        if len(iod_orbit) > 0:
            iod_orbit = iod_orbit.set_column("orbit_id", pa.array([linkage_id]))
            iod_orbit_members = iod_orbit_members.set_column(
                "orbit_id",
                pa.array([linkage_id for i in range(len(iod_orbit_members))]),
            )

        time_end = time.time()
        duration = time_end - time_start
        logger.debug(f"IOD for linkage {linkage_id} completed in {duration:.3f}s.")

        iod_orbits_list.append(iod_orbit)
        iod_orbit_members_list.append(iod_orbit_members)

    iod_orbits = qv.concatenate(iod_orbits_list)
    iod_orbit_members = qv.concatenate(iod_orbit_members_list)
    return iod_orbits, iod_orbit_members


iod_worker_remote = ray.remote(iod_worker)
iod_worker_remote.options(num_returns=1, num_cpus=1)


def iod(
    observations: Observations,
    min_obs: int = 6,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 0.0,
    rchi2_threshold: float = 200,
    observation_selection_method: Literal[
        "combinations", "first+middle+last", "thirds"
    ] = "combinations",
    iterate: bool = False,
    light_time: bool = True,
    propagator: Propagator = PYOORB(),
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Run initial orbit determination on a set of observations believed to belong to a single
    object.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Dataframe of observations with at least the following columns:
            "obs_id" : Observation IDs [str],
            "mjd_utc" : Observation time in MJD UTC [float],
            "RA_deg" : equatorial J2000 Right Ascension in degrees [float],
            "Dec_deg" : equatorial J2000 Declination in degrees [float],
            "RA_sigma_deg" : 1-sigma uncertainty in equatorial J2000 RA [float],
            "Dec_sigma_deg" : 1 sigma uncertainty in equatorial J2000 Dec [float],
            "observatory_code" : MPC recognized observatory code [str],
            "obs_x" : Observatory's heliocentric ecliptic J2000 x-position in au [float],
            "obs_y" : Observatory's heliocentric ecliptic J2000 y-position in au [float],
            "obs_z" : Observatory's heliocentric ecliptic J2000 z-position in au [float],
            "obs_vx" [Optional] : Observatory's heliocentric ecliptic J2000 x-velocity in au per day [float],
            "obs_vy" [Optional] : Observatory's heliocentric ecliptic J2000 y-velocity in au per day [float],
            "obs_vz" [Optional] : Observatory's heliocentric ecliptic J2000 z-velocity in au per day [float]
    min_obs : int, optional
        Minimum number of observations that must remain in the linkage. For example, if min_obs is set to 6 and
        a linkage has 8 observations, at most the two worst observations will be flagged as outliers if their individual
        chi2 values exceed the chi2 threshold.
    contamination_percentage : float, optional
        Maximum percent of observations that can flagged as outliers.
    rchi2_threshold : float, optional
        Maximum reduced chi2 required for an initial orbit to be accepted.
    observation_selection_method : {'first+middle+last', 'thirds', 'combinations'}, optional
        Selects which three observations to use for IOD depending on the method. The avaliable methods are:
            'first+middle+last' : Grab the first, middle and last observations in time.
            'thirds' : Grab the middle observation in the first third, second third, and final third.
            'combinations' : Return the observation IDs corresponding to every possible combination of three observations with
                non-coinciding observation times.
    iterate : bool, optional
        Iterate the preliminary orbit solution using the state transition iterator.
    light_time : bool, optional
        Correct preliminary orbit for light travel time.
    linkage_id_col : str, optional
        Name of linkage_id column in the linkage_members dataframe.
    backend : {'MJOLNIR', 'PYOORB'}, optional
        Which backend to use for ephemeris generation.
    backend_kwargs : dict, optional
        Settings and additional parameters to pass to selected
        backend.

    Returns
    -------
    iod_orbits : `~pandas.DataFrame`
        Dataframe with orbits found in linkages.
            "orbit_id" : Orbit ID, a uuid [str],
            "epoch" : Epoch at which orbit is defined in MJD TDB [float],
            "x" : Orbit's ecliptic J2000 x-position in au [float],
            "y" : Orbit's ecliptic J2000 y-position in au [float],
            "z" : Orbit's ecliptic J2000 z-position in au [float],
            "vx" : Orbit's ecliptic J2000 x-velocity in au per day [float],
            "vy" : Orbit's ecliptic J2000 y-velocity in au per day [float],
            "vz" : Orbit's ecliptic J2000 z-velocity in au per day [float],
            "arc_length" : Arc length in days [float],
            "num_obs" : Number of observations that were within the chi2 threshold
                of the orbit.
            "chi2" : Total chi2 of the orbit calculated using the predicted location of the orbit
                on the sky compared to the consituent observations.

    iod_orbit_members : `~pandas.DataFrame`
        Dataframe of orbit members with the following columns:
            "orbit_id" : Orbit ID, a uuid [str],
            "obs_id" : Observation IDs [str], one ID per row.
            "residual_ra_arcsec" : Residual (observed - expected) equatorial J2000 Right Ascension in arcseconds [float]
            "residual_dec_arcsec" : Residual (observed - expected) equatorial J2000 Declination in arcseconds [float]
            "chi2" : Observation's chi2 [float]
            "gauss_sol" : Flag to indicate which observations were used to calculate the Gauss soluton [int]
            "outlier" : Flag to indicate which observations are potential outliers (their chi2 is higher than
                the chi2 threshold) [float]
    """
    processable = True
    if len(observations) == 0:
        processable = False

    obs_ids_all = observations.id.to_numpy(zero_copy_only=False)
    coords_all = observations.coordinates
    observers_with_states = observations.get_observers()
    observers = observers_with_states.observers
    coords_obs_all = observers_with_states.observers.coordinates.r
    times_all = coords_all.time.mjd().to_numpy(zero_copy_only=False)

    chi2_sol = 1e10
    orbit_sol: FittedOrbits = FittedOrbits.empty()
    obs_ids_sol = None
    arc_length = None
    outliers = np.array([])
    converged = False
    num_obs = len(observations)
    if num_obs < min_obs:
        processable = False
    num_outliers = int(num_obs * contamination_percentage / 100.0)
    num_outliers = np.maximum(np.minimum(num_obs - min_obs, num_outliers), 0)

    # Select observation IDs to use for IOD
    obs_ids = select_observations(
        observations,
        method=observation_selection_method,
    )
    obs_ids = obs_ids[: (3 * (num_outliers + 1))]

    if len(obs_ids) == 0:
        processable = False

    j = 0
    while not converged and processable:
        if j == len(obs_ids):
            break

        ids = obs_ids[j]
        mask = np.isin(obs_ids_all, ids)

        # Grab sky-plane positions of the selected observations, the heliocentric ecliptic position of the observer,
        # and the times at which the observations occur
        coords = coords_all.apply_mask(mask)
        coords_obs = coords_obs_all[mask, :]
        times = times_all[mask]

        # Run IOD
        iod_orbits = gaussIOD(
            coords.values[:, 1:3],
            times,
            coords_obs,
            light_time=light_time,
            iterate=iterate,
            max_iter=100,
            tol=1e-15,
        )
        if len(iod_orbits) == 0:
            j += 1
            continue

        # Propagate initial orbit to all observation times
        ephemeris = propagator.generate_ephemeris(
            iod_orbits, observers, chunk_size=1, max_processes=1
        )

        # For each unique initial orbit calculate residuals and chi-squared
        # Find the orbit which yields the lowest chi-squared
        orbit_ids = iod_orbits.orbit_id.to_numpy(zero_copy_only=False)
        for i, orbit_id in enumerate(orbit_ids):
            ephemeris_orbit = ephemeris.select("orbit_id", orbit_id)

            # Calculate residuals and chi2
            residuals = Residuals.calculate(
                coords_all,
                ephemeris_orbit.coordinates,
            )
            chi2 = residuals.chi2.to_numpy()
            chi2_total = np.sum(chi2)
            rchi2 = chi2_total / (2 * num_obs - 6)

            # The reduced chi2 is above the threshold and no outliers are
            # allowed, this cannot be improved by outlier rejection
            # so continue to the next IOD orbit
            if rchi2 > rchi2_threshold and num_outliers == 0:
                # If we have iterated through all iod orbits and no outliers
                # are allowed for this linkage then no other combination of
                # observations will make it acceptable, so exit here.
                if (i + 1) == len(iod_orbits):
                    processable = False
                    break

                continue

            # If the total reduced chi2 is less than the threshold accept the orbit
            elif rchi2 <= rchi2_threshold:
                logger.debug("Potential solution orbit has been found.")
                orbit_sol = iod_orbits[i : i + 1]
                obs_ids_sol = ids
                chi2_total_sol = chi2_total
                chi2_sol = chi2
                rchi2_sol = rchi2
                residuals_sol = residuals
                outliers = np.array([])
                arc_length = times_all.max() - times_all.min()
                converged = True
                break

            # Let's now test to see if we can remove some outliers, we
            # anticipate that we get to this stage if the three selected observations
            # belonging to one object yield a good initial orbit but the presence of outlier
            # observations is skewing the sum total of the residuals and chi2
            elif num_outliers > 0:

                logger.debug("Attempting to identify possible outliers.")
                for o in range(num_outliers):
                    # Select i highest observations that contribute to
                    # chi2 (and thereby the residuals)
                    remove = chi2[~mask].argsort()[-(o + 1) :]

                    # Grab the obs_ids for these outliers
                    obs_id_outlier = obs_ids_all[~mask][remove]
                    logger.debug("Possible outlier(s): {}".format(obs_id_outlier))

                    # Subtract the outlier's chi2 contribution
                    # from the total chi2
                    # Then recalculate the reduced chi2
                    chi2_new = chi2_total - np.sum(chi2[~mask][remove])
                    num_obs_new = len(observations) - len(remove)
                    rchi2_new = chi2_new / (2 * num_obs_new - 6)

                    ids_mask = np.isin(obs_ids_all, obs_id_outlier, invert=True)
                    arc_length = times_all[ids_mask].max() - times_all[ids_mask].min()

                    # If the updated reduced chi2 total is lower than our desired
                    # threshold, accept the soluton. If not, keep going.
                    if rchi2_new <= rchi2_threshold and arc_length >= min_arc_length:
                        orbit_sol = iod_orbits[i : i + 1]
                        obs_ids_sol = ids
                        chi2_total_sol = chi2_new
                        rchi2_sol = rchi2_new
                        residuals_sol = residuals
                        outliers = obs_id_outlier
                        num_obs = num_obs_new
                        ids_mask = np.isin(obs_ids_all, outliers, invert=True)
                        arc_length = (
                            times_all[ids_mask].max() - times_all[ids_mask].min()
                        )
                        chi2_sol = chi2
                        converged = True
                        break

            else:
                continue

        j += 1

    if not converged or not processable:

        return FittedOrbits.empty(), FittedOrbitMembers.empty()

    else:

        orbit = FittedOrbits.from_kwargs(
            orbit_id=orbit_sol.orbit_id,
            object_id=orbit_sol.object_id,
            coordinates=orbit_sol.coordinates,
            arc_length=[arc_length],
            num_obs=[num_obs],
            chi2=[chi2_total_sol],
            reduced_chi2=[rchi2_sol],
        )

        orbit_members = FittedOrbitMembers.from_kwargs(
            orbit_id=np.full(len(obs_ids_all), orbit_sol.orbit_id[0].as_py()),
            obs_id=obs_ids_all,
            residuals=residuals_sol,
            solution=np.isin(obs_ids_all, obs_ids_sol),
            outlier=np.isin(obs_ids_all, outliers),
        )

    return orbit, orbit_members


def initial_orbit_determination(
    observations: Union[Observations, ray.ObjectRef],
    linkage_members: Union[ClusterMembers, FittedOrbitMembers, ray.ObjectRef],
    min_obs: int = 6,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 20.0,
    rchi2_threshold: float = 10**3,
    observation_selection_method: Literal[
        "combinations", "first+middle+last", "thirds"
    ] = "combinations",
    iterate: bool = False,
    light_time: bool = True,
    linkage_id_col: str = "cluster_id",
    propagator: Type[Propagator] = PYOORB,
    propagator_kwargs: dict = {},
    chunk_size: int = 1,
    max_processes: Optional[int] = 1,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Run initial orbit determination on linkages found in observations.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Dataframe of observations with at least the following columns:
            "obs_id" : Observation IDs [str],
            "mjd_utc" : Observation time in MJD UTC [float],
            "RA_deg" : equatorial J2000 Right Ascension in degrees [float],
            "Dec_deg" : equatorial J2000 Declination in degrees [float],
            "RA_sigma_deg" : 1-sigma uncertainty in equatorial J2000 RA [float],
            "Dec_sigma_deg" : 1 sigma uncertainty in equatorial J2000 Dec [float],
            "observatory_code" : MPC recognized observatory code [str],
            "obs_x" : Observatory's heliocentric ecliptic J2000 x-position in au [float],
            "obs_y" : Observatory's heliocentric ecliptic J2000 y-position in au [float],
            "obs_z" : Observatory's heliocentric ecliptic J2000 z-position in au [float],
            "obs_vx" [Optional] : Observatory's heliocentric ecliptic J2000 x-velocity in au per day [float],
            "obs_vy" [Optional] : Observatory's heliocentric ecliptic J2000 y-velocity in au per day [float],
            "obs_vz" [Optional] : Observatory's heliocentric ecliptic J2000 z-velocity in au per day [float]
    linkage_members : `~pandas.DataFrame`
        Dataframe of linkages with at least two columns:
            "linkage_id" : Linkage ID [str],
            "obs_id" : Observation IDs [str], one ID per row.
    observation_selection_method : {'first+middle+last', 'thirds', 'combinations'}, optional
        Selects which three observations to use for IOD depending on the method. The avaliable methods are:
            'first+middle+last' : Grab the first, middle and last observations in time.
            'thirds' : Grab the middle observation in the first third, second third, and final third.
            'combinations' : Return the observation IDs corresponding to every possible combination of three observations with
                non-coinciding observation times.
    min_obs : int, optional
        Minimum number of observations that must remain in the linkage. For example, if min_obs is set to 6 and
        a linkage has 8 observations, at most the two worst observations will be flagged as outliers. Only up t o
        the contamination percentage of observations of will be flagged as outliers, provided that at least min_obs
        observations remain in the linkage.
    rchi2_threshold : float, optional
        Minimum reduced chi2 for an initial orbit to be accepted. If an orbit
    contamination_percentage : float, optional
        Maximum percent of observations that can flagged as outliers.
    iterate : bool, optional
        Iterate the preliminary orbit solution using the state transition iterator.
    light_time : bool, optional
        Correct preliminary orbit for light travel time.
    linkage_id_col : str, optional
        Name of linkage_id column in the linkage_members dataframe.
    backend : {'MJOLNIR', 'PYOORB'}, optional
        Which backend to use for ephemeris generation.
    backend_kwargs : dict, optional
        Settings and additional parameters to pass to selected
        backend.
    chunk_size : int, optional
        Number of linkages to send to each job.
    num_jobs : int, optional
        Number of jobs to launch.
    parallel_backend : str, optional
        Which parallelization backend to use {'ray', 'mp', 'cf'}. Defaults to using Python's concurrent.futures module ('cf').

    Returns
    -------
    iod_orbits : `~pandas.DataFrame`
        Dataframe with orbits found in linkages.
            "orbit_id" : Orbit ID, a uuid [str],
            "epoch" : Epoch at which orbit is defined in MJD TDB [float],
            "x" : Orbit's ecliptic J2000 x-position in au [float],
            "y" : Orbit's ecliptic J2000 y-position in au [float],
            "z" : Orbit's ecliptic J2000 z-position in au [float],
            "vx" : Orbit's ecliptic J2000 x-velocity in au per day [float],
            "vy" : Orbit's ecliptic J2000 y-velocity in au per day [float],
            "vz" : Orbit's ecliptic J2000 z-velocity in au per day [float],
            "arc_length" : Arc length in days [float],
            "num_obs" : Number of observations that were within the chi2 threshold
                of the orbit.
            "chi2" : Total chi2 of the orbit calculated using the predicted location of the orbit
                on the sky compared to the consituent observations.

    iod_orbit_members : `~pandas.DataFrame`
        Dataframe of orbit members with the following columns:
            "orbit_id" : Orbit ID, a uuid [str],
            "obs_id" : Observation IDs [str], one ID per row.
            "residual_ra_arcsec" : Residual (observed - expected) equatorial J2000 Right Ascension in arcseconds [float]
            "residual_dec_arcsec" : Residual (observed - expected) equatorial J2000 Declination in arcseconds [float]
            "chi2" : Observation's chi2 [float]
            "gauss_sol" : Flag to indicate which observations were used to calculate the Gauss soluton [int]
            "outlier" : Flag to indicate which observations are potential outliers (their chi2 is higher than
                the chi2 threshold) [float]
    """
    time_start = time.time()
    logger.info("Running initial orbit determination...")

    iod_orbits_list = []
    iod_orbit_members_list = []
    if len(observations) > 0 and len(linkage_members) > 0:

        # Extract linkage IDs
        linkage_ids = linkage_members.column(linkage_id_col).unique()

        if max_processes is None or max_processes > 1:

            if not ray.is_initialized():
                ray.init(address="auto")

            observations_ref = ray.put(observations)
            linkage_members_ref = ray.put(linkage_members)

            futures = []
            for linkage_id_chunk in _iterate_chunks(linkage_ids, chunk_size):
                futures.append(
                    iod_worker_remote.remote(
                        linkage_id_chunk,
                        observations_ref,
                        linkage_members_ref,
                        min_obs=min_obs,
                        min_arc_length=min_arc_length,
                        contamination_percentage=contamination_percentage,
                        rchi2_threshold=rchi2_threshold,
                        observation_selection_method=observation_selection_method,
                        iterate=iterate,
                        light_time=light_time,
                        linkage_id_col=linkage_id_col,
                        propagator=propagator,
                        propagator_kwargs=propagator_kwargs,
                    )
                )

            while futures:
                finished, futures = ray.wait(futures, num_returns=1)
                result = ray.get(finished[0])
                iod_orbits_list.append(result[0])
                iod_orbit_members_list.append(result[1])

        else:
            for linkage_id_chunk in _iterate_chunks(linkage_ids, chunk_size):
                iod_orbits_chunk, iod_orbit_members_chunk = iod_worker(
                    linkage_id_chunk,
                    observations,
                    linkage_members,
                    min_obs=min_obs,
                    min_arc_length=min_arc_length,
                    contamination_percentage=contamination_percentage,
                    rchi2_threshold=rchi2_threshold,
                    observation_selection_method=observation_selection_method,
                    iterate=iterate,
                    light_time=light_time,
                    linkage_id_col=linkage_id_col,
                    propagator=propagator,
                    propagator_kwargs=propagator_kwargs,
                )
                iod_orbits_list.append(iod_orbits_chunk)
                iod_orbit_members_list.append(iod_orbit_members_chunk)

        iod_orbits = qv.concatenate(iod_orbits_list)
        iod_orbit_members = qv.concatenate(iod_orbit_members_list)

        iod_orbits, iod_orbit_members = iod_orbits.drop_duplicates(
            iod_orbit_members,
            subset=[
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.x",
                "coordinates.y",
                "coordinates.z",
                "coordinates.vx",
                "coordinates.vy",
                "coordinates.vz",
            ],
            keep="first",
        )

        logger.info("Found {} initial orbits.".format(len(iod_orbits)))

    else:
        iod_orbits = FittedOrbits.empty()
        iod_orbit_members = FittedOrbitMembers.empty()

    time_end = time.time()
    logger.info(
        "Initial orbit determination completed in {:.3f} seconds.".format(
            time_end - time_start
        )
    )

    return iod_orbits, iod_orbit_members
