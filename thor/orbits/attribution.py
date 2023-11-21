import logging
import time
from typing import Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.coordinates.residuals import Residuals
from adam_core.orbits import Orbits
from adam_core.propagator import PYOORB
from adam_core.propagator.utils import _iterate_chunks
from sklearn.neighbors import BallTree

from ..observations.observations import Observations
from ..orbit_determination import FittedOrbitMembers, FittedOrbits
from .od import differential_correction

logger = logging.getLogger(__name__)

__all__ = ["Attributions", "attribute_observations", "merge_and_extend_orbits"]


LATLOT_INDEX = np.array([2, 1])


class Attributions(qv.Table):
    orbit_id = qv.StringColumn()
    obs_id = qv.StringColumn()
    residuals = Residuals.as_column(nullable=True)
    distance = qv.Float64Column(nullable=True)

    def drop_coincident_attributions(
        self, observations: Observations
    ) -> "Attributions":
        """
        Drop attributions that are coincident in time: two or more observations attributed
        to the same orbit that occur at the same time. The observation with the lowest
        distance is kept.

        Parameters
        ----------
        observations : `~thor.observations.observations.Observations`
            Observations which will be used to get the observation times.

        Returns
        -------
        attributions : `~thor.orbits.attribution.Attributions`
            Attributions table with coincident attributions removed.
        """
        # Flatten the table so nested columns are dot-delimited at the top level
        flattened_table = self.flattened_table()

        # Drop the residual values (a list column) due to: https://github.com/apache/arrow/issues/32504
        flattened_table = flattened_table.drop(["residuals.values"])

        # Filter the observations to only include those that have been attributed
        # to an orbit
        observations_filtered = observations.apply_mask(
            pc.is_in(observations.id, flattened_table.column("obs_id"))
        )

        # Flatten the observations table
        flattened_observations = observations_filtered.flattened_table()

        # Only keep relevant columns
        flattened_observations = flattened_observations.select(
            ["id", "coordinates.time.days", "coordinates.time.nanos"]
        )

        # Join the time column back to the flattened attributions table
        flattened_table = flattened_table.join(
            flattened_observations, ["obs_id"], right_keys=["id"]
        )

        # Add index column
        flattened_table = flattened_table.add_column(
            0, "index", pa.array(np.arange(len(flattened_table)))
        )

        # Sort the table
        flattened_table = flattened_table.sort_by(
            [
                ("orbit_id", "ascending"),
                ("coordinates.time.days", "ascending"),
                ("coordinates.time.nanos", "ascending"),
            ]
        )

        # Group by orbit ID and observation time
        indices = (
            flattened_table.group_by(
                ["orbit_id", "coordinates.time.days", "coordinates.time.nanos"],
                use_threads=False,
            )
            .aggregate([("index", "first")])
            .column("index_first")
        )

        return self.take(indices)


def attribution_worker(
    orbit_ids: npt.NDArray[np.str_],
    observation_indices: npt.NDArray[np.int64],
    orbits: Union[Orbits, FittedOrbits],
    observations: Observations,
    radius: float = 1 / 3600,
    propagator: Literal["PYOORB"] = "PYOORB",
    propagator_kwargs: dict = {},
) -> Attributions:
    if propagator == "PYOORB":
        prop = PYOORB(**propagator_kwargs)
    else:
        raise ValueError(f"Invalid propagator '{propagator}'.")

    if isinstance(orbits, FittedOrbits):
        orbits = orbits.to_orbits()

    # Select the orbits and observations for this batch
    observations = observations.take(observation_indices)
    orbits = orbits.apply_mask(pc.is_in(orbits.orbit_id, orbit_ids))

    # Get the unique observers for this batch of observations
    observers_with_states = observations.get_observers()
    observers = observers_with_states.observers

    # Generate ephemerides for each orbit at the observation times
    ephemeris = prop.generate_ephemeris(
        orbits, observers, chunk_size=len(orbits), max_processes=1
    )

    # Round the ephemeris and observations to the nearest millisecond
    ephemeris = ephemeris.set_column(
        "coordinates.time", ephemeris.coordinates.time.rounded(precision="ms")
    )
    observations_rounded = observations.set_column(
        "coordinates.time", observations.coordinates.time.rounded(precision="ms")
    )

    # Create a linkage between the ephemeris and observations
    linkage = qv.MultiKeyLinkage(
        ephemeris,
        observations_rounded,
        left_keys={
            "days": ephemeris.coordinates.time.days,
            "nanos": ephemeris.coordinates.time.nanos,
            "code": ephemeris.coordinates.origin.code,
        },
        right_keys={
            "days": observations_rounded.coordinates.time.days,
            "nanos": observations_rounded.coordinates.time.nanos,
            "code": observations_rounded.coordinates.origin.code,
        },
    )

    # Loop through each unique exposure and visit, find the nearest observations within
    # eps (haversine metric)
    distances = []
    orbit_ids_associated = []
    obs_ids_associated = []
    obs_times_associated = []
    radius_rad = np.radians(radius)
    residuals = []
    for _, ephemeris_i, observations_i in linkage.iterate():

        # Extract the observation IDs and times
        obs_ids = observations_i.id.to_numpy(zero_copy_only=False)
        obs_times = observations_i.coordinates.time.mjd().to_numpy(zero_copy_only=False)
        orbit_ids = ephemeris_i.orbit_id.to_numpy(zero_copy_only=False)

        # Extract the spherical coordinates for both the observations
        # and ephemeris
        coords = observations_i.coordinates
        coords_predicted = ephemeris_i.coordinates

        # Haversine metric requires latitude first then longitude...
        coords_latlon = np.radians(coords.values[:, LATLOT_INDEX])
        coords_predicted_latlon = np.radians(coords_predicted.values[:, LATLOT_INDEX])

        num_obs = len(coords_predicted)
        k = np.minimum(3, num_obs)

        # Build BallTree with a haversine metric on predicted ephemeris
        tree = BallTree(coords_predicted_latlon, metric="haversine")
        # Query tree using observed RA, Dec
        d, i = tree.query(
            coords_latlon,
            k=k,
            return_distance=True,
            dualtree=True,
            breadth_first=False,
            sort_results=False,
        )

        # Select all observations with distance smaller or equal
        # to the maximum given distance
        mask = np.where(d <= radius_rad)

        if len(d[mask]) > 0:
            orbit_ids_associated.append(orbit_ids[i[mask]])
            obs_ids_associated.append(obs_ids[mask[0]])
            obs_times_associated.append(obs_times[mask[0]])
            distances.append(d[mask])

            residuals_i = Residuals.calculate(
                coords.take(mask[0]), coords_predicted.take(i[mask])
            )
            residuals.append(residuals_i)

    if len(distances) > 0:
        distances = np.degrees(np.concatenate(distances))
        orbit_ids_associated = np.concatenate(orbit_ids_associated)
        obs_ids_associated = np.concatenate(obs_ids_associated)
        obs_times_associated = np.concatenate(obs_times_associated)
        residuals = qv.concatenate(residuals)

        return Attributions.from_kwargs(
            orbit_id=orbit_ids_associated,
            obs_id=obs_ids_associated,
            residuals=residuals,
            distance=distances,
        )

    else:
        return Attributions.empty()


attribution_worker_remote = ray.remote(attribution_worker)
attribution_worker_remote.options(
    num_returns=1,
    num_cpus=1,
)


def attribute_observations(
    orbits: Union[Orbits, FittedOrbits],
    observations: Observations,
    radius: float = 5 / 3600,
    propagator: Literal["PYOORB"] = "PYOORB",
    propagator_kwargs: dict = {},
    orbits_chunk_size: int = 10,
    observations_chunk_size: int = 100000,
    max_processes: Optional[int] = 1,
) -> Attributions:
    logger.info("Running observation attribution...")
    time_start = time.time()

    if isinstance(orbits, FittedOrbits):
        orbits = orbits.to_orbits()

    orbit_ids = orbits.orbit_id
    observation_indices = np.arange(0, len(observations))

    attributions_list = []
    if max_processes is None or max_processes > 1:

        if not ray.is_initialized():
            ray.init(address="auto")

        observations_ref = ray.put(observations)
        orbits_ref = ray.put(orbits)

        futures = []
        for orbit_id_chunk in _iterate_chunks(orbit_ids, orbits_chunk_size):
            for observations_indices_chunk in _iterate_chunks(
                observation_indices, observations_chunk_size
            ):
                futures.append(
                    attribution_worker_remote.remote(
                        orbit_id_chunk,
                        observations_indices_chunk,
                        orbits_ref,
                        observations_ref,
                        radius=radius,
                        propagator=propagator,
                        propagator_kwargs=propagator_kwargs,
                    )
                )

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            attributions_list.append(ray.get(finished[0]))

    else:
        for orbit_id_chunk in _iterate_chunks(orbit_ids, orbits_chunk_size):
            for observations_indices_chunk in _iterate_chunks(
                observation_indices, observations_chunk_size
            ):
                attribution_df_i = attribution_worker(
                    orbit_id_chunk,
                    observations_indices_chunk,
                    orbits,
                    observations,
                    radius=radius,
                    propagator=propagator,
                    propagator_kwargs=propagator_kwargs,
                )
                attributions_list.append(attribution_df_i)

    attributions = qv.concatenate(attributions_list)
    attributions = attributions.sort_by(["orbit_id", "obs_id", "distance"])

    time_end = time.time()
    logger.info(
        f"Attributed {len(attributions.obs_id.unique())} observations to {len(attributions.orbit_id.unique())} orbits."
    )
    logger.info(
        "Attribution completed in {:.3f} seconds.".format(time_end - time_start)
    )
    return attributions


def merge_and_extend_orbits(
    orbits: Union[FittedOrbits, ray.ObjectRef],
    orbit_members: Union[FittedOrbitMembers, ray.ObjectRef],
    observations: Union[Observations, ray.ObjectRef],
    min_obs: int = 6,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 20.0,
    rchi2_threshold: float = 5,
    radius: float = 1 / 3600,
    delta: float = 1e-8,
    max_iter: int = 20,
    method: Literal["central", "finite"] = "central",
    fit_epoch: bool = False,
    propagator: Literal["PYOORB"] = "PYOORB",
    propagator_kwargs: dict = {},
    orbits_chunk_size: int = 10,
    observations_chunk_size: int = 100000,
    max_processes: Optional[int] = 1,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Attempt to extend an orbit's observational arc by running
    attribution on the observations. This is an iterative process: attribution
    is run, any observations found for each orbit are added to that orbit and differential correction is
    run. Orbits which are subset's of other orbits are removed. Iteration continues until there are no
    duplicate observation assignments.

    Parameters
    ----------
    orbit_chunk_size : int, optional
        Number of orbits to send to each job.
    observations_chunk_size : int, optional
        Number of observations to process per batch.
    num_jobs : int, optional
        Number of jobs to launch.
    parallel_backend : str, optional
        Which parallelization backend to use {'ray', 'mp', cf}. Defaults to using Python's concurrent.futures
        module ('cf').
    """
    time_start = time.perf_counter()
    logger.info("Running orbit extension and merging...")

    if isinstance(orbits, ray.ObjectRef):
        orbits = ray.get(orbits)
        logger.info("Retrieved orbits from the object store.")

    if isinstance(orbit_members, ray.ObjectRef):
        orbit_members = ray.get(orbit_members)
        logger.info("Retrieved orbit members from the object store.")

    if isinstance(observations, ray.ObjectRef):
        observations = ray.get(observations)
        logger.info("Retrieved observations from the object store.")

    # Set the running variables
    orbits_iter = orbits
    orbit_members_iter = orbit_members
    observations_iter = observations

    iterations = 0
    odp_orbits_list = []
    odp_orbit_members_list = []
    if len(orbits_iter) > 0 and len(observations_iter) > 0:

        converged = False
        while not converged:
            # Run attribution
            attributions = attribute_observations(
                orbits_iter,
                observations_iter,
                radius=radius,
                propagator=propagator,
                propagator_kwargs=propagator_kwargs,
                orbits_chunk_size=orbits_chunk_size,
                observations_chunk_size=observations_chunk_size,
                max_processes=max_processes,
            )

            # For orbits with coincident observations: multiple observations attributed at
            # the same time, keep only observation with smallest distance
            attributions = attributions.drop_coincident_attributions(observations)

            # Create a new orbit members table with the newly attributed observations and
            # filter the orbits to only include those that still have observations
            orbit_members_iter = FittedOrbitMembers.from_kwargs(
                orbit_id=attributions.orbit_id,
                obs_id=attributions.obs_id,
                residuals=attributions.residuals,
            )
            orbits_iter = orbits_iter.apply_mask(
                pc.is_in(orbits_iter.orbit_id, orbit_members_iter.orbit_id.unique())
            )

            # Run differential orbit correction
            orbits_iter, orbit_members_iter = differential_correction(
                orbits_iter,
                orbit_members_iter,
                observations_iter,
                rchi2_threshold=rchi2_threshold,
                min_obs=min_obs,
                min_arc_length=min_arc_length,
                contamination_percentage=contamination_percentage,
                delta=delta,
                method=method,
                max_iter=max_iter,
                fit_epoch=fit_epoch,
                propagator=propagator,
                propagator_kwargs=propagator_kwargs,
                chunk_size=orbits_chunk_size,
                max_processes=max_processes,
            )
            orbit_members_iter = orbit_members_iter.drop_outliers()

            # Remove any duplicate orbits
            orbits_iter = orbits_iter.sort_by([("reduced_chi2", "ascending")])
            orbits_iter, orbit_members_iter = orbits_iter.drop_duplicates(
                orbit_members_iter,
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

            # Remove the orbits that were not improved from the pool of available orbits. Orbits that were not improved
            # are orbits that have already iterated to their best-fit solution given the observations available. These orbits
            # are unlikely to recover more observations in subsequent iterations and so can be saved for output.
            not_improved_mask = pc.equal(orbits_iter.improved, False)
            orbits_out = orbits_iter.apply_mask(not_improved_mask)
            orbit_members_out = orbit_members_iter.apply_mask(
                pc.is_in(orbit_members_iter.orbit_id, orbits_out.orbit_id)
            )

            # If some of the orbits that haven't improved in their orbit fit still share observations
            # then assign those observations to the orbit with the most observations, longest arc length,
            # and lowest reduced chi2 (and remove the other instances of that observation)
            # We will one final iteration of OD on the output orbits later
            orbits_out, orbit_members_out = orbits_out.assign_duplicate_observations(
                orbit_members_out
            )

            # Add these orbits to the output list
            odp_orbits_list.append(orbits_out)
            odp_orbit_members_list.append(orbit_members_out)

            # Remove observations that have been added to the output list of orbits
            observations_iter = observations_iter.apply_mask(
                pc.invert(
                    pc.is_in(
                        observations_iter.id,
                        orbit_members_out.obs_id.unique(),
                    )
                )
            )

            # Identify the orbit that could still be improved more
            improved_mask = pc.invert(not_improved_mask)
            orbits_iter = orbits_iter.apply_mask(improved_mask)
            orbit_members_iter = orbit_members_iter.apply_mask(
                pc.is_in(orbit_members_iter.orbit_id, orbits_iter.orbit_id)
            )

            # Remove observations that have been added to the output list of orbits
            # from the orbits that we will continue iterating over
            orbit_members_iter = orbit_members_iter.apply_mask(
                pc.invert(
                    pc.is_in(
                        orbit_members_iter.obs_id,
                        orbit_members_out.obs_id.unique(),
                    )
                )
            )
            orbits_iter = orbits_iter.apply_mask(
                pc.is_in(orbits_iter.orbit_id, orbit_members_iter.orbit_id.unique())
            )

            iterations += 1
            if len(orbits_iter) == 0:
                converged = True

        odp_orbits = qv.concatenate(odp_orbits_list)
        odp_orbit_members = qv.concatenate(odp_orbit_members_list)

        if len(odp_orbits) > 0:
            # Do one final iteration of OD on the output orbits. This
            # will update any fits of orbits that might have had observations
            # removed during the assign_duplicate_observations step
            odp_orbits, odp_orbit_members = differential_correction(
                odp_orbits,
                odp_orbit_members,
                observations,
                rchi2_threshold=rchi2_threshold,
                min_obs=min_obs,
                min_arc_length=min_arc_length,
                contamination_percentage=contamination_percentage,
                delta=delta,
                method=method,
                max_iter=max_iter,
                fit_epoch=fit_epoch,
                propagator=propagator,
                propagator_kwargs=propagator_kwargs,
                chunk_size=orbits_chunk_size,
                max_processes=max_processes,
            )
            odp_orbit_members = odp_orbit_members.drop_outliers()

    else:
        odp_orbits = FittedOrbits.empty()
        odp_orbit_members = FittedOrbitMembers.empty()

    time_end = time.perf_counter()
    logger.info(
        f"Number of attribution / differential correction iterations: {iterations}"
    )
    logger.info(
        f"Extended and/or merged {len(orbits)} orbits into {len(odp_orbits)} orbits."
    )
    logger.info(
        "Orbit extension and merging completed in {:.3f} seconds.".format(
            time_end - time_start
        )
    )

    return odp_orbits, odp_orbit_members
