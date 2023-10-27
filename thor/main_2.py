import logging
import time
from typing import Any, Iterator, List, Optional, Union

import pandas as pd
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.coordinates import (
    CartesianCoordinates,
    OriginCodes,
    transform_coordinates,
)
from adam_core.propagator import PYOORB, Propagator

from .config import Config
from .main import clusterAndLink
from .observations.filters import ObservationFilter, TestOrbitRadiusObservationFilter
from .observations.observations import Observations, ObserversWithStates
from .orbit import TestOrbit, TestOrbitEphemeris
from .orbits import (
    differentialCorrection,
    initialOrbitDetermination,
    mergeAndExtendOrbits,
)
from .projections import GnomonicCoordinates

logger = logging.getLogger(__name__)


class TransformedDetections(qv.Table):
    id = qv.StringColumn()
    coordinates = GnomonicCoordinates.as_column()
    state_id = qv.Int64Column()


def range_and_transform_worker(
    ranged_detections: CartesianCoordinates,
    observations: Observations,
    ephemeris: TestOrbitEphemeris,
    state_id: int,
) -> TransformedDetections:
    """
    Given ranged detections and their original observations, transform these to the gnomonic tangent
    plane centered on the motion of the test orbit for a single state.

    Parameters
    ----------
    ranged_detections
        Cartesian detections ranged so that their heliocentric distance is the same as the test orbit
        for each state
    observations
        The observations from which the ranged detections originate. These should be sorted one-to-one
        with the ranged detections
    ephemeris
        Ephemeris from which to extract the test orbit's aberrated state.
    state_id
        The ID for this particular state.

    Returns
    -------
    transformed_detections
        Detections transformed to a gnomonic tangent plane centered on the motion of the
        test orbit.
    """
    # Select the detections and ephemeris for this state id
    mask = pc.equal(state_id, observations.state_id)
    ranged_detections_state = ranged_detections.apply_mask(mask)
    ephemeris_state = ephemeris.select("id", state_id)
    observations_state = observations.select("state_id", state_id)

    # Transform the detections into the co-rotating frame
    return TransformedDetections.from_kwargs(
        id=observations_state.detections.id,
        coordinates=GnomonicCoordinates.from_cartesian(
            ranged_detections_state,
            center_cartesian=ephemeris_state.ephemeris.aberrated_coordinates,
        ),
        state_id=observations_state.state_id,
    )


range_and_transform_remote = ray.remote(range_and_transform_worker)


def range_and_transform(
    test_orbit: TestOrbit,
    observations: Union[Observations, ray.ObjectRef],
    propagator: Propagator = PYOORB(),
    max_processes: Optional[int] = 1,
) -> TransformedDetections:
    """
    Range observations for a single test orbit and transform them into a
    gnomonic projection centered on the motion of the test orbit (co-rotating
    frame).

    Parameters
    ----------
    test_orbit : `~thor.orbit.TestOrbit`
        Test orbit to use to gather and transform observations.
    observations : `~thor.observations.observations.Observations`
        Observations from which range and transform the detections.
    propagator : `~adam_core.propagator.propagator.Propagator`
        Propagator to use to propagate the test orbit and generate
        ephemerides.
    max_processes : int, optional
        Maximum number of processes to use for parallelization. If
        an existing ray cluster is already running, this parameter
        will be ignored if larger than 1 or not None.

    Returns
    -------
    transformed_detections : `~thor.main.TransformedDetections`
        The transformed detections as gnomonic coordinates
        of the observations in the co-rotating frame.
    """
    time_start = time.perf_counter()
    logger.info("Running range and transform...")
    logger.info(f"Assuming r = {test_orbit.orbit.coordinates.r[0]} au")
    logger.info(f"Assuming v = {test_orbit.orbit.coordinates.v[0]} au/d")

    # Compute the ephemeris of the test orbit (this will be cached)
    ephemeris = test_orbit.generate_ephemeris_from_observations(
        observations,
        propagator=propagator,
        max_processes=max_processes,
    )

    # Assume that the heliocentric distance of all point sources in
    # the observations are the same as that of the test orbit
    ranged_detections_spherical = test_orbit.range_observations(
        observations,
        propagator=propagator,
        max_processes=max_processes,
    )

    # Transform from spherical topocentric to cartesian heliocentric coordinates
    ranged_detections_cartesian = transform_coordinates(
        ranged_detections_spherical.coordinates,
        representation_out=CartesianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )

    transformed_detection_list = []
    if max_processes is None or max_processes > 1:

        if not ray.is_initialized():
            logger.debug(
                f"Ray is not initialized. Initializing with {max_processes}..."
            )
            ray.init(num_cpus=max_processes)

        if isinstance(observations, ray.ObjectRef):
            observations_ref = observations
            observations = ray.get(observations_ref)
        else:
            observations_ref = ray.put(observations)

        if isinstance(ephemeris, ray.ObjectRef):
            ephemeris_ref = ephemeris
        else:
            ephemeris_ref = ray.put(ephemeris)

        ranged_detections_cartesian_ref = ray.put(ranged_detections_cartesian)

        # Get state IDs
        state_ids = observations.state_id.unique().sort()
        futures = []
        for state_id in state_ids:
            futures.append(
                range_and_transform_remote.remote(
                    ranged_detections_cartesian_ref,
                    observations_ref,
                    ephemeris_ref,
                    state_id,
                )
            )

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            transformed_detection_list.append(ray.get(finished[0]))

    else:
        # Get state IDs
        state_ids = observations.state_id.unique().sort()
        for state_id in state_ids:
            mask = pc.equal(state_id, observations.state_id)
            transformed_detection_list.append(
                range_and_transform_worker(
                    ranged_detections_cartesian.apply_mask(mask),
                    observations.select("state_id", state_id),
                    ephemeris.select("id", state_id),
                    state_id,
                )
            )

    transformed_detections = qv.concatenate(transformed_detection_list)
    transformed_detections = transformed_detections.sort_by(by=["state_id"])

    time_end = time.perf_counter()
    logger.info(f"Transformed {len(transformed_detections)} observations.")
    logger.info(
        f"Range and transform completed in {time_end - time_start:.3f} seconds."
    )
    return transformed_detections


def _observations_to_observations_df(observations: Observations) -> pd.DataFrame:
    """
    Convert THOR observations (v2.0) to the older format used by the rest of the
    pipeline. This will eventually be removed once the rest of the pipeline is
    updated to use the new format.

    Parameters
    ----------
    observations : `~thor.observations.observations.Observations`
        Observations to convert.

    Returns
    -------
    observations_df : `~pandas.DataFrame`
        Observations in the old format.
    """
    observations_df = observations.to_dataframe()
    observations_df.rename(
        columns={
            "detections.id": "obs_id",
            "detections.ra": "RA_deg",
            "detections.dec": "Dec_deg",
            "detections.ra_sigma": "RA_sigma_deg",
            "detections.dec_sigma": "Dec_sigma_deg",
            "detections.mag": "mag",
            "detections.mag_sigma": "mag_sigma",
        },
        inplace=True,
    )
    observations_df["mjd_utc"] = (
        observations.detections.time.rescale("utc").mjd().to_numpy(zero_copy_only=False)
    )
    return observations_df


def _observers_with_states_to_observers_df(
    observers: ObserversWithStates,
) -> pd.DataFrame:
    """
    Convert THOR observers (v2.0) to the older format used by the rest of the
    pipeline. This will eventually be removed once the rest of the pipeline is
    updated to use the new format.

    Parameters
    ----------
    observers : `~adam_core.observers.observers.Observers`
        Observers to convert to a dataframe.

    Returns
    -------
    observers_df : `~pandas.DataFrame`
        Observers in the old format.
    """
    observers_df = observers.to_dataframe()
    observers_df.rename(
        columns={
            "observers.coordinates.x": "obs_x",
            "observers.coordinates.y": "obs_y",
            "observers.coordinates.z": "obs_z",
            "observers.coordinates.vx": "obs_vx",
            "observers.coordinates.vy": "obs_vy",
            "observers.coordinates.vz": "obs_vz",
        },
        inplace=True,
    )
    return observers_df


def _transformed_detections_to_transformed_detections_df(
    transformed_detections: TransformedDetections,
) -> pd.DataFrame:
    """
    Convert THOR transformed detections (v2.0) to the older format used by the
    rest of the pipeline. This will eventually be removed once the rest of the
    pipeline is updated to use the new format.

    Parameters
    ----------
    transformed_detections : `~thor.main.TransformedDetections`
        Transformed detections to convert to a dataframe.

    Returns
    -------
    transformed_detections_df : `~pandas.DataFrame`
        Transformed detections in the old format.
    """
    transformed_detections_df = transformed_detections.to_dataframe()
    transformed_detections_df.rename(
        columns={
            "id": "obs_id",
            "coordinates.theta_x": "theta_x_deg",
            "coordinates.theta_y": "theta_y_deg",
        },
        inplace=True,
    )
    return transformed_detections_df


def link_test_orbit(
    test_orbit: TestOrbit,
    observations: Observations,
    filters: Optional[List[ObservationFilter]] = None,
    config: Optional[Config] = None,
) -> Iterator[Any]:
    """
    Run THOR for a single test orbit on the given observations. This function will yield
    results at each stage of the pipeline.
        1. transformed_detections
        2. clusters, cluster_members
        3. iod_orbits, iod_orbit_members
        4. od_orbits, od_orbit_members
        5. recovered_orbits, recovered_orbit_members

    Parameters
    ----------
    test_orbit : `~thor.orbit.TestOrbit`
        Test orbit to use to gather and transform observations.
    observations : `~thor.observations.observations.Observations`
        Observations from which range and transform the detections.
    filters : list of `~thor.observations.filters.ObservationFilter`, optional
        List of filters to apply to the observations before running THOR.
    propagator : `~adam_core.propagator.propagator.Propagator`
        Propagator to use to propagate the test orbit and generate
        ephemerides.
    max_processes : int, optional
        Maximum number of processes to use for parallelization.
    """
    time_start = time.perf_counter()
    logger.info("Running test orbit...")

    if config is None:
        config = Config()

    if config.propagator == "PYOORB":
        propagator = PYOORB()
    else:
        raise ValueError(f"Unknown propagator: {config.propagator}")

    use_ray = False
    if config.max_processes is None or config.max_processes > 1:
        # Initialize ray
        if not ray.is_initialized():
            logger.debug(
                f"Ray is not initialized. Initializing with {config.max_processes}..."
            )
            ray.init(num_cpus=config.max_processes)

        if not isinstance(observations, ray.ObjectRef):
            observations = ray.put(observations)

        use_ray = True

    # Apply filters to the observations
    filtered_observations = observations
    if filters is None:
        # By default we always filter by radius from the predicted position of the test orbit
        filters = [TestOrbitRadiusObservationFilter(radius=config.cell_radius)]
    for filter_i in filters:
        filtered_observations = filter_i.apply(
            filtered_observations, test_orbit, max_processes=config.max_processes
        )

    # Defragment the observations
    filtered_observations = qv.defragment(filtered_observations)

    # Observations are no longer needed, so we can delete them
    del observations

    if use_ray:
        filtered_observations = ray.put(filtered_observations)

    # Range and transform the observations
    transformed_detections = range_and_transform(
        test_orbit,
        filtered_observations,
        propagator=propagator,
        max_processes=config.max_processes,
    )
    yield transformed_detections

    # TODO: ray support for the rest of the pipeline has not yet been implemented
    # so we will convert the ray objects to regular objects for now
    if use_ray:
        filtered_observations = ray.get(filtered_observations)

    # Convert quivr tables to dataframes used by the rest of the pipeline
    observations_df = _observations_to_observations_df(filtered_observations)
    observers_df = _observers_with_states_to_observers_df(
        filtered_observations.get_observers()
    )
    transformed_detections_df = _transformed_detections_to_transformed_detections_df(
        transformed_detections
    )

    # Merge dataframes together
    observations_df = observations_df.merge(observers_df, on="state_id")
    transformed_detections_df = transformed_detections_df.merge(
        observations_df[["obs_id", "mjd_utc", "observatory_code"]], on="obs_id"
    )

    # Run clustering
    clusters, cluster_members = clusterAndLink(
        transformed_detections_df,
        vx_range=[config.vx_min, config.vx_max],
        vy_range=[config.vy_min, config.vy_max],
        vx_bins=config.vx_bins,
        vy_bins=config.vy_bins,
        vx_values=config.vx_values,
        vy_values=config.vy_values,
        eps=config.cluster_radius,
        min_obs=config.cluster_min_obs,
        min_arc_length=config.cluster_min_arc_length,
        alg=config.cluster_algorithm,
        num_jobs=config.max_processes,
        parallel_backend=config.parallel_backend,
    )
    yield clusters, cluster_members

    # Run initial orbit determination
    iod_orbits, iod_orbit_members = initialOrbitDetermination(
        observations_df,
        cluster_members,
        min_obs=config.iod_min_obs,
        min_arc_length=config.iod_min_arc_length,
        contamination_percentage=config.iod_contamination_percentage,
        rchi2_threshold=config.iod_rchi2_threshold,
        observation_selection_method=config.iod_observation_selection_method,
        backend=config.propagator,
        backend_kwargs={},
        chunk_size=config.iod_chunk_size,
        num_jobs=config.max_processes,
        parallel_backend=config.parallel_backend,
        # TODO: investigate whether these should be configurable
        iterate=False,
        identify_subsets=False,
        light_time=True,
        linkage_id_col="cluster_id",
    )
    yield iod_orbits, iod_orbit_members

    # Run differential correction
    od_orbits, od_orbit_members = differentialCorrection(
        iod_orbits,
        iod_orbit_members,
        observations_df,
        min_obs=config.od_min_obs,
        min_arc_length=config.od_min_arc_length,
        contamination_percentage=config.od_contamination_percentage,
        rchi2_threshold=config.od_rchi2_threshold,
        delta=config.od_delta,
        max_iter=config.od_max_iter,
        backend=config.propagator,
        backend_kwargs={},
        chunk_size=config.od_chunk_size,
        num_jobs=config.max_processes,
        parallel_backend=config.parallel_backend,
        # TODO: investigate whether these should be configurable
        method="central",
        fit_epoch=False,
        test_orbit=None,
    )
    yield od_orbits, od_orbit_members

    # Run arc extension
    recovered_orbits, recovered_orbit_members = mergeAndExtendOrbits(
        od_orbits,
        od_orbit_members,
        observations_df,
        min_obs=config.arc_extension_min_obs,
        min_arc_length=config.arc_extension_min_arc_length,
        contamination_percentage=config.arc_extension_contamination_percentage,
        rchi2_threshold=config.arc_extension_rchi2_threshold,
        eps=config.arc_extension_radius,
        delta=config.od_delta,
        max_iter=config.od_max_iter,
        backend=config.propagator,
        backend_kwargs={},
        orbits_chunk_size=config.arc_extension_chunk_size,
        num_jobs=config.max_processes,
        parallel_backend=config.parallel_backend,
        # TODO: investigate whether these should be configurable
        method="central",
        fit_epoch=False,
        observations_chunk_size=100000,
    )
    yield recovered_orbits, recovered_orbit_members

    time_end = time.perf_counter()
    logger.info(f"Test orbit completed in {time_end-time_start:.3f} seconds.")
