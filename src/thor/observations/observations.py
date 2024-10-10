import multiprocessing as mp
import pathlib
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.coordinates import CoordinateCovariances, Origin, SphericalCoordinates
from adam_core.observations import Exposures, PointSourceDetections
from adam_core.observers import Observers
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp

from thor.config import Config

from .photometry import Photometry
from .states import calculate_state_id_hashes

__all__ = [
    "InputObservations",
    "Observations",
    "convert_input_observations_to_observations",
    "input_observations_to_observations_worker",
]


import logging

logger = logging.getLogger(__name__)


def observations_iterator(
    observations: Union["Observations", str],
    chunk_size: int = 1_000_000,
) -> Iterator["Observations"]:
    """
    Create an iterator that yields chunks of observations from a table or parquet folder/file

    Parameters
    ----------
    observations : `~Observations` or str
        Either a table or a path to a file containing a table of observations.

    Yields
    ------
    observations_chunk : `Observations`
        A chunk of observations.
    """
    if isinstance(observations, str):

        # Discover if it's a single file or a folder using pathlib
        parquet_paths = []
        path = pathlib.Path(observations)
        if path.is_dir():
            # grab all parquet files from the folder
            # and its subdirectories
            for file in path.glob("**/*.parquet"):
                parquet_paths.append(str(file))
        else:
            parquet_paths.append(observations)

        for parquet_path in parquet_paths:
            parquet_file = pa.parquet.ParquetFile(parquet_path, memory_map=True)
            logger.debug(f"Starting to read {parquet_file}")
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                table = Observations.from_pyarrow(pa.Table.from_batches([batch]))

                # Reading the table as above will cause the time scale attribute to
                # be read in as the default 'tai' rather than 'utc'
                if table.coordinates.time.scale != "utc":
                    table = table.set_column(
                        "coordinates.time",
                        Timestamp.from_kwargs(
                            days=table.coordinates.time.days,
                            nanos=table.coordinates.time.nanos,
                            scale="utc",
                        ),
                    )

                # Similarly, the coordinates frame attribute will be read in as the
                # default 'unspecified' rather than 'equatorial'
                if table.coordinates.frame != "equatorial":
                    table = table.set_column(
                        "coordinates",
                        SphericalCoordinates.from_kwargs(
                            rho=table.coordinates.rho,
                            lon=table.coordinates.lon,
                            lat=table.coordinates.lat,
                            vrho=table.coordinates.vrho,
                            vlon=table.coordinates.vlon,
                            vlat=table.coordinates.vlat,
                            time=table.coordinates.time,
                            covariance=table.coordinates.covariance,
                            origin=table.coordinates.origin,
                            frame="equatorial",
                        ),
                    )

                yield table

    else:
        offset = 0
        while offset < len(observations):
            yield observations[offset : offset + chunk_size]
            offset += chunk_size


def _input_observations_iterator(
    input_observations: Union["InputObservations", str],
    chunk_size: int = 1_000_000,
) -> Iterator["InputObservations"]:
    """
    Create an iterator that yields chunks of InputObservations from a table or parquet folder/file

    Parameters
    ----------
    input_observations : `~InputObservations` or str
        Either a table or a path to a file containing a table of input observations.

    Yields
    ------
    input_observations_chunk : `InputObservatio`
        A chunk of input observations.
    """
    if isinstance(input_observations, str):

        # Discover if it's a single file or a folder using pathlib
        parquet_paths = []
        path = pathlib.Path(input_observations)
        if path.is_dir():
            for file in path.glob("**/*.parquet"):
                parquet_paths.append(str(file))
        else:
            parquet_paths.append(input_observations)

        for parquet_path in parquet_paths:
            parquet_file = pa.parquet.ParquetFile(parquet_path, memory_map=True)
            logger.debug(f"Starting to read {parquet_file}")
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                table = InputObservations.from_pyarrow(pa.Table.from_batches([batch]))
            ## TODO: This should be happening some other way
            time = Timestamp.from_kwargs(days=table.time.days, nanos=table.time.nanos, scale="utc")
            table = table.set_column("time", time)
            yield table

    else:
        offset = 0
        while offset < len(input_observations):
            yield input_observations[offset : offset + chunk_size]
            offset += chunk_size


def input_observations_to_observations_worker(
    input_observations_chunk: "InputObservations",
) -> "Observations":
    """
    Convert a chunk of input observations to observations.

    Parameters
    ----------
    input_observations_chunk : `~InputObservations`
        A chunk of input observations.

    Returns
    -------
    observations_chunk : `~Observations`
        A chunk of observations.
    """
    return Observations.from_input_observations(input_observations_chunk)


input_observations_to_observations_worker_remote = ray.remote(input_observations_to_observations_worker)


def _process_next_future_result(
    futures: list,
    output_observations: "Observations",
    output_writer: pa.parquet.ParquetWriter,
):
    finished, futures = ray.wait(futures, num_returns=1)
    observations_chunk = ray.get(finished[0])
    if output_writer is not None:
        output_writer.write_table(observations_chunk.table)
    else:
        output_observations = qv.concatenate([output_observations, observations_chunk])
        if output_observations.fragmented():
            output_observations = qv.defragment(output_observations)
    return futures, output_observations


def convert_input_observations_to_observations(
    input_observations: Union["InputObservations", str],
    config: Config,
    output_path: Optional[str] = None,
) -> Union["Observations", str]:
    """
    Converts input observations to observations, optionally reading and writing from files.

    Use files as input / output when they are exceedingly large.
    """
    input_iterator = _input_observations_iterator(input_observations)

    output_observations = Observations.empty()

    output_writer = None
    if output_path is not None:
        output_writer = pa.parquet.ParquetWriter(output_path, Observations.schema)

    if config.max_processes is None:
        max_processes = mp.cpu_count()
    else:
        max_processes = config.max_processes

    use_ray = initialize_use_ray(num_cpus=max_processes)
    if use_ray:
        futures: List[ray.ObjectRef] = []
        for input_observation_chunk in input_iterator:
            if len(futures) > max_processes * 1.5:
                futures, output_observations = _process_next_future_result(
                    futures, output_observations, output_writer
                )

            futures.append(input_observations_to_observations_worker_remote.remote(input_observation_chunk))

        while futures:
            futures, output_observations = _process_next_future_result(
                futures, output_observations, output_writer
            )
    else:
        for input_observation_chunk in input_iterator:
            observations_chunk = input_observations_to_observations_worker(input_observation_chunk)
            if output_writer is not None:
                output_writer.write_table(observations_chunk.table)
            else:
                output_observations = qv.concatenate([output_observations, observations_chunk])
                if output_observations.fragmented():
                    output_observations = qv.defragment(output_observations)

    if output_writer is not None and output_path is not None:
        output_writer.close()
        return output_path

    return output_observations


class ObserversWithStates(qv.Table):
    state_id = qv.LargeStringColumn()
    observers = Observers.as_column()


class InputObservations(qv.Table):
    id = qv.LargeStringColumn()
    exposure_id = qv.LargeStringColumn()
    time = Timestamp.as_column()
    ra = qv.Float64Column()
    dec = qv.Float64Column()
    ra_sigma = qv.Float64Column(nullable=True)
    dec_sigma = qv.Float64Column(nullable=True)
    ra_dec_cov = qv.Float64Column(nullable=True)
    mag = qv.Float64Column()
    mag_sigma = qv.Float64Column(nullable=True)
    filter = qv.LargeStringColumn()
    observatory_code = qv.LargeStringColumn()


class Observations(qv.Table):
    id = qv.LargeStringColumn()
    exposure_id = qv.LargeStringColumn()
    coordinates = SphericalCoordinates.as_column()
    photometry = Photometry.as_column()
    state_id = qv.LargeStringColumn()

    @classmethod
    def from_input_observations(cls, observations: InputObservations) -> "Observations":
        """
        Create a THOR observations table from an InputObservations table. The InputObservations table
        are sorted by ascending time and observatory code.

        Parameters
        ----------
        observations : `~InputObservations`
            A table of input observations.

        Returns
        -------
        observations : `~Observations`
            A table of THOR observations.
        """
        assert observations.time.scale == "utc", "Input observations must be in UTC"

        # Extract the sigma and covariance values for RA and Dec
        ra_sigma = observations.ra_sigma.to_numpy(zero_copy_only=False)
        dec_sigma = observations.dec_sigma.to_numpy(zero_copy_only=False)
        ra_dec_cov = observations.ra_dec_cov.to_numpy(zero_copy_only=False)

        # Create the covariance matrices
        covariance_matrices = np.full((len(observations), 6, 6), np.nan)
        covariance_matrices[:, 1, 1] = ra_sigma**2
        covariance_matrices[:, 2, 2] = dec_sigma**2
        covariance_matrices[:, 1, 2] = ra_dec_cov
        covariance_matrices[:, 2, 1] = ra_dec_cov
        covariances = CoordinateCovariances.from_matrix(covariance_matrices)

        # Create the coordinates table
        coords = SphericalCoordinates.from_kwargs(
            lon=observations.ra,
            lat=observations.dec,
            time=observations.time,
            covariance=covariances,
            origin=Origin.from_kwargs(code=observations.observatory_code),
            frame="equatorial",
        )

        # Create the photometry table
        photometry = Photometry.from_kwargs(
            filter=observations.filter,
            mag=observations.mag,
            mag_sigma=observations.mag_sigma,
        )

        return cls.from_kwargs(
            id=observations.id,
            exposure_id=observations.exposure_id,
            coordinates=coords,
            photometry=photometry,
            state_id=calculate_state_id_hashes(coords),
        )

    @classmethod
    def from_detections_and_exposures(
        cls, detections: PointSourceDetections, exposures: Exposures
    ) -> "Observations":
        """
        Create a THOR observations table from a PointSourceDetections table and an Exposures table.

        Note: This function will sort the detections by time and observatory code.

        Parameters
        ----------
        detections : `~adam_core.observations.detections.PointSourceDetections`
            A table of point source detections.
        exposures : `~adam_core.observations.exposures.Exposures`
            A table of exposures.

        Returns
        -------
        observations : `~Observations`
            A table of observations.
        """
        # TODO: One thing we could try in the future is truncating observation times to ~1ms and using those to group
        # into indvidual states (i.e. if two observations are within 1ms of each other, they are in the same state). Again,
        # this only matters for those detections that have times that differ from the midpoint time of the exposure (LSST)
        # If the detection times are not in UTC, convert them to UTC
        if detections.time.scale != "utc":
            detections = detections.set_column("time", detections.time.rescale("utc"))

        # Join the detections and exposures tables
        detections_flattened = detections.flattened_table()
        exposures_flattened = exposures.flattened_table()
        detections_exposures = detections_flattened.join(
            exposures_flattened, ["exposure_id"], right_keys=["id"]
        )

        # Combine chunks of joined table
        detections_exposures = detections_exposures.combine_chunks()

        # Create covariance matrices
        sigmas = np.zeros((len(detections_exposures), 6))
        sigmas[:, 1] = detections_exposures["ra_sigma"].to_numpy(zero_copy_only=False)
        sigmas[:, 2] = detections_exposures["dec_sigma"].to_numpy(zero_copy_only=False)
        covariances = CoordinateCovariances.from_sigmas(sigmas)

        # Create the coordinates table
        coordinates = SphericalCoordinates.from_kwargs(
            lon=detections_exposures["ra"],
            lat=detections_exposures["dec"],
            time=Timestamp.from_kwargs(
                days=detections_exposures["time.days"],
                nanos=detections_exposures["time.nanos"],
                scale="utc",
            ),
            covariance=covariances,
            origin=Origin.from_kwargs(code=detections_exposures["observatory_code"]),
            frame="equatorial",
        )

        # Create the photometry table
        photometry = Photometry.from_kwargs(
            filter=detections_exposures["filter"],
            mag=detections_exposures["mag"],
            mag_sigma=detections_exposures["mag_sigma"],
        )

        return cls.from_kwargs(
            id=detections_exposures["id"],
            exposure_id=detections_exposures["exposure_id"],
            coordinates=coordinates,
            photometry=photometry,
            state_id=calculate_state_id_hashes(coordinates),
        ).sort_by(
            [
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.origin.code",
            ]
        )

    def get_observers(self) -> ObserversWithStates:
        """
        Get the observers table for these observations. The observers table
        will have one row for each unique state ID in the observations table.

        Returns
        -------
        observers : `~thor.observations.observers.ObserversWithStates`
            The observers table for these observations with an added
            column representing the state ID.
        """
        observers_with_states = ObserversWithStates.empty()
        for code, observations_i in self.group_by_observatory_code():
            # Extract unique times and make sure they are sorted
            # by time in ascending order
            unique_times = observations_i.coordinates.time.unique()
            unique_times = unique_times.sort_by(["days", "nanos"])

            # States are defined by unique times and observatory codes and
            # are sorted by time in ascending order
            state_ids = observations_i.state_id.unique()

            # Get observers table for this observatory
            observers_i = Observers.from_code(code, unique_times)

            # Create the observers with states table
            observers_with_states_i = ObserversWithStates.from_kwargs(
                state_id=state_ids,
                observers=observers_i,
            )

            observers_with_states = qv.concatenate([observers_with_states, observers_with_states_i])
            if observers_with_states.fragmented():
                observers_with_states = qv.defragment(observers_with_states)

        return observers_with_states.sort_by("state_id")

    def select_exposure(self, exposure_id: int) -> "Observations":
        """
        Select observations from a single exposure.

        Parameters
        ----------
        exposure_id : int
            The exposure ID.

        Returns
        -------
        observations : `~Observations`
            Observations from the specified exposure.
        """
        return self.apply_mask(pc.equal(self.exposure_id, exposure_id))

    def group_by_exposure(self) -> Iterator[Tuple[str, "Observations"]]:
        """
        Group observations by exposure ID. Note that if this exposure has observations
        with varying observation times, then the exposure will have multiple unique state IDs.

        Returns
        -------
        observations : Iterator[`~thor.observations.observations.Observations`]
            Observations belonging to individual exposures.
        """
        exposure_ids = self.exposure_id
        for exposure_id in exposure_ids.unique().sort():
            yield exposure_id.as_py(), self.select_exposure(exposure_id)

    def select_observatory_code(self, observatory_code) -> "Observations":
        """
        Select observations from a single observatory.

        Parameters
        ----------
        observatory_code : str
            The observatory code.

        Returns
        -------
        observations : `~Observations`
            Observations from the specified observatory.
        """
        return self.apply_mask(pc.equal(self.coordinates.origin.code, observatory_code))

    def group_by_observatory_code(self) -> Iterator[Tuple[str, "Observations"]]:
        """
        Group observations by observatory code.

        Returns
        -------
        observations : Iterator[`~thor.observations.observations.Observations`]
            Observations belonging to individual observatories.
        """
        observatory_codes = self.coordinates.origin.code
        for observatory_code in observatory_codes.unique().sort():
            yield observatory_code.as_py(), self.select_observatory_code(observatory_code)
