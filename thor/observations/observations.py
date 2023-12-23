from typing import Iterator, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.coordinates import CoordinateCovariances, Origin, SphericalCoordinates
from adam_core.observations import Exposures, PointSourceDetections
from adam_core.observers import Observers
from adam_core.time import Timestamp

from .photometry import Photometry
from .states import calculate_state_ids

__all__ = [
    "InputObservations",
    "Observations",
]


class ObserversWithStates(qv.Table):
    state_id = qv.Int64Column()
    observers = Observers.as_column()


class InputObservations(qv.Table):
    id = qv.StringColumn()
    exposure_id = qv.StringColumn()
    time = Timestamp.as_column()
    ra = qv.Float64Column()
    dec = qv.Float64Column()
    ra_sigma = qv.Float64Column(nullable=True)
    dec_sigma = qv.Float64Column(nullable=True)
    ra_dec_cov = qv.Float64Column(nullable=True)
    mag = qv.Float64Column()
    mag_sigma = qv.Float64Column(nullable=True)
    filter = qv.StringColumn()
    observatory_code = qv.StringColumn()


class Observations(qv.Table):
    id = qv.StringColumn()
    exposure_id = qv.StringColumn()
    coordinates = SphericalCoordinates.as_column()
    photometry = Photometry.as_column()
    state_id = qv.Int64Column()

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
        # Sort the observations by time and observatory code
        observations_sorted = observations.sort_by(
            ["time.days", "time.nanos", "observatory_code"]
        )

        # If the times are not in UTC, convert them to UTC
        if observations_sorted.time.scale != "utc":
            observations_sorted = observations_sorted.set_column(
                "time", observations_sorted.time.rescale("utc")
            )

        # Extract the sigma and covariance values for RA and Dec
        ra_sigma = observations_sorted.ra_sigma.to_numpy(zero_copy_only=False)
        dec_sigma = observations_sorted.dec_sigma.to_numpy(zero_copy_only=False)
        ra_dec_cov = observations_sorted.ra_dec_cov.to_numpy(zero_copy_only=False)

        # Create the covariance matrices
        covariance_matrices = np.full((len(observations_sorted), 6, 6), np.nan)
        covariance_matrices[:, 1, 1] = ra_sigma**2
        covariance_matrices[:, 2, 2] = dec_sigma**2
        covariance_matrices[:, 1, 2] = ra_dec_cov
        covariance_matrices[:, 2, 1] = ra_dec_cov
        covariances = CoordinateCovariances.from_matrix(covariance_matrices)

        # Create the coordinates table
        coords = SphericalCoordinates.from_kwargs(
            lon=observations_sorted.ra,
            lat=observations_sorted.dec,
            time=observations_sorted.time,
            covariance=covariances,
            origin=Origin.from_kwargs(code=observations_sorted.observatory_code),
            frame="equatorial",
        )

        # Create the photometry table
        photometry = Photometry.from_kwargs(
            filter=observations_sorted.filter,
            mag=observations_sorted.mag,
            mag_sigma=observations_sorted.mag_sigma,
        )

        return cls.from_kwargs(
            id=observations_sorted.id,
            exposure_id=observations_sorted.exposure_id,
            coordinates=coords,
            photometry=photometry,
            state_id=calculate_state_ids(coords),
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
            state_id=calculate_state_ids(coordinates),
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
            state_ids = observations_i.state_id.unique().sort()

            # Get observers table for this observatory
            observers_i = Observers.from_code(code, unique_times)

            # Create the observers with states table
            observers_with_states_i = ObserversWithStates.from_kwargs(
                state_id=state_ids,
                observers=observers_i,
            )

            observers_with_states = qv.concatenate(
                [observers_with_states, observers_with_states_i]
            )
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
            yield observatory_code.as_py(), self.select_observatory_code(
                observatory_code
            )
