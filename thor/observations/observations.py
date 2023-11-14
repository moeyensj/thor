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


class ObserversWithStates(qv.Table):
    state_id = qv.Int64Column()
    observers = Observers.as_column()


class Observations(qv.Table):
    """
    The Observations table stored invidual point source detections and a state ID for each unique
    combination of detection time and observatory code. The state ID is used as reference to a specific
    observing geometry.

    The recommended constructor to use is `~Observations.from_detections_and_exposures`, as this function
    will sort the detections by time and observatory code, and for each unique combination of the two
    assign a unique state ID. If not using this constructor, please ensure that the detections are sorted
    by time and observatory code and that each unique combination of time and observatory code has a unique
    state ID.
    """

    id = qv.StringColumn()
    exposure_id = qv.StringColumn()
    coordinates = SphericalCoordinates.as_column()
    photometry = Photometry.as_column()
    state_id = qv.Int64Column()

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

        # Flatten the detections table (i.e. remove the nested columns). Unfortunately joins on tables
        # with nested columns are not all supported in pyarrow
        detections_flattened = pa.table(
            [
                detections.id,
                detections.exposure_id,
                detections.time.days,
                detections.time.nanos,
                detections.ra,
                detections.ra_sigma,
                detections.dec,
                detections.dec_sigma,
                detections.mag,
                detections.mag_sigma,
            ],
            names=[
                "id",
                "exposure_id",
                "days",
                "nanos",
                "ra",
                "ra_sigma",
                "dec",
                "dec_sigma",
                "mag",
                "mag_sigma",
            ],
        )

        # Extract the exposure IDs and the observatory codes from the exposures table
        exposure_filters_obscodes = pa.table(
            [exposures.id, exposures.filter, exposures.observatory_code],
            names=["exposure_id", "filter", "observatory_code"],
        )

        # Join the detection times and the exposure IDs so that each detection has an observatory code
        obscode_times = detections_flattened.join(
            exposure_filters_obscodes, ["exposure_id"]
        )

        # Group the detections by the observatory code and the detection times and then grab the unique ones
        unique_obscode_times = obscode_times.group_by(
            ["days", "nanos", "observatory_code"]
        ).aggregate([])

        # Now sort the unique detections by the observatory code and the detection time
        unique_obscode_times = unique_obscode_times.sort_by(
            [
                ("days", "ascending"),
                ("nanos", "ascending"),
                ("observatory_code", "ascending"),
            ]
        )

        # For each unique detection time and observatory code assign a unique state ID
        unique_obscode_times = unique_obscode_times.add_column(
            0,
            pa.field("state_id", pa.int64()),
            pa.array(np.arange(0, len(unique_obscode_times))),
        )

        # Join the unique observatory code and detections back to the original detections
        detections_with_states = obscode_times.join(
            unique_obscode_times, ["days", "nanos", "observatory_code"]
        )

        # Now sort the detections one final time by state ID
        detections_with_states = detections_with_states.sort_by(
            [("state_id", "ascending")]
        )

        sigmas = np.zeros((len(detections_with_states), 6))
        sigmas[:, 1] = detections_with_states["ra_sigma"].to_numpy(zero_copy_only=False)
        sigmas[:, 2] = detections_with_states["dec_sigma"].to_numpy(
            zero_copy_only=False
        )

        return cls.from_kwargs(
            id=detections_with_states["id"],
            exposure_id=detections_with_states["exposure_id"],
            coordinates=SphericalCoordinates.from_kwargs(
                lon=detections_with_states["ra"],
                lat=detections_with_states["dec"],
                time=Timestamp.from_kwargs(
                    days=detections_with_states["days"],
                    nanos=detections_with_states["nanos"],
                    scale="utc",
                ),
                covariance=CoordinateCovariances.from_sigmas(sigmas),
                origin=Origin.from_kwargs(
                    code=detections_with_states["observatory_code"]
                ),
                frame="equatorial",
            ),
            photometry=Photometry.from_kwargs(
                filter=detections_with_states["filter"],
                mag=detections_with_states["mag"],
                mag_sigma=detections_with_states["mag_sigma"],
            ),
            state_id=detections_with_states["state_id"],
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
        observers_with_states = []
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
            observers_with_states.append(observers_with_states_i)

        observers = qv.concatenate(observers_with_states)
        return observers.sort_by("state_id")

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
