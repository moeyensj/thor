from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.coordinates import Times
from adam_core.observations import Exposures, PointSourceDetections
from adam_core.observers import Observers


class Observations(qv.Table):
    """
    The Observations table stored invidual point source detections and a state ID for each unique
    combination of detection time and observatory code. The state ID is used as reference to a specific
    observing geometry.

    Columns
    -------
    detections : `~adam_core.observations.detections.PointSourceDetections`
        A table of point source detections.
    observatory_code : `~qv.StringColumn`
        The observatory code for each detection.
    state_id : `~qv.Int64Column`
        The state ID for each detection.
    """

    detections = PointSourceDetections.as_column()
    observatory_code = qv.StringColumn()
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
            detections = detections.set_column("time", detections.time.to_scale("utc"))

        # Flatten the detections table (i.e. remove the nested columns). Unfortunately joins on tables
        # with nested columns are not all supported in pyarrow
        detections_flattened = pa.table(
            [
                detections.id,
                detections.exposure_id,
                detections.time.jd1,
                detections.time.jd2,
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
                "jd1",
                "jd2",
                "ra",
                "ra_sigma",
                "dec",
                "dec_sigma",
                "mag",
                "mag_sigma",
            ],
        )

        # Extract the exposure IDs and the observatory codes from the exposures table
        exposure_obscodes = pa.table(
            [exposures.id, exposures.observatory_code],
            names=["exposure_id", "observatory_code"],
        )

        # Join the detection times and the exposure IDs so that each detection has an observatory code
        obscode_times = detections_flattened.join(exposure_obscodes, ["exposure_id"])

        # Group the detections by detection time and observatory code
        unique_obscode_times = obscode_times.group_by(
            ["jd1", "jd2", "observatory_code"]
        ).aggregate([])

        # Now sort the unique detections by time and observatory code
        unique_obscode_times = unique_obscode_times.sort_by(
            [
                ("jd1", "ascending"),
                ("jd2", "ascending"),
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
            unique_obscode_times, ["observatory_code", "jd1", "jd2"]
        )

        # Now sort the detections one final time by state ID
        detections_with_states = detections_with_states.sort_by(
            [("state_id", "ascending")]
        )

        return cls.from_kwargs(
            detections=PointSourceDetections.from_kwargs(
                id=detections_with_states["id"],
                exposure_id=detections_with_states["exposure_id"],
                ra=detections_with_states["ra"],
                ra_sigma=detections_with_states["ra_sigma"],
                dec=detections_with_states["dec"],
                dec_sigma=detections_with_states["dec_sigma"],
                mag=detections_with_states["mag"],
                mag_sigma=detections_with_states["mag_sigma"],
                time=Times.from_kwargs(
                    jd1=detections_with_states["jd1"],
                    jd2=detections_with_states["jd2"],
                    scale="utc",
                ),
            ),
            observatory_code=detections_with_states["observatory_code"],
            state_id=detections_with_states["state_id"],
        )

    def get_observers(self):
        """
        Get the observers table for these observations. The observers table
        will have one row for each unique state ID in the observations table.

        Returns
        -------
        observers : `~adam_core.observers.observers.Observers`
            The observers table for these observations.
        """
        observer_dfs = []
        for code, observations_i in self.group_by_observatory_code():
            unique_times = observations_i.detections.time.unique()
            unique_states = observations_i.state_id.unique()

            # Get observers table for this observatory
            observers_i = Observers.from_code(code, unique_times.to_astropy())

            # Here we fall back on pandas to do some book-keeping to get the state IDs for each
            # observer state
            observers_i_df = observers_i.to_dataframe()
            observers_i_df["state_id"] = unique_states.to_numpy()
            observer_dfs.append(observers_i_df)

        # Merge the observers tables for each observatory and sort by state ID
        observer_df = pd.concat(observer_dfs)
        observer_df.sort_values(by=["state_id"], inplace=True, ignore_index=True)
        return Observers.from_dataframe(observer_df)

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
        return self.apply_mask(pc.equal(self.detections.exposure_id, exposure_id))

    def group_by_exposure(self) -> Iterator[Tuple[str, "Observations"]]:
        """
        Group observations by exposure ID. Note that if this exposure has observations
        with varying observation times, then the exposure will have multiple unique state IDs.

        Returns
        -------
        observations : Iterator[`~thor.observations.observations.Observations`]
            Observations belonging to individual exposures.
        """
        exposure_ids = self.detections.exposure_id
        for exposure_id in exposure_ids.unique():
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
        return self.apply_mask(pc.equal(self.observatory_code, observatory_code))

    def group_by_observatory_code(self) -> Iterator[Tuple[str, "Observations"]]:
        """
        Group observations by observatory code.

        Returns
        -------
        observations : Iterator[`~thor.observations.observations.Observations`]
            Observations belonging to individual observatories.
        """
        observatory_codes = self.observatory_code
        for observatory_code in observatory_codes.unique():
            yield observatory_code.as_py(), self.select_observatory_code(
                observatory_code
            )
