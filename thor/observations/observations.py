from typing import Iterator, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
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
    ):
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

        # It would be cool if we could do the following with functionality in quivr rather than expensive
        # copies to pandas and back to arrow. A proposed interface is in the comments.

        # Lets set the time scale to UTC
        if detections.time.scale != "utc":
            detections = detections.set_column("time", detections.time.to_scale("utc"))

        ### 1. Create a table that attaches observatory codes to detections
        # Quivr interface (join -- create composite table schema from two tables)
        # times_obscodes = detections.join(exposures, on="exposure_id", columns=["exposure_id", "observatory_code"])

        # Create table of detections IDs, exposure IDs, and times
        detections_times = pa.table(
            [detections.id, detections.exposure_id, detections.time.mjd()],
            names=["obs_id", "exposure_id", "time_mjd"],
        )

        # Create table of exposure IDs and observatory codes
        exposure_obscodes = pa.table(
            [exposures.id, exposures.observatory_code],
            names=["exposure_id", "observatory_code"],
        )

        # Merge the two tables so that each detection has its corresponding observatory code
        times_obscodes = pd.merge(
            detections_times.to_pandas(),
            exposure_obscodes.to_pandas(),
            on="exposure_id",
        )

        ### 2. Get unique observatory codes and times
        # Quivr interface (drop duplicates -- drop duplicate rows optionally considering only a subset of columns)
        # unique_times_obscodes = times_obscodes.drop_duplicates(subset=["time", "observatory_code"])

        # Now lets get the unique times and observatory codes (these are the unique states at which we will need
        # to get the observatory coordinates and later test orbit coordinates)
        unique_time_obscodes = times_obscodes[
            ["time_mjd", "observatory_code"]
        ].drop_duplicates(subset=["time_mjd", "observatory_code"])
        unique_time_obscodes.reset_index(drop=True, inplace=True)

        ### 3. Calculate a unique state ID for each unique time and observatory code
        # Quivr interface (add column -- add a column to a table)
        # If add column is un-quivr like then I'd create a temporary table here
        # unique_time_obscodes = unique_time_obscodes.add_column("state_id", qv.Int64Column(pa.array()))
        unique_time_obscodes.insert(0, "state_id", unique_time_obscodes.index.values)

        ### 4. Merge the unique state IDs back into the times_obscodes table
        # Quivr interface (join -- create composite table schema from two tables)
        # times_obscodes = times_obscodes.join(unique_time_obscodes, on=["time_mjd", "observatory_code"])

        # Merge the unique state IDs back into the times_obscodes table
        times_obscodes = pd.merge(
            times_obscodes, unique_time_obscodes, on=["time_mjd", "observatory_code"]
        )

        ### 5. Merge the detections table with the times_obscodes table to get the state IDs for each detection
        # Quivr interface (join -- create composite table schema from two tables)
        # detections_states = detections.join(times_obscodes, on=["id", "exposure_id"], columns=["state_id"])

        # Now merge the detections table with the times_obscodes table to get the state IDs for each detection
        detections_states = pd.merge(
            detections.to_dataframe(),
            times_obscodes[["obs_id", "exposure_id", "state_id", "observatory_code"]],
            left_on=["id", "exposure_id"],
            right_on=["obs_id", "exposure_id"],
        )

        ### 6. Sort the detections by time and observatory code
        # Quivr interface (sort -- sort a table by one or more columns)
        # detections_states = detections_states.sort(by=["time_mjd", "observatory_code"], ascending=[True, True])
        detections_states.sort_values(
            by=["time.jd1", "time.jd2", "observatory_code"],
            inplace=True,
            ignore_index=True,
        )

        # In summary, with the proposed quivr interface, this function would look like:
        # times_obscodes = detections.join(exposures, on="exposure_id", columns=["exposure_id", "observatory_code"])
        # unique_times_obscodes = times_obscodes.drop_duplicates(subset=["time", "observatory_code"])
        # unique_times_obscodes = unique_times_obscodes.add_column("state_id", qv.Int64Column(pa.array()))
        # times_obscodes = times_obscodes.join(unique_times_obscodes, on=["time", "observatory_code"])
        # detections_states = detections.join(times_obscodes, on=["id", "exposure_id"], columns=["state_id"])
        # detections_states = detections_states.sort(by=["time", "observatory_code"], ascending=[True, True])
        return cls.from_kwargs(
            detections=PointSourceDetections.from_flat_dataframe(detections_states),
            observatory_code=detections_states["observatory_code"],
            state_id=detections_states["state_id"],
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
