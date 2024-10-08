from typing import Tuple

import numpy as np
import pyarrow as pa
import quivr as qv

from ..observations import Observations

__all__ = [
    "sort_by_id_and_time",
]


def sort_by_id_and_time(
    linkages: qv.AnyTable,
    members: qv.AnyTable,
    observations: Observations,
    linkage_column: str,
) -> Tuple[qv.AnyTable, qv.AnyTable]:
    """
    Sort linkages and linkage members by linkage ID and observation time.

    Parameters
    ----------
    linkages : qv.AnyTable
        Linkages to sort.
    members : qv.AnyTable
        Linkage members to sort.
    observations : Observations
        Observations from which linkage members were generated. Observations
        are used to determine the observation time of each linkage member.
    linkage_column : str
        Column name in the linkage table to use for sorting. For clusters
        this is "cluster_id" and for orbits this is "orbit_id".

    Returns
    -------
    linkages : qv.AnyTable
        Sorted linkages.
    members : qv.AnyTable
        Sorted linkage members.
    """
    # Grab the linkage ID column from the linkages table and add an index column
    linkage_table = linkages.table.select([linkage_column])
    linkage_table = linkage_table.add_column(0, "index", pa.array(np.arange(0, len(linkage_table))))

    # Grab the linkage ID and observation ID columns from the linkage members table and add an index column
    members_table = members.table.select([linkage_column, "obs_id"])
    members_table = members_table.add_column(0, "index", pa.array(np.arange(0, len(members_table))))

    # Grab the observation ID, observation time columns and join with the linkage members table on the observation ID
    observation_times = observations.flattened_table().select(
        ["id", "coordinates.time.days", "coordinates.time.nanos"]
    )
    member_times = members_table.join(observation_times, keys=["obs_id"], right_keys=["id"])

    # Sort the reduced linkages table by linkage ID and the linkage member times table by linkage ID and observation time
    linkage_table = linkage_table.sort_by([(linkage_column, "ascending")])
    member_times = member_times.sort_by(
        [
            (linkage_column, "ascending"),
            ("coordinates.time.days", "ascending"),
            ("coordinates.time.nanos", "ascending"),
        ]
    )

    linkages = linkages.take(linkage_table["index"])
    members = members.take(member_times["index"])

    if linkages.fragmented():
        linkages = qv.defragment(linkages)
    if members.fragmented():
        members = qv.defragment(members)
    return linkages, members
