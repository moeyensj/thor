from typing import List, Literal, Optional, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.orbit_determination import FittedOrbitMembers, FittedOrbits

from ..utils.quivr import drop_duplicates

__all__ = [
    "assign_duplicate_observations",
    "drop_duplicate_orbits",
    "drop_outliers",
]


def assign_duplicate_observations(
    orbits: FittedOrbits, orbit_members: FittedOrbitMembers
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Assigns observations that have been assigned to multiple orbits to the orbit with the
    most observations, longest arc length, and lowest reduced chi2.

    Parameters
    ----------
    orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
        Fitted orbit members.

    Returns
    -------
    filtered : `~thor.orbit_determination.FittedOrbits`
        Fitted orbits with duplicate assignments removed.
    filtered_orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
        Fitted orbit members with duplicate assignments removed.
    """
    # Sorting by priority criteria
    orbits = orbits.sort_by(
        [
            ("num_obs", "descending"),
            ("arc_length", "descending"),
            ("reduced_chi2", "ascending"),
        ]
    )

    # Extracting unique observation IDs
    unique_obs_ids = pc.unique(orbit_members.column("obs_id"))

    # Dictionary to store the best orbit for each observation
    best_orbit_for_obs = {}

    # Iterate over each unique observation ID
    for obs_id in unique_obs_ids:
        # Filter orbit_members for the current observation ID
        mask = pc.equal(orbit_members.column("obs_id"), obs_id)
        current_obs_members = orbit_members.where(mask)

        # Extract orbit IDs that this observation belongs to
        obs_orbit_ids = current_obs_members.column("orbit_id")

        # Find the best orbit for this observation based on the criteria
        for sorted_orbit_id in orbits.column("orbit_id"):
            if pc.any(pc.is_in(sorted_orbit_id, value_set=obs_orbit_ids)).as_py():
                best_orbit_for_obs[obs_id.as_py()] = sorted_orbit_id.as_py()
                break

    # Iteratively update orbit_members to drop rows where obs_id is the same,
    # but orbit_id is not the best orbit_id for that observation
    for obs_id, best_orbit_id in best_orbit_for_obs.items():
        mask_to_remove = pc.and_(
            pc.equal(orbit_members.column("obs_id"), pa.scalar(obs_id)),
            pc.not_equal(orbit_members.column("orbit_id"), pa.scalar(best_orbit_id)),
        )
        orbit_members = orbit_members.apply_mask(pc.invert(mask_to_remove))

    # Filtering self based on the filtered orbit_members
    orbits_mask = pc.is_in(
        orbits.column("orbit_id"), value_set=orbit_members.column("orbit_id")
    )
    filtered_orbits = orbits.apply_mask(orbits_mask)

    return filtered_orbits, orbit_members


def drop_duplicate_orbits(
    orbits: FittedOrbits,
    orbit_members: FittedOrbitMembers,
    subset: Optional[List[str]] = None,
    keep: Literal["first", "last"] = "first",
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Drop duplicate orbits from the fitted orbits and remove
    the corresponding orbit members.

    Parameters
    ----------
    orbits : `~thor.orbit_determination.FittedOrbits`
        Fitted orbits.
    orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
        Fitted orbit members.
    subset : list of str, optional
        Subset of columns to consider when dropping duplicates. If not specified all the columns
        specifying unique state are used: time, x, y, z, vx, vy, vz.
    keep : {'first', 'last'}, default 'first'
        If there are duplicate rows then keep the first or last row.

    Returns
    -------
    filtered : `~thor.orbit_determination.FittedOrbits`
        Fitted orbits without duplicates.
    filtered_orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
        Fitted orbit members without duplicates.
    """
    if subset is None:
        subset = [
            "coordinates.time.days",
            "coordinates.time.nanos",
            "coordinates.x",
            "coordinates.y",
            "coordinates.z",
            "coordinates.vx",
            "coordinates.vy",
            "coordinates.vz",
        ]

    filtered = drop_duplicates(orbits, subset=subset, keep=keep)
    filtered_orbit_members = orbit_members.apply_mask(
        pc.is_in(orbit_members.orbit_id, filtered.orbit_id)
    )
    return filtered, filtered_orbit_members


def drop_outliers(orbit_members: FittedOrbitMembers) -> FittedOrbitMembers:
    """
    Drop outliers from the fitted orbit members.

    Parameters
    ----------
    orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
        Fitted orbit members.

    Returns
    -------
    fitted_orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
        Fitted orbit members without outliers.
    """
    filtered = orbit_members.apply_mask(pc.equal(orbit_members.outlier, False))
    if filtered.fragmented():
        filtered = qv.defragment(filtered)
    return filtered
