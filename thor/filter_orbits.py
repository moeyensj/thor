from typing import Tuple

import pandas as pd

from .utils import calcDeltas

__all__ = ["filterKnownOrbits", "filterOrbits"]


def filterKnownOrbits(
    orbits: pd.DataFrame,
    orbit_observations: pd.DataFrame,
    associations: pd.DataFrame,
    min_obs: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove all observations of unknown objects, keeping only observations of objects with
    a known association. If any orbits have fewer than min_obs observations after removing
    unknown observations then remove those orbits as well.

    This function will also set the provisional and permanent designation columns as required
    by the ADES file format.

    Parameters
    ----------
    orbits : `~pandas.DataFrame`
        DataFrame of orbits.
    orbit_observations : `~pandas.DataFrame`
        Dataframe of orbit observations with at least one column with the orbit ID ('orbit_id') and
        one column with the 'obs_id'
    associations : `~pandas.DataFrame`
        DataFrame of known associations, with one column of containing the observation ID ('obs_id')
        and another column containing the association ('obj_id'). Any unknown objects should have
        been assigned an unknown ID. See preprocessObservations.
    min_obs : int
        The minimum number of observations for an object to be considered as recovered.

    Returns
    -------
    known_orbits : `~pandas.DataFrame`
        Orbits of previously known objects.
    known_orbit_observations : `~pandas.DataFrame`
        Observations of previously known objects, the constituent observations
        to which the orbits were fit.
    """
    # Merge associations with orbit observations
    labeled_observations = orbit_observations.merge(
        associations[["obs_id", "obj_id"]], on="obs_id", how="left"
    )

    # Keep only observations of known objects
    labeled_observations = labeled_observations[
        ~labeled_observations["obj_id"].str.contains(UNKNOWN_ID_REGEX, regex=True)
    ]

    # Keep only known objects with at least min_obs observations
    occurences = labeled_observations["orbit_id"].value_counts()
    orbit_ids = occurences.index.values[occurences.values >= min_obs]

    # Filter input orbits
    orbits_mask = orbits["orbit_id"].isin(orbit_ids)
    orbit_observations_mask = labeled_observations["orbit_id"].isin(orbit_ids)
    known_orbits = orbits[orbits_mask].copy()
    known_orbit_observations = labeled_observations[orbit_observations_mask].copy()

    # Split into permanent and provisional designations
    if len(known_orbit_observations) > 0:
        known_orbit_observations.loc[:, "permID"] = ""
        known_orbit_observations.loc[:, "provID"] = ""
    else:
        known_orbit_observations["permID"] = ""
        known_orbit_observations["provID"] = ""

    # Process permanent IDs first
    # TODO : add an equivalent for Comets
    perm_ids = known_orbit_observations["obj_id"].str.isnumeric()
    known_orbit_observations.loc[perm_ids, "permID"] = known_orbit_observations[
        perm_ids
    ]["obj_id"].values

    # Identify provisional IDs next
    prov_ids = (~known_orbit_observations["obj_id"].str.isnumeric()) & (
        ~known_orbit_observations["obj_id"].str.contains(UNKNOWN_ID_REGEX, regex=True)
    )
    known_orbit_observations.loc[prov_ids, "provID"] = known_orbit_observations[
        prov_ids
    ]["obj_id"].values

    # Reorder the columns to put the labels toward the front
    cols = known_orbit_observations.columns
    first = ["orbit_id", "permID", "provID", "obj_id", "obs_id"]
    cols_ordered = first + cols[~cols.isin(first)].tolist()
    known_orbit_observations = known_orbit_observations[cols_ordered]

    return known_orbits, known_orbit_observations


def filterOrbits(
    orbits: pd.DataFrame,
    orbit_observations: pd.DataFrame,
    associations: pd.DataFrame,
    min_obs: int = 5,
    min_time_separation: float = 30.0,
    delta_cols: list = ["mjd_utc", "mag", "RA_deg", "Dec_deg"],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter orbits into orbits of previously known objects and potential discovery candidates.

    Parameters
    ----------
    orbits : `~pandas.DataFrame`
        DataFrame of orbits.
    orbit_observations : `~pandas.DataFrame`
        Dataframe of orbit observations with at least one column with the orbit ID ('orbit_id') and
        one column with the 'obs_id'
    associations : `~pandas.DataFrame`
        DataFrame of known associations, with one column of containing the observation ID ('obs_id')
        and another column containing the association ('obj_id'). Any unknown objects should have
        been assigned an unknown ID. See preprocessObservations.
    min_obs : int
        The minimum number of observations for an object to be considered as recovered.
    min_time_separation : int
        The minimum time two observations should be separated in minutes. If any observations
        for a single orbit are seperated by less than this amount then only the first observation is kept.
        This is useful to prevent stationary sources from biasing orbit fits, although may decrease overall
        completeness.
    delta_cols : list[str]
        Columns for which to calculate deltas (must include mjd_utc).

    Returns
    -------
    discovery_candidates : (`~pandas.DataFrame`, `~pandas.DataFrame`)
        DataFrame of dicovery candidate orbits and discovery candidate observations.
    known_orbits : (`~pandas.DataFrame`, `~pandas.DataFrame`)
        DataFrame of known orbits and known orbit observations.
    """
    # Calculate deltas of a variety of quantities (this returns the orbit_observations dataframe
    # with the delta columns added)
    deltas = calcDeltas(
        orbit_observations, groupby_cols=["orbit_id", "night_id"], delta_cols=delta_cols
    )

    # Mark all observations within min_time of another as filtered
    deltas.loc[:, "filtered"] = 1
    deltas.loc[
        (deltas["dmjd_utc"].isna())
        | (deltas["dmjd_utc"] >= min_time_separation / 60 / 24),
        "filtered",
    ] = 0
    orbits_ = orbits.copy()
    orbit_observations_ = deltas.copy()

    # Identify known orbits (also remove any observations of unknown objects from these orbits)
    recovered_known_orbits, recovered_known_orbit_observations = filterKnownOrbits(
        orbits_, orbit_observations_, associations, min_obs=min_obs
    )

    # Remove the known orbits from the pool of orbits
    # The remaining orbits are potential candidates
    known_orbit_ids = recovered_known_orbits["orbit_id"].values
    candidate_orbits = orbits_[~orbits_["orbit_id"].isin(known_orbit_ids)]
    candidate_orbit_observations = orbit_observations_[
        ~orbit_observations_["orbit_id"].isin(known_orbit_ids)
    ]

    # Remove any observations of the candidate discoveries that are potentially
    # too close in time to eachother (removes stationary source that could bias results)
    # Any orbits that now have fewer than min_obs observations are also removed
    candidate_orbit_observations = candidate_orbit_observations[
        candidate_orbit_observations["filtered"] == 0
    ]
    occurences = candidate_orbit_observations["orbit_id"].value_counts()
    orbit_ids = occurences.index.values[occurences.values >= min_obs]
    candidate_orbits = orbits[orbits["orbit_id"].isin(orbit_ids)]
    candidate_orbit_observations = candidate_orbit_observations[
        candidate_orbit_observations["orbit_id"].isin(orbit_ids)
    ]

    # Add a trkSub column to the discovery candidates
    trk_subs = [
        f"t{i[0:4]}{i[-3:]}" for i in candidate_orbit_observations["orbit_id"].values
    ]
    candidate_orbit_observations.insert(1, "trkSub", trk_subs)

    discovery_candidates = (candidate_orbits, candidate_orbit_observations)
    known_orbits = (recovered_known_orbits, recovered_known_orbit_observations)

    return discovery_candidates, known_orbits
