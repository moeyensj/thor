import pandas as pd
from typing import Tuple

from .data_processing import UNKNOWN_ID_REGEX

def filterKnownOrbits(
        orbits: pd.DataFrame,
        orbit_observations: pd.DataFrame,
        associations: pd.DataFrame,
        min_obs: int = 5,
    ) -> Tuple[pd.DataFrame]:
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
    labeled_observations = orbit_observations.merge(associations[["obs_id", "obj_id"]], on="obs_id", how="left")

    # Keep only observations of known objects
    labeled_observations = labeled_observations[~labeled_observations["obj_id"].str.contains(UNKNOWN_ID_REGEX, regex=True)]

    # Keep only known objects with at least min_obs observations
    occurences = labeled_observations["orbit_id"].value_counts()
    orbit_ids = occurences.index.values[occurences.values >= min_obs]

    # Filter input orbits
    orbits_mask = orbits["orbit_id"].isin(orbit_ids)
    orbit_observations_mask = labeled_observations["orbit_id"].isin(orbit_ids)
    known_orbits = orbits[orbits_mask].copy()
    known_orbit_observations = labeled_observations[orbit_observations_mask].copy()

    # Split into permanent and provisional designations
    known_orbit_observations.loc[:, "permID"] = ""
    known_orbit_observations.loc[:, "provID"] = ""

    # Process permanent IDs first
    # TODO : add an equivalent for Comets
    perm_ids = known_orbit_observations["obj_id"].str.isnumeric()
    known_orbit_observations.loc[perm_ids, "permID"] = known_orbit_observations[perm_ids]["obj_id"].values

    # Identify provisional IDs next
    prov_ids = (
        (~known_orbit_observations["obj_id"].str.isnumeric())
        & (~known_orbit_observations["obj_id"].str.contains(UNKNOWN_ID_REGEX, regex=True))
    )
    known_orbit_observations.loc[prov_ids, "provID"] = known_orbit_observations[prov_ids]["obj_id"].values

    # Reorder the columns to put the labels toward the front
    cols = known_orbit_observations.columns
    first = ["orbit_id", "permID", "provID", "obj_id", "obs_id"]
    cols_ordered = first + cols[~cols.isin(first)].tolist()
    known_orbit_observations = known_orbit_observations[cols_ordered]

    return known_orbits, known_orbit_observations