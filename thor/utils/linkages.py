import logging
import time

import numpy as np
import pandas as pd

__all__ = [
    "generateCombinations",
    "sortLinkages",
    "identifySubsetLinkages",
    "removeDuplicateLinkages",
    "removeDuplicateObservations",
    "calcDeltas",
]

logger = logging.getLogger(__name__)


def generateCombinations(x, idx=None, ct=None, reps=None):
    # Magic from the wizard himself: Mario Juric
    # recursively generate all combinations of idx, assuming
    # ct is the list of repeat counts of idx
    if x is not None:
        # initialization; find indices of where the repetitions are
        _, idx, ct = np.unique(x, return_counts=True, return_index=True)
        reps = np.nonzero(ct > 1)[0]
    if len(reps) == 0:
        yield idx
        return
    i = reps[0]
    idx = idx.copy()
    for _ in range(ct[i]):
        yield from generateCombinations(None, idx, ct, reps[1:])
        idx[i] += 1


def sortLinkages(linkages, linkage_members, observations, linkage_id_col="orbit_id"):
    """
    Check that linkages and linkage_members have their linkage IDs in the same order. If not,
    sort both by linkage ID. Second, check that linkage_members is additionally sorted by
    mjd_utc. If linkage_members does not contain the mjd_utc column, then observations will be merged
    to retrieve the observation time.

    Parameters
    ----------
    linkages : `~pandas.DataFrame`
        DataFrame containing at least a linkage ID column (linkage_id_col). Each unique linkage should
        be only present once.
    linkage_members : `~pandas.DataFrame`
        DataFrame containing at least a linkage ID column (linkage_id_col) and an observation ID column ('obs_id'). The observation ID
        column is used to merge on observations so that the observation time can be retrieved.
    observations : `~pandas.DataFrame`
        DataFrame containing observations with at least an observation ID column ('obs_id') and a observation time
        column ('mjd_utc').
    linkage_id_col : str
        Name of the linkage ID column.

    Returns
    -------
    linkages : `~pandas.DataFrame`
        Linkages sorted by linkage IDs.
    linkage_members : `~pandas.DataFrame`
        Linkages sorted by linkage IDs and observation times.
    """
    time_start = time.time()
    logger.debug("Verifying linkages...")

    linkages_verified = linkages.copy()
    linkage_members_verified = linkage_members.copy()

    reset_index = False
    id_sorted = np.all(
        linkages_verified[linkage_id_col].values
        == linkage_members_verified[linkage_id_col].unique()
    )
    if not id_sorted:
        logger.debug(
            f"Linkages and linkage_members dataframes are not equally sorted by {linkage_id_col}. Sorting..."
        )
        # Sort by linkage_id
        sort_start = time.time()
        linkages_verified.sort_values(by=[linkage_id_col], inplace=True)
        linkage_members_verified.sort_values(by=[linkage_id_col], inplace=True)
        sort_end = time.time()
        duration = sort_end - sort_start
        logger.debug(f"Sorting completed in {duration:.3f}s.")
        reset_index = True

    time_present = True
    if "mjd_utc" not in linkage_members_verified.columns:
        logger.debug(
            "Observation time column ('mjd_utc') is not in linkage_members, merging with observations..."
        )

        # Merge with observations to get the observation time for each observation in linkage_members
        merge_start = time.time()
        linkage_members_verified = linkage_members_verified.merge(
            observations[["obs_id", "mjd_utc"]], on="obs_id", how="left"
        )
        merge_end = time.time()
        duration = merge_end - merge_start
        logger.debug(f"Merging completed in {duration:.3f}s.")
        time_present = False

    linkage_members_verified_ = linkage_members_verified.sort_values(
        by=[linkage_id_col, "mjd_utc"]
    )
    time_sorted = np.all(
        linkage_members_verified_[[linkage_id_col, "obs_id"]].values
        == linkage_members_verified[[linkage_id_col, "obs_id"]].values
    )
    if not time_sorted:
        logger.debug(
            f"Linkage_members is not sorted by {linkage_id_col} and mjd_utc. Sorting..."
        )

        # Sort by linkage_id, mjd_utc, and finally obs_id
        sort_start = time.time()
        linkage_members_verified.sort_values(
            by=[linkage_id_col, "mjd_utc", "obs_id"], inplace=True
        )
        sort_end = time.time()
        duration = sort_end - sort_start
        logger.debug(f"Sorting completed in {duration:.3f}s.")
        reset_index = True

    if reset_index:
        for df in [linkages_verified, linkage_members_verified]:
            df.reset_index(inplace=True, drop=True)

    if not time_present:
        linkage_members_verified.drop(columns=["mjd_utc"], inplace=True)

    time_end = time.time()
    duration = time_end - time_start
    logger.debug(f"Linkages verified in {duration:.3f}s.")
    return linkages_verified, linkage_members_verified


def identifySubsetLinkages(linkage_members, linkage_id_col="orbit_id"):
    """
    Identify subset linkages. A subset is defined as a linkage which contains
    the a subset of observation IDs of a larger or equally sized linkaged.

    For example, if a linkage B has constituent observations: obs0001, obs0002, obs0003, obs0004.
    Then a linkage A with constituent observations: obs0001, obs0002, obs0003; is a subset of
    B. If linkage A and B share exactly the same observations they will be identified as subsets of each
    other.

    Parameters
    ----------
    linkage_members : `~pandas.DataFrame`
        DataFrame containing at least a linkage ID column (linkage_id_col) and an observation ID column ('obs_id').
    linkage_id_col : str
        Name of the linkage ID column.

    Returns
    -------
    subsets : `~pandas.DataFrame`
        DataFrame containing a column with the linkage_id and a second column containing the linkages identified
        as subsets. A linkage with multiple subsets will appear once for every subset linkage found.
    """
    # Create a dictionary keyed on linkage ID with a set of each linkage's observation
    # ID as values
    time_start = time.time()
    linkage_dict = {}
    for linkage_id in linkage_members[linkage_id_col].unique():
        obs_ids = linkage_members[linkage_members[linkage_id_col] == linkage_id][
            "obs_id"
        ].values
        linkage_dict[linkage_id] = set(obs_ids)
    time_end = time.time()
    duration = time_end - time_start
    logger.debug(f"Linkage dictionary created in {duration:.3f}s.")

    time_start = time.time()
    subset_dict = {}
    for linkage_id_a in linkage_dict.keys():
        # Grab linkage A's observations
        obs_ids_a = linkage_dict[linkage_id_a]

        for linkage_id_b in linkage_dict.keys():
            # If linkage A is not linkage B then
            # check if linkage B is a subset of linkage A
            if linkage_id_b != linkage_id_a:

                # Grab linkage B's observations
                obs_ids_b = linkage_dict[linkage_id_b]
                if obs_ids_b.issubset(obs_ids_a):

                    # Linkage B is a subset of Linkage A, so lets
                    # add this result to the subset dictionary
                    if linkage_id_a not in subset_dict.keys():
                        subset_dict[linkage_id_a] = [linkage_id_b]
                    else:
                        subset_dict[linkage_id_a].append(linkage_id_b)
    time_end = time.time()
    duration = time_end - time_start
    logger.debug(f"Linkage dictionary scanned for subsets in {duration:.3f}s.")

    time_start = time.time()
    linkage_ids = []
    subset_linkages = []
    for linkage_id, subset_ids in subset_dict.items():
        subset_linkages += subset_ids
        linkage_ids += [linkage_id for i in range(len(subset_ids))]
    subsets = pd.DataFrame({"linkage_id": linkage_ids, "subset_ids": subset_linkages})
    time_end = time.time()
    duration = time_end - time_start
    logger.debug(f"Subset dataframe created in {duration:.3f}s.")

    return subsets


def removeDuplicateLinkages(linkages, linkage_members, linkage_id_col="orbit_id"):
    """
    Removes linkages that have identical observations as another linkage. Linkage quality is not taken
    into account.

    Parameters
    ----------
    linkages : `~pandas.DataFrame`
        DataFrame containing at least the linkage ID.
    linkage_members : `~pandas.DataFrame`
        Dataframe containing the linkage ID and the observation ID for each of the linkage's
        constituent observations. Each observation ID should be in a single row.
    linkage_id_col : str, optional
        Linkage ID column name (must be the same in both DataFrames).

    Returns
    -------
    linkages : `~pandas.DataFrame`
        DataFrame with duplicate linkages removed.
    linkage_members : `~pandas.DataFrame`
        DataFrame with duplicate linkages removed.
    """
    linkages_ = linkages.copy()
    linkage_members_ = linkage_members.copy()

    # Expand observation IDs into columns, then remove duplicates using pandas functionality
    expanded = (
        linkage_members_[[linkage_id_col, "obs_id"]]
        .groupby(by=[linkage_id_col])["obs_id"]
        .apply(list)
        .to_frame(name="obs_ids")
    )
    expanded = expanded["obs_ids"].apply(pd.Series)
    linkage_ids = expanded.drop_duplicates().index.values

    linkages_ = linkages_[linkages_[linkage_id_col].isin(linkage_ids)]
    linkage_members_ = linkage_members_[
        linkage_members_[linkage_id_col].isin(linkage_ids)
    ]

    for df in [linkages_, linkage_members_]:
        df.reset_index(inplace=True, drop=True)

    return linkages_, linkage_members_


def removeDuplicateObservations(
    linkages,
    linkage_members,
    min_obs=5,
    linkage_id_col="orbit_id",
    filter_cols=["num_obs", "arc_length"],
    ascending=[False, False],
):
    """
    Removes duplicate observations using the filter columns. The filter columns are used to sort the linkages
    as desired by the user. The first instance of the observation is kept and all other instances are removed.
    If any linkage's number of observations drops below min_obs, that linkage is removed.

    Parameters
    ----------
    linkages : `~pandas.DataFrame`
        DataFrame containing at least the linkage ID.
    linkage_members : `~pandas.DataFrame`
        Dataframe containing the linkage ID and the observation ID for each of the linkage's
        constituent observations. Each observation ID should be in a single row.
    min_obs : int, optional
        Minimum number of observations for a linkage to be viable.
    linkage_id_col : str, optional
        Linkage ID column name (must be the same in both DataFrames).
    filter_cols : list, optional
        List of column names to use to sort the linkages.
    ascending : list, optional
        Sort the filter_cols in ascending or descending order.

    Returns
    -------
    linkages : `~pandas.DataFrame`
        DataFrame with duplicate observations removed.
    linkage_members : `~pandas.DataFrame`
        DataFrame with duplicate observations removed.
    """
    linkages_ = linkages.copy()
    linkage_members_ = linkage_members.copy()

    # Sort linkages by the desired columns
    linkages_.sort_values(
        by=filter_cols, ascending=ascending, inplace=True, ignore_index=True
    )

    # Set both dataframe's indices to the linkage ID for
    # faster querying
    linkages_.set_index(linkage_id_col, inplace=True)
    linkage_members_.set_index(linkage_id_col, inplace=True)

    # Sort linkage members the same way as linkages
    linkage_members_ = linkage_members_.loc[linkages_.index.values]
    linkage_members_.reset_index(inplace=True)

    # Drop all but the first duplicate observation
    linkage_members_ = linkage_members_.drop_duplicates(subset=["obs_id"], keep="first")

    # Make sure that the remaining linkages have enough observations (>= min_obs)
    linkage_occurences = linkage_members_[linkage_id_col].value_counts()
    linkages_to_keep = linkage_occurences.index.values[
        linkage_occurences.values >= min_obs
    ]
    linkages_ = linkages_[linkages_.index.isin(linkages_to_keep)]
    linkage_members_ = linkage_members_[
        linkage_members_[linkage_id_col].isin(linkages_to_keep)
    ]

    # Reset indices
    linkages_.reset_index(inplace=True)
    linkage_members_.reset_index(inplace=True, drop=True)
    return linkages_, linkage_members_


def calcDeltas(
    linkage_members,
    observations=None,
    groupby_cols=["orbit_id", "night_id"],
    delta_cols=["mjd_utc", "RA_deg", "Dec_deg", "mag"],
):
    """
    Calculate deltas for the desired columns. For example, if groupby columns are given to be orbit_id and night id, then
    the linkages are grouped first by orbit_id then night_id, and then the difference in quantities are calculated for
    each column in delta_cols. This can be used to calculate the nightly time difference in observations per linkage, or the
    amount of motion a linkage has between observations, etc...

    Parameters
    ----------
    linkage_members : `~pandas.DataFrame`
        DataFrame containing at least a linkage ID column (linkage_id_col) and an observation ID column ('obs_id'). The observation ID
        column is used to merge on observations so that the columns from the observations dataframe can be retrieved if necessary.
    observations : `~pandas.DataFrame`
        DataFrame containing observations with at least an observation ID column ('obs_id').
    groupby_cols : list
        Columns by which to group the linkages and calculate deltas.
    delta_cols : list
        Columns for which to calculate deltas.

    Returns
    -------
    linkage_members : `~pandas.DataFrame`
        Copy of the linkage_members dataframe with the delta columns added.
    """
    linkage_members_ = linkage_members.copy()

    # Check to see if each column on which a delta should be
    # calculated is in linkage_members, if not look for it
    # in observations
    cols = []
    for col in delta_cols + groupby_cols:
        if col not in linkage_members_.columns:
            if col not in observations.columns or observations is None:
                err = f"{col} could not be found in either linkage_members or observations."
                raise ValueError(err)

            cols.append(col)

    if len(cols) > 0:
        linkage_members_ = linkage_members_.merge(
            observations[["obs_id"] + cols], on="obs_id", how="left"
        )

    nightly = linkage_members_.groupby(by=groupby_cols)

    deltas = nightly[delta_cols].diff()
    deltas.rename(columns={c: f"d{c}" for c in delta_cols}, inplace=True)
    linkage_members_ = linkage_members_.join(deltas)

    return linkage_members_
