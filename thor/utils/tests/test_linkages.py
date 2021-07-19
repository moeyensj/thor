import pytest
import numpy as np
import pandas as pd

from ..linkages import sortLinkages
from ..linkages import calcDeltas
from ..linkages import identifySubsetLinkages

### Create test data set
linkage_ids = ["a", "b", "c"]
linkage_lengths = [4, 5, 6]

linkage_members_ids = []
for i, lid in enumerate(linkage_ids):
    linkage_members_ids += [lid for j in range(linkage_lengths[i])]
obs_ids = [f"o{i:04d}" for i in range(len(linkage_members_ids))]

times = []
for i in range(len(linkage_ids)):
    times += [np.arange(59000, 59000 + linkage_lengths[i])]
times = np.concatenate(times)

LINKAGES = pd.DataFrame({
    "linkage_id" : linkage_ids
})
LINKAGE_MEMBERS = pd.DataFrame({
    "linkage_id" : linkage_members_ids,
    "obs_id" : obs_ids,
    "mjd_utc" : times,
})
OBSERVATIONS = pd.DataFrame({
    "obs_id" : obs_ids,
    "mjd_utc" : times,
})
OBSERVATIONS.sort_values(
    by=["mjd_utc", "obs_id"],
    inplace=True,
    ignore_index=True
)

def test_sortLinkages_timePresent():

    # Scramble the linkages dataframe
    len_linkages = len(LINKAGES)
    scramble = np.random.choice(len_linkages, len_linkages, replace=False)
    linkages_unsorted = LINKAGES.loc[scramble].reset_index(
        drop=True
    )

    # Scramble the linkage_members dataframe
    len_members = len(LINKAGE_MEMBERS)
    scramble = np.random.choice(len_members, len_members, replace=False)
    linkage_members_unsorted = LINKAGE_MEMBERS.loc[scramble].reset_index(
        drop=True
    )

    # Sort scrambled linkages
    linkages_sorted, linkage_members_sorted = sortLinkages(
        linkages_unsorted,
        linkage_members_unsorted,
        OBSERVATIONS,
        linkage_id_col="linkage_id"
    )

    # Make sure they returned dataframes match those created
    pd.testing.assert_frame_equal(LINKAGES, linkages_sorted)
    pd.testing.assert_frame_equal(LINKAGE_MEMBERS, linkage_members_sorted)

def test_sortLinkages_timeMissing():

    # Scramble the linkages dataframe
    len_linkages = len(LINKAGES)
    scramble = np.random.choice(len_linkages, len_linkages, replace=False)
    linkages_unsorted = LINKAGES.loc[scramble].reset_index(
        drop=True
    )

    # Scramble the linkage_members dataframe
    len_members = len(LINKAGE_MEMBERS)
    scramble = np.random.choice(len_members, len_members, replace=False)
    linkage_members_unsorted = LINKAGE_MEMBERS.loc[scramble].reset_index(
        drop=True
    )
    linkage_members_unsorted.drop(
        columns=["mjd_utc"],
        inplace=True
    )

    # Sort scrambled linkages
    linkages_sorted, linkage_members_sorted = sortLinkages(
        linkages_unsorted,
        linkage_members_unsorted,
        OBSERVATIONS,
        linkage_id_col="linkage_id"
    )

    # Make sure they returned dataframes match those created
    pd.testing.assert_frame_equal(LINKAGES, linkages_sorted)
    pd.testing.assert_frame_equal(LINKAGE_MEMBERS[["linkage_id", "obs_id"]], linkage_members_sorted)

def test_calcDeltas():

    # Calculate deltas for the time column and make sure they match the input dataset
    linkages_members_ = calcDeltas(
        LINKAGE_MEMBERS,
        OBSERVATIONS,
        groupby_cols=["linkage_id"],
        delta_cols=["mjd_utc"]
    )

    assert "dmjd_utc" in linkages_members_.columns
    assert linkages_members_[linkages_members_["linkage_id"] == "a"]["dmjd_utc"].sum() == 3
    assert linkages_members_[linkages_members_["linkage_id"] == "b"]["dmjd_utc"].sum() == 4
    assert linkages_members_[linkages_members_["linkage_id"] == "c"]["dmjd_utc"].sum() == 5

def test_calcDeltas_columnInObservations():

    # Calculate deltas for the time column and make sure they match the input dataset, but
    # this time remove the column from linkage_members so that calcDeltas looks for it
    # in observations
    linkages_members_ = calcDeltas(
        LINKAGE_MEMBERS[["linkage_id", "obs_id"]],
        OBSERVATIONS,
        groupby_cols=["linkage_id"],
        delta_cols=["mjd_utc"]
    )

    assert "dmjd_utc" in linkages_members_.columns
    assert linkages_members_[linkages_members_["linkage_id"] == "a"]["dmjd_utc"].sum() == 3
    assert linkages_members_[linkages_members_["linkage_id"] == "b"]["dmjd_utc"].sum() == 4
    assert linkages_members_[linkages_members_["linkage_id"] == "c"]["dmjd_utc"].sum() == 5

def test_calcDeltas_missingColumn():

    # Calculate deltas for the time column and make sure they match the input dataset
    with pytest.raises(ValueError):
        linkages_members_ = calcDeltas(
            LINKAGE_MEMBERS[["linkage_id", "obs_id"]],
            OBSERVATIONS[["obs_id"]],
            groupby_cols=["linkage_id"],
            delta_cols=["mjd_utc"]
        )

def test_identifySubsetLinkages_0subsets():

    # No subsets should be found in the default dataset
    subsets = identifySubsetLinkages(LINKAGE_MEMBERS, linkage_id_col="linkage_id")
    assert len(subsets) == 0

    return

def test_identifySubsetLinkages_3subsets():

    # Make a copy of the linkage members dataframe
    # Rename the observations so that A is a subset of B and C, and so that B is a subset of C.
    linkage_members = LINKAGE_MEMBERS.copy()
    for linkage_id in LINKAGE_MEMBERS["linkage_id"].unique():
        num_obs = len(linkage_members[linkage_members["linkage_id"].isin([linkage_id])])
        linkage_members.loc[linkage_members["linkage_id"].isin([linkage_id]), "obs_id"] = [f"o{i:04d}" for i in range(num_obs)]

    subsets = identifySubsetLinkages(linkage_members, linkage_id_col="linkage_id")

    # Make sure A and B have been identified as subsets of C
    C_subsets = subsets[subsets["linkage_id"] == "c"]["subset_ids"].values
    assert "a" in C_subsets
    assert "b" in C_subsets
    assert len(C_subsets) == 2

    # Make sure A has been identifed as a subsets of B
    B_subsets = subsets[subsets["linkage_id"] == "b"]["subset_ids"].values
    assert "a" in B_subsets
    assert len(B_subsets) == 1

    # Make sure A has no subsets
    A_subsets = subsets[subsets["linkage_id"] == "a"]["subset_ids"].values
    assert len(A_subsets) == 0

    return

def test_identifySubsetLinkages_3duplicates():

    # Make a copy of the linkage members dataframe
    # Rename the observations so that A is a subset of B and C, and so that B is a subset of C.
    linkage_members = LINKAGE_MEMBERS.copy()
    for linkage_id in LINKAGE_MEMBERS["linkage_id"].unique():
        num_obs = len(linkage_members[linkage_members["linkage_id"].isin([linkage_id])])
        linkage_members.loc[linkage_members["linkage_id"].isin([linkage_id]), "obs_id"] = [f"o{i:04d}" for i in range(num_obs)]

    # Trim the linkages so that they have exactly the same observations
    linkage_members = linkage_members[~linkage_members["obs_id"].isin(["o0004", "o0005"])].copy()

    subsets = identifySubsetLinkages(linkage_members, linkage_id_col="linkage_id")

    # Make sure A and B have been identified as subsets of C
    C_subsets = subsets[subsets["linkage_id"] == "c"]["subset_ids"].values
    assert "a" in C_subsets
    assert "b" in C_subsets
    assert len(C_subsets) == 2

    # Make sure A and C have been identified as subsets of B
    B_subsets = subsets[subsets["linkage_id"] == "b"]["subset_ids"].values
    assert "a" in B_subsets
    assert "c" in B_subsets
    assert len(B_subsets) == 2

    # Make sure B and C have been identified as subsets of C
    C_subsets = subsets[subsets["linkage_id"] == "c"]["subset_ids"].values
    assert "a" in C_subsets
    assert "b" in C_subsets
    assert len(C_subsets) == 2

    return