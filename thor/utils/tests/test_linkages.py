import numpy as np
import pandas as pd

from ..linkages import sortLinkages

### Create test data set
linkage_ids = ["a", "b", "c"]
linkage_lengths = [4, 5, 6]

linkage_members_ids = []
for i, lid in enumerate(linkage_ids):
    linkage_members_ids += [i for j in range(linkage_lengths[i])]
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

