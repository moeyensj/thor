import uuid
import warnings
import numpy as np
import pandas as pd

__all__ = [
    "sortLinkages",
    "verifyLinkages",
    "generateCombinations",
    "identifySubsetLinkages",
    "mergeLinkages"
]

def sortLinkages(
        linkages, 
        linkage_members, 
        observations,
        linkage_id_col="orbit_id"
    ):
    linkages_sorted = linkages.copy()
    linkage_members_sorted = linkage_members.copy()

    linkages_sorted.sort_values(
        by=[linkage_id_col],
        inplace=True
    )
    linkage_members_sorted = linkage_members_sorted.merge(
        observations[["obs_id", "mjd_utc"]],          
        on="obs_id", 
        how="left",
    )
    linkage_members_sorted.sort_values(
        by=[linkage_id_col, "mjd_utc", "obs_id"],
        inplace=True
    )
    linkage_members_sorted.drop(
        columns=["mjd_utc"],
        inplace=True
    )
    for df in [linkages_sorted, linkage_members_sorted]:
        df.reset_index(
            inplace=True,
            drop=True
        )
        
    return linkages_sorted, linkage_members_sorted

def verifyLinkages(
        linkages, 
        linkage_members, 
        observations, 
        linkage_id_col="orbit_id"
    ):
    
    linkages_verified = linkages.copy()
    linkage_members_verified = linkage_members.copy()
    
    reset_index = False
    if not np.all(linkages_verified[linkage_id_col].values == linkage_members_verified[linkage_id_col].unique()):
        warning = (
            "Linkages and linkage_members dataframes are not equally sorted by linkage ID.\n"
            "Sorting..."
        )
        warnings.warn(warning)
        
        linkages_verified.sort_values(
            by=[linkage_id_col], 
            inplace=True
        )
        linkage_members_verified.sort_values(
            by=[linkage_id_col], 
            inplace=True
        )
        reset_index = True
    
    linkage_members_verified = linkage_members_verified.merge(observations[["obs_id", "mjd_utc"]], 
        on="obs_id",
        how="left"
    )
    if not np.all(linkage_members_verified.sort_values(by=[linkage_id_col, "mjd_utc"])[["orbit_id", "obs_id"]].values == linkage_members_verified[["orbit_id", "obs_id"]].values):
        warning = (
            "Linkage_members is not sorted by {} and mjd_utc.\n"
            "Sorting..."
        )
        warnings.warn(warning.format(linkage_id_col))

        linkage_members_verified.sort_values(
            by=[linkage_id_col, "mjd_utc", "obs_id"], 
            inplace=True
        )
        reset_index = True

    if reset_index == True:
        for df in [linkages_verified, linkage_members_verified]:
            df.reset_index(
                inplace=True,
                drop=True
            )
    
    return linkages_verified, linkage_members_verified[[linkage_id_col, "obs_id"]]


def generateCombinations(
        x, 
        idx=None, 
        ct=None, 
        reps=None
    ):
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

def identifySubsetLinkages(
        all_linkages, 
        linkage_members, 
        linkage_id_col="orbit_id"
    ):
    """
    Identify each linkage that is a subset of a larger linkage. 
    
    Parameters
    ----------
    all_linkages : 
    
    
    
    
    
    """
    
    
    linkage_members_merged = linkage_members.copy()
    all_linkages_merged = all_linkages.copy()
    all_linkages_merged["subset_of"] = None

    counts = linkage_members["obs_id"].value_counts()
    duplicate_obs_ids = counts.index[counts.values > 1].values
    
    subset_linkages = []
    obs_ids_analyzed = set()
    i = 0
    while len(obs_ids_analyzed) != len(duplicate_obs_ids):

        obs_id = duplicate_obs_ids[i]

        if obs_id not in obs_ids_analyzed:

            # Find all linkages that contain this observation (that have not already been identified as a subset)
            linkage_ids = linkage_members_merged[linkage_members_merged["obs_id"].isin([obs_id])][linkage_id_col].values

            # Count the occurences of these linkages (the number of observations in each linkage)
            linkage_id_counts = linkage_members_merged[(
                linkage_members_merged[linkage_id_col].isin(linkage_ids) 
                & (~linkage_members_merged[linkage_id_col].isin(subset_linkages))
            )][linkage_id_col].value_counts()
            linkage_ids = linkage_id_counts.index.values

            for linkage_id_i in linkage_ids:

                # Has linkage i already been identified as a subset? If not, see if it has any subset linkages
                is_subset_i = all_linkages_merged[all_linkages_merged[linkage_id_col].isin([linkage_id_i])]["subset_of"].values[0]
                if not is_subset_i:

                    # Grab linkage i's observation IDs
                    obs_ids_i = linkage_members_merged[linkage_members_merged[linkage_id_col].isin([linkage_id_i])]["obs_id"].values

                    for linkage_id_j in linkage_ids[np.where(linkage_ids != linkage_id_i)]:

                        # If this linkage has not already been marked as a subset of another, check to see 
                        # if it is a subset 
                        is_subset_j = all_linkages_merged[all_linkages_merged[linkage_id_col].isin([linkage_id_j])]["subset_of"].values[0]
                        if not is_subset_j:

                            # Grab linkage j's observation IDs
                            obs_ids_j = linkage_members_merged[linkage_members_merged[linkage_id_col].isin([linkage_id_j])]["obs_id"].values

                            # If linkage j is a subset of linkage i, flag it as such
                            if set(obs_ids_j).issubset(set(obs_ids_i)):
                                all_linkages_merged.loc[all_linkages_merged[linkage_id_col].isin([linkage_id_j]), "subset_of"] = linkage_id_i

                                subset_linkages.append(linkage_id_j)
                                for j in obs_ids_j:
                                    obs_ids_analyzed.add(j)


            obs_ids_analyzed.add(obs_id)

        i += 1
        
    return all_linkages_merged, linkage_members_merged

def mergeLinkages(linkages, linkage_members, observations, linkage_id_col="orbit_id"):

    assert "mjd_utc" not in linkage_members.columns
    
    obs_id_occurences = linkage_members["obs_id"].value_counts()
    duplicate_obs_ids = obs_id_occurences.index.values[obs_id_occurences.values > 1]
    linkage_members_ = linkage_members.merge(observations[["obs_id", "mjd_utc"]], on="obs_id")

    if linkage_id_col == "orbit_id":
        columns = ["orbit_id", "epoch", "x", "y", "z", "vx", "vy", "vz"]
    else:
        columns = ["cluster_id", "vtheta_x_deg", "vtheta_y_deg"]

    merged_linkages = []
    merged_linkage_members = []
    merged_from = []
    while len(duplicate_obs_ids) > 0:

        duplicate_obs_id = duplicate_obs_ids[0]
        linkage_ids_i = linkage_members_[linkage_members_["obs_id"].isin([duplicate_obs_id])][linkage_id_col].unique()
        obs_ids = linkage_members_[linkage_members_[linkage_id_col].isin(linkage_ids_i)]["obs_id"].unique()
        times = linkage_members_[linkage_members_["obs_id"].isin(obs_ids)].drop_duplicates(subset=["obs_id"])["mjd_utc"].values

        obs_ids = obs_ids[np.argsort(times)]
        times = times[np.argsort(times)]
        for combination in generateCombinations(times):

            new_possible_linkages = linkages[linkages[linkage_id_col].isin(linkage_ids_i)].copy()
            new_linkage = new_possible_linkages.sort_values(by=["rchi2"])[:1]
            new_linkage_id = str(uuid.uuid4().hex)
            new_linkage[linkage_id_col] = new_linkage_id

            new_linkage_members = {
                linkage_id_col : [new_linkage_id for i in range(len(obs_ids[combination]))],
                "obs_id" : obs_ids[combination],
                "mjd_utc" : times[combination]
            }
            merged_from_i = {
                "orbit_id" : [new_linkage_id for i in range(len(linkage_ids_i))],
                "merged_from" : linkage_ids_i
            }
            merged_linkages.append(new_linkage)
            merged_linkage_members.append(pd.DataFrame(new_linkage_members))
            merged_from.append(pd.DataFrame(merged_from_i))

        duplicate_obs_ids = np.delete(duplicate_obs_ids, np.isin(duplicate_obs_ids, obs_ids))

    if len(merged_linkages) > 0:
        merged_linkages = pd.concat(merged_linkages) 
        merged_linkage_members = pd.concat(merged_linkage_members)
        merged_from = pd.concat(merged_from)
        
        merged_linkages.sort_values(
            by=[linkage_id_col],
            inplace=True
        )
        merged_linkage_members.sort_values(
            by=[linkage_id_col, "mjd_utc"],
            inplace=True
        )
        merged_from.sort_values(
            by=[linkage_id_col],
            inplace=True
        )
        
        for df in [merged_linkages, merged_linkage_members, merged_from]:
            df.reset_index(
                inplace=True,
                drop=True
            )
    
    else:
        
        merged_linkages = pd.DataFrame(
            columns=columns
        )
    
        merged_linkage_members = pd.DataFrame(
            columns=[linkage_id_col, "obs_id"]
        )

        merged_from = pd.DataFrame(
            columns=[linkage_id_col, "merged_from"]
        )
    return merged_linkages[columns], merged_linkage_members[[linkage_id_col, "obs_id"]], merged_from