import numpy as np
import pandas as pd

__all__ = [
    "identifySubsetLinkages"
]

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