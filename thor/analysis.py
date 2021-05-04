import os
import pandas as pd
from difi import analyzeLinkages
from difi import analyzeObservations

from .orbits import Orbits

__all__ = [
    "readOrbitDir",
    "analyzeTHOROrbit",
    "analyzeTHOR"
]


def readOrbitDir(orbit_dir):
    
    projected_observations = pd.read_csv(
        os.path.join(orbit_dir, "projected_observations.csv"),
        index_col=False,
        dtype={"obs_id" : str}
    )
    
    clusters = pd.read_csv(
        os.path.join(orbit_dir, "clusters.csv"),
        index_col=False
    )

    cluster_members = pd.read_csv(
        os.path.join(orbit_dir, "cluster_members.csv"),
        index_col=False,
        dtype={"obs_id" : str}
    )
    
    iod_orbits = Orbits.from_csv(
        os.path.join(orbit_dir, "iod_orbits.csv"),
    ).to_df(include_units=False)
        
    iod_orbit_members = pd.read_csv(
        os.path.join(orbit_dir, "iod_orbit_members.csv"),
        index_col=False,
        dtype={"obs_id" : str}
    )
    
    od_orbits = Orbits.from_csv(
        os.path.join(orbit_dir, "od_orbits.csv"),
    ).to_df(include_units=False)
        
    od_orbit_members = pd.read_csv(
        os.path.join(orbit_dir, "od_orbit_members.csv"),
        index_col=False,
        dtype={"obs_id" : str}
    )
    
    recovered_orbits = Orbits.from_csv(
        os.path.join(orbit_dir, "recovered_orbits.csv"),
    ).to_df(include_units=False)
        
    recovered_orbit_members = pd.read_csv(
        os.path.join(orbit_dir, "recovered_orbit_members.csv"),
        index_col=False,
        dtype={"obs_id" : str}
    )
    
    data_products = (
        projected_observations, 
        clusters, cluster_members, 
        iod_orbits, iod_orbit_members, 
        od_orbits, od_orbit_members, 
        recovered_orbits, recovered_orbit_members
    )
    return data_products

def analyzeTHOROrbit(
        preprocessed_associations,
        orbit_dir, 
        classes=None, 
        min_obs=5,
        contamination_percentage=20,
        metric="min_obs", 
        metric_kwargs={
            "min_obs" : 5,
        }):
    
    data_products = readOrbitDir(orbit_dir)
    (
        projected_observations,
        clusters,
        cluster_members, 
        iod_orbits,
        iod_orbit_members, 
        od_orbits,
        od_orbit_members, 
        recovered_orbits,
        recovered_orbit_members 
    ) = data_products
    
    analysis_observations = projected_observations.merge(
        preprocessed_associations, 
        on="obs_id",
        how="left"
    )
    
    
    column_mapping = {
        "obs_id" : "obs_id",
        "linkage_id" : "cluster_id",
        "truth" : "obj_id"    
    }

    all_truths, findable_observations, summary = analyzeObservations(
        analysis_observations,
        classes=classes,
        metric=metric,
        **metric_kwargs,
        column_mapping=column_mapping,
    )

    all_clusters, all_truths_clusters, summary_clusters = analyzeLinkages(
        analysis_observations,
        cluster_members,
        all_truths=all_truths,
        min_obs=min_obs,
        contamination_percentage=contamination_percentage,
        classes=classes,
        column_mapping=column_mapping
    )
    for df in [all_clusters, all_truths_clusters, summary_clusters]:
        df.insert(0, "component", "clustering")

    column_mapping["linkage_id"] = "orbit_id"

    all_iod_orbits, all_truths_iod, summary_iod = analyzeLinkages(
        analysis_observations,
        iod_orbit_members,
        all_truths=all_truths,
        min_obs=min_obs,
        contamination_percentage=contamination_percentage,
        classes=classes,
        column_mapping=column_mapping
    )
    for df in [all_iod_orbits, all_truths_iod, summary_iod]:
        df.insert(0, "component", "iod")

    all_od_orbits, all_truths_od, summary_od = analyzeLinkages(
        analysis_observations,
        od_orbit_members,
        all_truths=all_truths,
        min_obs=min_obs,
        contamination_percentage=0.0,
        classes=classes,
        column_mapping=column_mapping
    )
    for df in [all_od_orbits, all_truths_od, summary_od]:
        df.insert(0, "component", "od")
        
    all_recovered_orbits, all_truths_recovered, summary_recovered = analyzeLinkages(
        analysis_observations,
        recovered_orbit_members,
        all_truths=all_truths,
        min_obs=min_obs,
        contamination_percentage=0.0,
        classes=classes,
        column_mapping=column_mapping
    )
    for df in [all_recovered_orbits, all_truths_recovered, summary_recovered]:
        df.insert(0, "component", "od+a")


    summary = pd.concat([summary_clusters, summary_iod, summary_od, summary_recovered])
    summary.reset_index(
        inplace=True,
        drop=True
    )
    
    all_truths = pd.concat([all_truths_clusters, all_truths_iod, all_truths_od, all_truths_recovered])
    all_truths.reset_index(
        inplace=True,
        drop=True
    )
    
    all_linkages = pd.concat([all_clusters, all_iod_orbits, all_od_orbits, all_recovered_orbits])
    all_linkages.reset_index(
        inplace=True,
        drop=True
    )
    
    all_linkages = all_linkages[[
        "component", "cluster_id", "orbit_id", "num_obs", "num_members", "pure",
        "pure_complete", "partial", "mixed", "contamination_percentage",
        "found_pure", "found_partial", "found", "linked_truth"
    ]]
    all_linkages.loc[all_linkages["cluster_id"].isna(), "cluster_id"] = "None"
    all_linkages.loc[all_linkages["orbit_id"].isna(), "orbit_id"] = "None"
    
    summary["linkage_purity"] = 100 * (summary["pure_linkages"] / summary["linkages"])
    
    return all_linkages, all_truths, summary

def analyzeTHOR(
        preprocessed_associations, 
        out_dir, 
        min_obs=5, 
        contamination_percentage=20,
        classes=None,
        metric="min_obs",  
        metric_kwargs={
            "min_obs" : 5
        }
    ):

    # Read preprocessed observations from out dir
    preprocessed_observations = pd.read_csv(
        os.path.join(out_dir, "preprocessed_observations.csv"),
        index_col=False,
        dtype={"obs_id" : str}
    )

    # Merge with prepprocessed associations to create a set of 'analysis observations':
    # observations that contain any previously known labels
    analysis_observations = preprocessed_observations.merge(
        preprocessed_associations, 
        on="obs_id"
    )

    # Calculate which objects should be findable
    column_mapping = {
        'obs_id' : 'obs_id', 
        'truth' : 'obj_id',
        'linkage_id' : 'orbit_id'
    }
    all_truths, findable_observations, summary = analyzeObservations(
        analysis_observations,
        classes=classes,
        metric='min_obs',
        column_mapping=column_mapping,
        **metric_kwargs
    )

    # Read the recovered orbits and orbit members
    recovered_orbits = Orbits.from_csv(
        os.path.join(out_dir, "recovered_orbits.csv")
    )

    recovered_orbit_members = pd.read_csv(
        os.path.join(out_dir, "recovered_orbit_members.csv"),
        index_col=False,
        dtype={"obs_id" : str}
    )

    # Calculate which objects were actually recovered 
    all_recovered_orbits, all_recovered_truths, recovered_summary = analyzeLinkages(
        analysis_observations,
        recovered_orbit_members,
        all_truths=all_truths,
        contamination_percentage=0.0, # Recovered orbits should be contamination free (no partial orbits)
        classes=classes,
        column_mapping=column_mapping
    )

    # Read test_orbits file from out_dir
    test_orbits_file = os.path.join(out_dir, "test_orbits_out.csv")
    if not os.path.exists(test_orbits_file):
        raise ValueError("Cannot find test_orbits_out.csv.")
    else:
        test_orbits = Orbits.from_csv(test_orbits_file)

    test_orbits_df = test_orbits.to_df(include_units=False)

    all_linkages_dfs = []
    all_truths_dfs = []
    summary_dfs = []

    # Go through each test orbit directory and analyze the outputs
    for orbit_id in test_orbits_df["test_orbit_id"].unique():

        orbit_dir = os.path.join(out_dir, "orbit_{}".format(orbit_id))

        all_linkages, all_truths, summary = analyzeTHOROrbit(
            preprocessed_associations,
            orbit_dir, 
            classes=classes, 
            min_obs=min_obs,
            contamination_percentage=contamination_percentage,
            metric="min_obs",
            metric_kwargs=metric_kwargs,
        )

        for df in [all_linkages, all_truths, summary]:
            df.insert(0, "test_orbit_id", orbit_id)

        all_linkages_dfs.append(all_linkages)
        all_truths_dfs.append(all_truths)
        summary_dfs.append(summary)

    all_linkages = pd.concat(
        all_linkages_dfs,
        ignore_index=True
    )
    all_truths = pd.concat(
        all_truths_dfs,
        ignore_index=True
    )
    summary = pd.concat(
        summary_dfs,
        ignore_index=True
    )

    test_orbit_analysis = (all_linkages, all_truths, summary)
    run_analysis = (all_recovered_orbits, all_recovered_truths, recovered_summary)
    return run_analysis, test_orbit_analysis