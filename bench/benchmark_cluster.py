from thor import clusterVelocity
from thor import clusters
import numpy as np
import pandas as pd
import random
import cProfile
import time
import os

random.seed(1)


def load_df():
    return pd.read_csv("data/one_cluster_iteration.csv")


def load_cluster_input(max_observations=None, max_exposures=None):
    df = load_df()

    if max_exposures is not None and max_exposures != -1:
        exposure_times = df['dt'].unique()
        if len(exposure_times) > max_exposures:
            exposures = random.sample(list(exposure_times), max_exposures)
            df = df[df['dt'].isin(exposures)]

    if max_observations is not None and len(df) > max_observations:
        df = df[:max_observations]
    return df


def warm_numba_jit():
    df = load_cluster_input(max_observations=1000, max_exposures=-1)
    x = np.array(df['x'])
    y = np.array(df['y'])
    eps = 0.01
    min_samples = 5
    points = np.array((x, y)).T
    found = clusters.find_clusters(points, eps, min_samples, alg="hotspot_2d")
    clusters.filter_clusters_by_length(
        found, np.array(df['dt']), min_samples, 1.0,
    )

def run(df, eps=0.01, min_samples=5, alg="hotspot_2d", run_id="run"):
    obs_ids = np.array(df['obs_ids'])
    x = np.array(df['x'])
    y = np.array(df['y'])
    dt = np.array(df['dt'])
    vx = 0.0
    vy = 0.0
    eps = eps
    min_samples = min_samples
    min_arc_length = 1.0

    points = np.array((x, y)).T
    prof = cProfile.Profile()
    prof.enable()
    start = time.perf_counter()
    found = clusters.find_clusters(points, eps, min_samples, alg=alg)
    filtered = clusters.filter_clusters_by_length(
        found, dt, min_samples, min_arc_length,
    )
    end = time.perf_counter()
    prof.disable()

    profile_name = f"results/{run_id}/profiles/benchmark_cluster_{time.time():.6f}.profile"
    prof.dump_stats(profile_name)

    tags = {
        "n_obs": str(len(df['x'])),
        "n_exp": str(len(np.unique(df['dt']))),
        "eps": str(eps),
        "min_samples": str(min_samples),
        "alg": alg,
        "runtime": f"{end - start:.6f}",
        "n_found": str(len(found)),
        "n_postfilter": str(len(filtered)),
        "profile": profile_name,
    }

    msg = "\t".join(f"{k}={v}" for k, v in tags.items())
    print(msg)

    return tags


if __name__ == "__main__":
    print("warming numba")
    warm_numba_jit()

    runs = []
    run_id = int(time.time())
    print(f"running benchmarks\tid={run_id}")
    os.makedirs(f"results/{run_id}/profiles")
    for max_obs in (100, 1000, 10000, -1):
        for max_exp in (10, 50, 100, -1):
            df = load_cluster_input(max_obs, max_exp)
            for eps in (0.005, 0.01, 0.02):
                for min_samples in (4, 5, 6):
                    for alg in ("hotspot_2d", "dbscan"):
                        tags = run(
                            df,
                            eps=eps,
                            min_samples=min_samples,
                            alg=alg,
                            run_id=run_id,
                        )
                        runs.append(tags)
    tag_df = pd.DataFrame(runs)
    summary_csv = f"results/{run_id}/summary.csv"
    tag_df.to_csv(summary_csv)
    print(f"all benchmarks complete, metadata written to {summary_csv}")
