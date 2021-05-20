from thor import clusterVelocity
import numpy as np
import pandas as pd
import random
import cProfile

random.seed(1)


def load_df():
    return pd.read_csv("data/one_cluster_iteration.csv")


def load_cluster_input(max_observations=None, max_exposures=None):
    df = load_df()

    if max_exposures is not None:
        exposure_times = df['dt'].unique()
        if max_exposures > len(exposure_times):
            exposures = random.sample(list(exposure_times), max_exposures)
            df = df[df['dt'].isin(exposures)]

    if max_observations is not None and max_observations > len(df):
        df = df[:max_observations]
    return df


def run(max_obs=100, max_exp=-1):
    df = load_cluster_input(max_observations=max_obs, max_exposures=max_exp)
    obs_ids = np.array(df['obs_ids'])
    x = np.array(df['x'])
    y = np.array(df['y'])
    dt = np.array(df['dt'])
    vx = 0.0
    vy = 0.0
    eps = 0.005
    min_samples = 5
    min_arc_length = 1.0
    prof = cProfile.Profile()
    prof.enable()
    clusterVelocity(obs_ids, x, y, dt,
                    vx, vy, eps, min_samples, min_arc_length)
    prof.disable()
    prof.dump_stats("benchmark_cluster.profile")
    return prof


if __name__ == "__main__":
    run()
