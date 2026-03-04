"""
Run the clustering benchmark for multiple algorithms and produce a
combined comparison summary.

This script calls ``run_benchmark.py`` once per algorithm (sequentially),
then merges the per-algorithm ``summary_<alg>.parquet`` files into a
single ``summary_comparison.parquet`` (and ``.csv``) for easy analysis.

Usage
-----
    python experiments/run_algorithm_comparison.py \
        --algorithms dbscan optics kdtree fft hough \
        --benchmark_dir experiments/benchmark \
        --object_ids 433 \
        --linking_windows 15 \
        --cadences pairs \
        --noise_densities 0 100

Any extra arguments after the algorithm-comparison flags are forwarded
to ``run_benchmark.py`` verbatim.
"""

def generate_comparison_plots(combined: "pd.DataFrame", output_dir: "Path"):
    """
    Generate algorithm comparison plots from a combined summary DataFrame.

    Produces three HTML+PNG plots:
      1. Completeness vs noise density (one line per algorithm)
      2. Purity vs noise density (one line per algorithm)
      3. Wall time vs noise density (one line per algorithm)

    Parameters
    ----------
    combined : pd.DataFrame
        Merged summary with columns: cluster_algorithm, noise_density,
        findable, found_pure_clusters, pure_clusters, contaminated_clusters,
        pipeline_elapsed_s, etc.
    output_dir : Path
        Directory to write plot files into.
    """
    import plotly.graph_objects as go

    output_dir.mkdir(parents=True, exist_ok=True)
    algorithms = sorted(combined["cluster_algorithm"].unique())

    # ── Aggregate per (algorithm, noise_density) ──────────────────────
    # Completeness: fraction of findable samples where a pure cluster was found
    # Purity: pure_clusters / (pure_clusters + contaminated_clusters)
    # Wall time: mean pipeline_elapsed_s per (object, window, cadence, noise)

    rows = []
    for alg in algorithms:
        alg_df = combined[combined["cluster_algorithm"] == alg]
        for nd, grp in alg_df.groupby("noise_density"):
            # Completeness — only consider findable samples
            findable_mask = grp["findable"] == True  # noqa: E712
            n_findable = findable_mask.sum()
            if n_findable > 0:
                n_found = grp.loc[findable_mask, "found_pure_clusters"].sum()
                completeness = n_found / n_findable
            else:
                completeness = float("nan")

            # Purity — across all samples that have cluster counts
            total_pure = grp["pure_clusters"].sum()
            total_contaminated = grp["contaminated_clusters"].sum()
            total_clusters = total_pure + total_contaminated
            purity = total_pure / total_clusters if total_clusters > 0 else float("nan")

            # Wall time — mean over unique pipeline runs
            # Each (object, window, cadence, noise) combo is one pipeline run
            # but produces multiple sample rows. Deduplicate by taking one per run.
            run_keys = ["object_id", "linking_window", "cadence", "noise_density"]
            if all(k in grp.columns for k in run_keys):
                run_times = grp.drop_duplicates(subset=run_keys)["pipeline_elapsed_s"]
            else:
                run_times = grp["pipeline_elapsed_s"]
            mean_time = run_times.mean()
            total_time = run_times.sum()

            rows.append({
                "algorithm": alg,
                "noise_density": nd,
                "completeness": completeness,
                "purity": purity,
                "mean_wall_time_s": mean_time,
                "total_wall_time_s": total_time,
                "n_findable": n_findable,
                "n_found": n_found if n_findable > 0 else 0,
                "n_runs": len(run_times),
            })

    agg = __import__("pandas").DataFrame(rows)

    # ── Color palette ────────────────────────────────────────────────
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    alg_colors = {alg: colors[i % len(colors)] for i, alg in enumerate(algorithms)}

    # ── Plot 1: Completeness vs noise density ────────────────────────
    fig_comp = go.Figure()
    for alg in algorithms:
        d = agg[agg["algorithm"] == alg].sort_values("noise_density")
        fig_comp.add_trace(go.Scatter(
            x=d["noise_density"],
            y=d["completeness"],
            mode="lines+markers",
            name=alg,
            line=dict(color=alg_colors[alg], width=2),
            marker=dict(size=8),
        ))
    fig_comp.update_layout(
        title="Completeness vs Noise Density by Algorithm",
        xaxis_title="Noise Density (per exposure)",
        yaxis_title="Completeness (found / findable)",
        yaxis_range=[0, 1.05],
        template="plotly_white",
        legend=dict(title="Algorithm"),
    )
    fig_comp.write_html(str(output_dir / "completeness_vs_noise.html"))

    # ── Plot 2: Purity vs noise density ──────────────────────────────
    fig_pur = go.Figure()
    for alg in algorithms:
        d = agg[agg["algorithm"] == alg].sort_values("noise_density")
        fig_pur.add_trace(go.Scatter(
            x=d["noise_density"],
            y=d["purity"],
            mode="lines+markers",
            name=alg,
            line=dict(color=alg_colors[alg], width=2),
            marker=dict(size=8),
        ))
    fig_pur.update_layout(
        title="Purity vs Noise Density by Algorithm",
        xaxis_title="Noise Density (per exposure)",
        yaxis_title="Purity (pure / total clusters)",
        yaxis_range=[0, 1.05],
        template="plotly_white",
        legend=dict(title="Algorithm"),
    )
    fig_pur.write_html(str(output_dir / "purity_vs_noise.html"))

    # ── Plot 3: Wall time vs noise density ───────────────────────────
    fig_time = go.Figure()
    for alg in algorithms:
        d = agg[agg["algorithm"] == alg].sort_values("noise_density")
        fig_time.add_trace(go.Scatter(
            x=d["noise_density"],
            y=d["mean_wall_time_s"],
            mode="lines+markers",
            name=alg,
            line=dict(color=alg_colors[alg], width=2),
            marker=dict(size=8),
        ))
    fig_time.update_layout(
        title="Mean Wall Time vs Noise Density by Algorithm",
        xaxis_title="Noise Density (per exposure)",
        yaxis_title="Mean Wall Time (seconds)",
        template="plotly_white",
        legend=dict(title="Algorithm"),
    )
    fig_time.write_html(str(output_dir / "wall_time_vs_noise.html"))

    # ── Summary table ────────────────────────────────────────────────
    agg.to_csv(str(output_dir / "algorithm_comparison_summary.csv"), index=False)

    print(f"\n  Plots saved to {output_dir}/:")
    print("    - completeness_vs_noise.html")
    print("    - purity_vs_noise.html")
    print("    - wall_time_vs_noise.html")
    print("    - algorithm_comparison_summary.csv")

    return agg


if __name__ == "__main__":

    import argparse
    import subprocess
    import sys
    from pathlib import Path

    import pandas as pd

    ALL_ALGORITHMS = ["dbscan", "hotspot_2d", "optics", "kdtree", "fft", "hough"]

    parser = argparse.ArgumentParser(
        description="Run the clustering benchmark for multiple algorithms and merge results.",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=ALL_ALGORITHMS,
        help=f"Algorithms to benchmark (default: all). Choices: {ALL_ALGORITHMS}",
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default="experiments/benchmark",
        help="Path to benchmark directory created by generate_benchmark.py.",
    )
    parser.add_argument(
        "--plots_only",
        action="store_true",
        help="Skip running benchmarks; just merge existing summaries and generate plots.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip plot generation (just run benchmarks and merge).",
    )
    parser.add_argument("--max_processes", type=int, default=24)
    parser.add_argument(
        "--object_ids", type=str, nargs="+", default=None,
        help="Run only these object IDs (forwarded to run_benchmark.py).",
    )
    parser.add_argument(
        "--dynamical_classes", type=str, nargs="+", default=None,
        help="Run only objects in these dynamical classes.",
    )
    parser.add_argument(
        "--linking_windows", type=int, nargs="+", default=None,
        help="Run only these linking windows.",
    )
    parser.add_argument(
        "--cadences", type=str, nargs="+", default=None,
        help="Run only these cadences.",
    )
    parser.add_argument(
        "--noise_densities", type=int, nargs="+", default=None,
        help="Run only these noise densities.",
    )

    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    script = Path(__file__).parent / "run_benchmark.py"

    if not script.exists():
        sys.exit(f"ERROR: {script} not found.")

    # Build the common args to forward to run_benchmark.py
    forward_args = ["--max_processes", str(args.max_processes)]
    if args.object_ids:
        forward_args += ["--object_ids"] + args.object_ids
    if args.dynamical_classes:
        forward_args += ["--dynamical_classes"] + args.dynamical_classes
    if args.linking_windows:
        forward_args += ["--linking_windows"] + [str(w) for w in args.linking_windows]
    if args.cadences:
        forward_args += ["--cadences"] + args.cadences
    if args.noise_densities:
        forward_args += ["--noise_densities"] + [str(n) for n in args.noise_densities]

    # ── Run each algorithm ─────────────────────────────────────────────
    failed = []
    if not args.plots_only:
        for alg in args.algorithms:
            print("\n" + "=" * 70)
            print(f"  ALGORITHM: {alg}")
            print("=" * 70 + "\n")

            cmd = [
                sys.executable,
                str(script),
                "--benchmark_dir", str(benchmark_dir),
                "--cluster_algorithm", alg,
            ] + forward_args

            print(f"  Command: {' '.join(cmd)}\n")
            result = subprocess.run(cmd)

            if result.returncode != 0:
                print(f"\n  WARNING: {alg} exited with code {result.returncode}")
                failed.append(alg)
    else:
        print("\n  --plots_only: skipping benchmark runs.")

    # ── Merge per-algorithm summaries ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  Merging results")
    print("=" * 70)

    dfs = []
    for alg in args.algorithms:
        summary_path = benchmark_dir / f"summary_{alg}.parquet"
        if summary_path.exists():
            df = pd.read_parquet(summary_path)
            # Ensure algorithm column exists (for backward compat)
            if "cluster_algorithm" not in df.columns:
                df["cluster_algorithm"] = alg
            dfs.append(df)
            print(f"  {alg}: {len(df)} rows")
        else:
            print(f"  {alg}: no summary found")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        out_parquet = benchmark_dir / "summary_comparison.parquet"
        out_csv = benchmark_dir / "summary_comparison.csv"
        combined.to_parquet(str(out_parquet), index=False)
        combined.to_csv(str(out_csv), index=False)
        print(f"\n  Combined: {len(combined)} rows")
        print(f"  Saved to: {out_parquet}")
        print(f"  Saved to: {out_csv}")

        # ── Quick comparison table ─────────────────────────────────────
        if "found_pure_clusters" in combined.columns:
            print("\n  ── Algorithm comparison (mean found_pure_clusters) ──")
            pivot = combined.groupby(
                ["cluster_algorithm", "noise_density"]
            ).agg({
                "found_pure_clusters": "mean",
                "pipeline_elapsed_s": "mean",
            }).round(3)
            print(pivot.to_string())

        # ── Generate comparison plots ────────────────────────────────
        required_cols = {"findable", "found_pure_clusters", "pure_clusters",
                         "contaminated_clusters", "pipeline_elapsed_s"}
        if not args.no_plots and required_cols.issubset(combined.columns):
            print("\n  Generating comparison plots...")
            plots_dir = benchmark_dir / "plots"
            # Drop rows without analysis results before plotting
            plot_df = combined.dropna(subset=["findable", "pure_clusters"])
            if len(plot_df) > 0:
                generate_comparison_plots(plot_df, plots_dir)
            else:
                print("  No rows with complete analysis data; skipping plots.")
        elif not args.no_plots:
            missing = required_cols - set(combined.columns)
            print(f"\n  Skipping plots (missing columns: {missing}).")
    else:
        print("\n  No results to merge.")

    if failed:
        print(f"\n  WARNING: The following algorithms had errors: {failed}")

    print("\n" + "=" * 70)
    print("  Algorithm comparison complete.")
    print("=" * 70)
