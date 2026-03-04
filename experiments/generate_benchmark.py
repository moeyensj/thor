if __name__ == "__main__":

    import argparse
    import gc
    import json
    from pathlib import Path

    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import quivr as qv
    from adam_assist import ASSISTPropagator
    from adam_core.coordinates import (
        CoordinateCovariances,
        Origin,
        SphericalCoordinates,
    )
    from adam_core.observers import Observers
    from adam_core.orbits.orbits import Orbits
    from adam_core.orbits.query import query_sbdb_new
    from adam_core.orbits.variants import VariantOrbits
    from adam_core.time import Timestamp

    from thor.observations import Observations
    from thor.observations.photometry import Photometry
    from thor.observations.states import calculate_state_id_hashes

    # ── Object catalog ────────────────────────────────────────────────
    # All non-perturber objects from adam_core/utils/helpers/data/objects.csv
    # Excludes ASSIST perturbers: 434 (Hungaria), 2 (Pallas), 6 (Hebe)
    OBJECT_CATALOG = {
        "2020 AV2": "Atira",
        "163693": "Atira",
        "2010 TK7": "Aten",
        "3753": "Aten",
        "54509": "Apollo",
        "2063": "Apollo",
        "1221": "Amor",
        "433": "Amor",
        "3908": "Amor",
        "1876": "Inner Main Belt",
        "2001": "Inner Main Belt",
        "6522": "Main Belt",
        "10297": "Main Belt",
        "17032": "Main Belt",
        "202930": "Main Belt",
        "911": "Jupiter Trojan",
        "1143": "Jupiter Trojan",
        "1172": "Jupiter Trojan",
        "3317": "Jupiter Trojan",
        "5145": "Centaur",
        "5335": "Centaur",
        "15760": "Trans-Neptunian Object",
        "15788": "Trans-Neptunian Object",
        "15789": "Trans-Neptunian Object",
        "A/2017 U1": "Interstellar Object",
    }

    CADENCE_MAP = {
        "singletons": 1,
        "pairs": 2,
        "triplets": 3,
        "quads": 4,
    }

    ALL_DYNAMICAL_CLASSES = sorted(set(OBJECT_CATALOG.values()))

    def safe_filename(object_id):
        """Sanitize object ID for use as a filename/directory name."""
        return object_id.replace("/", "_").replace(" ", "_")

    # ── Argument parsing ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Generate benchmark observation datasets (signal + noise). "
        "Does NOT run the clustering pipeline. Use run_benchmark.py for that.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiments/benchmark",
    )
    parser.add_argument("--start_mjd", type=float, default=60999.25)
    parser.add_argument("--observatory_code", type=str, default="X05")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--astrometric_sigma_arcsec", type=float, default=0.1)
    parser.add_argument("--obs_separation_minutes", type=float, default=30.0)
    parser.add_argument("--cov_scale", type=float, default=1e6)
    parser.add_argument("--default_mag", type=float, default=22.0)
    parser.add_argument("--default_mag_sigma", type=float, default=0.1)
    parser.add_argument("--default_filter", type=str, default="r")
    parser.add_argument(
        "--object_ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific object IDs to run (default: all 25 non-perturber objects).",
    )
    parser.add_argument(
        "--dynamical_classes",
        type=str,
        nargs="+",
        default=None,
        help="Filter by dynamical class (e.g. Amor 'Main Belt').",
    )
    parser.add_argument(
        "--linking_windows",
        type=int,
        nargs="+",
        default=[15, 30, 60, 90],
        help="Linking window sizes in nights.",
    )
    parser.add_argument(
        "--cadences",
        type=str,
        nargs="+",
        default=["singletons", "pairs", "triplets", "quads"],
        help="Cadence names: singletons, pairs, triplets, quads.",
    )
    parser.add_argument(
        "--noise_densities",
        type=int,
        nargs="+",
        default=[0, 10, 50, 100, 500, 1000],
        help="Number of noise observations per exposure.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Resolve object list ───────────────────────────────────────────
    if args.object_ids is not None:
        object_ids = args.object_ids
        for oid in object_ids:
            if oid not in OBJECT_CATALOG:
                parser.error(f"Unknown object_id: {oid}. Valid: {list(OBJECT_CATALOG.keys())}")
    elif args.dynamical_classes is not None:
        object_ids = [
            oid for oid, cls in OBJECT_CATALOG.items()
            if cls in args.dynamical_classes
        ]
        if not object_ids:
            parser.error(
                f"No objects match dynamical_classes={args.dynamical_classes}. "
                f"Valid: {ALL_DYNAMICAL_CLASSES}"
            )
    else:
        object_ids = list(OBJECT_CATALOG.keys())

    for c in args.cadences:
        if c not in CADENCE_MAP:
            parser.error(f"Unknown cadence: {c}. Valid: {list(CADENCE_MAP.keys())}")

    # ── Output directory ──────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    orbits_dir = output_dir / "orbits"
    orbits_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Save experiment configuration ─────────────────────────────────
    experiment_config = {
        "object_ids": object_ids,
        "linking_windows": args.linking_windows,
        "cadences": args.cadences,
        "noise_densities": args.noise_densities,
        "start_mjd": args.start_mjd,
        "observatory_code": args.observatory_code,
        "n_samples": args.n_samples,
        "obs_separation_minutes": args.obs_separation_minutes,
        "cov_scale": args.cov_scale,
        "astrometric_sigma_arcsec": args.astrometric_sigma_arcsec,
        "default_mag": args.default_mag,
        "default_mag_sigma": args.default_mag_sigma,
        "default_filter": args.default_filter,
        "seed": args.seed,
    }
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)

    propagator = ASSISTPropagator()
    astro_sigma_deg = args.astrometric_sigma_arcsec / 3600.0

    n_objects = len(object_ids)
    n_windows = len(args.linking_windows)
    n_cadences = len(args.cadences)
    n_noise = len(args.noise_densities)
    n_signal_combos = n_objects * n_windows * n_cadences
    n_total_cases = n_signal_combos * n_noise

    print("=" * 70)
    print("Clustering Benchmark Data Generator")
    print("=" * 70)
    print(f"  Objects          : {n_objects}")
    print(f"  Linking windows  : {args.linking_windows}")
    print(f"  Cadences         : {args.cadences}")
    print(f"  Noise densities  : {args.noise_densities}")
    print(f"  Samples/object   : {args.n_samples}")
    print(f"  Signal combos    : {n_signal_combos}")
    print(f"  Total cases      : {n_total_cases}")
    print(f"  Output dir       : {output_dir}")
    print("=" * 70)

    # ── Step 1: Fetch orbits from SBDB ────────────────────────────────
    print("\n[Step 1] Fetching orbits from SBDB...")

    cached_orbits = {}
    uncached_ids = []
    for oid in object_ids:
        orbit_path = orbits_dir / f"{safe_filename(oid)}.parquet"
        if orbit_path.exists():
            cached_orbits[oid] = Orbits.from_parquet(str(orbit_path))
            print(f"  Cached: {oid}")
        else:
            uncached_ids.append(oid)

    if uncached_ids:
        print(f"  Querying SBDB for {len(uncached_ids)} objects: {uncached_ids}")
        fetched_orbits = query_sbdb_new(
            uncached_ids,
            orbit_id_from_input=True,
            allow_missing=True,
        )
        for oid in uncached_ids:
            mask = pc.equal(fetched_orbits.orbit_id, pa.scalar(oid, pa.large_string()))
            orbit_i = fetched_orbits.apply_mask(mask)
            if len(orbit_i) > 0:
                orbit_i.to_parquet(str(orbits_dir / f"{safe_filename(oid)}.parquet"))
                cached_orbits[oid] = orbit_i
                print(f"  Fetched: {oid}")
            else:
                print(f"  WARNING: {oid} not found in SBDB, skipping.")

    orbit_dict = cached_orbits
    active_object_ids = [oid for oid in object_ids if oid in orbit_dict]
    print(f"\n  {len(active_object_ids)} orbits available.")

    # ── Step 2: Generate data ─────────────────────────────────────────
    print("\n[Step 2] Generating observation datasets...")

    generated = 0
    skipped = 0

    for obj_idx, object_id in enumerate(active_object_ids):
        orbit = orbit_dict[object_id]
        dynamical_class = OBJECT_CATALOG[object_id]

        print(f"\n{'=' * 70}")
        print(f"[Object {obj_idx + 1}/{len(active_object_ids)}] "
              f"{object_id} ({dynamical_class})")
        print(f"{'=' * 70}")

        obj_data_dir = data_dir / safe_filename(object_id)
        obj_data_dir.mkdir(parents=True, exist_ok=True)

        # Inflate covariance for variant generation
        orbit_cov = orbit.coordinates.covariance.to_matrix()
        inflated_cov = orbit_cov * args.cov_scale
        orbit_inflated = Orbits.from_kwargs(
            orbit_id=orbit.orbit_id,
            object_id=orbit.object_id,
            coordinates=orbit.coordinates.set_column(
                "covariance", CoordinateCovariances.from_matrix(inflated_cov)
            ),
        )

        # Generate variant orbits (sample objects) — shared across all combos
        sample_orbits_path = obj_data_dir / "sample_orbits.parquet"
        if sample_orbits_path.exists():
            sample_orbits = Orbits.from_parquet(str(sample_orbits_path))
            print(f"  Sample orbits already exist ({len(sample_orbits)} variants)")
        else:
            print(f"  Drawing {args.n_samples} variant samples (cov_scale={args.cov_scale})...")
            variants = VariantOrbits.create(
                orbit_inflated, method="monte-carlo", num_samples=args.n_samples
            )
            sample_orbits = Orbits.from_kwargs(
                orbit_id=pc.binary_join_element_wise(
                    variants.orbit_id,
                    pc.cast(pa.scalar("_"), pa.large_string()),
                    variants.variant_id,
                ),
                object_id=variants.object_id,
                coordinates=variants.coordinates,
            )
            sample_orbits.to_parquet(str(sample_orbits_path))
            del variants

        # Reference sky position for noise generation
        ref_timestamp = Timestamp.from_mjd([args.start_mjd], scale="utc")
        ref_observers = Observers.from_code(args.observatory_code, ref_timestamp)
        ref_ephemeris = propagator.generate_ephemeris(orbit, ref_observers, max_processes=1)
        ref_ra = ref_ephemeris.coordinates.lon[0].as_py()
        ref_dec = ref_ephemeris.coordinates.lat[0].as_py()
        del ref_ephemeris
        print(f"  Reference sky position: RA={ref_ra:.4f}, Dec={ref_dec:.4f}")

        for n_nights in args.linking_windows:
            for cadence_name in args.cadences:
                n_obs_per_night = CADENCE_MAP[cadence_name]
                combo_label = f"w{n_nights:03d}_{cadence_name}"
                combo_dir = obj_data_dir / combo_label
                combo_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n  {'─' * 66}")
                print(f"  [{object_id}] window={n_nights}, cadence={cadence_name} "
                      f"({n_obs_per_night} obs/night)")

                # ── Build observation times ───────────────────────
                obs_times_mjd = []
                obs_nights = []
                obs_exposure_ids = []
                for night_idx in range(n_nights):
                    base_mjd = args.start_mjd + night_idx
                    for obs_idx in range(n_obs_per_night):
                        t = base_mjd + obs_idx * args.obs_separation_minutes / (60.0 * 24.0)
                        obs_times_mjd.append(t)
                        obs_nights.append(night_idx)
                        obs_exposure_ids.append(f"exp_n{night_idx:04d}_o{obs_idx:02d}")

                obs_times_mjd = np.array(obs_times_mjd)
                n_exposures = len(obs_times_mjd)
                obs_timestamp = Timestamp.from_mjd(obs_times_mjd, scale="utc")
                observers = Observers.from_code(args.observatory_code, obs_timestamp)

                # ── Generate signal observations (skip if exists) ─
                signal_obs_path = combo_dir / "signal_observations.parquet"
                labels_path = combo_dir / "labels.parquet"

                if signal_obs_path.exists() and labels_path.exists():
                    print(f"    Signal observations already exist, skipping ephemeris.")
                    combined_signal = Observations.from_parquet(str(signal_obs_path))
                    from thor.analysis import ObservationLabels
                    labels = ObservationLabels.from_parquet(str(labels_path))
                    all_label_obs_ids = labels.obs_id.to_pylist()
                    all_label_object_ids = labels.object_id.to_pylist()
                    n_total_signal = len(combined_signal)
                else:
                    all_signal_observations = []
                    all_label_obs_ids = []
                    all_label_object_ids = []

                    for sample_idx in range(len(sample_orbits)):
                        sample_orbit = sample_orbits[sample_idx]
                        sample_orbit_id = sample_orbit.orbit_id[0].as_py()

                        ephemeris = propagator.generate_ephemeris(
                            sample_orbit, observers, max_processes=1
                        )

                        n_signal = len(ephemeris)
                        sigmas = np.full((n_signal, 6), np.nan)
                        sigmas[:, 1] = astro_sigma_deg
                        sigmas[:, 2] = astro_sigma_deg

                        # Add astrometric jitter to signal positions
                        lon_arr = ephemeris.coordinates.lon.to_numpy(zero_copy_only=False)
                        lat_arr = ephemeris.coordinates.lat.to_numpy(zero_copy_only=False)
                        lat_jitter = rng.normal(0, astro_sigma_deg, n_signal)
                        lon_jitter = rng.normal(0, astro_sigma_deg, n_signal) / np.cos(
                            np.radians(lat_arr)
                        )
                        lon_arr = lon_arr + lon_jitter
                        lat_arr = lat_arr + lat_jitter

                        signal_coords = SphericalCoordinates.from_kwargs(
                            lon=pa.array(lon_arr),
                            lat=pa.array(lat_arr),
                            time=ephemeris.coordinates.time,
                            covariance=CoordinateCovariances.from_sigmas(sigmas),
                            origin=ephemeris.coordinates.origin,
                            frame=ephemeris.coordinates.frame,
                        )
                        signal_state_ids = calculate_state_id_hashes(signal_coords)

                        signal_ids = pa.array(
                            [f"signal_{sample_orbit_id}_{i:06d}" for i in range(n_signal)],
                            pa.large_string(),
                        )
                        signal_observations = Observations.from_kwargs(
                            id=signal_ids,
                            exposure_id=pa.array(obs_exposure_ids, pa.large_string()),
                            night=pa.array(obs_nights, pa.int64()),
                            coordinates=signal_coords,
                            photometry=Photometry.from_kwargs(
                                mag=pa.repeat(args.default_mag, n_signal),
                                mag_sigma=pa.repeat(args.default_mag_sigma, n_signal),
                                filter=pa.repeat(
                                    pa.scalar(args.default_filter, pa.large_string()), n_signal
                                ),
                            ),
                            state_id=signal_state_ids,
                        )
                        all_signal_observations.append(signal_observations)
                        all_label_obs_ids.extend(signal_ids.to_pylist())
                        all_label_object_ids.extend([sample_orbit_id] * n_signal)

                        del ephemeris
                        gc.collect()

                    combined_signal = qv.concatenate(all_signal_observations)
                    n_total_signal = len(combined_signal)
                    combined_signal.to_parquet(str(signal_obs_path))

                    # Save labels (signal only — noise labels added at combine time)
                    from thor.analysis import ObservationLabels
                    labels = ObservationLabels.from_kwargs(
                        obs_id=pa.array(all_label_obs_ids, pa.large_string()),
                        object_id=pa.array(all_label_object_ids, pa.large_string()),
                    )
                    labels.to_parquet(str(labels_path))

                    print(f"    Generated {n_total_signal} signal observations "
                          f"from {len(sample_orbits)} samples.")
                    del all_signal_observations
                    gc.collect()

                # ── Generate noise + combined observations per noise density ─
                for noise_density in args.noise_densities:
                    noise_label = f"noise_{noise_density:06d}"
                    noise_dir = combo_dir / noise_label
                    obs_out_path = noise_dir / "observations.parquet"
                    labels_out_path = noise_dir / "labels.parquet"

                    if obs_out_path.exists() and labels_out_path.exists():
                        skipped += 1
                        print(f"    [Noise {noise_density}] Already exists, skipping.")
                        continue

                    noise_dir.mkdir(parents=True, exist_ok=True)
                    generated += 1

                    if noise_density > 0:
                        n_noise_total = noise_density * n_exposures

                        noise_ra = rng.uniform(
                            ref_ra - 1.0, ref_ra + 1.0, n_noise_total
                        )
                        noise_dec = rng.uniform(
                            ref_dec - 1.0, ref_dec + 1.0, n_noise_total
                        )

                        noise_time_indices = np.repeat(np.arange(n_exposures), noise_density)
                        noise_times = Timestamp.from_mjd(
                            obs_times_mjd[noise_time_indices], scale="utc"
                        )
                        noise_nights_arr = np.array(obs_nights)[noise_time_indices]
                        noise_exposure_ids_arr = np.array(obs_exposure_ids)[noise_time_indices]

                        noise_sigmas = np.full((n_noise_total, 6), np.nan)
                        noise_sigmas[:, 1] = astro_sigma_deg
                        noise_sigmas[:, 2] = astro_sigma_deg

                        noise_coords = SphericalCoordinates.from_kwargs(
                            lon=pa.array(noise_ra),
                            lat=pa.array(noise_dec),
                            time=noise_times,
                            covariance=CoordinateCovariances.from_sigmas(noise_sigmas),
                            origin=Origin.from_kwargs(
                                code=pa.repeat(
                                    pa.scalar(args.observatory_code, pa.large_string()),
                                    n_noise_total,
                                )
                            ),
                            frame="equatorial",
                        )
                        noise_state_ids = calculate_state_id_hashes(noise_coords)

                        noise_ids = pa.array(
                            [f"noise_{i:08d}" for i in range(n_noise_total)],
                            pa.large_string(),
                        )
                        noise_observations = Observations.from_kwargs(
                            id=noise_ids,
                            exposure_id=pa.array(
                                noise_exposure_ids_arr.tolist(), pa.large_string()
                            ),
                            night=pa.array(noise_nights_arr.tolist(), pa.int64()),
                            coordinates=noise_coords,
                            photometry=Photometry.from_kwargs(
                                mag=pa.repeat(args.default_mag, n_noise_total),
                                mag_sigma=pa.repeat(args.default_mag_sigma, n_noise_total),
                                filter=pa.repeat(
                                    pa.scalar(args.default_filter, pa.large_string()),
                                    n_noise_total,
                                ),
                            ),
                            state_id=noise_state_ids,
                        )

                        combined_observations = qv.concatenate(
                            [combined_signal, noise_observations]
                        )
                        combined_observations = combined_observations.sort_by(
                            [
                                "coordinates.time.days",
                                "coordinates.time.nanos",
                                "coordinates.origin.code",
                            ]
                        )

                        combined_labels = ObservationLabels.from_kwargs(
                            obs_id=pa.concat_arrays([
                                pa.array(all_label_obs_ids, pa.large_string()),
                                noise_ids,
                            ]),
                            object_id=pa.concat_arrays([
                                pa.array(all_label_object_ids, pa.large_string()),
                                pa.nulls(n_noise_total, pa.large_string()),
                            ]),
                        )
                    else:
                        combined_observations = combined_signal
                        combined_labels = labels

                    combined_observations.to_parquet(str(obs_out_path))
                    combined_labels.to_parquet(str(labels_out_path))

                    n_total = len(combined_observations)
                    print(f"    [Noise {noise_density}] {n_total} obs "
                          f"({n_total_signal} signal + {n_total - n_total_signal} noise)")

                    if noise_density > 0:
                        del combined_observations, combined_labels
                    gc.collect()

                del combined_signal
                gc.collect()

        del sample_orbits
        gc.collect()

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Data generation complete.")
    print(f"  Generated: {generated} cases")
    print(f"  Skipped (already exist): {skipped} cases")
    print(f"  Output: {output_dir}")
    print()
    print("Next step: run the clustering pipeline with run_benchmark.py")
    print(f"  python experiments/run_benchmark.py --benchmark_dir {output_dir}")
    print("=" * 70)
