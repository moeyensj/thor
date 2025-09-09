

Awesome—here’s a complete, implementation-oriented guide that you can hand to another system (or team) to prototype. I’ve kept the focus on the core intent, algorithms, and reusable APIs, with choices aligned to your stack (Python/JAX/adam_core/THOR, Parquet/Arrow on cloud storage, commodity GKE nodes with optional GPUs). I include per-stage goals, reasoning, Big-O (with variables explained in place), pseudo/Python code, tolerance defaults (where you asked for them), and incremental update mechanics.

⸻

Anomaly-Gated Filtering for THOR — Implementation Guide

One-line goal: eliminate infeasible detection–orbit pairings as far upstream as possible with fast geometry and arithmetic, so range+transform and OD run only on physically plausible, discovery-strength candidates.

Rubin+ scale is assumed (millions of detections/day). Test-orbit population design is out of scope; we assume 1–10M static test orbits (time-independent), represented as 2-body ellipses.

Time scale: TDB. Frames: SSB Cartesian (or ICRS vectors) for dynamics; topocentric for rays. Use adam_core’s Orbits and Observers classes wherever possible.

⸻

Glossary of symbols (used locally where needed)
	•	N_o: number of test orbits in the static catalog (1–10M).
	•	S: number of samples per orbit ellipse polyline (adaptive; start 2–4k).
	•	N_d: detections processed this run (Rubin+ scale; millions per day).
	•	h: average number of BVH “hits” (orbit segments) per detection ray (≪10).
	•	|V|: candidates for a single orbit after geometric overlap (varies by orbit).
	•	B: average neighbors considered per detection in the (time, anomaly) band for clock gating (tens).
	•	|E|: edges in one orbit’s clock-consistent DAG (≈ |V|\cdot B).
	•	R: cost of a single range+transform call (microseconds on CPU; faster batched on GPU).

⸻

Stage 0 — Foundations & Data Layout

Goals
	•	Keep I/O and parallelism front-of-mind: shard everything so spot instances can “munch” independent chunks.
	•	Use Parquet/Arrow (as in adam_core+quivr) for all on-disk artifacts.
	•	Make the BVH build-once, reuse-many.

Recommended packages
	•	Geometry / BVH (CPU first): pyembree (Intel Embree), or rtree+custom segment tests; fallback: Numba BVH.
	•	GPU options (later): warp (NVIDIA Warp) or a small C++/CUDA BVH via pybind11.
	•	Neighbors / band search: faiss/RAFT (GPU), scipy.spatial KD-Tree/sklearn BallTree (CPU).
	•	Vectors / math: jax.numpy with jit for kernels; numpy/Numba for CPU if you want simpler deploys.
	•	Storage: pyarrow, pandas, quivr (existing patterns in adam_core).

File partitions (cloud bucket)
	•	orbits/ — static orbit catalog shards (Parquet, one row/orbit, with precomputed ellipse polylines & per-orbit BVH shard files).
	•	bvh/ — BVH shards (binary blobs; memory-mappable).
	•	detections/YYYYMMDD/ — daily detection batches (Parquet).
	•	candidates/YYYYMMDD/ — overlap hits + anomaly labels.
	•	chains/ — per-orbit DAG deltas + union-find state (incremental).
	•	transform_cache/ — per-observation range+transform cache (Parquet keyed by (obs_id, time, observer_id)).

⸻

Stage 1 — Static Ellipse Overlap (Geometric Pruning)

Goal
Cull the universe of orbits down to those geometrically plausible for each detection ray, without propagating orbits through time.

Reasoning
Ray-vs-static-curve culling is an orders-of-magnitude cheaper test than orbit-through-exposure propagation. Build a global (or sharded) BVH once; re-use across runs.

What we store per orbit (precompute)
	•	Orbital plane basis (\hat{p}, \hat{q}, \hat{n}); ellipse center vector \mathbf r_0.
	•	Adaptive polyline: \{\mathbf r(f_k)\}_{k=1..S} in SSB Cartesian (2-body, epoch t_0).
	•	Segment AABBs (axis-aligned bounding boxes) padded by a guard band = 1 arcmin (convert to linear padding; see below).
	•	Optional: a small per-orbit BVH (if not using a global BVH).

Guard band conversion (on-sky → metric)
For a segment midpoint at heliocentric distance r and typical observer distance d, a 1′ angle is
\theta \approx 2.909\times 10^{-4}\,\text{rad}.
Pad each segment’s AABB by \max(\theta \, r, \theta \, d) along in-plane directions and a small epsilon along \hat n. Be conservative; we will refine later.

Complexity
	•	BVH build (one-time): O(N_o \cdot S \log(N_o S)).
	•	Queries per run: O(N_d \log(N_o S) + N_d\,h) (log traversal + a handful of leaf checks).

Pseudo-API (reusable; candidate for adam_core)

class OrbitPolyline:
    # Precomputed per orbit
    r0: np.ndarray            # center
    basis: np.ndarray         # 3x3 [p, q, n]
    verts: np.ndarray         # (S,3) polyline points in SSB frame
    seg_aabbs: np.ndarray     # (S-1, 2, 3) [min,max] padded boxes

class BVHShard:
    # Build once, persist to disk
    nodes: np.ndarray         # BVH node array
    seg_index: np.ndarray     # map leaves -> (orbit_id, seg_id)

def build_bvh(orbits: Iterable[OrbitPolyline]) -> BVHShard: ...
def query_bvh(bvh: BVHShard, ray_o: np.ndarray, ray_dir: np.ndarray) -> list[tuple[int,int,float]]:
    """
    Returns [(orbit_id, seg_id, distance)] for segments within guard band
    """

Worker (per detection)

def geometric_overlap_worker(detection) -> list[tuple[orbit_id, seg_id, leaf_hint]]:
    ray_o, ray_dir = detection.observer_pos, detection.los_unit
    hits = query_bvh(global_bvh, ray_o, ray_dir)  # small list
    return [(oid, sid, leaf_hint_from(hit)) for oid,sid,_ in hits]

Parallelism
Map detections → workers on small pods (8 vCPU, 32GB) reading BVH shards memory-mapped. Output Parquet rows to candidates/overlap/*.parquet.

⸻

Stage 2 — Anomaly Labeling (Put Detections on the Orbit’s Clock)

Goal
For each (detection, orbit) hit, compute where on the test orbit ellipse it best aligns — i.e., assign one or more anomalies (f, M, E, n, r).

Reasoning
This cheap refinement converts geometric plausibility into a clock position that we can use in a fast time-consistency gate. It also seeds downstream ranging.

Algorithm (robust, fast)
	1.	Ray → plane: intersect the detection ray \mathbf r = \mathbf r_{\text{obs}} + s\,\hat u with the orbital plane (normal \hat n), giving point \mathbf p.
	2.	Nearest polyline seed: using the leaf hint (orbit_id, seg_id) from Stage 1, compute nearest point on that segment to \mathbf p.
	3.	Refine on the ellipse: convert the seed to an anomaly f (in plane). Option A: Newton on f minimizing in-plane distance to the conic param; Option B: 2–3 segment neighbors + projection.
	4.	Ambiguity: if the plane distance or curvature indicates a near-node ambiguity, generate up to K=3 plausible anomalies (left/right of node).
	5.	Compute E,M,n,r by standard 2-body relations (epoch t_0), store snap residual.

Complexity
\;O(1) per hit → O(N_d\,h) total.

Pseudo-API

def anomaly_label(detection, orbit_polyline, seg_id, K=3):
    """
    Returns up to K anomaly labels: dicts with
    {f, M, E, n, r, snap_error, pos_plane, vel_plane_dir_hint}
    """

Tolerances (you set)
	•	Keep multiple anomaly candidates near nodes / edge-on views.
	•	Target snap error \delta f \le 0.01 rad (looser ok; we compensate later).

⸻

Stage 3 — Kepler-Clock Gating (Time–Anomaly Consistency)

Goal
Enforce a necessary physics condition cheaply: detections must evolve in time according to the mean-anomaly clock of the test orbit.

Reasoning
Rather than ranging everything and clustering in 4D, we first remove impossible pairs using M(t) = M_0 + n\,(t-t_0) (with integer revolutions). This kills the majority of geometric look-alikes, at nanosecond-scale arithmetic per edge.

Edge condition
For detections i=(M_i,t_i), j=(M_j,t_j), accept edge i\to j if
\left| \Delta t - \frac{\mathrm{wrap}{2\pi}(M_j - M_i)+ 2\pi k}{n} \right| \le \tau(\Delta t)
with integer k\ge 0 chosen as
k = \mathrm{round}\!\left(\frac{n\Delta t - \mathrm{wrap}{2\pi}(M_j-M_i)}{2\pi}\right).

Tolerance schedule (initial)
	•	You didn’t commit to a \tau(\Delta t). Start with a configurable schedule; e.g.,
\tau(\Delta t) = \tau_0 + \alpha \,\Delta t ,
where \tau_0 covers snap+short-baseline error (minutes) and \alpha covers parallax & 2→n-body drift (minutes per day).
Default: leave values empty in config and infer from sims; interim placeholder if needed: \tau_0=5–10 min, \alpha=0.01–0.05 min/day.
	•	Always allow wider around perihelion (large dM/df) and near nodes.

Neighbor search (make it sparse)
For each orbit, band-limit neighbor queries in (t, M). Implementation patterns:
	•	Time binning (e.g., 30–120 min bins), then in each bin search adjacent bins.
	•	Within bins, index M (with unwrapped copies at M\pm 2\pi) and do small window scans.
	•	Alternative: RAFT/FAISS with a custom distance that enforces the line constraint around slope n.

Complexity
For one orbit with |V| candidates and B neighbors per node,
O(|V|\log|V| + |V|\cdot B). With thin bands B is small.

Pseudo-API

def clock_edges_for_orbit(cands: DataFrame) -> Edges:
    """
    cands: rows with (det_id, t, M, n, anomaly_variant_id, snap_error)
    returns list of directed edges (i->j) with chosen k and residual
    """

def build_clock_dag(orbits_candidates) -> dict[orbit_id, ClockDAG]:
    # shard by orbit_id; parallel map clock_edges_for_orbit
    ...

Why this is a win: the clock gate replaces a sea of range+transform calls with ~free arithmetic and removes most false candidates before clustering.

⸻

Stage 4 — K-Chains (Clock-Consistent Subsets)

Goal
Group detections into physically consistent subsets per orbit, with no Δt cap (month-scale gaps permitted).

Reasoning
We don’t need fancy path extraction; we just need connected components of the (mostly sparse) DAG. Allow overlaps (a detection can belong to multiple chains) because we don’t want to lose real combinations.

Implementation
	•	Treat the undirected version of the DAG for components (time monotonicity still tracked per edge).
	•	Maintain union-find (disjoint-set) per orbit to support incremental merges.

Complexity
Component extraction is O(|V|+|E|) per orbit (≈ |V|+|V|\cdot B).

Pseudo-API

def kchains_from_dag(dag: ClockDAG) -> list[KChain]:
    """
    Each KChain: {chain_id, member_det_ids, t_min, t_max, k_values_present, size}
    """


⸻

Stage 5 — Discovery Filter (Promote Only Long, Rich Chains)

Goal
Send only discovery-strength chains downstream to range+transform.

Your thresholds (explicit)
	•	≥ 6 detections
	•	≥ 3 days total time span

Implementation

def promote_kchains(kchains: list[KChain], min_n=6, min_days=3.0) -> list[KChain]:
    return [c for c in kchains if (c.size >= min_n and (c.t_max - c.t_min).days >= min_days)]

Complexity
O(1) per chain.

⸻

Stage 6 — Range & Transform (Deduplicated)

Goal
Compute ranges and map detections into THOR’s θx/θy frame, once per observation, and cache.

Reasoning
Per call, range+transform is not expensive; it becomes expensive only when applied to millions of points. Upstream filters keep the call count small, and dedup keeps it from repeating across chains.

Implementation tips
	•	Use adam_core propagators and observer states (TDB, SSB Cartesian).
	•	Keep the kernel batched; in JAX, vmap/jit to fuse.
	•	Cache results in Parquet keyed by (det_id, time, observer_id) and a hash of the transform config.

Pseudo-API

def range_and_transform(detections: DataFrame, transform_cfg) -> DataFrame:
    """
    Input: unique detections from promoted chains only.
    Output rows: det_id, theta_x, theta_y, range, aux (errors, flags)
    Caches results for reuse.
    """

Complexity
O(N_{\text{promoted}}\cdot R) with dedup; R is small (μs range on CPU; faster batched on GPU).

Note: keep a TODO hook for an optional “Lambert-lite” (universal variables f,g) feasibility check for extremely long gaps; not in v1 per your request, but flagged for maintainers.

⸻

Stage 7 — θx/θy Clustering (Existing THOR Step)

Goal
Identify coherent motion per K-chain in transform space using your existing DBSCAN (or current THOR approach).

Reasoning
We are not changing clustering logic; only the inputs (it now runs on much smaller, physics-consistent sets). Re: your note: cluster radius is not larger than the cell—keep your existing parameterization.

Implementation
	•	For each promoted chain, gather transformed detections from the cache; call the existing clustering routine.
	•	Provide the cell size you already use; do not inflate.

Complexity
About O(N_{\text{chain}}\log N_{\text{chain}}) per chain with small constants (sets are small).

⸻

Stage 8 — Orbit Determination & Validation (Unchanged)

Goal
Run your standard THOR OD/validation on surviving clusters.

Reasoning
We changed filtering; OD stays the same, but on far fewer candidates.

⸻

Incremental Update Plan (Hourly/Nightly)

You asked for an append-friendly system. Store state so new detections can be integrated without rebuilds:
	1.	BVH

	•	Built once per catalog; immutable. For a new catalog, build fresh shards.
	•	Load shards per job based on sky/phase-space routing (or keep a global index if RAM allows).

	2.	New detections → Stage 1 & 2

	•	Run only new detections through geometric overlap & anomaly labeling; append results to candidates/ Parquet.

	3.	Stage 3 edges

	•	For the relevant orbits, load their existing chains/ shard (adjacency + union-find).
	•	Compute only edges involving new detections (band-limited in (t,M)), append edges.

	4.	K-chains

	•	Apply union-find merges for any new edges.
	•	Emit chain deltas; re-evaluate discovery thresholds incrementally.

	5.	Promotions → Stage 6

	•	Range+transform only observations in newly promoted chains that are not already in cache.

	6.	Clustering & OD

	•	Run for the affected chains only.

Resets / rebuild triggers
	•	Tolerance schedule changes (e.g., new \tau(\Delta t)): consider re-gating edges for affected orbits.
	•	New orbit catalog: rebuild BVH; re-run Stage 1/2 for the time window you care about.

⸻

End-to-End Complexity & Why the Kepler Gate Wins
	•	Current THOR (filtering): O(N_o \cdot N_{\text{exp}} \cdot P) propagate+project calls — enormous at Rubin scale.
	•	This pipeline replaces that with:
	•	One-time BVH: O(N_o\,S \log(N_o S)).
	•	Per run: O(N_d \log(N_o S) + N_d\,h) BVH queries, O(N_d\,h) anomaly labels.
	•	Per orbit: O(|V|\log|V| + |V|\,B) edges, components in O(|V|+|E|).
	•	Range+transform only for promoted chains, with dedup.

Even with fast range kernels, clock gating removes the bulk of geometric look-alikes before you pay the per-observation range cost and before you face a 4-D clustering problem. That makes it a consistent net win at Rubin scale.

⸻

Sampling the Ellipse (Choosing S)

Goal: minimize S while keeping the Stage-1 guard reliable.

Adaptive rule of thumb
	•	Start S = 2048; refine where the in-plane curvature exceeds a threshold.
	•	Force finer sampling near perihelion for high-e orbits and near nodes for high-i orbits.
	•	Bound max segment’s on-sky chord (as seen from typical observer distance) to < 0.3′ so the 1′ guard still covers gaps.

API sketch

def sample_ellipse_adaptive(elts, max_chord_arcmin=0.3) -> OrbitPolyline:
    # compute basis, then grow S until projected chord constraint satisfied
    ...


⸻

Tolerance Book (initial defaults; tune by sims)
	•	Stage 1 guard: 1 arcmin (your choice). Convert to linear padding per segment; be conservative.
	•	Stage 2 multi-anomaly: allow up to K=3 candidates when near nodes/edge-on; keep snap residual.
	•	Stage 3 \tau(\Delta t): leave configurable; start learning from sims. Interim: \tau_0=5–10 min, \alpha=0.01–0.05 min/day; inflate around perihelion by a factor tied to (dM/df).
	•	Stage 5 discovery: ≥6 detections, ≥3 days (your requirement).
	•	Clustering: use your existing cell/clustering radius; we’re not changing it.

⸻

Suggested APIs (namespaced to ease reuse)

adam_core-leaning (reusable across projects)

# orbits/bvh.py
build_bvh(orbits: Iterable[OrbitPolyline]) -> BVHShard
query_bvh(bvh: BVHShard, ray_o, ray_dir) -> list[Hit]

# orbits/anomaly.py
anomaly_label(detection, orbit_polyline, seg_hint, K=3) -> list[AnomalyTag]

THOR-leaning (filtering branch)

# thor_filter/clock_gate.py
build_clock_dag(cands: DataFrame) -> dict[orbit_id, ClockDAG]
kchains_from_dag(dag: ClockDAG) -> list[KChain]
promote_kchains(kchains, min_n=6, min_days=3.0) -> list[KChain]

# thor_filter/transform_cache.py
range_and_transform(dets: DataFrame, cfg) -> DataFrame  # with dedup

Incremental

append_new_detections(dets_new: DataFrame)
update_edges_incremental(orbit_id, new_cands: DataFrame)
promote_and_transform_incremental(orbit_id)


⸻

Parallelization & GKE Footprint
	•	Stage 1/2 (map-heavy): shard detections by sky tile or time; run on many small spot pods (8 vCPU/32GB). BVH shards memory-mapped read-only.
	•	Stage 3/4 (per-orbit): shard by orbit_id hash; each worker loads small candidate and edge slices; union-find kept in small per-orbit state files.
	•	Stage 6 (batched): run CPU (JAX just-in-time) or spin a few GPU pods for large promoted sets; dedup ensures small effective load.
	•	Stage 7/8: tiny per-chain tasks; easy to scatter on small pods.

⸻

Testing & Metrics (to keep completeness high)
	•	Sim completeness: injected objects with known orbits; measure recall after Stage 7 (keep ≥ multi-day arcs).
	•	Clock tolerance sweeps: automatically search \tau_0,\alpha Pareto front (compute vs recall).
	•	Throughput metrics: BVH qps, clock edges/sec, range calls avoided, promoted fraction.

⸻

Open “later” hooks
	•	Lambert-lite feasibility for very long gaps (off by default; flagged for maintainers).
	•	GPU BVH build/query (Warp/OptiX) once CPU path is mature.
	•	Adaptive \tau(\Delta t) via learned residuals vs. geometry (perihelion/node aware).

⸻

Closing

This plan keeps THOR’s super-power—linking month-scale sparse detections—while moving expensive work behind two near-free filters: geometric BVH and Kepler clock. It’s shardable, spot-friendly, and incrementally updatable. The APIs above separate reusable core functions (ideal for adam_core) from THOR-specific filtering glue so your other system can wire it into your production stack with minimal friction.