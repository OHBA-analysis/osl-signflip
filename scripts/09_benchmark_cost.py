"""Benchmark the computational cost (wallclock + FLOPs) of the four sign-flipping methods.

This is a *cost* benchmark, kept deliberately separate from ``01_sign_flip.py`` (which
produces the deployed flips via joblib + disk I/O). For a fair speed comparison we must:

  1. run **serially, single-threaded** (no Dask, BLAS threads pinned to 1) so the wallclock
     reflects the algorithm, not the scheduler;
  2. time only the ``find_flips_*`` solver core, with the covariances already in memory,
     so the shared ``.fif`` load/save I/O (identical across methods) does not swamp the
     signal;
  3. split the deterministic methods into solver-core vs greedy-refine time;
  4. report FLOPs from an analytic model, driven by the problem size (n, E, D) and
     the configured iteration budgets (pure-NumPy hardware FLOP counting is unreliable).
     
NOTE: Wallclock is the empirical ground truth and the FLOP column is an order-of-magnitude
      metric. See the function ``flop_model()`` for the exact formulas and assumptions.

The refine settings mirror ``01_sign_flip.py`` (all new methods share one strategy) such that
the timing is apples-to-apples with the deployed run.

Env: ``osld`` (with ``osl-signflip`` installed)
Usage
-----
    python 09_benchmark_cost.py <dataset> [--methods <methods>] [--n-sessions N] \
        [--n-repeats N] [--n-warmup N]
"""

# Import packages
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# NOTE: We pin thread counts before numpy is imported so that eigh / matmuls run
#       single-threaded and the wallclock is stable and comparable across methods.

import argparse
import numpy as np
import time
from glob import glob
import osl_signflip as sf
from osl_signflip import io as sf_io


# Set directories
BASE_DIR = "/well/woolrich/users/olt015/osl-signflip"

# Set hyperparameters
COMMON_PARAMS = {
    "n_embeddings": 15,
    "standardize": True,
}  # shared by every sign flipping method

REFINE_PARAMS = {
    "refine": True,
    "n_refine": 5,
    "refine_strategy": "best",
}  # greedy-refine settings shared by spectral, SDP, and BP

METHOD_PARAMS = {
    "random": {
        "n_restarts": 3,
        "n_iter": 5000,
        "max_flips": 20,
        "random_state": 0,
    },
    "spectral": {},
    "belief_propagation": {
        "bp_n_iter": 200,
        "beta": 2.0,
        "damping": 0.5,
        "prior_strength": 1e-3,
        "bp_tol": 1e-6,
    },
    "sdp": {
        "sdp_n_restarts": 5,
        "sdp_max_iter": 1000,
        "sdp_tol": 1.0e-8,
        "gw_samples": 500,
        "random_state": 0,
    },
}  # specific to each sign flipping method

ALL_METHODS = list(METHOD_PARAMS)


# Define helper functions
def time_function_call(fn, n_repeats, n_warmup):
    """
    Runs a given function ``n_warmup`` (untimed) + ``n_repeats`` (timed) times.

    Returns ``(result, median_seconds, all_times)``, where ``result`` is the return value
    of the last function call. Uses ``perf_counter`` and reports the median to be robust
    to jitter.
    """
    for _ in range(n_warmup):
        result = fn()
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, float(np.median(times)), np.asarray(times)


def time_method(method, cov, template_cov, n_embeddings, n_repeats, n_warmup):
    """
    Return {'core': ~ seconds, 'refine': ~ seconds, 'total': ~ seconds} median
    wallclock for one method and session.

    This function measures per-session, per-method wallclock time (solver core
    vs greedy refine).

    - ``random`` has no refine stage, so core == total and refine == 0.
    - The remaining methods are timed with ``refine=False`` (core), then the greedy
      polish is timed separately on the core's output (refine). total = core + refine,
      which reproduces the production cost (find_flips_*(refine=True) rebuilds the
      affinity inside _greedy_refine, exactly as core + refine does here).
    """
    E = n_embeddings
    params = METHOD_PARAMS[method]

    if method == "random":
        fn = lambda: sf.find_flips_random(
            cov, template_cov, E,
            params["n_restarts"],
            params["n_iter"],
            params["max_flips"],
            use_tqdm=False,
            random_state=params["random_state"],
        )
        _, t_core, _ = time_function_call(fn, n_repeats, n_warmup)
        return {"core": t_core, "refine": 0.0, "total": t_core}

    if method == "spectral":
        core_fn = lambda: sf.find_flips_spectral(cov, template_cov, E, refine=False)
    elif method == "belief_propagation":
        core_fn = lambda: sf.find_flips_belief_propagation(
            cov, template_cov, E,
            bp_n_iter=params["bp_n_iter"],
            beta=params["beta"],
            damping=params["damping"],
            prior_strength=params["prior_strength"],
            bp_tol=params["bp_tol"],
            refine=False,
        )
    elif method == "sdp":
        core_fn = lambda: sf.find_flips_sdp(
            cov, template_cov, E,
            sdp_n_restarts=params["sdp_n_restarts"],
            sdp_max_iter=params["sdp_max_iter"],
            sdp_tol=params["sdp_tol"],
            gw_samples=params["gw_samples"],
            refine=False,
            random_state=params["random_state"],
        )
    else:
        raise ValueError(f"Unknown method '{method}'.")

    (flips_core, _), t_core, _ = time_function_call(core_fn, n_repeats, n_warmup)

    refine_fn = lambda: sf.greedy_refine(
        cov, template_cov, flips_core, E,
        REFINE_PARAMS["n_refine"], strategy=REFINE_PARAMS["refine_strategy"],
    )
    _, t_refine, _ = time_function_call(refine_fn, n_repeats, n_warmup)

    return {"core": t_core, "refine": t_refine, "total": t_core + t_refine}


def flop_model(n, E):
    """
    Order-of-magnitude analytic FLOP counts per method, as a dict of {'core','refine','total'}.

    - n = number of parcels
    - E = number of embeddings

    NOTE:
        * Constants are coarse (a metric evaluation is charged ~2 * D**2, one covariance pass +
          one correlation pass).
        * The iteration terms for BP/SDP use the configured caps (bp_n_iter, sdp_max_iter),
          so they are UPPER BOUNDS. BP/SDP typically converge in far fewer sweeps.
        * Wallclock time (measured above) is the ground truth; this column gives the scaling
          and relative magnitude.
    """
    # Get method parameters
    random = METHOD_PARAMS["random"]
    bp = METHOD_PARAMS["belief_propagation"]
    sdp = METHOD_PARAMS["sdp"]

    # Get relevant data dimensions
    D = n * E  # dimension of the covariance matrix
    D2 = D * D
    k = int(np.ceil(np.sqrt(2 * n))) + 1  # default mixing-method rank for SDP

    # Define FLOP cost for metric computation
    # - one apply_flips_to_covariance + one covariance_matrix_correlation
    metric = 2.0 * D2

    # Define FLOP cost for greedy refinement
    # - one O(D**2) build of the block-sum affinities, then O(n) per trial with
    #   the rank-1 evaluator (dominated by the build for these sizes)
    refine = 2.0 * D2 + REFINE_PARAMS["n_refine"] * n * n

    # Initialize output dictionary
    out = {}

    # [Random] n_restarts * n_iter full-metric evaluations (no refine)
    rand_core = random["n_restarts"] * random["n_iter"] * metric
    out["random"] = {"core": rand_core, "refine": 0.0, "total": rand_core}

    # [Spectral] eigh takes ~ (4/3) n**3 + O(D**2) affinity build
    spec_core = (4.0 / 3.0) * n ** 3 + 2.0 * D2
    out["spectral"] = {"core": spec_core, "refine": refine, "total": spec_core + refine}

    # [BP] affinity build (2*D2) + `bp_n_iter` message-passing iterations
    # Each iteration costs ~10*n**2: a handful of dense n-by-n array ops
    # (e.g., tanh/arctanh message updates, damping) and the O(n**2) metric recording;
    # the leading 10 is a coarse "~10 n**2-scale ops per iteration" constant. Iterations are
    # charged at the cap `bp_n_iter`, so this term is an upper bound.
    bp_core = 2.0 * D2 + bp["bp_n_iter"] * (10.0 * n * n)
    out["belief_propagation"] = {"core": bp_core, "refine": refine, "total": bp_core + refine}

    # [SDP] affinity build (2*D2) + mixing-method solve + Goemans-Williamson (GW) rounding
    #   - mixing-method solve: `sdp_n_restarts` restarts x `sdp_max_iter` sweeps, each sweep
    #     updating all n rows of the rank-k factor V (each row an n-by-k matvec ~= n**2 * k);
    #   - GW rounding: `gw_samples` random hyperplanes, each scoring a quadratic form ~ n**2.
    # Sweeps are charged at the cap `sdp_max_iter`, so this term is an upper bound.
    sdp_core = (2.0 * D2
                + sdp["sdp_n_restarts"] * sdp["sdp_max_iter"] * (n * n * k)
                + sdp["gw_samples"] * (n * n))
    out["sdp"] = {"core": sdp_core, "refine": refine, "total": sdp_core + refine}

    return out


# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark wallclock + FLOPs of the sign-flipping methods.",
    )
    parser.add_argument("dataset", help="Dataset name (e.g., camcan).")
    parser.add_argument(
        "--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS,
        help="Subset of methods to benchmark (default: all four).",
    )
    parser.add_argument(
        "--n-sessions", type=int, default=None,
        help="Benchmark on the first N sessions only (default: all). Random is slow, so a "
             "subset (e.g. 20) is usually enough for a stable per-session median.",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=5,
        help="Timed repeats per (session, method); the median is reported (default: 5).",
    )
    parser.add_argument(
        "--n-warmup", type=int, default=1,
        help="Untimed warm-up runs before timing (default: 1).",
    )
    return parser.parse_args()


# BENCHMARK ANALYSIS
if __name__ == "__main__":
    # Get input arguments
    args = parse_args()
    dataset = args.dataset
    E = COMMON_PARAMS["n_embeddings"]
    print(f"[INFO] Dataset: {dataset} | Methods: {', '.join(args.methods)}")
    print(f"[INFO] Threads pinned to 1 (OMP/OPENBLAS/MKL). "
          f"repeats={args.n_repeats}, warmup={args.n_warmup}")

    # Locate raw parcellated files (identical across src_<method> dirs)
    src_dir = None
    for m in args.methods:
        cand = os.path.join(BASE_DIR, dataset, f"src_{m}")
        if os.path.isdir(cand):
            src_dir = cand
            break
    if src_dir is None:
        raise FileNotFoundError(f"No src_<method> directory under {BASE_DIR}/{dataset}.")
    parc_files = sorted(glob(os.path.join(src_dir, "*/parc/lcmv-parc-raw.fif")))
    session_ids = [f.split("/")[-3] for f in parc_files]
    print(f"[INFO] Found {len(session_ids)} sessions in {src_dir}.")

    # Define template session (reusing found sessions)
    template_sessions = {
        "camcan": "sub-CC110606",
        "lemon": "sub-010295",
        "wakeman_henson": "sub15_run03",
    }
    template = template_sessions.get(dataset)
    if template is None:
        raise ValueError(f"No template defined for '{dataset}' dataset.")
    print(f"[INFO] Template: {template}")

    # Load data TDE covariances (expensive I/O, done a single time)
    print("[INFO] Loading covariances ...")
    covs = sf_io.load_covariances(
        parc_files,
        n_embeddings=E,
        standardize_data=COMMON_PARAMS["standardize"],
        use_tqdm=True,
    )
    template_cov = covs[session_ids.index(template)]
    n_parcels = covs.shape[1] // E
    print(f"[INFO] Covariance matrix dimension D = {covs.shape[1]} " +
          f"(n = {n_parcels} parcels, E = {E} embeddings)")

    # Get sessions to evaluate (excluding the template)
    eval_idx = [i for i in range(len(session_ids)) if session_ids[i] != template]
    if args.n_sessions is not None:
        eval_idx = eval_idx[:args.n_sessions]
    print(f"[INFO] Timing {len(eval_idx)} sessions x {len(args.methods)} methods ...")

    # Measure per-method, per-session wallclock times
    times = {m: [] for m in args.methods}  # list over sessions of {'core','refine','total'}
    for c, i in enumerate(eval_idx, 1):
        for m in args.methods:
            rec = time_method(m, covs[i], template_cov, E, args.n_repeats, args.n_warmup)
            times[m].append(rec)
        if c % 10 == 0 or c == len(eval_idx):
            print(f"\t[{c:3d}/{len(eval_idx)}] {session_ids[i]}")

    # Aggregate wallclock time measurements
    stages = ["core", "refine", "total"]
    agg = {m: {st: np.array([r[st] for r in times[m]]) for st in stages} for m in args.methods}

    # Compute and report median-over-sessions of the per-session median wallclock times
    print("\n" + "=" * 84)
    print(f"COST BENCHMARK  ({dataset}: n = {n_parcels}, E = {E}, D = {n_parcels * E}; "
          f"{len(eval_idx)} sessions)")
    print("=" * 84)
    print(f"\nWallclock per session, median [IQR] in ms  (serial, single-threaded):")
    print(f"\t{'method':<20}{'core (ms)':>18}{'refine (ms)':>18}{'total (ms)':>18}")
    for m in args.methods:
        cells = ""
        for st in stages:
            x = agg[m][st] * 1e3
            median = np.median(x)  # median over sessions
            iqr = np.subtract(*np.percentile(x, [75, 25]))
            cells += f"{median:10.2f} [{iqr:5.2f}]".rjust(18)
        print(f"\t{m:<20}{cells}")

    # Measure speed-up against the random method on total wallclock time
    rand_total = np.median(agg["random"]["total"]) if "random" in agg else None
    if rand_total:
        print(f"\nSpeed-up vs random (median total wallclock time):")
        for m in args.methods:
            speed_up = rand_total / np.median(agg[m]["total"])
            print(f"\t{m:<20}{speed_up:8.1f}x")

    # Get analytic FLOP counts per method
    flops = flop_model(n_parcels, E)

    print(f"\nAnalytic FLOP model (BP/SDP core use iteration CAPS -> upper bounds):")
    print(f"\t{'method':<20}{'core':>14}{'refine':>14}{'total':>14}")
    for m in args.methods:
        f = flops[m]
        print(f"\t{m:<20}{f['core']:14.3e}{f['refine']:14.3e}{f['total']:14.3e}")

    # Save result outputs
    # (includes per-session raw timings + the aggregate + FLOP model)
    out_dir = os.path.join(BASE_DIR, dataset, "data")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "benchmark_cost.npz")
    save = {
        "sessions": np.array([session_ids[i] for i in eval_idx]),
        "n_parcels": n_parcels, "n_embeddings": E, "template": template,
    }
    for m in args.methods:
        for st in stages:
            save[f"{m}_{st}"] = agg[m][st]
            save[f"flops_{m}_{st}"] = flops[m][st]
    np.savez(out, **save)
    print(f"\n[INFO] Output saved to {out}")
