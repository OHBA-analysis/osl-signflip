"""
Measure seed stability for the sign-flipping methods that include randomized process.

Background: ``random`` and ``sdp`` both draw random numbers, so their output can
vary with the RNG seed; ``spectral`` and ``belief_propagation`` are deterministic
by construction (no seed) and are therefore omitted. This script re-runs sign flipping
for several seeds per session and records the after-flip covariance-correlation metric,
so we can quantify the run-to-run variability.

Uses the same covariance construction and solver hyperparameters as ``01_sign_flip.py``
so the numbers are comparable with the deployed pipeline.

Env: ``osle`` (imports sign-flipping algorithms)
Usage:
    python 10_seed_stability.py <dataset> [--n-sessions N] [--n-seeds N] [--n-jobs N]
"""

# Import packages
import argparse
import numpy as np
import os
from glob import glob
from joblib import Parallel, delayed, parallel_backend
from osl_ephys.source_recon import sign_flipping as sf

# Set directories
BASE_DIR = "/well/woolrich/users/olt015/Cho2026_SignFlip"

# Set hyperparameters
N_EMBEDDINGS = 15
STANDARDIZE = True
REFINE_PARAMS = {
    "refine": True,
    "n_refine": 5,
    "refine_strategy": "best"
}

METHOD_PARAMS = {
    "random": {
        "n_init": 3,
        "n_iter": 5000,
        "max_flips": 20,
    },
    "sdp": {
        "sdp_n_restarts": 5,
        "sdp_max_iter": 1000,
        "sdp_tol": 1.0e-8,
        "gw_samples": 500,
    },
}
METHODS = list(METHOD_PARAMS)

# Define argument parser
def parse_args():
    p = argparse.ArgumentParser(
        description="Test seed-stability of random / SDP sign flipping."
    )
    p.add_argument(
        "dataset",
        help="Dataset name (e.g., camcan).",
    )
    p.add_argument(
        "--n-sessions", type=int, default=None,
        help="Use the first N non-template sessions (default: all)."
    )
    p.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of RNG seeds (default: 5)."
    )
    p.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel workers across sessions (default: 1, i.e. serial). "
             "BLAS threads per worker are pinned to 1 to avoid oversubscription."
    )
    return p.parse_args()

# Define a helper function
def run_once(method, cov, template_cov, seed):
    """Returns the after-flip metric for a single method, session, and seed."""
    params = METHOD_PARAMS[method]
    if method == "random":
        _, metrics = sf.find_flips(
            cov, template_cov, N_EMBEDDINGS,
            params["n_init"], params["n_iter"], params["max_flips"],
            use_tqdm=False, random_state=seed,
        )
    else:  # SDP
        _, metrics = sf.find_flips_sdp(
            cov, template_cov, N_EMBEDDINGS,
            sdp_n_restarts=params["sdp_n_restarts"], sdp_max_iter=params["sdp_max_iter"],
            sdp_tol=params["sdp_tol"], gw_samples=params["gw_samples"],
            random_state=seed, **REFINE_PARAMS,
        )
    return float(np.asarray(metrics).ravel()[-1])


def run_session(cov, template_cov, seeds):
    """
    Returns a dictionary with methods as keys and (n_seeds,) after-flip metric arrays
    as values for a single session.

    Independent per session, so this is the unit of work parallelized over.
    """
    return {
        m: np.array([run_once(m, cov, template_cov, s) for s in seeds])
        for m in METHODS
    }


# SEED STABILITY
if __name__ == "__main__":
    # Get input arguments
    args = parse_args()
    dataset = args.dataset
    seeds = list(range(args.n_seeds))
    print(
        f"[INFO] Dataset: {dataset} | Sign flipping methods: {', '.join(METHODS)} " +
        f"| Seeds: {seeds}"
    )

    # Locate raw parcellated files (identical across src_<method> dirs)
    src_dir = next((os.path.join(BASE_DIR, dataset, f"src_{m}") for m in METHODS
                    if os.path.isdir(os.path.join(BASE_DIR, dataset, f"src_{m}"))), None)
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
    print(f"[INFO] Template: {template}")

    # Load data TDE covariances
    print("[INFO] Loading covariances ...")
    covs = sf.load_covariances(
        parc_files,
        n_embeddings=N_EMBEDDINGS,
        standardize=STANDARDIZE,
        use_tqdm=True,
    )
    template_cov = covs[session_ids.index(template)]

    # Get sessions to evaluate (excluding the template)
    eval_idx = [i for i in range(len(session_ids)) if session_ids[i] != template]
    if args.n_sessions is not None:
        eval_idx = eval_idx[:args.n_sessions]
    print(f"[INFO] {len(eval_idx)} sessions x {len(METHODS)} methods x {len(seeds)} seeds "
          f"(n_jobs = {args.n_jobs}) ...")

    # Compute metrics per session
    # (parallelized over sessions, with each seed/method processed within the loop)
    tasks = (delayed(run_session)(covs[i], template_cov, seeds) for i in eval_idx)
    if args.n_jobs == 1:
        results = Parallel(n_jobs=1, verbose=10)(tasks)
    else:
        # Pin BLAS threads to 1 per worker so 16 processes don't oversubscribe the cores
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=args.n_jobs, verbose=10)(tasks)

    metric = {m: np.full((len(eval_idx), len(seeds)), np.nan) for m in METHODS}
    # metric[method]: (n_sessions, n_seeds)
    for si, res in enumerate(results):
        for m in METHODS:
            metric[m][si] = res[m]

    # Report per-session variability across seeds, summarized over sessions
    print("\nPer-session standard deviation across seeds:")
    for m in METHODS:
        s = metric[m].std(axis=1)  # standard deviation over seeds per session
        print(f"\t{m:<10} median std = {np.median(s):.2e}  "
              f"max std = {s.max():.2e} (mean metric {metric[m].mean():.4f})")

    # Save result outputs
    out_dir = os.path.join(BASE_DIR, dataset, "data")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "seed_stability.npz")
    save = {
        "dataset": dataset,
        "methods": np.array(METHODS),
        "sessions": np.array([session_ids[i] for i in eval_idx]),
        "seeds": np.array(seeds),
        "template": template,
        "n_parcels": covs.shape[1] // N_EMBEDDINGS,
        }
    for m in METHODS:
        save[f"metric_{m}"] = metric[m]
    np.savez(out, **save)
    print(f"[INFO] Output saved to {out}.")
