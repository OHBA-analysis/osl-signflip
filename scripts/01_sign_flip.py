"""Apply dipole sign flipping on the source space data.

For each session, this script (i) computes the time-delay-embedded (TDE) covariance of the
parcellated data, (ii) finds the sign flips that best align it to a template session with
``osl-signflip``, (iii) saves the sign-flipped parcel time courses, and (iv) records the
before/after metrics in the session's report data (read by ``02_collect_results.py``).

Env: ``osld`` (with ``osl-signflip`` installed)
Usage:
    python 01_sign_flip.py <dataset> <method> [--template SESSION] [--n-workers N]
"""

# Install dependencies
import argparse
import os
import pickle
from glob import glob

import mne
import osl_signflip as sf
from joblib import Parallel, delayed, parallel_backend
from osl_signflip import io as sf_io

# Set directories
BASE_DIR = "/well/woolrich/users/olt015/osl-signflip"

# Set hyperparameters
COMMON_PARAMS = {
    "n_embeddings": 15,
    "standardize": True,
}  # shared by every sign flipping method

METHOD_PARAMS = {
    "random": {
        "n_restarts": 3,
        "n_iter": 5000,
        "max_flips": 20,
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
    },
}  # specific to each sign flipping method

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply dipole sign flipping to the source space data.",
    )
    parser.add_argument(
        "dataset",
        help="Dataset name (e.g., lemon).",
    )
    parser.add_argument(
        "sign_flipping_method",
        choices=list(METHOD_PARAMS),
        help="Sign flipping method to apply.",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Session ID to use as the alignment template. "
             "If omitted, a good template is found automatically.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16).",
    )
    return parser.parse_args()

# Define helper functions
def load_covariance(parc_file):
    """Computes the TDE covariance of a parcellated data file."""
    return sf_io.load_covariances(
        [parc_file],
        n_embeddings=COMMON_PARAMS["n_embeddings"],
        standardize_data=COMMON_PARAMS["standardize"],
    )[0]


def add_to_report(report_file, info):
    """
    Adds info to a session's report data, keeping the entries already there
    (e.g., from source reconstruction), as osl-ephys's ``src_report.add_to_data`` does.
    """
    data = {}
    if os.path.exists(report_file):
        with open(report_file, "rb") as fh:
            data = pickle.load(fh)
    else:
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
    data.update(info)
    with open(report_file, "wb") as fh:
        pickle.dump(data, fh)


def sign_flip_session(src_dir, session, template, template_cov, method):
    """
    Finds and applies the sign flips for one session, saves the sign-flipped data,
    and records the metrics.

    Returns
    -------
    session : str
        Session ID.
    before : float
        Metric before sign flipping.
    after : float
        Metric of the applied sign flips.
    """
    # Get file paths
    parc_file = os.path.join(src_dir, session, "parc", "lcmv-parc-raw.fif")
    sflip_file = os.path.join(src_dir, session, f"{session}_sflip_lcmv-parc-raw.fif")
    report_file = os.path.join(src_dir, "src_report", session, "data.pkl")

    # Find the channels to flip
    cov = load_covariance(parc_file)
    flips, metrics = sf.find_flips(
        cov,
        template_cov,
        COMMON_PARAMS["n_embeddings"],
        method=method,
        **METHOD_PARAMS[method],
    )

    # Apply flips to the parcellated data and save
    raw = mne.io.read_raw_fif(parc_file, verbose=False)
    sf_io.apply_flips(raw, flips).save(sflip_file, overwrite=True, verbose=False)

    # Save info for the report
    add_to_report(report_file, {
        "fix_sign_ambiguity": True,
        "method": method,
        "template": template,
        **COMMON_PARAMS,
        **METHOD_PARAMS[method],
        "metrics": metrics,
    })

    return session, metrics[0], metrics[-1]


# SIGN FLIP DATA
if __name__ == "__main__":
    # Get input arguments
    args = parse_args()
    dataset = args.dataset
    sign_flipping_method = args.sign_flipping_method
    template = args.template
    print(f"[INFO] Dataset: {dataset} | Sign flipping method: {sign_flipping_method}")

    # Get session IDs
    src_dir = os.path.join(BASE_DIR, f"{dataset}/src_{sign_flipping_method}")
    file_paths = sorted(glob(os.path.join(src_dir, "*/parc/lcmv-parc-raw.fif")))
    session_ids = [file.split("/")[-3] for file in file_paths]
    print(f"[INFO] Number of available sessions: {len(session_ids)}")

    # Find a good template session to align other sessions to
    if template is None:
        covs = sf_io.load_covariances(
            file_paths,
            n_embeddings=COMMON_PARAMS["n_embeddings"],
            standardize_data=COMMON_PARAMS["standardize"],
            use_tqdm=True,
        )
        template_index = sf.find_template(
            covs, diag_offset=COMMON_PARAMS["n_embeddings"], use_tqdm=True
        )
        template = session_ids[template_index]
        template_cov = covs[template_index]
    else:
        if template not in session_ids:
            raise ValueError(f"Template session {template} not found in {src_dir}.")
        template_cov = load_covariance(file_paths[session_ids.index(template)])
    print(f"[INFO] Using session {template} as the template.")

    # Perform sign flipping
    tasks = (
        delayed(sign_flip_session)(
            src_dir, session, template, template_cov, sign_flipping_method
        )
        for session in session_ids
    )
    # Pin BLAS threads to 1 per worker so that the workers don't oversubscribe the cores
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.n_workers, verbose=10)(tasks)

    # Report metrics before and after sign flipping
    for session, before, after in results:
        print(f"\t{session}: metric before {before:.4f}, after {after:.4f}")

    print("Sign flipping complete.")
