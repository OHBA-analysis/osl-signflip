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

    # Set the template session (if omitted, a good template is found automatically)
    if template is not None and template not in session_ids:
        raise ValueError(f"Template session {template} not found in {src_dir}.")

    # Find and apply the sign flips, and save the sign-flipped data
    out_paths = [
        os.path.join(src_dir, session, f"{session}_sflip_lcmv-parc-raw.fif")
        for session in session_ids
    ]
    flips, metrics, template_index = sf_io.sign_flip_files(
        file_paths,
        out_paths,
        n_embeddings=COMMON_PARAMS["n_embeddings"],
        method=sign_flipping_method,
        template=None if template is None else session_ids.index(template),
        standardize_data=COMMON_PARAMS["standardize"],
        n_jobs=args.n_workers,
        **METHOD_PARAMS[sign_flipping_method],
    )
    template = session_ids[template_index]
    print(f"[INFO] Used session {template} as the template.")

    # Print metrics before and after sign flipping
    # (the template is not flipped against itself, so its metrics are NaN)
    for session, session_metrics in zip(session_ids, metrics):
        add_to_report(
            os.path.join(src_dir, "src_report", session, "data.pkl"),
            {
                "fix_sign_ambiguity": True,
                "method": sign_flipping_method,
                "template": template,
                **COMMON_PARAMS,
                **METHOD_PARAMS[sign_flipping_method],
                "metrics": session_metrics,
            },
        )  # save info for the report
        print(
            f"\t{session}: metric before {session_metrics[0]:.4f}, "
            f"after {session_metrics[-1]:.4f}"
        )

    print("Sign flipping complete.")
