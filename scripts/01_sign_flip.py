"""Apply dipole sign flipping on the source space data."""

# Install dependencies
import argparse
import os
import yaml
from dask.distributed import Client
from glob import glob
from osl_ephys import source_recon, utils

# Set directories
BASE_DIR = "/well/woolrich/users/olt015/Cho2026_SignFlip"
FSL_DIR = "/well/woolrich/projects/software/fsl"

# Set hyperparameters
COMMON_PARAMS = {
    "n_embeddings": 15,
    "standardize": True,
}  # shared by every sign flipping method

METHOD_PARAMS = {
    "random": {
        "n_init": 3,
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
        help="Subject ID to use as the alignment template. "
             "If omitted, a good template is found automatically.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=16,
        help="Number of Dask workers for parallel processing (default: 16).",
    )
    return parser.parse_args()

# Define a configuration builder
def build_config(sign_flipping_method, template):
    """
    Builds the osl-ephys source_recon config for the given method.
    
    Parameters
    ----------
    sign_flipping_method : str
        The sign flipping method to use.
    template : str
        The subject ID to use as the target alignment template.

    Returns
    -------
    config : str
        The YAML configuration for the osl-ephys source_recon.
    """
    fix_sign_ambiguity = {
        "method": sign_flipping_method,
        "template": template,
        **COMMON_PARAMS,
        **METHOD_PARAMS[sign_flipping_method],
    }
    config = {"source_recon": [{"fix_sign_ambiguity": fix_sign_ambiguity}]}
    return yaml.dump(config, sort_keys=False)


# SIGN FLIP DATA
if __name__ == "__main__":
    # Get input arguments
    args = parse_args()
    dataset = args.dataset
    sign_flipping_method = args.sign_flipping_method
    template = args.template
    print(f"[INFO] Dataset: {dataset} | Sign flipping method: {sign_flipping_method}")

    # Set logger
    utils.logger.set_up(level="INFO")

    # Set up FSL
    source_recon.setup_fsl(FSL_DIR)

    # Get subject IDs
    src_dir = os.path.join(BASE_DIR, f"{dataset}/src_{sign_flipping_method}")
    file_paths = sorted(glob(os.path.join(src_dir, "*/parc/lcmv-parc-raw.fif")))
    subject_ids = [file.split("/")[-3] for file in file_paths]
    print(f"[INFO] Number of available subjects: {len(subject_ids)}")

    # Find a good template subject to align other subjects to
    if template is None:
        template = source_recon.find_template_subject(
            src_dir, subject_ids, n_embeddings=15, standardize=True
        )
    print(f"[INFO] Using subject/session {template} as the template.")

    # Configure pipeline
    config = build_config(sign_flipping_method, template)

    # Set up parallel processing
    client = Client(n_workers=args.n_workers, threads_per_worker=1)

    # Perform sign flipping
    source_recon.run_src_batch(
        config,
        outdir=src_dir,
        subjects=subject_ids,
        dask_client=True,
        random_seed=0,
    )

    print("Sign flipping complete.")
