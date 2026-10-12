"""Train TDE-HMM on the sign-flipped source data for specified method.

Trains a single HMM run for a given ``(dataset, method, run_id)`` and saves
that run's outputs. The best run (lowest free energy) is selected later in
``07_dynamic_comparison.py``. Note that HMM state ordering is arbitrary from
run to run.

The random seed is tied to ``run_id`` only (not the method), so run ``r`` starts
from the *same* initialization for every sign-flipping method to ensure a fair
cross-method comparison.

Env: ``osld`` (to use osl-dynamics package)
Usage:
    python 06_dynamic_analysis.py <dataset> <method> <run_id>
"""

# Import packages
import argparse
import numpy as np
import os
import pickle
from glob import glob

from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.utils import set_random_seed


METHODS = ["random", "spectral", "sdp", "belief_propagation"]

# Set directories
BASE_DIR = "/well/woolrich/users/olt015/osl-signflip"

# Define analysis constants
FS = 250                # sampling frequency (Hz)
N_EMBEDDINGS = 15       # time-delay embeddings
N_PCA_COMPONENTS = 80   # PCA dimensionality after TDE (== HMM n_channels)

# Configure HMM hyperparameters
N_STATES = 6
SEQUENCE_LENGTH = 800
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
N_EPOCHS = 20
INIT_N_EPOCHS = 2       # epochs per random-state-time-course initialization
INIT_N_INIT = 5         # number of initializations

# Define helper functions
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a TDE-HMM on sign-flipped source data.",
    )
    parser.add_argument("dataset", help="Dataset name (e.g., lemon, camcan, wakeman_henson).")
    parser.add_argument("method", choices=METHODS, help="Sign flipping method applied to the data.")
    parser.add_argument("run_id", type=int, help="Run index (also seeds the initialization).")
    parser.add_argument(
        "--n-workers", type=int, default=16,
        help="Number of parallel jobs for data loading/preparation (default: 16).",
    )
    return parser.parse_args()


# DYNAMIC ANALYSIS
if __name__ == "__main__":
    # Get input arguments
    args = parse_args()
    dataset = args.dataset
    method = args.method
    run_id = args.run_id
    n_jobs = args.n_workers
    print(f"[INFO] Dataset: {dataset} | Method: {method} | Run: run{run_id}")

    # Set GPU memory growth
    tf_ops.gpu_growth()

    # Set random seed for Python random, NumPy, and TensorFlow.
    set_random_seed(run_id, op_determinism=False)
    # NOTE: The seed depends on run_id only, so a given run starts from the same
    #       initialization across methods (a fair cross-method comparison).

    # Set directories
    data_dir = os.path.join(BASE_DIR, dataset, f"src_{method}")
    hmm_dir = os.path.join(data_dir, f"hmm/run{run_id}")
    os.makedirs(hmm_dir, exist_ok=True)

    # Get data files
    file_paths = sorted(glob(os.path.join(data_dir, "*/*_sflip_lcmv-parc-raw.fif")))
    print(f"[INFO] Number of available sessions: {len(file_paths)}")

    # Prepare data (TDE + PCA + standardize)
    input_data = Data(
        file_paths, picks="misc", reject_by_annotation="omit",
        sampling_frequency=FS, n_jobs=n_jobs,
    )
    input_data.prepare({
        "tde_pca": {"n_embeddings": N_EMBEDDINGS, "n_pca_components": N_PCA_COMPONENTS},
        "standardize": {},
    })

    # Define the HMM configuration
    config = Config(
        n_states=N_STATES,
        n_channels=N_PCA_COMPONENTS,
        sequence_length=SEQUENCE_LENGTH,
        learn_means=False,
        learn_covariances=True,
        learn_trans_prob=True,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS,
    )

    # Build the model
    model = Model(config)
    model.summary()

    # Initialize weights
    model.random_state_time_course_initialization(
        input_data, n_epochs=INIT_N_EPOCHS, n_init=INIT_N_INIT,
    )

    # Train the model on the full dataset
    history = model.fit(input_data)

    # Save the trained model and training history
    model.save(f"{hmm_dir}/trained_model")
    with open(f"{hmm_dir}/history.pkl", "wb") as f:
        pickle.dump(history, f)

    # Get results
    loss = history["loss"]                       # training loss curve
    free_energy = model.free_energy(input_data)  # variational free energy
    alpha = model.get_alpha(input_data)          # inferred state probabilities (HMM gamma)
    tp = model.get_trans_prob()                  # inferred transition probability matrix
    cov = model.get_covariances()                # observation model state covariances (in TDE-PCA space)

    print("[INFO] Final loss:", loss[-1])
    print("[INFO] Free energy:", free_energy)

    # Save results
    outputs = {
        "loss": loss,
        "free_energy": free_energy,
        "alpha": alpha,
        "transition_probability": tp,
        "covariance": cov,  # in each method's own PCA basis; not cross-comparable
    }
    with open(f"{hmm_dir}/run_results.pkl", "wb") as f:
        pickle.dump(outputs, f)
    np.save(f"{hmm_dir}/free_energy.npy", free_energy)  # for fast best-run scanning
    # NOTE: We deliberately do NOT save the raw training time series here; it is
    #       run-independent and very large. 07 reloads and trims the raw parcel data
    #       itself to align with alpha.

    # Delete temporary data directory
    input_data.delete_dir()

    print("[INFO] Training complete.")
