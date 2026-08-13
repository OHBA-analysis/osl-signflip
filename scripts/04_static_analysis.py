"""Extract static features for specified sign-flipping method.

Computes per-subject and group-level static features from the sign-flipped
parcel data and saves them for cross-method comparison (``05_static_comparison.py``
aggregates all four methods).

Two kinds of feature:

* **Sign-invariant** PSD, power map and amplitude-envelope connectivity (AEC).
  These do not depend on per-parcel sign flips (PSD/power use ``|.|^2``; AEC
  correlates sign-blind amplitude envelopes). Saved so the comparison step can
  confirm they are identical across methods -- the sanity check that sign
  flipping preserves standard static biomarkers.

* **TDE covariance (sign-sensitive).** The source data are symmetric-
  orthogonalised, so the *zero-lag* signed correlation between parcels is ~0 by
  construction; the sign-sensitive structure lives at nonzero lag. We therefore
  use the time-delay-embedded covariance (``n_embeddings = 15`` -> D = 15 * C).
  Cross-parcel covariance blocks flip sign with the flips (within-parcel
  auto-covariance blocks are invariant), so each subject is scored by the similarity
  of its cross-parcel TDE covariance to the template subject's.

Env: ``osld`` (to use osl-dynamics package)
Usage:
    python 04_static_analysis.py <dataset> <sign_flipping_method> --n-workers <n>
"""

# Import packages
import argparse
import glob
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import numpy as np

from osl_dynamics.analysis import static, power
from osl_dynamics.data import Data


# Set directories
BASE_DIR = "/well/woolrich/users/olt015/Cho2026_SignFlip"

# Define analysis constants
FS = 250                # sampling frequency (Hz)
WELCH_WINDOW = 500      # Welch window length (samples)
FILTER_BAND = (1, 45)   # band for the amplitude envelope correlation (AEC) computation
N_EMBEDDINGS = 15       # time-delay embeddings -> D = n_parcels * N_EMBEDDINGS

# Define helper functions
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract static features from sign-flipped source data.",
    )
    parser.add_argument("dataset", help="Dataset name (e.g., lemon, wakeman_henson).")
    parser.add_argument("sign_flipping_method", help="Sign flipping method applied to the data.")
    parser.add_argument(
        "--n-workers", type=int, default=16,
        help="Number of Dask workers for parallel processing (default: 16).",
    )
    return parser.parse_args()


def get_template(dataset):
    """Reads the template subject (auto-selected by 01_sign_flip.py) from a data.pkl."""
    pattern = os.path.join(
        BASE_DIR, dataset, "src_sdp/src_report/*/data.pkl"
    )  # use `src_sdp` as a representative directory
    for f in sorted(glob.glob(pattern)):
        with open(f, "rb") as fh:
            t = pickle.load(fh).get("template")
        if t:
            return t
    return None


def subject_of(path):
    """src_{method}/{subject}/{subject}_sflip_lcmv-parc-raw.fif -> subject."""
    return os.path.basename(os.path.dirname(path))


def cross_parcel_triu(C, n_embeddings):
    """
    Strict-upper-triangle mask restricted to cross-parcel entries.

    osl-dynamics TDE ordering is channel-major, so embedded dim ``d`` belongs to
    parcel ``d // n_embeddings``. Within-parcel (auto-covariance) blocks are
    sign-invariant and excluded to isolate the sign-flip-sensitive structure.
    """
    D = C * n_embeddings
    parcel = np.arange(D) // n_embeddings
    cross = parcel[:, None] != parcel[None, :]
    triu = np.triu(np.ones((D, D), dtype=bool), 1)
    return triu & cross


def corr(a, b):
    """Computes Pearson product-moment correlation coefficients."""
    return np.corrcoef(a, b)[0, 1]


# STATIC ANALYSIS
if __name__ == "__main__":
    # Get input arguments
    args = parse_args()
    dataset = args.dataset
    method = args.sign_flipping_method
    n_jobs = args.n_workers
    print(f"[INFO] Dataset: {dataset} | Sign flipping method: {method}")

    # Get template subject/session
    template = get_template(dataset)
    print(f"[INFO] Template subject: {template}")

    # Get data file paths (aligned by subject order; template kept as TDE reference)
    src_dir = os.path.join(BASE_DIR, dataset, f"src_{method}")
    files = {subject_of(f): f for f in sorted(
        glob.glob(os.path.join(src_dir, "*", f"*_sflip_lcmv-parc-raw.fif"))
    )}
    subjects = sorted(files)
    assert template in subjects, "template subject missing from sflip files"
    flist = [files[s] for s in subjects]
    print(f"[INFO] Number of available subjects: {len(subjects)} (incl. template)")

    # Load data
    data = Data(
        flist, picks="misc", reject_by_annotation="omit",
        sampling_frequency=FS, n_jobs=n_jobs,
    )
    ts = data.time_series()  # raw time series
    # shape: list of (n_samples, n_channels); aligned to `subjects`
    C = ts[0].shape[1]  # number of channels

    # Compute sign-invariant features from raw time series (for quality control)
    f, psd = static.welch_spectra(
        data=ts, sampling_frequency=FS, window_length=WELCH_WINDOW,
        standardize=True, n_jobs=n_jobs,
    )  # psd.shape: (n_subjects, C, n_freqs)
    pwr = power.variance_from_spectra(f, psd) # pwr.shape: (n_subjects, C)

    # Compute TDE covariance (sign-sensitive)
    data.prepare({"tde": {"n_embeddings": N_EMBEDDINGS}, "standardize": {}})
    tde = static.functional_connectivity(
        data.time_series(), conn_type="cov", n_jobs=n_jobs
    )  # tde.shape: (n_subjects, D, D), where D = C * N_EMBEDDINGS
    del data

    # Compute AEC from a freshly reloaded raw object
    aec_data = Data(
        flist, picks="misc", reject_by_annotation="omit",
        sampling_frequency=FS, n_jobs=n_jobs,
    )
    aec_data.prepare({
        "filter": {"low_freq": FILTER_BAND[0], "high_freq": FILTER_BAND[1]},
        "amplitude_envelope": {},
        "standardize": {},
    })
    aec = static.functional_connectivity(aec_data.time_series(), n_jobs=n_jobs)
    # aec.shape: (n_subjects, C, C)
    del aec_data

    # Compute before-flip group TDE covariance (method-independent)
    before_files = [
        os.path.join(src_dir, s, "parc/lcmv-parc-raw.fif") for s in subjects
    ]
    before_data = Data(
        before_files, picks="misc", reject_by_annotation="omit",
        sampling_frequency=FS, n_jobs=n_jobs,
    )
    before_data.prepare({"tde": {"n_embeddings": N_EMBEDDINGS}, "standardize": {}})
    tde_group_before = static.functional_connectivity(
        before_data.time_series(), conn_type="cov", n_jobs=n_jobs
    ).mean(0)  # tde_group_before.shape: (D, D)
    del before_data

    # NOTE: The pre-flip parcel data (parc/lcmv-parc-raw.fif) are identical across methods;
    # averaging their TDE covariance across subjects -- each with an arbitrary per-parcel
    # sign -- gives the unresolved "before" group covariance that sign flipping aligns.
    # Computed here (once per method run; the result should be the same for every method)
    # so the comparison step can load it alongside each method's after-flip tde_group.

    # Compute per-subject similarity of the cross-parcel TDE covariance to the template's
    mask = cross_parcel_triu(C, N_EMBEDDINGS)
    ti = subjects.index(template)
    tmpl_vec = tde[ti][mask]
    nt_idx = [i for i, s in enumerate(subjects) if s != template]
    nt_subjects = [subjects[i] for i in nt_idx]
    tde_sim_template = np.array([corr(tde[i][mask], tmpl_vec) for i in nt_idx])

    # Get group-level features
    psd_group = psd.mean(0)
    power_group = pwr.mean(0)
    aec_group = aec.mean(0)
    tde_group = tde.mean(0)

    # Save per-subject features (non-template) and group means
    outdir = os.path.join(BASE_DIR, dataset, "data")
    os.makedirs(outdir, exist_ok=True)
    out_npz = os.path.join(outdir, f"static_{method}.npz")
    np.savez(
        out_npz,
        dataset=dataset,
        method=method,
        subjects=np.array(nt_subjects),
        template=template,
        n_parcels=C,
        n_embeddings=N_EMBEDDINGS,
        f=f,
        psd=psd[nt_idx].astype(np.float32),       # (nt, C, F)
        power=pwr[nt_idx].astype(np.float32),     # (nt, C)
        aec=aec[nt_idx].astype(np.float32),       # (nt, C, C)
        psd_group=psd_group,
        power_group=power_group,
        aec_group=aec_group,                      # (C, C) after-flip group AEC
        tde_group=tde_group,                      # (D, D) after-flip group TDE cov
        tde_group_before=tde_group_before,        # (D, D) before-flip group TDE cov
        tde_sim_template=tde_sim_template,        # (nt,)
    )  # nt: number of non-template subjects
    print(f"[INFO] Saved features to {out_npz}")

    print(f"[INFO] TDE similarity to template: mean {tde_sim_template.mean():.4f} "
          f"+/- {tde_sim_template.std():.4f} (n = {len(nt_subjects)})")
    print("[INFO] Done.")
