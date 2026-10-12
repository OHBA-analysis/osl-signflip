"""Aggregate TDE-HMM dynamic features across sign-flipping methods (per dataset).

Conducts a per-dataset analysis for the dynamic feature comparison. It picks the
best HMM run per method, matches the arbitrary HMM state order across methods to
SDP, and computes every feature required for the visualization step.

For a given dataset, the four methods are trained on the same recordings that
differ only by the per-parcel sign flips, so sessions and time points are aligned
across methods. Therefore, any difference in the inferred dynamics is attributable
to the sign-flipping method.

Writes into ``<dataset>/data/``:

* ``dynamic_features.npz`` -- per-method group features: state PSD/power/coherence
  (``gpsd``/``gpower``/``gcoh``), the sign-sensitive state TDE covariance
  (``tde_cov``), the per-session/per-state summary statistics (``fo``/``lt``/
  ``intv``/``sr``), and the hard state-label sequences (``labels``). Multitaper +
  state-covariance general matrix multiply (GEMM) is heavy to build -> cached and
  reused on re-runs.

* ``dynamic_results.npz`` -- figure-ready arrays: state-segmentation agreement and
  Dice (4x4 and per-session vs SDP), inter-method similarity of every group network
  feature, the state-matched observation-model covariances and transition matrices,
  the per-state Friedman omnibus for each summary statistic (``friedman_chi2``/``_p``,
  shape (n_stats, n_states), Holm-corrected across states as ``friedman_p_holm``), best
  run, and state-match quality.

Env: ``osld`` (to use osl-dynamics package)
Usage:
    python 07_dynamic_features.py <dataset> [--n-workers N] [--blas-threads N]
"""

# Import packages
import argparse
import matplotlib
matplotlib.use("Agg")
import numpy as np
import os
import pickle
import utils as ut

try:  # used to pin BLAS threads around the fused state-covariance GEMM
    from threadpoolctl import threadpool_limits
except Exception:  # falls back to the ambient BLAS thread count
    threadpool_limits = None

from glob import glob
from osl_dynamics.analysis import connectivity, power, spectral
from osl_dynamics.data import Data
from osl_dynamics.data.processing import time_embed
from osl_dynamics.inference import modes
from scipy.stats import friedmanchisquare

# Set directories
BASE_DIR = "/well/woolrich/users/olt015/osl-signflip"

# Define analysis constants (must match 06_dynamic_analysis.py)
METHODS = ["random", "spectral", "sdp", "belief_propagation"]
FS = 250                 # sampling frequency (Hz)
N_EMBEDDINGS = 15        # time-delay embeddings
SEQUENCE_LENGTH = 800    # HMM sequence length
N_STATES = 6

# Define settings for post hoc analysis
FREQ_RANGE = [1, 45]     # frequency range for the multitaper spectra
TIME_HALF_BANDWIDTH = 4  # time-half bandwidth for multitaper
N_TAPERS = 7             # number of multitaper tapers
POWER_BAND = [1, 20]     # band to integrate for power maps
COH_BAND = [1, 20]       # band to average for coherence maps
MT_BATCH = 32            # sessions per multitaper batch (bounds memory on large datasets)
STAT_NAMES = ["fo", "lt", "intv", "sr"]

# Set files for brain parcellations and source space visualizations
MASK_FILE = "MNI152_T1_8mm_brain.nii.gz"
PARCELLATION_FILE = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"  # 38 Giles

# Define helper functions
def parse_args():
    p = argparse.ArgumentParser(description="Aggregate TDE-HMM dynamics for the specified dataset.")
    p.add_argument("dataset", help="Dataset name (e.g., lemon, camcan, wakeman_henson).")
    p.add_argument("--n-workers", type=int, default=8,
                   help="Number of parallel jobs for data loading / multitaper (default: 8).")
    p.add_argument("--blas-threads", type=int, default=0,
                   help="BLAS threads for the state-covariance GEMM (0 = auto).")
    return p.parse_args()


def detect_threads():
    """Detects the number of threads to use for BLAS operations."""
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(v) if v else (os.cpu_count() or 1)


def corr(a, b):
    """Computes Pearson product-moment correlation coefficients."""
    a_flat = np.asarray(a).ravel()
    b_flat = np.asarray(b).ravel()
    return np.corrcoef(a_flat, b_flat)[0, 1]


def session_of(path):
    """src_{method}/{session}/{session}_sflip_lcmv-parc-raw.fif -> session."""
    return os.path.basename(os.path.dirname(path))


def sflip_files(dataset, method):
    """Returns a list of sign-flipped source files for the specified dataset and method."""
    return sorted(glob(os.path.join(
        BASE_DIR, dataset, f"src_{method}", "*/*_sflip_lcmv-parc-raw.fif"
    )))


def pick_best_run(dataset, method):
    """Returns an ID for the best HMM run using the saved free_energy metric."""
    free_energy = {}
    for p in sorted(glob(os.path.join(
        BASE_DIR, dataset, f"src_{method}", "hmm/run*/free_energy.npy"
    ))):
        run_id = int(os.path.basename(os.path.dirname(p)).replace("run", ""))
        free_energy[run_id] = float(np.load(p))
    if not free_energy:
        raise FileNotFoundError(f"no HMM runs found for {dataset}/{method}.")
    return min(free_energy, key=free_energy.get), free_energy


def load_run_results(dataset, method, run_id):
    """Loads the saved results of a specific HMM run."""
    result_path = os.path.join(
        BASE_DIR, dataset, f"src_{method}/hmm/run{run_id}/run_results.pkl"
    )
    with open(result_path, "rb") as fh:
        results = pickle.load(fh)
    return results


def align(ts, alpha):
    """
    Truncates raw time series and alpha time courses of each session
    to a common length (from the start).
    """
    ts_cropped, alpha_cropped = [], []
    for x, a in zip(ts, alpha):
        n = min(len(x), len(a))
        ts_cropped.append(np.asarray(x[:n]))
        alpha_cropped.append(np.asarray(a[:n]))
    return ts_cropped, alpha_cropped


def group_multitaper(ts, alpha, n_jobs):
    """Memory-bounded group-averaged multitaper spectra (length-weighted)."""
    f = None
    psd_sum = coh_sum = None
    total = 0.0
    for s in range(0, len(ts), MT_BATCH):
        tb, ab = ts[s:s + MT_BATCH], alpha[s:s + MT_BATCH]
        f, psd, coh, _ = spectral.multitaper_spectra(
            data=tb, alpha=ab, sampling_frequency=FS,
            frequency_range=FREQ_RANGE, time_half_bandwidth=TIME_HALF_BANDWIDTH,
            n_tapers=N_TAPERS, standardize=True, return_weights=True,
            keepdims=True, n_jobs=n_jobs,
        )
        w = np.array([len(x) for x in tb], dtype=float)
        psd_b = np.tensordot(w, psd, axes=(0, 0))
        coh_b = np.tensordot(w, coh, axes=(0, 0))
        if psd_sum is None:
            psd_sum, coh_sum = psd_b, coh_b
        else:
            psd_sum += psd_b
            coh_sum += coh_b
        total += w.sum()
        print(f"\tmultitaper: {min(s + MT_BATCH, len(ts))}/{len(ts)} sessions")
    return f, psd_sum / total, coh_sum / total


def group_state_cov(raw, alpha, n_states, E, n_threads=None):
    """
    Computes group-level, state-conditional TDE covariance matrices.

    ``C_k = sum_t gamma_k(t) z(t) z(t)^T / sum_t gamma_k(t)`` with ``z`` the
    per-column-standardized TDE of the raw sign-flipped parcel data (no PCA -> the
    parcel x lag axes mean the same thing for every method). The ``n_states``
    state-weighted copies are stacked into one wide GEMM so the BLAS can thread.
    """
    C = raw[0].shape[1]
    D = C * E
    acc = np.zeros((n_states, D, D), dtype=np.float64)
    wsum = np.zeros(n_states, dtype=np.float64)
    limiter = (threadpool_limits(limits=n_threads, user_api="blas")
               if (threadpool_limits is not None and n_threads) else None)
    try:
        for i, (x, a) in enumerate(zip(raw, alpha)):
            z = time_embed(np.asarray(x), E).astype(np.float32)
            z -= z.mean(0)
            z /= z.std(0) + np.float32(1e-6)
            n = min(len(z), len(a))
            z = z[:n]
            aa = np.asarray(a[:n], dtype=np.float32)
            zw = (z[:, None, :] * aa[:, :, None]).reshape(n, n_states * D)
            g = z.T @ zw
            acc += g.reshape(D, n_states, D).transpose(1, 0, 2).astype(np.float64)
            wsum += aa.sum(0, dtype=np.float64)
            if (i + 1) % 50 == 0 or i + 1 == len(raw):
                print(f"\tstate cov: {i + 1}/{len(raw)} sessions")
    finally:
        if limiter is not None:
            limiter.unregister()
    return (acc / wsum[:, None, None]).astype(np.float32)


def compute_label_agreement(la, lb):
    """
    Computes per-session fraction of time points with the same (matched)
    hard state.

    Every time point counts once, so states that are "on" for a lot of the recording
    dominate the score. This is a time-weighted (micro-averaged) overlap.
    """
    out = []
    for x, y in zip(la, lb):
        n = min(len(x), len(y))
        out.append(np.mean(x[:n] == y[:n]))
    return np.array(out)


def compute_label_dice(la, lb, n_states):
    """
    Computes per-session Dice coefficient between two hard state sequences,
    averaged over states.

    Each state gets its own overlap score, and then all states are averaged equally.
    This is a state-weighted (macro-averaged) overlap, i.e., a rare state counts just
    as much as a common one.
    """
    out = []
    for x, y in zip(la, lb):
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        ds = []
        for k in range(n_states):
            ax, bx = (x == k), (y == k)
            denom = ax.sum() + bx.sum()
            if denom:
                ds.append(2.0 * np.logical_and(ax, bx).sum() / denom)
        out.append(np.mean(ds) if ds else np.nan)
    return np.array(out)


# DYNAMIC FEATURE EXTRACTION
if __name__ == "__main__":
    # Get input arguments
    args = parse_args()
    dataset = args.dataset
    n_jobs = args.n_workers
    blas_threads = args.blas_threads if args.blas_threads > 0 else detect_threads()
    print(f"[INFO] Dataset: {dataset} | BLAS threads for state-covariance: {blas_threads}")

    # Set output directory
    outdir = os.path.join(BASE_DIR, dataset, "data")
    os.makedirs(outdir, exist_ok=True)

    # Identify best run per method and load run results
    run_results, best_run = {}, {}
    for m in METHODS:
        best_run[m], _ = pick_best_run(dataset, m)
        run_results[m] = load_run_results(dataset, m, best_run[m])
    print("[INFO] Best run per method (lowest free energy):")
    for m in METHODS:
        print(f"\t{m:<20} -> run{best_run[m]}")

    # Align sessions across sign-flipping methods (same sorted sessions)
    files = {m: sflip_files(dataset, m) for m in METHODS}
    sessions = [session_of(f) for f in files["sdp"]]
    for m in METHODS:
        assert [session_of(f) for f in files[m]] == sessions, f"session mismatch for {m}"
    n_sessions = len(sessions)
    print(f"[INFO] {n_sessions} sessions, {N_STATES} states")

    # Match arbitrary HMM state order across methods to SDP
    print("[INFO] Matching HMM state order ...")
    stc_concat = {
        m: np.concatenate(modes.argmax_time_courses(run_results[m]["alpha"]), axis=0)
        for m in METHODS
    }
    match_order = ["sdp"] + [m for m in METHODS if m != "sdp"]
    orders = modes.match_modes(*[stc_concat[m] for m in match_order], return_order=True)
    order = {match_order[i]: np.asarray(orders[i]) for i in range(len(match_order))}
    for m in METHODS:
        o = order[m]
        run_results[m]["alpha"] = [a[:, o] for a in run_results[m]["alpha"]]
        run_results[m]["covariance"] = run_results[m]["covariance"][o]
        run_results[m]["transition_probability"] = run_results[m]["transition_probability"][o][:, o]

    # Measure match quality (per-state correlation of each method's state time course with SDP's)
    sc = {
        m: np.concatenate(modes.argmax_time_courses(run_results[m]["alpha"]), axis=0).astype(float)
        for m in METHODS
    }
    match_quality = {
        m: np.mean([corr(sc[m][:, k], sc["sdp"][:, k]) for k in range(N_STATES)])
        for m in METHODS
    }

    # Compute and extract per-method group features (heavy; cached)
    cache = os.path.join(outdir, "dynamic_features.npz")
    if os.path.exists(cache):
        print(f"[INFO] Loading cached features from {cache}")
        Z = np.load(cache, allow_pickle=True)
        feats = Z["feats"].item()
        f_axis = Z["f_axis"]
        C = int(Z["n_parcels"])
    else:
        feats = {m: {} for m in METHODS}
        f_axis = None
        for m in METHODS:
            print(f"[INFO] Extracting features for {m} ...")

            # Load data
            data = Data(files[m], picks="misc", reject_by_annotation="omit",
                        sampling_frequency=FS, n_jobs=n_jobs)
            raw_full = data.time_series()
            ts_trim = data.trim_time_series(
                sequence_length=SEQUENCE_LENGTH,
                n_embeddings=N_EMBEDDINGS,
                prepared=False,
            )
            data.delete_dir()
            ts_mt, alpha_mt = align(ts_trim, run_results[m]["alpha"])
            C = ts_mt[0].shape[1]  # number of parcels

            # Compute spectral features
            f_axis, gpsd, gcoh = group_multitaper(ts_mt, alpha_mt, n_jobs)
            gpower = power.variance_from_spectra(f_axis, gpsd, frequency_range=POWER_BAND)
            gcoh_band = connectivity.mean_coherence_from_spectra(
                f_axis, gcoh, frequency_range=COH_BAND
            )
            tde_cov = group_state_cov(
                raw_full, run_results[m]["alpha"], N_STATES, N_EMBEDDINGS,
                n_threads=blas_threads,
            )

            # Save features
            stc = modes.argmax_time_courses(run_results[m]["alpha"])
            feats[m]["fo"] = modes.fractional_occupancies(stc)
            feats[m]["lt"] = modes.mean_lifetimes(stc, sampling_frequency=FS)
            feats[m]["intv"] = modes.mean_intervals(stc, sampling_frequency=FS)
            feats[m]["sr"] = modes.switching_rates(stc, sampling_frequency=FS)
            feats[m]["gpsd"] = gpsd.astype(np.float32)
            feats[m]["gpower"] = gpower.astype(np.float32)
            feats[m]["gcoh"] = gcoh_band.astype(np.float32)
            feats[m]["tde_cov"] = tde_cov
            feats[m]["labels"] = [s.argmax(1).astype(np.int8) for s in stc]  # hard state sequences
        np.savez(cache, feats=feats, f_axis=f_axis, n_parcels=C)
        print(f"[INFO] Cached features to {cache}")

    # Compute state-segmentation agreement and Dice coefficients
    # (4x4 matrix; per-session vs SDP)
    labels = {m: feats[m]["labels"] for m in METHODS}
    agreement = np.eye(len(METHODS))
    dice = np.eye(len(METHODS))
    for a in range(len(METHODS)):
        for b in range(len(METHODS)):
            agreement[a, b] = compute_label_agreement(labels[METHODS[a]], labels[METHODS[b]]).mean()
            dice[a, b] = np.nanmean(compute_label_dice(labels[METHODS[a]], labels[METHODS[b]], N_STATES))
    agree_sess = np.column_stack([compute_label_agreement(labels[m], labels["sdp"]) for m in METHODS])
    dice_sess = np.column_stack([compute_label_dice(labels[m], labels["sdp"], N_STATES) for m in METHODS])

    # Perform Friedman omnibus test on summary-statistics (Holm-corrected across states)
    friedman_chi2 = np.full((len(STAT_NAMES), N_STATES), np.nan)
    friedman_p = np.full((len(STAT_NAMES), N_STATES), np.nan)
    friedman_p_holm = np.full((len(STAT_NAMES), N_STATES), np.nan)
    for si, s in enumerate(STAT_NAMES):
        for k in range(N_STATES):
            cols = [np.asarray(feats[m][s])[:, k] for m in METHODS]
            try:
                res = friedmanchisquare(*cols)
                friedman_chi2[si, k], friedman_p[si, k] = res.statistic, res.pvalue
            except ValueError:
                pass  # e.g., a state with no cross-method variance -> leave NaN
        friedman_p_holm[si] = ut.holm_correct(friedman_p[si])
    # NOTE: A (session, state) pair is not an independent block (states within a session are
    # correlated), so we do NOT flatten session x state. Instead, for each summary statistic
    # we run Friedman across the four methods separately for every state, with the sessions
    # as the matched blocks (n = n_sessions). We then Holm-correct the resulting p-values
    # across the states. Arrays are shaped (n_stats, n_states); rows follow STAT_NAMES.

    # Quantify inter-method similarity of group network features
    cp_mask = ut.cross_parcel_triu(C, N_EMBEDDINGS)
    tri_c = np.triu_indices(C, 1)

    def group_vectors(kind):
        if kind == "tde_cov":
            return {m: np.concatenate([feats[m]["tde_cov"][k][cp_mask] for k in range(N_STATES)])
                    for m in METHODS}
        if kind == "gpsd":
            return {m: feats[m]["gpsd"].reshape(-1) for m in METHODS}
        if kind == "gcoh":
            return {m: np.concatenate([feats[m]["gcoh"][k][tri_c] for k in range(N_STATES)])
                    for m in METHODS}
        if kind == "gpower":
            return {m: feats[m]["gpower"].reshape(-1) for m in METHODS}
        if kind == "tp":
            return {m: run_results[m]["transition_probability"].reshape(-1) for m in METHODS}
        raise ValueError(kind)

    def corr_between_method(kind):
        vec = group_vectors(kind)
        M = np.eye(len(METHODS))
        for a in range(len(METHODS)):
            for b in range(a + 1, len(METHODS)):
                M[a, b] = M[b, a] = corr(vec[METHODS[a]], vec[METHODS[b]])
        return M

    sim = {kind: corr_between_method(kind) for kind in ["tde_cov", "gpsd", "gcoh", "gpower", "tp"]}

    # Get state-matched observation-model covariances and transition matrices
    obs_cov = np.stack([run_results[m]["covariance"] for m in METHODS])
    trans_prob = np.stack([run_results[m]["transition_probability"] for m in METHODS])
    # obs_cov.shape: (n_methods, n_states, n_channels, n_channels)
    # trans_prob.shape: (n_methods, n_states, n_states)

    # Save features for the visualization
    np.savez(
        os.path.join(outdir, "dynamic_results.npz"),
        dataset=dataset, methods=np.array(METHODS), sessions=np.array(sessions),
        n_states=N_STATES, n_embeddings=N_EMBEDDINGS, n_parcels=C,
        best_run=np.array([best_run[m] for m in METHODS]),
        match_quality=np.array([match_quality[m] for m in METHODS]),
        agreement=agreement, dice=dice, agree_sess=agree_sess, dice_sess=dice_sess,
        sim_tde_cov=sim["tde_cov"], sim_psd=sim["gpsd"], sim_coh=sim["gcoh"],
        sim_power=sim["gpower"], sim_tp=sim["tp"],
        friedman_stats=np.array(STAT_NAMES),
        friedman_chi2=friedman_chi2, friedman_p=friedman_p, friedman_p_holm=friedman_p_holm,
        obs_cov=obs_cov, trans_prob=trans_prob,
    )
    print(f"[INFO] Saved results to {os.path.join(outdir, 'dynamic_results.npz')}")

    # Dynamic brain network power and FC maps for SDP sign-flipped data
    try:
        power.save(
            power_map=feats["sdp"]["gpower"],
            mask_file=MASK_FILE,
            parcellation_file=PARCELLATION_FILE,
            filename=os.path.join(outdir, "dynamic_power_sdp.png"),
            subtract_mean=True, plot_kwargs={"symmetric_cbar": True},
        )
        print(f"[INFO] Wrote SDP power maps -> {outdir}/dynamic_power_sdp*.png")
    except Exception as e:
        print(f"[WARN] SDP power-map render skipped: {e}")
    try:
        gcoh_rel = feats["sdp"]["gcoh"].copy()
        gcoh_rel -= gcoh_rel.mean(0, keepdims=True)  # coherence relative to the state mean
        connectivity.save(
            connectivity_map=gcoh_rel,
            parcellation_file=PARCELLATION_FILE,
            threshold=0.97,
            filename=os.path.join(outdir, "dynamic_coh_sdp.png"),
        )
        print(f"[INFO] Wrote SDP coherence connectomes -> {outdir}/dynamic_coh_sdp*.png")
    except Exception as e:
        print(f"[WARN] SDP coherence render skipped: {e}")

    print("[INFO] Feature extraction complete.")
