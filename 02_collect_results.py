"""Accumulate sign-flipping results and compute comparison statistics.

Reads the per-subject results produced by ``01_sign_flip.py``:

* the before/after covariance-correlation metric from
  ``{dataset}/src_{method}/src_report/{subject}/data.pkl``
  (``metrics[0]`` = before flip; ``metrics[-1]`` = the metric of the *actually
  applied* flips -- note this is NOT ``metrics.max()``, which for belief
  propagation is a transient mid-trajectory peak the solver moved past);
* the final per-parcel sign-flip decisions, recovered *exactly* from the saved
  parcel time courses. Because the sign-flipped file is ``sflip = f * orig``
  channel-wise, ``f_p = sign(<orig_p, sflip_p>)``.

Usage
-----
    python 02_collect_results.py <dataset>       # e.g. lemon, wakeman_henson
"""

# Import packages
import argparse
import glob
import mne
import numpy as np
import os
import pickle
from scipy.stats import wilcoxon

mne.set_log_level("ERROR")

# Set directories
BASE_DIR = "/well/woolrich/users/olt015/Cho2026_SignFlip"

# Analysis constants
METHODS = ["random", "spectral", "sdp", "belief_propagation"]
N_METHODS = len(METHODS)
FLIP_SLICE = 100_000  # samples read to recover flip signs (sign is exact for any nonzero window)

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect sign-flipping metrics and flip decisions across methods.",
    )
    parser.add_argument(
        "dataset",
        help="Dataset name (e.g., lemon)."
    )
    return parser.parse_args()

# Define helper functions
def load_metrics(dataset, method):
    """
    Returns metrics before and after applying a sign-flipping algorithm
    for each subject and method.
        - ``before`` = unflipped metric (``metrics[0]``)
        - ``after`` = metric of the applied flips (``metrics[-1]``)
    """
    out = {}
    pattern = os.path.join(
        BASE_DIR, dataset, f"src_{method}/src_report/*/data.pkl"
    )
    template = None
    for f in sorted(glob.glob(pattern)):
        subject = f.split("/")[-2]
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        m = np.asarray(data["metrics"], dtype=float).ravel()
        out[subject] = (m[0], m[-1])
        template = data.get("template", template)
    return out, template


def _pick_parcels(raw):
    """
    Selects channels whose name contains ``"parcel"`` (falling back to all
    ``misc`` channels); excludes non-parcel channels such as the stim /
    trigger channels present in the Wakeman-Henson data (STI101/201/301).
    """
    parc = [ch for ch in raw.ch_names if "parcel" in ch]
    return parc if parc else "misc"


def recover_flips(dataset, method, subject):
    """
    Recovers the applied ±1 per-parcel flip vector from the saved files.

    ``sflip_tc[p] = f_p * orig_tc[p]`` exactly, so the sign of the (partial)
    inner product over time recovers ``f_p``.
    """
    # Load data time series
    src_dir = os.path.join(BASE_DIR, f"{dataset}/src_{method}/{subject}")
    orig_ts = mne.io.read_raw_fif(
        os.path.join(src_dir, "parc/lcmv-parc-raw.fif"),
    )
    sflip_ts = mne.io.read_raw_fif(
        os.path.join(src_dir, f"{subject}_sflip_lcmv-parc-raw.fif"),
    )

    # Select the time window to analyze
    picks = _pick_parcels(orig_ts)
    stop = min(FLIP_SLICE, orig_ts.n_times)
    orig = orig_ts.get_data(
        picks=picks, reject_by_annotation="omit", stop=stop,
    )
    sflip = sflip_ts.get_data(
        picks=picks, reject_by_annotation="omit", stop=stop,
    )

    # Recover the flip decisions
    dot = np.sum(orig * sflip, axis=1)
    flips = np.sign(dot)
    flips[flips == 0] = 1.0  # np.sign(0) -> 0; treat as +1
    return flips.astype(np.int8)


def matrix_lines(mat, fmt, diag_dash=False):
    """Formats a METHODS x METHODS matrix as aligned text rows (with header)."""
    lines = ["  " + " " * 20 + "".join(f"{m[:8]:>9}" for m in METHODS)]
    for a in range(N_METHODS):
        row = "  " + f"{METHODS[a]:<20}"
        for b in range(N_METHODS):
            row += f"{'--':>9}" if diag_dash and a == b else format(mat[a, b], fmt)
        lines.append(row)
    return lines


# COLLECT RESULTS
if __name__ == "__main__":
    # Get input arguments
    args = parse_args()
    dataset = args.dataset
    print(f"[INFO] Dataset: {dataset}")

    # Load metrics before and after sign-flipping for every method
    metrics, template = {}, None
    for method in METHODS:
        metrics[method], tmpl = load_metrics(dataset, method)
        template = tmpl or template
    print(f"[INFO] Template subject/session: {template}")
    # NOTE: The template subject/session should be identical for all methods given
    #       the same dataset.

    # Confirm common subjects across all four methods, excluding the template
    subjects = sorted(set.intersection(*[set(metrics[m]) for m in METHODS]))
    subjects = [s for s in subjects if s != template]
    n_subjects = len(subjects)
    print(f"[INFO] Common subjects (template excluded): {n_subjects}")

    # Get metrics (i.e., correlation between covariance matrices)
    before = np.array([metrics["sdp"][s][0] for s in subjects])  # same across all methods
    after = np.column_stack(
        [[metrics[m][s][1] for s in subjects] for m in METHODS]
    )  # after.shape: (n_subjects, N_METHODS)

    for m in METHODS:
        before_m = np.array([metrics[m][s][0] for s in subjects])
        max_dev = np.max(np.abs(before_m - before))
        assert max_dev < 1e-6, f"before-flip metric differs for {m}: max deviation = {max_dev:.2e}"

    # Recover the applied flip decisions for every subject and method
    print("[INFO] Recovering flip vectors from saved parcel time courses ...")
    flips = None
    for mi, method in enumerate(METHODS):
        for si, subject in enumerate(subjects):
            f = recover_flips(dataset, method, subject)
            if flips is None:
                flips = np.zeros((N_METHODS, n_subjects, f.shape[0]), dtype=np.int8)
            flips[mi, si] = f
        print(f"\t{method}: done")
    n_parcels = flips.shape[-1]

    # Compute statistics on the after-flip metrics
    summary = np.column_stack([
        after.mean(axis=0), np.median(after, axis=0), after.std(axis=0),
        after.min(axis=0), after.max(axis=0),
    ])  # summary.shape: (N_METHODS, n_statistics)

    sdp = METHODS.index("sdp")
    paired_meanD = np.full(N_METHODS, np.nan)
    paired_medianD = np.full(N_METHODS, np.nan)
    paired_winpct = np.full(N_METHODS, np.nan)
    paired_wilcoxp = np.full(N_METHODS, np.nan)
    for mi, method in enumerate(METHODS):
        if mi == sdp:
            continue
        diff = after[:, sdp] - after[:, mi]
        paired_meanD[mi] = diff.mean()
        paired_medianD[mi] = np.median(diff)
        paired_winpct[mi] = np.mean(diff > 0) * 100.0
        try:
            paired_wilcoxp[mi] = wilcoxon(after[:, sdp], after[:, mi]).pvalue
        except ValueError:
            paired_wilcoxp[mi] = np.nan

    # Compute win-rate matrix (row > col, %); diagonal is 0 (a > a is never true)
    winrate = (after[:, :, None] > after[:, None, :]).mean(axis=0) * 100.0

    # Compute pairwise agreement matrix (up to global sign), averaged over subjects
    same = (flips[:, None] == flips[None, :]).mean(axis=-1)
    # shape: (N_METHODS, N_METHODS, n_subjects)
    agreement = np.maximum(same, 1.0 - same).mean(axis=-1)

    # Gauge-align each subject's flip vectors to SDP
    aligned = flips.copy().astype(np.int8)
    for s in range(n_subjects):
        ref = flips[sdp, s]
        for mi in range(N_METHODS):
            if mi == sdp:
                continue
            if np.mean(aligned[mi, s] == ref) < 0.5:
                aligned[mi, s] = -aligned[mi, s]

    # Compute majority-vote consensus
    consensus = np.sign(aligned.sum(axis=0))  # (n_subjects, n_parcels); ties -> 0
    consensus[consensus == 0] = 1
    consensus = consensus.astype(np.int8)

    # Compute per-parcel disagreement rate (fraction of subjects where the four
    # methods are not unanimous on that parcel)
    parcel_disagreement = np.mean(
        (aligned != aligned[0:1]).any(axis=0), axis=0
    )  # shape: (n_parcels,)

    # Compute per-method consensus agreement (fraction of parcels matching the consensus)
    consensus_agreement = np.column_stack([
        np.mean(aligned[mi] == consensus, axis=1) for mi in range(N_METHODS)
    ])  # shape: (n_subjects, N_METHODS)

    # Save results
    out_dir = os.path.join(BASE_DIR, f"{dataset}/figures")
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, "analysis_results.npz")
    np.savez(
        save_path,
        dataset=dataset,
        methods=np.array(METHODS),
        subjects=np.array(subjects),
        template=template,
        n_parcels=n_parcels,
        before=before,
        after=after,
        flips=flips,
        summary=summary,
        paired_meanD=paired_meanD,
        paired_medianD=paired_medianD,
        paired_winpct=paired_winpct,
        paired_wilcoxp=paired_wilcoxp,
        winrate=winrate,
        agreement=agreement,
        parcel_disagreement=parcel_disagreement,
        consensus_agreement=consensus_agreement,
    )
    print(f"[INFO] Saved results to {save_path}")

    # Print and save human-readable summary
    lines = []
    lines.append(f"Sign-flipping method comparison -- {dataset}")
    lines.append(f"n = {n_subjects} subjects (template {template} excluded), {n_parcels} parcels")
    lines.append(f"before-flip metric: mean {before.mean():.4f} | median {np.median(before):.4f}\n")
    lines.append(f"{'method':<20}{'mean':>8}{'median':>8}{'std':>8}{'min':>8}{'max':>8}")
    order = np.argsort(-summary[:, 0])  # sort methods by mean after-flip correlation across subjects
    for mi in order:
        s = summary[mi]
        lines.append(
            f"{METHODS[mi]:<20}{s[0]:>8.4f}{s[1]:>8.4f}{s[2]:>8.4f}{s[3]:>8.4f}{s[4]:>8.4f}\n"
        )
    lines.append("Paired vs SDP (positive = SDP better), Wilcoxon two-sided:")
    lines.append(f"\t{'baseline':<20}{'meanD':>10}{'medD':>10}{'SDPwin%':>9}{'wilcox_p':>12}")
    for mi, method in enumerate(METHODS):
        if mi == sdp:
            continue
        lines.append(
            f"\t{method:<20}{paired_meanD[mi]:>+10.5f}{paired_medianD[mi]:>+10.5f}"
            f"{paired_winpct[mi]:>9.1f}{paired_wilcoxp[mi]:>12.2e}\n"
        )
    lines.append("Win-rate matrix (row > col, %):")
    lines += matrix_lines(winrate, ">9.1f", diag_dash=True)
    lines.append("")
    lines.append("Flip-decision agreement matrix (up to global sign, mean over subjects):")
    lines += matrix_lines(agreement, ">9.3f")
    lines.append("")
    lines.append("Per-method agreement with the 4-method consensus (mean over subjects):")
    for mi, method in enumerate(METHODS):
        lines.append(f"\t{method:<20}{consensus_agreement[:, mi].mean():>8.3f}")
    text = "\n".join(lines)
    print("\n" + text)
    txt_path = os.path.join(out_dir, "summary_stats.txt")
    with open(txt_path, "w") as fh:
        fh.write(text + "\n")
    print(f"\n[INFO] Saved summary to {txt_path}")
