"""Shared helpers for the sign-flipping analysis and visualization scripts.

The visualization scripts import from here, and figures are written to the
top-level ``figures/`` directory.
"""

# Import packages
import os
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

# Silence per-glyph INFO chatter by fontTools when matplotlib subsets fonts for PDF output
# (osl-dynamics/osl-ephys raise the root logger to INFO, so these would otherwise print)
logging.getLogger("fontTools").setLevel(logging.WARNING)

# Set directories
BASE_DIR = "/well/woolrich/users/olt015/Cho2026_SignFlip"  # project root
FIG_DIR = os.path.join(BASE_DIR, "figures")  # top-level figure directory

# Define datasets
PRIMARY = "camcan"
REPRO = ["lemon", "wakeman_henson"]
ALL_DATASETS = [PRIMARY] + REPRO
DATASET_LABELS = {
    "camcan": "Cam-CAN",
    "lemon": "LEMON",
    "wakeman_henson": "Wakeman-Henson",
}
# NOTE: Cam-CAN is the primary dataset (main figures); LEMON and Wakeman-Henson
# are used to show reproducibility and consistency of the results (supplements).

# Define visualization constants
METHODS = ["random", "spectral", "sdp", "belief_propagation"]  # canonical method identifiers
DISPLAY = ["sdp", "random", "spectral", "belief_propagation"]  # fixed display order
COLORS = {
    "sdp": "#d1495b",                # red
    "random": "#30638e",             # blue
    "belief_propagation": "#edae49", # amber
    "spectral": "#66a182",           # green
}
LABELS = {
    "sdp": "SDP",
    "random": "Random",
    "belief_propagation": "BP",
    "spectral": "Spectral",
}


# -------------------------------
# Visualization and data handling
# -------------------------------

def data_dir(dataset):
    """Fetches per-dataset artefact directory: ``<dataset>/data``."""
    return os.path.join(BASE_DIR, dataset, "data")


def set_style():
    """Applies the rcParams for the publication-ready figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,  # embed as TrueType (editable text in Illustrator)
        "ps.fonttype": 42,
    })


def panel_tag(ax, letter, dx=-0.02, dy=1.05):
    """Places a bold lower-case panel tag just outside the axis's top-left."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="right")


def save_fig(fig, name):
    """Saves a figure as a vector PDF + high-dpi PNG into ``figures/``."""
    os.makedirs(FIG_DIR, exist_ok=True)
    pdf = os.path.join(FIG_DIR, f"{name}.pdf")
    png = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=500)
    plt.close(fig)
    print(f"[INFO] Saved {pdf}.")
    print(f"[INFO] Saved {png}.")


def load_analysis_results(dataset):
    """
    Loads ``<dataset>/data/analysis_results.npz`` and a name->column map.

    Returns ``(D, midx)`` where ``D`` is a plain dict (with ``dataset`` decoded
    to ``str``) and ``midx`` maps each method name to its column in ``D['after']``.
    Always index ``after``/``flips``/etc. through ``midx`` -- the stored ``methods``
    order is not always guaranteed identical across datasets.
    """
    path = os.path.join(data_dir(dataset), "analysis_results.npz")
    D = dict(np.load(path, allow_pickle=True))
    D["dataset"] = str(D["dataset"])
    midx = {str(m): i for i, m in enumerate(D["methods"])}
    return D, midx


# -----------------
# Post-hoc analysis
# -----------------

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


# --------------------------------------------------------
# Statistics (recomputed from the raw per-subject metrics)
# --------------------------------------------------------

def holm_correct(pvals):
    """Holm-Bonferroni step-down correction. NaNs are ignored (kept as NaN)."""
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)
    finite = np.where(np.isfinite(pvals))[0]
    m = len(finite)
    order = finite[np.argsort(pvals[finite])]
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min((m - rank) * pvals[idx], 1.0)
        running = max(running, adj)  # enforce monotonic non-decreasing
        out[idx] = running
    return out


def rank_biserial(diff):
    """
    Matched-pairs rank-biserial effect size for a Wilcoxon signed-rank test.

    ``r = (T+ - T-) / (T+ + T-)`` over the non-zero differences, where ``T+/T-``
    are the summed ranks of the positive/negative ``|diff|``. Range [-1, 1];
    positive means the first condition (SDP) tends to exceed the baseline.
    """
    d = diff[diff != 0]
    if d.size == 0:
        return np.nan
    ranks = rankdata(np.abs(d))
    t_pos = ranks[d > 0].sum()
    t_neg = ranks[d < 0].sum()
    total = t_pos + t_neg
    return (t_pos - t_neg) / total if total > 0 else np.nan


def friedman_omnibus(scores, midx=None):
    """
    Friedman test across the four methods, with Kendall's W effect size.

    ``scores``: (n_subjects, n_methods) per-subject metric matrix, one column per
    method (indexed by ``midx``). The Friedman test treats subjects as the matched
    blocks and the method as the repeated measure.
    
    Returns a dict with ``chi2``, ``p``, ``kendalls_w`` and ``n`` (subjects).
    """
    cols = [scores[:, midx[m]] for m in METHODS] if midx else \
           [scores[:, i] for i in range(scores.shape[1])]
    chi2, p = friedmanchisquare(*cols)
    n, k = scores.shape
    w = chi2 / (n * (k - 1))  # Kendall's W in [0, 1]
    return {"chi2": float(chi2), "p": float(p), "kendalls_w": float(w), "n": n}


def paired_vs_sdp(scores, midx):
    """
    Wilcoxon signed-rank of SDP vs each baseline, Holm-corrected.

    ``scores``: (n_subjects, n_methods) per-subject metric matrix, one column per
    method (indexed by ``midx``). For each baseline, the SDP column and that
    baseline's column are the two *paired* samples over subjects, so this is a
    matched signed-rank test on the per-subject differences ``SDP - baseline``.

    Returns a dict with baseline method keys (in ``DISPLAY`` order, SDP omitted)
    with per-comparison ``meanD``, ``medianD``, ``winpct`` (SDP > baseline),
    ``p`` (raw two-sided), ``p_holm`` and ``rank_biserial``. ``meanD``/``medianD``
    are SDP - baseline.
    """
    sdp = scores[:, midx["sdp"]]
    baselines = [m for m in DISPLAY if m != "sdp"]
    raw_p = []
    rows = {}
    for m in baselines:
        diff = sdp - scores[:, midx[m]]
        try:
            p = float(wilcoxon(sdp, scores[:, midx[m]]).pvalue)
        except ValueError:  # all differences zero
            p = np.nan
        raw_p.append(p)
        rows[m] = {
            "meanD": float(diff.mean()),
            "medianD": float(np.median(diff)),
            "winpct": float(np.mean(diff > 0) * 100.0),
            "p": p,
            "rank_biserial": float(rank_biserial(diff)),
        }
    for m, ph in zip(baselines, holm_correct(raw_p)):
        rows[m]["p_holm"] = float(ph)
    return rows


def p_stars(p):
    """Compact significance annotation for a (corrected) p-value."""
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return f"$p$ = {p:.1e}"
    return f"$p$ = {p:.3f}"
