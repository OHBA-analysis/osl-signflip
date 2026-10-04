"""Visualize figures on computational cost and seed-stability.

Reads the per-dataset ``benchmark_cost.npz`` (from ``09_benchmark_cost.py``) and
``seed_stability.npz`` (from ``10_seed_stability.py``) for every dataset and writes
into the top-level ``figures/``:

    * ``fig2_cost``            -- per-session wall-clock and analytic FLOPs per method,
                                  with the geometric-mean speed-up vs random annotated.
    * ``figS8_seed_stability`` -- run-to-run spread of the after-flip metric across RNG
                                  seeds for the randomized methods (i.e., random, SDP).

Env: ``osld`` (or any env with numpy/matplotlib)
Usage:
    python 11_cost_stability_comparison.py        # produces every figure above
"""

# Import packages
import os
import numpy as np
import utils as ut
import matplotlib.pyplot as plt  # utils sets the Agg backend on import
from matplotlib.lines import Line2D


# Distinct marker per dataset (one series per dataset in the cost panels).
MARKERS = {"camcan": "o", "lemon": "s", "wakeman_henson": "^"}


# Define a helper function for loading npz file
def load_npz(dataset, name):
    path = os.path.join(ut.data_dir(dataset), name)
    return dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else None


# ---------------
# Reusable Panels
# ---------------

def _cost_panel(ax, bench, avail, value_fn, ylabel, annotate_speedup=False):
    """One method-by-dataset cost panel on a log-axis."""
    offsets = np.linspace(-0.24, 0.24, len(avail)) if len(avail) > 1 else [0.0]
    for di, ds in enumerate(avail):
        B = bench[ds]
        for mi, m in enumerate(ut.DISPLAY):
            med, lo, hi = value_fn(B, m)
            yerr = None if lo is None else [[med - lo], [hi - med]]
            ax.errorbar(mi + offsets[di], med, yerr=yerr, marker=MARKERS[ds],
                        color=ut.COLORS[m], ms=6, capsize=2, lw=1, mec="0.3", mew=0.4)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ut.DISPLAY)))
    ax.set_xticklabels([ut.LABELS[m] for m in ut.DISPLAY])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, which="both")
    if annotate_speedup:
        ytop = ax.get_ylim()[1]
        for mi, m in enumerate(ut.DISPLAY):
            if m == "random":
                continue
            su = [np.median(bench[ds]["random_total"]) / np.median(bench[ds][f"{m}_total"])
                  for ds in avail]
            gm = float(np.exp(np.mean(np.log(su))))  # geometric mean of per-dataset speed-ups
            ax.text(mi, ytop, f"{gm:.0f}$\\times$", ha="center", va="top",
                    fontsize=7, color="0.2")


# -----------------------------
# Figures (includes Fig. 2, S8)
# -----------------------------

def fig2_cost(bench, avail):
    """Figure 2: Wall-clock (measured) and FLOPs (analytic) per method."""
    def wall(B, m):
        tot = B[f"{m}_total"] * 1e3  # unit: ms
        return np.median(tot), *np.percentile(tot, [25, 75])

    def flops(B, m):
        return float(B[f"flops_{m}_total"]), None, None

    # Plot figures
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    _cost_panel(axes[0], bench, avail, wall, "Median per-session time (ms)",
                annotate_speedup=True)
    axes[0].set_title("Wall-clock (measured; single-threaded)")
    _cost_panel(axes[1], bench, avail, flops, "Analytic FLOPs (total)")
    axes[1].set_title("FLOPs (analytic; BP/SDP at iteration caps)")

    # Add dataset-marker legend
    handles = [Line2D([0], [0], marker=MARKERS[ds], color="0.3", ls="none",
                      label=ut.DATASET_LABELS[ds]) for ds in avail]
    axes[1].legend(
        handles=handles, frameon=False, fontsize=7,
        title="Dataset", title_fontsize=7,
    )
    for ax, tag in zip(axes, "ab"):
        ut.panel_tag(ax, tag)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ut.save_fig(fig, "fig2_cost")


def figS8_seed_stability(seed, avail, rng):
    """Figure S8: Per-session spread of the after-flip metric across RNG seeds."""
    # Set parameters
    methods = ["random", "sdp"]  # seed-dependent methods only
    floor = 1e-7                 # makes exact-zero SDP spreads visible on a log axis

    # Plot figures
    fig, axes = plt.subplots(1, len(avail), figsize=(3.6 * len(avail), 4.0), squeeze=False)
    for di, ds in enumerate(avail):
        S = seed[ds]
        ax = axes[0, di]
        n_seeds = len(S["seeds"])
        for i, m in enumerate(methods):
            std = np.maximum(S[f"metric_{m}"].std(axis=1), floor)  # across seeds
            n = std.size
            bp = ax.boxplot(std, positions=[i], widths=0.6, patch_artist=True,
                            showfliers=False, zorder=2, medianprops=dict(color="black", lw=1.2))
            bp["boxes"][0].set(facecolor=ut.COLORS[m], alpha=0.45, edgecolor="0.3", lw=0.8)
            ax.scatter(np.full(n, i) + rng.uniform(-0.14, 0.14, n), std,
                       s=7, color=ut.COLORS[m], alpha=0.5, zorder=3, edgecolors="none")
        ax.set_yscale("log")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([ut.LABELS[m] for m in methods])
        ax.set_title(f"{ut.DATASET_LABELS[ds]}")
        ax.grid(axis="y", alpha=0.25, which="both")
        if di == 0:
            ax.set_ylabel(rf"$\sigma$(after-flip correlation) (over {n_seeds} seeds)")
    for ax, tag in zip(axes.ravel(), "abc"):
        ut.panel_tag(ax, tag)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    ut.save_fig(fig, "figS8_seed_stability")


if __name__ == "__main__":
    # Set visual style
    ut.set_style()  # set plot style
    rng = np.random.default_rng(0)  # for jitters; deterministic

    # Load analysis results
    cost = {
        ds: b for ds in ut.ALL_DATASETS
        if (b := load_npz(ds, "benchmark_cost.npz"))
    }
    stability = {
        ds: s for ds in ut.ALL_DATASETS
        if (s := load_npz(ds, "seed_stability.npz"))
    }

    # Generate figures
    if cost:
        avail = [ds for ds in ut.ALL_DATASETS if ds in cost]
        print(f"[INFO] Cost benchmark available: {avail}")
        fig2_cost(cost, avail)
    else:
        print("[WARN] No benchmark_cost.npz found. Skipping Fig 2 (run 09_benchmark_cost.py).")

    if stability:
        avail = [ds for ds in ut.ALL_DATASETS if ds in stability]
        print(f"[INFO] Seed stability available: {avail}")
        figS8_seed_stability(stability, avail, rng)
    else:
        print("[WARN] No seed_stability.npz found. Skipping S9 (run 10_seed_stability.py).")

    print("[INFO] Visualization complete.")
