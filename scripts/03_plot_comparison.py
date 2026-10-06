"""Visualize figures on sign-flipping quality and flip decisions.

Reads ``<dataset>/data/analysis_results.npz`` (from ``02_collect_results.py``)
for every dataset and writes the following into the top-level ``figures/``:

    * ``fig1_performance``      -- Sign-flipping performance in Cam-CAN: metric distribution,
                                   per-session, paired SDP advantage, and cross-dataset SDP
                                   performance.
    * ``fig3_flip_decisions``   -- Cam-CAN flip decisions: agreement matrix, per-parcel
                                   disagreement, agreement with the 4-method consensus.
    * ``figS1_performance``     -- Fig 1a-c reproduced on LEMON and Wakeman-Henson.
    * ``figS2_sign_flip_curve`` -- Before and four after-flip curves, one row per dataset.
    * ``figS3_flip_decisions``  -- Fig 3 reproduced on LEMON and Wakeman-Henson.

Env: ``osld`` (or any env with numpy/scipy/matplotlib).
Usage:
    python 03_plot_comparison.py        # produces every figure above
"""

# Import packages
import numpy as np
import utils as ut
import matplotlib.pyplot as plt  # utils sets the Agg backend on import

# ---------------
# Reusable panels
# ---------------

def draw_distribution(ax, D, midx, rng, friedman=None, friedman_center=False):
    """
    After-flip metric distribution per method: box + jitter + mean diamond
    (Fig 1a, S1a, S1d).

    If ``friedman`` (a dict from ``utils.friedman_omnibus``) is passed, the omnibus
    result is reported in an inset box on the right: upper corner by default, or
    vertically centred when ``friedman_center=True`` (useful where the top corner is
    crowded, e.g. the Wakeman-Henson long tail).
    """
    after = D["after"]
    n = after.shape[0]
    for i, m in enumerate(ut.DISPLAY):
        x = after[:, midx[m]]
        bp = ax.boxplot(x, positions=[i], widths=0.6, patch_artist=True,
                        showfliers=False, zorder=2,
                        medianprops=dict(color="black", lw=1.2))
        bp["boxes"][0].set(facecolor=ut.COLORS[m], alpha=0.45, edgecolor="0.3", lw=0.8)
        ax.scatter(np.full(n, i) + rng.uniform(-0.14, 0.14, n), x,
                   s=4, color=ut.COLORS[m], alpha=0.45, zorder=3, edgecolors="none")
        ax.scatter([i], [x.mean()], marker="D", s=20, color="black", zorder=4)
        ax.text(i + 0.22, x.mean(), f"{x.mean():.3f}", va="center", ha="left", fontsize=6.5)
    ax.set_xticks(range(len(ut.DISPLAY)))
    ax.set_xticklabels([ut.LABELS[m] for m in ut.DISPLAY], ha="center")
    ax.set_ylabel("After-flip covariance correlation")
    ax.set_title(f"Sign-Flipping Performance (n = {n})")
    ax.grid(axis="y", alpha=0.25)

    # Report Friedman omnibus test
    if friedman is not None:
        txt = (f"Friedman $\\chi^2$ = {friedman['chi2']:.1f}\n"
               f"Kendall's $W$ = {friedman['kendalls_w']:.2f}\n"
               f"{ut.p_stars(friedman['p'])}\n"
               f"(template excluded)")
        if friedman_center:
            y, va = 0.5, "center"                    # vertically centred (right side)
        else:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.32 * (hi - lo))   # headroom only for the top-corner box
            y, va = 0.975, "top"
        ax.text(0.975, y, txt, transform=ax.transAxes, ha="right", va=va,
                fontsize=6, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="0.7", lw=0.6, alpha=0.92))


def draw_per_session_sorted(ax, D, midx):
    """
    Per-session metric, with sessions sorted by the SDP value (one line per method)
    (Fig 1b, S1b, S1e).
    """
    after = D["after"]
    n = after.shape[0]
    order = np.argsort(after[:, midx["sdp"]])
    xs = np.arange(n)
    for m in ut.DISPLAY:
        ax.plot(xs, after[order, midx[m]], lw=0.8, color=ut.COLORS[m],
                label=ut.LABELS[m], alpha=0.8)
    ax.set_xlabel("Session (sorted by SDP metric)")
    ax.set_ylabel("After-flip covariance correlation")
    ax.set_title("Per-session Performance Metric")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(alpha=0.25)


def draw_paired_comparison(ax, D, midx, rng):
    """
    Paired SDP advantage vs other algorithms (Fig 1c, S1c, S1f).

    The figure reports Holm-corrected Wilcoxon + effect size.
    """
    after = D["after"]
    n = after.shape[0]
    stats = ut.paired_vs_sdp(after, midx)
    baselines = [m for m in ut.DISPLAY if m != "sdp"]
    for i, m in enumerate(baselines):
        d = after[:, midx["sdp"]] - after[:, midx[m]]
        bp = ax.boxplot(d, positions=[i], widths=0.6, patch_artist=True,
                        showfliers=False, zorder=2,
                        medianprops=dict(color="black", lw=1.2))
        bp["boxes"][0].set(facecolor=ut.COLORS[m], alpha=0.45, edgecolor="0.3", lw=0.8)
        ax.scatter(np.full(n, i) + rng.uniform(-0.14, 0.14, n), d,
                   s=4, color=ut.COLORS[m], alpha=0.4, zorder=3, edgecolors="none")
    ax.axhline(0, color="black", lw=0.8, ls="--", zorder=1)
    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels([ut.LABELS[m] for m in baselines], ha="center")
    ax.set_ylabel(r"$\Delta$ metric  (SDP $-$ Baseline)")
    ax.set_title("Paired SDP Advantage")
    ax.grid(axis="y", alpha=0.25)

    # Place headroom for the annotations
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.30 * (hi - lo))

    # Annotate p_holm, win %, and effect size
    ytop = ax.get_ylim()[1]
    for i, m in enumerate(baselines):
        s = stats[m]
        ax.text(i, ytop, f"{ut.p_stars(s['p_holm'])}\nWin {s['winpct']:.0f}%\n"
                         f"$r$ = {s['rank_biserial']:.2f}",
                va="top", ha="center", fontsize=6)


def draw_cross_dataset_sdp(ax, data, rng):
    """SDP per-session metric as one box per dataset (Fig 1d)."""
    n_sessions = []
    for i, ds in enumerate(ut.ALL_DATASETS):
        D, midx = data[ds]
        x = D["after"][:, midx["sdp"]]
        n_sessions.append(x.size)
        bp = ax.boxplot(x, positions=[i], widths=0.6, patch_artist=True,
                        showfliers=False, zorder=2,
                        medianprops=dict(color="black", lw=1.2))
        bp["boxes"][0].set(facecolor=ut.COLORS["sdp"], alpha=0.45, edgecolor="0.3", lw=0.8)
        ax.scatter(np.full(n_sessions[-1], i) + rng.uniform(-0.14, 0.14, n_sessions[-1]), x,
                   s=4, color=ut.COLORS["sdp"], alpha=0.4, zorder=3, edgecolors="none")
    ax.set_xticks(range(len(ut.ALL_DATASETS)))
    ax.set_xticklabels(
        [ut.DATASET_LABELS[d] + f"\n(n = {n})" for d, n in zip(ut.ALL_DATASETS, n_sessions)],
        ha="center",
    )
    ax.set_ylabel("After-flip covariance correlation")
    ax.set_title("SDP Performance Across Datasets")
    ax.grid(axis="y", alpha=0.25)


def draw_sign_flip_curve(ax, D, midx):
    """
    Plots metric curves before and after sign-flipping in original
    session order (Fig S2).
    """
    after = D["after"]
    n = after.shape[0]  # number of sessions
    x = np.arange(n) + 1
    ax.plot(x, D["before"], lw=1.0, color="0.5", ls="--", label="Before", alpha=0.9)
    for m in ut.DISPLAY:
        ax.plot(x, after[:, midx[m]], lw=1.0, color=ut.COLORS[m], label=ut.LABELS[m], alpha=0.9)
    ax.set_ylabel("Covariance correlation")
    ax.set_xlim(0, n + 1)
    ax.grid(alpha=0.25)


def draw_agreement(ax, D, midx):
    """
    Flip-decision agreement matrix (fraction of parcels, up to a global sign)
    (Fig 3a, S3a, S3d).
    """
    idx = [midx[m] for m in ut.DISPLAY]
    A = D["agreement"][np.ix_(idx, idx)]
    labels = [ut.LABELS[m] for m in ut.DISPLAY]
    im = ax.imshow(A, vmin=0.7, vmax=1.0, cmap="magma")
    ax.set_xticks(range(len(ut.DISPLAY)))
    ax.set_yticks(range(len(ut.DISPLAY)))
    ax.set_xticklabels(labels, ha="center")
    ax.set_yticklabels(labels)
    mid = (A.min() + 1.0) / 2
    for i in range(len(ut.DISPLAY)):
        for j in range(len(ut.DISPLAY)):
            ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                    color="white" if A[i, j] < mid else "black")
    ax.set_title("Pairwise Flip Decision Agreement")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def draw_parcel_disagreement(ax, D):
    """
    Per-parcel disagreement rate across methods (parcels sorted, descending)
    (Fig 3b, S3b, S3d).
    """
    pd = D["parcel_disagreement"]
    order = np.argsort(-pd)
    ax.bar(range(pd.shape[0]), pd[order], color="#30638e", alpha=0.85)
    ax.set_xlabel("Parcel (sorted by disagreement)")
    ax.set_ylabel("Fraction of disagreeing sessions")
    ax.set_title(f"Per-Parcel Disagreement Rate ({pd.shape[0]} parcels)")
    ax.grid(axis="y", alpha=0.25)


def draw_consensus(ax, D, midx, rng):
    """
    Per-method agreement with the 4-method consensus (box + jitter)
    (Fig 3c, S3c, S3f).
    """
    cons = D["consensus_agreement"]
    n = cons.shape[0]
    for i, m in enumerate(ut.DISPLAY):
        x = cons[:, midx[m]]
        bp = ax.boxplot(x, positions=[i], widths=0.6, patch_artist=True,
                        showfliers=False, zorder=2,
                        medianprops=dict(color="black", lw=1.2))
        bp["boxes"][0].set(facecolor=ut.COLORS[m], alpha=0.45, edgecolor="0.3", lw=0.8)
        ax.scatter(np.full(n, i) + rng.uniform(-0.14, 0.14, n), x,
                   s=4, color=ut.COLORS[m], alpha=0.45, zorder=3, edgecolors="none")
    ax.set_xticks(range(len(ut.DISPLAY)))
    ax.set_xticklabels([ut.LABELS[m] for m in ut.DISPLAY], ha="center")
    ax.set_ylabel("Fraction of parcels matching consensus")
    ax.set_title("Agreement with 4-method Consensus")
    ax.grid(axis="y", alpha=0.25)


# ----------------------------------------
# Figures (includes Fig. 1, 3, S1, S2, S3)
# ----------------------------------------

def fig1_performance(data, rng):
    """Figure 1: Cam-CAN sign-flipping performance (a-c) + cross-dataset SDP metrics (d)."""
    # Load data and run Friedman test
    D, midx = data[ut.PRIMARY]
    fr = ut.friedman_omnibus(D["after"], midx)

    # Plot figures
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    draw_distribution(axes[0, 0], D, midx, rng, friedman=fr)
    draw_per_session_sorted(axes[0, 1], D, midx)
    draw_paired_comparison(axes[1, 0], D, midx, rng)
    draw_cross_dataset_sdp(axes[1, 1], data, rng)

    # Axis settings
    for ax, tag in zip(axes.ravel(), "abcd"):
        ut.panel_tag(ax, tag)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    ut.save_fig(fig, "fig1_performance")


def fig3_flip_decisions(data, rng):
    """Figure 3: Cam-CAN flip-decision agreement, disagreement, consensus."""
    # Load data
    D, midx = data[ut.PRIMARY]

    # Plot figures
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    draw_agreement(axes[0], D, midx)
    draw_parcel_disagreement(axes[1], D)
    draw_consensus(axes[2], D, midx, rng)

    # Axis settings
    for ax, tag in zip(axes, "abc"):
        ut.panel_tag(ax, tag)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ut.save_fig(fig, "fig3_flip_decisions")


def figS1_performance(data, rng):
    """Figure S1: Fig 1a-c reproduced on the two other datasets."""
    # Plot figures
    fig, axes = plt.subplots(len(ut.REPRO), 3, figsize=(11, 3.4 * len(ut.REPRO)))
    for r, ds in enumerate(ut.REPRO):
        D, midx = data[ds]
        fr = ut.friedman_omnibus(D["after"], midx)
        draw_distribution(axes[r, 0], D, midx, rng, friedman=fr,
                          friedman_center=(ds == "wakeman_henson"))
        draw_per_session_sorted(axes[r, 1], D, midx)
        draw_paired_comparison(axes[r, 2], D, midx, rng)
        axes[r, 0].set_ylabel(f"{ut.DATASET_LABELS[ds]}\nAfter-flip correlation")

    # Axis settings
    for ax, tag in zip(axes.ravel(), "abcdef"):
        ut.panel_tag(ax, tag)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    ut.save_fig(fig, "figS1_performance")


def figS2_sign_flip_curve(data):
    """Figure S2: Before and four after-flip curves; one row per dataset."""
    # Plot figures
    fig, axes = plt.subplots(
        len(ut.ALL_DATASETS), 1, figsize=(9, 2.6 * len(ut.ALL_DATASETS))
    )
    for row, ds in enumerate(ut.ALL_DATASETS):
        D, midx = data[ds]
        draw_sign_flip_curve(axes[row], D, midx)
        axes[row].set_title(f"{ut.DATASET_LABELS[ds]}  (n = {D['after'].shape[0]})")

    # Axis settings
    axes[-1].set_xlabel("Recording sessions")
    axes[0].legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.15))
    for ax, tag in zip(axes, "abc"):
        ut.panel_tag(ax, tag)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    ut.save_fig(fig, "figS2_sign_flip_curve")


def figS3_flip_decisions(data, rng):
    """Figure S3: Fig 3 reproduced on the two other datasets."""
    # Plot figures
    fig, axes = plt.subplots(len(ut.REPRO), 3, figsize=(11, 3.6 * len(ut.REPRO)))
    for r, ds in enumerate(ut.REPRO):
        D, midx = data[ds]
        draw_agreement(axes[r, 0], D, midx)
        draw_parcel_disagreement(axes[r, 1], D)
        draw_consensus(axes[r, 2], D, midx, rng)
        axes[r, 0].set_ylabel(ut.DATASET_LABELS[ds])

    # Axis settings
    for ax, tag in zip(axes.ravel(), "abcdef"):
        ut.panel_tag(ax, tag)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    ut.save_fig(fig, "figS3_flip_decisions")


if __name__ == "__main__":
    # Set visual style
    ut.set_style()  # set plot style
    rng = np.random.default_rng(0)  # for jitters; keep deterministic

    # Load analysis results
    data = {ds: ut.load_analysis_results(ds) for ds in ut.ALL_DATASETS}
    for ds in ut.ALL_DATASETS:
        D, _ = data[ds]
        print(f"[INFO] {ds}: n = {D['after'].shape[0]} sessions, "
              f"{int(D['n_parcels'])} parcels")

    # Generate figures
    fig1_performance(data, rng)
    fig3_flip_decisions(data, rng)
    figS1_performance(data, rng)
    figS2_sign_flip_curve(data)
    figS3_flip_decisions(data, rng)
    print("[INFO] Visualization complete.")
