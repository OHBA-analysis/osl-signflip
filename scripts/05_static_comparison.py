"""Visualize figures on static feature comparison across sign-flipping methods.

Aggregates the per-method features saved by ``04_static_analysis.py`` (run once
per dataset x method) for every dataset and writes into the top-level ``figures/``:

    * ``fig4_tde_covariance``     -- Cam-CAN TDE covariance: the group-level matrix before
                                     vs after flipping (random, SDP), inter-method similarity,
                                     per-subject similarity to template, paired SDP advantage.
    * ``figS4_tde_covariance``    -- Fig 4 reproduced on LEMON and Wakeman-Henson.
    * ``figS5_static_invariance`` -- sign-invariant static features (PSD, AEC) are identical
                                     across methods (max |method - SDP| ~ machine precision),
                                     one row per dataset.

The sign-invariant features (PSD/power/AEC) are for quality control (QC), showing that
sign flipping preserves standard static metrics; only the TDE covariance is sign-sensitive,
so the method comparison runs on that.

NOTE (interpretation): the per-subject TDE-covariance similarity to the template is
essentially the sign-flip objective itself, so Fig 4 is a sanity check that the
*optimized* object behaves as designed -- NOT independent downstream validation. The
independent evidence that the solver choice does not distort the science comes from
comparing the dynamic analysis results.

Env: ``osld`` (or any env with numpy/scipy/matplotlib)
Usage:
    python 05_static_comparison.py        # produces every figure above
"""

# Import packages
import os
import numpy as np
import utils as ut
import matplotlib.pyplot as plt  # utils sets the Agg backend on import
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Define a helper function for loading static features
def load_static_features(dataset):
    """
    Loads each method's static features for a specified dataset, subject-aligned.

    Returns a dict with the per-subject TDE similarity matrix ``sim`` (n, 4) and
    a name->column map ``midx``, the invariance QC deviations, the inter-method
    group-covariance similarity, the group TDE matrices (per method + before-flip),
    and the sign-invariant group features for the invariance panels.
    """
    # Load static features
    features = {}
    for m in ut.METHODS:
        path = os.path.join(ut.data_dir(dataset), f"static_{m}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {path} -- run 04_static_analysis.py {dataset} {m}")
        features[m] = dict(np.load(path, allow_pickle=True))

    # Get common subjects across methods, with order taken from SDP
    subj = {m: features[m]["subjects"].tolist() for m in ut.METHODS}
    common = set.intersection(*[set(subj[m]) for m in ut.METHODS])
    subjects = [s for s in subj["sdp"] if s in common]
    pos = {m: {s: i for i, s in enumerate(subj[m])} for m in ut.METHODS}
    order = {m: [pos[m][s] for s in subjects] for m in ut.METHODS}

    # Get data dimensions and metadata
    C = int(features["sdp"]["n_parcels"])
    n_emb = int(features["sdp"]["n_embeddings"])
    midx = {m: i for i, m in enumerate(ut.METHODS)}

    # Per-subject similarity of the cross-parcel TDE covariance to the template
    sim = np.column_stack([features[m]["tde_sim_template"][order[m]] for m in ut.METHODS])

    # Invariance QC (max |method - SDP| over subjects for the sign-invariant features)
    qc = {}
    for feat in ["psd", "power", "aec"]:
        st = {m: features[m][feat][order[m]] for m in ut.METHODS}
        qc[feat] = max(np.abs(st[m] - st["sdp"]).max() for m in ut.METHODS)

    # Inter-method similarity of the group TDE covariance (cross-parcel entries only)
    mask = ut.cross_parcel_triu(C, n_emb)
    gvec = {m: features[m]["tde_group"][mask] for m in ut.METHODS}
    inter = np.eye(len(ut.METHODS))
    for a in range(len(ut.METHODS)):
        for b in range(a + 1, len(ut.METHODS)):
            r = ut.corr(gvec[ut.METHODS[a]], gvec[ut.METHODS[b]])
            inter[a, b] = inter[b, a] = r

    return {
        "dataset": dataset,
        "n": len(subjects),
        "C": C, "n_emb": n_emb,
        "template": str(features["sdp"]["template"]),
        "sim": sim, "midx": midx,
        "qc": qc, "inter": inter,
        "tde_group": {m: features[m]["tde_group"] for m in ut.METHODS},
        "tde_group_before": features["sdp"].get("tde_group_before"),
        "psd_group": {m: features[m]["psd_group"] for m in ut.METHODS},
        "aec_group": features["sdp"]["aec_group"],
        "f": features["sdp"]["f"],
    }
    # NOTE: `tde_group_before` is identical across all methods.


# ---------------
# Reusable panels
# ---------------

def _box_jitter(ax, values_by_method, rng):
    """
    Box plots with jittered points.
    Mean values are marked by a diamond for each method in DISPLAY order.
    """
    n = next(iter(values_by_method.values())).shape[0]
    for i, m in enumerate(ut.DISPLAY):
        x = values_by_method[m]
        bp = ax.boxplot(x, positions=[i], widths=0.6, patch_artist=True,
                        showfliers=False, zorder=2,
                        medianprops=dict(color="black", lw=1.2))
        bp["boxes"][0].set(facecolor=ut.COLORS[m], alpha=0.45, edgecolor="0.3", lw=0.8)
        ax.scatter(np.full(n, i) + rng.uniform(-0.14, 0.14, n), x,
                   s=4, color=ut.COLORS[m], alpha=0.45, zorder=3, edgecolors="none")
        ax.scatter([i], [x.mean()], marker="D", s=22, color="black", zorder=4)
    ax.set_xticks(range(len(ut.DISPLAY)))
    ax.set_xticklabels([ut.LABELS[m] for m in ut.DISPLAY], ha="center")
    ax.grid(axis="y", alpha=0.25)


def tde_vmax(S):
    """
    Symmetric color limit from the cross-parcel entries of
    the group-level TDE covariance matrices.
    """
    mask = ut.cross_parcel_triu(S["C"], S["n_emb"])
    pool = [S["tde_group"]["random"][mask], S["tde_group"]["sdp"][mask]]
    if S["tde_group_before"] is not None:
        pool.append(S["tde_group_before"][mask])
    return float(np.percentile(np.abs(np.concatenate(pool)), 99))


def draw_tde_matrix(ax, mat, title, v):
    """A single group-level TDE-covariance heatmap."""
    ax.set_title(title)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Embedded dimension (parcel x lag)")
    ax.set_ylabel("Embedded dimension (parcel x lag)")
    return im


def draw_sim_matrix(ax, S, cbar="right"):
    """
    Inter-method similarity of the group-level TDE covariance matrices.

    ``cbar`` positions the colorbar: "right", "bottom", or None (omit it, e.g. when
    another block in the same figure already carries the shared bar).
    """
    di = [S["midx"][m] for m in ut.DISPLAY]
    M = S["inter"][np.ix_(di, di)]
    im = ax.imshow(M, cmap="viridis", vmin=0.88, vmax=1.0, aspect="auto")
    labels = [ut.LABELS[m] for m in ut.DISPLAY]
    ax.set_xticks(range(len(ut.DISPLAY))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(ut.DISPLAY))); ax.set_yticklabels(labels)
    mid = 0.5 * (0.85 + 1.0)  # text-color threshold for the fixed scale
    for i in range(len(ut.DISPLAY)):
        for j in range(len(ut.DISPLAY)):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] < mid else "black")
    ax.set_title("Inter-method Similarity between TDE Covariance")
    if cbar == "right":
        cax = inset_axes(ax, width="4%", height="100%", loc="center left",
                         bbox_to_anchor=(1.02, 0.0, 1, 1), bbox_transform=ax.transAxes,
                         borderpad=0)
        ax.figure.colorbar(im, cax=cax, label="Pearson correlation")
    elif cbar == "bottom":
        cax = inset_axes(ax, width="100%", height="4%", loc="lower center",
                         bbox_to_anchor=(0.0, -0.15, 1, 1), bbox_transform=ax.transAxes,
                         borderpad=0)
        ax.figure.colorbar(im, cax=cax, orientation="horizontal",
                           label="Pearson correlation")


def draw_sim_subject(ax, S, rng):
    """Per-subject TDE similarity to template per method, with Friedman omnibus test."""
    sim, midx = S["sim"], S["midx"]
    fr = ut.friedman_omnibus(sim, midx)
    _box_jitter(ax, {m: sim[:, midx[m]] for m in ut.DISPLAY}, rng)
    ax.set_ylabel("After-flip covariance correlation")
    ax.set_title("Per-subject Similarity")
    txt = (f"Friedman $\\chi^2$ = {fr['chi2']:.0f}\n"
           f"Kendall's $W$ = {fr['kendalls_w']:.2f}\n"
           f"{ut.p_stars(fr['p'])}\n"
           f"(template excluded)")
    if S["dataset"] == "wakeman_henson":
        y, va = 0.5, "center"
    else:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.32 * (hi - lo))  # headroom for the top-corner box
        y, va = 0.975, "top"
    ax.text(0.975, y, txt, transform=ax.transAxes, ha="right", va=va,
            fontsize=6, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="0.7", lw=0.6, alpha=0.92))


def draw_sim_paired(ax, S, rng):
    """
    Paired SDP advantage vs other algorithms.
    The figure reports Holm-corrected Wilcoxon + effect size.
    """
    sim, midx, n = S["sim"], S["midx"], S["n"]
    stats = ut.paired_vs_sdp(sim, midx)
    baselines = [m for m in ut.DISPLAY if m != "sdp"]
    for i, m in enumerate(baselines):
        d = sim[:, midx["sdp"]] - sim[:, midx[m]]
        bp = ax.boxplot(d, positions=[i], widths=0.6, patch_artist=True,
                        showfliers=False, zorder=2,
                        medianprops=dict(color="black", lw=1.2))
        bp["boxes"][0].set(facecolor=ut.COLORS[m], alpha=0.45, edgecolor="0.3", lw=0.8)
        ax.scatter(np.full(n, i) + rng.uniform(-0.14, 0.14, n), d,
                   s=4, color=ut.COLORS[m], alpha=0.4, zorder=3, edgecolors="none")
    ax.axhline(0, color="black", lw=0.8, ls="--", zorder=1)
    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels([ut.LABELS[m] for m in baselines], ha="center")
    ax.set_ylabel(r"$\Delta$ metric (SDP $-$ Baseline)")
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
                         f"$r$ = {s['rank_biserial']:.2f}", va="top", ha="center", fontsize=6)


def draw_psd(ax, S):
    """Group-level PSD for the four methods overlaid."""
    for m in ut.DISPLAY:
        ax.plot(
            S["f"], S["psd_group"][m].mean(0),
            color=ut.COLORS[m], lw=1.6, alpha=0.8, label=ut.LABELS[m],
        )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (a.u.)")
    ax.set_title("Group-level PSD")
    ax.legend(frameon=False, ncol=2)


def draw_aec(ax, S):
    """
    Group AEC connectivity matrix (SDP shown, diagonal blanked).

    The diagonal (self-connection = 1) is set to NaN and the colour scale is
    rescaled to the off-diagonal range, so the off-diagonal pattern is visible.
    Group AEC should ideally be identical across methods.
    """
    A = np.array(S["aec_group"], dtype=float).copy()
    np.fill_diagonal(A, np.nan)
    v = float(np.nanpercentile(np.abs(A), 99))
    im = ax.imshow(A, cmap="RdBu_r", vmin=-v, vmax=v)
    ticks = np.linspace(0, A.shape[0] - 1, 5).round().astype(int)
    ax.set_xticks(ticks); ax.set_xticklabels(ticks + 1)
    ax.set_yticks(ticks); ax.set_yticklabels(ticks + 1)
    ax.set_xlabel("Parcel"); ax.set_ylabel("Parcel")
    ax.set_title("Group-level AEC (for SDP)")
    ax.set_anchor("W")  # left-justify the plot
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.05)
    ax.figure.colorbar(im, cax=cax)


def draw_feature_deviation(ax, S):
    """Max |method - SDP| for the sign-invariant features (log scale)."""
    feats = ["psd", "power", "aec"]  # keys into S["qc"]
    labels = ["PSD", "Power", "AEC"]  # display labels
    vals = [max(S["qc"][fe], 1e-20) for fe in feats]
    x = np.arange(len(feats))
    ax.bar(x, vals, color="#888888")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.axhline(np.finfo(np.float32).eps, ls="--", color="k", lw=1, label="float32 eps")
    ax.set_ylabel("max(|Method - SDP|)")
    ax.set_title("Cross-method Deviation")
    ax.legend(frameon=False)
    for i, fe in enumerate(feats):
        ax.text(i, vals[i], f"{S['qc'][fe]:.0e}", ha="center", va="bottom", fontsize=6.5)


# ---------------------------------
# Figures (includes Fig. 4, S4, S5)
# ---------------------------------

def _tde_block(fig, ax_row, ax_stat, S, rng, v, inter_cbar="right"):
    """
    Fills a (matrix row, stat row) pair for one dataset's TDE-covariance figure
    (used for Fig 4 and S4).

    ``v`` is the shared symmetric color limit for the covariance matrices (unified
    across datasets); ``inter_cbar`` positions the inter-method colorbar.
    """
    specs = [("Before Flipping", S["tde_group_before"]),
             ("After Random Flipping", S["tde_group"]["random"]),
             ("After SDP Flipping", S["tde_group"]["sdp"])]
    ims = [draw_tde_matrix(ax, m, t, v) for ax, (t, m) in zip(ax_row, specs)]
    ims = [im for im in ims if im is not None]
    if ims:
        cax = inset_axes(ax_row[2], width="4%", height="100%", loc="center left",
                         bbox_to_anchor=(1.02, 0.0, 1, 1),
                         bbox_transform=ax_row[2].transAxes, borderpad=0)
        cbar = fig.colorbar(ims[0], cax=cax, label="Covariance")
        cbar.ax.yaxis.set_major_locator(MultipleLocator(0.01))  # 0.01 tick spacing
    draw_sim_matrix(ax_stat[0], S, cbar=inter_cbar)
    draw_sim_subject(ax_stat[1], S, rng)
    draw_sim_paired(ax_stat[2], S, rng)
    for a in list(ax_row) + list(ax_stat):
        a.set_box_aspect(1)  # square, equal-sized panels


def fig4_tde(data, rng, v):
    """Figure 4: Cam-CAN TDE covariance matrices and associated statistics."""
    # Plot figures
    S = data[ut.PRIMARY]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2), constrained_layout=True)
    _tde_block(fig, axes[0], axes[1], S, rng, v, inter_cbar="bottom")

    # Axis settings
    for ax, tag in zip(axes.ravel(), "abcdef"):
        ut.panel_tag(ax, tag)
    ut.save_fig(fig, "fig4_tde_covariance")


def figS4_tde(data, rng, v):
    """Figure S4: Fig 4 reproduced on the two other datasets."""
    # Plot figures
    fig, axes = plt.subplots(
        2 * len(ut.REPRO), 3,
        figsize=(11, 7.0 * len(ut.REPRO)),
        constrained_layout=True,
    )
    for di, ds in enumerate(ut.REPRO):
        S = data[ds]
        # Only the last block carries the shared colorbar
        last = di == len(ut.REPRO) - 1
        _tde_block(fig, axes[2 * di], axes[2 * di + 1], S, rng, v,
                   inter_cbar="bottom" if last else None)
        # axes[2 * di, 0].set_ylabel(ut.DATASET_LABELS[ds], fontsize=9)

    # Axis settings
    for ax, tag in zip(axes.ravel(), "abcdefghijkl"):
        ut.panel_tag(ax, tag)
    ut.save_fig(fig, "figS4_tde_covariance")


def figS5_invariance(data):
    """Figure S5: Sign-invariant features across methods, one row per dataset."""
    # Plot figures
    fig, axes = plt.subplots(len(ut.ALL_DATASETS), 3, figsize=(11, 3.2 * len(ut.ALL_DATASETS)))
    for di, ds in enumerate(ut.ALL_DATASETS):
        S = data[ds]
        draw_psd(axes[di, 0], S)
        draw_aec(axes[di, 1], S)
        draw_feature_deviation(axes[di, 2], S)
        axes[di, 0].set_ylabel(f"{ut.DATASET_LABELS[ds]}\nPSD (a.u.)")

    # Axis settings
    for ax, tag in zip(axes.ravel(), "abcdefghi"):
        ut.panel_tag(ax, tag)
    fig.suptitle("Static Feature Invariance to the Sign-Flipping Method", fontsize=9, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    ut.save_fig(fig, "figS5_static_invariance")


if __name__ == "__main__":
    # Set visual style
    ut.set_style()  # set plot style
    rng = np.random.default_rng(0)  # for jitters; deterministic

    # Load analysis results
    data = {ds: load_static_features(ds) for ds in ut.ALL_DATASETS}
    for ds in ut.ALL_DATASETS:
        S = data[ds]
        print(f"[INFO] {ds}: n = {S['n']}, {S['C']} parcels")
        for feat in ["psd", "power", "aec"]:
            print(f"\tinvariance QC {feat:<6} max|method-SDP| = {S['qc'][feat]:.2e}")

    # Get a limit for unified TDE-covariance color scale across datasets
    v = max(tde_vmax(val) for val in data.values())

    # Generate figures
    fig4_tde(data, rng, v)
    figS4_tde(data, rng, v)
    figS5_invariance(data)
    print("[INFO] Visualization complete.")
