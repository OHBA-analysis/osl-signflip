"""Visualize figures on static feature comparison across sign-flipping methods.

Aggregates the per-method features saved by ``04_static_analysis.py`` (run once
per dataset x method) for every dataset and writes into the top-level ``figures/``:

    * ``fig4_tde_covariance``     -- Cam-CAN TDE covariance: the group-level matrix before
                                     vs. after flipping and the inter-method similarity.
    * ``figS4_tde_covariance``    -- Fig 4 reproduced on LEMON and Wakeman-Henson.
    * ``figS5_static_invariance`` -- sign-invariant static features (PSD, AEC) are identical
                                     across methods (max |method - SDP| ~ machine precision),
                                     one row per dataset.

The sign-invariant features (PSD/power/AEC) are for quality control (QC), showing that
sign flipping preserves standard static metrics; only the TDE covariance is sign-sensitive,
so the method comparison runs on that.

NOTE (interpretation): Fig 4 is a qualitative display of the *optimized* object (the group
TDE covariance) and its cross-method agreement -- NOT independent downstream validation. The
independent evidence that the solver choice does not distort the science comes from the
dynamic analysis (Fig 5).

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


# Define a helper function for loading static features
def load_static_features(dataset):
    """
    Loads each method's static features for a specified dataset, session-aligned.

    Returns a dict with a name->column map ``midx``, the invariance QC deviations,
    the inter-method group-covariance similarity, the group TDE matrices (per method
    + before-flip), and the sign-invariant group features for the invariance panels.
    """
    # Load static features
    features = {}
    for m in ut.METHODS:
        path = os.path.join(ut.data_dir(dataset), f"static_{m}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {path} -- run 04_static_analysis.py {dataset} {m}")
        features[m] = dict(np.load(path, allow_pickle=True))

    # Get common sessions across methods, with order taken from SDP
    sess = {m: features[m]["sessions"].tolist() for m in ut.METHODS}
    common = set.intersection(*[set(sess[m]) for m in ut.METHODS])
    sessions = [s for s in sess["sdp"] if s in common]
    pos = {m: {s: i for i, s in enumerate(sess[m])} for m in ut.METHODS}
    order = {m: [pos[m][s] for s in sessions] for m in ut.METHODS}

    # Get data dimensions and metadata
    C = int(features["sdp"]["n_parcels"])
    n_emb = int(features["sdp"]["n_embeddings"])
    midx = {m: i for i, m in enumerate(ut.METHODS)}

    # Invariance QC (max |method - SDP| over sessions for the sign-invariant features)
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
        "n": len(sessions),
        "C": C, "n_emb": n_emb,
        "template": str(features["sdp"]["template"]),
        "midx": midx, "qc": qc, "inter": inter,
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
    im = ax.imshow(M, cmap="viridis", vmin=0.88, vmax=1.0)
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
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.05)
        ax.figure.colorbar(im, cax=cax, label="Pearson correlation")
    elif cbar == "bottom":
        cax = make_axes_locatable(ax).append_axes("bottom", size="4%", pad=0.1)
        ax.figure.colorbar(im, cax=cax, orientation="horizontal",
                           label="Pearson correlation")


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

def _tde_block(fig, axes, S, v):
    """
    Fills a 2x2 panel block for one dataset's TDE-covariance figure
    (used for Fig 4 and S4).

    Panels:
        [0, 0] After Random Flipping   [0, 1] After SDP Flipping
        [1, 0] Before Flipping         [1, 1] Inter-method similarity

    The three covariance heatmaps share the symmetric color limit ``v`` (unified across
    datasets) and one vertical colorbar; the inter-method similarity panel carries its
    own vertical colorbar.
    """
    # Group-level TDE covariance matrices before and after sign-flipping
    specs = [(axes[0, 0], "After Random Flipping", S["tde_group"]["random"]),
             (axes[0, 1], "After SDP Flipping",    S["tde_group"]["sdp"]),
             (axes[1, 0], "Before Flipping",       S["tde_group_before"])]
    ims = [draw_tde_matrix(ax, m, t, v) for ax, t, m in specs]

    # Inter-method similarity of the group covariance matrices
    draw_sim_matrix(axes[1, 1], S, cbar="right")

    # Plot shared vertical covariance colorbar
    cax = make_axes_locatable(axes[0, 1]).append_axes("right", size="4%", pad=0.05)
    cbar = fig.colorbar(ims[0], cax=cax, label="Covariance")
    cbar.ax.yaxis.set_major_locator(MultipleLocator(0.01))  # 0.01 tick spacing

    for a in axes.ravel():
        a.set_box_aspect(1)  # square, equal-sized panels


def fig4_tde(data, v):
    """Figure 4: Cam-CAN TDE covariance matrices and inter-method similarity."""
    # Plot figures
    S = data[ut.PRIMARY]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.2), constrained_layout=True)
    _tde_block(fig, axes, S, v)

    # Axis settings
    for ax, tag in zip(axes.ravel(), "abcd"):
        ut.panel_tag(ax, tag)
    ut.save_fig(fig, "fig4_tde_covariance")


def figS4_tde(data, v):
    """Figure S4: Fig 4 reproduced on the two other datasets."""
    # Plot figures
    fig, axes = plt.subplots(
        2 * len(ut.REPRO), 2,
        figsize=(7.6, 7.2 * len(ut.REPRO)),
        constrained_layout=True,
    )
    for di, ds in enumerate(ut.REPRO):
        _tde_block(fig, axes[2 * di:2 * di + 2, :], data[ds], v)

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
    fig4_tde(data, v)
    figS4_tde(data, v)
    figS5_invariance(data)
    print("[INFO] Visualization complete.")
