"""Visualize figures on dynamic feature comparison across sign-flipping methods.

Reads the per-dataset dynamic features saved by ``07_dynamic_features.py``
(``dynamic_results.npz`` + ``dynamic_features.npz``) and writes into the top-level
``figures/``:

    * ``fig5_dynamics``              -- Cam-CAN: Dice between state time courses, per-session Dice
                                        with SDP, inter-method state TDE-covariance similarity, and
                                        per-state summary statistics (fractional occupancy / lifetime
                                        / switching rate) across methods.
    * ``figS6_hmm_networks``         -- Cam-CAN SDP HMM networks: per-state PSD, power, and
                                        coherence.
    * ``tableS1_dynamic_invariance`` -- inter-method similarity (correlation with SDP) of state
                                        PSD, power, coherence, and transition probabilities,
                                        written as a Markdown table.
    * ``figS7_obs_cov``              -- Cam-CAN observation-model state covariances per method
                                        (qualitative; each method has its own PCA basis).

Note that these figures use the primary dataset (Cam-CAN) and that dynamic features were
extracted from the best TDE-HMM run.

Env: ``osld`` (requires osl-dynamics for source-space visualization)
Usage:
    python 08_dynamic_comparison.py        # produces every figure above
"""

# Import packages
import io
import os
import numpy as np
import utils as ut
import matplotlib.pyplot as plt  # utils sets the Agg backend on import
import matplotlib.ticker as mticker
from osl_dynamics.analysis import connectivity
from osl_dynamics.utils import plotting as osld_plotting


# Set files for brain parcellations and source space visualizations
MASK_FILE = "MNI152_T1_8mm_brain.nii.gz"
PARCELLATION_FILE = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"  # 38 Giles


# Define a helper function for loading dynamic features
def load_dynamic_features(dataset):
    """
    Loads each dataset's dynamic results and group-level features for
    visualization.
    """
    # Load dynamic features
    d = ut.data_dir(dataset)
    results = dict(np.load(os.path.join(d, "dynamic_results.npz"), allow_pickle=True))
    features = np.load(os.path.join(d, "dynamic_features.npz"), allow_pickle=True)
    raw_feats = features["feats"].item()

    # Keep only the data arrays the figures use
    feats = {
        m: {k: raw_feats[m][k]
        for k in ["gpsd", "gpower", "gcoh", "fo", "lt", "sr"]}
        for m in ut.METHODS
    }
    results["feats"] = feats
    results["f"] = features["f_axis"]
    results["midx"] = {str(m): i for i, m in enumerate(results["methods"])}
    results["dataset"] = str(results["dataset"])
    results["n_states"] = int(results["n_states"])
    return results


# ---------------
# Reusable Panels
# ---------------

def _fig_to_image(fig):
    """Rasterizes a standalone Matplotlib figure to an RGBA array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=500)
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def _power_brain(values):
    """Renders a power map to an image array for a single state."""
    fig, _ = osld_plotting.plot_brain_surface(
        values, mask_file=MASK_FILE, parcellation_file=PARCELLATION_FILE,
        symmetric_cbar=True, cbar_fontsize=24,
    )
    return _fig_to_image(fig)


def _coh_brain(conn):
    """Renders a coherence FC network to an image array for a single state."""
    fig, ax = plt.subplots(figsize=(5.7, 2.8))
    connectivity.save(
        connectivity_map=conn, parcellation_file=PARCELLATION_FILE,
        threshold=0.97, filename=None, axes=[ax],
        plot_kwargs={"edge_cmap": "RdBu_r", "figure": fig, "display_mode": "yz"},
    )
    cb_ax = fig.get_axes()[-1]
    pos = cb_ax.get_position()
    cb_ax.set_position([pos.x0 * 1.05, pos.y0, pos.width, pos.height])
    cb_ax.tick_params(labelsize=14)
    return _fig_to_image(fig)


def draw_matrix(
    ax,
    M,
    midx,
    title,
    cmap="viridis",
    fmt="{:.3f}",
    cbar_label=None,
    sci=False,
    cbar_tick_spacing=None,
):
    """
    A 4x4 method-by-method matrix, with colors scaled to [min, 1].

    - ``sci`` renders the colorbar ticks in scientific notation (factoring out the
      common offset), which keeps the near-1 similarity ticks compact.
    - ``cbar_tick_spacing`` (e.g. 0.05) pins the colorbar ticks to that spacing via a
      MultipleLocator, giving fewer, evenly-spaced ticks; None keeps the auto locator.
    """
    di = [midx[m] for m in ut.DISPLAY]
    Mv = M[np.ix_(di, di)]
    vmin = float(Mv.min())
    im = ax.imshow(Mv, cmap=cmap, vmin=vmin, vmax=1, aspect="auto")
    labels = [ut.LABELS[m] for m in ut.DISPLAY]
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_yticks(range(4)); ax.set_yticklabels(labels)
    mid = 0.5 * (vmin + 1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, fmt.format(Mv[i, j]), ha="center", va="center", fontsize=6.5,
                    color="white" if Mv[i, j] < mid else "black")
    ax.set_title(title)
    cax_ax = ax.inset_axes([0.0, -0.22, 1.0, 0.045])
    cbar = ax.figure.colorbar(im, cax=cax_ax, orientation="horizontal")
    cax = cbar.ax.xaxis
    if sci:
        sf = mticker.ScalarFormatter(useMathText=True, useOffset=True)
        sf.set_powerlimits((0, 0))
        cax.set_major_formatter(sf)
        cax.get_offset_text().set_fontsize(6)
        cbar.ax.tick_params(labelsize=6)
        cbar.ax.locator_params(nbins=4)
    if cbar_tick_spacing:
        cax.set_major_locator(mticker.MultipleLocator(cbar_tick_spacing))
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=7)


def draw_box(ax, per_sess, midx, ylabel, title, rng):
    """
    Per-session metric values against SDP for each method
    (draws box plots with jitter).
    """
    baselines = [m for m in ut.DISPLAY if m != "sdp"]
    n = per_sess.shape[0]  # number of sessions
    for i, m in enumerate(baselines):
        x = per_sess[:, midx[m]]
        bp = ax.boxplot(x, positions=[i], widths=0.6, patch_artist=True,
                        showfliers=False, zorder=2, medianprops=dict(color="black", lw=1.2))
        bp["boxes"][0].set(facecolor=ut.COLORS[m], alpha=0.45, edgecolor="0.3", lw=0.8)
        ax.scatter(np.full(n, i) + rng.uniform(-0.14, 0.14, n), x,
                   s=4, color=ut.COLORS[m], alpha=0.45, zorder=3, edgecolors="none")
    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels([ut.LABELS[m] for m in baselines])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def draw_grouped_bars(ax, feats, stat, ylabel):
    """
    Per-state summary statistics.
    Plots one bar per method within each state (mean ± sem).
    """
    K = np.asarray(feats["sdp"][stat]).shape[1]  # number of states
    x = np.arange(K)
    width = 0.8 / len(ut.DISPLAY)
    for i, m in enumerate(ut.DISPLAY):
        v = np.asarray(feats[m][stat])
        mean = v.mean(0)
        sem = v.std(0) / np.sqrt(v.shape[0])
        ax.bar(x + (i - (len(ut.DISPLAY) - 1) / 2) * width, mean, width, yerr=sem,
               color=ut.COLORS[m], alpha=0.85, label=ut.LABELS[m],
               error_kw=dict(lw=0.6, ecolor="0.3"))
    ax.set_xticks(x)
    ax.set_xticklabels([str(k + 1) for k in range(K)])
    ax.set_xlabel("State"); ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


# ------------------------------------------
# Figures (includes Fig. 5, S6, S7, Table 1)
# ------------------------------------------

def fig5_dynamics(R, rng):
    """
    Figure 5: Preservation of dynamics across sign-flipping methods.

    Uses all sessions (including template), and states are matched to SDP.
    """
    midx, feats = R["midx"], R["feats"]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 7.2))
    draw_matrix(axes[0, 0], R["dice"], midx, "Dice Overlap between State Time Courses",
                cbar_label="Dice coefficient")
    draw_box(axes[0, 1], R["dice_sess"], midx, "Dice coefficient",
             "Per-session Dice with SDP", rng)
    draw_matrix(axes[0, 2], R["sim_tde_cov"], midx, "Similarity in State-wise TDE Covariance",
                cbar_tick_spacing=0.05, cbar_label="Pearson correlation")
    draw_grouped_bars(axes[1, 0], feats, "fo", "Fractional occupancy")
    draw_grouped_bars(axes[1, 1], feats, "lt", "Mean lifetime (s)")
    draw_grouped_bars(axes[1, 2], feats, "sr", "Switching rate (Hz)")
    for ax in axes[1, :]:
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
        ax.legend(frameon=False, ncol=4, fontsize=6, loc="upper center",
                  columnspacing=1.0, handletextpad=0.4)
    for ax in axes.ravel():
        ax.set_box_aspect(1)  # square drawing box on every panel
    for ax, tag in zip(axes.ravel(), "abcdef"):
        ut.panel_tag(ax, tag, dx=-0.05, dy=1.07)
    fig.tight_layout(rect=[0, 0, 1, 0.96], w_pad=0.3, h_pad=2.2)
    ut.save_fig(fig, "fig5_dynamics")


def figS6_hmm_networks(R):
    """
    Figure S6: HMM network profiles for SDP sign-flipped data.

    Includes state-specific PSDs (mean ± sd), power maps, and FC coherence networks.
    The glass brains are rendered here, so the panels and colorbars can be sized
    to the grid.
    """
    # Get dynamic brain network features
    feats, f, ds = R["feats"], R["f"], R["dataset"]
    K = R["n_states"]
    gpsd = feats["sdp"]["gpsd"]      # (n_states, n_channels, n_freqs)
    gpower = feats["sdp"]["gpower"]  # (n_states, n_channels)
    gcoh = feats["sdp"]["gcoh"]      # (n_states, n_channels, n_channels)
    power_rel = gpower - gpower.mean(0, keepdims=True)  # power relative to the state mean
    coh_rel = gcoh - gcoh.mean(0, keepdims=True)        # coherence relative to the state mean

    # Plot figures
    fig = plt.figure(figsize=(10.0, 2.1 * K))
    gs = fig.add_gridspec(K, 3, width_ratios=[1.0, 1.3, 1.3], hspace=0.35, wspace=0.03,
                          top=0.93, bottom=0.05)
    for k in range(K):
        # State-specific PSD (shades mean ± standard deviation over parcels)
        ax0 = fig.add_subplot(gs[k, 0])
        mean, sd = gpsd[k].mean(0), gpsd[k].std(0)
        ax0.fill_between(f, mean - sd, mean + sd, color=ut.COLORS["sdp"], alpha=0.2, lw=0)
        ax0.plot(f, mean, color=ut.COLORS["sdp"], lw=1.2)
        ax0.set_ylabel("PSD (a.u.)", fontsize=8)
        if k == K - 1:
            ax0.set_xlabel("Frequency (Hz)", fontsize=8)
        ax0.tick_params(labelsize=8)

        # State-specific power map
        ax1 = fig.add_subplot(gs[k, 1]); ax1.axis("off")
        ax1.imshow(_power_brain(power_rel[k]))

        # State-specific coherence network
        ax2 = fig.add_subplot(gs[k, 2]); ax2.axis("off")
        ax2.imshow(_coh_brain(coh_rel[k]))

        # Set column titles
        if k == 0:
            ax0.set_title("State-wise PSD", fontsize=9)
            ax1.set_title("State-wise Power Maps", fontsize=9)
            ax2.set_title("State-wise FC Networks", fontsize=9)

    fig.suptitle(
        f"Group-level HMM Networks for {ut.DATASET_LABELS[ds]} (with SDP sign-flipping)",
        fontsize=9, y=0.985,
    )
    ut.save_fig(fig, "figS6_hmm_networks")


def tableS1_dynamic_invariance(data):
    """
    Table S1: Inter-method similarity of dynamic state-network features (as correlation
    with SDP), written as a markdown file.

    Each cell is the Pearson correlation of that method's group-level feature with SDP's
    (states matched to SDP).
    """
    feats = [("sim_psd", "State PSD"), ("sim_power", "State power"),
             ("sim_coh", "State coherence"), ("sim_tp", "Trans. prob.")]
    baselines = [m for m in ut.DISPLAY if m != "sdp"]  # methods compared against SDP

    header = "| Dataset | Method | " + " | ".join(name for _, name in feats) + " |"
    sep = "|" + "---|" * (2 + len(feats))
    lines = [
        "# Table S1. Inter-method similarity of dynamic state-network features",
        "",
        "Pearson correlation of each method's group-level state-network features with SDP's "
        "(states matched to SDP). Higher is more similar (maximum 1).",
        "",
        header, sep,
    ]
    for ds in ut.ALL_DATASETS:
        R = data[ds]
        midx = R["midx"]
        for i, m in enumerate(baselines):
            ds_cell = ut.DATASET_LABELS[ds] if i == 0 else ""
            vals = " | ".join(f"{R[key][midx['sdp'], midx[m]]:.4f}" for key, _ in feats)
            lines.append(f"| {ds_cell} | {ut.LABELS[m]} | {vals} |")

    os.makedirs(ut.FIG_DIR, exist_ok=True)
    out = os.path.join(ut.FIG_DIR, "tableS1_dynamic_invariance.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[INFO] Saved {out}.")


def figS7_obs_cov(R):
    """
    Figure S7: Observation model state covariances per method (blanked diagonals).

    Each state (column) has its own symmetric color scale, shared across the four methods
    (rows), so low-amplitude states are not washed out by the high-amplitude ones.

    Note that this figure is for a qualitative comparison, since each method has its own
    PCA basis (when the data were prepared before TDE-HMM fitting).
    """
    # Get state-specific covariance matrices
    obs = R["obs_cov"]  # (n_methods, n_states, 80, 80); state-matched
    midx, K = R["midx"], R["n_states"]
    covs = {}
    vmax = np.zeros(K)
    for k in range(K):
        pool = []
        for m in ut.DISPLAY:
            c = obs[midx[m]][k].astype(float).copy()
            np.fill_diagonal(c, np.nan)
            covs[(m, k)] = c
            pool.append(np.abs(c).ravel())
        vmax[k] = np.nanpercentile(np.concatenate(pool), 99)

    # Plot figures
    fig, axes = plt.subplots(
        len(ut.DISPLAY), K,
        figsize=(2.0 * K, 2.2 * len(ut.DISPLAY)),
        constrained_layout=True,
    )
    ims = [None] * K
    for r, m in enumerate(ut.DISPLAY):
        for k in range(K):
            ax = axes[r, k]
            ims[k] = ax.imshow(covs[(m, k)], cmap="RdBu_r", vmin=-vmax[k], vmax=vmax[k])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"State {k + 1}", fontsize=9)
            if k == 0:
                ax.set_ylabel(ut.LABELS[m], fontsize=9)
    for k in range(K):
        cbar = fig.colorbar(ims[k], ax=axes[:, k], location="bottom",
                            fraction=0.05, pad=0.02, aspect=12)
        cbar.locator = mticker.MaxNLocator(nbins=2, symmetric=True)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.xaxis.get_offset_text().set_fontsize(8)
    ut.save_fig(fig, "figS7_obs_cov")


if __name__ == "__main__":
    # Set visual style
    ut.set_style()  # set plot style
    rng = np.random.default_rng(0)  # for jitters; deterministic

    # Load analysis results
    data = {ds: load_dynamic_features(ds) for ds in ut.ALL_DATASETS}
    primary_data = data[ut.PRIMARY]  # CamCAN-only

    # Generate figures
    fig5_dynamics(primary_data, rng)
    figS6_hmm_networks(primary_data)
    tableS1_dynamic_invariance(data)
    figS7_obs_cov(primary_data)
    print("[INFO] Visualization complete.")
