"""Visualize the overview schematic of the sign-flipping problem and the proposed approach.

Writes ``fig1_overview`` into the top-level ``figures/``:

    (a) The problem: each session is source reconstructed independently, so every
        parcel time course carries an arbitrary sign (simulated signals).
    (b) Why it matters: (top) simulated sessions with random signs cancel in the group
        average and give correlations of either sign; (bottom) one cross-parcel block
        of the Cam-CAN TDE covariance, per session and averaged over sessions, before
        and after SDP sign-flipping.
    (c) The approach on a simulated session with known signs: time-delay embedding ->
        session and template TDE covariances -> affinity matrix -> Z2 synchronization ->
        spectral, SDP, and BP -> greedy refinement of the correlation objective ->
        recovered signs of every method next to the true signs.

Only the bottom row of (b) uses Cam-CAN data (no time series are shown); (a), the top
row of (b), and (c) are simulated. The example parcel pair in (b) is selected by a
sign-invariant rule: the pair with the largest per-session cross-parcel block magnitude
(RMS), averaged over sessions. The Cam-CAN quantities are computed once and cached in
``camcan/data/schematic.npz``; delete the cache to recompute.

Env: ``osld`` (needs ``osl_signflip`` and ``nilearn``).
Usage:
    python schematics.py        # computes the cache (if absent) and produces the figure
"""

# Import packages
import os
import numpy as np
import utils as ut
import matplotlib.pyplot as plt  # utils sets the Agg backend on import
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import blended_transform_factory

DATASET = ut.PRIMARY
N_EMBEDDINGS = 15
FS = 250
N_JOBS = 16
CACHE = os.path.join(ut.data_dir(DATASET), "schematic.npz")
PARCELLATION_FILE = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"  # 38 Giles

# Colors of the schematic figure
POS, NEG = "#c0392b", "#2c6fbb"  # positive / negative sign
GREY = "#8a8a8a"
PANEL_BG = "#f4f5f7"
CMAP = "RdBu_r"


# -----------
# Computation
# -----------

def parcel_file(session):
    """Pre-flip parcel time courses (identical for every method)."""
    return os.path.join(ut.BASE_DIR, DATASET, "src_sdp", session, "parc", "lcmv-parc-raw.fif")


def bp_beliefs(M, beta=2.0, damping=0.5, prior_strength=1e-3, n_iter=200, tol=1e-6):
    """Final BP beliefs, mirroring ``osl_signflip.solvers.belief_propagation`` (defaults)."""
    eps = 1e-12
    W = M / np.max(np.abs(M))
    row_strength = W.sum(1)
    h = prior_strength * row_strength / np.max(np.abs(row_strength))
    u = np.zeros_like(W)
    tanh_betaW = np.tanh(beta * W)
    for _ in range(n_iter):
        cavity = (h + u.sum(0))[:, None] - u.T
        new_u = np.arctanh(np.clip(tanh_betaW * np.tanh(cavity), -1 + eps, 1 - eps))
        np.fill_diagonal(new_u, 0.0)
        u_next = damping * u + (1 - damping) * new_u
        delta = np.max(np.abs(u_next - u))
        u = u_next
        if delta < tol:
            break
    return h + u.sum(0)


def compute():
    """Computes the Cam-CAN quantities shown in panel (b) and caches them."""
    from osl_signflip.io import load_covariances
    from osl_dynamics.meeg.parcellation import Parcellation

    D, midx = ut.load_analysis_results(DATASET)
    sessions = [str(s) for s in D["sessions"]]
    template = str(D["template"])
    N, n, E = len(sessions), int(D["n_parcels"]), N_EMBEDDINGS

    # Session and template TDE covariances (pre-flip)
    covs = load_covariances([parcel_file(s) for s in sessions], E, n_jobs=N_JOBS)
    T = load_covariances([parcel_file(template)], E, use_tqdm=False)[0]

    # Example pair: largest sign-invariant block magnitude, averaged over sessions
    blocks = covs.reshape(N, n, E, n, E).transpose(0, 1, 3, 2, 4)  # (session, i, j, E, E)
    strength = np.sqrt((blocks ** 2).mean((3, 4))).mean(0)
    iu = np.triu_indices(n, 1)
    k = np.argmax(strength[iu])
    i, j = int(iu[0][k]), int(iu[1][k])

    F = D["flips"][midx["sdp"]].astype(float)  # (session, parcel)
    rel = F[:, i] * F[:, j]  # relative sign applied to the pair
    pair_before = blocks[:, i, j].copy()

    np.savez(
        CACHE,
        sessions=np.array(sessions), template=template, pair=np.array([i, j]),
        strength=strength[iu], pair_before=pair_before.astype(np.float32), rel=rel,
        coords=Parcellation(PARCELLATION_FILE).roi_centers(),
    )
    print(f"[INFO] Saved {CACHE}.")


def ar2(n_samples, freq, rng, radius=0.97, burn=500):
    """Unit-variance AR(2) oscillation at ``freq`` Hz."""
    a1, a2 = 2 * radius * np.cos(2 * np.pi * freq / FS), -radius ** 2
    e = rng.standard_normal(n_samples + burn)
    z = np.zeros_like(e)
    for t in range(2, len(e)):
        z[t] = a1 * z[t - 1] + a2 * z[t - 2] + e[t]
    z = z[burn:]
    return z / z.std()


def simulate_approach(n=12, duration=60, deviation=0.5, noise=1.0, seed=1, min_weight=0.7,
                      keep_structure=True, methods=None):
    """
    Toy sign-flipping problem with known signs for panel (c). Every parcel mixes a few
    shared oscillations with parcel-specific delays (so cross-parcel blocks have lagged
    structure). The template and the session are independent realizations of the same
    network; the session's non-zero mixing weights deviate and its parcels carry the
    true signs ``f_true``. Every method is run with the pipeline's hyperparameters.
    """
    from osl_signflip.api import find_flips
    from osl_signflip.covariance import compute_covariance
    from osl_signflip.metric import FlipMetricEvaluator
    from osl_signflip.solvers.sdp import _solve_sdp_mixing

    rng = np.random.default_rng(seed)
    E, T = N_EMBEDDINGS, duration * FS
    freqs = [8, 10, 12, 20]
    W = rng.standard_normal((n, len(freqs))) * (rng.random((n, len(freqs))) < 0.6)
    for p in np.flatnonzero(np.abs(W).max(1) < min_weight):  # no parcel is pure noise
        W[p, rng.integers(len(freqs))] = rng.choice([-1, 1]) * (min_weight + abs(rng.normal()))
    delay = rng.integers(0, 8, size=W.shape)

    def realise(weights):
        z = [ar2(T + 10, f, rng) for f in freqs]
        x = np.stack([sum(weights[p, k] * z[k][10 - delay[p, k]:10 - delay[p, k] + T]
                          for k in range(len(freqs))) for p in range(n)], 1)
        return x + noise * rng.standard_normal(x.shape)

    x_template = realise(W)
    # Session-specific deviations change the strength of existing couplings only, so the
    # session shares the template's network structure
    mask = (W != 0) if keep_structure else 1
    x_session = realise(W + deviation * rng.standard_normal(W.shape) * mask)
    f_true = np.ones(n)
    f_true[rng.choice(n, n // 2 - 1, replace=False)] = -1
    x_session *= f_true

    C, Tm = compute_covariance(x_session, E), compute_covariance(x_template, E)
    M = FlipMetricEvaluator(C, Tm, E).M
    kwargs = {"random": dict(n_restarts=3, n_iter=5000, max_flips=min(20, n), use_tqdm=False)}
    flips, rho_after = [], []
    for m in methods or ut.DISPLAY:
        f, metrics = find_flips(C, Tm, E, method=m, **kwargs.get(m, {}))
        flips.append(f * np.sign(f @ f_true))  # common global sign: that of the truth
        rho_after.append(metrics[-1])
        rho_before = metrics[0]

    # Intermediate quantities of the three proposed methods, signed like the truth
    v1 = np.linalg.eigh(M)[1][:, -1]
    v1 *= np.sign(np.sign(v1) @ f_true)
    # One SDP restart at its random start (no sweeps) and at convergence
    _, V0 = _solve_sdp_mixing(M, n_restarts=1, max_iter=0, rng=np.random.default_rng(0))
    _, V = _solve_sdp_mixing(M, n_restarts=1, rng=np.random.default_rng(0))
    beliefs = bp_beliefs(M)
    beliefs *= np.sign(np.sign(beliefs) @ f_true)
    return dict(x=x_session[:, 0], C=C, T=Tm, M=M, v1=v1, V0=V0, V=V, beliefs=beliefs,
                f_true=f_true, flips=np.stack(flips), rho_before=rho_before,
                rho_after=np.array(rho_after))


# --------
# Plotting
# --------

def simulate(n_sessions=40, seed=0):
    """
    Toy example: every session shares an evoked response in parcel A and a delayed,
    attenuated copy in parcel B, but each parcel carries a random sign per session.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(-0.1, 0.5, 301)
    w = np.exp(-((t - 0.10) / 0.03) ** 2) - 0.6 * np.exp(-((t - 0.20) / 0.05) ** 2)
    wb = 0.8 * np.interp(t - 0.03, t, w)
    signs = rng.choice([-1.0, 1.0], size=(n_sessions, 2))
    a = signs[:, :1] * w + 0.15 * rng.standard_normal((n_sessions, t.size))
    b = signs[:, 1:] * wb + 0.15 * rng.standard_normal((n_sessions, t.size))
    return t, a, b, signs


def corr_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def boxed(ax, lw=0.6, color="k"):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
        sp.set_color(color)


def sign_badge(ax, x, y, s, size=7):
    ax.text(x, y, "+" if s > 0 else "-", transform=ax.transAxes, ha="center",
            va="center", color="white", fontsize=size, fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.15", fc=POS if s > 0 else NEG, ec="none"))


def arrow(fig, x0, x1, y, color=GREY):
    """Horizontal arrow between two x positions (figure coordinates)."""
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=8, lw=1.0, color=color,
                                   shrinkA=0, shrinkB=0))


def x_extent(fig, axes, artists=()):
    """Horizontal figure-coordinate extent of axes frames plus side artists (e.g. labels)."""
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [a.get_position() for a in axes]
    boxes += [a.get_window_extent(renderer).transformed(inv) for a in artists]
    return min(b.x0 for b in boxes), max(b.x1 for b in boxes)


def panel_extent(fig, axes, pad=0.012, top=0.03):
    """Figure-coordinate extent of a group of axes, including their labels and texts."""
    renderer = fig.canvas.get_renderer()
    boxes = [a.get_tightbbox(renderer).transformed(fig.transFigure.inverted()) for a in axes]
    return [min(b.x0 for b in boxes) - pad, min(b.y0 for b in boxes) - pad,
            max(b.x1 for b in boxes) + pad, max(b.y1 for b in boxes) + top]


def panel_background(fig, x0, y0, x1, y1):
    fig.add_artist(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
                                  boxstyle="round,pad=0,rounding_size=0.01",
                                  fc=PANEL_BG, ec="none", zorder=-10))


def panel_label(fig, x, y, letter, title):
    # Shared baseline so the letter and the title line up despite different font sizes
    fig.text(x + 0.008, y - 0.03, letter, fontsize=11, fontweight="bold", va="baseline")
    fig.text(x + 0.028, y - 0.03, title, fontsize=8, fontweight="bold", va="baseline")


def draw_brain(ax, coords, pair):
    """Glass brain with all parcels in grey and the example pair drawn on top."""
    from nilearn import plotting
    others = np.setdiff1d(np.arange(len(coords)), pair)
    display = plotting.plot_connectome(np.zeros((len(others),) * 2), coords[others],
                                       node_color="#d4d4d4", node_size=3, display_mode="z",
                                       axes=ax, annotate=False, colorbar=False)
    # The axial view plots (x, y) directly; nilearn draws nodes at zorder 1000, so the
    # pair goes above them (another parcel sits 2.6 mm from parcel A in this view)
    brain = display.axes["z"].ax
    for p, c in zip(pair, ["#7b3294", "#008837"]):
        brain.scatter(*coords[p][:2], s=30, color=c, ec="white", lw=0.5, zorder=2000)
    return brain


def draw_problem(fig, gs, S, sim):
    """Panel (a): arbitrary sign per parcel and session."""
    t, a, b, signs = sim
    sub = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs, height_ratios=[1.5, 1, 1, 1],
                                  hspace=0.35)
    ax_brain = fig.add_subplot(sub[0])
    clean(ax_brain)  # placeholder; the brain is drawn centered in the panel once it is laid out
    axes = [ax_brain]
    for r, s in enumerate([0, 1, 2]):
        ax = fig.add_subplot(sub[r + 1])
        ax.plot(t, a[s], color="#7b3294", lw=0.9)
        ax.plot(t, b[s] - 2.2, color="#008837", lw=0.9)
        ax.set_xlim(t[0], t[-1] + 0.12)
        ax.set_ylim(-3.6, 1.6)
        clean(ax)
        ax.text(-0.02, 0.5, f"Session {r + 1}", transform=ax.transAxes, ha="right",
                va="center", fontsize=6.5)
        sign_badge(ax, 0.95, 0.80, signs[s, 0], size=6)
        sign_badge(ax, 0.95, 0.30, signs[s, 1], size=6)
        axes.append(ax)
    # x is re-centered on the panel once its extent is known (fig1_overview)
    caption = axes[-1].text(0.5, -0.15, "Each session is source reconstructed independently:\n"
                            "parcel signs (±) are arbitrary (simulated signals)",
                            transform=axes[-1].transAxes, ha="center", va="top", fontsize=6,
                            color="#444444")
    axes[1].text(0.02, 1.05, "Parcel A", color="#7b3294", transform=axes[1].transAxes,
                 fontsize=6, va="bottom")
    axes[1].text(0.32, 1.05, "Parcel B", color="#008837", transform=axes[1].transAxes,
                 fontsize=6, va="bottom")
    return axes, caption


def draw_consequence(fig, gs, S, sim):
    """Panel (b): cancellation in the group average and in a real TDE block."""
    t, a, b, signs = sim
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs, height_ratios=[1, 1.15], hspace=0.45)
    top = GridSpecFromSubplotSpec(1, 2, subplot_spec=sub[0], wspace=0.45,
                                  width_ratios=[1.4, 1])
    axes = []

    # (top-left) group-averaged response of parcel A, before vs after sign alignment
    ax = fig.add_subplot(top[0])
    ax.axhline(0, color="#bbbbbb", lw=0.5)
    ax.plot(t, a.mean(0), color=GREY, lw=1.1, label="Arbitrary signs")
    ax.plot(t, (a * signs[:, :1]).mean(0), color=ut.COLORS["sdp"], lw=1.1,
            label="Aligned signs")
    ax.set_xlabel("Time (s)", labelpad=1)
    ax.set_ylabel("Group average (a.u.)", labelpad=1)
    ax.set_yticks([0])
    ax.set_title("Average over sessions", fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=5.5, loc="upper right", handlelength=1.2)
    axes.append(ax)

    # (top-right) per-session correlation between parcels A and B
    ax = fig.add_subplot(top[1])
    r_before = corr_rows(a, b)
    r_after = corr_rows(a * signs[:, :1], b * signs[:, 1:])
    rng = np.random.default_rng(1)
    for x, r, c in [(0, r_before, GREY), (1, r_after, ut.COLORS["sdp"])]:
        ax.scatter(x + rng.uniform(-0.15, 0.15, r.size), r, s=4, color=c, lw=0, alpha=0.8,
                   label="Sessions" if x == 0 else None)
        ax.plot([x - 0.25, x + 0.25], [r.mean()] * 2, color="k", lw=1.0,
                label="Group mean" if x == 0 else None)
    ax.axhline(0, color="#bbbbbb", lw=0.5)
    ax.set_xticks([0, 1], ["Arbitrary", "Aligned"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Corr(A, B)", labelpad=1)
    ax.set_title("Between-parcel correlations", fontsize=7, pad=3)
    ax.legend(frameon=False, fontsize=5.5, loc="lower right", handlelength=1.2,
              markerscale=1.5)
    axes.append(ax)

    # (bottom) real TDE cross-parcel block, per session and session-averaged
    blocks, rel = S["pair_before"], S["rel"]
    mag = np.sqrt((blocks ** 2).mean((1, 2)))
    pos, neg = np.flatnonzero(rel > 0), np.flatnonzero(rel < 0)
    order = list(pos[np.argsort(mag[pos])[::-1][:2]]) + list(neg[np.argsort(mag[neg])[::-1][:2]])
    bot = GridSpecFromSubplotSpec(1, 7, subplot_spec=sub[1], wspace=0.25,
                                  width_ratios=[1, 1, 1, 1, 0.55, 1.35, 1.35])
    v = np.percentile(np.abs(blocks[order]), 98)
    for c, s in enumerate(order):
        ax = fig.add_subplot(bot[c])
        ax.imshow(blocks[s], cmap=CMAP, vmin=-v, vmax=v)
        boxed(ax, lw=0.4, color="#666666")
        ax.set_title(f"Session {c + 1}", fontsize=5.5, pad=2)
        sign_badge(ax, 0.5, -0.16, rel[s], size=5.5)
        axes.append(ax)
    # Centered under the four session blocks (the midpoint of their centers is unaffected
    # by the equal-aspect shrinking of each axes)
    p0, p1 = axes[2].get_position(), axes[5].get_position()
    axes[2].text((p0.x0 + p0.x1 + p1.x0 + p1.x1) / 4, -0.36,
                 "Cam-CAN TDE covariance block of\n" \
                 "parcels A and B before flipping",
                 transform=blended_transform_factory(fig.transFigure, axes[2].transAxes),
                 ha="center", va="top", fontsize=5.5, color="#444444")
    vg = np.abs(blocks.mean(0)).max() * 1.15
    vg = max(vg, np.abs((blocks * rel[:, None, None]).mean(0)).max())
    for c, (mat, title) in enumerate([
            (blocks.mean(0), "Average before"),
            ((blocks * rel[:, None, None]).mean(0), "Average after SDP")]):
        ax = fig.add_subplot(bot[5 + c])
        ax.imshow(mat, cmap=CMAP, vmin=-vg, vmax=vg)
        boxed(ax, lw=0.6, color=ut.COLORS["sdp"] if c else "#666666")
        ax.set_title(title, fontsize=5.5, pad=2)
        axes.append(ax)

    # Row labels: simulated (top) vs real data (bottom), at a common x left of both rows
    x = axes[0].get_position().x0 - 0.066
    for ax, label in [(axes[0], "Simulated"), (axes[2], "Real data (Cam-CAN)")]:
        ax.text(x, 0.5, label, transform=blended_transform_factory(fig.transFigure,
                ax.transAxes), rotation=90, ha="center", va="center", fontsize=6.5,
                fontweight="bold", color="#444444")
    return axes


def draw_approach(fig, gs, S):
    """Panel (c): pipeline on a simulated session with known signs."""
    E, n = N_EMBEDDINGS, len(S["v1"])
    sub = GridSpecFromSubplotSpec(1, 6, subplot_spec=gs, wspace=0.42,
                                  width_ratios=[1.0, 1.25, 1.0, 1.1, 1.05, 1.0])
    axes = []

    # (c1) time-delay embedding of one parcel time course
    ax = fig.add_subplot(sub[0])
    x = S["x"][:FS] / np.std(S["x"])
    lags = [-7, -4, 0, 4, 7]
    lag_texts = []
    for r, lag in enumerate(lags):
        xs = np.roll(x, lag)[10:-10]
        ax.plot(xs - 3.2 * r, color="#7b3294", lw=0.6, alpha=1 if lag == 0 else 0.55)
        lag_texts.append(ax.text(len(xs) + 6, -3.2 * r, f"{lag:+d}" if lag else "0",
                                 fontsize=5, va="center", color="#555555"))
    ax.set_xlim(0, len(xs))
    clean(ax)
    ax.set_title("Time-delay embedding", fontsize=6.5, pad=3)
    ax.text(0.5, -0.06, f"{E} lagged copies per parcel", transform=ax.transAxes,
            ha="center", va="top", fontsize=5.5, color="#444444")
    axes.append(ax)

    # (c2) session vs template TDE covariance (all parcels). The most strongly coupled
    # pair in the template (largest block RMS, a sign-invariant rule) is outlined.
    sub2 = GridSpecFromSubplotSpec(2, 1, subplot_spec=sub[1], hspace=0.35)
    D = n * E
    cross = np.arange(D)[:, None] // E != np.arange(D)[None, :] // E
    v = np.percentile(np.abs(S["T"][cross]), 99)
    rms = np.sqrt((S["T"].reshape(n, E, n, E) ** 2).mean((1, 3)))
    iu = np.triu_indices(n, 1)
    k = np.argmax(rms[iu])
    pi, pj = iu[0][k], iu[1][k]
    pair_axes = []
    for r, (mat, title) in enumerate([(S["C"], "Session  $\\mathbf{C}_s$"),
                                      (S["T"], "Template  $\\mathbf{T}$")]):
        ax = fig.add_subplot(sub2[r])
        ax.imshow(mat, cmap=CMAP, vmin=-v, vmax=v)
        for g in range(1, n):
            ax.axhline(g * E - 0.5, color="white", lw=0.2)
            ax.axvline(g * E - 0.5, color="white", lw=0.2)
        ax.add_patch(Rectangle((pj * E - 0.5, pi * E - 0.5), E, E, fill=False, ec="k",
                               lw=0.6))
        boxed(ax, lw=0.4, color="#666666")
        ax.set_ylabel(title, fontsize=5.5, labelpad=1)
        pair_axes.append(ax)
    pair_axes[0].set_title("TDE covariances", fontsize=6.5, pad=3)
    axes.append(pair_axes[0])

    # (c3) affinity matrix
    ax = fig.add_subplot(sub[2])
    vm = np.abs(S["M"]).max()
    ax.imshow(S["M"], cmap=CMAP, vmin=-vm, vmax=vm)
    boxed(ax, lw=0.4, color="#666666")
    ax.set_title("Affinity  $\\mathbf{M}$", fontsize=6.5, pad=3)
    ax.text(0.5, -0.06, "$M_{ij} = \\sum_{(a,b)\\in\\mathcal{S}_{ij}} C_{ab}T_{ab}$",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.5)
    axes.append(ax)

    # (c4) Z2 synchronisation graph: numbered parcels, coloured by their recovered sign
    ax = fig.add_subplot(sub[3])
    ang = np.pi / 2 - np.linspace(0, 2 * np.pi, n, endpoint=False)  # clockwise from top
    xy = np.c_[np.cos(ang), np.sin(ang)]
    M = S["M"]
    iu = np.triu_indices(n, 1)
    top = np.argsort(np.abs(M[iu]))[::-1][:30]
    wmax = np.abs(M[iu][top]).max()
    for t in top:
        i, j = iu[0][t], iu[1][t]
        ax.plot(*xy[[i, j]].T, color=POS if M[i, j] > 0 else NEG,
                lw=0.2 + 1.4 * abs(M[i, j]) / wmax, alpha=0.7, zorder=1)
    f = S["flips"][0]
    ax.scatter(*xy.T, s=42, c=[POS if v > 0 else NEG for v in f], ec="white", lw=0.4,
               zorder=2)
    for p, (u, w) in enumerate(xy):
        ax.text(u, w, str(p + 1), ha="center", va="center", fontsize=4, color="white",
                fontweight="bold", zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-1.12, 1.12)  # tight around the nodes, so arrow gaps match the other stages
    ax.set_ylim(-1.12, 1.12)
    clean(ax)
    ax.set_title("$\\mathbb{Z}_2$ synchronisation", fontsize=6.5, pad=3)
    ax.text(0.5, -0.06, "$\\max_{\\mathbf{f}\\in\\{\\pm1\\}^n}\\ \\mathbf{f}^{\\top}"
            "\\mathbf{M}\\mathbf{f}$", transform=ax.transAxes, ha="center", va="top",
            fontsize=6)
    ax.text(0.5, -0.3, "edges: sign of $M_{ij}$; nodes: $f_i$", transform=ax.transAxes,
            ha="center", va="top", fontsize=5, color="#555555")
    axes.append(ax)

    # (c5) three proposed methods (bars in parcel order, matching the node numbers)
    sub5 = GridSpecFromSubplotSpec(3, 1, subplot_spec=sub[4], hspace=0.55)
    m_axes = []
    ax = fig.add_subplot(sub5[0])  # spectral: leading eigenvector
    vals = S["v1"]
    ax.bar(np.arange(n), vals, color=[POS if v > 0 else NEG for v in vals], width=0.75)
    ax.axhline(0, color="k", lw=0.4)
    m_axes.append((ax, "spectral", "Spectral: eigenvector"))
    ax = fig.add_subplot(sub5[1])  # SDP: unit vectors at the start and at convergence
    # Project onto the plane spanned by the direction separating the SDP signs at
    # convergence and the leading remaining direction of the converged vectors, so the
    # dashed line acts as the rounding hyperplane. The starting vectors are drawn faint
    # (their 2D projections are shorter than unit length); the converged ones bold.
    V0, V, f_sdp = S["V0"], S["V"], S["flips"][0]
    r = V.T @ f_sdp
    r /= np.linalg.norm(r)
    resid = V - np.outer(V @ r, r)
    q = np.linalg.svd(resid, full_matrices=False)[2][0]
    for Vk, lw, alpha in [(V0, 0.6, 0.45), (V, 1.0, 1.0)]:
        for p, s in zip(np.c_[Vk @ r, Vk @ q], f_sdp):
            ax.plot([0, p[0]], [0, p[1]], color=POS if s > 0 else NEG, lw=lw, alpha=alpha)
    ax.axvline(0, color="k", lw=0.6, ls="--")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    m_axes.append((ax, "sdp", "SDP: vectors + hyperplane"))
    ax = fig.add_subplot(sub5[2])  # BP: beliefs
    b = S["beliefs"]
    ax.bar(np.arange(n), np.tanh(b), color=[POS if v > 0 else NEG for v in b], width=0.75)
    ax.axhline(0, color="k", lw=0.4)
    m_axes.append((ax, "belief_propagation", "BP: beliefs"))
    for ax, m, title in m_axes:
        boxed(ax, lw=1.0, color=ut.COLORS[m])
        ax.set_title(title, fontsize=5.5, pad=2, color=ut.COLORS[m])
    axes.append(m_axes[0][0])

    # (c6) greedy refinement -> recovered signs of every method next to the true signs
    ax = fig.add_subplot(sub[5])
    cols = [("True", "k", S["f_true"])] + [
        (ut.LABELS[m], ut.COLORS[m], S["flips"][c]) for c, m in enumerate(ut.DISPLAY)]
    grid = np.stack([f for _, _, f in cols], 1)  # (parcel, column)
    ax.imshow(grid, cmap=CMAP, vmin=-1.6, vmax=1.6, aspect="auto", interpolation="nearest")
    for (p, c), val in np.ndenumerate(grid):
        ax.text(c, p, "+" if val > 0 else "\u2212", ha="center", va="center", fontsize=5,
                color="white", fontweight="bold")
    for g in range(1, len(cols)):
        ax.axvline(g - 0.5, color=PANEL_BG, lw=2.0 if g == 1 else 0.8)
    for g in range(1, n):
        ax.axhline(g - 0.5, color=PANEL_BG, lw=0.5)
    ax.set_xticks(range(len(cols)), [c for c, _, _ in cols], fontsize=5.5, rotation=90)
    ax.xaxis.tick_top()
    for tl, (_, color, _) in zip(ax.get_xticklabels(), cols):
        tl.set_color(color)
    ax.set_yticks(range(n), [str(p + 1) for p in range(n)], fontsize=4.5)
    ax.set_ylabel("Parcel", fontsize=5.5, labelpad=1)
    ax.tick_params(length=0, pad=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Refine $\\rho$ → signs $\\mathbf{f}^{\\bigstar}$", fontsize=6.5, pad=3)
    ax.text(0.5, -0.03, "+: keep   \u2212: flip\n"
            f"$\\rho$: {S['rho_before']:.2f} → {S['rho_after'][0]:.2f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.5)
    axes.append(ax)

    # Pipeline stages, left to right, as (axes, side artists) for spacing the arrows
    stages = [([axes[0]], lag_texts),
              (pair_axes, [a.yaxis.label for a in pair_axes]),
              ([axes[2]], []),
              ([axes[3]], []),
              ([m for m, _, _ in m_axes], []),
              ([ax], [ax.yaxis.label])]
    return axes, pair_axes, m_axes, stages


def fig1_overview(S):
    sim = simulate()
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82], width_ratios=[0.62, 1.38],
                          hspace=0.55, wspace=0.27, left=0.08, right=0.98, top=0.95,
                          bottom=0.06)
    a_axes, a_caption = draw_problem(fig, gs[0, 0], S, sim)
    b_axes = draw_consequence(fig, gs[0, 1], S, sim)
    sim_c = simulate_approach()
    print(f"[INFO] Simulated session: rho {sim_c['rho_before']:.3f} -> "
          + ", ".join(f"{m} {r:.3f}" for m, r in zip(ut.DISPLAY, sim_c["rho_after"]))
          + "; recovered = true signs: "
          + ", ".join(f"{m} {np.all(f == sim_c['f_true'])}"
                      for m, f in zip(ut.DISPLAY, sim_c["flips"])))
    c_axes, pair_axes, m_axes, stages = draw_approach(fig, gs[1, :], sim_c)

    # Panel (c): space the pipeline stages so every arrow slot has the same width
    fig.canvas.draw()
    xs = [x_extent(fig, axs, arts) for axs, arts in stages]
    slot = (xs[-1][1] - xs[0][0] - sum(x1 - x0 for x0, x1 in xs)) / (len(xs) - 1)
    x = xs[0][0]
    for (axs, _), (x0, x1) in zip(stages, xs):
        for a in axs:
            pos = a.get_position()
            a.set_position([pos.x0 + x - x0, pos.y0, pos.width, pos.height])
        x += x1 - x0 + slot

    # Panel backgrounds and labels (positions are fixed once the layout is drawn)
    fig.canvas.draw()
    ext = {"a": panel_extent(fig, a_axes, top=0.045),
           "b": panel_extent(fig, b_axes, top=0.045),
           "c": panel_extent(fig, c_axes + pair_axes + [m for m, _, _ in m_axes], top=0.045)}
    for k, f in ((1, min), (3, max)):  # panels (a) and (b) share the top row's extent
        ext["a"][k] = ext["b"][k] = f(ext["a"][k], ext["b"][k])
    ext["a"][2] = min(ext["a"][2], ext["b"][0] - 0.012)  # gap between (a) and (b)
    ext["c"][0], ext["c"][2] = ext["a"][0], ext["b"][2]  # (c) spans the top row's width
    titles = {"a": "Sign ambiguity", "b": "Why it matters",
              "c": "Sign-flipping as $\\mathbb{Z}_2$ synchronisation"}
    for letter, (x0, y0, x1, y1) in ext.items():
        panel_background(fig, x0, y0, x1, y1)
        panel_label(fig, x0, y1, letter, titles[letter])

    # Brain icon, centered horizontally in panel (a)
    pos = a_axes[0].get_position()
    ax = fig.add_axes([(ext["a"][0] + ext["a"][2] - pos.width) / 2, pos.y0, pos.width,
                       pos.height])
    clean(ax)
    draw_brain(ax, S["coords"], S["pair"])
    a_caption.set_x(((ext["a"][0] + ext["a"][2]) / 2 - pos.x0) / pos.width)  # same x span

    # Pipeline arrows: one common height (center of the affinity matrix) and the same gap
    # between every arrow and the stages it connects
    pos = c_axes[2].get_position()
    y, gap = pos.y0 + pos.height / 2, 0.012
    fig.canvas.draw()
    xs = [x_extent(fig, axs, arts) for axs, arts in stages]
    for (_, x1), (x0, _) in zip(xs[:-1], xs[1:]):
        arrow(fig, x1 + gap, x0 - gap, y)
    ut.save_fig(fig, "fig1_overview")


def main():
    if not os.path.exists(CACHE):
        compute()
    S = {k: v for k, v in np.load(CACHE, allow_pickle=True).items()}
    i, j = S["pair"]
    print(f"[INFO] Example pair ({i}, {j}); relative sign inverted by SDP in "
          f"{int((S['rel'] < 0).sum())}/{len(S['rel'])} sessions")
    ut.set_style()
    with plt.rc_context({"axes.facecolor": "none"}):  # let the panel backgrounds show
        fig1_overview(S)


if __name__ == "__main__":
    main()
