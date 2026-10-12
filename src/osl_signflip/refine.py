"""Greedy refinement of a flip vector on the exact metric."""

# Authors: SungJun Cho <sungjun.cho@ndcn.ox.ac.uk>

import numpy as np

from ._utils import apply_flip_convention
from .metric import FlipMetricEvaluator


def greedy_refine(
    cov,
    template_cov,
    flips,
    n_embeddings,
    n_refine,
    strategy="best",
    tol=1e-12,
):
    """Greedy coordinate-ascent refinement of a flip vector.

    Starting from ``flips``, we flip the sign of channels whenever doing so increases the
    covariance correlation metric. Unlike the affinity-based solvers, this optimises the
    exact evaluation metric.

    Each sweep costs ``O(n_channels)`` rather than ``O(n_channels * D**2)``, as it never
    touches the full covariance while scoring the identical exact metric.

    Two strategies are available:

    - ``"first"`` (first-improvement / Gauss-Seidel): sweep over the channels and apply each
      improving single-channel flip as soon as it is found. Cheaper, but the result depends on
      the (fixed) channel order, since later channels are evaluated against the already-updated
      flips.
    - ``"best"`` (best-improvement / steepest ascent): evaluate every single-channel flip from
      the current state and apply only the single best one, repeating until no flip improves.
      Order-invariant (the chosen flip does not depend on channel order) and usually slightly
      better, at the cost of more metric evaluations.

    Both converge to a 1-flip local optimum: a state where no single-channel flip improves the
    metric.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix we would like to sign flip.
    template_cov : numpy.ndarray
        Template covariance matrix.
    flips : numpy.ndarray
        Initial (n_channels,) array of 1s and -1s to refine.
    n_embeddings : int
        Number of time-delay embeddings.
    n_refine : int
        For ``strategy="first"``, the maximum number of sweeps over the channels (the loop
        stops early once a full sweep makes no improvement). For ``strategy="best"``, ``n_refine``
        only scales a generous safety cap that is not reached in practice.
    strategy : str, optional
        Either ``"best"`` (default) or ``"first"``.
    tol : float, optional
        A flip is accepted only if it improves the metric by more than ``tol``. This guards
        against float-noise cycling.

    Returns
    -------
    flips : numpy.ndarray
        Refined (n_channels,) array of 1s and -1s.
    metric : float
        Covariance correlation metric of the refined flips.
    """
    if strategy not in ("first", "best"):
        raise ValueError(f"strategy must be 'first' or 'best', got '{strategy}'.")

    obj = FlipMetricEvaluator(cov, template_cov, n_embeddings)
    M, S = obj.M, obj.S

    f = apply_flip_convention(flips).astype(float)
    n_channels = len(f)
    u = M @ f
    v = S @ f
    P = 0.5 * (f @ u)
    Q = 0.5 * (f @ v)
    cur = float(obj.correlation_from_PQ(P, Q))

    def _accept(k, P_k, Q_k, cand):
        # Flip channel k and update the running fields; quadratic forms in O(n_channels)
        nonlocal P, Q, cur
        df = -2.0 * f[k]
        u[:] += M[:, k] * df
        v[:] += S[:, k] * df
        f[k] = -f[k]
        P, Q, cur = float(P_k), float(Q_k), float(cand)

    if strategy == "first":
        # First-improvement: apply each improving single-channel flip as it is found
        for _ in range(int(n_refine)):
            improved = False
            for k in range(n_channels):
                P_k = P - 2.0 * f[k] * u[k]
                Q_k = Q - 2.0 * f[k] * v[k]
                cand = float(obj.correlation_from_PQ(P_k, Q_k))
                if cand > cur + tol:
                    _accept(k, P_k, Q_k, cand)
                    improved = True
            if not improved:
                break
    else:
        # Best-improvement: evaluate all single-channel flips, apply only the best one
        # The cap is a safety bound; the loop converges (and breaks) well before it.
        for _ in range(int(n_refine) * n_channels):
            P_new = P - 2.0 * f * u
            Q_new = Q - 2.0 * f * v
            cand = obj.correlation_from_PQ(P_new, Q_new)
            k_best = int(np.argmax(cand))
            if cand[k_best] > cur + tol:
                _accept(k_best, P_new[k_best], Q_new[k_best], cand[k_best])
            else:
                break

    return apply_flip_convention(f), cur
