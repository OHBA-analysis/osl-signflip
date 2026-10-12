"""Loopy belief propagation."""

# Authors: SungJun Cho <sungjun.cho@ndcn.ox.ac.uk>

import logging

import numpy as np

from .._utils import apply_flip_convention
from ..metric import FlipMetricEvaluator, covariance_matrix_correlation
from ..refine import greedy_refine

logger = logging.getLogger(__name__)


def find_flips_belief_propagation(
    cov,
    template_cov,
    n_embeddings,
    bp_n_iter=200,
    beta=2.0,
    damping=0.5,
    prior_strength=1e-3,
    bp_tol=1e-6,
    refine=True,
    n_refine=5,
    refine_strategy="best",
):
    """Finds channels to flip using loopy belief propagation.

    This is a deterministic solver for the Z2 (sign) synchronization problem by applying
    message passing on an Ising-style model whose pairwise couplings are the channel
    affinity matrix. Belief propagation on a fully-connected graph is approximate (it may
    not converge and can find a local optimum), but it provides a deterministic, reproducible
    alternative to the random search. By default the solution is then polished with greedy
    refinement, which optimizes the exact metric.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix we would like to sign flip.
    template_cov : numpy.ndarray
        Template covariance matrix.
    n_embeddings : int
        Number of time-delay embeddings.
    bp_n_iter : int, optional
        Maximum number of message-passing iterations.
    beta : float, optional
        Inverse temperature. Larger values give more decisive updates.
    damping : float, optional
        Damping factor in [0, 1). Higher values are slower but more stable.
    prior_strength : float, optional
        Small deterministic symmetry-breaking term. Without it, the zero-field model
        can stay at the trivial zero-message fixed point.
    bp_tol : float, optional
        Convergence tolerance on the message updates.
    refine : bool, optional
        Should we polish the belief propagation solution with greedy refinement?
    n_refine : int, optional
        Maximum number of greedy refinement sweeps (only used if ``refine=True``).
    refine_strategy : str, optional
        Greedy refinement strategy, ``"first"`` or ``"best"`` (default).
        See :func:`osl_signflip.refine.greedy_refine`.

    Returns
    -------
    best_flips : numpy.ndarray
        A (n_channels,) array of 1s and -1s indicating whether or not to flip a channel.
    metrics : numpy.ndarray
        Evaluation metric (correlation between covariance matrices) as a function of
        iterations, including the initial metric and (if ``refine=True``) a final entry
        for the refined solution. The history is padded to a fixed length so every
        session returns the same shape. Shape is (bp_n_iter + 2,) if ``refine`` else
        (bp_n_iter + 1,).
    """
    logger.info("find_flips_belief_propagation")

    # Set numeric types
    bp_n_iter = int(bp_n_iter)
    beta = float(beta)
    damping = float(damping)
    prior_strength = float(prior_strength)
    bp_tol = float(bp_tol)

    # Get the affinity matrix
    obj = FlipMetricEvaluator(cov, template_cov, n_embeddings)
    W = obj.M
    n = W.shape[0]  # number of channels
    eps = 1e-12

    # Normalize the coupling scale so that tanh(beta * W) is well-behaved
    w_scale = np.max(np.abs(W))
    if w_scale > eps:
        W = W / w_scale
    # NOTE: If not normalized, the magnitude affinity is otherwise unbounded. This keeps the
    #       relative couplings but gives beta a consistent meaning.

    # Deterministic tiny unary bias to break symmetry
    row_strength = np.sum(W, axis=1)
    scale = np.max(np.abs(row_strength))
    if scale < eps:
        h = np.zeros(n, dtype=float)
    else:
        h = prior_strength * row_strength / scale
    # NOTE: This bias keeps the method reproducible and avoids the trivial zero fixed point.

    # Preallocate messages u[i, j] (represents the cavity influence from i -> j)
    u = np.zeros((n, n), dtype=float)  # store full dense matrix for simplicity
    tanh_betaW = np.tanh(beta * W)

    # Score the metric history
    metric_before = covariance_matrix_correlation(cov, template_cov, n_embeddings)
    metrics = [metric_before]
    flips = apply_flip_convention(np.ones(n))

    for _ in range(bp_n_iter):
        # Cavity field for every ordered pair: cavity[i, j] = h[i] + incoming[i] - u[j, i]
        incoming = np.sum(u, axis=0)  # sum_k u[k, i] for each i
        cavity = (h + incoming)[:, np.newaxis] - u.T

        # Ising BP update, vectorized over all pairs
        # (This is the vectorized equivalent of looping over all ordered pairs (i, j) with i != j.)
        tanh_prod = np.clip(tanh_betaW * np.tanh(cavity), -1.0 + eps, 1.0 - eps)
        new_u = np.arctanh(tanh_prod)
        np.fill_diagonal(new_u, 0.0)  # no self-messages (i == j)

        # Damping
        u_next = damping * u + (1.0 - damping) * new_u
        delta = np.max(np.abs(u_next - u))
        u = u_next

        # Current beliefs / node fields
        beliefs = h + np.sum(u, axis=0)
        flips = apply_flip_convention(np.sign(beliefs))
        metrics.append(obj.metric(flips))
        # NOTE: The metric history is scored through the exact evaluator, which reproduces
        #       `covariance_matrix_correlation` to machine precision but costs only
        #       O(n_channels**2) per iteration instead of rebuilding the full covariance.

        if delta < bp_tol:
            break

    # Optionally polish with greedy refinement
    if refine:
        flips, metric_refined = greedy_refine(
            cov, template_cov, flips, n_embeddings, n_refine, strategy=refine_strategy
        )
        metrics.append(metric_refined)

    logger.info(f"metric before: {metrics[0]}, after: {metrics[-1]}")

    # Pad the history to a fixed length so all sessions return the same shape
    target_len = bp_n_iter + (2 if refine else 1)
    metrics = metrics + [metrics[-1]] * (target_len - len(metrics))
    metrics = np.asarray(metrics, dtype=float)

    return flips, metrics
