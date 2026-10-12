"""Semidefinite programming (SDP) relaxation with Goemans-Williamson rounding."""

# Authors: SungJun Cho <sungjun.cho@ndcn.ox.ac.uk>

import logging

import numpy as np

from .._utils import apply_flip_convention
from ..metric import (
    FlipMetricEvaluator,
    apply_flips_to_covariance,
    covariance_matrix_correlation,
)
from ..refine import greedy_refine

logger = logging.getLogger(__name__)


def _solve_sdp_mixing(
    M,
    rank=None,
    n_restarts=5,
    max_iter=1000,
    tol=1e-8,
    rng=None,
):
    """Solves the Z2 synchronization problem using SDP with the low-rank
    mixing method.

    Parameters
    ----------
    M : numpy.ndarray
        Symmetric (n_channels, n_channels) affinity matrix with a zero diagonal.
    rank : int, optional
        Factorization rank. Defaults to ``ceil(sqrt(2n)) + 1``, which guarantees the
        global-optimality property.
    n_restarts : int, optional
        Number of random restarts wherein the best objective is kept.
    max_iter : int, optional
        Maximum number of coordinate-descent sweeps per restart.
    tol : float, optional
        Convergence tolerance on the largest row update within a sweep.
    rng : numpy.random.Generator, optional
        Random generator used for the restarts (for reproducibility).

    Returns
    -------
    value : float
        Best SDP objective ``<M, X>`` over the restarts.
    V : numpy.ndarray
        (n_channels, rank) factor of the best solution; rows are unit-norm.
    """
    # Validate inputs
    n = M.shape[0]
    if rank is None:
        rank = int(np.ceil(np.sqrt(2 * n))) + 1
    if rng is None:
        rng = np.random.default_rng(0)
    eps = 1e-12

    # Apply SDP solver
    best_value = -np.inf
    best_V = None
    for _ in range(int(n_restarts)):
        V = rng.standard_normal((n, rank))
        V /= np.linalg.norm(V, axis=1, keepdims=True) + eps
        for _ in range(int(max_iter)):
            max_change = 0.0
            for i in range(n):
                g = (
                    M[i] @ V
                )  # sum_j M[i, j] v_j (the diagonal term vanishes as M[i, i] = 0)
                g_norm = np.linalg.norm(g)
                if g_norm > eps:
                    new_v = g / g_norm
                    max_change = max(max_change, np.linalg.norm(new_v - V[i]))
                    V[i] = new_v
            if max_change < tol:
                break
        value = float(np.sum(M * (V @ V.T)))
        if value > best_value:
            best_value = value
            best_V = V

    return best_value, best_V


def _gw_round(M, V, n_samples=500, rng=None):
    """Rounds an SDP factor to a flip vector using Goemans-Williamson
    hyperplane rounding.

    Given the factor ``V`` of the SDP solution, we sample random hyperplanes
    ``r`` and read off the cut ``f_i = sign(v_i . r)``, keeping the sample that
    maximises ``f.T @ M @ f``. This recovers a discrete ±1 assignment from the
    continuous relaxation.

    Parameters
    ----------
    M : numpy.ndarray
        Symmetric (n_channels, n_channels) affinity matrix.
    V : numpy.ndarray
        (n_channels, rank) SDP factor from :func:`_solve_sdp_mixing`.
    n_samples : int, optional
        Number of random hyperplanes to try.
    rng : numpy.random.Generator, optional
        Random generator used for the hyperplanes (for reproducibility).

    Returns
    -------
    flips : numpy.ndarray
        A (n_channels,) array of 1s and -1s (sign convention applied).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    best_flips = None
    best_value = -np.inf
    for _ in range(int(n_samples)):
        r = rng.standard_normal(V.shape[1])
        f = np.sign(V @ r)
        f[f == 0] = 1.0
        value = float(f @ M @ f)
        if value > best_value:
            best_value = value
            best_flips = f

    return apply_flip_convention(best_flips)


def find_flips_sdp(
    cov,
    template_cov,
    n_embeddings,
    sdp_rank=None,
    sdp_n_restarts=5,
    sdp_max_iter=1000,
    sdp_tol=1e-8,
    gw_samples=500,
    refine=True,
    n_refine=5,
    refine_strategy="best",
    random_state=0,
):
    """Finds channels to flip using a semidefinite relaxation.

    This solver targets the Z2 (sign) synchronization problem through the tighter
    semidefinite relaxation

        maximize  <M, X>   subject to   X >= 0 (positive semidefinite),  diag(X) = 1,

    where ``X`` relaxes the rank-one product ``flips @ flips.T``. Spectral synchronization
    is the looser relaxation that keeps only ``trace(X) = n``, so the SDP is at least as
    tight. The relaxation is solved with the low-rank mixing method, and the continuous
    solution is rounded back to ±1 by Goemans-Williamson hyperplane rounding. By default
    the solution is then polished with greedy refinement, which optimizes the exact metric.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix we would like to sign flip.
    template_cov : numpy.ndarray
        Template covariance matrix.
    n_embeddings : int
        Number of time-delay embeddings.
    sdp_rank : int, optional
        Rank of the mixing-method factorisation. Defaults to ``ceil(sqrt(2 n)) + 1``.
    sdp_n_restarts : int, optional
        Number of random restarts for the mixing method.
    sdp_max_iter : int, optional
        Maximum number of coordinate-descent sweeps per restart.
    sdp_tol : float, optional
        Convergence tolerance for the mixing method.
    gw_samples : int, optional
        Number of random hyperplanes for Goemans-Williamson rounding.
    refine : bool, optional
        Should we polish the rounded solution with greedy refinement?
    n_refine : int, optional
        Maximum number of greedy refinement sweeps (only used if ``refine=True``).
    refine_strategy : str, optional
        Greedy refinement strategy, ``"first"`` (first-improvement) or
        ``"best"`` (best-improvement / steepest ascent, default).
        See :func:`osl_signflip.refine.greedy_refine`.
    random_state : int, optional
        Seed for the mixing-method restarts and hyperplane rounding, so the result is
        reproducible.

    Returns
    -------
    best_flips : numpy.ndarray
        A (n_channels,) array of 1s and -1s indicating whether or not to flip a channel.
    metrics : numpy.ndarray
        Evaluation metric (correlation between covariance matrices) before and after
        flipping. Shape is (2,).
    """
    logger.info("find_flips_sdp")

    # Set random seed
    rng = np.random.default_rng(random_state)

    # Get the affinity matrix
    M = FlipMetricEvaluator(cov, template_cov, n_embeddings).M

    # Solve the SDP relaxation and round the continuous solution back to +/-1
    _, V = _solve_sdp_mixing(
        M,
        rank=sdp_rank,
        n_restarts=sdp_n_restarts,
        max_iter=sdp_max_iter,
        tol=sdp_tol,
        rng=rng,
    )
    flips = _gw_round(M, V, n_samples=gw_samples, rng=rng)

    # Apply flips and evaluate the metric
    metric_before = covariance_matrix_correlation(cov, template_cov, n_embeddings)
    if refine:
        flips, metric_after = greedy_refine(
            cov, template_cov, flips, n_embeddings, n_refine, strategy=refine_strategy
        )
    else:
        new_cov = apply_flips_to_covariance(cov, flips, n_embeddings)
        metric_after = covariance_matrix_correlation(new_cov, template_cov, n_embeddings)
    metrics = np.array([metric_before, metric_after], dtype=float)

    logger.info(f"metric before: {metric_before}, after: {metric_after}")

    return flips, metrics
