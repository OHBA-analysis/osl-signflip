"""Spectral synchronization."""

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


def find_flips_spectral(
    cov,
    template_cov,
    n_embeddings,
    refine=True,
    n_refine=5,
    refine_strategy="best",
):
    """Finds channels to flip using spectral synchronization.

    This is a deterministic solver for the underlying Z2 (sign) synchronization
    problem: given the channel-by-channel affinity matrix, the signs of the leading
    eigenvector maximize the quadratic agreement ``flips.T @ W @ flips`` over the unit
    sphere, which is the standard convex relaxation of the discrete sign assignment.
    It is fast and reproducible, with no random initializations. By default the spectral
    solution is then polished with greedy refinement, which optimizes the exact metric.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix we would like to sign flip.
    template_cov : numpy.ndarray
        Template covariance matrix.
    n_embeddings : int
        Number of time-delay embeddings.
    refine : bool, optional
        Should we polish the spectral solution with greedy refinement?
    n_refine : int, optional
        Maximum number of greedy refinement sweeps (only used if ``refine=True``).
    refine_strategy : str, optional
        Greedy refinement strategy, ``"first"`` or ``"best"`` (default).
        See :func:`osl_signflip.refine.greedy_refine`.

    Returns
    -------
    best_flips : numpy.ndarray
        A (n_channels,) array of 1s and -1s indicating whether or not to flip
        a channel.
    metrics : numpy.ndarray
        Evaluation metric (correlation between covariance matrices) before and
        after flipping. Shape is (2,).
    """
    logger.info("find_flips_spectral")

    # Get the affinity matrix
    W = FlipMetricEvaluator(cov, template_cov, n_embeddings).M

    # Signs of the eigenvector with the largest eigenvalue give the spectral solution
    eigvals, eigvecs = np.linalg.eigh(W)
    v = eigvecs[:, np.argmax(eigvals)]
    flips = apply_flip_convention(np.sign(v))

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
