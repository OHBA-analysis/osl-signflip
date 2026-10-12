"""Random search (the stochastic baseline).

Adapted from ``osl_ephys.source_recon.sign_flipping``.
(MIT License, OHBA Analysis Group)
"""

# Authors: SungJun Cho <sungjun.cho@ndcn.ox.ac.uk>

import logging

import numpy as np

from .._utils import apply_flip_convention, progress_range
from ..metric import (
    apply_flips_to_covariance,
    check_covariances,
    covariance_matrix_correlation,
)

logger = logging.getLogger(__name__)


def randomly_flip(flips, max_flips, rng=None):
    """Randomly flips some channels (or parcels).

    Parameters
    ----------
    flips : numpy.ndarray
        Vector of 1s and -1s indicating which channels to flip.
    max_flips : int
        Maximum number of channels to change in this function.
    rng : numpy.random.Generator, optional
        Random generator used to draw the flips. If ``None``, an unseeded generator is
        created, so the result is not reproducible; pass a seeded generator (e.g. from
        ``numpy.random.default_rng(seed)``) for reproducibility.

    Returns
    -------
    new_flips : numpy.ndarray
        Vector of 1s and -1s indicating which channels to flip.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Select the number of channels to flip
    n_channels_to_flip = rng.integers(1, max_flips + 1)

    # Select the channels to flip
    n_channels = flips.shape[0]
    random_channels_to_flip = rng.choice(
        n_channels, size=n_channels_to_flip, replace=False
    )
    new_flips = np.copy(flips)
    new_flips[random_channels_to_flip] *= -1

    return new_flips


def find_flips_random(
    cov,
    template_cov,
    n_embeddings,
    n_restarts=3,
    n_iter=2500,
    max_flips=20,
    use_tqdm=False,
    random_state=0,
):
    """Finds channels to flip using random search.

    We search for the channels to flip by randomly flipping them and saving the
    flips that maximise the correlation of the covariance matrices between sessions.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix we would like to sign flip.
    template_cov : numpy.ndarray
        Template covariance matrix.
    n_embeddings : int
        Number of time-delay embeddings.
    n_restarts : int, optional
        Number of independent random searches (restarts) to perform.
    n_iter : int, optional
        Number of sign flipping proposals per restart.
    max_flips : int, optional
        Maximum number of channels to flip in a proposal.
    use_tqdm : bool, optional
        Should we display a tqdm progress bar?
    random_state : int, optional
        Seed for the random search, so the result is reproducible (default 0).
        Pass ``None`` for an unseeded generator, in which case the search is
        not reproducible.

    Returns
    -------
    best_flips : numpy.ndarray
        A (n_channels,) array of 1s and -1s indicating whether or not to flip
        a channel.
    metrics : numpy.ndarray
        Evaluation metric (correlation between covariance matrices) before flipping,
        followed by the best metric after each restart. Shape is (n_restarts + 1,).
    """
    logger.info("find_flips_random")

    # Set random seed
    rng = np.random.default_rng(random_state)

    # Validate inputs and get the number of channels
    cov, template_cov = check_covariances(cov, template_cov, n_embeddings)
    n_channels = cov.shape[-1] // n_embeddings

    # Validation
    if max_flips > n_channels:
        raise ValueError(
            f"max_flips ({max_flips}) must not exceed the number of channels ({n_channels})"
        )

    # Find the best channels to flip
    best_flips = np.ones(n_channels)
    best_metric = (
        -np.inf
    )  # so the first restart is always kept, even if its metric is <= 0
    metrics = []
    for n in range(n_restarts):
        # Reset the flips and calculate the evaluation metric before sign flipping
        flips = np.ones(n_channels)
        metric = covariance_matrix_correlation(cov, template_cov, n_embeddings)
        if n == 0:
            metrics.append(metric)
            logger.info(f"restart #{n}, unflipped metric: {metric}")

        # Randomly permute the sign of different channels and calculate the metric
        for _ in progress_range(n_iter, desc="sign flipping", use_tqdm=use_tqdm):
            new_flips = randomly_flip(flips, max_flips, rng=rng)
            new_cov = apply_flips_to_covariance(cov, new_flips, n_embeddings)
            new_metric = covariance_matrix_correlation(
                new_cov, template_cov, n_embeddings
            )
            if new_metric > metric:
                # Save the improved solution
                flips = new_flips
                metric = new_metric

        # Update best_flips if this was the best search
        if metric > best_metric:
            best_flips = flips
            best_metric = metric

        # Save metric as a function of restarts
        metrics.append(best_metric)
        logger.info(f"restart #{n}, current best metric: {best_metric}")

    return apply_flip_convention(best_flips), np.asarray(metrics, dtype=float)
