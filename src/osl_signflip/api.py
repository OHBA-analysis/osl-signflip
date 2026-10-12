"""High-level interface: pick a method by name and align a group of sessions."""

# Authors: SungJun Cho <sungjun.cho@ndcn.ox.ac.uk>

import logging

import numpy as np

from .covariance import compute_covariances
from .solvers import (
    find_flips_belief_propagation,
    find_flips_random,
    find_flips_sdp,
    find_flips_spectral,
)
from .template import find_template

logger = logging.getLogger(__name__)

_SOLVERS = {
    "sdp": find_flips_sdp,
    "spectral": find_flips_spectral,
    "belief_propagation": find_flips_belief_propagation,
    "random": find_flips_random,
}

METHODS = tuple(_SOLVERS)  # names of the available sign-flipping methods


def find_flips(cov, template_cov, n_embeddings, method="sdp", **kwargs):
    """Finds the channels to flip so that a session's covariance best matches a template.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix we would like to sign flip.
        Shape must be (n_channels * n_embeddings, n_channels * n_embeddings).
    template_cov : numpy.ndarray
        Template covariance matrix.
        Shape must be (n_channels * n_embeddings, n_channels * n_embeddings).
    n_embeddings : int
        Number of time-delay embeddings used to build the covariances.
    method : str, optional
        One of ``"sdp"`` (semidefinite relaxation with Goemans-Williamson rounding, default),
        ``"spectral"`` (spectral synchronization), ``"belief_propagation"`` (loopy belief
        propagation) or ``"random"`` (random search).
    **kwargs
        Hyperparameters of the chosen method, e.g. ``sdp_n_restarts`` or ``random_state``.
        Passing a hyperparameter the chosen method does not accept raises a ``TypeError``.

    Returns
    -------
    flips : numpy.ndarray
        A (n_channels,) array of 1s and -1s indicating whether or not to flip a channel.
    metrics : numpy.ndarray
        Evaluation metric (correlation between covariance matrices). The first entry is
        before flipping and the last entry is that of the returned flips.
    """
    if method not in _SOLVERS:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of {', '.join(map(repr, METHODS))}."
        )
    return _SOLVERS[method](cov, template_cov, n_embeddings, **kwargs)


def apply_flips(x, flips):
    """Applies flips to parcel time series.

    Parameters
    ----------
    x : numpy.ndarray
        Parcel time series. Shape must be (..., n_channels), e.g. (n_samples, n_channels)
        or (n_epochs, n_samples, n_channels).
    flips : numpy.ndarray
        A (n_channels,) array of 1s and -1s.

    Returns
    -------
    x_flipped : numpy.ndarray
        Sign-flipped time series with the same shape as ``x``.
    """
    x = np.asarray(x)
    flips = np.asarray(flips)
    if flips.ndim != 1 or x.shape[-1] != flips.shape[0]:
        raise ValueError(
            f"flips of shape {flips.shape} is incompatible with data of shape {x.shape}; "
            "the last axis of the data must be the channel/parcel axis."
        )
    return x * flips


def align_signs(
    data,
    n_embeddings,
    method="sdp",
    template=None,
    standardize_data=True,
    use_tqdm=False,
    **kwargs,
):
    """Aligns the signs of a group of sessions to a common template.

    This computes the time-delay-embedded covariance of each session, selects a template
    session (unless one is given), and finds the flips that best align every other session
    to it.

    Parameters
    ----------
    data : list of numpy.ndarray
        Parcel time series for each session. Each array must be (n_samples, n_channels);
        ``n_samples`` may differ across sessions but ``n_channels`` should not.
    n_embeddings : int
        Number of time-delay embeddings. Must be odd.
    method : str, optional
        Sign-flipping method. See :func:`find_flips`.
    template : int, optional
        Index of the template session in ``data``. If ``None``, the template is selected
        with :func:`~osl_signflip.find_template`.
    standardize_data : bool, optional
        Should we standardize the embedded data before computing the covariances?
    use_tqdm : bool, optional
        Should we display a tqdm progress bar?
    **kwargs
        Hyperparameters of the chosen method. See :func:`find_flips`.

    Returns
    -------
    flips : numpy.ndarray
        Flips for each session. Shape is (n_sessions, n_channels). The template session's
        flips are all 1s.
    metrics : numpy.ndarray
        Evaluation metric history for each session. Shape is (n_sessions, n_metrics), where
        ``n_metrics`` depends on the method (see :func:`find_flips`). The template session's
        row is ``NaN``.
    template : int
        Index of the template session.
    """
    # Validate the designated method
    if method not in _SOLVERS:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of {', '.join(map(repr, METHODS))}."
        )

    covs = compute_covariances(data, n_embeddings, standardize_data, use_tqdm=use_tqdm)
    n_sessions = len(covs)
    n_channels = covs.shape[-1] // n_embeddings

    # Find the template session
    if template is None:
        template = find_template(covs, diag_offset=n_embeddings, use_tqdm=use_tqdm)
    template = int(template)
    if not 0 <= template < n_sessions:
        raise ValueError(
            f"template must be an index in [0, {n_sessions}), got {template}."
        )
    logger.info(f"using template: {template}")

    # Find sign flip solutions and compute corresponding metric values
    flips = np.ones((n_sessions, n_channels))
    metrics = [None] * n_sessions
    for i in range(n_sessions):
        if i == template:
            continue
        flips[i], metrics[i] = find_flips(
            covs[i], covs[template], n_embeddings, method=method, **kwargs
        )

    # The template is not flipped against itself; fill its metric row with NaN
    n_metrics = len(next(m for m in metrics if m is not None)) if n_sessions > 1 else 0
    metrics = np.array(
        [np.full(n_metrics, np.nan) if m is None else m for m in metrics], dtype=float
    )

    return flips, metrics, template
