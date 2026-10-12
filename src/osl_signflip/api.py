"""High-level interface: pick a method by name and align a group of sessions."""

# Authors: SungJun Cho <sungjun.cho@ndcn.ox.ac.uk>

import logging
from functools import partial

import numpy as np

from ._utils import parallel_map
from .covariance import compute_covariance, compute_covariances
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


def _check_method(method):
    """Raises an error if ``method`` is not an available sign-flipping method."""
    if method not in _SOLVERS:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of {', '.join(map(repr, METHODS))}."
        )


def _stack_metrics(metrics):
    """Stacks per-session metric histories, filling the template's row (None) with NaN."""
    done = [m for m in metrics if m is not None]
    n_metrics = len(done[0]) if done else 0
    return np.array(
        [np.full(n_metrics, np.nan) if m is None else m for m in metrics], dtype=float
    )


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
    _check_method(method)
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
    template_cov=None,
    standardize_data=True,
    apply=False,
    n_jobs=1,
    use_tqdm=False,
    **kwargs,
):
    """Aligns the signs of a group of sessions to a common template.

    This computes the time-delay-embedded covariance of each session, selects a template
    session (unless one is given), and finds the flips that best align every other session
    to it. For parcellated data stored in ``.fif`` files, see
    :func:`osl_signflip.io.sign_flip_files`.

    Parameters
    ----------
    data : list of numpy.ndarray
        Parcel time series for each session. Each array must be (n_samples, n_channels);
        ``n_samples`` may differ across sessions but ``n_channels`` should not.
    n_embeddings : int
        Number of time-delay embeddings. Must be odd.
    method : str, optional
        Sign-flipping method. See :func:`find_flips`.
    template : int or numpy.ndarray, optional
        The template to align to. Either the index of a session in ``data``, or the
        (n_samples, n_channels) time series of a template session that is not in
        ``data``. If ``None`` (and ``template_cov`` is not given), the template is
        selected from ``data`` with :func:`~osl_signflip.find_template`.
    template_cov : numpy.ndarray, optional
        Precomputed covariance of a template session, as an alternative to ``template``.
        Shape must be (n_channels * n_embeddings, n_channels * n_embeddings).
    standardize_data : bool, optional
        Should we standardize the embedded data before computing the covariances?
    apply : bool, optional
        Should we also return the sign-flipped data?
    n_jobs : int, optional
        Number of parallel worker processes. ``-1`` uses all CPU cores.
    use_tqdm : bool, optional
        Should we display a tqdm progress bar?
    **kwargs
        Hyperparameters of the chosen method. See :func:`find_flips`.

    Returns
    -------
    flips : numpy.ndarray
        Flips for each session. Shape is (n_sessions, n_channels). If the template is a
        session in ``data``, its flips are all 1s.
    metrics : numpy.ndarray
        Evaluation metric history for each session. Shape is (n_sessions, n_metrics), where
        ``n_metrics`` depends on the method (see :func:`find_flips`). If the template is a
        session in ``data``, its row is ``NaN``.
    template : int or None
        Index of the template session in ``data``, or ``None`` if the template is not
        one of the sessions in ``data``.
    data_flipped : list of numpy.ndarray
        Sign-flipped time series for each session. Only returned if ``apply=True``.
    """
    _check_method(method)
    if template is not None and template_cov is not None:
        raise ValueError("Pass either template or template_cov, not both.")

    # Compute the covariance of each session
    covs = compute_covariances(
        data, n_embeddings, standardize_data, n_jobs=n_jobs, use_tqdm=use_tqdm
    )
    n_sessions = len(covs)
    n_channels = covs.shape[-1] // n_embeddings

    # Find the template
    template_index = None
    if template_cov is None:
        if template is None:
            template_index = find_template(
                covs, diag_offset=n_embeddings, use_tqdm=use_tqdm
            )
        elif np.ndim(template) == 0:
            template_index = int(template)
            if not 0 <= template_index < n_sessions:
                raise ValueError(
                    f"template must be an index in [0, {n_sessions}), got {template}."
                )
        else:
            template_cov = compute_covariance(template, n_embeddings, standardize_data)
    if template_index is not None:
        template_cov = covs[template_index]
        logger.info(f"using template: {template_index}")

    # Find sign flip solutions and compute corresponding metric values
    # (the template session, if in data, is not flipped against itself)
    todo = [i for i in range(n_sessions) if i != template_index]
    results = parallel_map(
        partial(find_flips, method=method, **kwargs),
        [(covs[i], template_cov, n_embeddings) for i in todo],
        n_jobs=n_jobs,
        desc="Sign flipping",
        use_tqdm=use_tqdm,
    )
    flips = np.ones((n_sessions, n_channels))
    metrics = [None] * n_sessions
    for i, (f, m) in zip(todo, results):
        flips[i], metrics[i] = f, m
    metrics = _stack_metrics(metrics)

    if apply:
        data_flipped = [apply_flips(x, f) for x, f in zip(data, flips)]
        return flips, metrics, template_index, data_flipped
    return flips, metrics, template_index
