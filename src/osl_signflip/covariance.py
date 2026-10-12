"""Time-delay-embedded (TDE) covariance of parcel time series.

Adapted from ``osl_ephys.source_recon.sign_flipping``.
(MIT License, OHBA Analysis Group)
"""

# Authors: SungJun Cho <sungjun.cho@ndcn.ox.ac.uk>

import warnings

import numpy as np

from ._utils import progress_range


def _check_data(x, n_embeddings, where=""):
    """Validates the parcel time series of one session.

    Parameters
    ----------
    x : numpy.ndarray
        Parcel time series. Shape must be (n_samples, n_channels).
    n_embeddings : int
        Number of time-delay embeddings.
    where : str, optional
        Prefix identifying the session in error messages (e.g. ``"session 3: "``).

    Returns
    -------
    x : numpy.ndarray
        The validated data as an array.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(
            f"{where}x must be a 2D (n_samples, n_channels) array, got shape {x.shape}."
        )
    n_samples, n_channels = x.shape
    if n_samples < n_embeddings + 1:
        raise ValueError(
            f"{where}time-delay embedding with n_embeddings={n_embeddings} needs at least "
            f"{n_embeddings + 1} samples, got {n_samples}."
        )
    bad = np.flatnonzero(~np.all(np.isfinite(x), axis=0))
    if bad.size:
        raise ValueError(
            f"{where}data contain NaN or infinite values in channel(s) {bad.tolist()}."
        )
    flat = np.flatnonzero(np.all(x == x[0], axis=0))
    if flat.size:
        raise ValueError(
            f"{where}channel(s) {flat.tolist()} are constant (zero variance), so their sign "
            "cannot be determined. Remove them before sign flipping."
        )
    if n_channels > n_samples:
        warnings.warn(
            f"{where}data have more channels ({n_channels}) than samples ({n_samples}). "
            "Data must be (n_samples, n_channels); did you pass (n_channels, n_samples)?",
            UserWarning,
            stacklevel=4,
        )
    return x


def _compute_covariance(x, n_embeddings, standardize_data, where=""):
    """Validates the data, then computes its time-delay-embedded covariance."""
    x = _check_data(x, n_embeddings, where)
    x = time_embed(x, n_embeddings)
    if standardize_data:
        x = standardize(x)
    return np.cov(x, rowvar=False)


def _stack_covariances(covs, labels, n_embeddings):
    """Stacks per-session covariances, checking that they have the same shape."""
    shapes = {cov.shape for cov in covs}
    if len(shapes) > 1:
        counts = ", ".join(
            f"{label}: {cov.shape[0] // n_embeddings}" for label, cov in zip(labels, covs)
        )
        raise ValueError(
            f"All sessions must have the same number of channels; got {counts}."
        )
    return np.array(covs)


def time_embed(x, n_embeddings):
    """Performs time-delay embedding.

    Parameters
    ----------
    x : numpy.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    n_embeddings : int
        Number of samples in which to shift the data. Must be an odd number.

    Returns
    -------
    x_te : numpy.ndarray
        Time embedded data. Shape is (n_samples - n_embeddings + 1, n_channels * n_embeddings).
        The ordering is channel-major, i.e. embedded dimension ``d`` belongs to channel
        ``d // n_embeddings``.
    """
    if n_embeddings % 2 == 0:
        raise ValueError("n_embeddings must be an odd number.")

    te_shape = (x.shape[0] - (n_embeddings - 1), x.shape[1] * n_embeddings)

    return (
        np.lib.stride_tricks.sliding_window_view(x=x, window_shape=te_shape[0], axis=0)
        .T[..., ::-1]
        .reshape(te_shape)
    )


def standardize(x):
    """Standardizes (z-transforms) the data.

    Parameters
    ----------
    x : numpy.ndarray
        Data. Shape must be (n_samples, n_channels).

    Returns
    -------
    std_x : numpy.ndarray
        Standardized time series.
    """
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


def compute_covariance(x, n_embeddings=1, standardize_data=True):
    """Computes the TDE covariance of one session.

    Parameters
    ----------
    x : numpy.ndarray
        Parcel time series. Shape must be (n_samples, n_channels).
    n_embeddings : int, optional
        Number of time-delay embeddings. Must be odd. ``1`` gives the ordinary
        (zero-lag) covariance.
    standardize_data : bool, optional
        Should we standardize the embedded data before computing the covariance?

    Returns
    -------
    cov : numpy.ndarray
        Covariance matrix. Shape is (n_channels * n_embeddings, n_channels * n_embeddings).
    """
    return _compute_covariance(x, n_embeddings, standardize_data)


def compute_covariances(data, n_embeddings=1, standardize_data=True, use_tqdm=False):
    """Computes the TDE covariance of multiple sessions.

    Parameters
    ----------
    data : list of numpy.ndarray
        Parcel time series for each session. Each array must be (n_samples, n_channels);
        ``n_samples`` may differ across sessions but ``n_channels`` may not.
    n_embeddings : int, optional
        Number of time-delay embeddings. Must be odd.
    standardize_data : bool, optional
        Should we standardize the embedded data before computing the covariance?
    use_tqdm : bool, optional
        Should we display a tqdm progress bar?

    Returns
    -------
    covs : numpy.ndarray
        Covariance matrices. Shape is (n_sessions, n_channels * n_embeddings,
        n_channels * n_embeddings).
    """
    covs = []
    for i in progress_range(len(data), desc="Calculating covariances", use_tqdm=use_tqdm):
        covs.append(
            _compute_covariance(
                data[i], n_embeddings, standardize_data, where=f"session {i}: "
            )
        )
    return _stack_covariances(
        covs,
        [f"session {i}" for i in range(len(covs))],
        n_embeddings,
    )
