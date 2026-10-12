"""Shared synthetic data for the osl_signflip tests."""

import numpy as np
import pytest

import osl_signflip


def make_sessions(n_sessions=4, n_channels=10, n_samples=4000, noise=0.3, seed=0):
    """Builds sessions that share a parcel signal up to a per-session sign flip.

    Returns the time series of each session and the planted signs, so that
    ``x_s * planted[s]`` are all sign-consistent.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_samples, n_channels))
    # Mix parcels and smooth in time so both zero- and non-zero-lag covariances are non-trivial
    mix = rng.standard_normal((n_channels, n_channels))
    kernel = np.ones(5) / 5
    base = (
        np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), 0, base) @ mix
    )
    data, planted = [], []
    for _ in range(n_sessions):
        f = rng.choice([-1.0, 1.0], size=n_channels)
        data.append(base * f + noise * rng.standard_normal((n_samples, n_channels)))
        planted.append(f)
    return data, np.array(planted)


def same_up_to_global_sign(a, b):
    return np.array_equal(a, b) or np.array_equal(a, -b)


@pytest.fixture
def pair():
    """A (cov, template_cov, n_embeddings, true_flips) tuple.

    ``true_flips`` are the flips that align the session to the template.
    """
    n_embeddings = 3
    data, planted = make_sessions(n_sessions=2, seed=1)
    cov = osl_signflip.compute_covariance(data[0], n_embeddings)
    template_cov = osl_signflip.compute_covariance(data[1], n_embeddings)
    return cov, template_cov, n_embeddings, planted[0] * planted[1]
