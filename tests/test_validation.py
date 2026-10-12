"""Invalid inputs must raise clear errors (or warn) rather than crash or return garbage."""

import warnings

import numpy as np
import pytest
from conftest import make_sessions

import osl_signflip


@pytest.fixture
def data():
    return make_sessions(n_sessions=3, n_channels=6, n_samples=500)[0]


def test_nan_raises_with_session_and_channel(data):
    data[2][5, 1] = np.nan
    with pytest.raises(ValueError, match=r"session 2: .*NaN.*\[1\]"):
        osl_signflip.align_signs(data, n_embeddings=3)


def test_constant_channel_raises_with_session_and_channel(data):
    data[1][:, 4] = 0.0
    with pytest.raises(ValueError, match=r"session 1: channel\(s\) \[4\] are constant"):
        osl_signflip.align_signs(data, n_embeddings=3)


def test_different_channel_counts_raise(data):
    data[1] = data[1][:, :5]
    with pytest.raises(
        ValueError, match="same number of channels; got session 0: 6, session 1: 5"
    ):
        osl_signflip.compute_covariances(data, n_embeddings=3)


def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="needs at least 4 samples, got 2"):
        osl_signflip.compute_covariance(
            np.random.default_rng(0).standard_normal((2, 6)), n_embeddings=3
        )


def test_transposed_data_warns(data):
    with pytest.warns(UserWarning, match=r"did you pass \(n_channels, n_samples\)"):
        osl_signflip.compute_covariances([x.T for x in data], n_embeddings=3)


def test_valid_data_does_not_warn(data):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        osl_signflip.align_signs(data, n_embeddings=3)


@pytest.mark.parametrize("method", osl_signflip.METHODS)
def test_non_finite_covariance_raises(pair, method):
    cov, template_cov, n_embeddings, _ = pair
    cov = cov.copy()
    cov[0, 5] = np.nan
    with pytest.raises(ValueError, match="cov contains NaN"):
        osl_signflip.find_flips(cov, template_cov, n_embeddings, method=method)


def test_mismatched_covariances_raise(pair):
    cov, template_cov, n_embeddings, _ = pair
    with pytest.raises(ValueError, match="same shape"):
        osl_signflip.find_flips_random(
            cov, template_cov[:-3, :-3], n_embeddings, max_flips=5
        )
