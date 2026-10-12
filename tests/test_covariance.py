import numpy as np
import pytest

import osl_signflip


def test_time_embed_shape_and_lags():
    n_samples, n_channels, n_embeddings = 50, 3, 5
    x = np.random.default_rng(0).standard_normal((n_samples, n_channels))
    te = osl_signflip.time_embed(x, n_embeddings)
    n_valid = n_samples - n_embeddings + 1
    assert te.shape == (n_valid, n_channels * n_embeddings)

    # Channel-major ordering: each block of n_embeddings columns holds lagged copies of one channel
    for c in range(n_channels):
        block = te[:, c * n_embeddings : (c + 1) * n_embeddings]
        offsets = sorted(
            next(
                s
                for s in range(n_embeddings)
                if np.array_equal(block[:, k], x[s : s + n_valid, c])
            )
            for k in range(n_embeddings)
        )
        assert offsets == list(range(n_embeddings))


def test_time_embed_rejects_even():
    with pytest.raises(ValueError, match="odd"):
        osl_signflip.time_embed(np.zeros((10, 2)), 4)


def test_standardize():
    x = np.random.default_rng(0).normal(3.0, 2.0, size=(1000, 4))
    z = osl_signflip.standardize(x)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-12)


def test_compute_covariance():
    x = np.random.default_rng(0).standard_normal((500, 4))
    cov = osl_signflip.compute_covariance(x, n_embeddings=3)
    assert cov.shape == (12, 12)
    np.testing.assert_allclose(cov, cov.T)

    # n_embeddings=1 without standardization is the ordinary covariance
    np.testing.assert_allclose(
        osl_signflip.compute_covariance(x, 1, standardize_data=False),
        np.cov(x, rowvar=False),
    )

    with pytest.raises(ValueError, match="2D"):
        osl_signflip.compute_covariance(x[None], 3)


def test_compute_covariances_variable_lengths():
    rng = np.random.default_rng(0)
    data = [rng.standard_normal((n, 4)) for n in (300, 500, 400)]
    covs = osl_signflip.compute_covariances(data, n_embeddings=3)
    assert covs.shape == (3, 12, 12)
    np.testing.assert_allclose(covs[1], osl_signflip.compute_covariance(data[1], 3))
