import numpy as np

import osl_signflip
from osl_signflip.metric import FlipMetricEvaluator


def test_apply_flips_to_covariance(pair):
    cov, _, n_embeddings, _ = pair
    n_channels = cov.shape[0] // n_embeddings
    flips = np.random.default_rng(0).choice([-1.0, 1.0], size=n_channels)
    flipped = osl_signflip.apply_flips_to_covariance(cov, flips, n_embeddings)
    expected = (
        np.outer(np.repeat(flips, n_embeddings), np.repeat(flips, n_embeddings)) * cov
    )
    np.testing.assert_allclose(flipped, expected)
    # Flipping twice is the identity
    np.testing.assert_allclose(
        osl_signflip.apply_flips_to_covariance(flipped, flips, n_embeddings), cov
    )


def test_affinities_are_symmetric_with_zero_diagonal(pair):
    cov, template_cov, n_embeddings, _ = pair
    obj = FlipMetricEvaluator(cov, template_cov, n_embeddings)
    for A in (obj.M, obj.S):
        np.testing.assert_allclose(A, A.T)
        np.testing.assert_array_equal(np.diag(A), 0)


def test_evaluator_matches_full_metric(pair):
    cov, template_cov, n_embeddings, _ = pair
    n_channels = cov.shape[0] // n_embeddings
    obj = FlipMetricEvaluator(cov, template_cov, n_embeddings)
    rng = np.random.default_rng(0)
    for _ in range(20):
        flips = rng.choice([-1.0, 1.0], size=n_channels)
        full = osl_signflip.covariance_matrix_correlation(
            osl_signflip.apply_flips_to_covariance(cov, flips, n_embeddings),
            template_cov,
            n_embeddings,
        )
        np.testing.assert_allclose(obj.metric(flips), full, atol=1e-12)


def test_global_sign_invariance(pair):
    cov, template_cov, n_embeddings, _ = pair
    n_channels = cov.shape[0] // n_embeddings
    obj = FlipMetricEvaluator(cov, template_cov, n_embeddings)
    flips = np.random.default_rng(1).choice([-1.0, 1.0], size=n_channels)
    np.testing.assert_allclose(obj.metric(flips), obj.metric(-flips))
