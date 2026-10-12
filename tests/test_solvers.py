import numpy as np
import pytest
from conftest import same_up_to_global_sign

import osl_signflip
from osl_signflip.metric import FlipMetricEvaluator

# Small budgets keep the random search fast on the 10-channel test problem
FAST = {"random": {"n_iter": 300, "max_flips": 5}}


@pytest.mark.parametrize("method", osl_signflip.METHODS)
def test_recovers_planted_flips(pair, method):
    cov, template_cov, n_embeddings, true_flips = pair
    flips, metrics = osl_signflip.find_flips(
        cov, template_cov, n_embeddings, method=method, **FAST.get(method, {})
    )
    assert set(np.unique(flips)) <= {-1.0, 1.0}
    assert same_up_to_global_sign(flips, true_flips)
    assert metrics[-1] > metrics[0]
    # The last metric is that of the returned flips
    np.testing.assert_allclose(
        metrics[-1],
        FlipMetricEvaluator(cov, template_cov, n_embeddings).metric(flips),
        atol=1e-12,
    )


@pytest.mark.parametrize("method", osl_signflip.METHODS)
def test_reproducible(pair, method):
    cov, template_cov, n_embeddings, _ = pair
    kw = FAST.get(method, {})
    f1, m1 = osl_signflip.find_flips(cov, template_cov, n_embeddings, method=method, **kw)
    f2, m2 = osl_signflip.find_flips(cov, template_cov, n_embeddings, method=method, **kw)
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(m1, m2)


def test_sdp_seed_insensitive(pair):
    cov, template_cov, n_embeddings, _ = pair
    results = [
        osl_signflip.find_flips_sdp(cov, template_cov, n_embeddings, random_state=s)[0]
        for s in range(5)
    ]
    for f in results[1:]:
        np.testing.assert_array_equal(f, results[0])


@pytest.mark.parametrize("method", ["sdp", "spectral", "belief_propagation"])
@pytest.mark.parametrize("refine_strategy", ["first", "best"])
def test_refine_options(pair, method, refine_strategy):
    cov, template_cov, n_embeddings, _ = pair
    _, m_ref = osl_signflip.find_flips(
        cov, template_cov, n_embeddings, method=method, refine_strategy=refine_strategy
    )
    _, m_raw = osl_signflip.find_flips(
        cov, template_cov, n_embeddings, method=method, refine=False
    )
    assert m_ref[-1] >= m_raw[-1] - 1e-12


def test_metric_shapes(pair):
    cov, template_cov, n_embeddings, _ = pair
    assert osl_signflip.find_flips_sdp(cov, template_cov, n_embeddings)[1].shape == (2,)
    assert osl_signflip.find_flips_spectral(cov, template_cov, n_embeddings)[1].shape == (
        2,
    )
    assert osl_signflip.find_flips_belief_propagation(
        cov, template_cov, n_embeddings, bp_n_iter=50
    )[1].shape == (52,)
    assert osl_signflip.find_flips_belief_propagation(
        cov, template_cov, n_embeddings, bp_n_iter=50, refine=False
    )[1].shape == (51,)
    assert osl_signflip.find_flips_random(
        cov, template_cov, n_embeddings, n_restarts=2, n_iter=10, max_flips=3
    )[1].shape == (3,)


def test_greedy_refine_reaches_local_optimum(pair):
    cov, template_cov, n_embeddings, _ = pair
    n_channels = cov.shape[0] // n_embeddings
    start = np.random.default_rng(0).choice([-1.0, 1.0], size=n_channels)
    flips, metric = osl_signflip.greedy_refine(
        cov, template_cov, start, n_embeddings, n_refine=5
    )
    obj = FlipMetricEvaluator(cov, template_cov, n_embeddings)
    for k in range(n_channels):
        trial = flips.copy()
        trial[k] *= -1
        assert obj.metric(trial) <= metric + 1e-12


def test_random_rejects_too_many_flips(pair):
    cov, template_cov, n_embeddings, _ = pair
    with pytest.raises(ValueError, match="max_flips"):
        osl_signflip.find_flips_random(cov, template_cov, n_embeddings, max_flips=100)


def test_random_reports_true_metric_when_never_positive(pair):
    # With an anti-correlated template and no proposals, every restart ends below zero;
    # the returned metric must still be the real metric of the returned flips, not 0
    cov, _, n_embeddings, _ = pair
    flips, metrics = osl_signflip.find_flips_random(
        cov, -cov, n_embeddings, n_restarts=2, n_iter=0, max_flips=5
    )
    expected = osl_signflip.covariance_matrix_correlation(
        osl_signflip.apply_flips_to_covariance(cov, flips, n_embeddings),
        -cov,
        n_embeddings,
    )
    assert metrics[-1] < 0
    np.testing.assert_allclose(metrics, [expected] * 3)


@pytest.mark.parametrize("method", osl_signflip.METHODS)
def test_global_sign_convention(pair, method):
    # Every method flips the fewer channels and, on a tie, keeps the first channel unflipped
    cov, template_cov, n_embeddings, _ = pair
    for seed in range(3):
        kw = dict(FAST.get(method, {}))
        if method in ("random", "sdp"):
            kw["random_state"] = seed
        flips, _ = osl_signflip.find_flips(
            cov, template_cov, n_embeddings, method=method, **kw
        )
        assert flips.sum() > 0 or (flips.sum() == 0 and flips[0] == 1)
