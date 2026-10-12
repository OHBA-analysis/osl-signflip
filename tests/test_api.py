import numpy as np
import pytest
from conftest import make_sessions, same_up_to_global_sign

import osl_signflip


def test_find_flips_rejects_unknown_method(pair):
    cov, template_cov, n_embeddings, _ = pair
    with pytest.raises(ValueError, match="Invalid method"):
        osl_signflip.find_flips(cov, template_cov, n_embeddings, method="nope")


def test_find_flips_rejects_foreign_hyperparameter(pair):
    cov, template_cov, n_embeddings, _ = pair
    with pytest.raises(TypeError):
        osl_signflip.find_flips(
            cov, template_cov, n_embeddings, method="sdp", bp_n_iter=10
        )


def test_find_flips_default_is_sdp(pair):
    cov, template_cov, n_embeddings, _ = pair
    f_default, m_default = osl_signflip.find_flips(cov, template_cov, n_embeddings)
    f_sdp, m_sdp = osl_signflip.find_flips_sdp(cov, template_cov, n_embeddings)
    np.testing.assert_array_equal(f_default, f_sdp)
    np.testing.assert_array_equal(m_default, m_sdp)


@pytest.mark.parametrize("template", [None, 2])
def test_align_signs(template):
    data, planted = make_sessions(n_sessions=5, seed=3)
    flips, metrics, tmpl = osl_signflip.align_signs(
        data, n_embeddings=3, template=template
    )
    assert flips.shape == (5, 10)
    assert metrics.shape == (5, 2)
    if template is not None:
        assert tmpl == template
    np.testing.assert_array_equal(flips[tmpl], 1)
    assert np.all(np.isnan(metrics[tmpl]))
    for s in range(5):
        if s != tmpl:
            assert same_up_to_global_sign(flips[s], planted[s] * planted[tmpl])
            assert metrics[s, -1] > metrics[s, 0]

    # After flipping, all sessions are sign-consistent with the template
    aligned = [osl_signflip.apply_flips(x, f) for x, f in zip(data, flips)]
    covs = osl_signflip.compute_covariances(aligned, n_embeddings=3)
    for s in range(5):
        assert osl_signflip.covariance_matrix_correlation(covs[s], covs[tmpl], 3) > 0.9


def test_align_signs_rejects_bad_template():
    data, _ = make_sessions(n_sessions=3)
    with pytest.raises(ValueError, match="template"):
        osl_signflip.align_signs(data, n_embeddings=3, template=3)


def test_apply_flips():
    x = np.random.default_rng(0).standard_normal((2, 100, 4))
    flips = np.array([1.0, -1.0, 1.0, -1.0])
    y = osl_signflip.apply_flips(x, flips)
    np.testing.assert_array_equal(y[..., 1], -x[..., 1])
    np.testing.assert_array_equal(y[..., 0], x[..., 0])
    with pytest.raises(ValueError, match="channel/parcel axis"):
        osl_signflip.apply_flips(x.swapaxes(1, 2), flips)
