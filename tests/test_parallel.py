"""Parallel processing and the extended align_signs options."""

import numpy as np
import pytest
from conftest import make_sessions, same_up_to_global_sign

import osl_signflip
from osl_signflip._utils import parallel_map


def test_parallel_map_keeps_order():
    args = [(i, 3) for i in range(10)]
    assert parallel_map(divmod, args, n_jobs=2) == parallel_map(divmod, args, n_jobs=1)
    assert parallel_map(divmod, args, n_jobs=2, use_tqdm=True) == [
        divmod(*a) for a in args
    ]


def test_parallel_map_reraises_worker_errors():
    with pytest.raises(ValueError):
        parallel_map(int, [("1",), ("x",)], n_jobs=2)


def test_compute_covariances_parallel_matches_serial():
    data, _ = make_sessions(n_sessions=4)
    serial = osl_signflip.compute_covariances(data, n_embeddings=3, n_jobs=1)
    parallel = osl_signflip.compute_covariances(data, n_embeddings=3, n_jobs=2)
    np.testing.assert_allclose(parallel, serial, rtol=0, atol=1e-12)


@pytest.mark.parametrize("method", osl_signflip.METHODS)
def test_align_signs_parallel_matches_serial(method):
    data, _ = make_sessions(n_sessions=4)
    kw = {"n_iter": 200, "max_flips": 4} if method == "random" else {}
    f1, m1, t1 = osl_signflip.align_signs(data, 3, method=method, n_jobs=1, **kw)
    f2, m2, t2 = osl_signflip.align_signs(data, 3, method=method, n_jobs=2, **kw)
    assert t1 == t2
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_allclose(m1, m2, rtol=0, atol=1e-12)


def test_align_signs_apply():
    data, _ = make_sessions(n_sessions=3)
    flips, _, _, data_flipped = osl_signflip.align_signs(data, 3, apply=True)
    assert len(data_flipped) == len(data)
    for x, f, y in zip(data, flips, data_flipped):
        np.testing.assert_array_equal(y, x * f)


def test_align_signs_external_template():
    data, planted = make_sessions(n_sessions=4, seed=5)
    sessions, template_data = data[:3], data[3]

    # Template given as time series of a session that is not in data
    flips, metrics, template = osl_signflip.align_signs(
        sessions, 3, template=template_data
    )
    assert template is None
    assert not np.isnan(metrics).any()  # every session is flipped against the template
    for s in range(3):
        assert same_up_to_global_sign(flips[s], planted[s] * planted[3])

    # Template given as a precomputed covariance gives the same result
    template_cov = osl_signflip.compute_covariance(template_data, 3)
    flips2, metrics2, template2 = osl_signflip.align_signs(
        sessions, 3, template_cov=template_cov
    )
    assert template2 is None
    np.testing.assert_array_equal(flips2, flips)
    np.testing.assert_array_equal(metrics2, metrics)


def test_align_signs_rejects_template_and_template_cov():
    data, _ = make_sessions(n_sessions=3)
    cov = osl_signflip.compute_covariance(data[0], 3)
    with pytest.raises(ValueError, match="either template or template_cov"):
        osl_signflip.align_signs(data, 3, template=0, template_cov=cov)


def test_parallel_errors_and_warnings_reach_the_user():
    data, _ = make_sessions(n_sessions=3, n_channels=6, n_samples=500)
    bad = [x.copy() for x in data]
    bad[1][5, 2] = np.nan
    with pytest.raises(ValueError, match=r"session 1: .*NaN"):
        osl_signflip.compute_covariances(bad, n_embeddings=3, n_jobs=2)
    with pytest.warns(UserWarning, match="did you pass"):
        osl_signflip.compute_covariances([x.T for x in data], n_embeddings=3, n_jobs=2)
