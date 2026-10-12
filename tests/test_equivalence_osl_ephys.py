"""osl_signflip must give identical results to the osl-ephys implementation.

The two code bases are maintained separately, so this test guards against them drifting
apart. It is skipped when osl-ephys (with the Z2 synchronization methods) is not installed.
"""

import numpy as np
import pytest
from conftest import make_sessions

import osl_signflip

sf = pytest.importorskip("osl_ephys.source_recon.sign_flipping")
if not hasattr(sf, "find_flips_sdp"):
    pytest.skip(
        "installed osl-ephys predates the Z2 synchronisation methods",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def covs():
    data, _ = make_sessions(n_sessions=3, n_channels=12, seed=7)
    return osl_signflip.compute_covariances(data, n_embeddings=5), 5


def test_covariance_and_template(covs):
    covs_, n_embeddings = covs
    data, _ = make_sessions(n_sessions=3, n_channels=12, seed=7)
    for x, c in zip(data, covs_):
        ref = np.cov(sf.std_data(sf.time_embed(x, n_embeddings)), rowvar=False)
        np.testing.assert_array_equal(c, ref)
    assert osl_signflip.find_template(covs_, n_embeddings) == sf.find_template_subject(
        covs_, n_embeddings
    )


@pytest.mark.parametrize(
    "method, ours, theirs, kwargs",
    [
        ("sdp", "find_flips_sdp", "find_flips_sdp", {}),
        ("spectral", "find_flips_spectral", "find_flips_spectral", {}),
        (
            "belief_propagation",
            "find_flips_belief_propagation",
            "find_flips_belief_propagation",
            {},
        ),
        (
            "sdp-first",
            "find_flips_sdp",
            "find_flips_sdp",
            {"refine_strategy": "first", "random_state": 3},
        ),
        (
            "bp-norefine",
            "find_flips_belief_propagation",
            "find_flips_belief_propagation",
            {"refine": False},
        ),
    ],
)
def test_solvers_identical(covs, method, ours, theirs, kwargs):
    covs_, E = covs
    f1, m1 = getattr(osl_signflip, ours)(covs_[0], covs_[1], E, **kwargs)
    f2, m2 = getattr(sf, theirs)(covs_[0], covs_[1], E, **kwargs)
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(m1, np.asarray(m2))


def test_random_identical(covs):
    covs_, E = covs
    f1, m1 = osl_signflip.find_flips_random(
        covs_[0], covs_[1], E, n_restarts=2, n_iter=200, max_flips=6
    )
    f2, m2 = sf.find_flips(
        covs_[0], covs_[1], E, n_restarts=2, n_iter=200, max_flips=6, use_tqdm=False
    )
    # osl_signflip applies the global sign convention to random search; osl-ephys returns the flips as found
    np.testing.assert_array_equal(f1, sf._apply_flip_convention(f2))
    np.testing.assert_array_equal(m1, np.asarray(m2))


def test_default_hyperparameters_match():
    import inspect

    from osl_ephys.source_recon.wrappers import fix_sign_ambiguity

    theirs = inspect.signature(fix_sign_ambiguity).parameters
    for fn in (
        osl_signflip.find_flips_sdp,
        osl_signflip.find_flips_spectral,
        osl_signflip.find_flips_belief_propagation,
        osl_signflip.find_flips_random,
    ):
        for name, p in inspect.signature(fn).parameters.items():
            if p.default is not inspect.Parameter.empty and name in theirs:
                assert p.default == theirs[name].default, (
                    f"{fn.__name__}: default of {name} differs"
                )


def test_random_identical_when_never_positive(covs):
    covs_, E = covs
    f1, m1 = osl_signflip.find_flips_random(
        covs_[0], -covs_[1], E, n_restarts=2, n_iter=0, max_flips=5
    )
    f2, m2 = sf.find_flips(
        covs_[0], -covs_[1], E, n_restarts=2, n_iter=0, max_flips=5, use_tqdm=False
    )
    np.testing.assert_array_equal(f1, sf._apply_flip_convention(f2))
    np.testing.assert_array_equal(m1, np.asarray(m2))


def test_flip_convention_identical():
    from osl_signflip._utils import apply_flip_convention

    rng = np.random.default_rng(0)
    cases = [rng.choice([-1.0, 1.0], size=n) for n in (5, 6, 38) for _ in range(50)]
    cases += [np.array([-1.0, 1.0, 1.0, -1.0]), np.array([0.0, -1.0, 1.0, -1.0])]
    for f in cases:
        np.testing.assert_array_equal(
            apply_flip_convention(f), sf._apply_flip_convention(f)
        )
