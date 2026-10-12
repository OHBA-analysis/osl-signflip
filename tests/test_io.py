import numpy as np
import pytest
from conftest import make_sessions

import osl_signflip

mne = pytest.importorskip("mne")
from osl_signflip import io


def _raw(x, names=None):
    n_channels = x.shape[1]
    names = names or [f"parcel_{i}" for i in range(n_channels)]
    info = mne.create_info(names, sfreq=250.0, ch_types="misc")
    return mne.io.RawArray(x.T, info, verbose=False)


def test_get_parcel_channels():
    x = np.zeros((10, 3))
    assert io.get_parcel_channels(_raw(x)) == ["parcel_0", "parcel_1", "parcel_2"]
    assert io.get_parcel_channels(_raw(x, ["a", "b", "c"])) == "misc"


def test_get_data_and_apply_flips_raw():
    data, _ = make_sessions(n_sessions=1, n_channels=4, n_samples=500)
    raw = _raw(data[0])
    np.testing.assert_allclose(io.get_data(raw), data[0])

    flips = np.array([1.0, -1.0, -1.0, 1.0])
    flipped = io.apply_flips(raw, flips)
    np.testing.assert_allclose(io.get_data(flipped), data[0] * flips)
    np.testing.assert_allclose(io.get_data(raw), data[0])  # original untouched


def test_get_data_and_apply_flips_epochs():
    x = np.random.default_rng(0).standard_normal((3, 4, 100))  # (epochs, channels, time)
    info = mne.create_info(
        [f"parcel_{i}" for i in range(4)], sfreq=250.0, ch_types="misc"
    )
    epochs = mne.EpochsArray(x, info, verbose=False)
    expected = np.swapaxes(x, 1, 2).reshape(-1, 4)
    np.testing.assert_allclose(io.get_data(epochs), expected)

    flips = np.array([-1.0, 1.0, 1.0, -1.0])
    np.testing.assert_allclose(
        io.get_data(io.apply_flips(epochs, flips)), expected * flips
    )


def test_load_covariances(tmp_path):
    data, _ = make_sessions(n_sessions=2, n_channels=4, n_samples=500)
    fnames = []
    for i, x in enumerate(data):
        fname = tmp_path / f"s{i}_parc-raw.fif"
        _raw(x).save(fname, fmt="double", verbose=False)
        fnames.append(str(fname))
    covs = io.load_covariances(fnames, n_embeddings=3)
    np.testing.assert_allclose(
        covs, osl_signflip.compute_covariances(data, n_embeddings=3), atol=1e-12
    )

    # Custom loader
    covs = io.load_covariances([0, 1], n_embeddings=3, loader=lambda i: data[i])
    np.testing.assert_allclose(
        covs, osl_signflip.compute_covariances(data, n_embeddings=3)
    )

    with pytest.raises(ValueError, match="loader"):
        io.read_data(tmp_path / "unknown.npy")
