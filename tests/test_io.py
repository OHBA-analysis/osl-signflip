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


def _write_sessions(tmp_path, data, kind="raw"):
    fnames, out_fnames = [], []
    for i, x in enumerate(data):
        if kind == "raw":
            fname = tmp_path / f"s{i}_parc-raw.fif"
            _raw(x).save(fname, fmt="double", verbose=False)
            out_fnames.append(str(tmp_path / f"s{i}_sflip_parc-raw.fif"))
        else:
            n_epochs = 4
            n_times = x.shape[0] // n_epochs
            epochs = np.swapaxes(
                x[: n_epochs * n_times].reshape(n_epochs, n_times, -1), 1, 2
            )
            info = mne.create_info(
                [f"parcel_{c}" for c in range(x.shape[1])], sfreq=250.0, ch_types="misc"
            )
            fname = tmp_path / f"s{i}_parc-epo.fif"
            mne.EpochsArray(epochs, info, verbose=False).save(
                fname, fmt="double", verbose=False
            )
            out_fnames.append(str(tmp_path / f"s{i}_sflip_parc-epo.fif"))
        fnames.append(str(fname))
    return fnames, out_fnames


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_sign_flip_files_matches_align_signs(tmp_path, n_jobs):
    data, _ = make_sessions(n_sessions=4, n_channels=6, n_samples=2000)
    fnames, out_fnames = _write_sessions(tmp_path, data)
    flips, metrics, template = io.sign_flip_files(fnames, out_fnames, 3, n_jobs=n_jobs)

    ref_flips, ref_metrics, ref_template = osl_signflip.align_signs(data, 3)
    assert template == ref_template
    np.testing.assert_array_equal(flips, ref_flips)
    np.testing.assert_allclose(metrics, ref_metrics, rtol=0, atol=1e-12)

    # The saved files hold the sign-flipped data (the template's data unchanged)
    for x, f, out in zip(data, flips, out_fnames):
        np.testing.assert_allclose(io.read_data(out), x * f, rtol=1e-6, atol=0)
    np.testing.assert_array_equal(flips[template], 1)


def test_sign_flip_files_template_options(tmp_path):
    data, _ = make_sessions(n_sessions=4, n_channels=6, n_samples=2000)
    fnames, out_fnames = _write_sessions(tmp_path, data)

    # Index and path of a session in fnames give the same result
    by_index = io.sign_flip_files(fnames, out_fnames, 3, template=2)
    by_path = io.sign_flip_files(fnames, out_fnames, 3, template=fnames[2])
    assert by_index[2] == by_path[2] == 2
    np.testing.assert_array_equal(by_index[0], by_path[0])

    # A template outside fnames, given as a path or as a covariance
    outside = io.sign_flip_files(fnames[:3], out_fnames[:3], 3, template=fnames[3])
    cov = io.load_covariances([fnames[3]], 3)[0]
    by_cov = io.sign_flip_files(fnames[:3], out_fnames[:3], 3, template_cov=cov)
    assert outside[2] is None and by_cov[2] is None
    assert not np.isnan(outside[1]).any()
    np.testing.assert_array_equal(outside[0], by_cov[0])
    ref = osl_signflip.align_signs(data[:3], 3, template=data[3])
    np.testing.assert_array_equal(outside[0], ref[0])


def test_sign_flip_files_epochs(tmp_path):
    data, _ = make_sessions(n_sessions=3, n_channels=6, n_samples=2000)
    fnames, out_fnames = _write_sessions(tmp_path, data, kind="epochs")
    flips, _, _ = io.sign_flip_files(fnames, out_fnames, 3, n_jobs=2)
    for fname, out, f in zip(fnames, out_fnames, flips):
        np.testing.assert_allclose(io.read_data(out), io.read_data(fname) * f, rtol=1e-6)


def test_load_covariances_parallel(tmp_path):
    data, _ = make_sessions(n_sessions=3, n_channels=6, n_samples=2000)
    fnames, _ = _write_sessions(tmp_path, data)
    np.testing.assert_allclose(
        io.load_covariances(fnames, 3, n_jobs=2),
        io.load_covariances(fnames, 3, n_jobs=1),
        rtol=0,
        atol=1e-12,
    )


def test_sign_flip_files_rejects_bad_arguments(tmp_path):
    data, _ = make_sessions(n_sessions=3, n_channels=6, n_samples=2000)
    fnames, out_fnames = _write_sessions(tmp_path, data)
    with pytest.raises(ValueError, match="same length"):
        io.sign_flip_files(fnames, out_fnames[:2], 3)
    with pytest.raises(ValueError, match="overwrite an input file"):
        io.sign_flip_files(fnames, [out_fnames[0], fnames[1], out_fnames[2]], 3)
    cov = io.load_covariances([fnames[0]], 3)[0]
    with pytest.raises(ValueError, match="either template or template_cov"):
        io.sign_flip_files(fnames, out_fnames, 3, template=0, template_cov=cov)
    with pytest.raises(ValueError, match="template must be an index"):
        io.sign_flip_files(fnames, out_fnames, 3, template=5)
