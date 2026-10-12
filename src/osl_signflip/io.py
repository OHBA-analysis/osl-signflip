"""Helpers for MNE-Python objects and files.

- Requires MNE-Python (``pip install osl-signflip[mne]``).
- Expects parcellated data: an ``mne.io.Raw`` or ``mne.Epochs`` whose parcel time series
  are stored in channels named ``parcel_X`` (or, for older files, as ``misc`` channels),
  as written by ``osl-ephys``.
- Data from other pipelines can be passed to the array-based functions in
  :mod:`osl_signflip` directly.

Adapted from ``osl_ephys.source_recon.sign_flipping``.
(MIT License, OHBA Analysis Group)
"""

# Authors: SungJun Cho <sungjun.cho@ndcn.ox.ac.uk>

import numpy as np

from ._utils import progress_range
from .covariance import _compute_covariance, _stack_covariances


def _import_mne():
    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "This function requires MNE-Python. Install it with `pip install osl-signflip[mne]`."
        ) from e
    return mne


def get_parcel_channels(inst):
    """Gets parcel channel names in an mne.io.Raw or mne.Epochs object.

    Parameters
    ----------
    inst : mne.io.Raw or mne.Epochs
        Raw or Epochs object.

    Returns
    -------
    parc_chans : list of str or str
        Parcel channel names. If no channels called 'parcel_X' are found
        in the object, then we return 'misc'.
    """
    # Parcel channels are those called 'parcel_X'
    parc_chans = [ch for ch in inst.ch_names if "parcel" in ch]
    if len(parc_chans) == 0:
        # Old parc-raw.fif didn't use the 'parcel_X' naming convention for parcel channels,
        # so we select all misc channels for backwards compatibility
        parc_chans = "misc"
    return parc_chans


def get_data(inst, picks=None):
    """Gets parcel time series from an mne.io.Raw or mne.Epochs object.

    Parameters
    ----------
    inst : mne.io.Raw or mne.Epochs
        Raw or Epochs object.
    picks : str or list of str, optional
        Channels to use. Defaults to :func:`get_parcel_channels`.

    Returns
    -------
    x : numpy.ndarray
        Parcel time series. Shape is (n_samples, n_channels). Bad segments are omitted
        for Raw objects; epochs are concatenated along time for Epochs objects.
    """
    mne = _import_mne()
    if picks is None:
        picks = get_parcel_channels(inst)
    if isinstance(inst, mne.BaseEpochs):
        x = inst.get_data(picks=picks)  # (epochs, channels, time)
        x = np.swapaxes(x, 1, 2)
        return x.reshape(-1, x.shape[-1])  # (time, channels)
    x = inst.get_data(picks=picks, reject_by_annotation="omit", verbose=False)
    return x.T  # (channels, time) -> (time, channels)


def read_data(fname):
    """Reads parcel time series from a ``*raw.fif`` or ``*epo.fif`` file.

    Parameters
    ----------
    fname : str
        Path to the parcellated data file.

    Returns
    -------
    x : numpy.ndarray
        Parcel time series. Shape is (n_samples, n_channels).
    """
    mne = _import_mne()
    fname = str(fname)
    if "raw.fif" in fname:
        # We assume this is a parc-raw.fif file created in beamform_and_parcellate
        inst = mne.io.read_raw_fif(fname, verbose=False)
    elif "epo.fif" in fname:
        # We assume this is a parc-epo.fif file created in beamform_and_parcellate
        inst = mne.read_epochs(fname, verbose=False)
    else:
        raise ValueError(f"Don't know how to load {fname}. Please pass a loader.")
    return get_data(inst)


def load_covariances(
    fnames, n_embeddings=1, standardize_data=True, loader=None, use_tqdm=False
):
    """Loads parcellated data files and returns their covariance matrices.

    Parameters
    ----------
    fnames : list of str
        Paths to parcellated data files.
    n_embeddings : int, optional
        Number of time-delay embeddings to perform.
    standardize_data : bool, optional
        Should we standardize the data?
    loader : callable, optional
        Custom function mapping a file path to a (n_samples, n_channels) array.
        Defaults to :func:`read_data`.
    use_tqdm : bool, optional
        Should we display a tqdm progress bar?

    Returns
    -------
    covs : numpy.ndarray
        Covariance matrices. Shape is (n_sessions, n_channels * n_embeddings,
        n_channels * n_embeddings).
    """
    if loader is None:
        loader = read_data
    covs = []
    for i in progress_range(
        len(fnames), desc="Calculating covariances", use_tqdm=use_tqdm
    ):
        covs.append(
            _compute_covariance(
                loader(fnames[i]), n_embeddings, standardize_data, where=f"{fnames[i]}: "
            )
        )
    return _stack_covariances(covs, [str(f) for f in fnames], n_embeddings)


def apply_flips(inst, flips, picks=None):
    """Applies flips to an mne.io.Raw or mne.Epochs object.

    Parameters
    ----------
    inst : mne.io.Raw or mne.Epochs
        Raw or Epochs object. It is not modified.
    flips : numpy.ndarray
        A (n_channels,) array of 1s and -1s, ordered as the parcel channels.
    picks : str or list of str, optional
        Channels to flip. Defaults to :func:`get_parcel_channels`.

    Returns
    -------
    inst_flipped : mne.io.Raw or mne.Epochs
        Sign-flipped copy of ``inst`` (data loaded into memory).
    """
    mne = _import_mne()
    if picks is None:
        picks = get_parcel_channels(inst)
    flips = np.asarray(flips, dtype=float)

    sflip = inst.copy()
    sflip.load_data()

    # Flip the sign of the channels
    if isinstance(inst, mne.BaseEpochs):

        def flip(data):
            return data * flips[np.newaxis, :, np.newaxis]

    else:

        def flip(data):
            return data * flips[:, np.newaxis]

    sflip.apply_function(flip, picks=picks, channel_wise=False)
    return sflip
