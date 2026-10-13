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

import os

import numpy as np

from ._utils import parallel_map
from .api import _check_method, _stack_metrics, find_flips
from .covariance import _compute_covariance, _stack_covariances
from .template import find_template


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


def read_inst(fname):
    """Reads a ``*raw.fif`` or ``*epo.fif`` file as an mne.io.Raw or mne.Epochs object.

    Parameters
    ----------
    fname : str
        Path to the parcellated data file.

    Returns
    -------
    inst : mne.io.Raw or mne.Epochs
        Raw or Epochs object.
    """
    mne = _import_mne()
    fname = str(fname)
    if "raw.fif" in fname:
        # We assume this is a parc-raw.fif file created in beamform_and_parcellate
        return mne.io.read_raw_fif(fname, verbose=False)
    if "epo.fif" in fname:
        # We assume this is a parc-epo.fif file created in beamform_and_parcellate
        return mne.read_epochs(fname, verbose=False)
    raise ValueError(f"Don't know how to load {fname}. Please pass a loader.")


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
    return get_data(read_inst(fname))


def _load_covariance(fname, n_embeddings, standardize_data, loader):
    """Loads one file and computes its covariance (a picklable unit of parallel work)."""
    return _compute_covariance(
        loader(fname), n_embeddings, standardize_data, where=f"{fname}: "
    )


def load_covariances(
    fnames, n_embeddings=1, standardize_data=True, loader=None, n_jobs=1, use_tqdm=True
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
    n_jobs : int, optional
        Number of parallel worker processes. ``-1`` uses all CPU cores.
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
    covs = parallel_map(
        _load_covariance,
        [(f, n_embeddings, standardize_data, loader) for f in fnames],
        n_jobs=n_jobs,
        desc="Calculating covariances",
        use_tqdm=use_tqdm,
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

    # Flip the sign of the channels
    if isinstance(inst, mne.BaseEpochs):

        def flip(data):
            return data * flips[np.newaxis, :, np.newaxis]

    else:

        def flip(data):
            return data * flips[:, np.newaxis]

    with mne.utils.use_log_level(False):  # silences MNE's per-file messages
        sflip = inst.copy()
        sflip.load_data()
        sflip.apply_function(flip, picks=picks, channel_wise=False)
    return sflip


def _sign_flip_file(
    fname,
    out_fname,
    cov,
    template_cov,
    n_embeddings,
    standardize_data,
    method,
    kwargs,
    is_template,
):
    """Finds, applies, and saves the sign flips of one file (a unit of parallel work)."""
    inst = read_inst(fname)
    x = get_data(inst)
    if is_template:
        # The template is not flipped against itself
        flips, metrics = np.ones(x.shape[1]), None
    else:
        if cov is None:
            cov = _compute_covariance(
                x, n_embeddings, standardize_data, where=f"{fname}: ", warn=False
            )
        flips, metrics = find_flips(
            cov, template_cov, n_embeddings, method=method, **kwargs
        )
    apply_flips(inst, flips).save(out_fname, overwrite=True, verbose=False)
    return flips, metrics


def sign_flip_files(
    fnames,
    out_fnames,
    n_embeddings,
    method="sdp",
    template=None,
    template_cov=None,
    standardize_data=True,
    n_jobs=1,
    use_tqdm=True,
    **kwargs,
):
    """Sign-flips parcellated data files and saves the sign-flipped data.

    This is the file-based counterpart of :func:`osl_signflip.align_signs`. For each
    session, it loads the data file, finds the flips that best align the session to a
    template, and saves the sign-flipped data to a new file. In parallel mode, each worker
    process loads and saves one file at a time, so only a few sessions are held in memory.

    Parameters
    ----------
    fnames : list of str
        Paths to the parcellated data files (``*raw.fif`` or ``*epo.fif``), one per session.
    out_fnames : list of str
        Paths to save the sign-flipped data to, one per session. Existing files are
        overwritten. They must differ from ``fnames``.
    n_embeddings : int
        Number of time-delay embeddings. Must be odd.
    method : str, optional
        Sign-flipping method. See :func:`osl_signflip.find_flips`.
    template : int or str, optional
        The template to align to. Either the index of a session in ``fnames``, or the path
        to a template data file (which need not be in ``fnames``). If ``None`` (and
        ``template_cov`` is not given), the template is selected from ``fnames`` with
        :func:`osl_signflip.find_template`.
    template_cov : numpy.ndarray, optional
        Precomputed covariance of a template session, as an alternative to ``template``.
        Shape must be (n_channels * n_embeddings, n_channels * n_embeddings).
    standardize_data : bool, optional
        Should we standardize the embedded data before computing the covariances?
    n_jobs : int, optional
        Number of parallel worker processes. ``-1`` uses all CPU cores.
    use_tqdm : bool, optional
        Should we display a tqdm progress bar?
    **kwargs
        Hyperparameters of the chosen method. See :func:`osl_signflip.find_flips`.

    Returns
    -------
    flips : numpy.ndarray
        Flips for each session. Shape is (n_sessions, n_channels). If the template is a
        session in ``fnames``, its flips are all 1s (its data are saved unchanged).
    metrics : numpy.ndarray
        Evaluation metric history for each session. Shape is (n_sessions, n_metrics). If the
        template is a session in ``fnames``, its row is ``NaN``.
    template : int or None
        Index of the template session in ``fnames``, or ``None`` if the template is not
        one of the sessions in ``fnames``.
    """
    _import_mne()
    _check_method(method)
    if method == "random":
        # Show one overall progress bar, not one per session from each worker
        kwargs.setdefault("use_tqdm", False)
    fnames = [str(f) for f in fnames]
    out_fnames = [str(f) for f in out_fnames]
    if len(fnames) != len(out_fnames):
        raise ValueError(
            f"fnames and out_fnames must have the same length, "
            f"got {len(fnames)} and {len(out_fnames)}."
        )
    real = [os.path.realpath(f) for f in fnames]
    for o in out_fnames:
        if os.path.realpath(o) in real:
            raise ValueError(f"Output file {o} would overwrite an input file.")
    if template is not None and template_cov is not None:
        raise ValueError("Pass either template or template_cov, not both.")

    # Find the template
    covs = [None] * len(fnames)
    template_index = None
    if template_cov is None:
        if template is None:
            # Compute all covariances once (reused below) and select the template
            all_covs = load_covariances(
                fnames, n_embeddings, standardize_data, n_jobs=n_jobs, use_tqdm=use_tqdm
            )
            template_index = find_template(
                all_covs, diag_offset=n_embeddings, use_tqdm=use_tqdm
            )
            covs = list(all_covs)
        elif isinstance(template, (str, os.PathLike)):
            if os.path.realpath(template) in real:
                template_index = real.index(os.path.realpath(template))
            else:
                template_cov = load_covariances(
                    [template], n_embeddings, standardize_data, use_tqdm=False
                )[0]
        else:
            template_index = int(template)
            if not 0 <= template_index < len(fnames):
                raise ValueError(
                    f"template must be an index in [0, {len(fnames)}), got {template}."
                )
    if template_index is not None and template_cov is None:
        template_cov = (
            covs[template_index]
            if covs[template_index] is not None
            else load_covariances(
                [fnames[template_index]], n_embeddings, standardize_data, use_tqdm=False
            )[0]
        )

    # Find, apply, and save the sign flips of each session
    results = parallel_map(
        _sign_flip_file,
        [
            (
                fnames[i],
                out_fnames[i],
                covs[i],
                template_cov,
                n_embeddings,
                standardize_data,
                method,
                kwargs,
                i == template_index,
            )
            for i in range(len(fnames))
        ],
        n_jobs=n_jobs,
        desc="Sign flipping",
        use_tqdm=use_tqdm,
    )
    flips = np.array([f for f, _ in results])
    metrics = _stack_metrics([m for _, m in results])
    return flips, metrics, template_index
