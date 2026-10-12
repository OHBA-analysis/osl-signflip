"""Internal helpers shared across modules."""

# Authors: SungJun Cho <sungjun.cho@ndcn.ox.ac.uk>

import numpy as np
from tqdm import trange


def progress_range(n, desc=None, use_tqdm=False):
    """Returns ``range(n)``, wrapped in a tqdm progress bar if requested.

    Parameters
    ----------
    n : int
        Number of iterations.
    desc : str, optional
        Progress bar description.
    use_tqdm : bool, optional
        Should we display a tqdm progress bar?

    Returns
    -------
    iterator : iterable
        Iterator over ``range(n)``.
    """
    if use_tqdm:
        return trange(n, desc=desc)
    return range(n)


def apply_flip_convention(flips):
    """Fixes the arbitrary global sign of a flip vector.

    The global sign is a gauge freedom where ``flips`` and ``-flips`` produce the same
    flipped covariance. Hence, a deterministic convention is adopted to make the result
    reproducible: we flip the fewer channels (i.e. make the sum of flips non-negative), and
    if the sum is exactly zero, we keep the first channel unflipped (``flips[0] = 1``).

    Parameters
    ----------
    flips : numpy.ndarray
        Vector of 1s and -1s (zeros, e.g. from ``np.sign``, are treated as +1).
        Shape is (n_channels,).

    Returns
    -------
    flips : numpy.ndarray
        Vector of 1s and -1s with a non-negative sum (and ``flips[0] = 1`` if the sum is zero).
        Shape is (n_channels,).
    """
    flips = np.asarray(flips, dtype=float).copy()
    flips[flips == 0] = 1.0  # np.sign() can emit 0; a 0 "flip" would zero the channel
    total = np.sum(flips)  # a sum of +/-1s, so the comparisons below are exact
    if total < 0 or (total == 0 and flips[0] < 0):
        flips *= -1.0
    return flips
