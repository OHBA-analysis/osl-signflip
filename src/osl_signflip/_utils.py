"""Internal helpers shared across modules."""

# Authors: SungJun Cho <sungjun.cho@ndcn.ox.ac.uk>

import numpy as np
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm, trange


def parallel_map(func, args_list, n_jobs=1, desc=None, use_tqdm=False):
    """Applies a function to each set of arguments, optionally in parallel.

    With ``n_jobs=1``, the calls run serially in the current process. Otherwise, they
    run in separate worker processes (joblib's ``loky`` backend), each limited to one
    BLAS thread so that the workers do not oversubscribe the CPU cores. Exceptions
    raised in a worker are re-raised in the current process.

    Parameters
    ----------
    func : callable
        Function to apply.
    args_list : iterable of tuple
        Positional arguments for each call.
    n_jobs : int, optional
        Number of parallel worker processes. ``-1`` uses all CPU cores.
    desc : str, optional
        Progress bar description.
    use_tqdm : bool, optional
        Should we display a tqdm progress bar?

    Returns
    -------
    results : list
        Return values, in the same order as ``args_list``.
    """
    args_list = list(args_list)
    if n_jobs == 1:
        return [
            func(*args_list[i])
            for i in progress_range(len(args_list), desc=desc, use_tqdm=use_tqdm)
        ]
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(func)(*args) for args in args_list
        )
        if use_tqdm:
            results = tqdm(results, total=len(args_list), desc=desc)
        return list(results)


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
