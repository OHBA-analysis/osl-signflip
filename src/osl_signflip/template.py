"""Template session selection.

Adapted from ``osl_ephys.source_recon.sign_flipping``.
(MIT License, OHBA Analysis Group)
"""

# Authors: SungJun Cho <sungjun.cho@ndcn.ox.ac.uk>

import numpy as np

from ._utils import progress_range
from .metric import covariance_matrix_correlation


def find_template(covs, diag_offset=0, use_tqdm=False):
    """Finds a good template session to use to align dipoles.

    We select the median session after calculating the similarity between
    the (sign-invariant, absolute-valued) covariances of each session.

    Parameters
    ----------
    covs : numpy.ndarray
        Covariance of each session. Shape must be (n_sessions, n_channels, n_channels).
    diag_offset : int, optional
        Offset to apply when getting the upper triangle of the covariance matrix before
        calculating the correlation between covariances.
    use_tqdm : bool, optional
        Should we display a tqdm progress bar?

    Returns
    -------
    index : int
        Index of the template session.
    """
    # Calculate the similarity between sessions
    n_sessions = len(covs)
    metric = np.zeros([n_sessions, n_sessions])
    for i in progress_range(n_sessions, desc="Comparing sessions", use_tqdm=use_tqdm):
        for j in range(i + 1, n_sessions):
            metric[i, j] = covariance_matrix_correlation(
                covs[i], covs[j], diag_offset, mode="abs"
            )
            metric[j, i] = metric[i, j]

    # Get the median session
    metric_sum = np.sum(metric, axis=1)
    argmedian = np.argsort(metric_sum)[len(metric_sum) // 2]

    return int(argmedian)
