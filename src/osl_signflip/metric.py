"""The covariance correlation metric that sign flipping maximises.

Adapted from ``osl_ephys.source_recon.sign_flipping``.
(MIT License, OHBA Analysis Group)
"""

# Authors: SungJun Cho <sungjun.cho@ndcn.ox.ac.uk>

import numpy as np


def covariance_matrix_correlation(C1, C2, diag_offset=0, mode=None):
    """Calculates the Pearson correlation between covariance matrices.

    Parameters
    ----------
    C1 : numpy.ndarray
        First covariance matrix.
    C2 : numpy.ndarray
        Second covariance matrix.
    diag_offset : int, optional
        To calculate the distance we take the upper triangle.
        This argument allows us to specify an offset from the diagonal
        so we can choose not to take elements near the diagonal.
        Passing ``n_embeddings`` excludes the within-channel blocks on the diagonal.
    mode : str, optional
        Either ``"abs"``, ``"sign"``, or ``None``.

    Returns
    -------
    corr : float
        Pearson correlation between the upper triangles.
    """
    if mode == "abs":
        C1 = np.abs(C1)
        C2 = np.abs(C2)
    elif mode == "sign":
        C1 = np.sign(C1)
        C2 = np.sign(C2)

    # Get the upper triangles
    i, j = np.triu_indices(C1.shape[0], k=diag_offset)
    C1 = C1[i, j]
    C2 = C2[i, j]

    # Calculate correlation
    return np.corrcoef([C1, C2])[0, 1]


def apply_flips_to_covariance(cov, flips, n_embeddings=1):
    """Applies flips to a covariance matrix.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix to apply flips to.
        Shape must be (n_channels * n_embeddings, n_channels * n_embeddings).
    flips : numpy.ndarray
        Vector of 1s and -1s indicating whether or not to flip a channels.
        Shape must be (n_channels,).
    n_embeddings : int, optional
        Number of embeddings used when calculating the covariance.

    Returns
    -------
    cov : numpy.ndarray
        Flipped covariance matrix.
    """
    # While `flips` is a (n_channels,) array, the covariance matrix is
    # (n_channels*n_embeddings, n_channels*n_embeddings). We repeat the flips vector
    # to account for the extra channels due to time embedding.
    flips = np.repeat(flips, n_embeddings)[np.newaxis, ...]
    flips = flips.T @ flips
    return cov * flips


def check_covariances(cov, template_cov, n_embeddings):
    """Validates a session/template covariance pair.

    Parameters
    ----------
    cov : numpy.ndarray
        Session covariance.
    template_cov : numpy.ndarray
        Template covariance.
    n_embeddings : int
        Number of time-delay embeddings used to build the covariances.

    Returns
    -------
    cov, template_cov : numpy.ndarray
        The validated covariances as arrays.
    """
    cov = np.asarray(cov)
    template_cov = np.asarray(template_cov)
    if cov.shape != template_cov.shape:
        raise ValueError("cov and template_cov must have the same shape.")
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov and template_cov must both be square matrices.")
    if cov.shape[0] % n_embeddings != 0:
        raise ValueError("cov shape is not compatible with n_embeddings.")
    for name, c in (("cov", cov), ("template_cov", template_cov)):
        if not np.all(np.isfinite(c)):
            raise ValueError(
                f"{name} contains NaN or infinite values. This usually means the data contain NaNs "
                "or a constant (zero-variance) channel."
            )
    return cov, template_cov


class FlipMetricEvaluator:
    """Evaluator of the sign-flip correlation metric.

    The covariance correlation metric depends on the ``+/-1`` flip vector ``f`` only through
    two quadratic forms, ``P(f) = 0.5 * f @ M @ f`` (agreement with the template) and
    ``Q(f) = 0.5 * f @ S @ f`` (sum of the flipped entries), where ``M`` is the magnitude affinity
    and ``S`` the subject-only affinity. The exact metric is then a closed-form function of ``P``,
    ``Q`` and a few template constants, so it can be evaluated without rebuilding the full covariance.
    This reproduces ``covariance_matrix_correlation(apply_flips_to_covariance(cov, f, E), template_cov, E)``
    to machine precision, which is what lets :func:`osl_signflip.refine.greedy_refine` score every
    single-channel flip in ``O(n_channels)`` by maintaining the fields ``M @ f`` and ``S @ f``
    incrementally.

    Magnitude affinity: ``M_ij = sum_block(cov * template)``.

        This is the channel-by-channel coupling fed to the Z2 (sign) solvers: ``M[i, j]``
        measures how much channels ``i`` and ``j`` agree between subject and template, and
        the quadratic form ``flips.T @ M @ flips`` is, up to a flip-invariant normalisation,
        exactly the numerator of the covariance correlation metric. Keeping the raw block inner
        product (rather than a cosine-normalised one) preserves the magnitude information that
        weights strongly-coupled pairs above weak, noisy ones.

    Subject affinity: ``S_ij = sum_block(cov)``.

    NOTE: Both are symmetric (n_channels, n_channels) matrices with a zero diagonal.

    Parameters
    ----------
    cov : numpy.ndarray
        Subject covariance. Shape is (n_channels * n_embeddings, n_channels * n_embeddings).
    template_cov : numpy.ndarray
        Template covariance. Shape is (n_channels * n_embeddings, n_channels * n_embeddings).
    n_embeddings : int
        Number of time-delay embeddings used to build the covariances.
    """

    def __init__(self, cov, template_cov, n_embeddings):
        # Validate inputs
        cov, template_cov = check_covariances(cov, template_cov, n_embeddings)

        # Get data dimensions
        D = cov.shape[0]
        E = n_embeddings
        n_channels = D // E

        # Get the entries the metric compares
        rows, cols = np.triu_indices(D, k=E)  # strict upper triangle with offset E
        cov_vals = cov[rows, cols]
        tmpl_vals = template_cov[rows, cols]
        row_chan = rows // E  # channel each scored row belongs to
        col_chan = cols // E  # channel each scored column belongs to

        # Compute affinity matrices
        def _accumulate(values):
            out = np.zeros((n_channels, n_channels), dtype=float)
            np.add.at(out, (row_chan, col_chan), values)
            out += out.T  # the diagonal stays zero (row_chan < col_chan throughout)
            return out

        self.M = _accumulate(cov_vals * tmpl_vals)  # magnitude affinity
        self.S = _accumulate(cov_vals)  # subject affinity

        # Get scalar statistics of the scored entries
        self.N = float(len(rows))  # number of scored entries
        self.tmpl_mean = np.mean(tmpl_vals)  # mean template entries
        self.tmpl_std = np.std(tmpl_vals)  # standard deviation of template entries
        self.cov_mean_sq = float(
            (cov_vals**2).sum() / self.N
        )  # mean squared subject entries

    def correlation_from_PQ(self, P, Q):
        """Computes Pearson correlation between the flipped subject covariance and the
        template covariance (over the scored entries).

        This function gives exact correlation given the two quadratic forms ``P`` and ``Q``.
        ``P`` and ``Q`` may each be a scalar or an array (the best-improvement sweep scores
        every single-channel flip at once). A (near-)constant flipped covariance gives a zero
        denominator; we report 0 there rather than propagating ``NaN``.
        """
        mean_CF = Q / self.N
        std_CF = np.sqrt(np.maximum(self.cov_mean_sq - mean_CF**2, 0.0))
        num = P / self.N - mean_CF * self.tmpl_mean
        denom = std_CF * self.tmpl_std
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = num / denom
        return np.where(denom > 0, corr, 0.0)

    def metric(self, flips):
        """Computes the exact metric of a flip vector in ``O(n_channels**2)``.

        Parameters
        ----------
        flips : numpy.ndarray
            A (n_channels,) array of 1s and -1s.

        Returns
        -------
        metric : float
            Covariance correlation metric of the flipped covariance.
        """
        P = 0.5 * flips @ (self.M @ flips)
        Q = 0.5 * flips @ (self.S @ flips)
        return float(self.correlation_from_PQ(P, Q))
