"""OSL-signflip: sign-flipping for source-reconstructed M/EEG parcel time series.

Source-reconstructed parcel time courses have an arbitrary sign that can differ across
sessions. OSL-signflip aligns them to a template session by solving the underlying Z2 (sign)
synchronisation problem with semidefinite programming (default), spectral synchronisation,
loopy belief propagation, or random search.
"""

from .api import METHODS, align_signs, apply_flips, find_flips
from .covariance import compute_covariance, compute_covariances, standardize, time_embed
from .metric import apply_flips_to_covariance, covariance_matrix_correlation
from .refine import greedy_refine
from .solvers import (
    find_flips_belief_propagation,
    find_flips_random,
    find_flips_sdp,
    find_flips_spectral,
)
from .template import find_template

__version__ = "0.1.0.dev0"

__all__ = [
    "METHODS",
    "align_signs",
    "apply_flips",
    "apply_flips_to_covariance",
    "compute_covariance",
    "compute_covariances",
    "covariance_matrix_correlation",
    "find_flips",
    "find_flips_belief_propagation",
    "find_flips_random",
    "find_flips_sdp",
    "find_flips_spectral",
    "find_template",
    "greedy_refine",
    "standardize",
    "time_embed",
]
