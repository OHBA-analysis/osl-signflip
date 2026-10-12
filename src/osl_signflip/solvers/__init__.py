"""Sign-flipping solvers.

Each solver takes a session covariance, a template covariance, and the
number of time-delay embeddings, and returns ``(flips, metrics)``.
"""

from .belief_propagation import find_flips_belief_propagation
from .random import find_flips_random
from .sdp import find_flips_sdp
from .spectral import find_flips_spectral

__all__ = [
    "find_flips_belief_propagation",
    "find_flips_random",
    "find_flips_sdp",
    "find_flips_spectral",
]
