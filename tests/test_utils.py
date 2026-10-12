import numpy as np
import pytest

from osl_signflip._utils import apply_flip_convention


@pytest.mark.parametrize(
    "flips, expected",
    [
        ([1, 1, -1], [1, 1, -1]),  # already flips the fewer channels
        ([-1, -1, 1], [1, 1, -1]),  # flips the majority -> negated
        ([1, -1, -1, 1], [1, -1, -1, 1]),  # tie, first channel unflipped -> kept
        ([-1, 1, 1, -1], [1, -1, -1, 1]),  # tie, first channel flipped -> negated
        ([0, -1, -1], [-1, 1, 1]),  # zeros count as +1, then the majority is negated
    ],
)
def test_apply_flip_convention(flips, expected):
    np.testing.assert_array_equal(
        apply_flip_convention(np.array(flips, dtype=float)), expected
    )


def test_apply_flip_convention_picks_one_of_each_pair():
    # f and -f always map to the same vector, so the output depends only on the solution
    rng = np.random.default_rng(0)
    for _ in range(200):
        f = rng.choice([-1.0, 1.0], size=38)
        np.testing.assert_array_equal(apply_flip_convention(f), apply_flip_convention(-f))


def test_progress_range():
    from osl_signflip._utils import progress_range

    assert list(progress_range(3)) == [0, 1, 2]
    assert list(progress_range(3, desc="test", use_tqdm=True)) == [0, 1, 2]
