import numpy as np

from rps_beliefs.analysis import streak_lengths, switch_probs, switch_probs_streak

CHOICES = np.array([0, 0, 0, 2, 1, 0, 0, 0])
BOT_MOVES = np.array([1, 1, 1, 1, 0, 0, 2, 2])


def test_streak_lengths():
    outcomes, streak_len = streak_lengths(CHOICES, BOT_MOVES)
    np.testing.assert_array_equal(outcomes, [0, 0, 0, 1, 1, 2, 1, 1])
    assert streak_len == [1, 2, 3, 1, 2, 1, 1, 2]


def test_switch_probs_streak():
    probs, count = switch_probs_streak(CHOICES, BOT_MOVES)
    np.testing.assert_allclose(probs, [[0, 0, 1], [0.5, 1, np.nan], [0, np.nan, np.nan]], equal_nan=True)
    np.testing.assert_allclose(count, [[1, 1, 1], [2, 1, 0], [1, 0, 0]])


def test_switch_probs():
    choices = np.array([0, 0, 2, 2, 2, 1, 1])
    bot_moves = np.array([1, 2, 0, 2, 1, 1, 0])
    np.testing.assert_allclose(switch_probs(choices, bot_moves), [0, 1, 0])


def test_switch_probs_nan_for_missing_outcome():
    choices = np.array([0, 2, 2, 2])
    bot_moves = np.array([2, 1, 1, 0])
    np.testing.assert_allclose(switch_probs(choices, bot_moves), [np.nan, 1/3, np.nan], equal_nan=True)
