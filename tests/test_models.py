import numpy as np

from rps_beliefs.models import play_vs_markov, update_belief


def test_update_belief_hand_values():
    b = np.array([1/3, 1/3, 1/3])
    b = update_belief(b, 0, 0.06, 0.7)  # rock
    b = update_belief(b, 0, 0.06, 0.7)  # rock
    assert round(b[0], 3) == 0.903
    b = update_belief(b, 1, 0.06, 0.7)  # scissors
    assert round(b[0], 3) == 0.670


def test_play_vs_markov_learns_from_round_1():
    n = 150
    rng = np.random.default_rng(1)
    _, _, trans = play_vs_markov(n, 0.1, 3, 0.5, 0.8, rng)
    assert trans.sum() == n - 1  # round 0 is not counted
