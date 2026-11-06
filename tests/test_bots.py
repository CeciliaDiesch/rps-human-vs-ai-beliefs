import numpy as np

from rps_beliefs.bots import markov_bot_move, markov_learn, outcome


def test_outcome():
    assert outcome(0, 1) == 0  # rock beats scissors: win
    assert outcome(0, 2) == 1  # paper beats rock: loss
    assert outcome(0, 0) == 2  # same gesture: draw


def test_markov_bot_move_counters_most_frequent_gesture():
    trans = np.zeros((3, 3, 3))
    trans[2, 1, 0] = 2  # after (paper, loss) the person played rock twice
    rng = np.random.default_rng(1)
    assert markov_bot_move(trans, 2, 1, rng) == 2  # predicts rock, so plays paper


def test_markov_learn_counts_choice():
    trans = np.zeros((3, 3, 3))
    markov_learn(trans, 2, 1, 0)
    assert trans[2, 1, 0] == 1
    assert trans.sum() == 1
