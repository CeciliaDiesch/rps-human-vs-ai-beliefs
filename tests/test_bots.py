import numpy as np

from rps_beliefs.bots import markov_bot_move, markov_learn, outcome, markov_table


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


def test_markov_table():
    trans = markov_table([0, 1, 1, 2], [0, 1, 2, 0])
    assert trans[0, 0, 1] == 1   # i = 1: after (rock, win) came scissors
    assert trans[1, 1, 1] == 1   # i = 2: after (scissors, loss) came scissors
    assert trans[1, 2, 2] == 1   # i = 3: after (scissors, draw) came paper
    assert trans.sum() == 3      # round 0 is never a "next" gesture, so 4 rounds give 3 counts


def test_markov_moves_from_table_match_step_by_step():
    pid = 7                                                   # any participant id
    choices = np.random.default_rng(0).integers(0, 3, 150)    # random person, behavior does not matter here

    # (a) step by step, like play_vs_markov
    trans = np.zeros((3, 3, 3))
    moves_a = []
    outcomes = []
    last_choice, last_outcome = 0, 0
    for t in range(150):
        move = markov_bot_move(trans, last_choice, last_outcome, np.random.default_rng([pid, t]))
        moves_a.append(move)
        outcomes.append(outcome(choices[t], move))
        if t > 0:
            markov_learn(trans, last_choice, last_outcome, choices[t])
        last_choice, last_outcome = choices[t], outcomes[t]

    # (b) web app: recount the table before each round from the rounds before t
    moves_b = []
    for t in range(150):
        trans_t = markov_table(choices[:t], outcomes[:t])
        if t > 0:
            last_choice, last_outcome = choices[t - 1], outcomes[t - 1]
        else:
            last_choice, last_outcome = 0, 0
        moves_b.append(markov_bot_move(trans_t, last_choice, last_outcome, np.random.default_rng([pid, t])))

    assert moves_a == moves_b
