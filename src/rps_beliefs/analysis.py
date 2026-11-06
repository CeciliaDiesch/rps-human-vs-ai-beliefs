import numpy as np

from rps_beliefs.bots import outcome


def classify_switches(choices: np.ndarray, recs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the round and category of every switch made by the person.

    choices and recs (the ideal choices) have shape (n,), k is the window in rounds. Returns the round
    indices after each switch and their categories, both with shape (number of switches,). Categories:
    0 = too early, 1 = on time, 2 = too late, 3 = without cause, 4 = on time but wrong gesture, 5 = too late and wrong gesture.
    """
    idx_recs = np.where(np.diff(recs) != 0)[0] + 1
    idx_person = np.where(np.diff(choices) != 0)[0] + 1
    category = np.full(len(idx_person), -1)
    for s in range(len(idx_person)):
        t = idx_person[s]
        t_prev = idx_person[s-1] if s > 0 else 0
        prev = idx_recs[idx_recs <= t]
        last = prev[-1] if len(prev) > 0 else 0
        d = t - last

        if choices[t] == recs[t]:
            if d <= k:
                category[s] = 1
            else:
                category[s] = 2
        elif choices[t] in recs[t+1: t+k+1]:
            category[s] = 0
        elif last > t_prev and choices[t-1] != recs[t]:
            if d <= k:
                category[s] = 4
            else:
                category[s] = 5
        else:
            category[s] = 3
    return idx_person, category


def switch_probs(choices: np.ndarray, bot_moves: np.ndarray) -> np.ndarray:
    """Compute the person's switch probability after a win, a loss or a draw.

    choices and bot_moves have shape (n,). Returns shape (3,) in the order [win, loss, draw], with nan
    for an outcome that never occurred.
    """

    idx_person = np.where(np.diff(choices) != 0)[0]
    switch_count = np.zeros(3)
    for t in idx_person:
        if bot_moves[t] == (choices[t] + 1) % 3:
            switch_count[0] += 1
        elif choices[t] == (bot_moves[t] + 1) % 3:
            switch_count[1] += 1
        else:
            switch_count[2] += 1
    wins = np.sum(bot_moves[:-1] == (choices[:-1] + 1) % 3)
    loses = np.sum(choices[:-1] == (bot_moves[:-1] + 1) % 3)
    equal = np.sum(choices[:-1] == bot_moves[:-1])
    return switch_count / np.array([wins, loses, equal])


def streak_lengths(choices: np.ndarray, bot_moves: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return the outcome and streak length of every round.

    The streak length counts how many rounds up to and including this one had the same outcome in a row.
    choices and bot_moves have shape (n,). Returns the outcomes with shape (n,) and the lengths as a list.
    """
    outcomes = np.zeros(len(choices), dtype=int)
    for t in range(len(choices)):
        outcomes[t] = outcome(choices[t], bot_moves[t])
    streak_len = [1]
    for i in range(1, len(outcomes)):
        if outcomes[i] == outcomes[i-1]:
            streak_len.append(streak_len[-1] + 1)
        else:
            streak_len.append(1)
    return outcomes, streak_len


def switch_probs_streak(choices: np.ndarray, bot_moves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute switch probability and case count by outcome and streak length.

    choices and bot_moves have shape (n,). Returns two arrays with shape (3, 3): row = outcome, column =
    streak length 1, 2, 3+. Probabilities are nan where the case count is 0.
    """
    outcomes, streak_len = streak_lengths(choices, bot_moves)
    L = np.minimum(np.array(streak_len), 3)
    column_count_index = L - 1
    count = np.zeros((3, 3))
    switch_count = np.zeros((3, 3))
    for t in range(len(choices) - 1):
        count[outcomes[t], column_count_index[t]] += 1
        if choices[t + 1] != choices[t]:
            switch_count[outcomes[t], column_count_index[t]] += 1
    return switch_count / count, count
