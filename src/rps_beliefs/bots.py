import numpy as np


def simulate_bot(n: int, h: float, rng: np.random.Generator, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Simulate n rounds of the list bot in rock-paper-scissors.

    The favourite gesture changes with probability h per round and is played with probability q.
    Returns the favourite gestures and the played gestures (0/1/2), each with shape (n,).
    """

    moves = np.zeros(n, dtype=int)
    fav = np.zeros(n, dtype=int)
    fav[0] = rng.integers(0, 3)
    n_array_fav_true_q = rng.random(n) < q

    for t in range(1, n):
        if rng.random() < h:
            fav[t] = (fav[t-1] + rng.integers(1, 3)) % 3
        else:
            fav[t] = fav[t-1]

    for t in range(n):
        if n_array_fav_true_q[t]:
            moves[t] = fav[t]
        else:
            moves[t] = (fav[t] + rng.integers(1, 3)) % 3

    return fav, moves


def outcome(choice: int, bot_move: int) -> int:
    """Return the result of one round from the person's view.

    choice and bot_move are gestures (0/1/2). Returns 0 = win, 1 = loss, 2 = draw.
    """
    if bot_move == (choice + 1) % 3:
        return 0
    elif choice == (bot_move + 1) % 3:
        return 1
    return 2


def markov_bot_move(trans: np.ndarray, last_choice: int, last_outcome: int, rng: np.random.Generator) -> int:
    """Return the Markov bot's move for the coming round.

    trans has shape (3, 3, 3) = [last gesture, last outcome, next gesture] and holds counts. The bot predicts
    the most frequent next gesture (ties broken at random) and plays its counter. Returns a gesture (0/1/2).
    """
    trans_row = trans[last_choice, last_outcome]
    bot_tip = rng.choice(np.flatnonzero(trans_row == trans_row.max()))
    return int((bot_tip + 2) % 3)


def markov_learn(trans: np.ndarray, last_choice: int, last_outcome: int, choice: int) -> None:
    """Count the person's choice in the Markov bot's table, in place.

    trans has shape (3, 3, 3) = [last gesture, last outcome, next gesture]. Round 0 has no previous gesture,
    so the caller only uses this from round 1 on.
    """
    trans[last_choice, last_outcome, choice] += 1
