import numpy as np

from rps_beliefs.bots import markov_bot_move, markov_learn, outcome


def update_belief(b: np.ndarray, bot_move: int, h: float, q: float) -> np.ndarray:
    """Bayesian update of the belief about the favorite gesture after one bot gesture.

    b is the current belief with shape (3,), h the assumed switch rate and q the probability of playing
    the favorite. Returns the new belief with shape (3,).
    """
    b_prime = b * (1 - h) + (1 - b) * h / 2
    like = np.full(3, (1 - q) / 2)
    like[bot_move] = q
    b_new = b_prime * like / np.sum(b_prime * like)
    return b_new


def choice_probs(b: np.ndarray, h: float, beta: float, kappa: float, last_choice: int, q: float) -> np.ndarray:
    """Compute the probabilities of choosing rock, scissors or paper from the belief b.

    Softmax over expected value with inverse temperature beta, plus stickiness kappa for last_choice.
    Returns the probabilities with shape (3,).
    """

    b_prime_bot = b * (1 - h) + (1 - b) * h / 2
    p = q * b_prime_bot + (1 - q) / 2 * (1 - b_prime_bot)
    ev = np.zeros(3)
    for g in range(3):
        ev[g] = p[(g+1) % 3] - p[(g+2) % 3]
    stick = np.zeros(3)
    stick[last_choice] = 1

    z = beta * ev + kappa * stick
    x = np.exp(z - np.max(z))
    return x/np.sum(x)


def simulate_player(n: int, bot_moves: np.ndarray, h_obs: float, kappa: float, beta: float, q: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate a person's choices against a given sequence of bot moves.

    bot_moves has shape (n,), h_obs is the switch rate the person assumes. Returns the choices with shape (n,).
    """
    b = np.array([1/3, 1/3, 1/3])
    choices = np.zeros(n, dtype=int)
    last_choice = 0
    for t in range(n):
        kappa_t = kappa if t > 0 else 0
        probs = choice_probs(b, h_obs, beta, kappa_t, last_choice, q)
        choices[t] = rng.choice(3, p=probs)
        b = update_belief(b, bot_moves[t], h_obs, q)
        last_choice = choices[t]
    return choices


def ideal_choices(bot_moves: np.ndarray, h: float, q: float) -> np.ndarray:
    """Return the ideal choice per round, the gesture with the highest expected value.

    bot_moves has shape (n,). Returns the ideal choices with shape (n,).
    """
    b = np.array([1/3, 1/3, 1/3])
    ideal = np.zeros(len(bot_moves), dtype=int)
    for t in range(len(bot_moves)):
        probs = choice_probs(b, h, 1, 0, 0, q)
        ideal[t] = np.argmax(probs)
        b = update_belief(b, bot_moves[t], h, q)
    return ideal


def play_vs_markov(n: int, h_obs: float, beta: float, kappa: float, q: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a person playing against the bot that uses last gesture and last outcome.

    Returns the choices and bot moves, each with shape (n,), and the bot's count table trans with shape
    (3, 3, 3) = [last gesture, last outcome, next gesture].
    """
    bot_moves = np.zeros(n, dtype=int)
    trans = np.zeros((3, 3, 3))
    b = np.array([1/3, 1/3, 1/3])
    choices = np.zeros(n, dtype=int)
    last_choice = 0
    last_outcome = 0
    for t in range(n):
        kappa_t = kappa if t > 0 else 0
        bot_moves[t] = markov_bot_move(trans, last_choice, last_outcome, rng)
        probs = choice_probs(b, h_obs, beta, kappa_t, last_choice, q)
        choices[t] = rng.choice(3, p=probs)
        b = update_belief(b, bot_moves[t], h_obs, q)
        if t > 0:
            markov_learn(trans, last_choice, last_outcome, choices[t])
        last_choice = choices[t]
        last_outcome = outcome(choices[t], bot_moves[t])
    return choices, bot_moves, trans
