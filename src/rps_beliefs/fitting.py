import numpy as np

from rps_beliefs.models import choice_probs, update_belief


def log_likelihood(choices: np.ndarray, bot_moves: np.ndarray, h: float, beta: float, kappa: float, q: float) -> float:
    """Compute the log-likelihood of the choices under the Bayes-softmax model.

    choices and bot_moves have shape (n,). Returns the summed log-likelihood as a single number.
    """
    b = np.array([1/3, 1/3, 1/3])
    ll = 0
    last_choice = 0
    for t in range(len(choices)):
        kappa_t = kappa if t > 0 else 0
        probs = choice_probs(b, h, beta, kappa_t, last_choice, q)
        ll += np.log(probs[choices[t]])
        b = update_belief(b, bot_moves[t], h, q)
        last_choice = choices[t]
    return ll


def fit_model(choices: np.ndarray, bot_moves: np.ndarray, q: float, h_grid: np.ndarray, beta_grid: np.ndarray, kappa_grid: np.ndarray) -> tuple[float, float, float, np.ndarray, int, int, int]:
    """Grid search: log-likelihood for all combinations of h, β, κ and the best combination.

    Returns the best h, β and κ, the log-likelihoods with shape (len(h_grid), len(beta_grid), len(kappa_grid)),
    and the grid indices i, j, k of the best combination.
    """
    ll_grid = np.zeros((len(h_grid), len(beta_grid), len(kappa_grid)))
    for i, h in enumerate(h_grid):
        for j, beta in enumerate(beta_grid):
            for k, kappa in enumerate(kappa_grid):
                ll_grid[i, j, k] = log_likelihood(choices, bot_moves, h, beta, kappa, q)

    i, j, k = np.unravel_index(np.argmax(ll_grid), ll_grid.shape)
    return h_grid[i], beta_grid[j], kappa_grid[k], ll_grid, i, j, k
