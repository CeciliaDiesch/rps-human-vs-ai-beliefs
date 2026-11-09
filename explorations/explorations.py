# %% Setup
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import rps_beliefs

from rps_beliefs.analysis import classify_switches, streak_lengths, switch_probs, switch_probs_streak
from rps_beliefs.bots import simulate_bot
from rps_beliefs.fitting import fit_model
from rps_beliefs.models import choice_probs, ideal_choices, play_vs_markov, simulate_player, update_belief

np.set_printoptions(precision=3, suppress=True)

# %% Test profile of 2 simulated players
h = 0.10
n = 150
q = 0.8
rng = np.random.default_rng(1)
fav, bot_moves = simulate_bot(n, h, rng, q)
choices_player_1 = simulate_player(n, bot_moves, 0.1, 0.5, 3, q, rng)
choices_player_2 = simulate_player(n, bot_moves, 0.3, -0.5, 3, q, rng)
choices_ideal = ideal_choices(bot_moves, h, q)
idx_switches_1, category_1 = classify_switches(choices_player_1, choices_ideal, 2)
category_counts_1 = np.bincount(category_1, minlength=6)
idx_switches_2, category_2 = classify_switches(choices_player_2, choices_ideal, 2)
category_counts_2 = np.bincount(category_2, minlength=6)
switch_probs_player_1 = switch_probs(choices_player_1, bot_moves)
switch_probs_player_2 = switch_probs(choices_player_2, bot_moves)
switch_probs_ideal = switch_probs(choices_ideal, bot_moves)
switch_probs_streak_player_1, count_player_1 = switch_probs_streak(choices_player_1, bot_moves)
switch_probs_streak_player_2, count_player_2 = switch_probs_streak(choices_player_2, bot_moves)
switch_probs_streak_ideal, count_ideal = switch_probs_streak(choices_ideal, bot_moves)
print(f"{idx_switches_1, category_1=}")
print(f"{category_counts_1=}")
print(f"{idx_switches_2, category_2=}")
print(f"{category_counts_2=}")
print(f"{switch_probs_player_1=}")
print(f"{switch_probs_player_2=}")
print(f"{switch_probs_ideal=}")
print(f"{switch_probs_streak_ideal=}")
print(f"{switch_probs_streak_player_1=} ")
print(f"{switch_probs_streak_player_2=} ")
print(f"{count_player_1=}")
print(f"{count_player_2=}")
print(f"Win rate player 1: {np.mean(bot_moves == (choices_player_1 + 1) % 3):.2f}")
print(f"Win rate player 2: {np.mean(bot_moves == (choices_player_2 + 1) % 3):.2f}")
print(f"Win rate ideal: {np.mean(bot_moves == (choices_ideal + 1) % 3):.2f}")

# %% Profile table
names = ["too early", "on time", "too late", "without cause", "on time, wrong", "too late, wrong"]
print(f"{'':22}{'Player 1':>12}{'Player 2':>12}{'ideal':>8}")
print(f"{'Total switches':22}{len(idx_switches_1):>12}{len(idx_switches_2):>12}{np.sum(np.diff(choices_ideal) != 0):>8}")
for i in range(6):
    print(f"{names[i]:22}{category_counts_1[i]:>12}{category_counts_2[i]:>12}{'–':>8}")
outcome_names = ["after win", "after loss", "after draw"]
for i in range(3):
    print(f"{outcome_names[i]:22}{switch_probs_player_1[i]:>12.2f}{switch_probs_player_2[i]:>12.2f}{switch_probs_ideal[i]:>8.2f}")

# %% Test play_vs_markov: win rates
rng = np.random.default_rng(1)
player_wins = np.zeros(50)
bot_wins = np.zeros(50)
for rep in range(50):                                        # 50 games
    choices, bot_moves, _ = play_vs_markov(150, 0.1, 3, 0.5, 0.8, rng)
    player_wins[rep] = np.mean(bot_moves == (choices + 1) % 3)   # person beats bot
    bot_wins[rep] = np.mean(choices == (bot_moves + 1) % 3)      # bot beats person
print(f"{player_wins.mean()=:.3f}, {bot_wins.mean()=:.3f}")   # expected: about 0.235 and 0.524

# %% Test switch_probs_streak
choices = np.array([0, 0, 0, 2, 1, 0, 0, 0])
bot_moves = np.array([1, 1, 1, 1, 0, 0, 2, 2])
# switch_probs_by_streak, count = switch_probs_streak(choices, bot_moves)
# print(f"{switch_probs_by_streak=}")
choices, bot_moves, trans = play_vs_markov(150, 0.1, 3, 0.5, 0.8, rng)
print(f"{np.sum(trans)}")

# %% Test streak_lengths
choices = np.array([0, 0, 0, 2, 1, 0, 0, 0])
bot_moves = np.array([1, 1, 1, 1, 0, 0, 2, 2])
outcomes, streak_len = streak_lengths(choices, bot_moves)
print(f"{streak_len=}")

# %% Test switch_probs: simulated player vs ideal player
h = 0.10
n = 150
q = 0.8
rng = np.random.default_rng(5)
fav, bot_moves = simulate_bot(n, h, rng, q)

h_obs = 0.10
beta = 3
kappa = 0.5
choices_simulated_player = simulate_player(n, bot_moves, h_obs, kappa, beta, q, rng)
switch_probs_player = switch_probs(choices_simulated_player, bot_moves)
switch_probs_streak_player, count_player = switch_probs_streak(choices_simulated_player, bot_moves)
choices_ideal = ideal_choices(bot_moves, h_obs, q)
switch_probs_ideal = switch_probs(choices_ideal, bot_moves)
switch_probs_streak_ideal, count_ideal = switch_probs_streak(choices_ideal, bot_moves)
print(f"{switch_probs_player=}")
print(f"{switch_probs_ideal=}")
print(f"{switch_probs_streak_ideal=}")
print(f"{switch_probs_streak_player=} ")

# %% Test switch_probs
choices = np.array([0, 0, 2, 2, 2, 1, 1])
bot_moves = np.array([1, 2, 0, 2, 1, 1, 0])
print(f"{switch_probs(choices, bot_moves)=}")   # expected: [0, 1, 0]

choices = np.array([0, 2, 2, 2])
bot_moves = np.array([2, 1, 1, 0])
print(f"{switch_probs(choices, bot_moves)=}")   # expected: [nan, 0.333, nan]

# %% 20 simulated players for the recovery test
h = 0.10    # probability that the bot's favourite gesture changes
n = 150
q = 0.8
rng = np.random.default_rng(1)
h_grid = np.linspace(0.01, 0.4, 10)      # 10 values from 0.01 to 0.4
beta_grid = np.linspace(0.5, 10, 10)
kappa_grid = np.linspace(-1.5, 2, 8)
true_vals = np.zeros((20, 3))    # true h, β, κ per person
fit_vals = np.zeros((20, 3))     # estimated h, β, κ

for player in range(20):
    fav, bot_moves = simulate_bot(n, h, rng, q)
    h_obs = rng.uniform(0.05, 0.3)
    beta = rng.uniform(1, 8)
    kappa = rng.uniform(-1, 1)
    choices = simulate_player(n, bot_moves, h_obs, kappa, beta, q, rng)
    h_fit, beta_fit, kappa_fit, ll_grid, i, j, k = fit_model(choices, bot_moves, q, h_grid, beta_grid, kappa_grid)
    true_vals[player] = [h_obs, beta, kappa]
    fit_vals[player] = [h_fit, beta_fit, kappa_fit]
    print(f"max_ll_combination for player(h_true ={h_obs}, beta_true={beta},kappa_true={kappa}): h={h_fit}, beta={beta_fit}, kappa={kappa_fit}")

# %% Recovery plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
names = ["h", "β", "κ"]
for p in range(3):
    ax = axes[p]
    ax.plot(true_vals[:, p], fit_vals[:, p], "o")                 # one point per person
    lo = min(true_vals[:, p].min(), fit_vals[:, p].min())
    hi = max(true_vals[:, p].max(), fit_vals[:, p].max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray")              # diagonal: estimated = true
    ax.set_xlabel(f"true {names[p]}")
    ax.set_ylabel(f"estimated {names[p]}")
fig.tight_layout()
plt.show()

# %% Choice test
b = np.array([0.903, 0.048, 0.048])
h = 0.06
last_choice = 1
q = 0.7
for beta in [0.5, 2, 10]:
    for kappa in [0, 1]:
        print(f"{beta=}, {kappa=}: {choice_probs(b, h, beta, kappa, last_choice, q)}")

# %% Test player
h = 0.10    # probability that the bot's favourite gesture changes
n = 150      # number of rounds
q = 0.8
rng = np.random.default_rng(1)
fav, bot_moves = simulate_bot(n, h, rng, q)

h_obs = 0.15
beta = 3
kappa = 0.5
choices = simulate_player(n, bot_moves, h_obs, kappa, beta, q, rng)

h_grid = np.linspace(0.01, 0.4, 15)      # 15 values from 0.01 to 0.4
beta_grid = np.linspace(0.5, 10, 15)
kappa_grid = np.linspace(-1, 2, 9)
h_fit, beta_fit, kappa_fit, ll_grid, i, j, k = fit_model(choices, bot_moves, q, h_grid, beta_grid, kappa_grid)
print(f"max_ll_combination: h={h_fit}, beta={beta_fit}, kappa={kappa_fit}")

# %% Heatmap
fig, ax = plt.subplots()
im = ax.imshow(ll_grid[:, :, k], origin="lower", aspect="auto",
               extent=[beta_grid[0], beta_grid[-1], h_grid[0], h_grid[-1]], vmin=ll_grid.max() - 10)   # real values on the axes
fig.colorbar(im, ax=ax, label="Log-Likelihood")
ax.set_xlabel("β")
ax.set_ylabel("h")
plt.show()

# %% Ideal player
rng = np.random.default_rng(1)
n = 100

for h in [0, 0.04, 0.06, 0.1]:
    for q in [0.6, 0.7, 0.8]:
        rates = np.zeros(50)
        for rep in range(50):                       # 50 repetitions
            fav, bot_moves = simulate_bot(n, h, rng, q)
            choices = ideal_choices(bot_moves, h, q)
            wins = np.sum(bot_moves == (choices + 1) % 3)
            rates[rep] = wins / n
        print(f"{h=}, {q=}: {rates.mean():.3f}")


# %% Beliefs plot
h = 0.10
q = 0.8
n = 150
rng = np.random.default_rng(1)
fav, moves = simulate_bot(n, h, rng, q)

b = np.array([1/3, 1/3, 1/3])
beliefs = np.zeros((n, 3))
for t in range(n):
    b = update_belief(b, moves[t], h, q)    # ideal observer with the true h
    beliefs[t] = b

n_favourite_changes = (np.diff(fav) != 0).sum()
print(f"Number of favourite gesture changes: {n_favourite_changes}")

fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
ax.plot(moves, ".", label="Played gestures")
ax.plot(fav, "-", label="Favourite gestures")
ax.set_ylabel("Gesture")
ax.set_yticks([0, 1, 2], ["Rock", "Scissors", "Paper"])
ax.legend()

ax2.plot(beliefs[:, 0], "-", color="black", label="Belief rock")
ax2.plot(beliefs[:, 1], "-", color="silver", label="Belief scissors")
ax2.plot(beliefs[:, 2], "-", color="green", label="Belief paper")
ax2.set_xlabel("Round")
ax2.set_ylabel("Belief")
ax2.legend()

fig.tight_layout()
plt.show()
# %% Leak comparison
h = 0.06    # probability that the bot's favourite gesture changes
n = 30      # number of rounds
q = 0.7
L = np.zeros((2, n))            # row 0: observer with h=0, row 1: with h=0.06
L_simple = np.zeros(n)
L_simple_0 = 0
for t in range(n):
    L_simple_0 = (1 - 0.09) * L_simple_0 + 1.54
    L_simple[t] = L_simple_0

moves_stone = np.zeros(n, dtype=int)

for i, h_obs in enumerate([0, 0.06]):
    b = np.array([1/3, 1/3, 1/3])   # reset the belief for each h_obs
    for t in range(n):              # each pass takes the move at position t and the belief of the previous step; b is updated and its log-odds are stored in L
        b = update_belief(b, moves_stone[t], h_obs, q)
        L[i, t] = np.log(b[0]/b[1])

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(L[0], "-", color="black", label="Log-odds rock/scissors h=0 ")
ax.plot(L[1], "-", color="silver", label="Log-odds rock/scissors h=0.06")
ax.plot(L_simple, "--", color="red", label="L_simple")

ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Round")
ax.set_ylabel("Log-odds rock/scissors")
ax.legend()

fig.tight_layout()
plt.show()


# %% Choose lists X and Y for the web app
n, h, q = 150, 0.10, 0.8
n_seeds = 10000
wins = np.zeros(n_seeds, dtype=int)
n_switches = np.zeros(n_seeds, dtype=int)
fav_shares = np.zeros((n_seeds, 3))
for seed in range(n_seeds):
    fav, bot_moves = simulate_bot(n, h, np.random.default_rng(seed), q)
    choices = ideal_choices(bot_moves, h, q)
    wins[seed] = np.sum(bot_moves == (choices + 1) % 3)
    n_switches[seed] = np.sum(fav[1:] != fav[:-1])
    fav_shares[seed] = np.bincount(fav, minlength=3) / n

# win rate near 0.63, each favourite about 1/3 of the rounds, 15 switches (expected: 149 * 0.10)
ok = (np.abs(wins / n - 0.63) <= 0.01) & np.all(np.abs(fav_shares - 1/3) <= 0.05, axis=1) & (n_switches == 15)
candidates = np.flatnonzero(ok)
spread = np.sum((fav_shares - 1/3) ** 2, axis=1)   # how far the favourite shares are from 1/3

# per number of wins the two most even lists, then the pair with the smaller total spread
best = None
for w in np.unique(wins[candidates]):
    group = candidates[wins[candidates] == w]
    pair = group[np.argsort(spread[group])[:2]]
    if len(pair) == 2 and (best is None or spread[pair].sum() < spread[best].sum()):
        best = pair
seed_x, seed_y = np.random.default_rng(0).permutation(best)   # X or Y by lot

for seed in candidates:
    print(f"{seed=}, wins={wins[seed]}, switches={n_switches[seed]}, shares={fav_shares[seed]}, spread={spread[seed]:.5f}")
print(f"{seed_x=}, {seed_y=}")

bot_lists = {}
for name, seed in [("X", seed_x), ("Y", seed_y)]:
    fav, bot_moves = simulate_bot(n, h, np.random.default_rng(seed), q)
    bot_lists[name] = {"seed": int(seed), "fav": fav.tolist(), "moves": bot_moves.tolist()}
with open(Path(rps_beliefs.__file__).parent / "bot_lists.json", "w") as f:
    json.dump(bot_lists, f)
