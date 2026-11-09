# Rock-paper-scissors: human vs. AI beliefs

A small pilot project on **what makes people switch strategy** in a changing environment, and whether this depends on **who they believe their opponent is**.

Participants play rock-paper-scissors against an opponent with a hidden favorite gesture that changes at random times. A Bayesian observer model serves as the normative reference, and a softmax choice model is fitted to the participants' choices.

**Status:** simulation, model, fitting, switch analysis and the learning bot are implemented and tested in part; the web-based experiment is the next step.

## Experimental design (planned)

| Block | Opponent announced as | Actual opponent |
|---|---|---|
| A | Human | Pre-generated sequence (hidden favorite gesture, hazard rate *h*) |
| B | AI 1 | Different pre-generated sequence with identical statistics |
| C | AI 2 | Learning bot that counters the participant's own patterns |

The blocks vary one factor at a time: A → B changes only the believed identity of the opponent, B → C changes only its actual policy.

The experiment is conducted in German.

## Model

**Opponent (blocks A and B).** In each round, the opponent plays its favorite gesture with probability *q* and each of the other two gestures with probability (1 − *q*)/2. Between rounds, the favorite gesture switches to one of the other two with hazard rate *h*.

**Learning bot (block C).** The bot counts how often the participant played each gesture after a given combination of their last gesture and last outcome (win, loss, draw). It predicts the most frequent next gesture (ties broken at random) and plays the gesture that beats it.

**Bayesian observer.** The belief *b* over the three possible favorite gestures is updated after each round in two steps:

**Step 1, prediction** (the favorite gesture may have switched):

```math
b'_i = b_i\,(1-h) + (1-b_i)\,\frac{h}{2}
```

**Step 2, update** with the observed gesture *g* (Bayes' rule):

```math
b_i^{\text{new}} \propto b'_i \cdot P(g \mid i), \qquad
P(g \mid i) =
\begin{cases}
q & \text{if } g = i \\
(1-q)/2 & \text{if } g \neq i
\end{cases}
```

In log-odds, this update behaves approximately like a leaky evidence accumulator, with a leak set by *h*.

**Choice rule.** From the predicted gesture probabilities, the expected value of each choice is EV = P(win) − P(lose). Choices follow a softmax with inverse temperature β and a stickiness term κ for repeating one's last choice:

```math
P(c) \propto \exp\big(\beta\,\mathrm{EV}_c + \kappa\,[c = c_{t-1}]\big)
```

**Fitting.** The parameters (*h*, β, κ) are estimated by maximum likelihood on a grid, with *q* fixed to its true value. Parameter recovery on simulated players checks which parameters can be identified with the planned number of rounds.

## Code

The functions live in `src/rps_beliefs/`. `explorations/explorations.py` is organized in cells (`# %%`) that can be run one by one in VS Code:

| Module | Function | Purpose |
|---|---|---|
| `bots` | `simulate_bot` | Generates the opponent's favorite gestures and played gestures |
| `bots` | `outcome` | Result of one round from the participant's view |
| `bots` | `markov_bot_move` | Learning bot's move: counters the participant's most frequent next gesture |
| `bots` | `markov_learn` | Updates the learning bot's count table with the participant's choice |
| `models` | `update_belief` | Bayesian belief update over the favorite gesture |
| `models` | `choice_probs` | Softmax choice probabilities with β and κ |
| `models` | `simulate_player` | Simulates a player's choices against a given opponent sequence |
| `models` | `ideal_choices` | Ideal choice per round: the gesture with the highest expected value |
| `models` | `play_vs_markov` | Simulates a player against the learning bot |
| `fitting` | `log_likelihood` | Log-likelihood of a sequence of choices given (*h*, β, κ) |
| `fitting` | `fit_model` | Grid search for the maximum-likelihood parameters |
| `analysis` | `classify_switches` | Assigns each switch to one of six categories relative to the ideal observer |
| `analysis` | `switch_probs` | Switch probability after a win, loss or draw |
| `analysis` | `streak_lengths` | Outcome and streak length of each round |
| `analysis` | `switch_probs_streak` | Switch probability and case count by outcome and streak length |

Further cells simulate players, plot beliefs and log-odds, compare the observer with a simple leaky accumulator, compute win rates of an ideal player, and run parameter recovery.

## Setup

The project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python explorations/explorations.py
uv run pytest
```

## Preliminary simulation results

- With *q* = 0.8 and *h* = 0.10, an ideal Bayesian player wins about 63 % of rounds. These values were chosen as a medium difficulty.
- With 150 rounds per block, β and κ are recovered well and *h* moderately well.
- The ideal player never switches after a win, and switches after about 70 % of losses and about 40 % of draws.
- Against a simulated Bayesian player, the learning bot wins about 52 % of rounds (player: about 24 %).

## Next steps

- Build the web app with a shared database
- Run a pilot with a few participants
- Compare the Bayesian model with alternatives (win-stay/lose-shift, Q-learning, a "reactive opponent" model, a Markov player), per block (planned after the pilot)

## Author

Cecilia Diesch

## License

[MIT](LICENSE)
