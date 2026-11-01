Rock-paper-scissors: human vs. AI beliefs

A small pilot project on what makes people switch strategy in a changing environment, and whether this depends on who they believe their opponent is.

Participants play rock-paper-scissors against an opponent with a hidden preferred gesture that changes at random times. A Bayesian observer model serves as the normative reference, and a softmax choice model is fitted to the participants' choices.

Status: simulation, model and fitting pipeline are implemented; the web-based experiment is in development.

Experimental design (planned)
Block Opponent announced as Actual opponent
A Human Pre-generated sequence (hidden preferred gesture, hazard rate h)
B AI 1 Different pre-generated sequence with identical statistics
C AI 2 Learning bot that counters the participant's own patterns

The blocks vary one factor at a time: A → B changes only the believed identity of the opponent, B → C changes only its actual policy.

Model

Opponent (blocks A and B). In each round, the opponent plays its preferred gesture with probability q and each of the other two gestures with probability (1 − q)/2. Between rounds, the preferred gesture switches to one of the other two with hazard rate h.

Bayesian observer. The belief b over the three possible preferred gestures is updated after each round in two steps:

Prediction (the preferred gesture may have switched):
$$b'_i = b_i\,(1-h) + (1-b_i)\,\tfrac{h}{2}$$
Update with the observed gesture g (Bayes' rule):
$$b_i^{\text{new}} \propto b'_i \cdot P(g \mid i), \qquad P(g \mid i) = \begin{cases} q & g = i \\ (1-q)/2 & g \neq i \end{cases}$$

In log-odds, this update behaves approximately like a leaky evidence accumulator, with a leak set by h.

Choice rule. From the predicted gesture probabilities, the expected value of each choice is EV = P(win) − P(lose). Choices follow a softmax with inverse temperature β and a stickiness term κ for repeating one's last choice:

$$P(c) \propto \exp\big(\beta\,\mathrm{EV}_c + \kappa\,[c = c_{t-1}]\big)$$

Fitting. The parameters (h, β, κ) are estimated by maximum likelihood on a grid, with q fixed to its true value. Parameter recovery on simulated players checks which parameters can be identified with the planned number of rounds.

Code

bot.py is organised in cells (# %%) that can be run one by one in VS Code:

Function Purpose
simulate_bot Generates the opponent's preferred gestures and played gestures
update_belief Bayesian belief update over the preferred gesture
choice_probs Softmax choice probabilities with β and κ
log_likelihood Log-likelihood of a sequence of choices given (h, β, κ)
fit_model Grid search for the maximum-likelihood parameters
simulate_player Simulates a player's choices against a given opponent sequence

Further cells simulate players, plot beliefs and log-odds, compare the observer with a simple leaky accumulator, compute win rates of an ideal player, and run parameter recovery.

Setup

The project uses uv:

bash
uv sync
uv run python bot.py
Preliminary simulation results
With q = 0.8 and h = 0.10, an ideal Bayesian player wins about 63 % of rounds. These values were chosen as a medium difficulty.
With 150 rounds per block, β and κ are recovered well and h moderately well.
Next steps
Build the web app with a shared database and run a pilot with a few participants
Implement the learning bot for block C
Classify each strategy switch relative to the ideal observer (too early, on time, too late, without reason)
Compare the Bayesian model with simpler alternatives (win-stay/lose-shift, Q-learning)
Author

Cecilia Diesch
