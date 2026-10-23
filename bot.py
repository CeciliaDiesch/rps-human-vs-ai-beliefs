# %% Setup
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)

# %% Functions


def simulate_bot(n, h, rng):
    """Simuliert die n Züge eines Bots im Schere-Stein-Papier-Spiel: gibt die Lieblingszüge und die tatsächlich gespielten Züge (0/1/2) zurück."""

    moves = np.zeros(n, dtype=int)                  # Array, gespielten Züge des Bots pro Runde
    fav = np.zeros(n, dtype=int)                    # Array, Lieblingszüge pro Runde
    fav[0] = rng.integers(0, 3)                     # Zufälliger Start-Lieblingszug des Bots
    n_array_fav_true_70 = rng.random(n) < 0.7       # Array, das angibt, ob der Lieblingszug des Bots gewählt wird, mit Verteilung 70% true und 30% false Einträgen

    for t in range(1, n):
        if rng.random() < h:                        # also falls eine zufällig gewählte Zahl kleiner als h ist, ändert sich der Lieblingszug
            fav[t] = (fav[t-1] + rng.integers(1, 3)) % 3
        else:
            fav[t] = fav[t-1]

    for t in range(n):
        if n_array_fav_true_70[t]:                  # Einträge sind 70% true, dann wird Lieblingszug gewählt als move
            moves[t] = fav[t]
        else:
            moves[t] = (fav[t] + rng.integers(1, 3)) % 3

    return fav, moves


def update_belief(b, bot_move, h):
    """Bayes-Update der Überzeugung über die Lieblingsgeste nach einer Bot-Geste."""
    b_prime = b * (1 - h) + (1 - b) * h / 2            # Teilschritt a
    like = np.full(3, 0.15)                            # Likelihood für jede Geste initialisieren, alle 0.15
    like[bot_move] = 0.7                               # verändert die Likelihood zu 0.7 für die tatsächlich gespielte Geste
    b_new = b_prime * like / np.sum(b_prime * like)    # Teilschritt b
    return b_new


def choice_probs(b, h, beta, kappa, last_choice):
    """Wahlwahrscheinlichkeiten für Stein/Schere/Papier aus der Überzeugung b."""

    b_prime_bot = b * (1 - h) + (1 - b) * h / 2         # Teilschritt a: Lieblingsgeste könnte wechseln
    p = 0.7 * b_prime_bot + 0.15 * (1 - b_prime_bot)    # Wahrscheinlichkeit, dass der Bot die jeweilige Geste spielt
    ev = np.zeros(3)
    for g in range(3):
        ev[g] = p[(g+1) % 3] - p[(g+2) % 3]             # Erwartungswert: Gewinnchance minus Verlustchance
    stick = np.zeros(3)
    stick[last_choice] = 1                              # 1 bei der eigenen letzten Wahl, sonst 0

    z = beta * ev + kappa * stick                       # Exponent der Softmax
    x = np.exp(z - np.max(z))                           # größten Wert abziehen: gleiches Ergebnis, kein Überlauf
    return x/np.sum(x)


def log_likelihood(choices, bot_moves, h, beta, kappa):  # selbe Schleife wie bei der Spielerin, nur Wk von echten Wahl aufsummieren, statt choice würfeln
    """Log-Likelihood der Wahlen unter dem Bayes-Softmax-Modell."""
    b = np.array([1/3, 1/3, 1/3])
    ll = 0
    last_choice = 0
    for t in range(len(choices)):  # Schritt 2: vorhersagen, ln addieren, dann b aktualisieren
        k = kappa if t > 0 else 0   # Runde 0: noch keine letzte Wahl
        probs = choice_probs(b, h, beta, k, last_choice)    # 1. Vorhersage
        ll += np.log(probs[choices[t]])                     # 2. ln der Wahrscheinlichkeit der echten Wahl der Spielerin
        b = update_belief(b, bot_moves[t], h)               # 3. Bot-Zug sehen, lernen
        last_choice = choices[t]
    return ll


# %% Choice test
b = np.array([0.903, 0.048, 0.048])
h = 0.06
last_choice = 1

for beta in [0.5, 2, 10]:
    for kappa in [0, 1]:
        print(f"{beta=}, {kappa=}: {choice_probs(b, h, beta, kappa, last_choice)}")

# %% Test player
h = 0.06    # Wahrscheinlichkeit, dass sich der Lieblingszug des Bots ändert
n = 150      # Anzahl der Runden
rng = np.random.default_rng(1)
fav, bot_moves = simulate_bot(n, h, rng)

h_obs = 0.06
beta = 3
kappa = 0.5

b = np.array([1/3, 1/3, 1/3])
choices = np.zeros(n, dtype=int)
last_choice = 0

for t in range(n):              # pro Runde: Vorhersage → Wahl → Bot-Zug sehen → lernen
    k = kappa if t > 0 else 0   # Runde 0: noch keine letzte Wahl
    probs = choice_probs(b, h_obs, beta, k, last_choice)   # Wahlwahrscheinlichkeiten aus der aktuellen Überzeugung b
    choices[t] = rng.choice(3, p=probs)                    # Spielerin wählt gewichtet nach diesen Wahrscheinlichkeiten
    b = update_belief(b, bot_moves[t], h_obs)  # jetzt sieht man die gespielte Geste vom Bot; damit b aktualisieren für die nächste Runde
    last_choice = choices[t]
print(f"Choices: {choices}")


ll = log_likelihood(choices, bot_moves, h, beta, kappa)
print(f"log_likelihood: {ll}")

ll_false_h0 = log_likelihood(choices, bot_moves, 0, beta, kappa)
print(f"log_likelihood false h=0: {ll_false_h0}")

ll_false_beta0_kappa0 = log_likelihood(choices, bot_moves, h, 0, 0)
print(f"log_likelihood false beta=0 and kappa=0; reines Raten: {ll_false_beta0_kappa0}")

# %% Leak comparison
h = 0.06    # Wahrscheinlichkeit, dass sich der Lieblingszug des Bots ändert
n = 30      # Anzahl der Runden
rng = np.random.default_rng(1)
fav, moves = simulate_bot(n, h, rng)

L = np.zeros((2, n))            # Zeile 0: Beobachter mit h=0, Zeile 1: mit h=0.06
L_simple = np.zeros(n)
L_simple_0 = 0
for t in range(n):
    L_simple_0 = (1 - 0.09) * L_simple_0 + 1.54
    L_simple[t] = L_simple_0

moves_stone = np.zeros(n, dtype=int)

for i, h_obs in enumerate([0, 0.06]):
    b = np.array([1/3, 1/3, 1/3])   # Reset belief for each h_obs
    for t in range(n):              # in jedem Durchlauf geht, der move an der Stelle t und das belief vom vorherigen Schritt hinein, b wird aktualisiert und wird im beliefs-Array an der Stelle t gespeichert, bzw. die Log-Odds werden in L gespeichert
        b = update_belief(b, moves_stone[t], h_obs)
        L[i, t] = np.log(b[0]/b[1])

anzahl_aenderungen = (np.diff(fav) != 0).sum()
print(f"Anzahl der Änderungen der Lieblingszüge: {anzahl_aenderungen}")

# %% Plot
# fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
fig, ax2 = plt.subplots(figsize=(10, 8))
# ax.plot(moves, ".", label="Tatsächlich gespielte Züge")
# ax.plot(fav, "-", label="Lieblingszüge")
# ax.set_ylabel("Zug")
# ax.set_yticks([0, 1, 2], ["Stein", "Schere", "Papier"])
# ax.legend()

# ax2.plot(beliefs[:, 0], "-", color="black", label="Überzeugung Stein")
# ax2.plot(beliefs[:, 1], "-", color="silver", label="Überzeugung Schere")
# ax2.plot(beliefs[:, 2], "-", color="green", label="Überzeugung Papier")
ax2.plot(L[0], "-", color="black", label="Log-Odds Stein/Schere h=0 ")
ax2.plot(L[1], "-", color="silver", label="Log-Odds Stein/Schere h=0.06")
ax2.plot(L_simple, "--", color="red", label="L_simple")

ax2.axhline(0, color="gray", linestyle="--")
ax2.set_xlabel("Runde")
# ax2.set_ylabel("Überzeugung")
ax2.set_ylabel("Log-Odds Stein/Schere")
ax2.legend()

fig.tight_layout()
plt.show()
