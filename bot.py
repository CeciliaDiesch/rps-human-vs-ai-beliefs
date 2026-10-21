# %% Setup
import numpy as np
import matplotlib.pyplot as plt

# %% Functions


def simulate_bot(n, h, rng):
    """Simuliert die n Züge eines Bots im Schere-Stein-Papier-Spiel: gibt die Lieblingszüge und die tatsächlich gespielten Züge (0/1/2) zurück."""

    moves = np.zeros(n, dtype=int)  # Array, gespielten Züge des Bots pro Runde
    fav = np.zeros(n, dtype=int)  # Array, Lieblingszüge pro Runde
    fav[0] = rng.integers(0, 3)  # Zufälliger Start-Lieblingszug des Bots
    n_array_fav_true_70 = rng.random(n) < 0.7  # Array, das angibt, ob der Lieblingszug des Bots gewählt wird, mit Verteilung 70% true und 30% false Einträgen

    for t in range(1, n):
        if rng.random() < h:  # also falls eine zufällig gewählte Zahl kleiner als h ist, ändert sich der Lieblingszug
            fav[t] = (fav[t-1] + rng.integers(1, 3)) % 3
        else:
            fav[t] = fav[t-1]

    for t in range(n):
        if n_array_fav_true_70[t]:  # Einträge sind 70% true, dann wird Lieblingszug gewählt als move
            moves[t] = fav[t]
        else:
            moves[t] = (fav[t] + rng.integers(1, 3)) % 3

    return fav, moves


def update_belief(b, bot_move, h):
    """Bayes-Update der Überzeugung über die Lieblingsgeste nach einer Bot-Geste."""
    b_prime = b * (1 - h) + (1 - b) * h / 2            # Teilschritt a
    like = np.full(3, 0.15)                            # Likelihood für jede Geste initialisieren, alle 0.15
    like[bot_move] = 0.7                                  # verändert die Likelihood zu 0.7 für die tatsächlich gespielte Geste
    b_new = b_prime * like / np.sum(b_prime * like)    # Teilschritt b
    return b_new


# %% Simulate
h = 0.06   # Wahrscheinlichkeit, dass sich der Lieblingszug des Bots ändert
n = 150    # Anzahl der Runden
rng = np.random.default_rng(1)
fav, moves = simulate_bot(n, h, rng)

b = np.array([1/3, 1/3, 1/3])  # Initiale Überzeugung über die Lieblingsgeste des Bots, am Anfang alle gleich wahrscheinlich
beliefs = np.zeros((n, 3))

for t in range(n):  # in jedem Durchlauf geht, der move an der Stelle t und das belief vom vorherigen Schritt hinein, b wird aktualisiert und wird im beliefs-Array an der Stelle t gespeichert
    b = update_belief(b, moves[t], h)
    beliefs[t] = b


anzahl_aenderungen = (np.diff(fav) != 0).sum()
print(f"Anzahl der Änderungen der Lieblingszüge: {anzahl_aenderungen}")

# %% Plot
fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
ax.plot(moves, ".", label="Tatsächlich gespielte Züge")
ax.plot(fav, "-", label="Lieblingszüge")
ax.set_ylabel("Zug")
ax.set_yticks([0, 1, 2], ["Stein", "Schere", "Papier"])
ax.legend()

ax2.plot(beliefs[:, 0], "-", color="black", label="Überzeugung Stein")
ax2.plot(beliefs[:, 1], "-", color="silver", label="Überzeugung Schere")
ax2.plot(beliefs[:, 2], "-", color="green", label="Überzeugung Papier")
ax2.set_xlabel("Runde")
ax2.set_ylabel("Überzeugung")
ax2.legend()

fig.tight_layout()
plt.show()
