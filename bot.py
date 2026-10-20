# %% Setup
import numpy as np
import matplotlib.pyplot as plt

# %% Setup


def simulate_bot(n, h, rng):
    """Simuliert die n Züge eines Bots im Schere-Stein-Papier-Spiel: speichert die Lieblingszüge und die tatsächlich gespielten Züge (0/1/2)."""
    stone = 0
    scissors = 1
    paper = 2

    # Array, das die gespielten Züge des Bots speichert
    moves = np.zeros(n, dtype=int)
    fav = np.zeros(n, dtype=int)  # Array, das die Lieblingszüge pro Runde
    fav[0] = rng.integers(0, 3)  # Zufälliger Start-Lieblingszug des Bots
    # Array, das angibt, ob der Lieblingszug des Bots mit Wahrscheinlichkeit 0.7 gewählt wird
    n_array_fav_true_70 = rng.random(n) < 0.7

    for t in range(1, n):
        if rng.random() < h:
            fav[t] = (fav[t-1] + rng.integers(1, 3)) % 3
        else:
            fav[t] = fav[t-1]

    for t in range(n):
        if n_array_fav_true_70[t]:
            moves[t] = fav[t]
        else:
            moves[t] = (fav[t] + rng.integers(1, 3)) % 3

    return fav, moves


# %% Simulation ausführen
rng = np.random.default_rng(1)
fav, moves = simulate_bot(150, 0.06, rng)
anzahl_aenderungen = (np.diff(fav) != 0).sum()
print(f"Anzahl der Änderungen der Lieblingszüge: {anzahl_aenderungen}")

# %% Plotten
fig, ax = plt.subplots()
ax.plot(moves, ".", label="Tatsächlich gespielte Züge")
ax.plot(fav, "-", label="Lieblingszüge")
ax.set_xlabel("Runde")
ax.set_ylabel("Zug")
ax.set_yticks([0, 1, 2], ["Stein", "Schere", "Papier"])
ax.legend()
plt.show()

# %%
