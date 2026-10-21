import numpy as np


def update_belief(b, bot_move, h):
    """Bayes-Update der Überzeugung über die Lieblingsgeste nach einer Bot-Geste."""
    b_prime = b * (1 - h) + (1 - b) * h / 2            # Teilschritt a
    like = np.full(3, 0.15)
    like[bot_move] = 0.7
    b_new = b_prime * like / np.sum(b_prime * like)              # Teilschritt b
    return b_new


b = np.array([1/3, 1/3, 1/3])
h = 0.06
for m in [0, 0, 1]:  # Bot moves Abfolge
    # in jedem Durchlauf geht das belief vom vorherigen Schritt hinein und wird aktualisiert
    b = update_belief(b, m, h)
    print(b)
