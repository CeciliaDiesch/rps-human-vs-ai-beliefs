# %% Setup
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)

# %% Functions


def simulate_bot(n, h, rng, q):
    """Simuliert die n Züge eines Bots im Stein-Schere-Papier-Spiel: gibt die Lieblingszüge und die tatsächlich gespielten Züge (0/1/2) zurück."""

    moves = np.zeros(n, dtype=int)                  # Array, gespielten Züge des Bots pro Runde
    fav = np.zeros(n, dtype=int)                    # Array, Lieblingszüge pro Runde
    fav[0] = rng.integers(0, 3)                     # Zufälliger Start-Lieblingszug des Bots
    n_array_fav_true_q = rng.random(n) < q          # Array, das angibt, ob der Lieblingszug des Bots gewählt wird, mit Verteilung q true und 1- q false Einträgen

    for t in range(1, n):
        if rng.random() < h:                        # also falls eine zufällig gewählte Zahl kleiner als h ist, ändert sich der Lieblingszug
            fav[t] = (fav[t-1] + rng.integers(1, 3)) % 3
        else:
            fav[t] = fav[t-1]

    for t in range(n):
        if n_array_fav_true_q[t]:                  # Einträge sind zu q*100 % true, dann wird Lieblingszug gewählt als move
            moves[t] = fav[t]
        else:
            moves[t] = (fav[t] + rng.integers(1, 3)) % 3

    return fav, moves


def update_belief(b, bot_move, h, q):
    """Bayes-Update der Überzeugung über die Lieblingsgeste nach einer Bot-Geste."""
    b_prime = b * (1 - h) + (1 - b) * h / 2            # Teilschritt a
    like = np.full(3, (1 - q) / 2)                     # Likelihood für jede Geste initialisieren, alle (1 - q) / 2
    like[bot_move] = q                                 # verändert die Likelihood zu q für die tatsächlich gespielte Geste
    b_new = b_prime * like / np.sum(b_prime * like)    # Teilschritt b
    return b_new


def choice_probs(b, h, beta, kappa, last_choice, q):
    """Wahlwahrscheinlichkeiten für Stein/Schere/Papier aus der Überzeugung b."""

    b_prime_bot = b * (1 - h) + (1 - b) * h / 2         # Teilschritt a: Lieblingsgeste könnte wechseln
    p = q * b_prime_bot + (1 - q) / 2 * (1 - b_prime_bot)    # Wahrscheinlichkeit, dass der Bot die jeweilige Geste spielt
    ev = np.zeros(3)
    for g in range(3):
        ev[g] = p[(g+1) % 3] - p[(g+2) % 3]             # Erwartungswert: Gewinnchance minus Verlustchance
    stick = np.zeros(3)
    stick[last_choice] = 1                              # 1 bei der eigenen letzten Wahl, sonst 0

    z = beta * ev + kappa * stick                       # Exponent der Softmax
    x = np.exp(z - np.max(z))                           # größten Wert abziehen: gleiches Ergebnis, kein Überlauf
    return x/np.sum(x)


def log_likelihood(choices, bot_moves, h, beta, kappa, q):  # selbe Schleife wie bei der Person, nur Wk von echten Wahl aufsummieren, statt choice würfeln
    """Log-Likelihood der Wahlen unter dem Bayes-Softmax-Modell."""
    b = np.array([1/3, 1/3, 1/3])
    ll = 0
    last_choice = 0
    for t in range(len(choices)):  # vorhersagen, ln addieren, dann b aktualisieren
        kappa_t = kappa if t > 0 else 0                              # Runde 0: noch keine letzte Wahl
        probs = choice_probs(b, h, beta, kappa_t, last_choice, q)    # 1. Vorhersage
        ll += np.log(probs[choices[t]])                        # 2. ln der Wahrscheinlichkeit der echten Wahl der Person
        b = update_belief(b, bot_moves[t], h, q)               # 3. Bot-Zug sehen, lernen
        last_choice = choices[t]
    return ll


def fit_model(choices, bot_moves, q, h_grid, beta_grid, kappa_grid):
    """Gittersuche: LL für alle Kombinationen aus h, β, κ und die beste Kombination."""
    ll_grid = np.zeros((len(h_grid), len(beta_grid), len(kappa_grid)))
    for i, h in enumerate(h_grid):
        for j, beta in enumerate(beta_grid):
            for k, kappa in enumerate(kappa_grid):
                ll_grid[i, j, k] = log_likelihood(choices, bot_moves, h, beta, kappa, q)

    i, j, k = np.unravel_index(np.argmax(ll_grid), ll_grid.shape)
    return h_grid[i], beta_grid[j], kappa_grid[k], ll_grid, i, j, k


def simulate_player(n, bot_moves, h_obs, kappa, beta, q, rng):
    """Simuliert die Wahlen einer Person gegen eine gegebene Bot-Zugfolge."""
    b = np.array([1/3, 1/3, 1/3])
    choices = np.zeros(n, dtype=int)
    last_choice = 0
    for t in range(n):              # pro Runde: Vorhersage → Wahl → Bot-Zug sehen → lernen
        kappa_t = kappa if t > 0 else 0   # Runde 0: noch keine letzte Wahl
        probs = choice_probs(b, h_obs, beta, kappa_t, last_choice, q)   # Wahlwahrscheinlichkeiten aus der aktuellen Überzeugung b
        choices[t] = rng.choice(3, p=probs)                       # Person wählt gewichtet nach diesen Wahrscheinlichkeiten eine der Gesten aus, bei Gleichverteilung zufällig
        b = update_belief(b, bot_moves[t], h_obs, q)              # jetzt sieht man die gespielte Geste vom Bot; damit b aktualisieren für die nächste Runde
        last_choice = choices[t]
    return choices


def ideal_choices(bot_moves, h, q):
    """Ideale Wahl in jeder Runde: Geste mit dem höchsten Erwartungswert."""
    b = np.array([1/3, 1/3, 1/3])
    ideal = np.zeros(len(bot_moves), dtype=int)
    for t in range(len(bot_moves)):                     # vorhersagen → Bot-Zug sehen → lernen
        probs = choice_probs(b, h, 1, 0, 0, q)          # β = 1, κ = 0: Reihenfolge wie beim EV
        ideal[t] = np.argmax(probs)                     # Geste mit dem höchsten Erwartungswert
        b = update_belief(b, bot_moves[t], h, q)        # Bot-Zug sehen, Überzeugung aktualisieren
    return ideal


def classify_switches(choices, recs, k):
    """Gibt für jeden Wechsel der Person die Runde und Kategorie zurück"""
    idx_recs = np.where(np.diff(recs) != 0)[0] + 1  # array mit Rundenindizes nach allen Wechsel des idealen Players
    idx_person = np.where(np.diff(choices) != 0)[0] + 1  # array mit Rundenindizes nach allen Wechsel der Person
    category = np.full(len(idx_person), -1)
    for s in range(len(idx_person)):
        t = idx_person[s]                       # Rundenindex nach Wechsel der Person
        t_prev = idx_person[s-1] if s > 0 else 0   # Rundenindex vom letzten Wechsel der Person (Index nach Wechsel)
        prev = idx_recs[idx_recs <= t]          # alle Beobachter-Wechsel bis Runde t
        last = prev[-1] if len(prev) > 0 else 0  # Rundenindex nach letzten Beobachter-Wechsel, falls noch keiner stattfand, dann ist Rundenindex 0 der letzte Wechsel
        d = t - last

        if choices[t] == recs[t]:  # falls die neue Wahl der aktuellen idealen Wahl entspricht
            if d <= k:
                category[s] = 1       # category 1 steht für rechtzeitig
            else:
                category[s] = 2       # category 2 steht für zu spät
        elif choices[t] in recs[t+1: t+k+1]:
            category[s] = 0       # category 0 steht für zu früh
        elif last > t_prev and choices[t-1] != recs[t]:
            if d <= k:
                category[s] = 4  # category 4 steht für rechtzeitig aber falsche Geste
            else:
                category[s] = 5  # category 5 steht für zu spät aber falsche Geste
        else:
            category[s] = 3         # category 3 steht für ohne Anlass
    return idx_person, category


def switch_probs(choices, bot_moves):
    """Wechselwahrscheinlichkeit der Person nach Sieg, Niederlage oder Unentschieden"""

    idx_person = np.where(np.diff(choices) != 0)[0]  # Rundenindizes vor allen Wechseln der Person
    switch_count = np.zeros(3)                       # Anzahl Wechsel nach [Sieg, Niederlage, Unentschieden]
    for t in idx_person:
        if bot_moves[t] == (choices[t] + 1) % 3:
            switch_count[0] += 1
        elif choices[t] == (bot_moves[t] + 1) % 3:
            switch_count[1] += 1
        else:
            switch_count[2] += 1
    wins = np.sum(bot_moves[:-1] == (choices[:-1] + 1) % 3)
    loses = np.sum(choices[:-1] == (bot_moves[:-1] + 1) % 3)
    equal = np.sum(choices[:-1] == bot_moves[:-1])
    return switch_count / np.array([wins, loses, equal])


def outcome(choice, bot_move):
    """Ergebnis einer Runde für die Person: 0 = Sieg, 1 = Niederlage, 2 = Unentschieden"""
    if bot_move == (choice + 1) % 3:
        return 0
    elif choice == (bot_move + 1) % 3:
        return 1
    return 2


def streak_lengths(choices, bot_moves):
    """Serienlänge jeder Runde: wie viele Runden bis einschließlich dieser hintereinander dasselbe Ergebnis hatten."""
    outcomes = np.zeros(len(choices), dtype=int)  # outcomes = array mit einträgen wins = 0, lose = 1, equal = 2
    for t in range(len(choices)):
        outcomes[t] = outcome(choices[t], bot_moves[t])
    streak_len = [1]
    for i in range(1, len(outcomes)):
        if outcomes[i] == outcomes[i-1]:
            streak_len.append(streak_len[-1] + 1)
        else:
            streak_len.append(1)
    return outcomes, streak_len


def switch_probs_streak(choices, bot_moves):
    """Wechselwahrscheinlichkeit und Fallzahl nach Ergebnis (Zeile) und Serienlänge 1, 2, 3+ (Spalte)"""
    outcomes, streak_len = streak_lengths(choices, bot_moves)
    L = np.minimum(np.array(streak_len), 3)                # Serienlänge pro Runde, Längen ab 3 zu 3 zusammengefasst
    column_count_index = L - 1                             # Serienlänge 1, 2, 3+ dem Spaltenindex 0, 1, 2 zuordnen
    count = np.zeros((3, 3))                               # 3×3-Array: wie oft kam die Kombination vor (Zeile Ergebnis, Spalte Serienlänge):[[Sieg mit Länge 1, Sieg L2, Sieg L3+], [Niederlage L1,..], [Unentsch L1,..]]
    switch_count = np.zeros((3, 3))                        # wie oft wurde danach gewechselt
    for t in range(len(choices) - 1):                      # alle Runden außer der letzten
        count[outcomes[t], column_count_index[t]] += 1     # jede Runde zählen und in dem richtigen Feld von count eintragen
        if choices[t + 1] != choices[t]:
            switch_count[outcomes[t], column_count_index[t]] += 1
    return switch_count / count, count


def play_vs_markov(n, h_obs, beta, kappa, q, rng):
    """Spiel einer simulierten Person gegen den Markov-Bot"""
    bot_moves = np.zeros(n, dtype=int)                  # Array, gespielten Züge des Bots pro Runde
    trans = np.zeros((3, 3))                            # 3x3-Array: wie oft folgt der letzten Geste 0,1,2 die aktuelle Geste 0,1,2
    b = np.array([1/3, 1/3, 1/3])
    choices = np.zeros(n, dtype=int)
    last_choice = rng.integers(0, 3)                    # Zufälliger Start des simulierten Players
    for t in range(n):                                  # pro Runde: Vorhersage → Wahl → Bot-Zug sehen → lernen
        kappa_t = kappa if t > 0 else 0                 # Runde 0: noch keine letzte Wahl
        trans_row = trans[last_choice]                      # was die Person bisher nach last_choice gespielt hat
        bot_tip = rng.choice(np.flatnonzero(trans_row == trans_row.max()))  # häufigste Geste nach last_choice, bei Gleichstand zufällig
        bot_moves[t] = (bot_tip + 2) % 3                                    # Bot spielt die Kontergeste zum Tipp
        probs = choice_probs(b, h_obs, beta, kappa_t, last_choice, q)   # Wahlwahrscheinlichkeiten aus der aktuellen Überzeugung b
        choices[t] = rng.choice(3, p=probs)                       # Person wählt gewichtet nach diesen Wahrscheinlichkeiten
        b = update_belief(b, bot_moves[t], h_obs, q)              # jetzt sieht man die gespielte Geste vom Bot; damit b aktualisieren für die nächste Runde
        if t > 0:                                           # Runde 0: noch keine letzte Wahl, also kein Übergang
            trans[last_choice, choices[t]] += 1             # Bot lernt: Person hat nach last_choice jetzt choices[t] gespielt
        last_choice = choices[t]
    return choices, bot_moves


# %% Test play_vs_markov: Gewinnquoten
rng = np.random.default_rng(1)
player_wins = np.zeros(50)
bot_wins = np.zeros(50)
for rep in range(50):                                        # 50 Spiele
    choices, bot_moves = play_vs_markov(150, 0.1, 3, 0.5, 0.8, rng)
    player_wins[rep] = np.mean(bot_moves == (choices + 1) % 3)   # Person schlägt Bot
    bot_wins[rep] = np.mean(choices == (bot_moves + 1) % 3)      # Bot schlägt Person
print(f"{player_wins.mean()=:.3f}, {bot_wins.mean()=:.3f}")   # erwartet: ca. 0,32 und 0,48

# %% Test switch_probs_streak
choices = np.array([0, 0, 0, 2, 1, 0, 0, 0])
bot_moves = np.array([1, 1, 1, 1, 0, 0, 2, 2])
switch_probs_serie, count = switch_probs_streak(choices, bot_moves)
print(f"{switch_probs_serie=}")

# %%test streak_lengths
choices = np.array([0, 0, 0, 2, 1, 0, 0, 0])
bot_moves = np.array([1, 1, 1, 1, 0, 0, 2, 2])
outcomes, streak_len = streak_lengths(choices, bot_moves)
print(f"{streak_len=}")

# %%test switch_probs simulierter Player vs idealer Player
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


# %% Simulierte 20 Player for recoevry test
h = 0.10    # Wahrscheinlichkeit, dass sich der Lieblingszug des Bots ändert
n = 150
q = 0.8
rng = np.random.default_rng(1)
h_grid = np.linspace(0.01, 0.4, 10)      # 10 Werte von 0,01 bis 0,4
beta_grid = np.linspace(0.5, 10, 10)
kappa_grid = np.linspace(-1.5, 2, 8)
true_vals = np.zeros((20, 3))    # wahre h, β, κ pro Person
fit_vals = np.zeros((20, 3))     # geschätzte h, β, κ

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
    ax.plot(true_vals[:, p], fit_vals[:, p], "o")                 # ein Punkt pro Person
    lo = min(true_vals[:, p].min(), fit_vals[:, p].min())
    hi = max(true_vals[:, p].max(), fit_vals[:, p].max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray")              # Diagonale: geschätzt = wahr
    ax.set_xlabel(f"wahr {names[p]}")
    ax.set_ylabel(f"geschätzt {names[p]}")
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
h = 0.10    # Wahrscheinlichkeit, dass sich der Lieblingszug des Bots ändert
n = 150      # Anzahl der Runden
q = 0.8
rng = np.random.default_rng(1)
fav, bot_moves = simulate_bot(n, h, rng, q)

h_obs = 0.15
beta = 3
kappa = 0.5
choices = simulate_player(n, bot_moves, h_obs, kappa, beta, q, rng)

h_grid = np.linspace(0.01, 0.4, 15)      # 15 Werte von 0,01 bis 0,4
beta_grid = np.linspace(0.5, 10, 15)
kappa_grid = np.linspace(-1, 2, 9)
h_fit, beta_fit, kappa_fit, ll_grid, i, j, k = fit_model(choices, bot_moves, q, h_grid, beta_grid, kappa_grid)
print(f"max_ll_combination: h={h_fit}, beta={beta_fit}, kappa={kappa_fit}")

# %% Heatmap
fig, ax = plt.subplots()
im = ax.imshow(ll_grid[:, :, k], origin="lower", aspect="auto",
               extent=[beta_grid[0], beta_grid[-1], h_grid[0], h_grid[-1]], vmin=ll_grid.max() - 10)   # echte Werte an den Achsen
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
        for rep in range(50):                       # 50 Wiederholungen
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
    b = update_belief(b, moves[t], h, q)    # idealer Beobachter mit dem wahren h
    beliefs[t] = b

anzahl_aenderungen = (np.diff(fav) != 0).sum()
print(f"Anzahl der Änderungen der Lieblingszüge: {anzahl_aenderungen}")

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
# %% Leak comparison
h = 0.06    # Wahrscheinlichkeit, dass sich der Lieblingszug des Bots ändert
n = 30      # Anzahl der Runden
q = 0.7
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
        b = update_belief(b, moves_stone[t], h_obs, q)
        L[i, t] = np.log(b[0]/b[1])

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(L[0], "-", color="black", label="Log-Odds Stein/Schere h=0 ")
ax.plot(L[1], "-", color="silver", label="Log-Odds Stein/Schere h=0.06")
ax.plot(L_simple, "--", color="red", label="L_simple")

ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Runde")
ax.set_ylabel("Log-Odds Stein/Schere")
ax.legend()

fig.tight_layout()
plt.show()
