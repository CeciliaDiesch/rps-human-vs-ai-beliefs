import json
from importlib.resources import files


BLOCK_INFO = {"A": ("Mensch", "Y"), "B": ("KI 1", "X"), "C": ("KI 2", None)}
BLOCK_ORDER = {1: ["A", "B", "C"], 2: ["B", "A", "C"]}


def next_participant(n_existing: int) -> tuple[str, int]:
    """Return code and variant for the next participant.

    n_existing is the number of participants already stored. Variants alternate 1, 2, 1, ..."""
    n = n_existing + 1
    variant = 1 if n % 2 == 1 else 2
    return f"P{n:02d}", variant


def load_bot_lists() -> dict:
    """Load the bot lists bundled with the study package.

    Returns a dict with keys "X" and "Y", each holding seed, fav and moves (150 entries each)."""
    text = files("rps_beliefs").joinpath("bot_lists.json").read_text(encoding="utf-8")
    return json.loads(text)
