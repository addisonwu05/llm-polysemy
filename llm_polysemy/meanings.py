"""Load meanings.json (tolerating hand-edited trailing commas) and derive labels."""

import json
import re
from pathlib import Path

from .registry import EXTRA_LABELS

DEFAULT_MEANINGS_FILE = Path("data/meanings.json")


def load_meanings(path: Path = DEFAULT_MEANINGS_FILE) -> dict:
    raw = Path(path).read_text()
    clean = re.sub(r",(\s*[}\]])", r"\1", raw)  # strip trailing commas
    return json.loads(clean)


def sense_keys(meanings: dict, word: str) -> list[str]:
    """Just the listed senses (no multiple/unclear/other)."""
    return [s["sense"] for s in meanings[word]]


def allowed_labels(meanings: dict, word: str) -> list[str]:
    return sense_keys(meanings, word) + EXTRA_LABELS
