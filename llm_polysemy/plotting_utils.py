"""Shared helpers for the plot scripts — consolidates the get_senses / get_vocab /
model_dist / text_dist / JS helpers that were reimplemented across the plots, plus
modality + judge resolution so every figure can be produced for image or text.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

from . import config
from .meanings import load_meanings
from .registry import EXTRA_LABELS, models_for, sense_colors  # noqa: F401 (re-exported)

JUDGE_DISPLAY = {"gpt": "GPT", "gemini": "Gemini", "consensus": "GPT+Gemini consensus"}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(modality: str):
    """Return (histograms_dict, meanings_dict) or (None, None) if the histograms
    file for this modality doesn't exist yet (e.g. text side not built)."""
    hist = config.CLS[modality].hist_file
    if not Path(hist).exists():
        return None, None
    return json.loads(Path(hist).read_text()), load_meanings()


# ── Sense / vocab / distribution helpers ───────────────────────────────────────

def get_senses(word: str, meanings: dict) -> list[str]:
    return [s["sense"] for s in meanings.get(word, [])]


def _counts(word_data_model: dict, judge: str) -> dict:
    node = word_data_model.get(judge)
    return node["_counts"] if node else {}


def get_vocab(word: str, word_data: dict, meanings: dict, judge: str = "gpt") -> list[str]:
    """Base senses + extra labels + any label the judge(s) actually emitted."""
    seen = set()
    judges = ("gpt", "gemini") if judge == "consensus" else (judge,)
    for m in word_data.values():
        for j in judges:
            seen.update(_counts(m, j).keys())
    vocab = get_senses(word, meanings)[:]
    for lbl in EXTRA_LABELS + sorted(seen):
        if lbl not in vocab:
            vocab.append(lbl)
    return vocab


def model_dist(counts: dict, vocab: list[str]):
    """Normalized probability vector over vocab, or None if empty."""
    vec = np.array([counts.get(s, 0) for s in vocab], dtype=float)
    total = vec.sum()
    return vec / total if total > 0 else None


def dist_for(word_data_model: dict, vocab: list[str], judge: str = "gpt"):
    """Distribution for a model under a judge. judge='consensus' averages the two
    judges' normalized distributions (falling back to whichever is present)."""
    if judge == "consensus":
        ds = [d for j in ("gpt", "gemini")
              if (d := model_dist(_counts(word_data_model, j), vocab)) is not None]
        if not ds:
            return None
        m = np.mean(ds, axis=0)
        s = m.sum()
        return m / s if s > 0 else None
    return model_dist(_counts(word_data_model, judge), vocab)


def text_dist(word: str, vocab: list[str], freq: dict):
    tf = freq.get(word, {})
    vec = np.array([tf.get(s, 0.0) for s in vocab], dtype=float)
    s = vec.sum()
    return vec / s if s > 0 else None


def js(p, q) -> float:
    return jensenshannon(p, q, base=2)


def mean_pairwise_js(dists: list) -> float:
    return float(np.mean([js(a, b) for a, b in combinations(dists, 2)]))


# ── Figure output paths ────────────────────────────────────────────────────────

FIG_ROOT = Path("figures")

# Text-frequency prior files (word-level, shared across modalities). Updated by the
# freq-script consolidation / data migration steps.
PRIOR_FILES = {
    "manual": Path("data/freq_manual.json"),
    "wiki":   Path("data/freq_wiki.json"),
    "trends": Path("data/freq_trends.json"),
}


def load_prior(name: str) -> dict:
    p = PRIOR_FILES[name]
    if not Path(p).exists():
        raise SystemExit(f"prior {name!r} ({p}) not found")
    return json.loads(Path(p).read_text())


def fig_dir(modality: str, subdir: str | None = None) -> Path:
    d = FIG_ROOT / modality
    if subdir:
        d = d / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── CLI ────────────────────────────────────────────────────────────────────────

def base_parser(description: str = "") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--modality", choices=["image", "text"], default="image")
    p.add_argument("--judge", choices=["gpt", "gemini", "consensus"], default="gpt")
    return p


def require_data(modality: str):
    """Load data or print a friendly message and exit 0 if not yet available."""
    data, meanings = load_data(modality)
    if data is None:
        print(f"No histograms for modality={modality} "
              f"({config.CLS[modality].hist_file} missing). Run classify --histograms first.")
        raise SystemExit(0)
    return data, meanings
