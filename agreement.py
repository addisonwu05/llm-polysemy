#!/usr/bin/env python3
"""
Inter-judge agreement between the GPT and Gemini annotations in
classifications.json.

For every image labelled by *both* judges we compare their chosen label
(a 'multiple' verdict counts as the single label "multiple"). Reports:
  - overall raw agreement (% of images where the two judges match)
  - Cohen's kappa (chance-corrected agreement)
  - the same broken down per image-model
  - the label pairs the judges most often disagree on

    python agreement.py
"""

import json
from collections import Counter
from pathlib import Path

CLASS_FILE = Path("classifications.json")


def label_of(verdict) -> str:
    """A verdict is a label string or {'label': 'multiple', 'senses': [...]}."""
    return verdict["label"] if isinstance(verdict, dict) else verdict


def cohens_kappa(pairs) -> float:
    """pairs: list of (gpt_label, gemini_label). Returns Cohen's kappa."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    gpt_marg = Counter(a for a, _ in pairs)
    gem_marg = Counter(b for _, b in pairs)
    labels = set(gpt_marg) | set(gem_marg)
    pe = sum((gpt_marg[l] / n) * (gem_marg[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def collect():
    """Walk classifications.json -> {model_key: [(gpt_label, gemini_label), ...]}."""
    data = json.loads(CLASS_FILE.read_text())
    by_model: dict[str, list] = {}
    for word, models in data.items():
        for mk, idxs in models.items():
            for idx, judges in idxs.items():
                if "gpt" in judges and "gemini" in judges:
                    pair = (label_of(judges["gpt"]), label_of(judges["gemini"]))
                    by_model.setdefault(mk, []).append(pair)
    return by_model


def report():
    by_model = collect()
    all_pairs = [p for pairs in by_model.values() for p in pairs]

    def line(name, pairs):
        n = len(pairs)
        same = sum(a == b for a, b in pairs)
        raw = same / n if n else float("nan")
        k = cohens_kappa(pairs)
        print(f"  {name:<16} n={n:<6} raw={raw:6.1%}  kappa={k:.3f}")

    print(f"\nInter-judge agreement (GPT vs Gemini)  —  {len(all_pairs)} dual-labelled images\n")
    print("Per image-model:")
    for mk in sorted(by_model, key=lambda m: -len(by_model[m])):
        line(mk, by_model[mk])
    print("\nOverall:")
    line("ALL", all_pairs)

    # Most common disagreements (unordered label pair).
    disagree = Counter(
        tuple(sorted((a, b))) for a, b in all_pairs if a != b
    )
    if disagree:
        print("\nTop disagreements (gpt-label / gemini-label, unordered):")
        for (a, b), c in disagree.most_common(15):
            print(f"  {c:>5}  {a}   vs  {b}")


if __name__ == "__main__":
    report()
