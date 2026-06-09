"""Hero Fig A: 2D scatter — frequency vs depictability of predicted winner,
colored by which modality won (image / text / both / split).

Usage:
    python -m llm_polysemy.analysis.t1_plot
    python -m llm_polysemy.analysis.t1_plot --out figures/t1_hero.pdf

Reads: data/t1_results.json  (produced by t1_eval.py)
       data/freq_wiki.json
       data/depictability.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


COLORS = {
    "image":  "#4C72B0",  # image modality wins
    "text":   "#DD8452",  # text modality wins
    "both":   "#55A868",  # both agree on same sense
    "split":  "#C44E52",  # both win something, but neither matches the other
}
LABELS = {
    "image": "Image wins (depictable sense)",
    "text":  "Text wins (frequent sense)",
    "both":  "Both agree",
    "split": "Disagree (neither axis wins both)",
}


def load_json(path):
    return json.loads(Path(path).read_text())


def classify_outcome(entry) -> str:
    obs_i = entry.get("obs_image")
    obs_t = entry.get("obs_text")
    pred_i = entry.get("pred_image")  # depictability winner
    pred_t = entry.get("pred_text")   # frequency winner

    if obs_i is None and obs_t is None:
        return None

    if obs_i == obs_t and obs_i is not None:
        return "both"

    img_correct = (obs_i is not None and obs_i == pred_i)
    txt_correct = (obs_t is not None and obs_t == pred_t)

    if img_correct and not txt_correct:
        return "image"
    if txt_correct and not img_correct:
        return "text"
    if img_correct and txt_correct:
        return "both"
    return "split"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures/t1_hero.pdf")
    args = p.parse_args()

    results = load_json("data/t1_results.json")
    freq = load_json("data/freq_wiki.json")
    depict = load_json("data/depictability.json")

    xs, ys, colors, labels_text, dissociation_flags = [], [], [], [], []

    for entry in results:
        word = entry["word"]
        pred_i = entry.get("pred_image")  # depictability-dominant sense
        pred_t = entry.get("pred_text")   # frequency-dominant sense
        outcome = classify_outcome(entry)
        if outcome is None:
            continue

        # x = frequency of the frequency-predicted sense
        x = freq.get(word, {}).get(pred_t or "", 0.0) if pred_t else 0.0
        # y = depictability of the depictability-predicted sense (normalized 0–1)
        raw_d = depict.get(word, {}).get(pred_i or "", {}).get("score", 3) if pred_i else 3
        y = (raw_d - 1) / 4

        xs.append(x)
        ys.append(y)
        colors.append(COLORS[outcome])
        labels_text.append(word)
        dissociation_flags.append(entry.get("dissociation", False))

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Background quadrant shading
    ax.axhspan(0.5, 1.0, xmin=0, xmax=0.5, alpha=0.04, color=COLORS["text"])
    ax.axhspan(0.5, 1.0, xmin=0.5, xmax=1.0, alpha=0.04, color=COLORS["both"])
    ax.axhspan(0.0, 0.5, xmin=0.0, xmax=0.5, alpha=0.04, color=COLORS["split"])
    ax.axhspan(0.0, 0.5, xmin=0.5, xmax=1.0, alpha=0.04, color=COLORS["image"])

    # Points
    for x, y, c, word, diss in zip(xs, ys, colors, labels_text, dissociation_flags):
        marker = "D" if diss else "o"
        size = 70 if diss else 45
        ax.scatter(x, y, c=c, s=size, marker=marker, alpha=0.85,
                   edgecolors="white", linewidths=0.5, zorder=3)

    # Labels for dissociation words
    for x, y, word, diss in zip(xs, ys, labels_text, dissociation_flags):
        if diss:
            ax.annotate(word, (x, y),
                        fontsize=7.5, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points",
                        color="#333333")

    # Diagonal guide (equal frequency = equal depictability)
    diag = np.linspace(0, 1, 100)
    ax.plot(diag, diag, lw=0.8, ls="--", color="#aaaaaa", zorder=1)

    ax.set_xlabel("Wikipedia frequency of frequency-dominant sense", fontsize=11)
    ax.set_ylabel("Depictability of depictability-dominant sense (0–1)", fontsize=11)
    ax.set_title("Depictability × Frequency Predicts Modality-Dominant Sense\n"
                 "(diamonds = dissociation words)", fontsize=12)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.axhline(0.5, color="#cccccc", lw=0.6, zorder=0)
    ax.axvline(0.5, color="#cccccc", lw=0.6, zorder=0)

    # Legend
    handles = [
        mpatches.Patch(color=COLORS[k], label=LABELS[k])
        for k in ("image", "text", "both", "split")
    ]
    handles.append(
        plt.scatter([], [], marker="D", c="#888888", s=60, label="Dissociation word")
    )
    ax.legend(handles=handles, fontsize=8.5, loc="lower right", framealpha=0.9)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
