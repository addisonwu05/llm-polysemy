"""
For each word, compute the mean pairwise JS divergence across all model pairs.
High score = models wildly disagree on which sense to pick.
Low score  = models all converge on the same sense.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from itertools import combinations

data = json.load(open("histograms.json"))
meanings = json.load(open("meanings.json"))

MODELS = [
    "flux", "flux_1pro", "flux_flex", "flux_max",
    "gemini", "gemini_25", "gemini_pro",
    "grok", "grok_base", "grok_pro",
    "openai", "openai_1", "openai_15", "openai_mini",
    "qwen", "qwen_1", "qwen_20",
]


def get_vocab(word, word_data):
    raw = meanings.get(word, [])
    base = [s["sense"] for s in raw] if isinstance(raw, list) else list(raw.keys())
    seen = set()
    for m in word_data.values():
        seen.update(m["gpt"]["_counts"].keys())
    vocab = base[:]
    for lbl in ["multiple", "unclear", "other"] + sorted(seen):
        if lbl not in vocab:
            vocab.append(lbl)
    return vocab


def model_dist(counts, vocab):
    vec = np.array([counts.get(s, 0) for s in vocab], dtype=float)
    s = vec.sum()
    return vec / s if s > 0 else None


word_scores = {}

for word, word_data in data.items():
    vocab = get_vocab(word, word_data)

    dists = []
    for m in MODELS:
        if m not in word_data:
            continue
        d = model_dist(word_data[m]["gpt"]["_counts"], vocab)
        if d is not None:
            dists.append(d)

    if len(dists) < 2:
        continue

    js_vals = [jensenshannon(a, b, base=2) for a, b in combinations(dists, 2)]
    word_scores[word] = float(np.mean(js_vals))

# Sort
ranked = sorted(word_scores.items(), key=lambda x: x[1])
most_similar = ranked[:20]    # lowest mean JS = models agree
most_different = ranked[-20:] # highest mean JS = models disagree

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, items, title, color in [
    (axes[0], most_similar,  "Most agreed-upon words\n(models converge on the same sense)",  "#4CAF50"),
    (axes[1], list(reversed(most_different)), "Most disagreed-upon words\n(models pick different senses)", "#E53935"),
]:
    words  = [w for w, _ in items]
    scores = [s for _, s in items]
    y = np.arange(len(words))
    ax.barh(y, scores, color=color, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(words, fontsize=10)
    ax.set_xlabel("Mean pairwise JS divergence (log₂)", fontsize=9)
    ax.set_xlim(0, max(word_scores.values()) * 1.1)
    ax.set_title(title, fontsize=11)
    for i, (w, s) in enumerate(items):
        ax.text(s + 0.003, i, f"{s:.3f}", va="center", fontsize=8)

plt.suptitle("Per-word inter-model disagreement\n(mean pairwise JS divergence across all 17 models, GPT classifier)",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("word_disagreement.png", dpi=150, bbox_inches="tight")
print("saved word_disagreement.png")

print("\nTop 10 most agreed:")
for w, s in most_similar[:10]:
    print(f"  {w:15s} {s:.4f}")
print("\nTop 10 most disagreed:")
for w, s in list(reversed(most_different))[:10]:
    print(f"  {w:15s} {s:.4f}")
