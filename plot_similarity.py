"""
Compute pairwise Jensen-Shannon divergence between models, averaged over all words,
then plot as a clustered heatmap similarity matrix.

JS divergence is used over KL because it is symmetric and always well-defined
(KL is undefined when a zero-probability sense in Q has positive mass in P).
JS is in [0, 1] when using log base 2.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.cluster.hierarchy import linkage, leaves_list

data = json.load(open("histograms.json"))
meanings = json.load(open("meanings.json"))

MODELS = [
    "flux", "flux_1pro", "flux_flex", "flux_max",
    "gemini", "gemini_25", "gemini_pro",
    "grok", "grok_base", "grok_pro",
    "openai", "openai_1", "openai_15", "openai_mini",
    "qwen", "qwen_1", "qwen_20",
]

MODEL_LABELS = {
    "openai":      "gpt-image-2",
    "openai_15":   "gpt-image-1.5",
    "openai_1":    "gpt-image-1",
    "openai_mini": "gpt-image-1-mini",
    "grok":        "grok-quality",
    "grok_base":   "grok-base",
    "grok_pro":    "grok-pro",
    "gemini":      "gemini-3.1-flash",
    "gemini_25":   "gemini-2.5-flash",
    "gemini_pro":  "gemini-3-pro",
    "qwen":        "Qwen-2.0-Pro",
    "qwen_20":     "Qwen-2.0",
    "qwen_1":      "Qwen-v1",
    "flux":        "FLUX.2-pro",
    "flux_max":    "FLUX.2-max",
    "flux_flex":   "FLUX.2-flex",
    "flux_1pro":   "FLUX.1.1-pro",
}


def get_all_senses(word):
    raw = meanings.get(word, [])
    if isinstance(raw, list):
        return [s["sense"] for s in raw]
    return list(raw.keys())


def model_dist(word_data, model, vocab):
    """Return a probability vector over vocab for this model/word (GPT classifier)."""
    if model not in word_data:
        return None
    counts = word_data[model]["gpt"]["_counts"]
    total = sum(counts.values())
    if total == 0:
        return None
    vec = np.array([counts.get(s, 0) for s in vocab], dtype=float)
    return vec / vec.sum()


# Accumulate per-word JS distances
n = len(MODELS)
sum_js = np.zeros((n, n))
word_count = np.zeros((n, n))  # how many words contributed to each pair

for word, word_data in data.items():
    # Union of meanings vocab + every label the classifiers actually used
    seen_labels = set()
    for m in word_data.values():
        seen_labels.update(m["gpt"]["_counts"].keys())
    vocab = get_all_senses(word)
    for lbl in ["multiple", "unclear", "other"]:
        if lbl not in vocab:
            vocab.append(lbl)
    for lbl in seen_labels:
        if lbl not in vocab:
            vocab.append(lbl)

    dists = {}
    for i, m in enumerate(MODELS):
        d = model_dist(word_data, m, vocab)
        if d is not None:
            dists[i] = d

    for i in dists:
        for j in dists:
            if i >= j:
                continue
            js = jensenshannon(dists[i], dists[j], base=2)
            sum_js[i, j] += js
            sum_js[j, i] += js
            word_count[i, j] += 1
            word_count[j, i] += 1

# Average JS distance (0 = identical, 1 = maximally different)
with np.errstate(invalid="ignore"):
    avg_js = np.where(word_count > 0, sum_js / word_count, np.nan)
np.fill_diagonal(avg_js, 0.0)

# Convert to similarity: 1 - JS distance
similarity = 1 - avg_js

# ── Hierarchical clustering to reorder rows/cols ──────────────────────────────
# Fill any NaN pairs (no shared words) with the mean JS distance
nan_mask = np.isnan(avg_js)
mean_js = np.nanmean(avg_js)
avg_js_filled = np.where(nan_mask, mean_js, avg_js)
np.fill_diagonal(avg_js_filled, 0.0)

condensed = avg_js_filled[np.triu_indices(n, k=1)]
Z = linkage(condensed, method="average")
order = leaves_list(Z)

sim_ordered = similarity[np.ix_(order, order)]
tick_labels = [MODEL_LABELS[MODELS[i]] for i in order]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))

im = ax.imshow(sim_ordered, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
plt.colorbar(im, ax=ax, label="JS similarity  (1 − JS divergence, log₂)")

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(tick_labels, fontsize=9)

# Annotate cells
for i in range(n):
    for j in range(n):
        val = sim_ordered[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if val > 0.65 else "white")

ax.set_title("Model similarity matrix\n(1 − mean JS divergence over 100 words, GPT classifier, hierarchically clustered)",
             fontsize=11)

plt.tight_layout()
plt.savefig("model_similarity.png", dpi=150, bbox_inches="tight")
print("saved model_similarity.png")

# ── Also print the top/bottom pairs ──────────────────────────────────────────
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        pairs.append((avg_js[i, j], MODELS[i], MODELS[j]))
pairs.sort()

print("\nMost similar pairs (lowest JS distance):")
for d, a, b in pairs[:5]:
    print(f"  {MODEL_LABELS[a]:20s}  {MODEL_LABELS[b]:20s}  JS={d:.4f}")

print("\nMost different pairs (highest JS distance):")
for d, a, b in pairs[-5:]:
    print(f"  {MODEL_LABELS[a]:20s}  {MODEL_LABELS[b]:20s}  JS={d:.4f}")

# ── Family cohesion analysis ──────────────────────────────────────────────────
FAMILY = {
    "flux": "FLUX", "flux_1pro": "FLUX", "flux_flex": "FLUX", "flux_max": "FLUX",
    "gemini": "Gemini", "gemini_25": "Gemini", "gemini_pro": "Gemini",
    "grok": "Grok", "grok_base": "Grok", "grok_pro": "Grok",
    "openai": "OpenAI", "openai_1": "OpenAI", "openai_15": "OpenAI", "openai_mini": "OpenAI",
    "qwen": "Qwen", "qwen_1": "Qwen", "qwen_20": "Qwen",
}
FAMILY_COLOR = {
    "FLUX": "#4C72B0", "Gemini": "#55A868",
    "Grok": "#C44E52", "OpenAI": "#DD8452", "Qwen": "#8172B3",
}
families = list(dict.fromkeys(FAMILY.values()))

from itertools import combinations

rows_cohesion = []
within_all, between_all = [], []

for fam in families:
    fam_idx   = [i for i, m in enumerate(MODELS) if FAMILY[m] == fam]
    other_idx = [i for i, m in enumerate(MODELS) if FAMILY[m] != fam]
    within  = [avg_js[i, j] for i, j in combinations(fam_idx, 2) if not np.isnan(avg_js[i, j])]
    between = [avg_js[i, j] for i in fam_idx for j in other_idx if not np.isnan(avg_js[i, j])]
    w, b = np.mean(within), np.mean(between)
    rows_cohesion.append((fam, w, b, b / w))
    within_all.extend(within)
    between_all.extend(between)

w_all = np.mean(within_all)
b_all = np.mean(between_all)
overall_ratio = b_all / w_all

print(f"\n{'─'*52}")
print("Family cohesion  (mean pairwise JS distance)")
print(f"{'Family':<10}  {'within':>8}  {'between':>8}  {'ratio':>7}")
print(f"{'─'*52}")
for fam, w, b, r in rows_cohesion:
    print(f"{fam:<10}  {w:8.4f}  {b:8.4f}  {r:6.2f}x")
print(f"{'─'*52}")
print(f"{'OVERALL':<10}  {w_all:8.4f}  {b_all:8.4f}  {overall_ratio:6.2f}x")
print(f"\nModels are {overall_ratio:.2f}x closer to members of their own family")
print("than to models from other families.")

# ── Bar chart ─────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))

x = np.arange(len(families))
width = 0.35
colors = [FAMILY_COLOR[f] for f in families]

bars_w = ax2.bar(x - width/2, [r[1] for r in rows_cohesion], width,
                 color=colors, alpha=0.9, label="within family")
bars_b = ax2.bar(x + width/2, [r[2] for r in rows_cohesion], width,
                 color=colors, alpha=0.4, label="between families")

# Overall reference lines
ax2.axhline(w_all, color="gray", linewidth=1, linestyle="--", alpha=0.6)
ax2.axhline(b_all, color="gray", linewidth=1, linestyle=":",  alpha=0.6)

# Ratio annotations
for i, (fam, w, b, r) in enumerate(rows_cohesion):
    ax2.text(i, max(w, b) + 0.008, f"{r:.2f}×", ha="center", fontsize=9, fontweight="bold",
             color=FAMILY_COLOR[fam])

ax2.set_xticks(x)
ax2.set_xticklabels(families, fontsize=11)
ax2.set_ylabel("Mean pairwise JS divergence (log₂)")
ax2.set_title(
    f"Family cohesion: within- vs between-family JS distance\n"
    f"Overall: models are {overall_ratio:.2f}× closer to their own family",
    fontsize=11,
)
ax2.legend(fontsize=9)
ax2.set_ylim(0, max(r[2] for r in rows_cohesion) * 1.18)

plt.tight_layout()
plt.savefig("family_cohesion.png", dpi=150, bbox_inches="tight")
print("\nsaved family_cohesion.png")
