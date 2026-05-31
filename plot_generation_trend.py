"""
Does text-prior alignment change across model generations within each family?

For each model we already have mean JS(text_freq || model) from plot_vs_textfreq.py.
Here we re-compute that, assign a generation index to each model, and plot
JS divergence vs. generation — one line per family — with a linear-regression
trendline and Pearson r.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr

data      = json.load(open("histograms.json"))
meanings  = json.load(open("meanings.json"))
text_freq = json.load(open("text_frequencies.json"))

# Generation order within each family (oldest → newest)
# openai_mini treated as a side-branch (parallel to openai_1), gen=1.5
FAMILIES = {
    "FLUX": [
        ("flux_1pro",  1, "FLUX.1.1-pro"),
        ("flux_flex",  2, "FLUX.2-flex"),
        ("flux",       3, "FLUX.2-pro"),
        ("flux_max",   4, "FLUX.2-max"),
    ],
    "Gemini": [
        ("gemini_25",  1, "2.5-flash"),
        ("gemini",     2, "3.1-flash"),
        ("gemini_pro", 3, "3-pro"),
    ],
    "Grok": [
        ("grok_base",  1, "base"),
        ("grok_pro",   2, "pro"),
        ("grok",       3, "quality"),
    ],
    "OpenAI": [
        ("openai_1",    1,   "gpt-image-1"),
        ("openai_mini", 1.5, "gpt-image-1-mini"),
        ("openai_15",   2,   "gpt-image-1.5"),
        ("openai",      3,   "gpt-image-2"),
    ],
    "Qwen": [
        ("qwen_1",  1, "Qwen-v1"),
        ("qwen_20", 2, "Qwen-2.0"),
        ("qwen",    3, "Qwen-2.0-Pro"),
    ],
}

FAMILY_COLOR = {
    "FLUX": "#4C72B0", "Gemini": "#55A868",
    "Grok": "#C44E52", "OpenAI": "#DD8452", "Qwen": "#8172B3",
}


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


def text_dist(word, vocab):
    tf = text_freq.get(word, {})
    vec = np.array([tf.get(s, 0.0) for s in vocab], dtype=float)
    s = vec.sum()
    return vec / s if s > 0 else None


# Compute per-model mean JS from text prior
model_js = {}
for family_models in FAMILIES.values():
    for model, gen, label in family_models:
        js_vals = []
        for word, word_data in data.items():
            if model not in word_data:
                continue
            vocab = get_vocab(word, word_data)
            tf = text_dist(word, vocab)
            if tf is None:
                continue
            d = model_dist(word_data[model]["gpt"]["_counts"], vocab)
            if d is None:
                continue
            js = jensenshannon(tf, d, base=2)
            if not np.isnan(js):
                js_vals.append(js)
        if js_vals:
            model_js[model] = np.mean(js_vals)


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(FAMILIES), figsize=(18, 5), sharey=True)
fig.suptitle("Does text-prior alignment change across model generations?\n"
             "(lower JS = closer to text-frequency prior; lines connect consecutive versions)",
             fontsize=11)

all_gens, all_js = [], []

for ax, (family, models) in zip(axes, FAMILIES.items()):
    color = FAMILY_COLOR[family]
    gens    = [gen   for m, gen, lbl in models if m in model_js]
    js_vals = [model_js[m] for m, gen, lbl in models if m in model_js]
    labels  = [lbl for m, gen, lbl in models if m in model_js]

    ax.plot(gens, js_vals, color=color, linewidth=2, marker="o",
            markersize=9, zorder=3)
    for g, j, lbl in zip(gens, js_vals, labels):
        ax.text(g, j + 0.004, lbl, ha="center", va="bottom", fontsize=8, color=color)

    # Linear regression trendline (skip if <3 distinct generation values)
    unique_gens = list(set(gens))
    if len(unique_gens) >= 3:
        coeffs = np.polyfit(gens, js_vals, 1)
        xs = np.linspace(min(gens) - 0.2, max(gens) + 0.2, 50)
        ax.plot(xs, np.polyval(coeffs, xs), color=color, linewidth=1,
                linestyle="--", alpha=0.5)
        r, p = pearsonr(gens, js_vals)
        ax.text(0.97, 0.97, f"r = {r:+.2f}\n(p={p:.2f})",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=color)

    all_gens.extend(gens); all_js.extend(js_vals)

    ax.set_title(family, fontsize=11, fontweight="bold", color=color)
    ax.set_xlabel("Generation (within family)", fontsize=9)
    ax.set_xticks(sorted(set(gens)))
    ax.set_ylim(0.45, 0.75)
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Mean JS divergence from text-freq prior (log₂)")

# Overall trendline across all families in a small inset
ax_inset = axes[-1].inset_axes([0.55, 0.05, 0.42, 0.40])
ax_inset.scatter(all_gens, all_js, color="gray", s=30, zorder=3)
if len(set(all_gens)) >= 3:
    coeffs = np.polyfit(all_gens, all_js, 1)
    xs = np.linspace(min(all_gens)-0.1, max(all_gens)+0.1, 50)
    ax_inset.plot(xs, np.polyval(coeffs, xs), color="black", linewidth=1.2)
    r, p = pearsonr(all_gens, all_js)
    ax_inset.set_title(f"all families\nr={r:+.2f} p={p:.2f}", fontsize=7)
ax_inset.set_xlabel("gen", fontsize=7); ax_inset.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig("model_generation_trend.png", dpi=150, bbox_inches="tight")
print("Saved model_generation_trend.png")
