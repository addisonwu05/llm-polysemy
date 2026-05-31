"""
Two outputs:
  1. plots_textfreq/<word>.png  — per-word stacked bars like before but with a
     "text freq" reference column prepended.
  2. model_text_divergence.png  — per model, mean JS(text_freq || model_dist)
     averaged across all words; shows which models deviate most from text priors.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

data          = json.load(open("histograms.json"))
meanings      = json.load(open("meanings.json"))
text_freq     = json.load(open("text_frequencies.json"))

os.makedirs("plots_textfreq", exist_ok=True)

MODELS = [
    "flux", "flux_1pro", "flux_flex", "flux_max",
    "gemini", "gemini_25", "gemini_pro",
    "grok", "grok_base", "grok_pro",
    "openai", "openai_1", "openai_15", "openai_mini",
    "qwen", "qwen_1", "qwen_20",
]

MODEL_LABELS = {
    "openai": "gpt-image-2", "openai_15": "gpt-image-1.5",
    "openai_1": "gpt-image-1", "openai_mini": "gpt-image-1-mini",
    "grok": "grok-quality", "grok_base": "grok-base", "grok_pro": "grok-pro",
    "gemini": "gemini-3.1-flash", "gemini_25": "gemini-2.5-flash", "gemini_pro": "gemini-3-pro",
    "qwen": "Qwen-2.0-Pro", "qwen_20": "Qwen-2.0", "qwen_1": "Qwen-v1",
    "flux": "FLUX.2-pro", "flux_max": "FLUX.2-max",
    "flux_flex": "FLUX.2-flex", "flux_1pro": "FLUX.1.1-pro",
}

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#CCB974",
]


def get_senses(word):
    raw = meanings.get(word, [])
    return [s["sense"] for s in raw] if isinstance(raw, list) else list(raw.keys())


def get_vocab(word, word_data):
    base = get_senses(word)
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


# ── 1. Per-word histograms with text-freq reference bar ───────────────────────
for word, word_data in data.items():
    vocab = get_vocab(word, word_data)
    senses = get_senses(word)
    all_labels = senses + [l for l in ["multiple", "unclear", "other"] if l in vocab]

    color_map = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(senses)}
    color_map.update({"multiple": "#BBBBBB", "unclear": "#555555", "other": "#333333"})

    tf_vec = text_dist(word, vocab)

    cols = ["[text freq]"] + [m for m in MODELS if m in word_data]
    x = np.arange(len(cols))

    fig, ax = plt.subplots(figsize=(max(10, len(cols) * 0.8), 4.5))
    bottoms = np.zeros(len(cols))

    for label in all_labels:
        idx = vocab.index(label)
        vals = []
        for col in cols:
            if col == "[text freq]":
                vals.append(tf_vec[idx] if tf_vec is not None else 0.0)
            else:
                d = model_dist(word_data[col]["gpt"]["_counts"], vocab)
                vals.append(d[idx] if d is not None else 0.0)
        vals = np.array(vals)
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, 0.65, bottom=bottoms, color=color_map.get(label, "#999999"), label=label)
        bottoms += vals

    # Highlight the text-freq column
    ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--")

    tick_labels = ["text\nfreq"] + [MODEL_LABELS.get(m, m) for m in MODELS if m in word_data]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion")
    ax.set_title(f'"{word}" — model distributions vs text-frequency prior', fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    plt.tight_layout()
    plt.savefig(f"plots_textfreq/{word}.png", dpi=120, bbox_inches="tight")
    plt.close()

print("Saved plots_textfreq/")

# ── 2. Per-model mean JS(text_freq || model) across all words ─────────────────
model_js = {m: [] for m in MODELS}

for word, word_data in data.items():
    vocab = get_vocab(word, word_data)
    tf_vec = text_dist(word, vocab)
    if tf_vec is None:
        continue
    for m in MODELS:
        if m not in word_data:
            continue
        d = model_dist(word_data[m]["gpt"]["_counts"], vocab)
        if d is None:
            continue
        js = jensenshannon(tf_vec, d, base=2)
        if not np.isnan(js):
            model_js[m].append(js)

means = {m: np.mean(v) for m, v in model_js.items() if v}
ranked = sorted(means.items(), key=lambda x: x[1])

labels = [MODEL_LABELS.get(m, m) for m, _ in ranked]
values = [v for _, v in ranked]

# Color by family
FAMILY_COLOR = {
    "flux": "#4C72B0", "flux_1pro": "#4C72B0", "flux_flex": "#4C72B0", "flux_max": "#4C72B0",
    "gemini": "#55A868", "gemini_25": "#55A868", "gemini_pro": "#55A868",
    "grok": "#C44E52", "grok_base": "#C44E52", "grok_pro": "#C44E52",
    "openai": "#DD8452", "openai_1": "#DD8452", "openai_15": "#DD8452", "openai_mini": "#DD8452",
    "qwen": "#8172B3", "qwen_1": "#8172B3", "qwen_20": "#8172B3",
}
colors = [FAMILY_COLOR.get(m, "#999999") for m, _ in ranked]

fig, ax = plt.subplots(figsize=(9, 7))
y = np.arange(len(ranked))
ax.barh(y, values, color=colors, alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Mean JS divergence from text-frequency prior (log₂)", fontsize=9)
ax.set_title("How much does each model deviate from text-frequency priors?\n"
             "(lower = model sense distribution matches text frequency; "
             "higher = model has its own bias)", fontsize=10)
for i, (m, v) in enumerate(ranked):
    ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)

# Family legend
from matplotlib.patches import Patch
legend_items = [
    Patch(color="#4C72B0", label="FLUX"),
    Patch(color="#55A868", label="Gemini"),
    Patch(color="#C44E52", label="Grok"),
    Patch(color="#DD8452", label="OpenAI"),
    Patch(color="#8172B3", label="Qwen"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig("model_text_divergence.png", dpi=150, bbox_inches="tight")
print("Saved model_text_divergence.png")

print("\nModel deviation from text priors (lowest → highest):")
for m, v in ranked:
    print(f"  {MODEL_LABELS.get(m,m):22s}  JS={v:.4f}")
