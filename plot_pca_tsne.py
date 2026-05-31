"""
PCA and t-SNE on model sense distributions.

Each model is represented as a concatenation of its per-word probability
vectors over senses (GPT classifier). Words with missing data for a model
get a zero vector for that word's block. The resulting 17×D matrix is
projected to 2D with both PCA and t-SNE.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

data     = json.load(open("histograms.json"))
meanings = json.load(open("meanings.json"))

MODELS = [
    "flux", "flux_1pro", "flux_flex", "flux_max",
    "gemini", "gemini_25", "gemini_pro",
    "grok", "grok_base", "grok_pro",
    "openai", "openai_1", "openai_15", "openai_mini",
    "qwen", "qwen_1",
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

# ── Build feature matrix ───────────────────────────────────────────────────────
# For each word, build a per-word vocab and a block of size len(vocab).
# Each model gets one row = concatenation of all blocks.

words = list(data.keys())
feature_blocks = []   # list of (word, vocab)

for word in words:
    raw = meanings.get(word, [])
    base = [s["sense"] for s in raw] if isinstance(raw, list) else list(raw.keys())
    seen = set()
    for m in data[word].values():
        seen.update(m["gpt"]["_counts"].keys())
    vocab = base[:]
    for lbl in ["multiple", "unclear", "other"] + sorted(seen):
        if lbl not in vocab:
            vocab.append(lbl)
    feature_blocks.append((word, vocab))

# Build matrix: rows = models, cols = concatenated probability vectors
rows = []
for model in MODELS:
    vec = []
    for word, vocab in feature_blocks:
        if model in data[word]:
            counts = data[word][model]["gpt"]["_counts"]
            block = np.array([counts.get(s, 0) for s in vocab], dtype=float)
            s = block.sum()
            block = block / s if s > 0 else np.zeros(len(vocab))
        else:
            block = np.zeros(len(vocab))
        vec.extend(block)
    rows.append(vec)

X = np.array(rows)   # (17, D)
print(f"Feature matrix: {X.shape}")

# ── PCA and t-SNE ─────────────────────────────────────────────────────────────
X_scaled = StandardScaler().fit_transform(X)

pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

tsne   = TSNE(n_components=2, perplexity=5, random_state=42, max_iter=2000)
X_tsne = tsne.fit_transform(X_scaled)


def make_plot(coords, title, filename, variance_explained=None):
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, model in enumerate(MODELS):
        family = FAMILY[model]
        color  = FAMILY_COLOR[family]
        ax.scatter(coords[i, 0], coords[i, 1], color=color, s=120, zorder=3,
                   edgecolors="white", linewidths=0.8)
        label = MODEL_LABELS[model]
        txt = ax.text(coords[i, 0], coords[i, 1] + 0.015 * (np.ptp(coords[:, 1])),
                      label, fontsize=8, ha="center", va="bottom", color=color)
        txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    # Draw lines connecting models within the same family in version order
    FAMILY_ORDER = {
        "FLUX":   ["flux_1pro", "flux_flex", "flux", "flux_max"],
        "Gemini": ["gemini_25", "gemini", "gemini_pro"],
        "Grok":   ["grok_base", "grok_pro", "grok"],
        "OpenAI": ["openai_1", "openai_15", "openai", "openai_mini"],
        "Qwen":   ["qwen_1", "qwen"],
    }
    idx = {m: i for i, m in enumerate(MODELS)}
    for family, order in FAMILY_ORDER.items():
        color = FAMILY_COLOR[family]
        xs = [coords[idx[m], 0] for m in order]
        ys = [coords[idx[m], 1] for m in order]
        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.5, zorder=2)

    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=f) for f, c in FAMILY_COLOR.items()]
    ax.legend(handles=handles, fontsize=9, loc="best")

    xlabel = f"PC 1 ({variance_explained[0]:.1%} var)" if variance_explained is not None else "dim 1"
    ylabel = f"PC 2 ({variance_explained[1]:.1%} var)" if variance_explained is not None else "dim 2"
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.4, linestyle="--")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


make_plot(X_pca, "PCA of model sense distributions\n(each model = concatenated per-word probability vectors, GPT classifier)",
          "model_pca.png",
          variance_explained=pca.explained_variance_ratio_)

make_plot(X_tsne, "t-SNE of model sense distributions\n(perplexity=5, 17 models)",
          "model_tsne.png")

print(f"\nPCA variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
