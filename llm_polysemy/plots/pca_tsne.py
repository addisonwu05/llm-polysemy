"""PCA and t-SNE of model sense distributions (each model = concatenated per-word
probability vectors)."""

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from ..plotting_utils import (base_parser, require_data, models_for, get_vocab,
                              dist_for, fig_dir, JUDGE_DISPLAY)
from ..registry import display_label, family_of, family_color, family_label, family_groups


def main():
    args = base_parser(__doc__).parse_args()
    data, meanings = require_data(args.modality)
    MODELS = models_for(args.modality)
    out = fig_dir(args.modality)
    suffix = "" if args.modality == "image" else "_text"

    words = list(data)
    blocks = [(w, get_vocab(w, data[w], meanings, args.judge)) for w in words]

    rows = []
    for model in MODELS:
        vec = []
        for word, vocab in blocks:
            d = dist_for(data[word][model], vocab, args.judge) if model in data[word] else None
            vec.extend(d if d is not None else np.zeros(len(vocab)))
        rows.append(vec)
    X = np.array(rows)
    print(f"Feature matrix: {X.shape}")

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    perplexity = min(5, len(MODELS) - 1)
    X_tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=2000).fit_transform(X_scaled)

    groups = family_groups(args.modality)

    def make_plot(coords, title, filename, variance=None):
        fig, ax = plt.subplots(figsize=(9, 7))
        for i, model in enumerate(MODELS):
            color = family_color(family_of(model))
            ax.scatter(coords[i, 0], coords[i, 1], color=color, s=120, zorder=3,
                       edgecolors="white", linewidths=0.8)
            txt = ax.text(coords[i, 0], coords[i, 1] + 0.015 * np.ptp(coords[:, 1]),
                          display_label(model), fontsize=8, ha="center", va="bottom", color=color)
            txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])
        idx = {m: i for i, m in enumerate(MODELS)}
        for fam, keys in groups.items():
            color = family_color(fam)
            ax.plot([coords[idx[m], 0] for m in keys], [coords[idx[m], 1] for m in keys],
                    color=color, linewidth=1.2, alpha=0.5, zorder=2)
        handles = [Patch(color=family_color(f), label=family_label(f)) for f in groups]
        ax.legend(handles=handles, fontsize=9, loc="best")
        if variance is not None:
            ax.set_xlabel(f"PC 1 ({variance[0]:.1%} var)"); ax.set_ylabel(f"PC 2 ({variance[1]:.1%} var)")
        else:
            ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
        ax.set_title(title, fontsize=12)
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.4, linestyle="--")
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {filename}")

    jd = JUDGE_DISPLAY[args.judge]
    make_plot(X_pca, f"PCA of model sense distributions\n(each model = concatenated per-word probability vectors, {jd})",
              out / f"model_pca{suffix}.png", variance=pca.explained_variance_ratio_)
    make_plot(X_tsne, f"t-SNE of model sense distributions\n(perplexity={perplexity}, {len(MODELS)} models, {jd})",
              out / f"model_tsne{suffix}.png")
    print(f"\nPCA variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}")


if __name__ == "__main__":
    main()
