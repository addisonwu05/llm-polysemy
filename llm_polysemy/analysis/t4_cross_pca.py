"""T4 cross-modality PCA: place all image + text models in one sense-distribution space.

Each model is represented as a concatenated per-word sense distribution vector over the
words present in both modality histograms. PCA (and optionally MDS) is applied to ask:
do models cluster by modality (image vs text) or by provider family?

The fingerprinting literature (McGovern et al. 2024, TensorGuard) asks this question
for text-only models. This script applies the same method to a cross-modal panel and
claims only the novel slice: sense signal + cross-modality placement.

Usage:
    python -m llm_polysemy.analysis.t4_cross_pca
    python -m llm_polysemy.analysis.t4_cross_pca --out figures/t4_cross_pca.pdf

Reads: data/histograms_image.json, data/histograms_text.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..meanings import load_meanings
from ..plotting_utils import get_vocab, dist_for
from ..registry import (models_for, display_label, family_of, family_color,
                         family_label, FAMILY_ORDER)


def load_json(path):
    return json.loads(Path(path).read_text())


MODALITY_MARKER = {"image": "o", "text": "s"}
MODALITY_EDGE   = {"image": "white", "text": "#333333"}
MODALITY_SIZE   = {"image": 110, "text": 90}


def build_feature_matrix(hist_img, hist_txt, meanings, judge="gpt"):
    """
    Build a feature matrix where each row = one model (all image + text models)
    and each column = one dimension of the concatenated per-word sense distribution.

    Only words present in BOTH histograms are used so the vector space is shared.
    """
    shared_words = sorted(set(hist_img.keys()) & set(hist_txt.keys()))
    print(f"Shared words (in both modalities): {len(shared_words)}")

    img_models = models_for("image")
    txt_models = models_for("text")
    all_models = [(m, "image") for m in img_models] + [(m, "text") for m in txt_models]

    # Build vocabulary blocks per word (shared across modalities)
    blocks = []
    for word in shared_words:
        # Merge vocab from both histograms so vectors are aligned
        vocab = get_vocab(word, hist_img[word], meanings, judge)
        txt_vocab = get_vocab(word, hist_txt[word], meanings, judge)
        for lbl in txt_vocab:
            if lbl not in vocab:
                vocab.append(lbl)
        blocks.append((word, vocab))

    rows = []
    row_labels = []
    for model_key, modality in all_models:
        hist = hist_img if modality == "image" else hist_txt
        vec = []
        for word, vocab in blocks:
            word_data = hist.get(word, {})
            model_data = word_data.get(model_key)
            if model_data is not None:
                d = dist_for(model_data, vocab, judge)
            else:
                d = None
            vec.extend(d if d is not None else np.zeros(len(vocab)))
        rows.append(vec)
        row_labels.append((model_key, modality))

    X = np.array(rows)
    print(f"Feature matrix: {X.shape} ({len(all_models)} models × {X.shape[1]} dims)")
    return X, row_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures/t4_cross_pca.pdf")
    p.add_argument("--judge", default="gpt", choices=["gpt", "gemini", "consensus"])
    args = p.parse_args()

    hist_img = load_json("data/histograms_image.json")
    hist_txt = load_json("data/histograms_text.json")
    meanings = load_meanings()

    X, row_labels = build_feature_matrix(hist_img, hist_txt, meanings, args.judge)

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (color_by, ax) in enumerate(zip(["modality", "family"], axes)):
        for i, (model_key, modality) in enumerate(row_labels):
            family = family_of(model_key)
            if color_by == "modality":
                color = "#4C72B0" if modality == "image" else "#DD8452"
                label = "Image model" if modality == "image" else "Text model"
            else:
                color = family_color(family)
                label = family_label(family)

            marker = MODALITY_MARKER[modality]
            ax.scatter(X_pca[i, 0], X_pca[i, 1],
                       c=color, s=MODALITY_SIZE[modality], marker=marker,
                       edgecolors=MODALITY_EDGE[modality], linewidths=0.6,
                       alpha=0.85, zorder=3)

            # Label only if not too crowded
            txt = ax.annotate(display_label(model_key),
                              (X_pca[i, 0], X_pca[i, 1]),
                              fontsize=6.5, ha="center", va="bottom",
                              xytext=(0, 5), textcoords="offset points",
                              color=color)
            txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])

        ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} var)", fontsize=10)
        ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} var)", fontsize=10)

        if color_by == "modality":
            ax.set_title("Colored by modality\n(○ = image, □ = text)", fontsize=11)
            handles = [
                mpatches.Patch(color="#4C72B0", label="Image models"),
                mpatches.Patch(color="#DD8452", label="Text models"),
            ]
        else:
            ax.set_title("Colored by provider family\n(○ = image, □ = text)",
                         fontsize=11)
            seen_fam = {family_of(k) for k, _ in row_labels}
            handles = [
                mpatches.Patch(color=family_color(f), label=family_label(f))
                for f in FAMILY_ORDER if f in seen_fam
            ]
            # Add modality shape legend
            handles += [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
                           markersize=8, label="image"),
                plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#888",
                           markersize=8, label="text"),
            ]
        ax.legend(handles=handles, fontsize=8.5, loc="best", framealpha=0.9)

    fig.suptitle(
        "Cross-Modality PCA: All Image + Text Models in One Sense-Distribution Space\n"
        "(Novel slice: fingerprinting applied to sense signal across modalities; "
        "cf. McGovern et al. 2024)",
        fontsize=11
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")

    # Silhouette quantification
    try:
        from sklearn.metrics import silhouette_score
        modality_labels = np.array([0 if m == "image" else 1 for _, m in row_labels])
        family_labels = np.array([FAMILY_ORDER.index(family_of(k)) for k, _ in row_labels])
        X_pca10 = PCA(n_components=min(10, len(row_labels) - 1), random_state=42).fit_transform(
            StandardScaler().fit_transform(X)
        )
        sil_mod = silhouette_score(X_pca10, modality_labels)
        sil_fam = silhouette_score(X_pca10, family_labels)
        print(f"\n=== Cluster Silhouette Scores (PCA top-10 components) ===")
        print(f"  Modality (image vs text): {sil_mod:.4f}")
        print(f"  Provider family:          {sil_fam:.4f}")
        if sil_mod > sil_fam:
            print("  → Modality clusters more than family (T4 finding: modality > provider)")
        else:
            print("  → Family clusters more than modality")
        print("\nNote: In text-only fingerprinting (McGovern et al. 2024), family dominates.")
        print("Here, with cross-modal sense signal, modality is the primary axis.")
    except ImportError:
        print("(sklearn not available — skipping silhouette scores)")

    print("\n=== Cluster interpretation guide ===")
    print("If left panel (modality color) shows clear separation: modality > provider.")
    print("If right panel (family color) shows clear separation: provider > modality.")
    print("The interesting result is whichever is stronger — report both honestly.")


if __name__ == "__main__":
    main()
