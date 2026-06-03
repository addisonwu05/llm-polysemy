"""Model distributions vs a text-frequency prior:
  1. per-word stacked bars with a [text freq] reference column prepended
  2. per-model mean JS(text_freq || model) across all words.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..plotting_utils import (base_parser, require_data, models_for, get_senses,
                              get_vocab, dist_for, text_dist, js, load_prior,
                              sense_colors, fig_dir, JUDGE_DISPLAY)
from ..registry import display_label, family_of, family_label, family_color, families


def main():
    p = base_parser(__doc__)
    p.add_argument("--prior", choices=["manual", "wiki", "trends"], default="manual")
    args = p.parse_args()
    data, meanings = require_data(args.modality)
    freq = load_prior(args.prior)
    MODELS = models_for(args.modality)
    out = fig_dir(args.modality, "textfreq")
    out_static = fig_dir(args.modality)
    suffix = "" if args.modality == "image" else "_text"

    # 1. per-word histograms with text-freq reference bar
    for word, word_data in data.items():
        vocab = get_vocab(word, word_data, meanings, args.judge)
        senses = get_senses(word, meanings)
        all_labels = senses + [l for l in ["multiple", "unclear", "other"] if l in vocab]
        color_map = sense_colors(senses)
        tf_vec = text_dist(word, vocab, freq)
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
                    d = dist_for(word_data[col], vocab, args.judge)
                    vals.append(d[idx] if d is not None else 0.0)
            vals = np.array(vals)
            if vals.sum() == 0:
                continue
            ax.bar(x, vals, 0.65, bottom=bottoms, color=color_map.get(label, "#999999"), label=label)
            bottoms += vals
        ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--")
        tick_labels = ["text\nfreq"] + [display_label(m) for m in MODELS if m in word_data]
        ax.set_xticks(x); ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Proportion")
        ax.set_title(f'"{word}" — model distributions vs {args.prior} text-frequency prior', fontsize=11)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        plt.tight_layout()
        plt.savefig(out / f"{word}.png", dpi=120, bbox_inches="tight")
        plt.close()
    print(f"Saved {out}/*.png")

    # 2. per-model mean JS(text_freq || model)
    model_js = {m: [] for m in MODELS}
    for word, word_data in data.items():
        vocab = get_vocab(word, word_data, meanings, args.judge)
        tf_vec = text_dist(word, vocab, freq)
        if tf_vec is None:
            continue
        for m in MODELS:
            if m not in word_data:
                continue
            d = dist_for(word_data[m], vocab, args.judge)
            if d is None:
                continue
            v = js(tf_vec, d)
            if not np.isnan(v):
                model_js[m].append(v)

    ranked = sorted(((m, np.mean(v)) for m, v in model_js.items() if v), key=lambda x: x[1])
    labels = [display_label(m) for m, _ in ranked]
    values = [v for _, v in ranked]
    colors = [family_color(family_of(m)) for m, _ in ranked]

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(ranked))
    ax.barh(y, values, color=colors, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Mean JS divergence from text-frequency prior (log₂)", fontsize=9)
    ax.set_title(f"How much does each model deviate from {args.prior} text-frequency priors?\n"
                 f"(lower = matches text frequency; higher = own bias; {JUDGE_DISPLAY[args.judge]})", fontsize=10)
    for i, (m, v) in enumerate(ranked):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    ax.legend(handles=[Patch(color=family_color(f), label=family_label(f)) for f in families(args.modality)],
              loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_static / f"model_text_divergence{suffix}.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_static}/model_text_divergence{suffix}.png")


if __name__ == "__main__":
    main()
