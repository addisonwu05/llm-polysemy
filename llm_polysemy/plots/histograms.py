"""Per-word stacked bar charts: sense distribution per model."""

import matplotlib.pyplot as plt
import numpy as np

from .. import config
from ..plotting_utils import (base_parser, require_data, models_for, get_senses,
                              dist_for, get_vocab, sense_colors, fig_dir, JUDGE_DISPLAY)


def main():
    args = base_parser(__doc__).parse_args()
    data, meanings = require_data(args.modality)
    models = models_for(args.modality)
    out = fig_dir(args.modality, "histograms")
    artifact = config.GEN[args.modality].artifact_word

    for word, word_data in data.items():
        senses = get_senses(word, meanings)
        all_labels = senses + ["multiple", "unclear"]
        color_map = sense_colors(senses)
        vocab = get_vocab(word, word_data, meanings, args.judge)

        model_counts = {}
        for m in models:
            if m not in word_data:
                continue
            d = dist_for(word_data[m], vocab, args.judge)
            if d is None:
                continue
            model_counts[m] = {lbl: d[vocab.index(lbl)] if lbl in vocab else 0.0
                               for lbl in all_labels}
        if not model_counts:
            continue

        present = list(model_counts)
        x = np.arange(len(present))
        fig, ax = plt.subplots(figsize=(max(10, len(present) * 0.75), 4.5))
        bottoms = np.zeros(len(present))
        for label in all_labels:
            vals = np.array([model_counts[m][label] for m in present])
            if vals.sum() == 0:
                continue
            ax.bar(x, vals, 0.65, bottom=bottoms, color=color_map[label], label=label)
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(present, rotation=40, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"Proportion of {artifact}")
        ax.set_title(f'"{word}" — sense distribution per model ({JUDGE_DISPLAY[args.judge]})', fontsize=12)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        plt.tight_layout()
        plt.savefig(out / f"{word}.png", dpi=120)
        plt.close()
    print(f"saved {out}/*.png ({len(data)} words)")


if __name__ == "__main__":
    main()
