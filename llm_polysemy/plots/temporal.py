"""Per-word sense distributions across model generations, one panel per family."""

import matplotlib.pyplot as plt
import numpy as np

from .. import config
from ..plotting_utils import (base_parser, require_data, get_senses, dist_for,
                              get_vocab, sense_colors, fig_dir, JUDGE_DISPLAY)
from ..registry import generation_order, family_label


def main():
    args = base_parser(__doc__).parse_args()
    data, meanings = require_data(args.modality)
    gen_order = generation_order(args.modality)   # family -> [(key, gen, label)]
    out = fig_dir(args.modality, "temporal")
    artifact = config.GEN[args.modality].artifact_word

    for word, word_data in data.items():
        senses = get_senses(word, meanings)
        all_labels = senses + ["multiple", "unclear"]
        color_map = sense_colors(senses)
        vocab = get_vocab(word, word_data, meanings, args.judge)

        fams = list(gen_order)
        fig, axes = plt.subplots(1, len(fams), figsize=(18, 4), sharey=True)
        fig.suptitle(f'"{word}" — sense distribution over model generations ({JUDGE_DISPLAY[args.judge]})', fontsize=12)

        for ax, fam in zip(axes, fams):
            present = [(k, lbl) for k, gen, lbl in gen_order[fam] if k in word_data]
            if not present:
                ax.set_visible(False)
                continue
            keys, labels = zip(*present)
            x = np.arange(len(keys))
            bottoms = np.zeros(len(keys))
            dists = [dist_for(word_data[k], vocab, args.judge) for k in keys]
            for sense in all_labels:
                idx = vocab.index(sense) if sense in vocab else None
                vals = np.array([(d[idx] if (d is not None and idx is not None) else 0.0) for d in dists])
                if vals.sum() == 0:
                    continue
                ax.bar(x, vals, 0.6, bottom=bottoms, color=color_map[sense], label=sense)
                bottoms += vals
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_title(family_label(fam), fontsize=10, fontweight="bold")
            ax.set_ylim(0, 1.05)
            if ax is axes[0]:
                ax.set_ylabel(f"Proportion of {artifact}")

        handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[s]) for s in all_labels if s in color_map]
        leg = [s for s in all_labels if s in color_map]
        fig.legend(handles, leg, loc="lower center", ncol=len(leg),
                   fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(out / f"{word}.png", dpi=120, bbox_inches="tight")
        plt.close()
    print(f"saved {out}/*.png ({len(data)} words)")


if __name__ == "__main__":
    main()
