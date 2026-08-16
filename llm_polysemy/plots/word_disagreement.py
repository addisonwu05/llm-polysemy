"""Per-word inter-model disagreement: mean pairwise JS divergence across models."""

import matplotlib.pyplot as plt
import numpy as np

from ..plotting_utils import (base_parser, require_data, models_for, get_vocab,
                              dist_for, mean_pairwise_js, fig_dir, JUDGE_DISPLAY)


def main():
    args = base_parser(__doc__).parse_args()
    data, meanings = require_data(args.modality)
    models = models_for(args.modality)
    out = fig_dir(args.modality)

    word_scores = {}
    for word, word_data in data.items():
        vocab = get_vocab(word, word_data, meanings, args.judge)
        dists = [d for m in models if m in word_data
                 and (d := dist_for(word_data[m], vocab, args.judge)) is not None]
        if len(dists) < 2:
            continue
        word_scores[word] = mean_pairwise_js(dists)

    ranked = sorted(word_scores.items(), key=lambda x: x[1])
    most_similar, most_different = ranked[:20], ranked[-20:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, items, title, color in [
        (axes[0], most_similar, "Most agreed-upon words\n(models converge on the same sense)", "#4CAF50"),
        (axes[1], list(reversed(most_different)), "Most disagreed-upon words\n(models pick different senses)", "#E53935"),
    ]:
        words = [w for w, _ in items]
        scores = [s for _, s in items]
        y = np.arange(len(words))
        ax.barh(y, scores, color=color, alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels(words, fontsize=10)
        ax.set_xlabel("Mean pairwise JS divergence (log₂)", fontsize=9)
        ax.set_xlim(0, max(word_scores.values()) * 1.1)
        ax.set_title(title, fontsize=11)
        for i, (w, s) in enumerate(items):
            ax.text(s + 0.003, i, f"{s:.3f}", va="center", fontsize=8)

    plt.suptitle(f"Per-word inter-model disagreement\n(mean pairwise JS divergence across {len(models)} "
                 f"{args.modality} models, {JUDGE_DISPLAY[args.judge]})", fontsize=12, y=1.01)
    plt.tight_layout()
    path = out / "word_disagreement.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")
    print("\nTop 10 most agreed:")
    for w, s in most_similar[:10]:
        print(f"  {w:15s} {s:.4f}")
    print("\nTop 10 most disagreed:")
    for w, s in list(reversed(most_different))[:10]:
        print(f"  {w:15s} {s:.4f}")


if __name__ == "__main__":
    main()
