"""NEW cross-modality figure: per family, does the image model resolve polysemy
the same way the text model does?

For each word and each family present in BOTH modalities, aggregate the family's
image-model sense distribution and its text-model distribution over a shared vocab,
then take JS divergence. Requires both histograms.json and histograms_text.json.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .. import config
from ..meanings import load_meanings
from ..plotting_utils import (base_parser, load_data, get_vocab, dist_for, js,
                              fig_dir, JUDGE_DISPLAY)
from ..registry import families, family_groups, family_label, family_color


def _agg(word_data, keys, vocab, judge, mode):
    present = [k for k in keys if k in word_data]
    if not present:
        return None
    if mode == "latest":  # keys are ordered oldest→newest
        for k in reversed(present):
            d = dist_for(word_data[k], vocab, judge)
            if d is not None:
                return d
        return None
    dists = [d for k in present if (d := dist_for(word_data[k], vocab, judge)) is not None]
    if not dists:
        return None
    m = np.mean(dists, axis=0)
    s = m.sum()
    return m / s if s > 0 else None


def main():
    p = base_parser(__doc__)
    p.add_argument("--agg", choices=["mean", "latest"], default="mean",
                   help="aggregate a family's models by mean distribution or its newest model")
    args = p.parse_args()

    data_img, _ = load_data("image")
    data_txt, _ = load_data("text")
    if data_img is None or data_txt is None:
        print("Need both histograms.json and histograms_text.json. "
              "Run classify --histograms for both modalities first.")
        raise SystemExit(0)
    meanings = load_meanings()
    out = fig_dir("cross")

    fams = [f for f in families("image") if f in families("text")]
    img_groups, txt_groups = family_groups("image"), family_groups("text")
    common_words = [w for w in data_img if w in data_txt]

    per_family = {f: [] for f in fams}        # family -> [JS per word]
    per_word = {w: [] for w in common_words}  # word -> [JS per family]

    for word in common_words:
        merged = {**data_img[word], **data_txt[word]}
        vocab = get_vocab(word, merged, meanings, args.judge)
        for f in fams:
            img = _agg(data_img[word], img_groups[f], vocab, args.judge, args.agg)
            txt = _agg(data_txt[word], txt_groups[f], vocab, args.judge, args.agg)
            if img is None or txt is None:
                continue
            v = js(img, txt)
            if not np.isnan(v):
                per_family[f].append(v)
                per_word[word].append(v)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: per-family mean image-vs-text JS
    fam_means = [(f, float(np.mean(per_family[f]))) for f in fams if per_family[f]]
    fam_means.sort(key=lambda x: x[1])
    ax = axes[0]
    y = np.arange(len(fam_means))
    ax.barh(y, [m for _, m in fam_means], color=[family_color(f) for f, _ in fam_means], alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels([family_label(f) for f, _ in fam_means], fontsize=11)
    ax.set_xlabel("Mean JS(image ‖ text) over words (log₂)", fontsize=9)
    ax.set_title(f"Per-family image-vs-text divergence ({args.agg} aggregate)\n"
                 "higher = a lab's image & text models resolve polysemy differently", fontsize=10)
    for i, (f, m) in enumerate(fam_means):
        ax.text(m + 0.003, i, f"{m:.3f}", va="center", fontsize=8)

    # Panel 2: words that diverge most between seeing vs reading
    word_means = sorted(((w, float(np.mean(v))) for w, v in per_word.items() if v),
                        key=lambda x: x[1])
    top = word_means[-20:]
    ax2 = axes[1]
    y2 = np.arange(len(top))
    ax2.barh(y2, [m for _, m in top], color="#8E44AD", alpha=0.8)
    ax2.set_yticks(y2); ax2.set_yticklabels([w for w, _ in top], fontsize=10)
    ax2.set_xlabel("Mean cross-modality JS across families (log₂)", fontsize=9)
    ax2.set_title("Words that diverge most between seeing vs reading", fontsize=10)
    for i, (w, m) in enumerate(top):
        ax2.text(m + 0.003, i, f"{m:.3f}", va="center", fontsize=8)

    plt.suptitle(f"Image vs text sense resolution ({JUDGE_DISPLAY[args.judge]}, {len(common_words)} words)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = out / f"model_image_vs_text_js_{args.agg}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")
    print("\nPer-family mean image-vs-text JS:")
    for f, m in fam_means:
        print(f"  {family_label(f):8s}  {m:.4f}  (n={len(per_family[f])} words)")


if __name__ == "__main__":
    main()
