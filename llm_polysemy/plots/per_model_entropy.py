"""Per-model sense-distribution entropy.

For each model (both modalities), compute the Shannon entropy of its sense
distribution for each word, then average across words.  Displays text and image
models side by side so the modality gap is immediately visible.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import entropy as scipy_entropy

from ..meanings import load_meanings
from ..plotting_utils import (base_parser, load_data, get_vocab, dist_for, fig_dir,
                               JUDGE_DISPLAY)
from ..registry import (models_for, display_label, family_of, family_color,
                        family_label, FAMILY_ORDER)


def _model_entropies(data, modality, meanings, judge):
    """Return dict {model_key: mean_entropy} over all words the model was judged on."""
    keys = models_for(modality)
    totals = {k: 0.0 for k in keys}
    counts = {k: 0 for k in keys}
    for word, word_data in data.items():
        merged = word_data  # vocab built from this modality only is fine for H
        vocab = get_vocab(word, merged, meanings, judge)
        for k in keys:
            if k not in word_data:
                continue
            d = dist_for(word_data[k], vocab, judge)
            if d is None:
                continue
            h = float(scipy_entropy(d + 1e-12, base=2))
            totals[k] += h
            counts[k] += 1
    return {k: totals[k] / counts[k] for k in keys if counts[k] > 0}


def main():
    p = base_parser(__doc__)
    args = p.parse_args()

    data_img, _ = load_data("image")
    data_txt, _ = load_data("text")
    if data_img is None or data_txt is None:
        print("Need both histograms JSON files.")
        raise SystemExit(0)
    meanings = load_meanings()

    h_txt = _model_entropies(data_txt, "text",  meanings, args.judge)
    h_img = _model_entropies(data_img, "image", meanings, args.judge)

    # ── Print table ─────────────────────────────────────────────────────────────
    print(f"\nMean per-word entropy (bits, judge={args.judge})\n")
    print(f"  {'TEXT MODEL':<24}  {'H':>6}    {'IMAGE MODEL':<24}  {'H':>6}")
    txt_sorted = sorted(h_txt.items(), key=lambda x: x[1])
    img_sorted = sorted(h_img.items(), key=lambda x: x[1])
    for i in range(max(len(txt_sorted), len(img_sorted))):
        t = f"{display_label(txt_sorted[i][0]):<24}  {txt_sorted[i][1]:.4f}" if i < len(txt_sorted) else ""
        g = f"{display_label(img_sorted[i][0]):<24}  {img_sorted[i][1]:.4f}" if i < len(img_sorted) else ""
        print(f"  {t}    {g}")

    txt_global = np.mean(list(h_txt.values()))
    img_global = np.mean(list(h_img.values()))
    print(f"\n  Text mean: {txt_global:.4f} bits    Image mean: {img_global:.4f} bits"
          f"    Δ = {txt_global - img_global:+.4f}")

    # ── Figure ──────────────────────────────────────────────────────────────────
    fig, (ax_t, ax_i) = plt.subplots(1, 2, figsize=(14, 7), sharey=False)

    def _bar(ax, h_dict, modality_label):
        pairs = sorted(h_dict.items(), key=lambda x: x[1])
        labels = [display_label(k) for k, _ in pairs]
        values = [v for _, v in pairs]
        colors = [family_color(family_of(k)) for k, _ in pairs]
        y = np.arange(len(pairs))
        ax.barh(y, values, color=colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Mean entropy (bits)", fontsize=10)
        ax.set_title(f"{modality_label} models", fontsize=11)
        ax.axvline(np.mean(values), color="black", linestyle="--", linewidth=1, alpha=0.6,
                   label=f"mean = {np.mean(values):.3f}")
        ax.legend(fontsize=8)
        for i, v in enumerate(values):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)

    _bar(ax_t, h_txt, "Text")
    _bar(ax_i, h_img, "Image")

    # Align x-axes
    xmax = max(ax_t.get_xlim()[1], ax_i.get_xlim()[1])
    ax_t.set_xlim(0, xmax)
    ax_i.set_xlim(0, xmax)

    families_present = list(dict.fromkeys(
        family_of(k) for k in list(h_txt) + list(h_img)
    ))
    legend_patches = [Patch(color=family_color(f), label=family_label(f))
                      for f in families_present]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(families_present),
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(
        f"Sense-distribution entropy per model ({JUDGE_DISPLAY[args.judge]})\n"
        f"Text mean {txt_global:.3f} bits  vs  Image mean {img_global:.3f} bits  "
        f"(Δ = {txt_global - img_global:+.3f})",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    out = fig_dir("cross")
    path = out / f"per_model_entropy_{args.judge}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
