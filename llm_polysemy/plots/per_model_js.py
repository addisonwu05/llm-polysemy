"""Per-model image-vs-text JS divergence.

For every (text_model, image_model) pair, compute mean JS divergence across all
words that both models were judged on, then visualise as a heatmap plus marginal
bar charts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ..meanings import load_meanings
from ..plotting_utils import (base_parser, load_data, get_vocab, dist_for, js,
                               fig_dir, JUDGE_DISPLAY)
from ..registry import models_for, display_label, family_of, family_color, FAMILY_ORDER


def _build_matrix(data_txt, data_img, meanings, judge):
    """Return (txt_keys, img_keys, matrix, word_counts) where matrix[i,j] is the
    mean JS between text model i and image model j, averaged over all words where
    both have data; word_counts[i,j] is how many words contributed."""
    txt_keys = [k for k in models_for("text") if any(k in data_txt.get(w, {}) for w in data_txt)]
    img_keys = [k for k in models_for("image") if any(k in data_img.get(w, {}) for w in data_img)]

    common_words = [w for w in data_txt if w in data_img]

    mat = np.full((len(txt_keys), len(img_keys)), np.nan)
    cnt = np.zeros((len(txt_keys), len(img_keys)), dtype=int)

    for wi, word in enumerate(common_words):
        merged = {**data_img[word], **data_txt[word]}
        vocab = get_vocab(word, merged, meanings, judge)
        for ti, tk in enumerate(txt_keys):
            dt = dist_for(data_txt[word].get(tk, {}), vocab, judge)
            if dt is None:
                continue
            for ii, ik in enumerate(img_keys):
                di = dist_for(data_img[word].get(ik, {}), vocab, judge)
                if di is None:
                    continue
                v = js(dt, di)
                if np.isnan(v):
                    continue
                if np.isnan(mat[ti, ii]):
                    mat[ti, ii] = 0.0
                mat[ti, ii] += v
                cnt[ti, ii] += 1

    with np.errstate(invalid="ignore"):
        mat = np.where(cnt > 0, mat / cnt, np.nan)

    return txt_keys, img_keys, mat, cnt


def main():
    p = base_parser(__doc__)
    args = p.parse_args()

    data_img, _ = load_data("image")
    data_txt, _ = load_data("text")
    if data_img is None or data_txt is None:
        print("Need both histograms JSON files. Run classify --histograms for both modalities.")
        raise SystemExit(0)
    meanings = load_meanings()

    txt_keys, img_keys, mat, cnt = _build_matrix(data_txt, data_img, meanings, args.judge)

    txt_labels = [display_label(k) for k in txt_keys]
    img_labels = [display_label(k) for k in img_keys]

    # ── Print table ─────────────────────────────────────────────────────────────
    col_w = max(len(l) for l in img_labels) + 2
    row_w = max(len(l) for l in txt_labels) + 2
    header = " " * row_w + "".join(f"{l:>{col_w}}" for l in img_labels)
    print(f"\nMean JS divergence (text model rows × image model cols), judge={args.judge}")
    print(header)
    for ti, tk in enumerate(txt_keys):
        row = f"{txt_labels[ti]:<{row_w}}"
        for ii in range(len(img_keys)):
            v = mat[ti, ii]
            row += f"{'—' if np.isnan(v) else f'{v:.4f}':>{col_w}}"
        print(row)

    # Marginals
    txt_means = np.nanmean(mat, axis=1)
    img_means = np.nanmean(mat, axis=0)
    print("\nText-model marginals (mean JS over all image models):")
    for k, m in sorted(zip(txt_keys, txt_means), key=lambda x: x[1]):
        print(f"  {display_label(k):<22s}  {m:.4f}")
    print("\nImage-model marginals (mean JS over all text models):")
    for k, m in sorted(zip(img_keys, img_means), key=lambda x: x[1]):
        print(f"  {display_label(k):<22s}  {m:.4f}")

    # ── Figure ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(1, 3, width_ratios=[10, 3, 3], wspace=0.35)

    # Heatmap
    ax_heat = fig.add_subplot(gs[0])
    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    im = ax_heat.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax,
                        interpolation="nearest")
    ax_heat.set_xticks(range(len(img_keys)))
    ax_heat.set_xticklabels(img_labels, rotation=45, ha="right", fontsize=8)
    ax_heat.set_yticks(range(len(txt_keys)))
    ax_heat.set_yticklabels(txt_labels, fontsize=9)
    ax_heat.set_xlabel("Image model", fontsize=10)
    ax_heat.set_ylabel("Text model", fontsize=10)
    ax_heat.set_title(f"JS(text‖image) per model pair\n({JUDGE_DISPLAY[args.judge]}, "
                      f"mean over {len([w for w in data_txt if w in data_img])} words)", fontsize=10)
    for ti in range(len(txt_keys)):
        for ii in range(len(img_keys)):
            v = mat[ti, ii]
            if not np.isnan(v):
                color = "white" if v > (vmin + 0.6 * (vmax - vmin)) else "black"
                ax_heat.text(ii, ti, f"{v:.3f}", ha="center", va="center",
                             fontsize=6, color=color)
    plt.colorbar(im, ax=ax_heat, label="JS divergence (log₂ bits)")

    # Text-model marginals bar chart
    ax_txt = fig.add_subplot(gs[1])
    order_t = np.argsort(txt_means)
    colors_t = [family_color(family_of(txt_keys[i])) for i in order_t]
    ax_txt.barh(range(len(txt_keys)), txt_means[order_t], color=colors_t, alpha=0.85)
    ax_txt.set_yticks(range(len(txt_keys)))
    ax_txt.set_yticklabels([txt_labels[i] for i in order_t], fontsize=8)
    ax_txt.set_xlabel("Mean JS (all image models)", fontsize=9)
    ax_txt.set_title("Text model\nmarginals", fontsize=10)
    for i, idx in enumerate(order_t):
        ax_txt.text(txt_means[idx] + 0.002, i, f"{txt_means[idx]:.3f}", va="center", fontsize=7)

    # Image-model marginals bar chart
    ax_img = fig.add_subplot(gs[2])
    order_i = np.argsort(img_means)
    colors_i = [family_color(family_of(img_keys[i])) for i in order_i]
    ax_img.barh(range(len(img_keys)), img_means[order_i], color=colors_i, alpha=0.85)
    ax_img.set_yticks(range(len(img_keys)))
    ax_img.set_yticklabels([img_labels[i] for i in order_i], fontsize=8)
    ax_img.set_xlabel("Mean JS (all text models)", fontsize=9)
    ax_img.set_title("Image model\nmarginals", fontsize=10)
    for i, idx in enumerate(order_i):
        ax_img.text(img_means[idx] + 0.002, i, f"{img_means[idx]:.3f}", va="center", fontsize=7)

    # Family-color legend
    families_present = list(dict.fromkeys(
        family_of(k) for k in txt_keys + img_keys
    ))
    from matplotlib.patches import Patch
    from ..registry import family_label
    legend_patches = [Patch(color=family_color(f), label=family_label(f)) for f in families_present]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(families_present),
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    out = fig_dir("cross")
    path = out / f"per_model_js_{args.judge}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
