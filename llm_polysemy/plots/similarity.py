"""Model similarity matrix (1 − mean JS divergence) + family-cohesion analysis."""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list

from ..plotting_utils import (base_parser, require_data, models_for, get_vocab,
                              dist_for, js, fig_dir, JUDGE_DISPLAY)
from ..registry import display_label, family_of, family_label, family_color, families


def main():
    args = base_parser(__doc__).parse_args()
    data, meanings = require_data(args.modality)
    MODELS = models_for(args.modality)
    out = fig_dir(args.modality)
    suffix = "" if args.modality == "image" else "_text"
    n = len(MODELS)

    sum_js = np.zeros((n, n))
    word_count = np.zeros((n, n))
    for word, word_data in data.items():
        vocab = get_vocab(word, word_data, meanings, args.judge)
        dists = {i: d for i, m in enumerate(MODELS) if m in word_data
                 and (d := dist_for(word_data[m], vocab, args.judge)) is not None}
        for i in dists:
            for j in dists:
                if i >= j:
                    continue
                v = js(dists[i], dists[j])
                sum_js[i, j] += v; sum_js[j, i] += v
                word_count[i, j] += 1; word_count[j, i] += 1

    with np.errstate(invalid="ignore"):
        avg_js = np.where(word_count > 0, sum_js / word_count, np.nan)
    np.fill_diagonal(avg_js, 0.0)
    similarity = 1 - avg_js

    nan_mask = np.isnan(avg_js)
    avg_js_filled = np.where(nan_mask, np.nanmean(avg_js), avg_js)
    np.fill_diagonal(avg_js_filled, 0.0)
    Z = linkage(avg_js_filled[np.triu_indices(n, k=1)], method="average")
    order = leaves_list(Z)
    sim_ordered = similarity[np.ix_(order, order)]
    tick_labels = [display_label(MODELS[i]) for i in order]

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(sim_ordered, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="JS similarity  (1 − JS divergence, log₂)")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            val = sim_ordered[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if val > 0.65 else "white")
    ax.set_title(f"Model similarity matrix\n(1 − mean JS divergence over words, "
                 f"{JUDGE_DISPLAY[args.judge]}, hierarchically clustered)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out / f"model_similarity{suffix}.png", dpi=150, bbox_inches="tight")
    print(f"saved {out}/model_similarity{suffix}.png")

    pairs = sorted((avg_js[i, j], MODELS[i], MODELS[j])
                   for i in range(n) for j in range(i + 1, n))
    print("\nMost similar pairs (lowest JS distance):")
    for d, a, b in pairs[:5]:
        print(f"  {display_label(a):20s}  {display_label(b):20s}  JS={d:.4f}")
    print("\nMost different pairs (highest JS distance):")
    for d, a, b in pairs[-5:]:
        print(f"  {display_label(a):20s}  {display_label(b):20s}  JS={d:.4f}")

    # ── Family cohesion ───────────────────────────────────────────────────────
    fams = families(args.modality)
    rows, within_all, between_all = [], [], []
    for fam in fams:
        fam_idx = [i for i, m in enumerate(MODELS) if family_of(m) == fam]
        other_idx = [i for i, m in enumerate(MODELS) if family_of(m) != fam]
        within = [avg_js[i, j] for i, j in combinations(fam_idx, 2) if not np.isnan(avg_js[i, j])]
        between = [avg_js[i, j] for i in fam_idx for j in other_idx if not np.isnan(avg_js[i, j])]
        w = np.mean(within) if within else np.nan
        b = np.mean(between) if between else np.nan
        rows.append((fam, w, b, b / w if w else np.nan))
        within_all.extend(within); between_all.extend(between)
    w_all, b_all = np.mean(within_all), np.mean(between_all)
    overall_ratio = b_all / w_all

    print(f"\n{'─'*52}\nFamily cohesion  (mean pairwise JS distance)")
    print(f"{'Family':<10}  {'within':>8}  {'between':>8}  {'ratio':>7}\n{'─'*52}")
    for fam, w, b, r in rows:
        print(f"{family_label(fam):<10}  {w:8.4f}  {b:8.4f}  {r:6.2f}x")
    print(f"{'─'*52}\n{'OVERALL':<10}  {w_all:8.4f}  {b_all:8.4f}  {overall_ratio:6.2f}x")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(fams))
    colors = [family_color(f) for f in fams]
    ax2.bar(x - 0.175, [r[1] for r in rows], 0.35, color=colors, alpha=0.9, label="within family")
    ax2.bar(x + 0.175, [r[2] for r in rows], 0.35, color=colors, alpha=0.4, label="between families")
    ax2.axhline(w_all, color="gray", linewidth=1, linestyle="--", alpha=0.6)
    ax2.axhline(b_all, color="gray", linewidth=1, linestyle=":", alpha=0.6)
    for i, (fam, w, b, r) in enumerate(rows):
        ax2.text(i, max(w, b) + 0.008, f"{r:.2f}×", ha="center", fontsize=9,
                 fontweight="bold", color=family_color(fam))
    ax2.set_xticks(x); ax2.set_xticklabels([family_label(f) for f in fams], fontsize=11)
    ax2.set_ylabel("Mean pairwise JS divergence (log₂)")
    ax2.set_title(f"Family cohesion: within- vs between-family JS distance\n"
                  f"Overall: models are {overall_ratio:.2f}× closer to their own family", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, max(r[2] for r in rows) * 1.18)
    plt.tight_layout()
    plt.savefig(out / f"family_cohesion{suffix}.png", dpi=150, bbox_inches="tight")
    print(f"\nsaved {out}/family_cohesion{suffix}.png")


if __name__ == "__main__":
    main()
