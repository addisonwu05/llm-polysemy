"""Does text-prior alignment change across model generations within each family?
JS(text_freq || model) vs generation index, one line per family, with regression."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from ..plotting_utils import (base_parser, require_data, get_vocab, dist_for,
                              text_dist, js, load_prior, fig_dir, JUDGE_DISPLAY)
from ..registry import generation_order, family_label, family_color


def main():
    p = base_parser(__doc__)
    p.add_argument("--prior", choices=["manual", "wiki", "trends"], default="manual")
    args = p.parse_args()
    data, meanings = require_data(args.modality)
    freq = load_prior(args.prior)
    gen_order = generation_order(args.modality)
    out = fig_dir(args.modality)
    suffix = "" if args.modality == "image" else "_text"

    # per-model mean JS from prior
    model_js = {}
    for fam, models in gen_order.items():
        for key, gen, label in models:
            vals = []
            for word, word_data in data.items():
                if key not in word_data:
                    continue
                vocab = get_vocab(word, word_data, meanings, args.judge)
                tf = text_dist(word, vocab, freq)
                d = dist_for(word_data[key], vocab, args.judge)
                if tf is None or d is None:
                    continue
                v = js(tf, d)
                if not np.isnan(v):
                    vals.append(v)
            if vals:
                model_js[key] = np.mean(vals)

    fams = list(gen_order)
    fig, axes = plt.subplots(1, len(fams), figsize=(18, 5), sharey=True)
    fig.suptitle(f"Does text-prior alignment change across model generations? ({JUDGE_DISPLAY[args.judge]})\n"
                 "(lower JS = closer to text-frequency prior; lines connect consecutive versions)", fontsize=11)

    all_gens, all_js = [], []
    for ax, fam in zip(axes, fams):
        color = family_color(fam)
        triples = [(gen, model_js[k], lbl) for k, gen, lbl in gen_order[fam] if k in model_js]
        if not triples:
            ax.set_visible(False)
            continue
        gens, jsv, labels = zip(*triples)
        ax.plot(gens, jsv, color=color, linewidth=2, marker="o", markersize=9, zorder=3)
        for g, j, lbl in zip(gens, jsv, labels):
            ax.text(g, j + 0.004, lbl, ha="center", va="bottom", fontsize=8, color=color)
        if len(set(gens)) >= 3:
            coeffs = np.polyfit(gens, jsv, 1)
            xs = np.linspace(min(gens) - 0.2, max(gens) + 0.2, 50)
            ax.plot(xs, np.polyval(coeffs, xs), color=color, linewidth=1, linestyle="--", alpha=0.5)
            r, pv = pearsonr(gens, jsv)
            ax.text(0.97, 0.97, f"r = {r:+.2f}\n(p={pv:.2f})", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color=color)
        all_gens.extend(gens); all_js.extend(jsv)
        ax.set_title(family_label(fam), fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Generation (within family)", fontsize=9)
        ax.set_xticks(sorted(set(gens)))
        ax.grid(axis="y", alpha=0.3)

    # data-driven y-limits with a little padding
    if all_js:
        lo, hi = min(all_js), max(all_js)
        pad = (hi - lo) * 0.15 or 0.05
        axes[0].set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel("Mean JS divergence from text-freq prior (log₂)")

    ax_inset = axes[-1].inset_axes([0.55, 0.05, 0.42, 0.40])
    ax_inset.scatter(all_gens, all_js, color="gray", s=30, zorder=3)
    if len(set(all_gens)) >= 3:
        coeffs = np.polyfit(all_gens, all_js, 1)
        xs = np.linspace(min(all_gens) - 0.1, max(all_gens) + 0.1, 50)
        ax_inset.plot(xs, np.polyval(coeffs, xs), color="black", linewidth=1.2)
        r, pv = pearsonr(all_gens, all_js)
        ax_inset.set_title(f"all families\nr={r:+.2f} p={pv:.2f}", fontsize=7)
    ax_inset.set_xlabel("gen", fontsize=7); ax_inset.tick_params(labelsize=6)

    plt.tight_layout()
    path = out / f"model_generation_trend{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
