"""T2 analysis: Does the 'superposition' (multiple-sense) rate correlate with
image model architecture (diffusion vs. other)?

Usage:
    python -m llm_polysemy.analysis.t2_architecture
    python -m llm_polysemy.analysis.t2_architecture --out figures/t2_superposition.pdf

Reads:
    data/histograms_image.json
    data/model_architectures.json

Superposition signal: fraction of samples labeled 'multiple' by either judge,
averaged across all words for that model.

Architecture labels: data/model_architectures.json (see that file for honesty about
closed models — only FLUX is confirmed diffusion at lock time).
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from ..registry import MODELS, models_for, display_label, family_color

AMBIGUOUS = {"multiple", "unclear", "other"}


def load_json(path):
    return json.loads(Path(path).read_text())


def compute_superposition_rates(hist_img: dict) -> dict[str, dict]:
    """
    For each image model, compute fraction of samples labeled 'multiple'
    (using the 'gpt' judge, falling back to first available judge).

    Returns {model_key: {"multiple_rate": float, "n_samples": int, "n_words": int}}
    """
    rates = defaultdict(lambda: {"multiple": 0, "total": 0, "words": set()})

    for word, models in hist_img.items():
        for model_key, model_data in models.items():
            # Get primary judge counts
            primary = model_data.get("gpt") or next(
                (v for k, v in model_data.items() if k != "_agreement"), None
            )
            if primary is None:
                continue
            counts = primary.get("_counts", {})
            total = sum(counts.values())
            multiple = counts.get("multiple", 0)
            if total > 0:
                rates[model_key]["multiple"] += multiple
                rates[model_key]["total"] += total
                rates[model_key]["words"].add(word)

    result = {}
    for model_key, d in rates.items():
        if d["total"] > 0:
            result[model_key] = {
                "multiple_rate": d["multiple"] / d["total"],
                "n_samples": d["total"],
                "n_words": len(d["words"]),
            }
    return result


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return (max(0, center - half), min(1, center + half))


def print_table(rates: dict, archs: dict):
    print("\n=== Superposition (multiple-sense) Rates by Image Model ===\n")
    print(f"{'Model':<20} {'Arch':<14} {'Multiple%':>10}  {'n':>6}  CI 95%")
    print("-" * 65)

    for key in models_for("image"):
        if key not in rates:
            continue
        r = rates[key]
        arch = archs.get(key, {}).get("arch", "unknown")
        label = display_label(key)
        n = r["n_samples"]
        k = int(r["multiple_rate"] * n)
        lo, hi = wilson_ci(k, n)
        print(f"{label:<20} {arch:<14} {r['multiple_rate']:>9.1%}  {n:>6}  "
              f"[{lo:.3f}–{hi:.3f}]")

    # Summary by architecture
    print("\n=== Mean Superposition Rate by Architecture ===\n")
    by_arch = defaultdict(list)
    for key, r in rates.items():
        arch = archs.get(key, {}).get("arch", "unknown")
        by_arch[arch].append(r["multiple_rate"])

    for arch, vals in sorted(by_arch.items()):
        mean = np.mean(vals)
        sd = np.std(vals)
        print(f"  {arch:<14}: n={len(vals):2d} models  mean={mean:.1%}  sd={sd:.3f}")

    # Point-biserial correlation (if we have diffusion + unknown)
    all_archs = {key: archs.get(key, {}).get("arch", "unknown") for key in rates}
    diffusion_rates = [rates[k]["multiple_rate"] for k, a in all_archs.items()
                       if a == "diffusion"]
    other_rates = [rates[k]["multiple_rate"] for k, a in all_archs.items()
                   if a != "diffusion"]
    if diffusion_rates and other_rates:
        from scipy import stats
        stat, p = stats.mannwhitneyu(diffusion_rates, other_rates,
                                     alternative="greater")
        print(f"\nMann-Whitney U (diffusion > other): U={stat:.0f}, p={p:.4f}")
        print("  (Note: only FLUX confirmed diffusion at lock time — treat with caution)")


def plot_figure(rates: dict, archs: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))

    arch_colors = {
        "diffusion":      "#4C72B0",
        "autoregressive": "#DD8452",
        "unknown":        "#AAAAAA",
    }
    arch_labels = {
        "diffusion":      "Diffusion (confirmed)",
        "autoregressive": "Autoregressive",
        "unknown":        "Unknown / closed",
    }

    keys_ordered = [k for k in models_for("image") if k in rates]
    x_pos = np.arange(len(keys_ordered))
    bar_colors = [arch_colors[archs.get(k, {}).get("arch", "unknown")]
                  for k in keys_ordered]
    bar_vals = [rates[k]["multiple_rate"] for k in keys_ordered]
    bar_n = [rates[k]["n_samples"] for k in keys_ordered]
    bar_labels = [display_label(k) for k in keys_ordered]

    ci_lo = []
    ci_hi = []
    for k in keys_ordered:
        n = rates[k]["n_samples"]
        mult = int(rates[k]["multiple_rate"] * n)
        lo, hi = wilson_ci(mult, n)
        ci_lo.append(rates[k]["multiple_rate"] - lo)
        ci_hi.append(hi - rates[k]["multiple_rate"])

    bars = ax.bar(x_pos, bar_vals, color=bar_colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ax.errorbar(x_pos, bar_vals, yerr=[ci_lo, ci_hi], fmt="none",
                color="#333333", capsize=3, linewidth=1)

    # Sample size annotations on bars where n < 300
    for i, (n, v) in enumerate(zip(bar_n, bar_vals)):
        if n < 300:
            ax.text(i, v + 0.005, f"n={n}", ha="center", va="bottom",
                    fontsize=6.5, color="#555555")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of samples labeled 'multiple'", fontsize=11)
    ax.set_title("Superposition Rate by Image Model\n"
                 "(Does 'both-senses' behavior cluster by architecture?)", fontsize=12)
    ax.set_ylim(0, min(1.0, max(bar_vals) * 1.35 + 0.05))

    # Legend
    handles = [mpatches.Patch(color=arch_colors[a], label=arch_labels[a], alpha=0.85)
               for a in ("diffusion", "autoregressive", "unknown")
               if any(archs.get(k, {}).get("arch") == a for k in keys_ordered)]
    ax.legend(handles=handles, fontsize=9, loc="upper right")

    # Family separator lines
    from ..registry import family_groups
    fg = family_groups("image")
    x = -0.5
    for family, fkeys in fg.items():
        x += len(fkeys)
        ax.axvline(x, color="#cccccc", lw=0.8, zorder=0)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures/t2_superposition.pdf")
    args = p.parse_args()

    hist_img = load_json("data/histograms_image.json")
    archs = load_json("data/model_architectures.json")

    rates = compute_superposition_rates(hist_img)

    print_table(rates, archs)

    try:
        plot_figure(rates, archs, args.out)
    except ImportError as e:
        print(f"(Skipping plot: {e})")


if __name__ == "__main__":
    main()
