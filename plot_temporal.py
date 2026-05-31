import json
import os
import matplotlib.pyplot as plt
import numpy as np

data = json.load(open("histograms.json"))
meanings = json.load(open("meanings.json"))

os.makedirs("plots_temporal", exist_ok=True)

# Families in chronological/version order, with display labels
FAMILIES = {
    "OpenAI": {
        "models": ["openai_1", "openai_15", "openai_mini", "openai"],
        "labels": ["gpt-image-1", "gpt-image-1.5", "gpt-image-1-mini", "gpt-image-2"],
    },
    "Grok": {
        "models": ["grok_base", "grok_pro", "grok"],
        "labels": ["imagine-base", "imagine-pro", "imagine-quality"],
    },
    "Gemini": {
        "models": ["gemini_25", "gemini", "gemini_pro"],
        "labels": ["2.5-flash", "3.1-flash", "3-pro"],
    },
    "Qwen": {
        "models": ["qwen_1", "qwen_20", "qwen"],
        "labels": ["Qwen-Image v1", "Qwen-Image 2.0", "Qwen-Image 2.0-Pro"],
    },
    "FLUX": {
        "models": ["flux_1pro", "flux_flex", "flux", "flux_max"],
        "labels": ["1.1-pro", "2-flex", "2-pro", "2-max"],
    },
}

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#CCB974",
]


def get_senses(word):
    raw = meanings.get(word, [])
    if isinstance(raw, list):
        return [s["sense"] for s in raw]
    return list(raw.keys())


for word, word_data in data.items():
    senses = get_senses(word)
    all_labels = senses + ["multiple", "unclear"]

    color_map = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(senses)}
    color_map["multiple"] = "#BBBBBB"
    color_map["unclear"] = "#555555"

    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(18, 4), sharey=True)
    fig.suptitle(f'"{word}" — sense distribution over model generations (GPT classifier)', fontsize=12)

    for ax, (family_name, family) in zip(axes, FAMILIES.items()):
        models = family["models"]
        labels = family["labels"]

        # only keep models present in the data
        present = [(m, l) for m, l in zip(models, labels) if m in word_data]
        if not present:
            ax.set_visible(False)
            continue

        present_models, present_labels = zip(*present)
        x = np.arange(len(present_models))
        bottoms = np.zeros(len(present_models))

        for sense in all_labels:
            vals = np.array([
                word_data[m]["gpt"]["_counts"].get(sense, 0) / max(sum(word_data[m]["gpt"]["_counts"].values()), 1)
                for m in present_models
            ])
            if vals.sum() == 0:
                continue
            ax.bar(x, vals, 0.6, bottom=bottoms, color=color_map[sense], label=sense)
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(present_labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(family_name, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)
        if ax == axes[0]:
            ax.set_ylabel("Proportion of images")

    # Single legend from the last visible axis
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[s]) for s in all_labels if s in color_map]
    labels_leg = [s for s in all_labels if s in color_map]
    fig.legend(handles, labels_leg, loc="lower center", ncol=len(labels_leg),
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(f"plots_temporal/{word}.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"saved plots_temporal/{word}.png")

print("done")
