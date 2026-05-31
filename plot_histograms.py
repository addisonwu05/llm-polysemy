import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

data = json.load(open("histograms.json"))
meanings = json.load(open("meanings.json"))

os.makedirs("plots", exist_ok=True)

MODELS = [
    "flux", "flux_1pro", "flux_flex", "flux_max",
    "gemini", "gemini_25", "gemini_pro",
    "grok", "grok_base", "grok_pro",
    "openai", "openai_1", "openai_15", "openai_mini",
    "qwen", "qwen_1", "qwen_20",
]

# Sense colors: fixed senses get a color, "unclear" and "multiple" always gray/dark
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]


def get_senses(word):
    raw = meanings.get(word, [])
    if isinstance(raw, list):
        return [s["sense"] for s in raw]
    return list(raw.keys())


for word, word_data in data.items():
    senses = get_senses(word)
    all_labels = senses + ["multiple", "unclear"]

    color_map = {}
    for i, s in enumerate(senses):
        color_map[s] = PALETTE[i % len(PALETTE)]
    color_map["multiple"] = "#BBBBBB"
    color_map["unclear"] = "#666666"

    # Collect counts per model (GPT classifier only)
    model_counts = {}
    for model in MODELS:
        if model not in word_data:
            continue
        counts = word_data[model]["gpt"]["_counts"]
        total = sum(counts.values()) or 1
        model_counts[model] = {label: counts.get(label, 0) / total for label in all_labels}

    if not model_counts:
        continue

    present_models = list(model_counts.keys())
    x = np.arange(len(present_models))
    width = 0.65

    fig, ax = plt.subplots(figsize=(max(10, len(present_models) * 0.75), 4.5))

    bottoms = np.zeros(len(present_models))
    for label in all_labels:
        vals = np.array([model_counts[m][label] for m in present_models])
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, width, bottom=bottoms, color=color_map[label], label=label)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(present_models, rotation=40, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion of images")
    ax.set_title(f'"{word}" — sense distribution per model (GPT classifier)', fontsize=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    plt.tight_layout()
    plt.savefig(f"plots/{word}.png", dpi=120)
    plt.close()
    print(f"saved plots/{word}.png")

print("done")
