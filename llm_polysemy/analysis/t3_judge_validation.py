"""T3 judge validation: sample (word, model, idx) cells for human labeling,
then compute Cohen's κ and check whether judge error is modality-balanced.

Two-step usage:

  Step 1 — generate sample for human labeling:
      python -m llm_polysemy.analysis.t3_judge_validation --sample
      # writes data/t3_human_sample.json — fill in 'human_label' for each row

  Step 2 — compute κ and modality balance report:
      python -m llm_polysemy.analysis.t3_judge_validation --report

The sample is stratified to over-represent 'unclear' and 'multiple' labels (the
most consequential error cases) and balanced across image/text modalities.

For image rows, the corresponding PNG path is included so you can open the file.
For text rows, the sentence is included inline.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..meanings import load_meanings

SAMPLE_SIZE = 180        # total cells to hand-label
AMBIG_OVERSAMPLE = 0.40  # fraction from unclear/multiple/other cells
MODALITY_BALANCE = 0.50  # fraction from each modality (before rounding)
SEED = 42


def load_classifications(path: str) -> list[dict]:
    """Flatten classifications JSON to a list of cell dicts."""
    raw = json.loads(Path(path).read_text())
    cells = []
    for word, models in raw.items():
        for model_key, idxs in models.items():
            for idx, judges in idxs.items():
                gpt_label = _label_of(judges.get("gpt"))
                gemini_label = _label_of(judges.get("gemini"))
                if gpt_label is None:
                    continue
                cells.append({
                    "word": word,
                    "model_key": model_key,
                    "idx": int(idx),
                    "gpt_label": gpt_label,
                    "gemini_label": gemini_label,
                    "agree": gpt_label == gemini_label,
                })
    return cells


def _label_of(verdict) -> str | None:
    if verdict is None:
        return None
    return verdict["label"] if isinstance(verdict, dict) else verdict


def stratified_sample(cells: list[dict], n: int, rng: random.Random) -> list[dict]:
    """
    Sample n cells:
      - ~40% from ambiguous (unclear/multiple/other) by GPT label
      - ~60% from clear senses
    Shuffle to avoid word/model clusters.
    """
    ambiguous = {"unclear", "multiple", "other"}
    ambig_cells = [c for c in cells if c["gpt_label"] in ambiguous]
    clear_cells  = [c for c in cells if c["gpt_label"] not in ambiguous]

    n_ambig = min(int(n * AMBIG_OVERSAMPLE), len(ambig_cells))
    n_clear = min(n - n_ambig, len(clear_cells))

    sample = rng.sample(ambig_cells, n_ambig) + rng.sample(clear_cells, n_clear)
    rng.shuffle(sample)
    return sample


def enrich_with_content(cells: list[dict], modality: str) -> list[dict]:
    """Add output path (image) or sentence text (text) to each cell."""
    for cell in cells:
        if modality == "image":
            path = (Path("data/outputs_image") / cell["word"] / cell["model_key"]
                    / f"{cell['idx']:02d}.png")
            cell["content"] = str(path) if path.exists() else None
            cell["content_type"] = "image_path"
        else:
            path = (Path("data/outputs_text") / cell["word"] / cell["model_key"]
                    / f"{cell['idx']:02d}.txt")
            if path.exists():
                cell["content"] = path.read_text().strip()
            else:
                cell["content"] = None
            cell["content_type"] = "sentence"
        cell["human_label"] = None
    return cells


def generate_sample():
    rng = random.Random(SEED)
    meanings = load_meanings()

    all_cells = []
    for modality in ("image", "text"):
        cls_path = f"data/classifications_{modality}.json"
        if not Path(cls_path).exists():
            print(f"  Skipping {modality} — {cls_path} not found")
            continue
        cells = load_classifications(cls_path)
        n_mod = int(SAMPLE_SIZE * MODALITY_BALANCE)
        sample = stratified_sample(cells, n_mod, rng)
        sample = enrich_with_content(sample, modality)
        for cell in sample:
            cell["modality"] = modality
            cell["word_senses"] = [e["sense"] for e in meanings.get(cell["word"], [])]
        all_cells.extend(sample)

    rng.shuffle(all_cells)

    out = Path("data/t3_human_sample.json")
    out.write_text(json.dumps(all_cells, indent=2))
    print(f"\nWrote {len(all_cells)} cells to {out}")

    by_mod = Counter(c["modality"] for c in all_cells)
    by_label = Counter(c["gpt_label"] for c in all_cells)
    print(f"By modality: {dict(by_mod)}")
    print(f"By GPT label (top 10): {dict(by_label.most_common(10))}")
    print(f"\nFill in 'human_label' for each row, then run:")
    print(f"  python -m llm_polysemy.analysis.t3_judge_validation --report")


def cohen_kappa(a_labels, b_labels, categories):
    n = len(a_labels)
    p_obs = sum(a == b for a, b in zip(a_labels, b_labels)) / n
    a_counts = Counter(a_labels)
    b_counts = Counter(b_labels)
    p_exp = sum((a_counts[k] / n) * (b_counts[k] / n) for k in categories)
    return (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else 0.0


def compute_report():
    p = Path("data/t3_human_sample.json")
    if not p.exists():
        raise SystemExit("Run --sample first to generate the sample file.")

    cells = json.loads(p.read_text())
    labeled = [c for c in cells if c.get("human_label") is not None]
    if not labeled:
        raise SystemExit("No human labels found. Fill in 'human_label' in the sample file.")

    print(f"Labeled cells: {len(labeled)} / {len(cells)}")
    categories = sorted({c["gpt_label"] for c in labeled} |
                        {c["human_label"] for c in labeled})

    gpt_labels = [c["gpt_label"] for c in labeled]
    human_labels = [c["human_label"] for c in labeled]
    kappa_overall = cohen_kappa(gpt_labels, human_labels, categories)

    print(f"\n=== Overall Judge Validation ===")
    print(f"GPT vs Human")
    print(f"  Observed agreement: {sum(g==h for g,h in zip(gpt_labels, human_labels))/len(labeled):.3f}")
    print(f"  Cohen's κ:          {kappa_overall:.3f}")
    if kappa_overall >= 0.60:
        print(f"  ✓ κ ≥ 0.60 — judge quality sufficient per T1_rule.md §5b")
    else:
        print(f"  ✗ κ < 0.60 — review depictability rating procedure")

    # Per-modality breakdown — critical for T3 acceptance criteria
    print(f"\n=== Per-Modality Breakdown (critical: must be balanced) ===")
    for modality in ("image", "text"):
        mod_cells = [c for c in labeled if c["modality"] == modality]
        if not mod_cells:
            print(f"  {modality}: no labeled cells")
            continue
        g = [c["gpt_label"] for c in mod_cells]
        h = [c["human_label"] for c in mod_cells]
        kappa = cohen_kappa(g, h, categories)
        agree = sum(a == b for a, b in zip(g, h)) / len(mod_cells)
        print(f"  {modality:<5}: n={len(mod_cells):3d}  agreement={agree:.3f}  κ={kappa:.3f}")

    # Check: is error rate modality-balanced?
    img_cells = [c for c in labeled if c["modality"] == "image"]
    txt_cells = [c for c in labeled if c["modality"] == "text"]
    if img_cells and txt_cells:
        img_err = sum(c["gpt_label"] != c["human_label"] for c in img_cells) / len(img_cells)
        txt_err = sum(c["gpt_label"] != c["human_label"] for c in txt_cells) / len(txt_cells)
        diff = abs(img_err - txt_err)
        print(f"\n  Error rate image: {img_err:.3f}   text: {txt_err:.3f}   |diff|={diff:.3f}")
        if diff <= 0.05:
            print("  ✓ Error rate balanced across modalities (|diff| ≤ 0.05)")
        elif diff <= 0.10:
            print("  ⚠ Modest imbalance (0.05 < |diff| ≤ 0.10) — note in paper")
        else:
            print("  ✗ Substantial imbalance (|diff| > 0.10) — modality effect may be artifact")

    # Unclear/multiple cell accuracy — most important to check
    print(f"\n=== Accuracy on 'unclear'/'multiple' Cells ===")
    ambig_cells = [c for c in labeled if c["gpt_label"] in ("unclear", "multiple")]
    if ambig_cells:
        g = [c["gpt_label"] for c in ambig_cells]
        h = [c["human_label"] for c in ambig_cells]
        agree = sum(a == b for a, b in zip(g, h)) / len(ambig_cells)
        print(f"  n={len(ambig_cells)}  GPT-human agreement on gray cells: {agree:.3f}")
    else:
        print("  No unclear/multiple cells in sample.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true",
                   help="Generate human-labeling sample")
    p.add_argument("--report", action="store_true",
                   help="Compute κ and modality-balance report from filled sample")
    args = p.parse_args()

    if args.sample:
        generate_sample()
    elif args.report:
        compute_report()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
