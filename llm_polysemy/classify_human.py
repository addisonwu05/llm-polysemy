"""Classify the human Prolific responses with the same GPT + Gemini judges.

The human study used the same 100-word English panel as the LLM benchmark, in
two conditions:
  - "meaning" — participants used the word in a sentence  (analog of LLM *text*)
  - "image"   — participants described the image it evokes (analog of LLM *image*)

This reuses the whole classify engine: a converter first materializes
`data/human_responses.json` into the `data/outputs_human_<condition>/<word>/human/NN.txt`
file tree the runner expects, then `run_classify` labels each response by sense
with human-framed prompts. Output histograms (`histograms_human_<condition>.json`)
compare directly to `histograms_text.json` / `histograms_image.json`.

    python -m llm_polysemy.classify_human --convert                 # JSON -> file tree
    python -m llm_polysemy.classify_human --condition meaning       # classify (both judges)
    python -m llm_polysemy.classify_human --condition image --judge gpt --word bank --n 1
    python -m llm_polysemy.classify_human --histograms              # build both histograms
"""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from . import config, runner
from .config import JUDGES
from .histograms import build_histograms
from .meanings import load_meanings

HUMAN_JSON = Path("data/human_responses.json")
CONDITIONS = ["meaning", "image"]
MODEL_KEY = "human"   # single pseudo-"model" axis so the existing tree layout fits


def convert():
    """Materialize human_responses.json into the per-condition output file tree."""
    recs = json.loads(HUMAN_JSON.read_text())
    counters: dict = {}
    n = 0
    for r in recs:
        cond, word = r.get("prompt_id"), r.get("word")
        resp = (r.get("response") or "").strip()
        if cond not in CONDITIONS or not word or not resp:
            continue
        d = config.human_cls_modality(cond).output_dir / word / MODEL_KEY
        d.mkdir(parents=True, exist_ok=True)
        i = counters.get((cond, word), 0)
        counters[(cond, word)] = i + 1
        # name must match runner.artifact_path's f"{idx:02d}.txt" so resume round-trips
        (d / f"{i:02d}.txt").write_text(resp)
        n += 1
    print(f"materialized {n} human responses -> data/outputs_human_<condition>/")


def cli_main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--convert", action="store_true",
                   help="materialize human_responses.json into the file tree and exit")
    p.add_argument("--condition", choices=CONDITIONS + ["both"], default="both")
    p.add_argument("--judge", choices=list(JUDGES), default=None, help="one judge (default: both)")
    p.add_argument("--word", default=None, help="classify only this word")
    p.add_argument("--n", type=int, default=None, help="cap responses per (word) — cheap test slice")
    p.add_argument("--histograms", action="store_true", help="build histograms and exit")
    args = p.parse_args()

    if args.convert:
        convert()
        return

    conds = CONDITIONS if args.condition == "both" else [args.condition]
    meanings = load_meanings()  # English meanings.json — same panel as the human study
    judges = [args.judge] if args.judge else list(JUDGES)

    for cond in conds:
        mod = config.human_cls_modality(cond)
        if args.histograms:
            build_histograms(mod.class_file, mod.hist_file)
            continue
        words = [w for w in meanings if not args.word or w == args.word]
        print(f"\n=== classifying human '{cond}' ===")
        asyncio.run(runner.run_classify(mod, meanings, words, judges, n_cap=args.n))


if __name__ == "__main__":
    cli_main()
