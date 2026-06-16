"""Merged classification entrypoint.

    python -m llm_polysemy.classify --modality image
    python -m llm_polysemy.classify --modality text --word bank --n 1
    python -m llm_polysemy.classify --modality image --histograms

Two judges (GPT + Gemini) label each artifact by which sense of the word it uses,
then `--histograms` aggregates into per-(word, model, judge) sense histograms.
"""

import argparse
import asyncio

from dotenv import load_dotenv

from . import config, runner
from .config import JUDGES
from .histograms import build_histograms
from .meanings import load_meanings


def cli_main(default_modality: str | None = None):
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=["image", "text", "text-imagine"],
                   default=default_modality or "image")
    p.add_argument("--lang", choices=["en", "tr", "fr"], default="en",
                   help="which language's outputs + meanings to classify (default: en)")
    p.add_argument("--judge", choices=list(JUDGES), default=None,
                   help="run only this judge (default: both)")
    p.add_argument("--model", default=None, help="classify only this model_key")
    p.add_argument("--word", default=None, help="classify only this word")
    p.add_argument("--n", type=int, default=None,
                   help="cap samples per (word, model) — handy for a cheap test run")
    p.add_argument("--histograms", action="store_true",
                   help="build the histograms file from the classifications and exit")
    args = p.parse_args()

    mod = config.cls_modality(args.modality, args.lang)
    if args.histograms:
        build_histograms(mod.class_file, mod.hist_file)
        return

    meanings = load_meanings(mod.meanings_file)
    words = [w for w in meanings if not args.word or w == args.word]
    judges = [args.judge] if args.judge else list(JUDGES)
    asyncio.run(runner.run_classify(mod, meanings, words, judges,
                                    model_filter=args.model, n_cap=args.n))


if __name__ == "__main__":
    cli_main()
