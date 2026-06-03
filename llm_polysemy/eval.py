"""Merged generation entrypoint.

    python -m llm_polysemy.eval --modality image          # all image models
    python -m llm_polysemy.eval --modality text --model openai_55

30 samples/word/model; the prompt is the bare word (image) or the one-sentence
instruction (text). Resume is filesystem-driven — missing artifacts are regenerated.
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from . import config, runner
from .registry import models_for

WORDS_FILE = Path("synonyms.txt")


def load_words() -> list[str]:
    return [l.strip() for l in WORDS_FILE.read_text().splitlines() if l.strip()]


def cli_main(default_modality: str | None = None):
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=["image", "text"],
                   default=default_modality or "image")
    p.add_argument("--model", default=None, help="run only this model_key (default: all)")
    args = p.parse_args()

    keys = models_for(args.modality)
    if args.model:
        if args.model not in keys:
            p.error(f"--model {args.model!r} is not a {args.modality} model; "
                    f"choose from: {', '.join(keys)}")
        keys = [args.model]

    words = load_words()[:100]
    asyncio.run(runner.run_generation(config.GEN[args.modality], keys, words))


if __name__ == "__main__":
    cli_main()
