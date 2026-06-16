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
from .registry import models_for, family_of

WORDS_FILES = {
    "en": Path("data/synonyms.txt"),
    "tr": Path("data/synonyms_tr.txt"),
    "fr": Path("data/synonyms_fr.txt"),
}

# Families skipped for a given language's default ("all models") run. Turkish
# drops Qwen + FLUX (weak/uninteresting on Turkish). An explicit --model still runs.
LANG_EXCLUDE_FAMILIES = {
    "tr": {"qwen", "flux"},
    "fr": {"qwen", "flux"},
}


def load_words(lang: str = "en") -> list[str]:
    return [l.strip() for l in WORDS_FILES[lang].read_text().splitlines() if l.strip()]


def cli_main(default_modality: str | None = None):
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=["image", "text", "text-imagine"],
                   default=default_modality or "image")
    p.add_argument("--lang", choices=["en", "tr", "fr"], default="en",
                   help="word list + generation-prompt language (default: en)")
    p.add_argument("--model", default=None, help="run only this model_key (default: all)")
    args = p.parse_args()

    if args.modality == "text-imagine" and args.lang != "en":
        p.error("--modality text-imagine is English only")

    # text-imagine reuses the text models (it is a prompt variant, not a new modality).
    keys = models_for("text" if args.modality == "text-imagine" else args.modality)
    if args.model:
        if args.model not in keys:
            p.error(f"--model {args.model!r} is not a {args.modality} model; "
                    f"choose from: {', '.join(keys)}")
        keys = [args.model]
    else:
        exclude = LANG_EXCLUDE_FAMILIES.get(args.lang, set())
        keys = [k for k in keys if family_of(k) not in exclude]

    words = load_words(args.lang)[:100]
    asyncio.run(runner.run_generation(config.gen_modality(args.modality, args.lang), keys, words))


if __name__ == "__main__":
    cli_main()
