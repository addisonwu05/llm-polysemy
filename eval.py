#!/usr/bin/env python3
"""Shim — image generation. See `llm_polysemy/eval.py`.

    python eval.py [--model KEY]                 # image (this shim's default)
    python -m llm_polysemy.eval --modality text  # canonical merged form
"""
from llm_polysemy.eval import cli_main

if __name__ == "__main__":
    cli_main(default_modality="image")
