#!/usr/bin/env python3
"""Shim — text classification. See `llm_polysemy/classify.py`.

    python classify_text.py [--histograms] [--word W] [--n N]   # text (default)
    python -m llm_polysemy.classify --modality text             # canonical merged form
"""
from llm_polysemy.classify import cli_main

if __name__ == "__main__":
    cli_main(default_modality="text")
