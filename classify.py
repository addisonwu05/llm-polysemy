#!/usr/bin/env python3
"""Shim — image classification. See `llm_polysemy/classify.py`.

    python classify.py [--histograms] [--word W] [--n N]   # image (default)
    python -m llm_polysemy.classify --modality text        # canonical merged form
"""
from llm_polysemy.classify import cli_main

if __name__ == "__main__":
    cli_main(default_modality="image")
