#!/usr/bin/env python3
"""Shim — inter-judge agreement. See `llm_polysemy/agreement.py`.

    python agreement.py [--modality {image,text}]
"""
from llm_polysemy.agreement import cli_main

if __name__ == "__main__":
    cli_main(default_modality="image")
