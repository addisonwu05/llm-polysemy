#!/bin/bash
set -e

HF_REPO="addisonwu05/llm-polysemy-outputs"

echo "Uploading outputs/ to Hugging Face dataset: $HF_REPO"
hf upload-large-folder "$HF_REPO" ./outputs --repo-type=dataset

echo "Done. View at: https://huggingface.co/datasets/$HF_REPO"
