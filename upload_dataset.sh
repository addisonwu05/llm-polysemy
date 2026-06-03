#!/bin/bash
set -e

# Upload a generated corpus to Hugging Face.
#   ./upload_dataset.sh           # image outputs (default)
#   ./upload_dataset.sh text      # text outputs
MODALITY="${1:-image}"
HF_REPO="addisonwu05/llm-polysemy-outputs"
DIR="data/outputs_${MODALITY}"

echo "Uploading $DIR to Hugging Face dataset: $HF_REPO"
hf upload-large-folder "$HF_REPO" "$DIR" --repo-type=dataset --path-in-repo "outputs_${MODALITY}"

echo "Done. View at: https://huggingface.co/datasets/$HF_REPO"
