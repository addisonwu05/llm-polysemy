#!/bin/bash
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")"
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CLASSIFY_ENV:?set CLASSIFY_ENV to a conda env with the classify deps (openai / google-genai / python-dotenv)}"
for KEY in simplear_sft simplear_rl; do
  echo "=== CLASSIFY(bulk) $KEY :: $(date) ==="
  python -m llm_polysemy.classify --modality image --model "$KEY" 2>&1 | grep -vE "it/s\]" | tail -15
done
echo "=== BULK CLASSIFY PASS DONE :: $(date) ==="
