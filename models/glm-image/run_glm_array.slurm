#!/bin/bash
# Parallel GLM-Image generation: ONE JOB PER WORD (100-task array), one word's 30
# images per GPU. Resume is per-file so tasks never conflict.
#   sbatch models/glm-image/run_glm_array.slurm
#SBATCH --job-name=glm-arr
#SBATCH --account=griffith
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-99
#SBATCH --output=models/glm-image/arr-%x-%A_%a.out
set -e
NSHARDS=100
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")"
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glm
echo "=== glm-image word#$SHARD :: $(date) :: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -m llm_polysemy.gen_glm_image --shard "$SHARD" --num-shards "$NSHARDS"
echo "=== word#$SHARD done :: $(date) ==="
