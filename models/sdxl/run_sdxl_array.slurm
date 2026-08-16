#!/bin/bash
# SDXL per-word array. Pass the variant (base|dpo) as $1.
#   sbatch models/sdxl/run_sdxl_array.slurm base
#   sbatch models/sdxl/run_sdxl_array.slurm dpo
#SBATCH --job-name=sdxl-arr
#SBATCH --account=griffith
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-99
#SBATCH --output=models/sdxl/arr-%x-%A_%a.out
set -e
VARIANT="${1:-base}"
NSHARDS=100
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")"
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glm
echo "=== sdxl:$VARIANT word#$SHARD :: $(date) :: $(hostname) ==="
python -m llm_polysemy.gen_sdxl --variant "$VARIANT" --shard "$SHARD" --num-shards "$NSHARDS"
echo "=== done :: $(date) ==="
