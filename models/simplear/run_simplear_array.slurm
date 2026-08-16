#!/bin/bash
# SimpleAR per-word array. Pass the variant (sft|rl) as $1.
#   sbatch models/simplear/run_simplear_array.slurm sft
#   sbatch models/simplear/run_simplear_array.slurm rl
#SBATCH --job-name=simplear-arr
#SBATCH --account=griffith
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-99
#SBATCH --output=models/simplear/arr-%x-%A_%a.out
set -e
VARIANT="${1:-rl}"
NSHARDS=100
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")"
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate simplear
echo "=== simplear:$VARIANT word#$SHARD :: $(date) :: $(hostname) ==="
python -m llm_polysemy.gen_simplear --variant "$VARIANT" --shard "$SHARD" --num-shards "$NSHARDS"
echo "=== done :: $(date) ==="
