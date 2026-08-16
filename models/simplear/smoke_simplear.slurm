#!/bin/bash
#SBATCH --job-name=simplear-smoke
#SBATCH --account=griffith
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=models/simplear/smoke-%j.out
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")"
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate simplear
echo "=== simplear smoke :: $(date) :: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "--- SFT ---"; python -m llm_polysemy.gen_simplear --variant sft --limit 2 --n 1
echo "--- RL  ---"; python -m llm_polysemy.gen_simplear --variant rl  --limit 2 --n 1
echo "--- artifacts ---"; ls data/outputs_image/*/simplear_sft/ data/outputs_image/*/simplear_rl/ 2>/dev/null | head
echo "=== done :: $(date) ==="
