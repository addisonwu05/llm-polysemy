#!/bin/bash
#SBATCH --job-name=sdxl-smoke
#SBATCH --account=griffith
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=models/sdxl/smoke-%j.out
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")"
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glm
echo "=== sdxl smoke :: $(date) :: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "--- BASE ---"; python -m llm_polysemy.gen_sdxl --variant base --limit 2 --n 1
echo "--- DPO ---";  python -m llm_polysemy.gen_sdxl --variant dpo  --limit 2 --n 1
echo "--- artifacts ---"; ls data/outputs_image/*/sdxl_base/ data/outputs_image/*/sdxl_dpo/ 2>/dev/null | head
echo "=== done :: $(date) ==="
