#!/bin/bash
#SBATCH --job-name=glm-smoke
#SBATCH --account=griffith
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=models/glm-image/smoke-%j.out
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")"
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glm
echo "=== glm smoke :: $(date) :: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -m llm_polysemy.gen_glm_image --limit 3 --n 1
echo "--- artifacts ---"; ls -la data/outputs_image/*/glm_image/ 2>/dev/null | head
echo "=== done :: $(date) ==="
