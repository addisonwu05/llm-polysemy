#!/bin/bash
# Build conda env 'simplear' for the SimpleAR SFT/RL text-to-image pair.
# Inference-only subset (skips RL/reward libs: hpsv2, trl, deepspeed, vllm, flash-attn).
# transformers MUST be the repo-pinned git commit: the model imports `LogitsWarper`
# from transformers, which was removed in newer releases.
set -x
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
set -e
conda create -y -n simplear python=3.10
conda activate simplear
# torch per the repo's pyproject (2.4.1 / cu121-class); use cu124 wheels available on della.
pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
# transformers pinned to the exact commit SimpleAR was built against (LogitsWarper present)
pip install --no-cache-dir "transformers@git+https://github.com/huggingface/transformers.git@7bbc62474391aff64f63fcc064c975752d1fa4de"
pip install --no-cache-dir accelerate safetensors huggingface-hub sentencepiece protobuf
pip install --no-cache-dir einops ftfy shortuuid mediapy pillow "numpy<2" tqdm loguru
set +e
echo "=== ENV BUILD DONE ==="
python - <<'PY'
import sys
sys.path.insert(0, "/scratch/gpfs/GRIFFITHS/aw2418/models/simplear-repo")
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__)
from simpar.model.builder import load_pretrained_model
from simpar.model.tokenizer.cosmos_tokenizer.video_lib import CausalVideoTokenizer
print("SimpleAR + Cosmos imports OK")
PY
