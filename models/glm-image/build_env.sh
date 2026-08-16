#!/bin/bash
# Build conda env 'glm' for GLM-Image (diffusers GlmImagePipeline).
set -x
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
set -e
conda create -y -n glm python=3.11
conda activate glm
pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# GlmImagePipeline / GlmImageForConditionalGeneration are only on the bleeding edge
# of the OFFICIAL HF repos (model_index.json shows diffusers 0.37.0.dev0).
pip install --no-cache-dir "git+https://github.com/huggingface/transformers.git"
pip install --no-cache-dir "git+https://github.com/huggingface/diffusers.git"
pip install --no-cache-dir accelerate sentencepiece protobuf pillow numpy tqdm safetensors
set +e
echo "=== ENV BUILD DONE ==="
python - <<'PY'
import torch, transformers, diffusers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__, "diffusers", diffusers.__version__)
try:
    from diffusers import GlmImagePipeline
    print("GlmImagePipeline import OK (top-level)")
except Exception as e:
    print("top-level import failed:", e)
    from diffusers.pipelines.glm_image import GlmImagePipeline
    print("GlmImagePipeline import OK (submodule)")
PY
