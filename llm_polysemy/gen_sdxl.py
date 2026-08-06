"""SDXL generator for the RLHF-ablation arm: SDXL-base vs Diffusion-DPO SDXL.

Two models that differ ONLY in the preference-tuning (DPO) stage:
  --variant base : stabilityai/stable-diffusion-xl-base-1.0           -> model_key "sdxl_base"
  --variant dpo  : same pipeline, UNet swapped for mhdang/dpo-sdxl...  -> model_key "sdxl_dpo"

Both are diffusion text-to-image, so the prompt is the bare word (benchmark protocol).
Runs in the `glm` conda env (diffusers + torch already present). Sharding + resume
identical to the other local generators.

    python -m llm_polysemy.gen_sdxl --variant base --shard 0 --num-shards 100
    python -m llm_polysemy.gen_sdxl --variant dpo  --limit 1 --n 1     # smoke
"""

import argparse
import io
import os
import sys
from pathlib import Path

from . import config, runner

MODEL_KEY = {"base": "sdxl_base", "dpo": "sdxl_dpo"}
# Public HF checkpoints (override via --base-id / --dpo-id or these env vars).
# Weights download on first use and cache in ~/.cache/huggingface (override with HF_HOME).
SDXL_BASE = os.environ.get("SDXL_BASE_ID", "stabilityai/stable-diffusion-xl-base-1.0")
DPO_UNET = os.environ.get("SDXL_DPO_ID", "mhdang/dpo-sdxl-text2image-v1")
STEPS, GUIDANCE = 40, 5.0   # identical settings for both variants -> the only diff is the UNet


def load_words(limit):
    words = [l.strip() for l in Path("data/synonyms.txt").read_text().splitlines() if l.strip()][:100]
    return words[:limit] if limit else words


def load_pipe(variant, base_id, dpo_id):
    import torch
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
    print(f"[sdxl:{variant}] loading base pipeline {base_id} ...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    if variant == "dpo":
        print(f"[sdxl:dpo] swapping in DPO-tuned UNet {dpo_id} ...", flush=True)
        unet = UNet2DConditionModel.from_pretrained(
            dpo_id, subfolder="unet", torch_dtype=torch.float16)
        pipe.unet = unet.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"[sdxl:{variant}] loaded", flush=True)
    return pipe


def gen_image(pipe, word, idx):
    import torch
    g = torch.Generator(device="cuda").manual_seed(idx)
    image = pipe(prompt=word, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                 generator=g).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["base", "dpo"], required=True)
    p.add_argument("--base-id", default=SDXL_BASE,
                   help="HF repo id or local dir for the SDXL base pipeline")
    p.add_argument("--dpo-id", default=DPO_UNET,
                   help="HF repo id or local dir holding the Diffusion-DPO UNet (subfolder 'unet')")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    mod = config.gen_modality("image")
    model_key = MODEL_KEY[args.variant]
    n_per_word = args.n or mod.n_per_word
    words = load_words(args.limit)
    if args.num_shards > 1:
        words = words[args.shard::args.num_shards]
        print(f"[sdxl:{args.variant}] shard {args.shard}/{args.num_shards}: {len(words)} words", flush=True)

    pipe = load_pipe(args.variant, args.base_id, args.dpo_id)

    n_ok = n_err = n_skip = 0
    for w_i, word in enumerate(words):
        todo = [i for i in range(n_per_word)
                if not runner.artifact_path(mod.output_dir, word, model_key, i, mod.ext).exists()]
        n_skip += n_per_word - len(todo)
        for idx in todo:
            path = runner.artifact_path(mod.output_dir, word, model_key, idx, mod.ext)
            try:
                png = gen_image(pipe, word, idx)
                path.parent.mkdir(parents=True, exist_ok=True)
                mod.save(path, png)
                n_ok += 1
            except Exception as exc:
                n_err += 1
                print(f"  [err ] {word}/{idx:02d}: {exc}", flush=True)
        print(f"[{w_i+1}/{len(words)}] {word}: ok={n_ok} err={n_err} skip={n_skip}", flush=True)

    print(f"\n[sdxl:{args.variant}] done. {n_ok} saved, {n_err} errors, {n_skip} present "
          f"-> {mod.output_dir}/<word>/{model_key}", flush=True)
    if n_ok == 0 and n_err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
