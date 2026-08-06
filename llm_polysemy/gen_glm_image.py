"""Standalone GLM-Image generator for the benchmark's image modality.

GLM-Image is a local diffusion text-to-image model (diffusers `GlmImagePipeline`),
so it does not go through the API runner. It is a true text-to-image model, so the
prompt is the bare word — identical to the API image models' protocol (no
instruction wrapper needed).

Writes the canonical layout so classification/histograms/plots pick it up:
    data/outputs_image/<word>/glm_image/<NN>.png   (model_key "glm_image")

Resume is filesystem-driven; sharding (--shard/--num-shards) supports the same
per-word SLURM job array as the other local generators.

    python -m llm_polysemy.gen_glm_image                          # all 100 words
    python -m llm_polysemy.gen_glm_image --shard 0 --num-shards 100   # one word (array task)
    python -m llm_polysemy.gen_glm_image --limit 1 --n 1          # smoke test

Requires the `glm` conda env (official git transformers+diffusers). See
models/glm-image/build_env.sh and models/glm-image/run_glm_array.slurm.
"""

import argparse
import io
import os
import sys
from pathlib import Path

from . import config, prompts, runner

MODEL_KEY = "glm_image"
# Public HF checkpoint (override via --model-path or GLM_IMAGE_ID). Weights download on
# first use and cache in ~/.cache/huggingface (override with HF_HOME).
DEFAULT_MODEL_PATH = os.environ.get("GLM_IMAGE_ID", "zai-org/GLM-Image")
# Recipe defaults from the model card (the resolution/steps GLM-Image is tuned for).
H, W = 1024, 1152
STEPS, GUIDANCE = 50, 1.5


def load_words(limit):
    words = [l.strip() for l in Path("data/synonyms.txt").read_text().splitlines() if l.strip()][:100]
    return words[:limit] if limit else words


def load_pipe(model_path):
    import torch
    from diffusers import GlmImagePipeline
    print(f"[glm] loading {model_path} ...", flush=True)
    pipe = GlmImagePipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    print("[glm] loaded", flush=True)
    return pipe


def gen_image(pipe, word, idx, h, w, steps, guidance):
    """Bare word -> PNG bytes. Per-sample seed (=idx) gives 30 diverse samples/word."""
    import torch
    g = torch.Generator(device="cuda").manual_seed(idx)
    image = pipe(prompt=word, height=h, width=w, num_inference_steps=steps,
                 guidance_scale=guidance, generator=g).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--height", type=int, default=H)
    p.add_argument("--width", type=int, default=W)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--guidance", type=float, default=GUIDANCE)
    p.add_argument("--limit", type=int, default=None, help="first N words only (smoke test)")
    p.add_argument("--n", type=int, default=None, help="override samples/word (smoke test)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    mod = config.gen_modality("image")
    n_per_word = args.n or mod.n_per_word
    words = load_words(args.limit)
    if args.num_shards > 1:
        words = words[args.shard::args.num_shards]
        print(f"[glm] shard {args.shard}/{args.num_shards}: {len(words)} words", flush=True)

    pipe = load_pipe(args.model_path)

    n_ok = n_err = n_skip = 0
    for w_i, word in enumerate(words):
        todo = [i for i in range(n_per_word)
                if not runner.artifact_path(mod.output_dir, word, MODEL_KEY, i, mod.ext).exists()]
        n_skip += n_per_word - len(todo)
        for idx in todo:
            path = runner.artifact_path(mod.output_dir, word, MODEL_KEY, idx, mod.ext)
            try:
                png = gen_image(pipe, word, idx, args.height, args.width, args.steps, args.guidance)
                path.parent.mkdir(parents=True, exist_ok=True)
                mod.save(path, png)
                n_ok += 1
            except Exception as exc:
                n_err += 1
                print(f"  [err ] {word}/{idx:02d}: {exc}", flush=True)
        print(f"[{w_i+1}/{len(words)}] {word}: ok={n_ok} err={n_err} skip={n_skip}", flush=True)

    print(f"\n[glm] done. {n_ok} images saved, {n_err} errors, {n_skip} already present "
          f"-> {mod.output_dir}", flush=True)
    if n_ok == 0 and n_err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
