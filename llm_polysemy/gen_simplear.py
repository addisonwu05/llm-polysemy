"""SimpleAR autoregressive T2I pair: SFT vs RL (RLHF ablation).

The cleanest ablation in this benchmark: SimpleAR ships two *full* checkpoints that
differ only by the final training stage — `SimpleAR-1.5B-SFT` (supervised fine-tune) and
`SimpleAR-1.5B-RL` (the same model after GRPO-style RL on reward models). No weight-swapping
is needed: each variant is loaded straight from_pretrained, and the VQ
tokenizer (NVIDIA Cosmos DV8x16x16), text tokenizer, prompt format, and sampling params are
all identical across the two, so the only difference is the RL stage.

  --variant sft : Daniel0724/SimpleAR-1.5B-SFT  -> model_key "simplear_sft"
  --variant rl  : Daniel0724/SimpleAR-1.5B-RL   -> model_key "simplear_rl"

SimpleAR is a Qwen2-based AR model: it emits 4096 (=64x64 @ 1024px) discrete visual tokens
which the Cosmos VQ decoder turns into pixels. The prompt is wrapped in the model's standard
T2I control tokens (`<|t2i|> A highly realistic image of {word} <|soi|>`) — a fixed neutral
wrapper applied to every word, so no sense-disambiguating context is added (benchmark protocol).

Runs in the `simplear` conda env on a GPU (no internet needed — all weights are local).

    python -m llm_polysemy.gen_simplear --variant rl  --shard 0 --num-shards 100
    python -m llm_polysemy.gen_simplear --variant sft --limit 2 --n 1     # smoke
"""

import argparse
import io
import sys
from pathlib import Path

from . import config, runner

REPO = "/scratch/gpfs/GRIFFITHS/aw2418/models/simplear-repo"
CKPT = {
    "sft": "/scratch/gpfs/GRIFFITHS/aw2418/models/simplear-sft",
    "rl":  "/scratch/gpfs/GRIFFITHS/aw2418/models/simplear-rl",
}
COSMOS = "/scratch/gpfs/GRIFFITHS/aw2418/models/cosmos-DV8x16x16"
MODEL_KEY = {"sft": "simplear_sft", "rl": "simplear_rl"}

IMAGE_SIZE = 1024              # SimpleAR's primary training resolution
DOWNSAMPLE = 16               # Cosmos DV8x16x16 spatial compression
CODEBOOK_SIZE = 64000
# Matched sampling for BOTH variants (the only diff must be the LLM weights). Repo defaults.
GEN_CFG = dict(cfg_scale=6.0, do_sample=True, temperature=1.0, top_p=1.0, top_k=64000)

POS_PREFIX = "A highly realistic image of "
NEG_PROMPT = ("An image of aerial view, overexposed, low quality, deformation, a poor "
              "composition, bad hands, bad teeth, bad eyes, bad limbs, distortion")


def load_words(limit):
    words = [l.strip() for l in Path("data/synonyms.txt").read_text().splitlines() if l.strip()][:100]
    return words[:limit] if limit else words


def load_model(variant):
    import torch
    sys.path.insert(0, REPO)
    from simpar.model.builder import load_pretrained_model
    from simpar.model.tokenizer.cosmos_tokenizer.networks import TokenizerConfigs
    from simpar.model.tokenizer.cosmos_tokenizer.video_lib import CausalVideoTokenizer as CosmosTokenizer

    print(f"[simplear:{variant}] loading Cosmos VQ tokenizer ...", flush=True)
    cfg = TokenizerConfigs["DV"].value
    cfg.update(dict(spatial_compression=16, temporal_compression=8))
    vq_model = CosmosTokenizer(
        checkpoint_enc=f"{COSMOS}/encoder.jit",
        checkpoint_dec=f"{COSMOS}/decoder.jit",
        tokenizer_config=cfg)
    vq_model.eval()
    vq_model.requires_grad_(False)

    print(f"[simplear:{variant}] loading LLM {CKPT[variant]} ...", flush=True)
    tokenizer, model, _, _ = load_pretrained_model(
        CKPT[variant], attn_implementation="sdpa", device_map="cuda:0")
    model.eval()
    print(f"[simplear:{variant}] ready", flush=True)
    return model, tokenizer, vq_model


def gen_image(model, tokenizer, vq_model, word, idx):
    import torch
    from PIL import Image
    latent = IMAGE_SIZE // DOWNSAMPLE
    max_new = latent ** 2

    fmt = "<|t2i|>" + POS_PREFIX + word + "<|soi|>"
    uncond = "<|t2i|>" + NEG_PROMPT + "<|soi|>"
    input_ids = tokenizer(fmt, return_tensors="pt").input_ids.to("cuda:0")
    uncond_ids = tokenizer(uncond, return_tensors="pt").input_ids.to("cuda:0")

    torch.manual_seed(idx)   # per-sample diversity, reproducible
    with torch.inference_mode():
        out = model.generate_visual(
            input_ids, negative_prompt_ids=uncond_ids, max_new_tokens=max_new,
            use_cache=True, **GEN_CFG)
        idx_sample = out[:, input_ids.shape[1]: input_ids.shape[1] + max_new].clone()
        idx_sample = idx_sample - len(tokenizer)
        idx_sample = torch.clamp(idx_sample, min=0, max=CODEBOOK_SIZE - 1)
        idx_sample = idx_sample.reshape(-1, latent, latent).unsqueeze(1)
        samples = vq_model.decode(idx_sample)          # (1,3,1,H,W) in [-1,1]

    samples = samples.squeeze(2).squeeze(0)            # (3,H,W)
    arr = ((samples.clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).float().cpu().numpy().astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["sft", "rl"], required=True)
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
        print(f"[simplear:{args.variant}] shard {args.shard}/{args.num_shards}: {len(words)} words", flush=True)

    model, tokenizer, vq_model = load_model(args.variant)

    n_ok = n_err = n_skip = 0
    for w_i, word in enumerate(words):
        todo = [i for i in range(n_per_word)
                if not runner.artifact_path(mod.output_dir, word, model_key, i, mod.ext).exists()]
        n_skip += n_per_word - len(todo)
        for idx in todo:
            path = runner.artifact_path(mod.output_dir, word, model_key, idx, mod.ext)
            try:
                png = gen_image(model, tokenizer, vq_model, word, idx)
                path.parent.mkdir(parents=True, exist_ok=True)
                mod.save(path, png)
                n_ok += 1
            except Exception as exc:
                n_err += 1
                print(f"  [err ] {word}/{idx:02d}: {exc}", flush=True)
        print(f"[{w_i+1}/{len(words)}] {word}: ok={n_ok} err={n_err} skip={n_skip}", flush=True)

    print(f"\n[simplear:{args.variant}] done. {n_ok} saved, {n_err} errors, {n_skip} present "
          f"-> {mod.output_dir}/<word>/{model_key}", flush=True)
    if n_ok == 0 and n_err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
