#!/usr/bin/env python3
"""
Eval script: generate images for each polysemous word using 5 models.
10 generations per word per model.
Results saved to outputs/<model>/<word>/<n>.png; progress tracked in results.json.
"""

import asyncio
import base64
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

WORDS_FILE = Path("synonyms.txt")
OUTPUT_DIR = Path("outputs")
RESULTS_FILE = Path("results.json")
N_PER_WORD = 10

MODELS = {
    "gemini": "gemini-3.1-flash-image-preview",
    "grok":   "grok-imagine-image-pro",
    "openai": "gpt-image-2",
    "qwen":   "Qwen/Qwen-Image-2.0-Pro",
    "flux":   "black-forest-labs/FLUX.2-pro",
}


def load_words():
    return [l.strip() for l in WORDS_FILE.read_text().splitlines() if l.strip()]


def image_path(model_key: str, word: str, idx: int) -> Path:
    return OUTPUT_DIR / model_key / word / f"{idx:02d}.png"


def save_image(model_key: str, word: str, idx: int, data: bytes) -> Path:
    path = image_path(model_key, word, idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def pending_indices(model_key: str, word: str) -> list[int]:
    return [i for i in range(N_PER_WORD) if not image_path(model_key, word, i).exists()]


# ── OpenAI (gpt-image-2) ──────────────────────────────────────────────────────

def _openai_generate(word: str) -> bytes:
    import urllib.request
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.images.generate(
        model=MODELS["openai"],
        prompt=word,
        n=1,
        size="1024x1024",
    )
    item = resp.data[0]
    if getattr(item, "b64_json", None):
        return base64.b64decode(item.b64_json)
    with urllib.request.urlopen(item.url) as r:
        return r.read()


# ── xAI Grok (grok-imagine-image-pro) ────────────────────────────────────────

def _grok_generate(word: str) -> bytes:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )
    resp = client.images.generate(
        model=MODELS["grok"],
        prompt=word,
        n=1,
        response_format="b64_json",
    )
    return base64.b64decode(resp.data[0].b64_json)


# ── Google Gemini (gemini-3.1-flash-image-preview) ────────────────────────────

def _gemini_generate(word: str) -> bytes:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=MODELS["gemini"],
        contents=word,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )
    for part in resp.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data
    raise ValueError(f"Gemini returned no image for '{word}'")


# ── Together.ai (Qwen / FLUX) ─────────────────────────────────────────────────

_TOGETHER_KEYS = {
    "qwen": "TOGETHER_API_KEY_QWEN",
    "flux": "TOGETHER_API_KEY_BF",
}

def _together_generate(model_key: str, word: str) -> bytes:
    import urllib.request
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ[_TOGETHER_KEYS[model_key]],
        base_url="https://api.together.xyz/v1",
    )
    resp = client.images.generate(
        model=MODELS[model_key],
        prompt=word,
        n=1,
        response_format="b64_json",
    )
    item = resp.data[0]
    if getattr(item, "b64_json", None):
        return base64.b64decode(item.b64_json)
    req = urllib.request.Request(
        item.url,
        headers={"Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"},
    )
    with urllib.request.urlopen(req) as r:
        return r.read()

def _qwen_generate(word: str) -> bytes:
    return _together_generate("qwen", word)

def _flux_generate(word: str) -> bytes:
    return _together_generate("flux", word)


# ── Dispatch table ────────────────────────────────────────────────────────────

GENERATORS = {
    "gemini": _gemini_generate,
    "grok":   _grok_generate,
    "openai": _openai_generate,
    "qwen":   _qwen_generate,
    "flux":   _flux_generate,
}


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_one(model_key: str, word: str, idx: int, results: dict, sem: asyncio.Semaphore):
    key = f"{model_key}/{word}/{idx:02d}"
    async with sem:
        for attempt in range(5):
            try:
                t0 = time.time()
                data = await asyncio.to_thread(GENERATORS[model_key], word)
                save_image(model_key, word, idx, data)
                elapsed = time.time() - t0
                print(f"  [ ok ] {key}  ({elapsed:.1f}s)")
                results.setdefault(word, {}).setdefault(model_key, {})[str(idx)] = "ok"
                return
            except Exception as exc:
                retryable = "429" in str(exc) or "no image" in str(exc)
                if retryable and attempt < 4:
                    wait = 2 ** attempt * 2  # 2, 4, 8, 16s
                    print(f"  [rty ] {key}  retry {attempt+1}/4 in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    print(f"  [err ] {key}: {exc}")
                    results.setdefault(word, {}).setdefault(model_key, {})[str(idx)] = f"error: {exc}"
                    return


async def main(model_keys: list[str]):
    words = load_words()
    results: dict = {}
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())

    sem = asyncio.Semaphore(5)

    for word in words:
        tasks = []
        skipped = 0
        for mk in model_keys:
            todo = pending_indices(mk, word)
            skipped += N_PER_WORD - len(todo)
            tasks.extend(run_one(mk, word, i, results, sem) for i in todo)

        label = f"=== {word} === ({len(tasks)} to generate"
        if skipped:
            label += f", {skipped} already done"
        print(f"\n{label})")

        await asyncio.gather(*tasks)
        RESULTS_FILE.write_text(json.dumps(results, indent=2))

    total = sum(
        1 for w in results.values()
        for m in w.values()
        for s in m.values()
        if s in ("ok", "done")
    )
    errors = sum(
        1 for w in results.values()
        for m in w.values()
        for s in m.values()
        if s.startswith("error")
    )
    print(f"\nDone. {total} images saved, {errors} errors. Results -> {RESULTS_FILE}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=list(MODELS.keys()), default=None,
        help="run only this model (default: all)",
    )
    args = parser.parse_args()
    keys = [args.model] if args.model else list(MODELS.keys())
    asyncio.run(main(keys))
