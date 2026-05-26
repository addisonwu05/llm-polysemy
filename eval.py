#!/usr/bin/env python3
"""
Eval script: generate images for each polysemous word using multiple models.
30 generations per word per model.
Results saved to outputs/<model>/<word>/<n>.png; progress tracked in results.json.
"""

import asyncio
import base64
import itertools
import json
import os
import threading
import time
from pathlib import Path

from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

WORDS_FILE = Path("synonyms.txt")
OUTPUT_DIR = Path("outputs")
RESULTS_FILE = Path("results.json")
N_PER_WORD = 30

MODELS = {
    # ── OpenAI ────────────────────────────────────────────────────────────────
    "openai":      "gpt-image-2",
    "openai_15":   "gpt-image-1.5",
    "openai_1":    "gpt-image-1",
    "openai_mini": "gpt-image-1-mini",
    # ── xAI Grok ─────────────────────────────────────────────────────────────
    "grok":        "grok-imagine-image-quality",  # grok-imagine-image-pro deprecated May 15 2026
    "grok_base":   "grok-imagine-image",
    "grok_pro":    "grok-imagine-image-pro",
    # ── Google Gemini ─────────────────────────────────────────────────────────
    "gemini":      "gemini-3.1-flash-image-preview",
    "gemini_25":   "gemini-2.5-flash-image",
    "gemini_pro":  "gemini-3-pro-image-preview",
    # ── Qwen (Together.ai) ────────────────────────────────────────────────────
    "qwen":        "Qwen/Qwen-Image-2.0-Pro",
    "qwen_20":     "Qwen/Qwen-Image-2.0",
    "qwen_1":      "Qwen/Qwen-Image",
    # ── FLUX (Together.ai / Black Forest Labs) ────────────────────────────────
    "flux":        "black-forest-labs/FLUX.2-pro",
    "flux_max":    "black-forest-labs/FLUX.2-max",
    "flux_flex":   "black-forest-labs/FLUX.2-flex",
    "flux_1pro":   "black-forest-labs/FLUX.1.1-pro",
}


# ── Key pools ─────────────────────────────────────────────────────────────────

class _KeyPool:
    """Thread-safe round-robin pool over a list of API keys."""
    def __init__(self, keys: list[str], name: str):
        if not keys:
            raise RuntimeError(f"No API keys found for pool '{name}'")
        self._keys = keys
        self._cycle = itertools.cycle(keys)
        self._lock = threading.Lock()
        print(f"  [keys] {name}: {len(keys)} key(s)")

    def __len__(self) -> int:
        return len(self._keys)

    def next(self) -> str:
        with self._lock:
            return next(self._cycle)


def _gather_keys(prefix: str) -> list[str]:
    p = prefix.lower()
    return [v.strip() for k, v in os.environ.items()
            if k.lower().startswith(p) and v.strip()]


_POOL_OPENAI   = _KeyPool(_gather_keys("openai_api_key"), "openai")
_POOL_GROK     = _KeyPool(_gather_keys("xai_api_key"),    "xai")
_POOL_GEMINI   = _KeyPool(_gather_keys("gemini_api_key"), "gemini")
_POOL_QWEN     = _KeyPool(_gather_keys("together_api_key_qwen"), "qwen")
_POOL_FLUX     = _KeyPool(_gather_keys("together_api_key_bf"),   "flux")

# Per-family semaphores so each provider gets exactly its own key-count of slots.
_SEM_OPENAI = None  # initialised in main() (must be inside an event loop)
_SEM_GROK   = None
_SEM_GEMINI = None
_SEM_QWEN   = None
_SEM_FLUX   = None

_FAMILY = {
    "openai":      "openai",
    "openai_15":   "openai",
    "openai_1":    "openai",
    "openai_mini": "openai",
    "grok":        "grok",
    "grok_base":   "grok",
    "grok_pro":    "grok",
    "gemini":      "gemini",
    "gemini_25":   "gemini",
    "gemini_pro":  "gemini",
    "qwen":        "qwen",
    "qwen_20":     "qwen",
    "qwen_1":      "qwen",
    "flux":        "flux",
    "flux_max":    "flux",
    "flux_flex":   "flux",
    "flux_1pro":   "flux",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_words():
    return [l.strip() for l in WORDS_FILE.read_text().splitlines() if l.strip()]


def image_path(model_key: str, word: str, idx: int) -> Path:
    return OUTPUT_DIR / word / model_key / f"{idx:02d}.png"


def save_image(model_key: str, word: str, idx: int, data: bytes) -> Path:
    path = image_path(model_key, word, idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def pending_indices(model_key: str, word: str) -> list[int]:
    return [i for i in range(N_PER_WORD) if not image_path(model_key, word, i).exists()]


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _openai_generate(word: str, model: str) -> bytes:
    import urllib.request
    from openai import OpenAI
    client = OpenAI(api_key=_POOL_OPENAI.next())
    resp = client.images.generate(
        model=model,
        prompt=word,
        n=1,
        size="1024x1024",
    )
    item = resp.data[0]
    if getattr(item, "b64_json", None):
        return base64.b64decode(item.b64_json)
    with urllib.request.urlopen(item.url) as r:
        return r.read()


# ── xAI Grok ─────────────────────────────────────────────────────────────────

def _grok_generate(word: str, model: str) -> bytes:
    from openai import OpenAI
    client = OpenAI(
        api_key=_POOL_GROK.next(),
        base_url="https://api.x.ai/v1",
    )
    resp = client.images.generate(
        model=model,
        prompt=word,
        n=1,
        response_format="b64_json",
    )
    return base64.b64decode(resp.data[0].b64_json)


# ── Google Gemini ─────────────────────────────────────────────────────────────

def _gemini_generate(word: str, model: str) -> bytes:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_POOL_GEMINI.next())
    resp = client.models.generate_content(
        model=model,
        contents=word,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )
    for part in resp.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data
    raise ValueError(f"Gemini returned no image for '{word}'")


# ── Together.ai (Qwen, FLUX family) ──────────────────────────────────────────

def _together_generate(word: str, model: str, pool: _KeyPool) -> bytes:
    import urllib.request
    from openai import OpenAI
    api_key = pool.next()
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.together.xyz/v1",
    )
    resp = client.images.generate(
        model=model,
        prompt=word,
        n=1,
        response_format="b64_json",
    )
    item = resp.data[0]
    if getattr(item, "b64_json", None):
        return base64.b64decode(item.b64_json)
    req = urllib.request.Request(
        item.url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req) as r:
        return r.read()


# ── Dispatch table ────────────────────────────────────────────────────────────

GENERATORS = {
    "openai":      lambda w: _openai_generate(w, MODELS["openai"]),
    "openai_15":   lambda w: _openai_generate(w, MODELS["openai_15"]),
    "openai_1":    lambda w: _openai_generate(w, MODELS["openai_1"]),
    "openai_mini": lambda w: _openai_generate(w, MODELS["openai_mini"]),
    "grok":        lambda w: _grok_generate(w, MODELS["grok"]),
    "grok_base":   lambda w: _grok_generate(w, MODELS["grok_base"]),
    "grok_pro":    lambda w: _grok_generate(w, MODELS["grok_pro"]),
    "gemini":      lambda w: _gemini_generate(w, MODELS["gemini"]),
    "gemini_25":   lambda w: _gemini_generate(w, MODELS["gemini_25"]),
    "gemini_pro":  lambda w: _gemini_generate(w, MODELS["gemini_pro"]),
    "qwen":        lambda w: _together_generate(w, MODELS["qwen"],      _POOL_QWEN),
    "qwen_20":     lambda w: _together_generate(w, MODELS["qwen_20"],   _POOL_QWEN),
    "qwen_1":      lambda w: _together_generate(w, MODELS["qwen_1"],    _POOL_QWEN),
    "flux":        lambda w: _together_generate(w, MODELS["flux"],      _POOL_FLUX),
    "flux_max":    lambda w: _together_generate(w, MODELS["flux_max"],  _POOL_FLUX),
    "flux_flex":   lambda w: _together_generate(w, MODELS["flux_flex"], _POOL_FLUX),
    "flux_1pro":   lambda w: _together_generate(w, MODELS["flux_1pro"], _POOL_FLUX),
}


# ── Runner ────────────────────────────────────────────────────────────────────

def _sem_for(model_key: str) -> asyncio.Semaphore:
    return {"openai": _SEM_OPENAI, "grok": _SEM_GROK,
            "gemini": _SEM_GEMINI, "qwen": _SEM_QWEN, "flux": _SEM_FLUX}[_FAMILY[model_key]]


async def run_one(model_key: str, word: str, idx: int, results: dict, bar: tqdm):
    key = f"{model_key}/{word}/{idx:02d}"
    async with _sem_for(model_key):
        for attempt in range(3):
            try:
                t0 = time.time()
                data = await asyncio.to_thread(GENERATORS[model_key], word)
                save_image(model_key, word, idx, data)
                elapsed = time.time() - t0
                tqdm.write(f"  [ ok ] {key}  ({elapsed:.1f}s)")
                results.setdefault(word, {}).setdefault(model_key, {})[str(idx)] = "ok"
                bar.update(1)
                return
            except Exception as exc:
                retryable = "429" in str(exc) or "no image" in str(exc)
                if retryable and attempt < 2:
                    wait = 2 ** attempt * 2  # 2, 4s
                    tqdm.write(f"  [rty ] {key}  retry {attempt+1}/2 in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    tqdm.write(f"  [err ] {key}: {exc}")
                    results.setdefault(word, {}).setdefault(model_key, {})[str(idx)] = f"error: {exc}"
                    bar.update(1)
                    return


async def run_model(model_key: str, words: list[str], results: dict, bar: tqdm):
    for word in words:
        todo = pending_indices(model_key, word)
        bar.update(N_PER_WORD - len(todo))
        tasks = [run_one(model_key, word, i, results, bar) for i in todo]
        if tasks:
            await asyncio.gather(*tasks)
        RESULTS_FILE.write_text(json.dumps(results, indent=2))


async def main(model_keys: list[str]):
    words = load_words()[:100]
    results: dict = {}
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())

    global _SEM_OPENAI, _SEM_GROK, _SEM_GEMINI, _SEM_QWEN, _SEM_FLUX
    _SEM_OPENAI = asyncio.Semaphore(len(_POOL_OPENAI))
    _SEM_GROK   = asyncio.Semaphore(len(_POOL_GROK))
    _SEM_GEMINI = asyncio.Semaphore(len(_POOL_GEMINI))
    _SEM_QWEN   = asyncio.Semaphore(len(_POOL_QWEN))
    _SEM_FLUX   = asyncio.Semaphore(len(_POOL_FLUX))
    total_slots = len(_POOL_OPENAI) + len(_POOL_GROK) + len(_POOL_GEMINI) + len(_POOL_QWEN) + len(_POOL_FLUX)
    tqdm.write(f"\nConcurrency: openai={len(_POOL_OPENAI)} grok={len(_POOL_GROK)} gemini={len(_POOL_GEMINI)} qwen={len(_POOL_QWEN)} flux={len(_POOL_FLUX)} (total={total_slots})")

    bars = {
        mk: tqdm(total=len(words) * N_PER_WORD, desc=mk, position=i, leave=True)
        for i, mk in enumerate(model_keys)
    }

    await asyncio.gather(*(
        run_model(mk, words, results, bars[mk])
        for mk in model_keys
    ))

    for bar in bars.values():
        bar.close()

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
    tqdm.write(f"\nDone. {total} images saved, {errors} errors. Results -> {RESULTS_FILE}")


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
