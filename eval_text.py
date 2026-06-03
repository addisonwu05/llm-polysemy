#!/usr/bin/env python3
"""
Text analog of eval.py: probe how LLMs resolve polysemous words.

For each word we send the bare word (no disambiguating context) wrapped in a
fixed "use it in one sentence" instruction, sample N_PER_WORD times at
temperature 1.0, and save each generated sentence. A later classify pass buckets
each sentence into a sense from meanings.json -> per-word per-model sense
histograms, directly comparable to the image-model distributions from eval.py.

Output layout: outputs_text/<word>/<model_key>/<NN>.txt (word first, then model).
Resume is filesystem-driven (pending_indices); results_text.json is a status log.
"""

import asyncio
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
OUTPUT_DIR = Path("outputs_text")
RESULTS_FILE = Path("results_text.json")
N_PER_WORD = 30
TEMPERATURE = 1.0

PROMPT_TEMPLATE = (
    "Use the following word in a single sentence: {word}\n"
    "Reply with only the sentence and nothing else."
)

MODELS = {
    # ── OpenAI ────────────────────────────────────────────────────────────────
    "openai_55":     "gpt-5.5",
    "openai_54mini": "gpt-5.4-mini",
    "openai_54nano": "gpt-5.4-nano",
    "openai_5":      "gpt-5",
    "openai_4o":     "gpt-4o",
    "openai_35":     "gpt-3.5-turbo",
    # ── xAI Grok ─────────────────────────────────────────────────────────────
    "grok_43":       "grok-4.3",
    "grok_420":      "grok-4.20-0309-reasoning",
    "grok_420_nr":   "grok-4.20-0309-non-reasoning",
    # ── Google Gemini ─────────────────────────────────────────────────────────
    "gemini_31pro":  "gemini-3.1-pro-preview",
    "gemini_35flash":"gemini-3.5-flash",
    "gemini_25flash":"gemini-2.5-flash",
    # ── Qwen (Together.ai, serverless) ────────────────────────────────────────
    "qwen_35_397b":  "Qwen/Qwen3.5-397B-A17B",
    "qwen_35_9b":    "Qwen/Qwen3.5-9B",
    "qwen_25_7b":    "Qwen/Qwen2.5-7B-Instruct-Turbo",
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


_POOL_OPENAI = _KeyPool(_gather_keys("openai_api_key"), "openai")
_POOL_GROK   = _KeyPool(_gather_keys("xai_api_key"),    "xai")
_POOL_GEMINI = _KeyPool(_gather_keys("gemini_api_key"), "gemini")
_POOL_QWEN   = _KeyPool(_gather_keys("together_api_key_qwen"), "qwen")

# Per-family semaphores so each provider gets exactly its own key-count of slots.
_SEM_OPENAI = None  # initialised in main() (must be inside an event loop)
_SEM_GROK   = None
_SEM_GEMINI = None
_SEM_QWEN   = None

_FAMILY = {
    "openai_55":     "openai",
    "openai_54mini": "openai",
    "openai_54nano": "openai",
    "openai_5":      "openai",
    "openai_4o":     "openai",
    "openai_35":     "openai",
    "grok_43":       "grok",
    "grok_420":      "grok",
    "grok_420_nr":   "grok",
    "gemini_31pro":  "gemini",
    "gemini_35flash":"gemini",
    "gemini_25flash":"gemini",
    "qwen_35_397b":  "qwen",
    "qwen_35_9b":    "qwen",
    "qwen_25_7b":    "qwen",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_words():
    return [l.strip() for l in WORDS_FILE.read_text().splitlines() if l.strip()]


def text_path(model_key: str, word: str, idx: int) -> Path:
    return OUTPUT_DIR / word / model_key / f"{idx:02d}.txt"


def save_text(model_key: str, word: str, idx: int, text: str) -> Path:
    path = text_path(model_key, word, idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def pending_indices(model_key: str, word: str) -> list[int]:
    return [i for i in range(N_PER_WORD) if not text_path(model_key, word, i).exists()]


def _prompt(word: str) -> str:
    return PROMPT_TEMPLATE.format(word=word)


# ── OpenAI-SDK families (OpenAI, Grok, Qwen) ───────────────────────────────────

def _chat_generate(word: str, model: str, pool: _KeyPool, base_url: str | None) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=pool.next(), **({"base_url": base_url} if base_url else {}))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _prompt(word)}],
        temperature=TEMPERATURE,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise ValueError(f"no text returned for '{word}'")
    return text


def _openai_generate(word: str, model: str) -> str:
    return _chat_generate(word, model, _POOL_OPENAI, None)


def _grok_generate(word: str, model: str) -> str:
    return _chat_generate(word, model, _POOL_GROK, "https://api.x.ai/v1")


def _qwen_generate(word: str, model: str) -> str:
    return _chat_generate(word, model, _POOL_QWEN, "https://api.together.xyz/v1")


# ── Google Gemini ─────────────────────────────────────────────────────────────

def _gemini_generate(word: str, model: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_POOL_GEMINI.next())
    resp = client.models.generate_content(
        model=model,
        contents=_prompt(word),
        config=types.GenerateContentConfig(temperature=TEMPERATURE),
    )
    text = (resp.text or "").strip()
    if not text:
        raise ValueError(f"no text returned for '{word}'")
    return text


# ── Dispatch table ────────────────────────────────────────────────────────────

GENERATORS = {
    "openai_55":     lambda w: _openai_generate(w, MODELS["openai_55"]),
    "openai_54mini": lambda w: _openai_generate(w, MODELS["openai_54mini"]),
    "openai_54nano": lambda w: _openai_generate(w, MODELS["openai_54nano"]),
    "openai_5":      lambda w: _openai_generate(w, MODELS["openai_5"]),
    "openai_4o":     lambda w: _openai_generate(w, MODELS["openai_4o"]),
    "openai_35":     lambda w: _openai_generate(w, MODELS["openai_35"]),
    "grok_43":       lambda w: _grok_generate(w, MODELS["grok_43"]),
    "grok_420":      lambda w: _grok_generate(w, MODELS["grok_420"]),
    "grok_420_nr":   lambda w: _grok_generate(w, MODELS["grok_420_nr"]),
    "gemini_31pro":  lambda w: _gemini_generate(w, MODELS["gemini_31pro"]),
    "gemini_35flash":lambda w: _gemini_generate(w, MODELS["gemini_35flash"]),
    "gemini_25flash":lambda w: _gemini_generate(w, MODELS["gemini_25flash"]),
    "qwen_35_397b":  lambda w: _qwen_generate(w, MODELS["qwen_35_397b"]),
    "qwen_35_9b":    lambda w: _qwen_generate(w, MODELS["qwen_35_9b"]),
    "qwen_25_7b":    lambda w: _qwen_generate(w, MODELS["qwen_25_7b"]),
}


# ── Runner ────────────────────────────────────────────────────────────────────

def _sem_for(model_key: str) -> asyncio.Semaphore:
    return {"openai": _SEM_OPENAI, "grok": _SEM_GROK,
            "gemini": _SEM_GEMINI, "qwen": _SEM_QWEN}[_FAMILY[model_key]]


async def run_one(model_key: str, word: str, idx: int, results: dict, bar: tqdm):
    key = f"{model_key}/{word}/{idx:02d}"
    async with _sem_for(model_key):
        for attempt in range(3):
            try:
                t0 = time.time()
                text = await asyncio.to_thread(GENERATORS[model_key], word)
                save_text(model_key, word, idx, text)
                elapsed = time.time() - t0
                tqdm.write(f"  [ ok ] {key}  ({elapsed:.1f}s)")
                results.setdefault(word, {}).setdefault(model_key, {})[str(idx)] = "ok"
                bar.update(1)
                return
            except Exception as exc:
                retryable = "429" in str(exc) or "no text" in str(exc)
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

    global _SEM_OPENAI, _SEM_GROK, _SEM_GEMINI, _SEM_QWEN
    _SEM_OPENAI = asyncio.Semaphore(len(_POOL_OPENAI))
    _SEM_GROK   = asyncio.Semaphore(len(_POOL_GROK))
    _SEM_GEMINI = asyncio.Semaphore(len(_POOL_GEMINI))
    _SEM_QWEN   = asyncio.Semaphore(len(_POOL_QWEN))
    total_slots = len(_POOL_OPENAI) + len(_POOL_GROK) + len(_POOL_GEMINI) + len(_POOL_QWEN)
    tqdm.write(f"\nConcurrency: openai={len(_POOL_OPENAI)} grok={len(_POOL_GROK)} gemini={len(_POOL_GEMINI)} qwen={len(_POOL_QWEN)} (total={total_slots})")

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
    tqdm.write(f"\nDone. {total} sentences saved, {errors} errors. Results -> {RESULTS_FILE}")


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
