#!/usr/bin/env python3
"""
Classify generated *sentences* by which sense of the polysemous word they use.

Text analog of classify.py. The same two judges (GPT-5.4 and Gemini 3.1/3.5)
that label the images read each sentence in outputs_text/<word>/<model>/<NN>.txt
and pick one label from that word's candidate senses (meanings.json) plus:
"multiple" / "unclear" / "other".

Per-sentence labels are saved to classifications_text.json (resumable). Aggregate
into per-(word, model, judge) sense histograms with:

    python classify_text.py --histograms
"""

import argparse
import asyncio
import itertools
import json
import os
import re
import threading
from pathlib import Path

from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

MEANINGS_FILE = Path("meanings.json")
OUTPUT_DIR    = Path("outputs_text")
CLASS_FILE    = Path("classifications_text.json")
HIST_FILE     = Path("histograms_text.json")

# ── Judge models — identical to classify.py so both modalities are scored the
#    same way (don't change without re-running the image side too). ────────────
JUDGES = {
    "gpt":    "gpt-5.4",
    "gemini": "gemini-3.5-flash",
}

# Labels offered on top of each word's listed senses.
EXTRA_LABELS = ["multiple", "unclear", "other"]

# Judge calls are cheap text completions with generous rate limits.
PER_KEY_CONCURRENCY = 4


# ── API key pools (prefix-matched env vars, same scheme as eval.py) ──────────

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
_POOL_GEMINI = _KeyPool(_gather_keys("gemini_api_key"), "gemini")

_SEM_GPT    = None  # initialised in main() (must be inside an event loop)
_SEM_GEMINI = None


# ── meanings.json (tolerate the hand-edited trailing commas) ─────────────────

def load_meanings() -> dict:
    raw = MEANINGS_FILE.read_text()
    clean = re.sub(r",(\s*[}\]])", r"\1", raw)  # strip trailing commas
    return json.loads(clean)


MEANINGS = load_meanings()


def allowed_labels(word: str) -> list[str]:
    return [s["sense"] for s in MEANINGS[word]] + EXTRA_LABELS


def sense_keys(word: str) -> list[str]:
    """Just the listed senses (no multiple/unclear/other)."""
    return [s["sense"] for s in MEANINGS[word]]


def build_prompt(word: str, sentence: str) -> str:
    lines = [f'- "{s["sense"]}": {s["meaning"]}' for s in MEANINGS[word]]
    lines.append('- "multiple": the sentence clearly uses two or more of the above meanings together')
    lines.append('- "unclear": you cannot tell, or it matches none of the above')
    lines.append('- "other": it clearly uses a different, specific meaning of the word not listed above')
    opts = "\n".join(lines)
    return (
        f'A language model was given only the single word "{word}" as its prompt, with no '
        f'other context, and asked to use it in one sentence. Here is the sentence it produced:\n\n'
        f'"{sentence}"\n\n'
        f'Decide which meaning of "{word}" the sentence uses.\n\n'
        f'Choose exactly one of these labels:\n{opts}\n\n'
        f'Respond with only the label.'
    )


def build_multiple_prompt(word: str, sentence: str) -> str:
    lines = [f'- "{s["sense"]}": {s["meaning"]}' for s in MEANINGS[word]]
    opts = "\n".join(lines)
    return (
        f'This sentence (generated from the single word "{word}") clearly uses two or '
        f'more distinct meanings of "{word}" together:\n\n"{sentence}"\n\n'
        f'Of the meanings below, list every one that is actually present in the sentence:\n{opts}\n\n'
        f'Return the senses that apply (at least two).'
    )


def text_path(word: str, model_key: str, idx: int) -> Path:
    return OUTPUT_DIR / word / model_key / f"{idx:02d}.txt"


# ── Judge calls ──────────────────────────────────────────────────────────────
# Text-only completions returning parsed JSON matching the given schema:
#   single  -> {"label": <one enum value>}
#   multiple-> {"senses": [<two or more sense keys>]}

def _gpt_call(prompt: str, schema: dict) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=_POOL_OPENAI.next())
    resp = client.chat.completions.create(
        model=JUDGES["gpt"],
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "out", "strict": True, "schema": schema}},
    )
    return json.loads(resp.choices[0].message.content)


def _gemini_call(prompt: str, schema):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=_POOL_GEMINI.next())
    resp = client.models.generate_content(
        model=JUDGES["gemini"],
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    return json.loads(resp.text)


def _single_schema_gpt(word):
    return {"type": "object", "additionalProperties": False,
            "properties": {"label": {"type": "string", "enum": allowed_labels(word)}},
            "required": ["label"]}


def _multi_schema_gpt(word):
    return {"type": "object", "additionalProperties": False,
            "properties": {"senses": {"type": "array", "minItems": 2,
                "items": {"type": "string", "enum": sense_keys(word)}}},
            "required": ["senses"]}


def _single_schema_gemini(word):
    from google.genai import types
    return types.Schema(type=types.Type.OBJECT, required=["label"],
        properties={"label": types.Schema(type=types.Type.STRING, enum=allowed_labels(word))})


def _multi_schema_gemini(word):
    from google.genai import types
    return types.Schema(type=types.Type.OBJECT, required=["senses"],
        properties={"senses": types.Schema(type=types.Type.ARRAY, min_items=2,
            items=types.Schema(type=types.Type.STRING, enum=sense_keys(word)))})


def _classify(judge: str, word: str, path: Path):
    """Return a single label string, or for 'multiple' a dict with the breakdown:
        {"label": "multiple", "senses": ["finance", "river"]}
    """
    sentence = path.read_text().strip()
    if judge == "gpt":
        label = _gpt_call(build_prompt(word, sentence), _single_schema_gpt(word))["label"]
    else:
        label = _gemini_call(build_prompt(word, sentence), _single_schema_gemini(word))["label"]

    if label not in allowed_labels(word):
        label = "unclear"
    if label != "multiple":
        return label

    # Follow-up: ask the same judge which specific senses are present.
    if judge == "gpt":
        senses = _gpt_call(build_multiple_prompt(word, sentence), _multi_schema_gpt(word))["senses"]
    else:
        senses = _gemini_call(build_multiple_prompt(word, sentence), _multi_schema_gemini(word))["senses"]
    senses = [s for s in dict.fromkeys(senses) if s in sense_keys(word)]  # dedupe + validate
    return {"label": "multiple", "senses": senses}


# ── Runner ───────────────────────────────────────────────────────────────────

def _sem_for(judge: str) -> asyncio.Semaphore:
    return {"gpt": _SEM_GPT, "gemini": _SEM_GEMINI}[judge]


async def run_one(word, model_key, idx, judge, results, bar):
    key = f"{judge}/{word}/{model_key}/{idx:02d}"
    path = text_path(word, model_key, idx)
    async with _sem_for(judge):
        for attempt in range(3):
            try:
                verdict = await asyncio.to_thread(_classify, judge, word, path)
                (results.setdefault(word, {}).setdefault(model_key, {})
                        .setdefault(str(idx), {}))[judge] = verdict
                bar.update(1)
                return
            except Exception as exc:
                msg = str(exc)
                retryable = ("429" in msg or "500" in msg or "503"
                             in msg or "timeout" in msg.lower())
                if retryable and attempt < 2:
                    await asyncio.sleep(2 ** attempt * 2)  # 2s, 4s
                else:
                    tqdm.write(f"  [err ] {key}: {exc}")
                    bar.update(1)
                    return


def already_done(results, word, model_key, idx, judge) -> bool:
    return (results.get(word, {}).get(model_key, {})
            .get(str(idx), {}).get(judge) is not None)


def enumerate_tasks(results, words, model_filter, judges, n_cap):
    """Yield (word, model_key, idx, judge) for existing .txt files not yet labelled."""
    for word in words:
        wdir = OUTPUT_DIR / word
        if not wdir.is_dir():
            continue
        for mdir in sorted(wdir.iterdir()):
            if not mdir.is_dir():
                continue
            model_key = mdir.name
            if model_filter and model_key != model_filter:
                continue
            idxs = sorted(int(p.stem) for p in mdir.glob("*.txt") if p.stem.isdigit())
            if n_cap is not None:
                idxs = idxs[:n_cap]
            for idx in idxs:
                for judge in judges:
                    if not already_done(results, word, model_key, idx, judge):
                        yield (word, model_key, idx, judge)


async def main(args):
    words = [w for w in MEANINGS.keys() if not args.word or w == args.word]
    judges = [args.judge] if args.judge else list(JUDGES.keys())

    results = json.loads(CLASS_FILE.read_text()) if CLASS_FILE.exists() else {}

    tasks_meta = list(enumerate_tasks(results, words, args.model, judges, args.n))
    by_judge = {j: [t for t in tasks_meta if t[3] == j] for j in judges}
    print(f"\nPending labels: " +
          " ".join(f"{j}={len(by_judge[j])}" for j in judges) +
          f"  (total={len(tasks_meta)})")
    if not tasks_meta:
        print("Nothing to do.")
        return

    global _SEM_GPT, _SEM_GEMINI
    _SEM_GPT    = asyncio.Semaphore(len(_POOL_OPENAI) * PER_KEY_CONCURRENCY)
    _SEM_GEMINI = asyncio.Semaphore(len(_POOL_GEMINI) * PER_KEY_CONCURRENCY)

    bars = {j: tqdm(total=len(by_judge[j]), desc=j, position=i, leave=True)
            for i, j in enumerate(judges)}

    async def worker(meta):
        await run_one(*meta, results, bars[meta[3]])

    BATCH = 200
    for i in range(0, len(tasks_meta), BATCH):
        await asyncio.gather(*(worker(m) for m in tasks_meta[i:i + BATCH]))
        CLASS_FILE.write_text(json.dumps(results, indent=2))

    for bar in bars.values():
        bar.close()
    CLASS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nDone. Labels -> {CLASS_FILE}")


# ── Histograms ───────────────────────────────────────────────────────────────

def label_of(verdict) -> str:
    """A stored verdict is either a label string or {'label': 'multiple', 'senses': [...]}"""
    return verdict["label"] if isinstance(verdict, dict) else verdict


def build_histograms():
    results = json.loads(CLASS_FILE.read_text())
    hist: dict = {}
    for word, models in results.items():
        for mk, idxs in models.items():
            agree = same = 0
            for idx, judges in idxs.items():
                for judge, verdict in judges.items():
                    label = label_of(verdict)
                    d = (hist.setdefault(word, {}).setdefault(mk, {})
                             .setdefault(judge, {"_counts": {}, "_multiple": {}}))
                    d["_counts"][label] = d["_counts"].get(label, 0) + 1
                    if isinstance(verdict, dict) and verdict.get("senses"):
                        combo = "+".join(sorted(verdict["senses"]))
                        d["_multiple"][combo] = d["_multiple"].get(combo, 0) + 1
                if "gpt" in judges and "gemini" in judges:
                    agree += 1
                    same += (label_of(judges["gpt"]) == label_of(judges["gemini"]))
            if agree:
                hist[word][mk]["_agreement"] = {
                    "same": same, "total": agree, "rate": round(same / agree, 3)}
    HIST_FILE.write_text(json.dumps(hist, indent=2))
    print(f"Histograms -> {HIST_FILE}  ({len(hist)} words)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=list(JUDGES.keys()), default=None,
                        help="run only this judge (default: both)")
    parser.add_argument("--model", default=None,
                        help="classify only this text model_key (default: all)")
    parser.add_argument("--word", default=None,
                        help="classify only this word (default: all)")
    parser.add_argument("--n", type=int, default=None,
                        help="cap sentences per (word, model) — handy for a cheap test run")
    parser.add_argument("--histograms", action="store_true",
                        help="build histograms_text.json from classifications_text.json and exit")
    args = parser.parse_args()

    if args.histograms:
        build_histograms()
    else:
        asyncio.run(main(args))
