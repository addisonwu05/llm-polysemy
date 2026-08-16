"""Generic async engine shared by generation and classification.

Two thin orchestrators keep their genuinely-distinct behavior (save cadence,
concurrency formula, resume mechanism) while sharing one retry helper. Behavior is
preserved exactly from the original eval.py / classify.py.
"""

import asyncio
import json
from pathlib import Path

from tqdm import tqdm

from . import schemas
from .config import GenModality, ClassifyModality, JUDGES, PER_KEY_CONCURRENCY, GEN_PER_KEY_CONCURRENCY
from .keys import build_pools
from .meanings import allowed_labels, sense_keys
from .registry import FAMILY_PROVIDER, family_of, models_for, spec
from . import providers


# ── Shared retry helper ────────────────────────────────────────────────────────

async def _attempt_with_retry(make_coro_factory, retryable, key, bar):
    """Run a blocking callable in a thread, up to 3 attempts, 2s/4s backoff on
    retryable errors. Returns (ok, value_or_exc)."""
    for attempt in range(3):
        try:
            value = await asyncio.to_thread(make_coro_factory)
            return True, value
        except Exception as exc:
            if retryable(str(exc)) and attempt < 2:
                wait = 2 ** attempt * 2  # 2s, 4s
                tqdm.write(f"  [rty ] {key}  retry {attempt+1}/2 in {wait}s")
                await asyncio.sleep(wait)
            else:
                return False, exc


# ── Paths / resume ─────────────────────────────────────────────────────────────

def artifact_path(out_dir: Path, word: str, model_key: str, idx: int, ext: str) -> Path:
    return out_dir / word / model_key / f"{idx:02d}.{ext}"


def pending_indices(mod: GenModality, model_key: str, word: str) -> list[int]:
    return [i for i in range(mod.n_per_word)
            if not artifact_path(mod.output_dir, word, model_key, i, mod.ext).exists()]


# ── Generation ─────────────────────────────────────────────────────────────────

async def run_generation(mod: GenModality, model_keys: list[str], words: list[str]):
    # Build only the pools the selected families need (text never needs flux).
    fams = sorted({family_of(k) for k in model_keys})
    pools = build_pools({f: FAMILY_PROVIDER[f]["key_prefix"] for f in fams})
    sems = {f: asyncio.Semaphore(len(pools[f]) * GEN_PER_KEY_CONCURRENCY) for f in fams}
    total = sum(len(pools[f]) * GEN_PER_KEY_CONCURRENCY for f in fams)
    tqdm.write("\nConcurrency: "
               + " ".join(f"{f}={len(pools[f])}x{GEN_PER_KEY_CONCURRENCY}" for f in fams)
               + f" (total={total})")

    generators = {k: providers.build_generator(spec(k), pools[family_of(k)], mod.temperature, mod.gen_prompt)
                  for k in model_keys}

    results: dict = {}
    if mod.results_file.exists():
        results = json.loads(mod.results_file.read_text())

    bars = {k: tqdm(total=len(words) * mod.n_per_word, desc=k, position=i, leave=True)
            for i, k in enumerate(model_keys)}

    def retryable(msg):
        m = msg.lower()
        return "429" in m or "timeout" in m or "timed out" in m or mod.retry_token in msg

    async def gen_one(model_key, word, idx):
        key = f"{model_key}/{word}/{idx:02d}"
        async with sems[family_of(model_key)]:
            ok, val = await _attempt_with_retry(
                lambda: generators[model_key](word), retryable, key, bars[model_key])
            if ok:
                path = artifact_path(mod.output_dir, word, model_key, idx, mod.ext)
                path.parent.mkdir(parents=True, exist_ok=True)
                mod.save(path, val)
                results.setdefault(word, {}).setdefault(model_key, {})[str(idx)] = "ok"
            else:
                tqdm.write(f"  [err ] {key}: {val}")
                results.setdefault(word, {}).setdefault(model_key, {})[str(idx)] = f"error: {val}"
            bars[model_key].update(1)

    async def run_model(model_key):
        for word in words:
            todo = pending_indices(mod, model_key, word)
            bars[model_key].update(mod.n_per_word - len(todo))
            if todo:
                await asyncio.gather(*(gen_one(model_key, word, i) for i in todo))
            mod.results_file.write_text(json.dumps(results, indent=2))

    await asyncio.gather(*(run_model(k) for k in model_keys))
    for b in bars.values():
        b.close()

    ok = sum(1 for w in results.values() for m in w.values()
             for s in m.values() if s in ("ok", "done"))
    err = sum(1 for w in results.values() for m in w.values()
              for s in m.values() if s.startswith("error"))
    tqdm.write(f"\nDone. {ok} {mod.artifact_word} saved, {err} errors. Results -> {mod.results_file}")


# ── Classification ─────────────────────────────────────────────────────────────

def _classify(mod: ClassifyModality, judge_name, judge_call, meanings, word, artifact):
    labels = allowed_labels(meanings, word)
    senses = sense_keys(meanings, word)
    single = (schemas.single_schema_gpt(labels) if judge_name == "gpt"
              else schemas.single_schema_gemini(labels))
    image = artifact if mod.judge_input_kind == "image" else None

    label = judge_call(mod.build_prompt(meanings, word, artifact), single, image=image)["label"]
    if label not in labels:
        label = "unclear"
    if label != "multiple":
        return label

    multi = (schemas.multi_schema_gpt(senses) if judge_name == "gpt"
             else schemas.multi_schema_gemini(senses))
    got = judge_call(mod.build_multiple_prompt(meanings, word, artifact), multi, image=image)["senses"]
    got = [s for s in dict.fromkeys(got) if s in senses]
    return {"label": "multiple", "senses": got}


def _already_done(results, word, model_key, idx, judge) -> bool:
    return (results.get(word, {}).get(model_key, {})
            .get(str(idx), {}).get(judge) is not None)


def _enumerate_tasks(mod, results, words, model_filter, judges, n_cap):
    for word in words:
        wdir = mod.output_dir / word
        if not wdir.is_dir():
            continue
        for mdir in sorted(wdir.iterdir()):
            if not mdir.is_dir():
                continue
            model_key = mdir.name
            if model_filter and model_key != model_filter:
                continue
            idxs = sorted(int(p.stem) for p in mdir.glob(f"*.{mod.ext}") if p.stem.isdigit())
            if n_cap is not None:
                idxs = idxs[:n_cap]
            for idx in idxs:
                for judge in judges:
                    if not _already_done(results, word, model_key, idx, judge):
                        yield (word, model_key, idx, judge)


async def run_classify(mod: ClassifyModality, meanings, words, judges,
                       model_filter=None, n_cap=None):
    pools = build_pools({"gpt": FAMILY_PROVIDER["openai"]["key_prefix"],
                         "gemini": FAMILY_PROVIDER["gemini"]["key_prefix"]})
    sems = {j: asyncio.Semaphore(len(pools[j]) * PER_KEY_CONCURRENCY) for j in judges}
    judge_calls = {j: providers.build_judge(j, JUDGES[j], pools[j]) for j in judges}

    results = json.loads(mod.class_file.read_text()) if mod.class_file.exists() else {}
    tasks = list(_enumerate_tasks(mod, results, words, model_filter, judges, n_cap))
    by_judge = {j: [t for t in tasks if t[3] == j] for j in judges}
    print("\nPending labels: " + " ".join(f"{j}={len(by_judge[j])}" for j in judges)
          + f"  (total={len(tasks)})")
    if not tasks:
        print("Nothing to do.")
        return

    bars = {j: tqdm(total=len(by_judge[j]), desc=j, position=i, leave=True)
            for i, j in enumerate(judges)}

    def retryable(msg):
        return ("429" in msg or "500" in msg or "503" in msg or "timeout" in msg.lower())

    async def cls_one(word, model_key, idx, judge):
        key = f"{judge}/{word}/{model_key}/{idx:02d}"
        path = artifact_path(mod.output_dir, word, model_key, idx, mod.ext)
        async with sems[judge]:
            artifact = mod.load_artifact(path)
            ok, val = await _attempt_with_retry(
                lambda: _classify(mod, judge, judge_calls[judge], meanings, word, artifact),
                retryable, key, bars[judge])
            if ok:
                (results.setdefault(word, {}).setdefault(model_key, {})
                        .setdefault(str(idx), {}))[judge] = val
            else:
                tqdm.write(f"  [err ] {key}: {val}")
            bars[judge].update(1)

    BATCH = 200
    for i in range(0, len(tasks), BATCH):
        await asyncio.gather(*(cls_one(*t) for t in tasks[i:i + BATCH]))
        mod.class_file.write_text(json.dumps(results, indent=2))

    for b in bars.values():
        b.close()
    mod.class_file.write_text(json.dumps(results, indent=2))
    print(f"\nDone. Labels -> {mod.class_file}")
