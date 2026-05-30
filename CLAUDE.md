# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A benchmark probing how text-to-image models resolve **polysemous words** (words with
multiple meanings, e.g. "bank", "trunk", "bolt"). For each word, the bare word alone is
sent as the prompt — no disambiguating context — and the model's chosen meaning is observed
across many samples. `meanings.json` enumerates the candidate senses per word so generated
images can later be classified/histogrammed by which sense the model picked.

## Commands

```bash
# Generate images for ALL models (resumes automatically; skips images already on disk)
python eval.py

# Generate for a single model (keys from the MODELS dict in eval.py)
python eval.py --model flux
python eval.py --model gemini_pro

# Publish the outputs/ image corpus to Hugging Face (requires `hf` CLI auth)
./upload_dataset.sh
```

There is no test suite, linter, or build step — this is a research/eval repo.

## How `eval.py` works

- **Words** come from `synonyms.txt` (one per line); `main()` uses the first 100.
- **30 samples per word per model** (`N_PER_WORD`). The prompt sent to the model is the
  raw word string, nothing else.
- **Output layout**: `outputs/<word>/<model_key>/<NN>.png` (word first, then model).
  Note the module docstring says `<model>/<word>` — that is stale; trust `image_path()`.
- **Resume is filesystem-driven**: `pending_indices()` regenerates only images whose PNG
  is missing. Deleting a PNG re-queues it; `results.json` is a status log, not the source
  of truth for what gets regenerated.
- **`results.json`** records per-image `"ok"` / `"error: ..."` status, keyed
  `[word][model_key][idx]`. Rewritten after each word completes.

### Concurrency / API key model

This is the trickiest part of the architecture:

- Each provider family (openai, grok, gemini, qwen, flux) has its own **`_KeyPool`** of API
  keys gathered from environment variables by **prefix match** (`_gather_keys`). Any env var
  whose name starts with the prefix is included, so multiple keys are added by suffixing:
  `OPENAI_API_KEY`, `OPENAI_API_KEY_7`, etc.
- Prefixes → pools: `openai_api_key` → OpenAI, `xai_api_key` → Grok, `gemini_api_key` →
  Gemini, `together_api_key_qwen` → Qwen, `together_api_key_bf` → FLUX (Black Forest Labs).
  **Qwen and FLUX both run on Together.ai but draw from separate key pools.**
- Per-family `asyncio.Semaphore` sized to `len(pool)` — so a family runs exactly as many
  concurrent requests as it has keys. Keys are handed out round-robin per request.
- Blocking SDK calls are pushed to threads via `asyncio.to_thread`. Retries: up to 3
  attempts, only on `429` or `"no image"` errors, with 2s/4s backoff.

### Provider integration

All providers except Gemini use the **OpenAI SDK** pointed at different `base_url`s
(`api.x.ai`, `api.together.xyz`). Gemini uses `google.genai`. Each has a `_*_generate`
function returning raw PNG bytes; the `GENERATORS` dispatch table maps each model_key to a
generator. To add a model: add to `MODELS`, `_FAMILY`, and `GENERATORS`.

## Environment

API keys live in `.env` (gitignored, loaded via `python-dotenv`). Add capacity by adding
more suffixed keys (e.g. `XAI_API_KEY_4`) — they are auto-discovered by prefix.

## Gotchas

- `outputs/` and `results.json` are large generated artifacts; `outputs/` is gitignored.
  Some words' image dirs are intentionally sparse (model errored or was skipped).
