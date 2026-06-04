# llm-polysemy

A benchmark probing how **text-to-image** and **text** models resolve **polysemous
words** — words with multiple meanings (e.g. *bank* → finance vs river, *bolt* →
fastener vs lightning). Each model is given the **bare word** with no disambiguating
context; we sample many times and observe which sense it picks, then compare the
sense distributions across models, model generations, families, and modalities.

## The pipeline (4 stages)

```
 generate  ─▶  classify  ─▶  histograms  ─▶  plot / compare
 (eval)        (judges)      (--histograms)   (plots/*)
```

1. **generate** — for each word, sample 30 outputs per model.
   - *image*: prompt = the bare word → a PNG.
   - *text*: prompt = `Use the following word in a single sentence: <word> / Reply with
     only the sentence and nothing else.` → a sentence.
2. **classify** — two judges (GPT + Gemini) label each output by which sense it uses.
3. **histograms** — aggregate the per-sample labels into per-(word, model, judge)
   sense distributions.
4. **plot / compare** — figures per word, per family, across generations, vs a
   text-frequency prior, and a cross-modality image-vs-text comparison.

Everything is driven by a single model registry (`llm_polysemy/registry.py`) — adding
a model is one row there.

## Setup

```bash
pip install openai google-genai python-dotenv tqdm numpy scipy scikit-learn matplotlib requests
# (pytrends only if you fetch Google Trends priors)
```

API keys live in `.env` (gitignored), gathered into per-provider pools **by prefix** —
add capacity by adding suffixed keys:

```
OPENAI_API_KEY=...        OPENAI_API_KEY_2=...     # OpenAI (GPT + gpt-image)
XAI_API_KEY=...                                    # xAI Grok
GEMINI_API_KEY=...                                 # Google Gemini
TOGETHER_API_KEY_QWEN=...                          # Qwen   (Together.ai)
TOGETHER_API_KEY_BF=...                            # FLUX   (Together.ai / Black Forest Labs)
```
A provider family runs exactly as many concurrent requests as it has keys.

## Commands

All commands take `--modality {image,text}`. Canonical form is `python -m
llm_polysemy.<entrypoint>`; thin root shims (`python eval.py`, `python eval_text.py`,
`python classify.py`, `python classify_text.py`, `python agreement.py`) still work.

### Generate

```bash
python -m llm_polysemy.eval --modality image                 # all image models
python -m llm_polysemy.eval --modality text                  # all text models
python -m llm_polysemy.eval --modality text --model openai_55 # one model
```
Resume is automatic & filesystem-driven: only missing outputs are regenerated
(delete an output file to re-queue just that sample).

### Classify

```bash
python -m llm_polysemy.classify --modality image                       # both judges
python -m llm_polysemy.classify --modality text --word bank --n 1      # cheap test slice
python -m llm_polysemy.classify --modality image --judge gemini        # one judge
python -m llm_polysemy.classify --modality image --histograms          # build histograms
python -m llm_polysemy.classify --modality text  --histograms
```

### Inter-judge agreement

```bash
python -m llm_polysemy.agreement --modality image      # raw agreement + Cohen's kappa
```

### Plots

Each plot accepts `--modality {image,text}` and `--judge {gpt,gemini,consensus}`
(default `gpt`); figures land under `figures/<modality>/` (or `figures/cross/`):

```bash
python -m llm_polysemy.plots.histograms        --modality image          # per-word sense bars
python -m llm_polysemy.plots.temporal          --modality image          # per-family, across generations
python -m llm_polysemy.plots.similarity        --modality image          # JS similarity matrix + family cohesion
python -m llm_polysemy.plots.pca_tsne          --modality image          # PCA / t-SNE of model distributions
python -m llm_polysemy.plots.word_disagreement --modality image          # most agreed / disagreed words
python -m llm_polysemy.plots.vs_textfreq       --modality text --prior manual   # vs a text-frequency prior
python -m llm_polysemy.plots.generation_trend  --modality text --prior manual   # prior alignment vs generation
python -m llm_polysemy.plots.cross_modality    --agg mean                # image-vs-text per family (needs both)
```

### Frequency priors (for the `vs_textfreq` / `generation_trend` plots)

```bash
python -m llm_polysemy.freq.build_freq                       # hand-curated -> data/freq_manual.json
python -m llm_polysemy.freq.fetch_freq --source wiki         # Wikipedia views -> data/freq_wiki.json
python -m llm_polysemy.freq.fetch_freq --source trends       # Google Trends  -> data/freq_trends.json
```

### Publish the corpus

```bash
./upload_dataset.sh image      # or: ./upload_dataset.sh text
```

## What you get (artifacts)

All generated data lives under `data/`, all figures under `figures/`:

| Path | What |
|---|---|
| `data/outputs_image/<word>/<model>/<NN>.png` | generated images |
| `data/outputs_text/<word>/<model>/<NN>.txt` | generated sentences |
| `data/results_{image,text}.json` | per-sample generation status log |
| `data/classifications_{image,text}.json` | per-sample judge labels |
| `data/histograms_{image,text}.json` | aggregated sense distributions |
| `data/meanings.json` | candidate senses per word |
| `data/synonyms.txt` | the word list (first 100 used) |
| `data/freq_{manual,wiki,trends}.json` | text-frequency priors |
| `figures/{image,text}/...` | per-modality figures |
| `figures/cross/...` | cross-modality (image vs text) figures |

## Adding a model

Add **one `ModelSpec` row** to `llm_polysemy/registry.py` (key, api_id, family,
modality, label, generation_index). Everything else — pools, concurrency, dispatch,
plot lists, family grouping, colors — derives from it. If you add a brand-new provider
family, also add its entry to `FAMILY_PROVIDER` (SDK / base_url / key prefix).

## Layout

```
llm_polysemy/
  registry.py        # SINGLE SOURCE OF TRUTH: every model + families/colors/gen-order
  config.py          # Modality config objects (image vs text) + JUDGES
  keys.py            # API-key pools (prefix-gathered, round-robin)
  meanings.py schemas.py prompts.py histograms.py agreement.py
  plotting_utils.py  # shared plot helpers + judge resolver
  runner.py          # async engine (generation + classification)
  providers/         # openai_sdk.py, google_genai.py
  eval.py classify.py # merged --modality entrypoints
  plots/             # 7 modality-aware plots + cross_modality.py
  freq/              # build_freq.py, fetch_freq.py
eval.py eval_text.py classify.py classify_text.py agreement.py   # thin root shims
data/  figures/
```

There is no test suite or build step — this is a research/eval repo.
