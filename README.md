# llm-polysemy

## What is novel here

Prior work establishes that language and vision models have **sense bias** (Rassin et al.
2022; the MFS literature) and that models cluster by provider family (McGovern et al.
2024; TensorGuard; "What Makes Two LMs Think Alike" 2025). This work adds two things
that survive prior-art search:

**1. A predictive rule.**  
Depictability × corpus-frequency predicts which sense each modality defaults to,
*including* the dissociation case where the frequency-dominant text sense is
non-depictable (*mine* → possessive). Prior work observes the bias; no prior work
gives a falsifiable rule for which sense wins per modality.  
See `preregistration/T1_rule.md` for the pre-registered analysis plan.

**2. Cross-modality + cross-architecture placement.**  
Image and text generators are placed in one sense-distribution similarity space.
The "superposition / both-senses" behavior — where a model depicts multiple senses at
once — appears to be architecture-bound: confirmed-diffusion models (FLUX.2) show
**6.6× higher** multi-sense rates than closed-architecture models (0.9%).
The fingerprinting literature is text-only; White & Cotterell (2022) proposed
superposition as diffusion-specific but did not test it across the current model zoo.

---

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
python -m llm_polysemy.eval --modality text --lang tr        # Turkish word list + Turkish prompt
python -m llm_polysemy.eval --modality text-imagine          # text models, "what image comes to mind?" (en only)
```
Resume is automatic & filesystem-driven: only missing outputs are regenerated
(delete an output file to re-queue just that sample).

**Languages (`--lang {en,tr,fr}`, default `en`).** `en` uses `data/synonyms.txt`
(100 words) with the English prompt and the original output paths. `tr` / `fr`
use `data/synonyms_{tr,fr}.txt` (25 words each) with a Turkish / French generation
prompt and write to **`_<lang>`-suffixed** paths (`outputs_text_tr/`,
`histograms_text_fr.json`, senses from `data/meanings_<lang>.json`) so the languages
never collide. The image prompt is the bare word, so it's already language-correct.
Pass the same `--lang` to `classify`. For `tr`, the default run **drops the Qwen and
FLUX families** (`LANG_EXCLUDE_FAMILIES` in `eval.py`) — image `tr` runs 10 models,
text `tr` runs 12; `fr` currently runs all families. An explicit `--model KEY` still
runs any single model.

**`--modality text-imagine`** is a third generation condition (English only): the
same text models, but prompted *"What image comes to mind when you think of the word
`<word>`? Describe it briefly."* — the verbatim wording of the human Prolific `image`
survey. It's a framing-matched control: holding the substrate fixed (still an LLM) it
isolates *task framing* (depict vs use-in-a-sentence) from *modality*, so its sense
distribution (`histograms_text_imagine.json`) is directly comparable to both the human
`image` baseline and the real image models. Outputs/logs use `_text_imagine` paths.

### Classify

```bash
python -m llm_polysemy.classify --modality image                       # both judges
python -m llm_polysemy.classify --modality text --word bank --n 1      # cheap test slice
python -m llm_polysemy.classify --modality image --judge gemini        # one judge
python -m llm_polysemy.classify --modality image --histograms          # build histograms
python -m llm_polysemy.classify --modality text  --histograms
python -m llm_polysemy.classify --modality text-imagine                # classify the imagine run
python -m llm_polysemy.classify --modality text-imagine --histograms   # -> histograms_text_imagine.json
python -m llm_polysemy.classify --modality text --lang tr              # classify the Turkish run
python -m llm_polysemy.classify --modality text --lang tr --histograms
```

### Human baseline (Prolific) classification

Classify the human Prolific responses (`data/human_responses.json`, the same
100-word English panel) with the **same GPT + Gemini judges**, in two conditions
that mirror the LLM modalities — `meaning` (used the word in a sentence ≈ text)
and `image` (described the image it evokes ≈ image). Human-framed judge prompts
live in `prompts.human_*`; senses come from `data/meanings.json`.

```bash
python -m llm_polysemy.classify_human --convert            # human_responses.json -> file tree
python -m llm_polysemy.classify_human --condition both     # classify (both judges)
python -m llm_polysemy.classify_human --histograms         # -> histograms_human_{meaning,image}.json
```
`histograms_human_meaning.json` / `histograms_human_image.json` are directly
comparable to `histograms_text.json` / `histograms_image.json`.

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

### T1/T2 predictive-rule analysis

```bash
# Score depictability (requires ANTHROPIC_API_KEY; run once, resumable)
python -m llm_polysemy.freq.score_depictability

# Spot-check sample for human inter-rater κ (fill human_score in output JSON, then:)
python -m llm_polysemy.freq.score_depictability --check
python -m llm_polysemy.analysis.t1_eval --kappa

# Pre-registered accuracy table + baselines + dissociation analysis
python -m llm_polysemy.analysis.t1_eval

# Hero Fig A: depictability × frequency scatter
python -m llm_polysemy.analysis.t1_plot

# T2: superposition rate by image model architecture
python -m llm_polysemy.analysis.t2_architecture

# T3: generate human-labeling sample (180 cells, 90 image / 90 text)
python -m llm_polysemy.analysis.t3_judge_validation --sample
# (fill human_label in data/t3_human_sample.json, then:)
python -m llm_polysemy.analysis.t3_judge_validation --report

# T4: cross-modality PCA — all image + text models in one sense space
python -m llm_polysemy.analysis.t4_cross_pca
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
| `data/outputs_text_imagine/<word>/<model>/<NN>.txt` | `text-imagine` mental-image descriptions (en) |
| `data/results_{image,text,text_imagine}.json` | per-sample generation status log |
| `data/classifications_{image,text,text_imagine}.json` | per-sample judge labels |
| `data/histograms_{image,text,text_imagine}.json` | aggregated sense distributions |
| `data/meanings.json` | candidate senses per word |
| `data/synonyms.txt` | the word list (first 100 used) |
| `data/synonyms_{tr,fr}.txt` | Turkish / French word lists (25 words each, `--lang {tr,fr}`) |
| `data/meanings_{tr,fr}.json` | Turkish / French candidate senses (English glosses for the judges + `meaning_<lang>`) |
| `data/{outputs,results,classifications,histograms}_*_{tr,fr}.*` | non-English run artifacts (`_<lang>` suffix) |
| `data/freq_{manual,wiki,trends}.json` | text-frequency priors |
| `data/depictability.json` | depictability scores per (word, sense), rated by Claude |
| `data/model_architectures.json` | architecture labels for image models (T2) |
| `preregistration/T1_rule.md` | pre-registered analysis plan for the predictive rule |
| `figures/{image,text}/...` | per-modality figures |
| `figures/cross/...` | cross-modality (image vs text) figures |
| `figures/t1_hero.pdf` | 2D depictability × frequency scatter (hero fig A) |
| `figures/t2_superposition.pdf` | superposition rate by image model architecture (hero fig B) |
| `figures/t4_cross_pca.pdf` | cross-modality PCA: all models in one sense space |
| `data/t3_human_sample.json` | 180-cell human-labeling sample for judge validation |

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
  freq/              # build_freq.py, fetch_freq.py, score_depictability.py
  analysis/          # t1_eval.py (T1 rule), t2_architecture.py (T2 superposition)
preregistration/     # T1_rule.md (pre-registered analysis plan), observed_set.txt
eval.py eval_text.py classify.py classify_text.py agreement.py   # thin root shims
data/  figures/
```

There is no test suite or build step — this is a research/eval repo.
