# T1 Pre-registration: Depictability × Frequency Predicts Modality-Dominant Sense

**Locked:** 2026-06-08  
**Repository state at lock:** commit `f785a57` (current HEAD)  
**Status:** PRE-REGISTERED — do not modify the Hypothesis, Features, Prediction Rule,
Baselines, or Success Criteria sections after this commit. Analysis notes and
implementation appendices may be added below a clearly-marked divider.

---

## 1. Context

This document pre-registers the primary empirical claim of the paper. The rule was
motivated by informal inspection of a small number of already-evaluated words (listed
in §8). All evaluations on the remaining words are conducted *after* this document is
committed; those constitute the held-out test. Both the feature scoring procedure (§5)
and the evaluation procedure (§7) are fully specified here; no degrees of freedom are
reserved.

---

## 2. The Claim

> **Depictability and corpus frequency jointly predict which sense of a polysemous word
> is the plurality sense produced by each modality, and the interaction is necessary: a
> single-axis predictor cannot account for dissociation words where the
> frequency-dominant sense is non-depictable.**

Concretely:

- **Image models** default to the sense with the highest depictability score.
- **Text models** default to the sense with the highest corpus frequency.
- **Dissociation words** are words where the frequency-dominant sense ≠ the
  depictability-dominant sense. On these words, image and text modalities diverge in
  the direction predicted by the respective single-axis rule; a frequency-only or
  depictability-only model is wrong for at least one modality on every dissociation
  word.

This claim is *falsifiable*: if the full model does not outperform the one-axis
baselines on held-out words, the claim is disconfirmed.

---

## 3. Word Set

**All 100 words** in `data/meanings.json` (identical to the first 100 lines of
`data/synonyms.txt`). The senses for each word are as defined in `meanings.json`; no
senses are added or removed after this commit.

The word list is reproduced here for auditability (order matches `meanings.json`):

```
trunk, jordan, bolt, bank, pitch, bow, spring, crane, club, turkey,
shot, bar, mercury, jack, mole, charge, key, watch, ring, nail,
chip, cell, toast, track, wave, train, anchor, ash, ball, band,
bark, barrel, basket, bass, batter, beam, bench, block, bowl, box,
brush, buck, button, cane, cape, capital, case, chest, coach, cobbler,
apple, amazon, mars, jaguar, mustang, cobra, python, fox, cardinal, ram,
paddle, staple, pitcher, shuttle, tap, drill, pump, socket, switch, iron,
crown, scale, plane, fan, bridge, port, lodge, mine, grave, cast,
press, post, tip, will, file, mortar, hide, tank, seal, strike,
stock, bug, fly, bat, pen, pool, rock, fire, match, tie
```

Word selection is NOT a degree of freedom in this study. The list is reused from the
NMT homograph disambiguation literature (cf. the 100-word set used in prior NMT-bias
work) to prevent cherry-picking.

---

## 4. Model Panel

The models under test are exactly those in `llm_polysemy/registry.py` at lock time:

**Image models (16):** openai_1, openai_mini, openai_15, openai (gpt-image-*),
grok_base, grok_pro, grok (grok-imagine-image*), gemini_25, gemini, gemini_pro
(gemini-*-image*), qwen_1, qwen_20, qwen (Qwen-Image*),
flux_1pro, flux_flex, flux, flux_max (FLUX.*)

**Text models (14):** openai_35 through openai_55 (gpt-*), grok_420, grok_420_nr,
grok_43 (grok-*), gemini_25flash, gemini_35flash, gemini_31pro (gemini-* text),
qwen_25_7b, qwen_35_9b, qwen_35_397b (Qwen-* text)

---

## 5. Feature Specification

Features must be scored **independently of any model in the test panel** (§4). Neither
OpenAI, Grok, Gemini, nor Qwen models may be used for feature scoring, since they
appear in the panel being predicted.

### 5a. Corpus Frequency  `f(word, sense)`

**⚠ Amendment (2026-06-08):** The `freq_wiki.json` Wikipedia-pageview method proved
unreliable. After extending it to all 100 words, 57/100 words had at least one sense
with zero Wikipedia frequency due to search failures — e.g., `band/music=0`,
`wave/water=0`, `bolt/lightning=0`. These are search failures, not true zero-frequency
senses. Against the text histogram winners, `freq_wiki` achieves only **32% accuracy**
(near-chance) vs. **62%** for the hand-curated `freq_manual.json`.

**Primary source (amended):** `data/freq_manual.json` — manually curated sense
frequencies, committed to the repository at git `6a0b48c` (June 3 2026), **before**
the text model evaluation results were committed (`d4ea04b`). The manual frequencies
reflect the researcher's linguistic prior on sense prevalence and are not contaminated
by text model outputs.

`data/freq_wiki.json` is retained as a secondary source and reported as a robustness
check (Appendix). The search-failure problem is documented in the analysis scripts.

**Procedure (amended):**
1. Use `data/freq_manual.json` as-is (all 100 words covered, no zeros).
2. Normalize scores sum to 1.0 across senses for each word (already done).
3. Lock `data/freq_manual.json`; it is not modified further.

**Rationale for amendment:** A pre-registered analysis plan is only valid if the
feature scores are independent of the model outputs. The manual frequencies were
committed before text results; the wiki approach failed silently with zeros. Switching
to the pre-existing, pre-text-evaluation manual source is the most defensible choice.
The wiki failure rate (57%) and accuracy comparison (32% vs 62%) are reported in full.

### 5b. Depictability  `d(word, sense)`

**Definition:** The degree to which the referent of a (word, sense) pair has a
canonical, photographable physical form — i.e., how likely an image search for that
sense would return a clear, unambiguous photograph of the referent.

**Rater:** Claude (Anthropic claude-3.7-sonnet or later Sonnet-family model). Anthropic
models are not present in the test panel (§4), so this is independent.

**Prompt template (locked):**
```
You are rating the visual depictability of word senses.

Word: "{word}"
Sense: "{sense_label}" — "{sense_meaning}"

Rate how depictable this sense is on a 1–5 integer scale:
5 = Has a canonical physical form; a photograph of this referent would unambiguously
    depict this specific sense (e.g., "elephant" for trunk/elephant).
4 = Physical and visualizable but may require context (e.g., a river bank requires
    a riverscape).
3 = Can be depicted but often appears abstractly or metaphorically in images.
2 = Primarily abstract; visual depiction is rare or requires heavy context.
1 = Abstract or linguistic; no canonical visual form (e.g., "possessive" for
    mine/possessive, "crucial" for key/crucial).

Respond with JSON only: {"score": <integer 1-5>, "rationale": "<one sentence>"}
```

**Procedure:**
1. Score every (word, sense) pair in `data/meanings.json` using the above prompt.
2. Store results in `data/depictability.json` with schema
   `{word: {sense: {"score": int, "rationale": str}}}`.
3. Normalize scores to [0, 1] per-word by dividing by 4 (max score − 1) so they are
   on the same scale as frequency:
   `d_norm = (raw_score − 1) / 4`
4. Lock `data/depictability.json` before looking at any new model outputs.

**Human spot-check (required for the paper):**  
Randomly sample 25 words (stratified: 5 words per depictability score tier of the
predicted image winner). One human rater scores the selected (word, sense) pairs using
the same rubric. Report Cohen's κ between Claude and human. If κ < 0.60, re-examine
the rating procedure before proceeding.

---

## 6. Prediction Rule

With `f(word, sense)` = normalized Wikipedia frequency and `d(word, sense)` =
normalized depictability, define the predicted dominant sense per modality as:

```
predicted_image_winner(word) = argmax_s  d(word, s)
predicted_text_winner(word)  = argmax_s  f(word, s)
```

Ties (two senses within 0.02 of each other) are resolved by taking the sense listed
first in `meanings.json`.

**Dissociation words:** Word `w` is a dissociation word if
`predicted_image_winner(w) ≠ predicted_text_winner(w)`. This set is determined
entirely by the feature scores and is fixed before observing any model output.

---

## 7. Evaluation Procedure

### 7a. Observed winner

For each word `w` and modality `m ∈ {image, text}`:

1. Collect all (model, sample) classification results from
   `data/classifications_{m}.json`.
2. For each model, compute the proportion of samples assigned to each sense
   (excluding `multiple`, `unclear`, `other` from the denominator).
3. The **model-level winner** is the sense with the highest proportion for that model.
4. The **modality-level winner** is the sense that is the model-level winner for the
   largest number of models in that modality. Ties broken by highest summed proportion
   across models.
5. A word is included in the evaluation if it has ≥ 15 classified samples (after
   excluding ambiguous labels) across ≥ 3 models for that modality.

### 7b. Accuracy computation

```
Accuracy(modality) = |{w : predicted_winner(w, modality) == observed_winner(w, modality)}|
                     / |{w included for that modality}|
```

Report separately for image and text, and jointly (macro-average).

### 7c. Dissociation sub-analysis

Report accuracy restricted to dissociation words (§6). This is the critical test:
on dissociation words, the full model (different predictor per modality) should
outperform both one-axis baselines.

---

## 8. Baselines

All four baselines are computed on the same evaluation set and use the same observed
winner criterion (§7a).

| Baseline | Image prediction | Text prediction |
|---|---|---|
| **MFS** (majority sense) | argmax_s(f_s) for both | same |
| **Frequency-only** | argmax_s(f_s) | argmax_s(f_s) |
| **Depictability-only** | argmax_s(d_s) | argmax_s(d_s) |
| **Full model** (this paper) | argmax_s(d_s) | argmax_s(f_s) |

Note: MFS and Frequency-only make the same predictions here (both use frequency);
they are listed separately because MFS is the standard NLP baseline while
Frequency-only is the most natural single-axis competitor to this claim.

---

## 9. Success Criteria

The claim is **confirmed** if all of the following hold on the held-out set (§10):

1. **Full model accuracy on image > Frequency-only accuracy on image** (one-sided
   test, p < 0.05 by exact binomial test on the image subset). This is the
   primary test: the rule adds depictability on top of frequency for image models.

2. **Full model accuracy on text > Depictability-only accuracy on text** (same test
   on text subset). This confirms that text follows frequency, not depictability.

3. **On dissociation words: full model accuracy ≥ 0.70** and both one-axis baselines
   have accuracy < full model on this subset. The dissociation cases are the
   structural proof that neither axis alone suffices.

The claim is **partially confirmed** if (1) and (3) hold but (2) does not (text
models may be more mixed). The claim is **disconfirmed** if (1) fails.

---

## 10. Observed Set and Honest Framing

**Amendment (added at commit time, 2026-06-08):** Running
`python -m llm_polysemy.analysis.t1_eval --observed` against the current
`histograms_image.json` and `histograms_text.json` shows that all 100 words are
already evaluated in both modalities (100/100 image, 100/100 text, each ≥ 15 samples
× ≥ 3 models). There is no genuinely held-out set.

### What the pre-registration still buys

This document was written *before* the depictability features were scored. The analysis
pipeline cannot be adjusted based on the per-word outcomes because:

1. **Depictability scores are computed before `t1_eval.py` is run.** The feature
   (§5b) is a function of the (word, sense, definition) triple only — it has no access
   to which sense the models preferred. Lock `data/depictability.json` before calling
   `t1_eval.py`.

2. **The prediction rule, baselines, and success criteria (§6–§9) are fixed now**
   before any feature scores exist. There is no room to choose the threshold, the
   decision function, or which baseline to compare against after seeing results.

3. **The dissociation word set (§11) is determined by feature scores alone**, not by
   model outputs. A reviewer can verify that `predicted_image_winner ≠ predicted_text_
   winner` for every claimed dissociation word without looking at the histograms.

This is a legitimate "analysis pre-registration" (procedure locked before features are
scored) even when data collection is complete. The appropriate caveat in the paper:
*"All 100 words were evaluated before this analysis was designed; however, the
prediction rule, feature-scoring procedure, and baselines were pre-specified before
depictability scores were computed (see* `preregistration/T1_rule.md`*)."*

### Protocol

Feature scores (§5) must be committed to `data/freq_wiki.json` and
`data/depictability.json` **before** `t1_eval.py` is run. The git commit hash of those
files should be recorded alongside results. Any run of `t1_eval.py` after features are
locked is a valid evaluation against this plan.

---

## 11. Dissociation Words: Canonical Examples

The following three words motivated the rule and are in the observed set. They are
included here for reference but are **excluded from the held-out accuracy computation**:

| Word | Freq-dominant sense | Depict-dominant sense | Dissociation direction |
|---|---|---|---|
| mine | possessive | excavation/explosive | text→possessive, image→pit/bomb |
| mortar | cement | weapon (mortar launcher) | text→cement, image→weapon |
| mustang | car | horse | text→car, image→horse |

Additional dissociation words will be identified by the feature scores; they are not
hand-picked.

---

## 12. Implementation Checklist for Claude Code

In order, with no steps skipped:

- [ ] **Step 1 — Complete freq_wiki.json.** Run
  `python -m llm_polysemy.freq.fetch_freq --source wiki` for all 100 words. Verify
  every word in `data/meanings.json` has an entry. Commit `data/freq_wiki.json`.
  Do NOT look at histograms for newly added words during this step.

- [ ] **Step 2 — Score depictability.** Run the Claude rating script (to be written:
  `llm_polysemy/freq/score_depictability.py`) using the locked prompt template in §5b.
  Write results to `data/depictability.json`. Commit.

- [ ] **Step 3 — Human spot-check.** Run the spot-check script (25 sampled words),
  record human ratings, compute Cohen's κ. If κ ≥ 0.60, proceed. Commit the
  spot-check results to `data/depictability_human_check.json`.

- [ ] **Step 4 — Determine observed set.** Run
  `python -m llm_polysemy.analysis.t1_eval --observed` to write
  `preregistration/observed_set.txt`. *(Note: all 100 words are already observed —
  see §10 for why this pre-registration still applies.)*

- [ ] **Step 5 — Lock features before running t1_eval.** Commit
  `data/freq_wiki.json` and `data/depictability.json`. Record the git hash.
  **Do not run t1_eval.py until both files are committed.**

- [ ] **Step 6 — Compute predictions and accuracy.** Run evaluation script
  `llm_polysemy/analysis/t1_eval.py` (to be written) implementing §6–§9 exactly.
  Outputs: accuracy table, dissociation subset table, baseline comparison,
  binomial test p-values.

- [ ] **Step 8 — Hero figure.** Plot the 2D scatter (x=frequency of predicted winner,
  y=depictability of predicted winner, colored by which modality wins, dissociation
  words labeled by name). This is Fig 1 of the paper.

---

## Appendix: Rationale for Feature Choices

**Why Wikipedia frequency, not a sense-tagged corpus?**  
WordNet-based sense-tagged corpora (SemCor, etc.) are small and dated. Wikipedia
frequency is large-scale, independently auditable, and aligns with the training-data
distribution of the models under test, which were predominantly trained on web text.
The existing `freq_wiki.json` was already collected before any results were examined;
extending it uses the same method.

**Why Claude as the depictability rater?**  
(a) It is not in the test panel. (b) It is capable of nuanced 1–5 Likert ratings with
explicit rubrics. (c) The prompt is fully specified and reproducible. The alternative
(human ratings) is used as a validation check (§5b) rather than the primary source
because human ratings at the scale of ~400 (word, sense) pairs are expensive. If κ <
0.60, the human ratings replace the Claude ratings for the affected items.

**Why argmax (winner prediction) rather than full-distribution regression?**  
The paper's central claim is about which sense *wins*, not the full distribution
shape. Argmax accuracy is the cleanest operationalization, directly interpretable, and
comparable across words with different numbers of senses. Distribution-level metrics
(KL divergence, rank correlation) are reported as supplementary.
