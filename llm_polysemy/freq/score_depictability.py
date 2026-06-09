"""Score depictability for every (word, sense) pair using Claude (Anthropic).

Claude is not in the test panel (OpenAI/Grok/Gemini/Qwen/FLUX), so it is a valid
independent rater per the T1 pre-registration (preregistration/T1_rule.md §5b).

Usage:
    python -m llm_polysemy.freq.score_depictability          # score all pairs
    python -m llm_polysemy.freq.score_depictability --check  # spot-check sample for human κ

Requires: ANTHROPIC_API_KEY env var.
Writes:   data/depictability.json  (raw scores + rationale, resumable)
          data/depictability_spotcheck.json  (25-word sample, with --check)
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import anthropic

from ..meanings import load_meanings

OUT = Path("data/depictability.json")
SPOTCHECK_OUT = Path("data/depictability_spotcheck.json")
MODEL = "claude-sonnet-4-6"

PROMPT_TEMPLATE = """\
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

Respond with JSON only: {{"score": <integer 1-5>, "rationale": "<one sentence>"}}\
"""


def score_pair(client, word: str, sense_label: str, sense_meaning: str,
               retries: int = 3) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        word=word, sense_label=sense_label, sense_meaning=sense_meaning
    )
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=128,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
            assert "score" in result and "rationale" in result
            assert result["score"] in (1, 2, 3, 4, 5)
            return result
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    ERROR: {e} — defaulting to score=3")
                return {"score": 3, "rationale": f"ERROR: {e}"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true",
                   help="Generate spot-check sample for human inter-rater reliability")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY before running this script.")

    client = anthropic.Anthropic(api_key=api_key)
    meanings = load_meanings()

    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    if results:
        done = sum(len(v) for v in results.values())
        print(f"Resuming — {done} sense-pairs already scored.")

    if args.check:
        _generate_spotcheck(results, meanings, args.seed)
        return

    total_pairs = sum(len(senses) for senses in meanings.values())
    done_count = sum(len(v) for v in results.values())
    print(f"Scoring {total_pairs - done_count} remaining pairs "
          f"({total_pairs} total) with {MODEL}...")

    for word, sense_list in meanings.items():
        if word not in results:
            results[word] = {}
        for entry in sense_list:
            sense = entry["sense"]
            if sense in results[word]:
                continue
            result = score_pair(client, word, sense, entry["meaning"])
            results[word][sense] = result
            print(f"  {word}/{sense}: {result['score']} — {result['rationale']}")
            OUT.write_text(json.dumps(results, indent=2))
            time.sleep(0.15)

    print(f"\nDone — {OUT}")
    _print_summary(results, meanings)


def _generate_spotcheck(results: dict, meanings: dict, seed: int):
    """
    Select 25 words stratified by predicted image-winner depictability tier
    (5 tiers × 5 words). Saves a CSV-friendly JSON for manual human rating.
    """
    if not results:
        raise SystemExit("Run without --check first to produce depictability.json.")

    rng = random.Random(seed)
    # Group words by the raw score of their top depictability sense
    tiers: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for word, sense_scores in results.items():
        if not sense_scores:
            continue
        top_score = max(v["score"] for v in sense_scores.values())
        tiers[top_score].append(word)

    sample_words = []
    for tier in (1, 2, 3, 4, 5):
        words_in_tier = tiers[tier]
        rng.shuffle(words_in_tier)
        sample_words.extend(words_in_tier[:5])

    # Fill up to 25 if any tier was short
    all_words = list(results.keys())
    rng.shuffle(all_words)
    for w in all_words:
        if len(sample_words) >= 25:
            break
        if w not in sample_words:
            sample_words.append(w)

    spotcheck = []
    for word in sample_words[:25]:
        for entry in meanings[word]:
            sense = entry["sense"]
            spotcheck.append({
                "word": word,
                "sense": sense,
                "meaning": entry["meaning"],
                "claude_score": results[word].get(sense, {}).get("score"),
                "claude_rationale": results[word].get(sense, {}).get("rationale"),
                "human_score": None,
            })

    SPOTCHECK_OUT.write_text(json.dumps(spotcheck, indent=2))
    print(f"Spot-check sample written to {SPOTCHECK_OUT} ({len(spotcheck)} rows).")
    print("Fill in 'human_score' (1-5) for each row, then run t1_eval.py --kappa.")


def _print_summary(results: dict, meanings: dict):
    from collections import Counter
    freq = Counter()
    for word_scores in results.values():
        for entry in word_scores.values():
            freq[entry["score"]] += 1
    print("\nScore distribution:")
    for s in (1, 2, 3, 4, 5):
        print(f"  {s}: {freq[s]:3d}")


if __name__ == "__main__":
    main()
