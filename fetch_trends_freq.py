"""
For each word, compare senses via Google Trends (2024) using
disambiguating search queries built from meanings.json.
Normalises relative interest across senses → trends_frequencies.json.

Google Trends returns relative values (0–100 within a batch), so we
compare all senses of a word in a single request when ≤5 senses,
or chain batches via an anchor term when >5.
"""
import json, time, re
import numpy as np
from pytrends.request import TrendReq

meanings = json.load(open("meanings.json"))

pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 30), retries=3, backoff_factor=1)


def sense_query(word, sense, meaning):
    """Build a short unambiguous search query for this sense."""
    # Strip trailing parenthetical notes, take first clause
    clean = re.split(r"[;,(]", meaning)[0].strip().rstrip(".")
    # Avoid just repeating the word — add the first distinctive noun/adjective
    words = clean.lower().split()
    # Drop the bare word itself if it appears
    words = [w for w in words if w != word.lower()]
    extra = " ".join(words[:3])
    return f"{word} {extra}".strip()


def fetch_trends(kw_list, timeframe="2024-01-01 2024-12-31"):
    """Return dict {kw: mean_interest} or None on failure."""
    try:
        pytrends.build_payload(kw_list, timeframe=timeframe, geo="")
        df = pytrends.interest_over_time()
        if df.empty:
            return None
        return {kw: float(df[kw].mean()) for kw in kw_list if kw in df.columns}
    except Exception as e:
        print(f"    trends error: {e}")
        return None


try:
    results = json.load(open("trends_frequencies.json"))
    print(f"Resuming — {len(results)} words done.")
except FileNotFoundError:
    results = {}

for word, senses in meanings.items():
    if word in results:
        continue
    sense_list = senses if isinstance(senses, list) else [
        {"sense": k, "meaning": v} for k, v in senses.items()
    ]
    print(f"\n{word}")

    queries = {entry["sense"]: sense_query(word, entry["sense"], entry["meaning"])
               for entry in sense_list}
    print("  queries:", queries)

    # Google Trends allows max 5 keywords per request
    sense_scores = {}
    sense_keys = list(queries.keys())

    if len(sense_keys) <= 5:
        kws = list(queries.values())
        res = fetch_trends(kws)
        if res:
            for sense, kw in queries.items():
                sense_scores[sense] = res.get(kw, 0.0)
        else:
            sense_scores = {s: 0.0 for s in sense_keys}
    else:
        # Batch in groups of 4, using first sense as anchor for normalisation
        anchor_sense = sense_keys[0]
        anchor_kw = queries[anchor_sense]
        anchor_val = None
        batches = [sense_keys[i:i+4] for i in range(0, len(sense_keys), 4)]
        for batch in batches:
            kws = [queries[s] for s in batch]
            if anchor_kw not in kws:
                kws = [anchor_kw] + kws
            res = fetch_trends(kws)
            if res is None:
                for s in batch:
                    sense_scores[s] = 0.0
                continue
            if anchor_val is None:
                anchor_val = res.get(anchor_kw, 1.0) or 1.0
                sense_scores[anchor_sense] = anchor_val
            scale = res.get(anchor_kw, anchor_val) or anchor_val
            for s in batch:
                kw = queries[s]
                sense_scores[s] = res.get(kw, 0.0) * (anchor_val / scale)
            time.sleep(1.5)

    total = sum(sense_scores.values())
    results[word] = ({s: v / total for s, v in sense_scores.items()}
                     if total > 0 else {s: 1 / len(sense_scores) for s in sense_scores})

    print("  →", {s: round(v, 3) for s, v in results[word].items()})
    with open("trends_frequencies.json", "w") as f:
        json.dump(results, f, indent=2)
    time.sleep(2.0)   # be polite to Google

print("\nDone — trends_frequencies.json")
