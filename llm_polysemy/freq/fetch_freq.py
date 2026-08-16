"""Fetch a text-frequency prior per word+sense, normalized across senses.

Two sources (merged from the old fetch_text_freq/fetch_wiki_freq/fetch_trends_freq):

    python -m llm_polysemy.freq.fetch_freq --source wiki   --out freq_wiki.json
    python -m llm_polysemy.freq.fetch_freq --source trends --out freq_trends.json

- wiki:   best-matching Wikipedia article per sense, summed 2024 monthly pageviews.
- trends: Google Trends 2024 mean interest for a disambiguating query per sense
          (batched via an anchor term when a word has >5 senses).

Resumable: saves after every word.
"""

import argparse
import json
import re
import time
from pathlib import Path

from ..meanings import load_meanings

WIKI_PAGEVIEW = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
    "/en.wikipedia/all-access/all-agents/{title}/monthly/20240101/20241201"
)


# ── Wikipedia source ───────────────────────────────────────────────────────────

def _wiki_fns():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "llm-polysemy-research/1.0 (academic)"

    def search(query, retries=5):
        for attempt in range(retries):
            try:
                r = s.get("https://en.wikipedia.org/w/api.php", params={
                    "action": "query", "list": "search",
                    "srsearch": query, "format": "json", "srlimit": 1}, timeout=12)
                if not r.text.strip() or r.status_code == 429:
                    time.sleep(2 ** attempt); continue
                hits = r.json().get("query", {}).get("search", [])
                return hits[0]["title"] if hits else None
            except Exception:
                time.sleep(2 ** attempt)
        return None

    def views(title, retries=3):
        url = WIKI_PAGEVIEW.format(title=title.replace(" ", "_"))
        for attempt in range(retries):
            try:
                r = s.get(url, timeout=12)
                if r.status_code == 429:
                    time.sleep(2 ** attempt); continue
                if r.status_code != 200:
                    return 0
                return sum(i["views"] for i in r.json().get("items", []))
            except Exception:
                time.sleep(2 ** attempt)
        return 0

    def score_word(word, sense_list):
        sv = {}
        for entry in sense_list:
            sense, meaning = entry["sense"], entry["meaning"]
            title = search(f"{word} {meaning}")
            if title is None:
                print(f"  [{sense}] not found"); sv[sense] = 0; continue
            v = views(title)
            print(f"  [{sense}] '{title}' → {v:,}")
            sv[sense] = v
            time.sleep(0.4)
        return sv

    return score_word


# ── Google Trends source ───────────────────────────────────────────────────────

def _trends_fns():
    from pytrends.request import TrendReq
    pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 30), retries=3, backoff_factor=1)

    def sense_query(word, meaning):
        clean = re.split(r"[;,(]", meaning)[0].strip().rstrip(".")
        words = [w for w in clean.lower().split() if w != word.lower()]
        return f"{word} {' '.join(words[:3])}".strip()

    def fetch(kw_list, timeframe="2024-01-01 2024-12-31"):
        try:
            pytrends.build_payload(kw_list, timeframe=timeframe, geo="")
            df = pytrends.interest_over_time()
            if df.empty:
                return None
            return {kw: float(df[kw].mean()) for kw in kw_list if kw in df.columns}
        except Exception as e:
            print(f"    trends error: {e}")
            return None

    def score_word(word, sense_list):
        queries = {e["sense"]: sense_query(word, e["meaning"]) for e in sense_list}
        print("  queries:", queries)
        keys = list(queries)
        scores = {}
        if len(keys) <= 5:
            res = fetch(list(queries.values()))
            for sense, kw in queries.items():
                scores[sense] = res.get(kw, 0.0) if res else 0.0
        else:
            anchor_sense = keys[0]
            anchor_kw = queries[anchor_sense]
            anchor_val = None
            for batch in (keys[i:i + 4] for i in range(0, len(keys), 4)):
                kws = [queries[s] for s in batch]
                if anchor_kw not in kws:
                    kws = [anchor_kw] + kws
                res = fetch(kws)
                if res is None:
                    for s in batch:
                        scores[s] = 0.0
                    continue
                if anchor_val is None:
                    anchor_val = res.get(anchor_kw, 1.0) or 1.0
                    scores[anchor_sense] = anchor_val
                scale = res.get(anchor_kw, anchor_val) or anchor_val
                for s in batch:
                    scores[s] = res.get(queries[s], 0.0) * (anchor_val / scale)
                time.sleep(1.5)
        time.sleep(2.0)
        return scores

    return score_word


SOURCES = {"wiki": _wiki_fns, "trends": _trends_fns}
DEFAULT_OUT = {"wiki": "data/freq_wiki.json", "trends": "data/freq_trends.json"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=list(SOURCES), required=True)
    p.add_argument("--out", default=None, help="output JSON (default: freq_<source>.json)")
    args = p.parse_args()
    out = Path(args.out or DEFAULT_OUT[args.source])

    meanings = load_meanings()
    score_word = SOURCES[args.source]()

    results = json.loads(out.read_text()) if out.exists() else {}
    if results:
        print(f"Resuming — {len(results)} words done.")

    for word, sense_list in meanings.items():
        assert isinstance(sense_list, list), f"meanings[{word!r}] must be list form"
        if word in results:
            continue
        print(f"\n{word}")
        sv = score_word(word, sense_list)
        total = sum(sv.values())
        results[word] = ({s: v / total for s, v in sv.items()} if total > 0
                         else {s: 1 / len(sv) for s in sv})
        out.write_text(json.dumps(results, indent=2))

    print(f"\nDone — {out}")


if __name__ == "__main__":
    main()
