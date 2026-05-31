"""
For each word+sense in meanings.json, find the best-matching Wikipedia article
and fetch its 2024 page view count. Normalize across senses per word to get a
proxy text-frequency distribution. Saves to text_frequencies.json.

No API key needed — uses the free Wikipedia REST API.
"""
import json
import time
import requests

meanings = json.load(open("meanings.json"))

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "llm-polysemy-research/1.0 (academic)"

PAGEVIEW_BASE = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
    "/en.wikipedia/all-access/all-agents/{title}/monthly/20240101/20241201"
)


def wikipedia_search(query, limit=1, retries=4):
    for attempt in range(retries):
        try:
            r = SESSION.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "list": "search",
                "srsearch": query, "format": "json", "srlimit": limit,
            }, timeout=10)
            if r.status_code == 429 or not r.text.strip():
                wait = 2 ** attempt
                print(f"    rate limited, waiting {wait}s…")
                time.sleep(wait)
                continue
            results = r.json().get("query", {}).get("search", [])
            return results[0]["title"] if results else None
        except Exception as e:
            time.sleep(2 ** attempt)
    return None


def pageviews(title, retries=3):
    url = PAGEVIEW_BASE.format(title=title.replace(" ", "_"))
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=10)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if r.status_code != 200:
                return 0
            return sum(item["views"] for item in r.json().get("items", []))
        except Exception:
            time.sleep(2 ** attempt)
    return 0


# Resume from existing file if present
try:
    results = json.load(open("text_frequencies.json"))
    print(f"Resuming — {len(results)} words already done.")
except FileNotFoundError:
    results = {}

for word, senses in meanings.items():
    if word in results:
        continue
    sense_list = senses if isinstance(senses, list) else [
        {"sense": k, "meaning": v} for k, v in senses.items()
    ]
    print(f"\n{word}")
    sense_views = {}

    for entry in sense_list:
        sense = entry["sense"]
        meaning = entry["meaning"]
        query = f"{word} {meaning}"
        title = wikipedia_search(query)
        if title is None:
            print(f"  [{sense}] no article found")
            sense_views[sense] = 0
            continue
        views = pageviews(title)
        print(f"  [{sense}] '{title}' → {views:,} views")
        sense_views[sense] = views
        time.sleep(0.3)   # be polite

    total = sum(sense_views.values())
    if total > 0:
        results[word] = {s: v / total for s, v in sense_views.items()}
    else:
        n = len(sense_views)
        results[word] = {s: 1 / n for s in sense_views}

    with open("text_frequencies.json", "w") as f:
        json.dump(results, f, indent=2)

print("\nSaved text_frequencies.json")
