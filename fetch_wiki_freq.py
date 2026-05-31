"""
For each word+sense, find the best Wikipedia article and sum 2024 monthly
page views. Normalise across senses → wiki_frequencies.json.
Saves after every word so it can be resumed.
"""
import json, time, requests

meanings = json.load(open("meanings.json"))

S = requests.Session()
S.headers["User-Agent"] = "llm-polysemy-research/1.0 (academic)"

PAGEVIEW = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
    "/en.wikipedia/all-access/all-agents/{title}/monthly/20240101/20241201"
)


def search(query, retries=5):
    for attempt in range(retries):
        try:
            r = S.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "list": "search",
                "srsearch": query, "format": "json", "srlimit": 1,
            }, timeout=12)
            if not r.text.strip() or r.status_code == 429:
                time.sleep(2 ** attempt); continue
            hits = r.json().get("query", {}).get("search", [])
            return hits[0]["title"] if hits else None
        except Exception:
            time.sleep(2 ** attempt)
    return None


def views(title, retries=3):
    url = PAGEVIEW.format(title=title.replace(" ", "_"))
    for attempt in range(retries):
        try:
            r = S.get(url, timeout=12)
            if r.status_code == 429:
                time.sleep(2 ** attempt); continue
            if r.status_code != 200:
                return 0
            return sum(i["views"] for i in r.json().get("items", []))
        except Exception:
            time.sleep(2 ** attempt)
    return 0


try:
    results = json.load(open("wiki_frequencies.json"))
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
    sense_views = {}
    for entry in sense_list:
        sense, meaning = entry["sense"], entry["meaning"]
        title = search(f"{word} {meaning}")
        if title is None:
            print(f"  [{sense}] not found"); sense_views[sense] = 0; continue
        v = views(title)
        print(f"  [{sense}] '{title}' → {v:,}")
        sense_views[sense] = v
        time.sleep(0.4)

    total = sum(sense_views.values())
    results[word] = ({s: v / total for s, v in sense_views.items()}
                     if total > 0 else {s: 1 / len(sense_views) for s in sense_views})
    with open("wiki_frequencies.json", "w") as f:
        json.dump(results, f, indent=2)

print("\nDone — wiki_frequencies.json")
