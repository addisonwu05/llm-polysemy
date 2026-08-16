"""Aggregate per-sample classifications into per-(word, model, judge) histograms.

Logic extracted verbatim from the original classify.py / classify_text.py so the
output is byte-identical given the same classifications file.
"""

import json
from pathlib import Path


def label_of(verdict) -> str:
    """A stored verdict is either a label string or {'label':'multiple','senses':[...]}"""
    return verdict["label"] if isinstance(verdict, dict) else verdict


def build_histograms(class_file: Path, hist_file: Path) -> None:
    results = json.loads(Path(class_file).read_text())
    hist: dict = {}
    for word, models in results.items():
        for mk, idxs in models.items():
            agree = same = 0
            for idx, judges in idxs.items():
                for judge, verdict in judges.items():
                    label = label_of(verdict)
                    d = (hist.setdefault(word, {}).setdefault(mk, {})
                             .setdefault(judge, {"_counts": {}, "_multiple": {}}))
                    d["_counts"][label] = d["_counts"].get(label, 0) + 1
                    if isinstance(verdict, dict) and verdict.get("senses"):
                        combo = "+".join(sorted(verdict["senses"]))
                        d["_multiple"][combo] = d["_multiple"].get(combo, 0) + 1
                if "gpt" in judges and "gemini" in judges:
                    agree += 1
                    same += (label_of(judges["gpt"]) == label_of(judges["gemini"]))
            if agree:
                hist[word][mk]["_agreement"] = {
                    "same": same, "total": agree, "rate": round(same / agree, 3)}
    Path(hist_file).write_text(json.dumps(hist, indent=2))
    print(f"Histograms -> {hist_file}  ({len(hist)} words)")
