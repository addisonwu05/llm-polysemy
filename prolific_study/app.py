"""Prolific study server — show one bare word, collect a free-form description.

Run:             python prolific_study/app.py
Participant URL: http://<host>:<port>/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}

Responses are appended one JSON line per (participant, word) to
responses/responses.jsonl. Word assignment is balanced: each participant gets
the word(s) with the fewest collected + in-flight responses, so coverage
fills evenly across the 100-word panel.
"""

import json
import random
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import study_config as cfg

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
WORDS_FILE = ROOT / "data" / "synonyms.txt"
RESPONSES_DIR = BASE / "responses"
RESPONSES_FILE = RESPONSES_DIR / "responses.jsonl"
PROMPT_FILE = BASE / "prompt.txt"
CONSENT_FILE = BASE / "consent.txt"

# How long an unfinished assignment keeps its words reserved before they are
# handed out again (covers participants who abandon mid-task).
ASSIGNMENT_TTL = 30 * 60

app = Flask(__name__)

WORDS = [w.strip() for w in WORDS_FILE.read_text().splitlines() if w.strip()][: cfg.N_WORDS]

_lock = threading.Lock()
_in_flight: dict[str, tuple[list[str], float]] = {}  # assignment_id -> (words, started_at)


def _response_counts() -> dict[str, int]:
    counts = {w: 0 for w in WORDS}
    if RESPONSES_FILE.exists():
        for line in RESPONSES_FILE.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("word") in counts:
                counts[row["word"]] += 1
    return counts


def _assign_words(assignment_id: str) -> list[str]:
    """Pick the least-covered words, counting saved and in-flight responses."""
    with _lock:
        now = time.time()
        for aid in [a for a, (_, t) in _in_flight.items() if now - t > ASSIGNMENT_TTL]:
            del _in_flight[aid]
        counts = _response_counts()
        for words, _ in _in_flight.values():
            for w in words:
                if w in counts:
                    counts[w] += 1
        pool = sorted(WORDS, key=lambda w: (counts[w], random.random()))
        chosen = pool[: cfg.WORDS_PER_PARTICIPANT]
        _in_flight[assignment_id] = (chosen, now)
    return chosen


@app.get("/")
def index():
    assignment_id = uuid.uuid4().hex
    return render_template(
        "study.html",
        consent_text=CONSENT_FILE.read_text().strip(),
        prompt_template=PROMPT_FILE.read_text().strip(),
        words=_assign_words(assignment_id),
        assignment_id=assignment_id,
        min_chars=cfg.MIN_RESPONSE_CHARS,
        prolific_pid=request.args.get("PROLIFIC_PID", ""),
        study_id=request.args.get("STUDY_ID", ""),
        session_id=request.args.get("SESSION_ID", ""),
    )


@app.post("/submit")
def submit():
    data = request.get_json(silent=True) or {}
    responses = data.get("responses", [])
    if not responses:
        return jsonify(error="no responses"), 400
    rows = []
    for i, r in enumerate(responses):
        word = r.get("word", "")
        text = (r.get("response") or "").strip()
        if word not in WORDS:
            return jsonify(error=f"unknown word {word!r}"), 400
        if len(text) < cfg.MIN_RESPONSE_CHARS:
            return jsonify(error=f"response for {word!r} is too short"), 400
        rows.append({
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "assignment_id": data.get("assignment_id", ""),
            "prolific_pid": data.get("prolific_pid", ""),
            "study_id": data.get("study_id", ""),
            "session_id": data.get("session_id", ""),
            "word": word,
            "response": text,
            "seconds_on_word": r.get("seconds"),
            "word_index": i,
            "words_total": len(responses),
        })
    with _lock:
        RESPONSES_DIR.mkdir(exist_ok=True)
        with RESPONSES_FILE.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        _in_flight.pop(data.get("assignment_id", ""), None)
    return jsonify(redirect="/done")


@app.get("/done")
def done():
    configured = cfg.PROLIFIC_COMPLETION_CODE != "REPLACE_ME"
    return render_template(
        "done.html",
        configured=configured,
        completion_code=cfg.PROLIFIC_COMPLETION_CODE,
        completion_url=cfg.COMPLETION_URL,
    )


@app.get("/status")
def status():
    counts = _response_counts()
    at_target = sum(1 for c in counts.values() if c >= cfg.TARGET_RESPONSES_PER_WORD)
    return jsonify(
        total_responses=sum(counts.values()),
        words_at_target=f"{at_target}/{len(WORDS)}",
        target_per_word=cfg.TARGET_RESPONSES_PER_WORD,
        counts=counts,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=cfg.PORT)
