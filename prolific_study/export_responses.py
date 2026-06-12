"""Export responses/responses.jsonl -> responses/responses.csv."""

import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "responses" / "responses.jsonl"
DST = BASE / "responses" / "responses.csv"

FIELDS = ["ts", "prolific_pid", "study_id", "session_id", "assignment_id",
          "word", "response", "seconds_on_word", "word_index", "words_total"]


def main():
    if not SRC.exists():
        raise SystemExit(f"no responses yet ({SRC} missing)")
    rows = [json.loads(line) for line in SRC.read_text().splitlines() if line.strip()]
    with DST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {DST}")


if __name__ == "__main__":
    main()
