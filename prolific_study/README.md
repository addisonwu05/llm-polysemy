# Prolific human-baseline study

Shows a participant **one bare word** and collects a free-form description of
what it means to them — the human counterpart to the model `text` modality.

## Run locally

```bash
pip install -r prolific_study/requirements.txt
python prolific_study/app.py            # serves on http://localhost:5050
```

Open `http://localhost:5050/` to preview. Live status: `http://localhost:5050/status`.

## Edit the wording

- **`prompt.txt`** — the task instruction shown above the word. A literal
  `{word}` is replaced with the assigned word (optional; the word is also shown
  big on its own).
- **`consent.txt`** — the first screen. Drop in your IRB-approved language.

Both files are re-read on every page load — edit and refresh, no restart needed.

## Other knobs — `study_config.py`

- `WORDS_PER_PARTICIPANT` (default 1), `MIN_RESPONSE_CHARS`, `N_WORDS` (100),
  `TARGET_RESPONSES_PER_WORD`, `PORT`.
- `PROLIFIC_COMPLETION_CODE` — **set this** to the code from your Prolific study
  page. Until set, the final page warns instead of redirecting.

## Prolific setup

1. Deploy somewhere public (see below) over **HTTPS**.
2. In Prolific, set the study URL to:
   `https://YOUR_HOST/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}`
   Prolific fills those placeholders in; the app records all three.
3. Choose **"I'll redirect them using a URL"** for completion and copy the
   completion code into `study_config.py`.

Word assignment is balanced automatically: each participant is handed the
least-covered word(s), counting both saved and in-flight responses, so the
100-word panel fills evenly. Recruit ~`100 × TARGET_RESPONSES_PER_WORD`
submissions for full coverage.

## Deploy

Any host that runs a Python web process works (Render / Railway / Fly / a VPS):

```bash
gunicorn -w 2 -b 0.0.0.0:$PORT --chdir prolific_study app:app
```

For a quick public tunnel during testing: `ngrok http 5050` (or `cloudflared`).

## Get the data out

Responses append to `prolific_study/responses/responses.jsonl` (one JSON line
per word; gitignored). To flatten to CSV:

```bash
python prolific_study/export_responses.py     # -> responses/responses.csv
```

Each row: `ts, prolific_pid, study_id, session_id, assignment_id, word,
response, seconds_on_word, word_index, words_total`.
