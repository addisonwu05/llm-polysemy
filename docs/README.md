# Prolific word-description study (static / GitHub Pages)

A pure HTML/JS page - no server. GitHub Pages serves it; responses are saved to
a **Google Sheet** via a Google Apps Script web app. This is the human baseline
for the `text` modality: show one bare word, collect a free-form description.

```
docs/
  index.html      ← the whole study (consent → word → thank-you)
  config.js       ← YOUR settings: Apps Script URL, completion code, etc.
  words.json          ← first 100 words of data/synonyms.txt (regenerate if it changes)
  prompt_meaning.txt  ← prompt variant A: "use it in a sentence"  (edit freely)
  prompt_image.txt    ← prompt variant B: "what image comes to mind" (edit freely)
  consent.txt         ← the consent screen   (edit freely)
  apps_script.gs      ← paste this into Google Apps Script (the backend)
```

## Setup (one time, ~10 min)

### 1. Backend - Google Sheet + Apps Script
1. Create a Google Sheet.
2. **Extensions ▸ Apps Script**, delete the stub, paste all of `apps_script.gs`.
3. **Deploy ▸ New deployment ▸ Web app**: *Execute as* **Me**, *Who has access*
   **Anyone**. Deploy, authorize, and copy the **Web app URL** (ends in `/exec`).

### 2. Configure the page - edit `config.js`
- `SCRIPT_URL` → the `/exec` URL from step 1.
- `PROLIFIC_COMPLETION_CODE` → the code from your Prolific study page.
- `WORDS_PER_PARTICIPANT` (default 1), `MIN_RESPONSE_WORDS` (default 10).

### 3. Publish on GitHub Pages
Commit `docs/` and push, then in the repo: **Settings ▸ Pages ▸ Source = Deploy
from a branch**, branch `main`, folder **`/docs`**. Your page goes live at
`https://<user>.github.io/<repo>/`.

### 4. Point Prolific at it
In Prolific set the study URL to (placeholders are filled in by Prolific):
```
https://<user>.github.io/<repo>/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}
```
Choose **"I'll redirect them using a URL"** for completion (the page redirects to
Prolific with your completion code at the end).

## Prompt variants
Each participant is shown **exactly one** prompt, chosen at random from the
`PROMPTS` list in `config.js` (currently `meaning` and `image`). The chosen
variant's `id` is recorded in the Sheet (`prompt_id` column) so you can split
results by condition. `{word}` in a prompt is replaced with the assigned word,
rendered **bold**. Add/remove variants by editing the `.txt` files and the
`PROMPTS` list.

## Editing the wording
The prompt `.txt` files and `consent.txt` are plain text fetched at page load -
edit, commit, push, and the live page updates.

## Preview locally
`fetch()` is blocked on `file://`, so serve over HTTP:
```bash
cd docs && python3 -m http.server 8077   # open http://localhost:8077/
```
With `SCRIPT_URL` still unset, submissions are logged to the browser console
(DevTools) instead of saved - so you can walk the whole flow before wiring the Sheet.

## Getting the data
Responses land in your Google Sheet (`Sheet1`), one row per word:
`ts, prolific_pid, study_id, session_id, assignment_id, prompt_id, prompt_text,
word, response, seconds_on_word, word_index, words_total, user_agent`.
**File ▸ Download ▸ CSV** when you're done.

## Coordinated assignment (balanced coverage)
On page load the frontend calls the Apps Script (`?action=assign`, via JSONP),
which reads current coverage from `Sheet1` + active reservations and returns the
N least-covered words under the least-covered prompt type, then reserves them in
a `Reservations` tab (20-min TTL; cleared on submit). This drives coverage to
**`TARGET` (=30) responses per (word × prompt) cell** instead of random scatter.
A simulated full run converges to exactly 30/cell in ~1,350 participants.

If the coordinator is unreachable, the page falls back to random assignment so a
participant is never blocked (logged to console).

**Keep in sync:** `WORDS`, `PROMPTS`, and `TARGET` are duplicated in
`apps_script.gs` — if you change `words.json` or `config.js` `PROMPTS`, update the
script too, then redeploy a **New version** of the web app.
