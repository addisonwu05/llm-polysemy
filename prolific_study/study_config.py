"""All study knobs in one place.

Participant-facing TEXT is not here — edit `prompt.txt` (the task instruction;
a literal {word} placeholder is substituted) and `consent.txt` (the first
screen). Both files are re-read on every page load, so edits take effect
without restarting the server.
"""

# ── Task shape ─────────────────────────────────────────────────────────────
WORDS_PER_PARTICIPANT = 1      # words each participant describes, one at a time
MIN_RESPONSE_CHARS = 20        # submit unlocks at this length (also enforced server-side)
TARGET_RESPONSES_PER_WORD = 30 # progress yardstick for /status; assignment always
                               # gives out the least-covered words regardless

# ── Word list ──────────────────────────────────────────────────────────────
N_WORDS = 100                  # first N words of data/synonyms.txt — the same
                               # slice the image/text models were evaluated on

# ── Prolific ───────────────────────────────────────────────────────────────
# Paste the completion code shown on your Prolific study page. Until this is
# set, the done-page shows a warning instead of redirecting participants.
PROLIFIC_COMPLETION_CODE = "REPLACE_ME"
COMPLETION_URL = f"https://app.prolific.com/submissions/complete?cc={PROLIFIC_COMPLETION_CODE}"

# ── Server ─────────────────────────────────────────────────────────────────
PORT = 5050
