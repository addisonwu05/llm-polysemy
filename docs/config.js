// ── Study configuration - edit these, commit, push. ────────────────────────
window.STUDY_CONFIG = {

  // PASTE your Google Apps Script Web App URL here (see apps_script.gs / README).
  // Until you do, submissions are logged to the browser console instead of saved.
  SCRIPT_URL: "https://script.google.com/macros/s/AKfycbx-BTTBlidNkZzWM36WnCbxsneypJZbDi30Afs9QnwFxLW9jCSDL7D4GHa-j734q1JO/exec",

  // PASTE the completion code from your Prolific study page.
  // Until set, the final screen warns instead of redirecting to Prolific.
  PROLIFIC_COMPLETION_CODE: "CUNIZJ2O",

  WORDS_PER_PARTICIPANT: 5,   // words each participant describes, one at a time
  MIN_RESPONSE_WORDS: 10,     // submit unlocks once the response has this many words

  // Each participant is shown exactly ONE of these prompts, picked at random.
  // The chosen `id` is recorded in the Sheet so you can tell them apart.
  // Edit the .txt files freely; {word} is replaced with the assigned word (bold).
  PROMPTS: [
    { id: "meaning", file: "prompt_meaning.txt" },
    { id: "image",   file: "prompt_image.txt" },
  ],
};
