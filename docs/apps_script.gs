/**
 * Google Apps Script backend for the Prolific word-description study.
 *
 * Two jobs:
 *   - doPost: append one row per response to the data sheet (Sheet1), and clear
 *     the participant's reservation.
 *   - doGet?action=assign: COORDINATED ASSIGNMENT. Reads current coverage from
 *     the sheet + active reservations, hands the participant the N least-covered
 *     words under the least-covered prompt type, and reserves them (TTL). This is
 *     what makes coverage converge to TARGET per (word x prompt) cell instead of
 *     being random. Returns JSONP so the GitHub Pages frontend can read it.
 *
 * SETUP (one time):
 *   1. Create a Google Sheet. Extensions > Apps Script. Paste this whole file.
 *   2. Deploy > New deployment > Web app. Execute as: Me. Who has access: Anyone.
 *      Deploy, authorize, COPY the /exec URL into docs/config.js as SCRIPT_URL.
 *   3. AFTER ANY EDIT to this file: Deploy > Manage deployments > edit (pencil)
 *      > Version: New version > Deploy (keeps the same URL).
 *
 * IMPORTANT: WORDS, PROMPTS and TARGET below must match the study. If you change
 * docs/words.json or the PROMPTS list in docs/config.js, update them here too.
 */

var SHEET_NAME = "Sheet1";       // data sheet (one row per response)
var RES_SHEET  = "Reservations"; // assignment_id | prompt_id | words(csv) | ts
var TARGET     = 30;             // responses wanted per (word x prompt) cell
var TTL_MS     = 20 * 60 * 1000; // a reservation frees after this if not submitted
var PROMPTS    = ["meaning", "image"];

var WORDS = [
  "trunk", "jordan", "bolt", "bank", "pitch", "bow", "spring", "crane",
  "club", "turkey", "shot", "bar", "mercury", "jack", "mole", "charge",
  "key", "watch", "ring", "nail", "chip", "cell", "toast", "track",
  "wave", "train", "anchor", "ash", "ball", "band", "bark", "barrel",
  "basket", "bass", "batter", "beam", "bench", "block", "bowl", "box",
  "brush", "buck", "button", "cane", "cape", "capital", "case", "chest",
  "coach", "cobbler", "apple", "amazon", "mars", "jaguar", "mustang", "cobra",
  "python", "fox", "cardinal", "ram", "paddle", "staple", "pitcher", "shuttle",
  "tap", "drill", "pump", "socket", "switch", "iron", "crown", "scale",
  "plane", "fan", "bridge", "port", "lodge", "mine", "grave", "cast",
  "press", "post", "tip", "will", "file", "mortar", "hide", "tank",
  "seal", "strike", "stock", "bug", "fly", "bat", "pen", "pool",
  "rock", "fire", "match", "tie",
];

var HEADERS = ["ts", "prolific_pid", "study_id", "session_id", "assignment_id",
               "prompt_id", "prompt_text", "word", "response", "seconds_on_word",
               "word_index", "words_total", "user_agent"];

// ── helpers ────────────────────────────────────────────────────────────────

function getSheet(name) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  return ss.getSheetByName(name) || ss.insertSheet(name);
}

function dataSheet() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  return ss.getSheetByName(SHEET_NAME) || ss.getActiveSheet();
}

function ensureResHeader(sh) {
  if (sh.getLastRow() === 0) sh.appendRow(["assignment_id", "prompt_id", "words", "ts"]);
}

// counts[prompt][word] from submitted responses only
function submittedCounts() {
  var counts = {};
  PROMPTS.forEach(function (p) { counts[p] = {}; WORDS.forEach(function (w) { counts[p][w] = 0; }); });
  var data = dataSheet().getDataRange().getValues();
  if (data.length > 1) {
    var head = data[0];
    var pi = head.indexOf("prompt_id"), wi = head.indexOf("word");
    if (pi >= 0 && wi >= 0) {
      for (var r = 1; r < data.length; r++) {
        var p = data[r][pi], w = data[r][wi];
        if (counts[p] && counts[p][w] !== undefined) counts[p][w]++;
      }
    }
  }
  return counts;
}

// returns active reservation rows, purging expired ones from the sheet
function activeReservations(now) {
  var sh = getSheet(RES_SHEET);
  ensureResHeader(sh);
  var data = sh.getDataRange().getValues();
  if (data.length <= 1) return [];
  var keep = [data[0]], active = [];
  for (var r = 1; r < data.length; r++) {
    if (now - Number(data[r][3]) <= TTL_MS) { keep.push(data[r]); active.push(data[r]); }
  }
  if (keep.length !== data.length) {
    sh.clearContents();
    sh.getRange(1, 1, keep.length, keep[0].length).setValues(keep);
  }
  return active;
}

function clearReservation(assignmentId) {
  if (!assignmentId) return;
  var sh = getSheet(RES_SHEET);
  var data = sh.getDataRange().getValues();
  if (data.length <= 1) return;
  var keep = [data[0]];
  for (var r = 1; r < data.length; r++) {
    if (String(data[r][0]) !== String(assignmentId)) keep.push(data[r]);
  }
  if (keep.length !== data.length) {
    sh.clearContents();
    sh.getRange(1, 1, keep.length, keep[0].length).setValues(keep);
  }
}

// the core: pick least-covered prompt, then its N least-covered words; reserve
function assign(assignmentId, n) {
  var now = Date.now();
  var counts = submittedCounts();
  activeReservations(now).forEach(function (row) {
    var p = row[1], ws = String(row[2]).split(",");
    ws.forEach(function (w) { if (counts[p] && counts[p][w] !== undefined) counts[p][w]++; });
  });

  // prompt with the most remaining work (sum of unmet TARGET over words)
  var best = PROMPTS[0], bestRem = -1;
  PROMPTS.forEach(function (p) {
    var rem = 0;
    WORDS.forEach(function (w) { rem += Math.max(0, TARGET - counts[p][w]); });
    if (rem > bestRem) { bestRem = rem; best = p; }
  });

  // least-covered words under that prompt (random tie-break)
  var sorted = WORDS.slice().sort(function (a, b) {
    var d = counts[best][a] - counts[best][b];
    return d !== 0 ? d : (Math.random() - 0.5);
  });
  var words = sorted.slice(0, n);

  var sh = getSheet(RES_SHEET);
  ensureResHeader(sh);
  sh.appendRow([assignmentId, best, words.join(","), now]);
  return { prompt_id: best, words: words, remaining: bestRem };
}

// ── endpoints ──────────────────────────────────────────────────────────────

function doGet(e) {
  var p = (e && e.parameter) ? e.parameter : {};
  if (p.action === "assign") {
    var lock = LockService.getScriptLock();
    lock.waitLock(30000);
    var out;
    try { out = assign(p.assignment_id || "", Number(p.n) || 5); }
    catch (err) { out = { error: String(err) }; }
    finally { lock.releaseLock(); }
    var body = JSON.stringify(out);
    if (p.callback) {
      return ContentService.createTextOutput(p.callback + "(" + body + ")")
        .setMimeType(ContentService.MimeType.JAVASCRIPT);
    }
    return ContentService.createTextOutput(body).setMimeType(ContentService.MimeType.JSON);
  }
  return ContentService.createTextOutput("Study backend is live.");
}

function doPost(e) {
  var lock = LockService.getScriptLock();
  lock.waitLock(30000);
  try {
    var payload = JSON.parse(e.postData.contents);
    var sheet = dataSheet();
    if (sheet.getLastRow() === 0) sheet.appendRow(HEADERS);

    var responses = payload.responses || [];
    for (var i = 0; i < responses.length; i++) {
      var r = responses[i];
      sheet.appendRow([
        new Date().toISOString(),
        payload.prolific_pid || "",
        payload.study_id || "",
        payload.session_id || "",
        payload.assignment_id || "",
        payload.prompt_id || "",
        payload.prompt_text || "",
        r.word || "",
        r.response || "",
        r.seconds == null ? "" : r.seconds,
        i,
        responses.length,
        payload.user_agent || "",
      ]);
    }
    clearReservation(payload.assignment_id || "");
    return ContentService
      .createTextOutput(JSON.stringify({ ok: true, saved: responses.length }))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService
      .createTextOutput(JSON.stringify({ ok: false, error: String(err) }))
      .setMimeType(ContentService.MimeType.JSON);
  } finally {
    lock.releaseLock();
  }
}
