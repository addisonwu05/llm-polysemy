/**
 * Google Apps Script backend for the Prolific word-description study.
 *
 * SETUP (one time, ~5 min):
 *   1. Create a Google Sheet. Note its tab name (default "Sheet1").
 *   2. Extensions ▸ Apps Script. Delete the stub, paste this whole file.
 *   3. (Optional) change SHEET_NAME below to your tab name.
 *   4. Deploy ▸ New deployment ▸ type "Web app".
 *        Execute as:  Me
 *        Who has access:  Anyone
 *      Click Deploy, authorize, and COPY the "Web app" URL
 *      (looks like https://script.google.com/macros/s/..../exec).
 *   5. Paste that URL into docs/config.js as SCRIPT_URL.
 *
 * Each participant submission appends one row per word. The first write also
 * writes a header row. Open the Sheet anytime to watch responses arrive.
 */

var SHEET_NAME = "Sheet1";

var HEADERS = ["ts", "prolific_pid", "study_id", "session_id", "assignment_id",
               "word", "response", "seconds_on_word", "word_index", "words_total",
               "user_agent"];

function doPost(e) {
  var lock = LockService.getScriptLock();
  lock.waitLock(30000);
  try {
    var payload = JSON.parse(e.postData.contents);
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(SHEET_NAME)
              || SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
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
        r.word || "",
        r.response || "",
        r.seconds == null ? "" : r.seconds,
        i,
        responses.length,
        payload.user_agent || "",
      ]);
    }
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

function doGet() {
  return ContentService.createTextOutput("Study backend is live.");
}
