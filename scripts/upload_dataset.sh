#!/bin/bash
# Upload a generated corpus to the Hugging Face dataset repo.
#
#   ./upload_dataset.sh                    # image outputs (default)
#   ./upload_dataset.sh text
#   ./upload_dataset.sh image --dry-run    # print the plan, upload nothing
#   ./upload_dataset.sh --all              # every sector that exists locally
#
# WHY THIS IS NOT A ONE-LINER
# ---------------------------
# `hf upload-large-folder` has NO `--path-in-repo` flag (check: `hf upload-large-folder
# --help`). It uploads a folder to the REPO ROOT, mirroring relative paths. The dataset,
# however, is organized into numbered sectors (01_main_english/, 05_ablations/, ...).
# So we stage a tree whose relative paths already equal the target repo paths, then
# upload the staging root. Staging uses HARD LINKS -- no data is duplicated (the PNG
# corpus is ~20 GB), so staging costs ~0 bytes as long as $STAGE is on the same
# filesystem as data/ (it is by default).
#
# Sector layout is defined by the dataset card:
#   https://huggingface.co/datasets/addisonwu05/llm-polysemy-outputs
set -euo pipefail

HF_REPO="${HF_REPO:-addisonwu05/llm-polysemy-outputs}"
STAGE="${STAGE:-$(dirname "$(readlink -f "$0")")/.hf_staging}"
DRY_RUN=0
MODALITIES=()

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --all)     MODALITIES=(image text human_image human_meaning image_tr text_tr image_fr text_fr text_imagine) ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    -*)        echo "unknown flag: $arg" >&2; exit 2 ;;
    *)         MODALITIES+=("$arg") ;;
  esac
done
[ ${#MODALITIES[@]} -eq 0 ] && MODALITIES=(image)

# ── local dir -> repo sector path ─────────────────────────────────────────────
repo_path_for() {
  case "$1" in
    image)         echo "01_main_english/image" ;;
    text)          echo "01_main_english/text" ;;
    human_image)   echo "02_human_baseline/image" ;;
    human_meaning) echo "02_human_baseline/meaning" ;;
    image_tr)      echo "03_cross_lingual/turkish/image" ;;
    text_tr)       echo "03_cross_lingual/turkish/text" ;;
    image_fr)      echo "03_cross_lingual/french/image" ;;
    text_fr)       echo "03_cross_lingual/french/text" ;;
    text_imagine)  echo "04_stated_vs_revealed/text_imagine" ;;
    *)             echo "" ;;
  esac
}

# Image-modality model keys that do NOT belong in the main panel.
# Routed to their own sector; the card is the source of truth for these lists.
ABLATION_KEYS="sdxl_base sdxl_dpo simplear_sft simplear_rl"
ABLATION_PATH="05_ablations/preference_tuning"

in_list() { case " $2 " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# ── stage ─────────────────────────────────────────────────────────────────────
rm -rf "$STAGE"; mkdir -p "$STAGE"
link_or_copy() { ln "$1" "$2" 2>/dev/null || cp "$1" "$2"; }

total=0
for MOD in "${MODALITIES[@]}"; do
  SRC="data/outputs_${MOD}"
  DEST_REL="$(repo_path_for "$MOD")"
  if [ -z "$DEST_REL" ]; then echo "!! unknown modality '$MOD' -- skipping" >&2; continue; fi
  if [ ! -d "$SRC" ]; then echo "!! $SRC missing -- skipping" >&2; continue; fi

  n_main=0; n_abl=0
  for worddir in "$SRC"/*/; do
    word="$(basename "$worddir")"
    for keydir in "$worddir"*/; do
      [ -d "$keydir" ] || continue
      key="$(basename "$keydir")"

      if [ "$MOD" = "image" ] && in_list "$key" "$ABLATION_KEYS"; then
        target="$STAGE/$ABLATION_PATH/$word/$key"; counter=abl
      else
        target="$STAGE/$DEST_REL/$word/$key"; counter=main
      fi

      mkdir -p "$target"
      for f in "$keydir"*; do
        [ -f "$f" ] || continue
        link_or_copy "$f" "$target/$(basename "$f")"
        [ "$counter" = abl ] && n_abl=$((n_abl+1)) || n_main=$((n_main+1))
      done
    done
  done
  total=$((total + n_main + n_abl))
  echo "[$MOD] -> $DEST_REL : $n_main files"
  [ "$n_abl" -gt 0 ] && echo "[$MOD] -> $ABLATION_PATH : $n_abl files (ablation keys)"
done

if [ "$total" -eq 0 ]; then echo "nothing staged; aborting." >&2; rm -rf "$STAGE"; exit 1; fi

echo
echo "Staged $total files under $STAGE"
echo "Repo paths to be written:"
(cd "$STAGE" && find . -mindepth 3 -maxdepth 3 -type d | sed 's|^\./||' | sed 's|/[^/]*$||' | sort -u | head -20)

if [ "$DRY_RUN" -eq 1 ]; then
  echo; echo "--dry-run: nothing uploaded. Staging left at $STAGE for inspection."
  exit 0
fi

# ── upload ────────────────────────────────────────────────────────────────────
# Resumable. The Hub rate-limits at 1000 API calls / 5 min; upload-large-folder backs
# off and shrinks its commit batch on 429, so those log lines are normal, not failures.
# Re-running skips already-committed files, hence the retry loop.
echo; echo "Uploading to $HF_REPO ..."
for attempt in 1 2 3 4 5; do
  if hf upload-large-folder "$HF_REPO" "$STAGE" --repo-type=dataset --num-workers=8; then
    echo "Done. View at: https://huggingface.co/datasets/$HF_REPO"
    exit 0
  fi
  echo "-- pass $attempt did not complete; resuming in 60s --" >&2
  sleep 60
done

echo "!! upload did not finish after 5 passes. Staging kept at $STAGE; re-run to resume." >&2
exit 1
