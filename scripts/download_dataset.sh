#!/bin/bash
# Download the published outputs from the Hugging Face dataset into the local
# layout the pipeline expects (data/outputs_<mod>/<word>/<model>/NN.{png,txt}),
# so you can reproduce classification / histograms / figures WITHOUT a GPU --
# no need to regenerate anything.
#
#   ./download_dataset.sh                    # image outputs (default)
#   ./download_dataset.sh text
#   ./download_dataset.sh image --dry-run    # print the plan, download nothing
#   ./download_dataset.sh --all              # every sector
#
# This is the inverse of upload_dataset.sh. The dataset is organized into numbered
# sectors on the Hub (01_main_english/, 05_ablations/, ...); this pulls the sector(s)
# for the requested modality and remaps them back to data/outputs_<mod>/. Files are
# hard-linked from the HF cache into place -- no duplication as long as $CACHE is on
# the same filesystem as data/ (it is by default).
set -euo pipefail

HF_REPO="${HF_REPO:-addisonwu05/llm-polysemy-outputs}"
CACHE="${CACHE:-$(dirname "$(readlink -f "$0")")/.hf_cache}"
DRY_RUN=0
MODALITIES=()

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --all)     MODALITIES=(image text human_image human_meaning image_tr text_tr image_fr text_fr text_imagine) ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    -*)        echo "unknown flag: $arg" >&2; exit 2 ;;
    *)         MODALITIES+=("$arg") ;;
  esac
done
[ ${#MODALITIES[@]} -eq 0 ] && MODALITIES=(image)

# ── local dir <- repo sector path (same table as upload_dataset.sh) ───────────
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
# Preference-tuning keys live in their own sector but belong under outputs_image/.
ABLATION_PATH="05_ablations/preference_tuning"

link_or_copy() { ln "$1" "$2" 2>/dev/null || cp "$1" "$2"; }

# Remap one downloaded sector tree (<cache>/<sector>/<word>/<key>/*) into a local
# outputs dir (data/outputs_<mod>/<word>/<key>/*). Echoes the file count.
remap_sector() {   # $1 = cache sector dir, $2 = local outputs dir
  local src="$1" dst="$2" n=0 word key f
  [ -d "$src" ] || { echo 0; return 0; }
  for worddir in "$src"/*/; do
    [ -d "$worddir" ] || continue
    word="$(basename "$worddir")"
    for keydir in "$worddir"*/; do
      [ -d "$keydir" ] || continue
      key="$(basename "$keydir")"
      mkdir -p "$dst/$word/$key"
      for f in "$keydir"*; do
        [ -f "$f" ] || continue
        link_or_copy "$f" "$dst/$word/$key/$(basename "$f")"
        n=$((n+1))
      done
    done
  done
  echo "$n"
}

for MOD in "${MODALITIES[@]}"; do
  SECTOR="$(repo_path_for "$MOD")"
  if [ -z "$SECTOR" ]; then echo "!! unknown modality '$MOD' -- skipping" >&2; continue; fi
  DEST="data/outputs_${MOD}"

  echo "[$MOD] Hub:$SECTOR -> $DEST"
  [ "$MOD" = image ] && echo "[$MOD] Hub:$ABLATION_PATH -> $DEST (preference-tuning keys)"
  if [ "$DRY_RUN" -eq 1 ]; then continue; fi

  # Download only the sector(s) we need (plus the ablation sector for image).
  INCLUDES=(--include "$SECTOR/**")
  [ "$MOD" = image ] && INCLUDES+=(--include "$ABLATION_PATH/**")
  hf download "$HF_REPO" --repo-type=dataset --local-dir "$CACHE" "${INCLUDES[@]}"

  n=$(remap_sector "$CACHE/$SECTOR" "$DEST")
  echo "[$MOD] remapped $n files -> $DEST"
  if [ "$MOD" = image ]; then
    na=$(remap_sector "$CACHE/$ABLATION_PATH" "$DEST")
    echo "[$MOD] remapped $na preference-tuning files -> $DEST"
  fi
done

echo
echo "Done. Local outputs are under data/outputs_*/ ."
echo "Next: python -m llm_polysemy.classify --modality image   (then --histograms), then the plots."
