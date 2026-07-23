#!/usr/bin/env bash
#
# download_data.sh — fetch CAMELYON16 whole-slide images (+ annotations, if any)
# from the public AWS Open Data mirror.
#
# Destination: set DEST (default "slides"). On Colab, point it at mounted Drive:
#   DEST=/content/drive/MyDrive/camelyon16 bash scripts/download_data.sh tumor_001..003
#
# Slide names: pass individual names and/or ranges "prefix_LO..HI":
#   bash scripts/download_data.sh tumor_084
#   bash scripts/download_data.sh tumor_001..003 test_001..003 normal_001..003
#
# Notes:
#   - Slides are large (normal ~1.2 GB, tumor ~3 GB, test ~1 GB each).
#   - Only tumor and test slides have annotations; normal slides don't (the .xml
#     download is skipped and validation simply won't run for those).
#   - Already-downloaded slides are skipped (safe to re-run / resume).
set -euo pipefail

BASE="https://camelyon-dataset.s3.amazonaws.com/CAMELYON16"
DEST="${DEST:-slides}"
mkdir -p "$DEST"

# --- expand any "prefix_LO..HI" ranges into individual names ---
expand() {
  local arg="$1"
  if [[ "$arg" == *".."* ]]; then
    local lo="${arg%..*}" hi="${arg#*..}"           # tumor_001 .. 015
    local prefix="${lo%_*}" start="${lo##*_}" end="$hi"
    local width=${#start}
    for ((i=10#$start; i<=10#$end; i++)); do
      printf "%s_%0*d\n" "$prefix" "$width" "$i"
    done
  else
    echo "$arg"
  fi
}

names=()
for a in "$@"; do
  while IFS= read -r n; do names+=("$n"); done < <(expand "$a")
done
[ ${#names[@]} -eq 0 ] && names=("tumor_084")

echo "Destination: $DEST"
echo "Slides (${#names[@]}): ${names[*]}"

for NAME in "${names[@]}"; do
  echo "==> $NAME"
  if [ -f "$DEST/$NAME.tif" ]; then
    echo "    slide already present, skipping"
  else
    curl -fL --progress-bar -o "$DEST/$NAME.tif" "$BASE/images/$NAME.tif"
  fi

  if [ -f "$DEST/$NAME.xml" ]; then
    echo "    annotation already present"
  elif curl -fL --progress-bar -o "$DEST/$NAME.xml" "$BASE/annotations/$NAME.xml"; then
    echo "    + annotation"
  else
    rm -f "$DEST/$NAME.xml"
    echo "    (no annotation for $NAME — validation will be skipped for it)"
  fi
done

echo "Done. Files are in $DEST/"
