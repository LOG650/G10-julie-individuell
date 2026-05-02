#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTML_TMP="/private/tmp/rapport-review.html"
CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CHROME_PROFILE="$(mktemp -d /private/tmp/chrome-headless-report.XXXXXX)"

cleanup() {
  rm -rf "${CHROME_PROFILE}"
}

trap cleanup EXIT

cd "${SCRIPT_DIR}"

pandoc "Rapport.md" \
  --standalone \
  --embed-resources \
  --mathml \
  --from=markdown+raw_attribute+raw_html+tex_math_dollars \
  --css="${SCRIPT_DIR}/report-print.css" \
  --lua-filter="${SCRIPT_DIR}/report-format.lua" \
  --output="${HTML_TMP}"

"${CHROME_BIN}" \
  --headless \
  --disable-gpu \
  --no-first-run \
  --allow-file-access-from-files \
  --user-data-dir="${CHROME_PROFILE}" \
  --virtual-time-budget=5000 \
  --print-to-pdf="${SCRIPT_DIR}/Rapport G10.pdf" \
  "file://${HTML_TMP}"
