#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFERENCE_DOC="${SCRIPT_DIR}/../000 templates/Mal prosjekt LOG650 v2.docx"

cd "${SCRIPT_DIR}"

pandoc "Rapport.md" \
  --standalone \
  --from=markdown+raw_attribute+raw_html+tex_math_dollars \
  --lua-filter="${SCRIPT_DIR}/report-format.lua" \
  --reference-doc="${REFERENCE_DOC}" \
  --output="Rapport.docx"
