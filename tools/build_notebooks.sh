#!/usr/bin/env bash
# Convert the jupytext "percent" sources in book/_src/ into the .ipynb files the
# book actually ships. Authoring in .py keeps diffs readable; students get .ipynb.
#
#   ./tools/build_notebooks.sh            # convert all
#   ./tools/build_notebooks.sh part3_analysis/01_decode_and_slim
#
# Notebooks are NOT executed here -- outputs are added by running them on Euler
# against the real data, then saving.
set -euo pipefail
cd "$(dirname "$0")/.."
JUPYTEXT="${JUPYTEXT:-jupytext}"

targets=()
if [ $# -gt 0 ]; then
  for name in "$@"; do targets+=("book/_src/${name}.py"); done
else
  while IFS= read -r f; do targets+=("$f"); done < <(find book/_src -name '*.py' | sort)
fi

for src in "${targets[@]}"; do
  out="${src/book\/_src/book}"; out="${out%.py}.ipynb"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ]; then
    # --update rewrites the code and markdown but KEEPS the stored outputs of
    # cells that did not change. Without it, every rebuild throws away results
    # that took a cluster session to produce.
    "$JUPYTEXT" --to ipynb --update --output "$out" "$src" >/dev/null
    echo "  updated $out"
  else
    "$JUPYTEXT" --to ipynb --output "$out" "$src" >/dev/null
    echo "  created $out"
  fi
done
