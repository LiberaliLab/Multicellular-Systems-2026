#!/usr/bin/env bash
# Mirror the shipping notebooks into readable plain-text sources.
#
# The .ipynb files under book/ are the SOURCE OF TRUTH: they are what students
# open, what Jupyter Book builds, and what carries the outputs produced on the
# cluster. book/_src/*.py is a GENERATED mirror of them, kept only so that a
# diff or a review shows prose and code instead of JSON.
#
#   ./tools/build_notebooks.sh            # regenerate every mirror
#   ./tools/build_notebooks.sh --check    # fail if any mirror is out of date
#   ./tools/build_notebooks.sh part3_analysis/1_preparation/03_normalisation
#
# Edit the NOTEBOOK. An edit made in book/_src/ is overwritten the next time
# this runs, and --check in CI will not catch it, because the notebook is what
# it compares against.
#
# This used to run the other way -- .py to .ipynb. It was reversed after three
# chapters were edited directly as notebooks and the old direction would have
# regenerated them from stale sources, discarding the work and its outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
JUPYTEXT="${JUPYTEXT:-jupytext}"

check=0
args=()
for arg in "$@"; do
  case "$arg" in
    --check) check=1 ;;
    *) args+=("$arg") ;;
  esac
done

targets=()
if [ ${#args[@]} -gt 0 ]; then
  for name in "${args[@]}"; do targets+=("book/${name}.ipynb"); done
else
  while IFS= read -r f; do targets+=("$f"); done \
    < <(find book -name '*.ipynb' -not -path 'book/_build/*' | sort)
fi

# A plain string, not an array: bash 3.2 (the macOS default) trips over
# expanding an empty array under `set -u`.
stale=""
# jupytext picks its output format from the file extension, so the scratch copy
# has to keep the .py name -- a bare mktemp file silently fails to convert.
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

for nb in "${targets[@]}"; do
  out="book/_src/${nb#book/}"; out="${out%.ipynb}.py"
  mkdir -p "$(dirname "$out")"
  if [ "$check" -eq 1 ]; then
    tmp="$scratch/$(basename "$out")"
    "$JUPYTEXT" --to py:percent --output "$tmp" "$nb" >/dev/null
    if ! diff -q "$out" "$tmp" >/dev/null 2>&1; then
      stale="$stale  $out"$'\n'
    fi
  else
    "$JUPYTEXT" --to py:percent --output "$out" "$nb" >/dev/null
    echo "  wrote $out"
  fi
done

if [ "$check" -eq 1 ]; then
  if [ -n "$stale" ]; then
    echo "book/_src is out of date with the notebooks:" >&2
    printf '%s' "$stale" >&2
    echo >&2
    echo "Run ./tools/build_notebooks.sh and commit the result." >&2
    exit 1
  fi
  echo "  book/_src matches all ${#targets[@]} notebooks"
fi
