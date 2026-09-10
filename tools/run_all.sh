#!/usr/bin/env bash
# Execute the Part 3 and Part 4 notebooks against real data, in dependency order.
#
#   ./tools/run_all.sh                  # everything
#   ./tools/run_all.sh --from 06_pca    # from that chapter onward
#   KERNEL=python3 ./tools/run_all.sh   # use a different kernel than the notebooks declare
#
# ORDER MATTERS. The chapters form a chain: chapter 03 writes mcs2026_clean.h5ad,
# chapter 04 cuts it into mcs2026_controls.h5ad (Stage 2) and mcs2026_sketch.h5ad
# (Part 4), and chapters 06-10 each add a slot (X_pca, X_umap, cell_state, paga,
# X_diffmap) to the controls file. Re-running an early chapter on its own therefore
# REBUILDS those files and silently discards everything the later ones wrote into
# them -- they then fail with a missing obsm or obs key. If you re-run one chapter,
# re-run the rest of the chain after it.
#
# 01_columns_to_markers needs the full 13 GB feature table; the others read what
# the chapter before them wrote.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
KERNEL_ARGS=()
[ -n "${KERNEL:-}" ] && KERNEL_ARGS=(--kernel "$KERNEL")

CHAPTERS=(
  part3_analysis/1_preparation/00_what_you_are_given
  part3_analysis/1_preparation/01_columns_to_markers
  part3_analysis/1_preparation/02_quality_control
  part3_analysis/1_preparation/03_normalisation
  part3_analysis/1_preparation/04_subsetting_and_sketching
  part3_analysis/2_controls/05_the_anndata_object
  part3_analysis/2_controls/06_pca
  part3_analysis/2_controls/07_umap
  part3_analysis/2_controls/08_cell_type_annotation
  part3_analysis/2_controls/09_paga
  part3_analysis/2_controls/10_diffusion_map
  part3_analysis/11_your_turn
  part4_final_solutions/00_full_dataset_analysis
  part4_final_solutions/01_signaling
  part4_final_solutions/02_mechanics
  part4_final_solutions/03_metabolism
  part4_final_solutions/04_organelles
  part4_final_solutions/05_integration
)

start=0
if [ "${1:-}" = "--from" ]; then
  [ $# -ge 2 ] || { echo "usage: $0 --from <chapter-name>" >&2; exit 2; }
  for i in "${!CHAPTERS[@]}"; do
    case "${CHAPTERS[$i]}" in *"$2"*) start=$i; break;; esac
  done
  echo "starting at ${CHAPTERS[$start]}"
fi

targets=()
for ((i = start; i < ${#CHAPTERS[@]}; i++)); do
  targets+=("book/${CHAPTERS[$i]}.ipynb")
done

"$PYTHON" tools/execute_notebooks.py "${KERNEL_ARGS[@]}" "${targets[@]}"
