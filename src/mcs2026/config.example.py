"""Local paths. Copy to ``config.py`` and edit; ``config.py`` is gitignored.

    cp src/mcs2026/config.example.py src/mcs2026/config.py

Nothing else in the course hard-codes a path, so this is the only file you
need to touch when the data moves.
"""

from pathlib import Path

# Root of the course data on Euler. Your instructor gives you this path.
DATA_ROOT = Path("/cluster/work/liberali/COURSE/mcs2026")

# --- Part 1 and 2: one OME-Zarr HCS plate -----------------------------------
PLATE_PATH = DATA_ROOT / "plate" / "HNES1_MP_PL1_mip.zarr"

# --- Part 3: the single-cell feature table ----------------------------------
# The full table, used once by part3/01_decode_and_slim.ipynb.
H5AD_FULL = DATA_ROOT / "tables" / "1_FE_pooled1.h5ad"
# The slim table that chapter 01 writes and every later chapter reads.
H5AD_SLIM = DATA_ROOT / "tables" / "mcs2026_slim.h5ad"

# --- the plate layout workbook (small, lives in the repo) -------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
LAYOUT_XLSX = REPO_ROOT / "metadata" / "L_ayout_384_Haralick_Thresholds.xlsx"
METADATA_DIR = REPO_ROOT / "metadata"

# Where your own outputs go. Keep this off the repo (it is gitignored anyway).
OUTPUT_DIR = Path.home() / "mcs2026_outputs"
