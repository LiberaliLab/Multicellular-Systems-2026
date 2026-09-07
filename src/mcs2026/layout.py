"""Read the 384-well plate layout workbook.

The workbook (``L_ayout_384_Haralick_Thresholds.xlsx``) is the decoder ring for
the whole experiment. It has four sheets, all describing the same plate:

===================  =========================================================
``MediumLayout``     which condition (``Cnd1``..``Cnd18``) is in each well,
                     plus the legend mapping those codes to compound names
``OtherLayout``      the timepoint in each well (36/48/60/84 h)
``LineLayout``       the cell line in each well (uniform here)
``StainingLayout``   which antibody was imaged in which ``(round, channel)``
===================  =========================================================

The last one matters most: the feature columns of the single-cell table are
named by *channel and round*, never by marker, so ``StainingLayout`` is the
only place the biology is written down.

Every function returns a tidy :class:`pandas.DataFrame`. Nothing here needs the
big feature table, so this module runs on a laptop in under a second.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

SHEET_CONDITION = "MediumLayout"
SHEET_STAINING = "StainingLayout"
SHEET_LINE = "LineLayout"
SHEET_TIMEPOINT = "OtherLayout"

#: Imaging channels, in the order they appear in each staining round.
CHANNELS: tuple[str, ...] = ("Texas Red", "FITC", "Cy5", "DAPI")

#: Column offsets within one antibody block of ``StainingLayout``.
#: The sheet repeats this 7-column pattern four times per round.
_STAIN_FIELDS = (
    "marker",
    "species",
    "manufacturer",
    "catalogue",
    "intensity_threshold",
    "haralick_comment",
    "channel",
)

_EMPTY_MARKER = {"x", "-", ""}


# --------------------------------------------------------------------------
# finding things in a hand-made spreadsheet
# --------------------------------------------------------------------------
def _read_raw(xlsx: str | Path, sheet: str) -> pd.DataFrame:
    """Read a sheet with no header at all, so cell positions are preserved."""
    return pd.read_excel(xlsx, sheet_name=sheet, header=None)


def _find_well_grid(raw: pd.DataFrame) -> tuple[int, int, list[str], list[int]]:
    """Locate the 384-well grid inside a hand-made sheet.

    Rather than hard-coding cell addresses (which break the moment somebody
    inserts a row), we look for the structure itself: a column holding the row
    letters ``A``..``P``, and a row holding the column numbers ``1``..``24``.

    Returns ``(letter_col, number_row, row_letters, col_numbers)``.
    """
    letter_col = number_row = None
    for col in raw.columns:
        values = {str(v).strip() for v in raw[col].dropna()}
        if {"A", "B", "O", "P"} <= values:
            letter_col = col
            break
    if letter_col is None:
        raise ValueError("no column holds the plate row letters A..P")

    for idx in raw.index:
        numbers = {
            int(v)
            for v in raw.loc[idx]
            if isinstance(v, (int, float, np.integer, np.floating))
            and not pd.isna(v)
            and float(v).is_integer()
        }
        if {1, 12, 24} <= numbers:
            number_row = idx
            break
    if number_row is None:
        raise ValueError("no row holds the plate column numbers 1..24")

    letters = [
        str(raw.at[i, letter_col]).strip()
        for i in raw.index
        if i > number_row
        and isinstance(raw.at[i, letter_col], str)
        and re.fullmatch(r"[A-P]", str(raw.at[i, letter_col]).strip())
    ]
    numbers = {
        col: int(raw.at[number_row, col])
        for col in raw.columns
        if isinstance(raw.at[number_row, col], (int, float, np.integer, np.floating))
        and not pd.isna(raw.at[number_row, col])
    }
    return letter_col, number_row, letters, numbers


def read_plate_grid(xlsx: str | Path, sheet: str, value_name: str) -> pd.DataFrame:
    """Read one plate-shaped sheet into tidy ``(row, column, well, <value>)`` rows.

    Empty wells are dropped, so the result has one row per *used* well.
    """
    raw = _read_raw(xlsx, sheet)
    letter_col, number_row, letters, numbers = _find_well_grid(raw)

    records = []
    for i in raw.index:
        if i <= number_row:
            continue
        letter = raw.at[i, letter_col]
        if not isinstance(letter, str) or letter.strip() not in letters:
            continue
        letter = letter.strip()
        for col, number in numbers.items():
            value = raw.at[i, col]
            if pd.isna(value):
                continue
            records.append(
                {
                    "row": letter,
                    "column": number,
                    "well": f"{letter}{number:02d}",
                    value_name: value,
                }
            )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------
# the four sheets
# --------------------------------------------------------------------------
def read_conditions(xlsx: str | Path) -> pd.DataFrame:
    """The ``Cnd*`` -> compound legend from ``MediumLayout``.

    Returns columns ``condition_code``, ``compound``, ``is_control``.
    """
    raw = _read_raw(xlsx, SHEET_CONDITION)

    # The legend lives under a "Conditions" header, to the right of the plate
    # grid. Anchor on that header: the grid itself is full of Cnd codes, so
    # "the column with Cnd values in it" would match the wrong thing.
    code_col = None
    for i in raw.index:
        for col in raw.columns:
            if str(raw.at[i, col]).strip().lower() == "conditions":
                code_col = col
                break
        if code_col is not None:
            break
    if code_col is None:
        raise ValueError(f"no 'Conditions' legend header found in {SHEET_CONDITION!r}")

    records = []
    for i in raw.index:
        code = raw.at[i, code_col]
        if not isinstance(code, str) or not re.fullmatch(r"Cnd\d+", code.strip()):
            continue
        compound = raw.at[i, code_col + 1]
        records.append(
            {
                "condition_code": code.strip(),
                "compound": str(compound).strip() if not pd.isna(compound) else None,
            }
        )
    out = pd.DataFrame.from_records(records)
    out["is_control"] = out["compound"].str.upper().isin({"DMSO", "PBS"})
    return out.sort_values(
        "condition_code", key=lambda s: s.str.removeprefix("Cnd").astype(int)
    ).reset_index(drop=True)


def read_stainings(xlsx: str | Path) -> pd.DataFrame:
    """The ``(round, channel)`` -> antibody table from ``StainingLayout``.

    This is the decoder for every feature column in the single-cell table.
    Returns one row per stain, with columns ``round``, ``channel``, ``marker``,
    ``species``, ``manufacturer``, ``catalogue``, ``intensity_threshold``,
    ``haralick_threshold`` and ``failed``.

    Two conventions in the sheet are worth knowing:

    * ``x`` in the marker cell means *no antibody in this slot this round*.
    * ``failed`` in the threshold cell means the stain did not work. Those
      columns are still present in the feature table and must be dropped.
    """
    raw = _read_raw(xlsx, SHEET_STAINING)

    header_row, round_col = None, None
    for i in raw.index:
        for col in raw.columns:
            if str(raw.at[i, col]).strip().lower() == "round":
                header_row, round_col = i, col
                break
        if header_row is not None:
            break
    if header_row is None:
        raise ValueError(f"no 'Round' header found in {SHEET_STAINING!r}")

    # Antibody blocks start at each 'Name' header on the header row.
    blocks = [
        col
        for col in raw.columns
        if col > round_col and str(raw.at[header_row, col]).strip().lower() == "name"
    ]
    if not blocks:
        raise ValueError("no antibody 'Name' blocks found")

    records = []
    for i in raw.index:
        if i <= header_row:
            continue
        rnd = raw.at[i, round_col]
        if pd.isna(rnd):
            continue
        for base in blocks:
            entry = {
                field: raw.at[i, base + offset] if base + offset in raw.columns else None
                for offset, field in enumerate(_STAIN_FIELDS)
            }
            marker = entry["marker"]
            if pd.isna(marker) or str(marker).strip().lower() in _EMPTY_MARKER:
                continue
            channel = entry["channel"]
            if pd.isna(channel):
                continue
            threshold = entry["intensity_threshold"]
            failed = "fail" in str(threshold).lower()
            records.append(
                {
                    "round": int(rnd),
                    "channel": str(channel).strip(),
                    "marker_full": str(marker).strip(),
                    "species": _clean(entry["species"]),
                    "manufacturer": _clean(entry["manufacturer"]),
                    "catalogue": _clean(entry["catalogue"]),
                    "intensity_threshold": None if failed else _number(threshold),
                    "haralick_threshold": _haralick(entry["haralick_comment"]),
                    "failed": failed,
                }
            )

    out = pd.DataFrame.from_records(records)
    out["marker"] = out["marker_full"].map(short_marker_name)
    cols = [
        "round", "channel", "marker", "marker_full", "species", "manufacturer",
        "catalogue", "intensity_threshold", "haralick_threshold", "failed",
    ]
    return out[cols].sort_values(["round", "channel"]).reset_index(drop=True)


def read_wells(xlsx: str | Path) -> pd.DataFrame:
    """One row per used well, joining condition, timepoint and cell line."""
    condition = read_plate_grid(xlsx, SHEET_CONDITION, "condition_code")
    timepoint = read_plate_grid(xlsx, SHEET_TIMEPOINT, "timepoint_h")
    line = read_plate_grid(xlsx, SHEET_LINE, "cell_line")

    wells = condition.merge(
        timepoint[["well", "timepoint_h"]], on="well", how="outer"
    ).merge(line[["well", "cell_line"]], on="well", how="outer")

    legend = read_conditions(xlsx)
    wells = wells.merge(legend, on="condition_code", how="left")
    wells["timepoint_h"] = pd.to_numeric(wells["timepoint_h"], errors="coerce").astype("Int64")
    return wells.sort_values(["row", "column"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _clean(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return None if text.lower() in _EMPTY_MARKER else text


def _number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _haralick(value) -> float | None:
    """Pull the number out of a comment like ``HLK:3000`` or ``HLK: 750``."""
    if pd.isna(value):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else None


#: Long catalogue descriptions -> the short name used in plots and panels.
#: Matched case-insensitively as substrings, first hit wins, so order matters:
#: put the more specific pattern first (``Foxo3a`` before ``Foxo1``).
MARKER_ALIASES: tuple[tuple[str, str], ...] = (
    ("Foxo3a", "Foxo3a"),
    ("Foxo1", "Foxo1"),
    ("beta-Catenin", "beta-Catenin"),
    ("FGFR1", "FGFR1"),
    ("PDGF R alpha", "PDGFRa"),
    ("YAP1", "YAP1"),
    ("Phospho S6", "p-S6"),
    ("RNA polymerase II", "RNAPII-pS5"),
    ("c-Myc", "c-Myc"),
    ("Phospho-AKT", "p-AKT"),
    ("Phospho-Myosin", "p-MyosinIIa"),
    ("ZO-1", "ZO-1"),
    ("Tubulin", "a-Tubulin"),
    ("E-cadherin", "E-cadherin"),
    ("Fibronectin", "Fibronectin"),
    ("LAMA4", "LAMA4"),
    ("Lamin B1", "LaminB1"),
    ("Lamin A", "LaminA"),
    ("Mitochondria", "Mitochondria"),
    ("Pmp70", "Pmp70"),
    ("GRP78", "GRP78"),
    ("HSP90", "HSP90"),
    ("Calreticulin", "Calreticulin"),
    ("Integrin beta 1", "Integrin-b1"),
    ("EEA1", "EEA1"),
    ("GM130", "GM130"),
    ("Giantin", "Giantin"),
    ("LAMP1", "LAMP1"),
    ("DDX6", "DDX6"),
    ("Oct3/4", "Oct4"),
    ("Nanog", "Nanog"),
    ("GATA-6", "GATA6"),
    ("Gata-4", "GATA4"),
    ("Sox2", "Sox2"),
    ("SOX17", "SOX17"),
    ("GATA3", "GATA3"),
    ("Cyclin A2", "CyclinA2"),
    ("p21", "p21"),
    ("Ki 67", "Ki67"),
    ("DAPI", "DAPI"),
)


def short_marker_name(full_name: str) -> str:
    """Map a catalogue description to a short plotting name.

    ``'Anti-LAMP1 antibody - Lysosome Marker (ab24170)'`` -> ``'LAMP1'``.
    Unrecognised names are returned trimmed, so a new antibody shows up in
    plots rather than silently vanishing.
    """
    haystack = str(full_name).lower()
    for pattern, short in MARKER_ALIASES:
        if pattern.lower() in haystack:
            return short
    return str(full_name).strip()[:24]


# --------------------------------------------------------------------------
# writing the tidy tables the notebooks read
# --------------------------------------------------------------------------
def write_metadata(xlsx: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Write ``conditions.csv``, ``wells.csv`` and ``stainings.csv``.

    These are committed to the repo so the layout is readable without opening
    Excel, and so a diff shows up if the workbook ever changes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "conditions": read_conditions(xlsx),
        "wells": read_wells(xlsx),
        "stainings": read_stainings(xlsx),
    }
    written = {}
    for name, table in tables.items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path
    return written
