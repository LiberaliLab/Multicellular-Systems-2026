#!/usr/bin/env python
"""Generate the explanatory diagrams for Parts 1 and 3.

Standard library only -- no matplotlib, no network. Each figure is written in a
light and a dark variant; the book shows the right one with the theme's
``only-light`` / ``only-dark`` classes, which follow the page's own toggle.

    python tools/make_figures.py

Edit the constants at the top of a ``fig_*`` function and re-run. The SVG is
plain text, so a diff shows exactly what moved.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "book" / "images"

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

THEMES = {
    "light": dict(
        fg="#16233d", muted="#5d6b85", hair="#c3cddd",
        panel="#f2f5fa", panel_edge="#d5deeb",
        cell="#dfe8f5", cell_edge="#b9c8de",
        accent="#1f7a8c", accent_soft="#bfe0e6",
        warm="#c4622d", warm_soft="#f6ddcb",
    ),
    "dark": dict(
        fg="#e6edf3", muted="#9fb0c4", hair="#3b4757",
        panel="#161d28", panel_edge="#2b3646",
        cell="#222d3d", cell_edge="#3a4759",
        accent="#5cc2d6", accent_soft="#1d3f49",
        warm="#eb9a68", warm_soft="#43281a",
    ),
}


# --------------------------------------------------------------------------
# tiny SVG helpers
# --------------------------------------------------------------------------
def esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def rect(x, y, w, h, fill, stroke, *, rx=3, sw=1.2, dash=None, opacity=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    extra += f' opacity="{opacity}"' if opacity is not None else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>')


def text(x, y, s, *, fill, size=13, anchor="start", weight=400, mono=False, style=""):
    family = MONO if mono else FONT
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"'
            f'{f" font-style={chr(34)}{style}{chr(34)}" if style else ""}>{esc(s)}</text>')


def line(x1, y1, x2, y2, stroke, *, sw=1.2, dash=None, marker=False):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    extra += ' marker-end="url(#arrow)"' if marker else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}" stroke-linecap="round"{extra}/>')


def svg(width, height, body, c) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img">\n'
        f'  <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{c["muted"]}"/></marker></defs>\n  '
        + "\n  ".join(body) + "\n</svg>\n"
    )


# --------------------------------------------------------------------------
# figure 1 — chunks
# --------------------------------------------------------------------------
def fig_chunks(c) -> str:
    W, H = 880, 396
    cols, rows = 8, 5
    cw, ch = 74, 46
    x0, y0 = 48, 84
    b = [text(48, 40, "One pyramid level is not one file", fill=c["fg"], size=17, weight=600),
         text(48, 63, "it is a grid of chunks, each a small file on disk",
              fill=c["muted"], size=13)]

    for r in range(rows):
        for k in range(cols):
            x, y = x0 + k * cw, y0 + r * ch
            hot = r in (1, 2) and k in (2, 3)
            b.append(rect(x, y, cw - 4, ch - 4,
                          c["accent_soft"] if hot else c["cell"],
                          c["accent"] if hot else c["cell_edge"],
                          sw=1.6 if hot else 1.0))
    # the read window, padded 8 px around the four highlighted chunks
    wx, wy = x0 + 2 * cw - 8, y0 + ch - 8
    ww, wh = 2 * cw + 12, 2 * ch + 12
    b.append(rect(wx, wy, ww, wh, "none", c["warm"], rx=5, sw=2.2, dash="6 4"))
    b.append(line(wx + ww, wy + wh / 2, wx + ww + 26, wy + wh / 2, c["warm"], sw=1.4))
    b.append(text(wx + ww + 34, wy + wh / 2 + 4, "a 512 × 512 read",
                  fill=c["warm"], size=12.5, weight=600))

    b.append(text(x0, y0 + rows * ch + 30,
                  "touches 4 chunk files — not the other 36, and not the whole array",
                  fill=c["fg"], size=13.5))
    b.append(text(x0, y0 + rows * ch + 52,
                  "the same holds on a local disk, a network filesystem or an S3 bucket",
                  fill=c["muted"], size=12.5))
    b.append(text(W - 48, y0 + rows * ch + 30, "0/2.1   0/2.2", fill=c["muted"],
                  size=11.5, anchor="end", mono=True))
    b.append(text(W - 48, y0 + rows * ch + 48, "0/3.1   0/3.2", fill=c["muted"],
                  size=11.5, anchor="end", mono=True))
    return svg(W, H, b, c)


# --------------------------------------------------------------------------
# figure 2 — multiscale pyramid
# --------------------------------------------------------------------------
def fig_pyramid(c) -> str:
    W, H = 880, 366
    b = [text(48, 40, "The same image, stored at several resolutions",
              fill=c["fg"], size=17, weight=600),
         text(48, 63, "to draw a thumbnail you read the smallest level, "
                      "not the largest one downsampled", fill=c["muted"], size=13)]

    levels = [("0", 176, "10000 × 10000", "0.65 µm/px"),
              ("1", 124, "5000 × 5000", "1.30 µm/px"),
              ("2", 86, "2500 × 2500", "2.60 µm/px"),
              ("3", 58, "1250 × 1250", "5.20 µm/px")]
    x = 60
    base_y = 246
    for i, (name, size, shape, scale) in enumerate(levels):
        y = base_y - size
        b.append(rect(x, y, size, size, c["accent_soft"] if i == 0 else c["cell"],
                      c["accent"] if i == 0 else c["cell_edge"], rx=4,
                      sw=1.8 if i == 0 else 1.2))
        b.append(text(x + size / 2, base_y + 24, f"level {name}", fill=c["fg"],
                      size=13, weight=600, anchor="middle", mono=True))
        b.append(text(x + size / 2, base_y + 43, shape, fill=c["muted"], size=11.5,
                      anchor="middle", mono=True))
        b.append(text(x + size / 2, base_y + 60, scale, fill=c["muted"], size=11.5,
                      anchor="middle", mono=True))
        x += size + 46

    b.append(line(688, 150, 812, 150, c["muted"], sw=1.4, marker=True))
    b.append(text(750, 138, "zoom out", fill=c["muted"], size=12, anchor="middle"))
    b.append(line(812, 178, 688, 178, c["muted"], sw=1.4, marker=True))
    b.append(text(750, 199, "zoom in", fill=c["muted"], size=12, anchor="middle"))
    b.append(text(48, H - 16,
                  "each level is its own array of chunks; the whole pyramid costs "
                  "about 33 % more than level 0 alone", fill=c["muted"], size=12.5))
    return svg(W, H, b, c)


# --------------------------------------------------------------------------
# figure 3 — the plate hierarchy
# --------------------------------------------------------------------------
def fig_hierarchy(c) -> str:
    W, H = 880, 364
    b = [text(48, 40, "The directory tree follows the plate",
              fill=c["fg"], size=17, weight=600),
         text(48, 63, "so the path B/03/0 is row B, column 3, image 0",
              fill=c["muted"], size=13)]

    # --- plate map ---------------------------------------------------------
    px, py, cell = 48, 96, 15.5
    ncol, nrow = 12, 8
    b.append(rect(px - 10, py - 20, ncol * cell + 26, nrow * cell + 34,
                  c["panel"], c["panel_edge"], rx=6))
    for r in range(nrow):
        for k in range(ncol):
            hot = (r == 1 and k == 2)
            b.append(rect(px + k * cell, py + r * cell, cell - 3, cell - 3,
                          c["accent"] if hot else c["cell"],
                          c["accent"] if hot else c["cell_edge"], rx=7, sw=0.9))
    for r, letter in enumerate("ABCDEFGH"):
        b.append(text(px - 8, py + r * cell + 9, letter, fill=c["muted"], size=9,
                      anchor="middle", mono=True))
    for k in range(ncol):
        b.append(text(px + k * cell + 6, py - 6, f"{k + 1:02d}", fill=c["muted"],
                      size=8, anchor="middle", mono=True))
    b.append(text(px + 40, py + nrow * cell + 26, "well B/03 highlighted",
                  fill=c["accent"], size=12, weight=600))

    # --- arrow -------------------------------------------------------------
    b.append(line(px + ncol * cell + 30, 175, px + ncol * cell + 74, 175,
                  c["muted"], sw=1.6, marker=True))

    # --- tree --------------------------------------------------------------
    tx, ty, lh = 330, 108, 25
    tree = [
        (0, "plate.zarr/", None),
        (1, ".zattrs", "rows, columns, acquisitions"),
        (1, "B/", "row"),
        (2, "03/", "column"),
        (3, "0/", "the image"),
        (4, ".zattrs", "multiscales, omero"),
        (4, "0/  1/  2/  3/  4/", "pyramid levels"),
        (4, "labels/", "segmentation masks"),
        (4, "tables/", "ROIs and measurements"),
    ]
    b.append(rect(tx - 16, ty - 24, W - tx - 32, len(tree) * lh + 30,
                  c["panel"], c["panel_edge"], rx=6))
    for i, (depth, name, note) in enumerate(tree):
        y = ty + i * lh
        hot = name in ("B/", "03/", "0/")
        b.append(text(tx + depth * 20, y, ("└── " if depth else "") + name,
                      fill=c["accent"] if hot else c["fg"], size=12.5,
                      weight=600 if hot else 400, mono=True))
        if note:
            b.append(text(W - 48, y, note, fill=c["muted"], size=11.5, anchor="end"))
    return svg(W, H, b, c)


# --------------------------------------------------------------------------
# figure 4 — the anatomy of the feature table
# --------------------------------------------------------------------------
def fig_table_anatomy(c) -> str:
    W, H = 880, 462
    b = [text(48, 40, "What you are given", fill=c["fg"], size=17, weight=600),
         text(48, 63, "one AnnData object: a matrix, a row table, and a column table",
              fill=c["muted"], size=13)]

    ox, oy = 48, 162                 # obs block
    xw, xh = 380, 210                # X block
    xx = ox + 172 + 18
    vy = oy - 46                     # var block sits above X

    # --- obs (row annotation) ---
    b.append(rect(ox, oy, 172, xh, c["panel"], c["panel_edge"], rx=5))
    b.append(text(ox + 10, oy - 10, "obs — one row per cell", fill=c["fg"], size=12, weight=600))
    obs_cols = ["well_name", "condition", "timepoint_h", "ROI", "label",
                "is_border_external", "…"]
    for i, name in enumerate(obs_cols):
        b.append(text(ox + 12, oy + 26 + i * 25, name, fill=c["muted"], size=11, mono=True))
    b.append(text(ox + 10, oy + xh + 20, "733,556 rows × 12", fill=c["muted"], size=11, mono=True))

    # --- var (column annotation) — the empty one ---
    b.append(rect(xx, vy - 26, xw, 30, c["warm_soft"], c["warm"], rx=5, sw=1.8))
    b.append(text(xx + 10, vy - 6, "var — one row per feature", fill=c["warm"],
                  size=12, weight=600))
    b.append(text(xx + xw - 10, vy - 6, "4,464 names and NOTHING ELSE",
                  fill=c["warm"], size=11.5, weight=600, anchor="end", mono=True))

    # --- X ---
    b.append(rect(xx, oy, xw, xh, c["accent_soft"], c["accent"], rx=5, sw=1.8))
    b.append(text(xx + xw / 2, oy + xh / 2 - 14, "X", fill=c["accent"], size=30,
                  weight=700, anchor="middle", mono=True))
    b.append(text(xx + xw / 2, oy + xh / 2 + 14, "733,556 × 4,464   float32",
                  fill=c["fg"], size=12.5, anchor="middle", mono=True))
    b.append(text(xx + xw / 2, oy + xh / 2 + 36, "13.1 GB dense",
                  fill=c["muted"], size=12, anchor="middle", mono=True))

    # --- what is missing ---
    nx = xx + xw + 26
    b.append(text(nx, oy + 12, "no layers", fill=c["muted"], size=11.5, mono=True))
    b.append(text(nx, oy + 34, "no obsm", fill=c["muted"], size=11.5, mono=True))
    b.append(text(nx, oy + 56, "no uns", fill=c["muted"], size=11.5, mono=True))
    b.append(text(nx, oy + 78, "no var", fill=c["warm"], size=11.5, weight=600, mono=True))
    b.append(text(nx, oy + 100, "columns", fill=c["warm"], size=11.5, weight=600, mono=True))

    b.append(text(48, H - 44,
                  "A column name says the channel and the imaging round — never the antibody.",
                  fill=c["fg"], size=13))
    b.append(text(48, H - 22,
                  "Filling that empty var table is what Steps 7 to 9 do.",
                  fill=c["muted"], size=12.5))
    return svg(W, H, b, c)


# --------------------------------------------------------------------------
# figure 5 — renaming a feature with its marker
# --------------------------------------------------------------------------
def fig_rename(c) -> str:
    W, H = 880, 412
    b = [text(48, 40, "Renaming a feature with its marker", fill=c["fg"], size=17, weight=600),
         text(48, 63, "the column name carries the channel and the round; the sheet turns "
                      "those into an antibody", fill=c["muted"], size=13)]

    # --- the raw column name, split into parts ---
    parts = [("cells_", c["muted"], 62), ("Intensity_", c["fg"], 92),
             ("mean_intensity_", c["fg"], 136), ("Texas Red", c["accent"], 92),
             ("_", c["muted"], 12), ("0", c["accent"], 14)]
    x, y = 62, 122
    b.append(rect(48, y - 26, W - 96, 42, c["panel"], c["panel_edge"], rx=5))
    for label, colour, w in parts:
        weight = 700 if colour == c["accent"] else 400
        b.append(text(x, y, label, fill=colour, size=14, mono=True, weight=weight))
        x += w
    b.append(text(62, y + 30, "family", fill=c["muted"], size=10.5, mono=True))
    b.append(text(216, y + 30, "statistic", fill=c["muted"], size=10.5, mono=True))
    b.append(text(352, y + 30, "channel", fill=c["accent"], size=10.5, weight=600, mono=True))
    b.append(text(452, y + 30, "round", fill=c["accent"], size=10.5, weight=600, mono=True))

    # --- arrow down ---
    b.append(line(200, y + 48, 200, y + 78, c["muted"], sw=1.4, marker=True))
    b.append(text(214, y + 70, "look up (channel, round) in the staining sheet",
                  fill=c["muted"], size=12))

    # --- the sheet row ---
    ty = y + 108
    b.append(rect(48, ty, W - 96, 74, c["panel"], c["panel_edge"], rx=5))
    heads = [("round", 66), ("channel", 150), ("marker", 300), ("threshold", 500), ("failed", 640)]
    for label, hx in heads:
        b.append(text(hx, ty + 24, label, fill=c["muted"], size=11, weight=600, mono=True))
    row = [("0", 66), ("Texas Red", 150), ("Foxo3a", 300), ("500", 500), ("no", 640)]
    for label, hx in row:
        hot = label in ("Texas Red", "Foxo3a", "0")
        b.append(text(hx, ty + 52, label, fill=c["accent"] if hot else c["fg"],
                      size=13, weight=700 if hot else 400, mono=True))

    # --- arrow down to the result ---
    b.append(line(200, ty + 92, 200, ty + 118, c["muted"], sw=1.4, marker=True))
    ry = ty + 148
    b.append(rect(48, ry - 24, 470, 40, c["accent_soft"], c["accent"], rx=5, sw=1.8))
    b.append(text(64, ry + 2, "var['marker'] = 'Foxo3a'", fill=c["accent"], size=14,
                  weight=700, mono=True))
    b.append(text(300, ry + 2, "var['theme'] = 'signaling'", fill=c["fg"], size=13, mono=True))
    b.append(text(544, ry + 2, "3,654 of 4,464 columns get a marker this way",
                  fill=c["muted"], size=11.5))
    return svg(W, H, b, c)


# --------------------------------------------------------------------------
# figure 6 — what Stage 1 hands over (the answer to figure 4)
# --------------------------------------------------------------------------
def fig_clean_object(c) -> str:
    W, H = 880, 470
    b = [text(48, 40, "What Stage 1 hands over", fill=c["fg"], size=17, weight=600),
         text(48, 63, "mcs2026_clean.h5ad — the same object, with every slot filled in",
              fill=c["muted"], size=13)]

    ox, oy = 48, 168
    ow, oh = 172, 196
    xx = ox + ow + 18
    xw, xh = 340, 150
    ux = xx + xw + 26
    uw = W - ux - 48

    # --- obs ---
    b.append(rect(ox, oy, ow, oh, c["panel"], c["panel_edge"], rx=5))
    b.append(text(ox + 10, oy - 10, "obs — one row per cell", fill=c["fg"], size=12, weight=600))
    for i, name in enumerate(["well", "condition", "timepoint_h",
                              "replicate", "area", "dapi", "…"]):
        new = name in {"replicate", "area", "dapi"}
        weight = 700 if new else 400
        fill = c["accent"] if new else c["muted"]
        b.append(text(ox + 12, oy + 26 + i * 24, name, fill=fill, size=11,
                      weight=weight, mono=True))
    b.append(text(ox + 10, oy + oh + 20, "~653,000 rows", fill=c["muted"], size=11, mono=True))

    # --- var, now filled ---
    b.append(rect(xx, oy - 36, xw, 30, c["accent_soft"], c["accent"], rx=5, sw=1.8))
    b.append(text(xx + 10, oy - 16, "var", fill=c["accent"], size=12, weight=600))
    b.append(text(xx + xw - 10, oy - 16, "marker · channel · round · theme",
                  fill=c["accent"], size=11.5, weight=600, anchor="end", mono=True))

    # --- X ---
    b.append(rect(xx, oy, xw, xh, c["accent_soft"], c["accent"], rx=5, sw=1.8))
    b.append(text(xx + xw / 2, oy + xh / 2 - 12, "X", fill=c["accent"], size=28,
                  weight=700, anchor="middle", mono=True))
    b.append(text(xx + xw / 2, oy + xh / 2 + 14, "~653,000 × 38   float32",
                  fill=c["fg"], size=12.5, anchor="middle", mono=True))
    b.append(text(xx + xw / 2, oy + xh / 2 + 36, "control-cell SDs, within timepoint",
                  fill=c["muted"], size=11.5, anchor="middle"))

    # --- layers ---
    ly = oy + xh + 12
    b.append(rect(xx, ly, xw, 34, c["panel"], c["panel_edge"], rx=5))
    b.append(text(xx + 12, ly + 22, 'layers["raw"]', fill=c["fg"], size=11.5,
                  weight=600, mono=True))
    b.append(text(xx + xw - 12, ly + 22, "intensity as measured",
                  fill=c["muted"], size=11, anchor="end"))

    # --- uns ---
    b.append(rect(ux, oy - 36, uw, oh + 82, c["warm_soft"], c["warm"], rx=5, sw=1.8))
    b.append(text(ux + 12, oy - 16, 'uns["provenance"]', fill=c["warm"], size=12,
                  weight=600, mono=True))
    notes = ["built_by", "layout_workbook", "cells_removed", "wells_removed",
             "features", "X", "layers_raw", "units"]
    for i, name in enumerate(notes):
        b.append(text(ux + 12, oy + 18 + i * 24, name, fill=c["fg"], size=11, mono=True))
    b.append(text(ux + 12, oy + 18 + len(notes) * 24 + 6,
                  "why, not just what", fill=c["warm"], size=11, style="italic"))

    b.append(text(48, H - 44,
                  "One read_h5ad, and every later chapter starts with no set-up.",
                  fill=c["fg"], size=13))
    b.append(text(48, H - 22,
                  "The wide 2,587-column table stays on disk as mcs2026_slim.h5ad.",
                  fill=c["muted"], size=12.5))
    return svg(W, H, b, c)


FIGURES = {
    "zarr_chunks": fig_chunks,
    "zarr_pyramid": fig_pyramid,
    "zarr_hierarchy": fig_hierarchy,
    "feature_table_anatomy": fig_table_anatomy,
    "feature_rename": fig_rename,
    "clean_object": fig_clean_object,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, builder in FIGURES.items():
        for theme, colours in THEMES.items():
            path = OUT / f"{name}_{theme}.svg"
            path.write_text(builder(colours), encoding="utf-8")
            print(f"  wrote {path.relative_to(OUT.parents[1])}  ({path.stat().st_size / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
