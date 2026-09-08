#!/usr/bin/env python
"""Generate the explanatory diagrams for Part 1, chapter 1.

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


FIGURES = {
    "zarr_chunks": fig_chunks,
    "zarr_pyramid": fig_pyramid,
    "zarr_hierarchy": fig_hierarchy,
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
