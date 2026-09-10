#!/usr/bin/env python
"""Check that every relative link in the book points at a file that exists.

`suppress_warnings: [ref.doc, myst.xref_missing]` in `_config.yml` exists so that
withholding Part 4 does not fail the build -- but it also silences the warning
you would want for a genuine typo, which then renders as plain text and looks
fine. This is the replacement: it reads the sources, so it does not care what
Sphinx decided to render.

    python tools/check_links.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parents[1] / "book"
LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")
IMAGE = re.compile(r"^\s*#?\s*:?:?\{?image\}?\s*(\S+\.svg|\S+\.png)", re.M)
CARD = re.compile(r"^\s*:link:\s*(\S+)\s*$", re.M)
SKIP = ("http://", "https://", "mailto:", "data:")

# Part 4 may be withheld, in which case links into it are meant to fall back to
# plain text. They still have to point at a file that exists on disk.
def texts(path: Path):
    if path.suffix == ".ipynb":
        for cell in json.loads(path.read_text())["cells"]:
            if cell["cell_type"] == "markdown":
                yield "".join(cell["source"])
    else:
        yield path.read_text()


def main() -> int:
    broken: list[str] = []
    checked = 0
    for path in sorted(BOOK.rglob("*")):
        if path.suffix not in (".md", ".ipynb") or "_build" in path.parts:
            continue
        for text in texts(path):
            targets = [(m, False) for m in LINK.findall(text)]
            targets += [(m, False) for m in IMAGE.findall(text)]
            targets += [(m, True) for m in CARD.findall(text)]
            for target, is_card in targets:
                if target.startswith(SKIP):
                    continue
                checked += 1
                if is_card:                       # sphinx-design :link: with :link-type: doc
                    candidates = [BOOK / f"{target}.md", BOOK / f"{target}.ipynb"]
                else:
                    candidates = [(path.parent / target).resolve()]
                    stem = candidates[0].with_suffix("")
                    candidates += [stem.with_suffix(".md"), stem.with_suffix(".ipynb")]
                if not any(c.exists() for c in candidates):
                    broken.append(f"{path.relative_to(BOOK)} -> {target}")

    print(f"{checked} source links checked")
    print(f"broken: {len(broken)}")
    for item in sorted(set(broken)):
        print("  ", item)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
