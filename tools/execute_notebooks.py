#!/usr/bin/env python
"""Execute course notebooks in place, keeping their declared kernel.

The book ships notebooks *with* outputs so the website builds without cluster
data. This runs them against the real data and saves the results.

    python tools/execute_notebooks.py book/part3_analysis/00_experiment_and_layout.ipynb
    python tools/execute_notebooks.py book/part3_analysis          # a whole part
    python tools/execute_notebooks.py --kernel python3 <path>      # local check

The notebook's own ``kernelspec`` (``Python (MCS 2026)``) is restored after
execution, so running with a different kernel locally does not leak into the
committed file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def execute(path: Path, kernel: str | None, timeout: int) -> bool:
    notebook = nbformat.read(path, as_version=4)
    declared = notebook.metadata.get("kernelspec")

    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel or (declared or {}).get("name", "python3"),
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    try:
        client.execute()
    except CellExecutionError as error:
        # Do NOT write. A half-executed notebook would replace committed outputs
        # with a traceback, and the site ships whatever is in the file.
        print(f"  FAILED {path}  (left unchanged)\n    {str(error).splitlines()[-1]}")
        return False

    if declared is not None:
        notebook.metadata["kernelspec"] = declared
    nbformat.write(notebook, path)
    print(f"  ok     {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument("--kernel", default=None, help="override the notebook's kernel")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    paths: list[Path] = []
    for target in args.targets:
        if target.is_dir():
            paths.extend(sorted(target.glob("*.ipynb")))
        else:
            paths.append(target)

    failures = [p for p in paths if not execute(p, args.kernel, args.timeout)]
    if failures:
        print(f"\n{len(failures)} of {len(paths)} notebooks failed")
        return 1
    print(f"\n{len(paths)} notebooks executed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
