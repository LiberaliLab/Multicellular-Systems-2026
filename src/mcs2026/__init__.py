"""Helpers for the Multicellular Systems 2026 hands-on course.

Three modules do the real work:

``layout``   parse the 384-well plate layout workbook
``decode``   turn 4,464 bare column names into an annotated ``var`` table
``panels``   the four biological themes, and how to select features for one
"""

__version__ = "0.1.0"

from mcs2026 import analysis, decode, layout, panels  # noqa: F401

__all__ = ["analysis", "decode", "layout", "panels"]
