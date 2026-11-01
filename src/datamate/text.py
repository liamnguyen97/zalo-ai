"""Text helpers shared by the data layer, the agents and the coder's sandbox.

Two comparison forms exist on purpose (they are not interchangeable):
- `fold`: no accents, lowercase, single spaces, punctuation KEPT. Item keys in `item_lines.key` are built with it and the
  coder compares names with `key == fold(name)`, so names like "Ống PVC D50 (4m)" keep their punctuation.
- `fold_text`: like `fold` but punctuation becomes a space. Used to match text a user typed against data values and
  against the question (find_values, evidence quotes), where punctuation is noise.
"""
from __future__ import annotations

import re
import unicodedata


def nodia(s: str) -> str:
    """Remove Vietnamese diacritics (đ -> d), keep case."""
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")


def fold(s) -> str:
    return re.sub(r"\s+", " ", nodia(str(s)).lower()).strip()


def fold_text(s) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s./%-]", " ", s)).strip()
