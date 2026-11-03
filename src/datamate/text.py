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


# Telex typing left in a word ("naof", "nhaats", "khoong", "quys") is undone (from zalo-challenge `norm_q`).
FINALS = re.compile(r"(?:[aeiouy]|c|ch|m|n|ng|nh|p|t)$")
# a folded Vietnamese syllable: initial consonant, 1-3 vowels, final consonant. A trailing tone letter (s f r x j) of a longer token is
# a telex slip when what is left is a syllable and the token is not: "thangs" (44 questions), "khachs", "nhieuf", "thoaij"; while
# "thanks", "class", "plus" are left alone
SYLLABLE = re.compile(r"^(?:ngh|ng|gh|gi|kh|nh|ph|qu|th|tr|ch|[bcdghklmnprstvx])?[aeiouy]{1,3}(?:ng|nh|ch|[cmnpt])?$")


def _telex(tok: str) -> str:
    if not tok.isalpha() or len(tok) < 2:
        return tok
    t = re.sub(r"dd", "d", tok)
    t = re.sub(r"(aa|ee|oo)", lambda m: m.group(1)[0], t)
    t = re.sub(r"(a|o|u)w", r"\1", t)
    if t[-1] in "sfrxj" and FINALS.search(t[:-1]) and (t != tok or len(t) <= 5 or (SYLLABLE.match(t[:-1]) and not SYLLABLE.match(t))):
        t = t[:-1]
    return t


def norm(text: str) -> str:
    """Folded (no accents, lower case, punctuation as spaces), telex slips undone, the shop owners' spellings in their standard form
    (datamate/lexicon.py: "đh" -> "don hang", "bnhiu" -> "bao nhieu"; accented variants are read before folding)."""
    from datamate import lexicon      # imported here: lexicon imports text for its glossary
    low = lexicon.apply_accented(unicodedata.normalize("NFC", (text or "").lower()))
    return lexicon.apply_folded(" ".join(_telex(w) for w in fold_text(low).split()))
