"""LEXICON: how shop owners write the words the code reads, mapped to one standard form before any rule reads them (text.norm).

The question readers (agents/question.py, agents/periods.py, agents/guideline.py, agents/facts.py) are regexes on folded text. Without
this table every regex listed its own variants ("bao nhieu|bao nhiu|bnhiu|bn"), and a variant one list missed changed what the code
decided. An edge case is now one line here, not a branch of code.

Each entry is (variant, standard, gloss):
- matched as a whole word (or run of words);
- a variant written WITH diacritics is matched on the lower-case accented text, before folding: folded, "o đờ" would also be
  "ở đó" and "đh" would be any "dh". The others are matched on the folded text, after the telex slips are undone;
- a variant is not expanded where the text already reads as its standard ("bao nhiêu" keeps its "nhiêu");
- `gloss` (optional): the meaning shown to the model in the spec prompt (agents/question.QuestionFacts.prompt). Only for words the
  model misread or cannot be expected to know: "Quý 4 tiệm chốt được bao nhiêu o đờ" was read as revenue (public q_37a4aca072a6).

Each entry was counted in the questions before it was added (scripts/oov_discovery.py lists the candidates), and its contexts read.
An entry that maps a phrase to itself protects it: the longest variant wins, so "mô tô" stays a motorbike while "hàng mô" = "hàng nào".
Left out on purpose: `dt` / `đt` (doanh thu, but "máy đt" = điện thoại in the phone shops), `hd` (a product: "Telfast HD 180mg"), `hong` without
diacritics (họng, hỏng or không), `k` ("không", but also "nghìn"), `kh` ("khách" or "không"), and teencode RULES (k->c, z->d...):
even checked against the vocabulary they gave wrong words ("zay" -> "day" for "vậy", "cko" -> "co" for "cho"); "chừ" (agents/periods.py reads
"tới chừ" itself; "giờ" would not), "răng" ("kem đánh răng").

DATAMATE_LEXICON=0 turns the table off (A/B).
"""
from __future__ import annotations

import os
import re
import unicodedata

import datamate.config  # noqa: F401 - loads .env before the switches below are read, whatever imports this module first

ENTRIES: list[tuple[str, str, str | None]] = [
    # orders (the variants named in the ticket; "o đờ" was read as revenue)
    ("o đờ", "đơn hàng", "đơn hàng (order)"),
    ("oder", "don hang", "đơn hàng (order)"),
    ("order", "don hang", "đơn hàng (order)"),
    ("đh", "đơn hàng", "đơn hàng (order)"),
    ("dh", "don hang", "đơn hàng (order)"),
    # measures and objects
    ("dthu", "doanh thu", None),
    ("sp", "san pham", None),
    # payment methods
    ("ck", "chuyen khoan", "chuyển khoản (bank transfer)"),
    ("tm", "tien mat", "tiền mặt (cash)"),
    ("xi ô đi", "cod", "COD (thu tiền khi giao hàng)"),
    # how many ("nhiêu" alone, with its circumflex: "là nhiêu" = "là bao nhiêu"; not "nhiều")
    ("bnhiu", "bao nhieu", None),
    ("bnhieu", "bao nhieu", None),
    ("bao nhiu", "bao nhieu", None),
    ("bn", "bao nhieu", None),
    ("nhiêu", "bao nhiêu", None),
    ("nhiu", "nhieu", None),
    # time
    ("hnay", "hom nay", None),
    ("bua nay", "hom nay", None),
    ("hqua", "hom qua", None),
    ("bua qua", "hom qua", None),
    ("trc", "truoc", None),
    # grammar words and teencode
    ("dc", "duoc", None),
    ("ko", "khong", None),
    ("hok", "khong", None),
    ("hông", "không", None),
    ("j", "gi", None),
    ("mik", "minh", None),
    ("mk", "minh", None),
    ("wan", "quan", None),
    ("jum", "gium", None),
    ("jup", "giup", None),
    ("trog", "trong", None),
    ("cko", "cho", None),
    ("kua", "cua", None),
    ("kon", "con", None),
    ("kac", "cac", None),
    ("tih", "tinh", None),
    ("nv", "nhan vien", None),
    ("ncc", "nha cung cap", None),
    ("ngui", "nguoi", None),
    # southern / central dialect ("tháng ni" = "tháng này", "hàng mô" = "hàng nào", "nhứt" = "nhất")
    ("nhứt", "nhất", None),
    ("nhut", "nhat", None),
    ("ni", "nay", None),
    ("mô", "nào", None),
    ("mô tô", "mô tô", None),          # protected: a motorbike, not "nào tô"
    ("rứa", "vậy", None),
    ("mún", "muốn", None),
    ("bít", "biết", None),
]

_on = lambda name, default: os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}   # noqa: E731
ENABLED = _on("DATAMATE_LEXICON", "1")
GLOSSARY = _on("DATAMATE_GLOSSARY", "1")             # the glossed words in the spec prompt


def _accented(s: str) -> bool:
    return any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", s)) or "đ" in s


def _compile(entries):
    table = {unicodedata.normalize("NFC", v): unicodedata.normalize("NFC", std) for v, std, _ in entries}
    if not table:
        return None, table
    alt = "|".join(re.escape(v) for v in sorted(table, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alt})(?!\w)"), table


ACCENTED, ACCENTED_MAP = _compile([e for e in ENTRIES if _accented(e[0])])
FOLDED, FOLDED_MAP = _compile([e for e in ENTRIES if not _accented(e[0])])
GLOSS = [(unicodedata.normalize("NFC", v), g) for v, _, g in ENTRIES if g]


def _sub(rx: re.Pattern | None, table: dict, text: str) -> str:
    if rx is None:
        return text

    def one(m: re.Match) -> str:
        v, std = m.group(0), table[m.group(0)]
        head = std[: -len(v)]
        if std.endswith(" " + v) and m.string[: m.start()].endswith(head):
            return v                                   # already the standard: "bao nhiêu" keeps its "nhiêu"
        return std
    return rx.sub(one, text)


def apply_accented(text: str) -> str:
    """Lower-case NFC text with the accented variants replaced (before folding)."""
    return _sub(ACCENTED, ACCENTED_MAP, text) if ENABLED else text


def apply_folded(text: str) -> str:
    """Folded text with the folded variants replaced."""
    return _sub(FOLDED, FOLDED_MAP, text) if ENABLED else text


def glossary(question: str) -> list[tuple[str, str]]:
    """[(word as the question writes it, meaning)] for the glossed entries the question uses, in reading order, once each."""
    if not (ENABLED and GLOSSARY) or not question:
        return []
    from datamate.text import fold_text    # no import cycle at module load: text imports this module
    low = unicodedata.normalize("NFC", question.lower())
    folded = fold_text(question)
    found: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for v, g in GLOSS:
        where = low if _accented(v) else folded
        m = re.search(rf"(?<!\w){re.escape(v)}(?!\w)", where)
        if m and g not in seen:
            seen.add(g)
            found.append((m.start(), v, g))
    return [(v, g) for _, v, g in sorted(found)]
