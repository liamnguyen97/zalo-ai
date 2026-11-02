"""QUESTION READING IN CODE: what the question names, asks for and when, read before any model call.

The Explainer used to spend model calls on this: a turn asking for lookups (6 of 30 turns on the first local runs did only
that), an independent "blind" model reading of the answer type, and a clarity-check model reading of what the spec lost or
added. Three things in a question can be read without a model, in a way that does not depend on the kind of shop:

  names    the phrases of the question that are values of the shop's own data (item, group, customer, unit, channel...).
           No word list: the shop's values are the dictionary, so a new kind of shop brings its own.
  type, k  the answer type from Vietnamese question grammar ("bao nhiêu %", "ngày nào", "mặt hàng nào", "có ... không",
           "top 3"). Grammar words, not shop words. On the 2,930 answerable training questions: 95.2% right overall and
           98-100% right whenever a specific rule fires (only the default "number" is weaker, 93.7%); k right on 136 of 142.
  periods  calendar phrases the rulebook defines (agents/periods.py), listed for the model.

`sure` marks a reading precise enough to override the model; the default reading never does.
Telex typing left in a word ("naof", "nhaats", "khoong") is undone first (from zalo-challenge `norm_q`).
"""
from __future__ import annotations

import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from datamate.agents.periods import BALANCE, DAY_LAST, DAY_THIS, MONTH, MONTH_LAST, MONTH_THIS, QUARTER, WEEK_LAST, WEEK_THIS, _implied_year, _month_bounds
from datamate.text import fold_text

# ---------------------------------------------------------------------------------------------- normalising
FINALS = re.compile(r"(?:[aeiouy]|c|ch|m|n|ng|nh|p|t)$")


def _telex(tok: str) -> str:
    if not tok.isalpha() or len(tok) < 2:
        return tok
    t = re.sub(r"dd", "d", tok)
    t = re.sub(r"(aa|ee|oo)", lambda m: m.group(1)[0], t)
    t = re.sub(r"(a|o|u)w", r"\1", t)
    if t[-1] in "sfrxj" and FINALS.search(t[:-1]) and (t != tok or len(t) <= 5):
        t = t[:-1]
    return t


def norm(text: str) -> str:
    """Folded (no accents, lower case, punctuation as spaces), telex slips undone."""
    return " ".join(_telex(w) for w in fold_text(text or "").split())


# Question grammar and time phrases: removed before names are looked up, so "bao nhiêu" does not match the unit "bao",
# "phần trăm" the unit "phần" or "từ tháng" the customer "Từ Thắng Hải" (all three seen on train questions).
GRAMMAR = re.compile(
    r"\b(?:bao nhieu|bao nhiu|bnhiu|bn|phan tram|ti le|ty le|ti trong|ty trong|la gi|la bao nhieu|cho hoi|xem giup|coi giup|"
    r"nhieu nhat|it nhat|cao nhat|thap nhat|lon nhat|nho nhat|tong cong|tat ca|cua hang|so voi|cung ky|tang hay giam|"
    r"co phai|hay khong|phai khong|dung khong)\b")
TIME = re.compile(
    r"\b(?:tu|den|toi|qua|truoc|sau|trong|cuoi|dau|het)? ?(?:ngay|thang|quy|tuan|nam|hom|bua|th|t|q)\s*\d{1,2}(?:\s*[/-]\s*\d{1,4})*\b|"
    r"\b\d{1,2}\s*/\s*\d{1,2}(?:\s*/\s*\d{2,4})?\b|"
    r"\b(?:tu|den|toi|qua) (?:ngay|thang|quy|tuan|nam|hom|dau|cuoi|het)\b|"
    r"\b(?:hom|bua|tuan|thang|nam|quy) (?:nay|qua|truoc|roi|sau|ngoai|kia)\b|\b\d+ ngay\b|\bhnay\b|\bhqua\b")


# ---------------------------------------------------------------------------------------------- names
@dataclass
class Name:
    phrase: str                       # as folded in the question
    how: str                          # "exact" | "inside one value"
    hits: list[tuple[str, str, str]]  # (table, column, value as stored)

    @property
    def value(self) -> str:
        return self.hits[0][2]

    @property
    def words(self) -> int:
        return len(self.phrase.split())


_INDEX: dict[int, dict] = {}
_LOCK = threading.Lock()
MAX_DISTINCT = 20000


def name_index(tables: dict) -> dict:
    """folded value -> {(table, column, value)} over the text columns of a shop (built once per shop, 1-15 s)."""
    key = id(tables)
    with _LOCK:
        if key in _INDEX:
            return _INDEX[key]
        idx: dict = defaultdict(set)
        for t, df in tables.items():
            for c in df.columns:
                s = df[c]
                if s.dtype.kind in "iufbM" or str(s.dtype).startswith("datetime") or c in ("source_sheet", "status_std"):
                    continue
                u = s.dropna().astype(str).unique()
                if len(u) > MAX_DISTINCT:
                    continue
                for v in u:
                    f = norm(v)
                    if len(f) >= 3 and re.search(r"[a-z]", f):
                        idx[f].add((t, str(c), v))
        _INDEX[key] = dict(idx)
        return _INDEX[key]


def find_names(question: str, tables: dict, max_n: int = 8) -> list[Name]:
    """Longest phrases of the question (grammar and time phrases removed) that are a whole data value, or a run of >= 2
    words covering >= half of exactly one data value ("mì omachi" -> "Mì Omachi sườn hầm")."""
    idx = name_index(tables)
    text = TIME.sub(" | ", GRAMMAR.sub(" | ", norm(question)))
    out: list[Name] = []
    for chunk in text.split("|"):
        toks = chunk.split()
        taken = [False] * len(toks)
        for n in range(min(max_n, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                if any(taken[i:i + n]):
                    continue
                ph = " ".join(toks[i:i + n])
                hits, how = idx.get(ph), "exact"
                if not hits and n >= 2:
                    inside = [f for f in idx if f" {ph} " in f" {f} " and n >= 0.5 * len(f.split())]
                    if len({v for f in inside for (_, _, v) in idx[f]}) == 1:
                        hits, how = idx[inside[0]], "inside one value"
                if hits:
                    taken[i:i + n] = [True] * n
                    out.append(Name(ph, how, sorted(hits)))
    return out


# ---------------------------------------------------------------------------------------------- type and k
NUM_WORDS = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}
OBJ = r"(?:mat hang|hang|san pham|sp|mon|thuoc|khach(?: hang)?|nhan vien|nv|nha cung cap|ncc|kenh|nhom|loai|size|xe|tho|lo|ngay|thang)"
PERCENT = re.compile(r"%|phan tram|\bti le\b|\bty le\b|\bti trong\b|\bty trong\b|chiem (?:bao nhieu|bn|bnhiu|may|nhieu)")
DATE = re.compile(r"\b(?:ngay|hom|bua|thang|tuan|nam|quy) nao\b|\bkhi nao\b|\bluc nao\b|\bhan (?:su dung|dung) (?:nao|gan nhat|som nhat)")
BOOL = re.compile(r"\b(?:co|da|duoc|vuot|cao hon|thap hon|nhieu hon|it hon|tang|giam)\b.*\b(?:khong|ko|k|hong)\s*$|"
                  r"\bphai khong\b|\bdung khong\b|\bhay khong\b|\bhay chua\b|\bchua\s*$")
WHICH = re.compile(r"\b(?:mat hang|hang|san pham|sp|mon|thuoc|khach(?: hang)?|nhan vien|nv|nha cung cap|ncc|kenh|nhom|loai|size|xe|tho|ca|lo)"
                   r"\b[^?]{0,40}?\bnao\b|\bai\b")
WHAT = re.compile(r"\bla (?:nhung )?(?:gi|mon gi|hang gi|ai|nhung (?:mon|hang|mat hang|san pham|sp) nao)\b|\bnhat\b[^?]{0,30}\bla gi\b|\btop ?\d")
TOPK = re.compile(r"\btop ?(\d{1,2})\b|\b(\d{1,2}|" + "|".join(NUM_WORDS) + r") " + OBJ + r"\b")
LIST = re.compile(r"\bnhung\b|\bcac\b.*\bnao\b|\bliet ke\b|\bke ten\b|\bdanh sach\b|\btheo thu tu\b|\blan luot\b|\btung\b.*\bnao\b")
TABLE = re.compile(r"\bkem\b|\bva so (?:no|tien|luong)\b|\bva (?:doanh thu|so tien|tien no|so no)\b|\bvoi so\b|"
                   r"\bmoi (?:khach|mat hang|nguoi)\b.*\b(?:bao nhieu|bn)\b")
COUNT = re.compile(r"\b(?:bao nhieu|may|bn|bnhiu|nhieu) (?:mat hang|hang|san pham|sp|mon|loai|khach|nguoi|don|lan|ngay|ly|ca|xe|phieu)\b")
LIST_NUMBERS = re.compile(r"\b(?:tung thang|moi thang|tung ngay|lan luot|tung quy)\b")


@dataclass
class Reading:
    answer_type: str
    k: int | None
    rule: str
    sure: bool                         # a specific rule fired (98-100% precise on train); the default never overrides


def read_type(question: str, names: list[Name] | None = None) -> Reading:
    """Answer type and k from the question's grammar. Item names of >= 2 words are removed first ('Otrivin 0,1% xịt mũi')."""
    s = norm(question)
    for n in sorted((n for n in names or [] if n.words >= 2), key=lambda n: -len(n.phrase)):
        s = s.replace(n.phrase, " ")
    s = re.sub(r"\s+", " ", s).strip()
    k = None
    m = TOPK.search(s)
    if m:
        v = m.group(1) or m.group(2)
        k = int(v) if v.isdigit() else NUM_WORDS.get(v)
    if PERCENT.search(s):
        return Reading("percent", k, "percent words", True)
    if DATE.search(s):
        return Reading("date", k, "which day / month", True)
    if LIST_NUMBERS.search(s):
        return Reading("list", k, "one number per period", True)
    if (WHICH.search(s) or WHAT.search(s)) and not COUNT.search(s):
        if TABLE.search(s):
            return Reading("table", k, "which, with amounts", True)
        if (k and k > 1) or LIST.search(s):
            return Reading("list", k, "which, several", True)
        return Reading("string", k, "which, one", True)
    if BOOL.search(s):
        return Reading("boolean", k, "yes / no", True)
    return Reading("number", k, "default", False)


# ---------------------------------------------------------------------------------------------- periods
@dataclass
class PeriodMention:
    label: str
    start: date
    end: date


def mentioned_periods(question: str, as_of: str) -> list[PeriodMention]:
    """Calendar phrases in reading order, with the rulebook's dates (§2: a month without a year is this year's unless it
    lies after today). Windows, balances and comparisons are not resolved here: they stay the model's (and repair_spec's)."""
    q = norm(question)
    today = date.fromisoformat(as_of[:10])
    if BALANCE.search(q):          # a balance "cuối ngày hôm nay" runs from the start of the data: not named as a one-day period
        q = DAY_THIS.sub(" ", DAY_LAST.sub(" ", q))
    monday = today - __import__("datetime").timedelta(days=today.weekday())
    found: list[tuple[int, PeriodMention]] = []
    one_day = __import__("datetime").timedelta(days=1)
    for rx, label, span in ((DAY_LAST, "yesterday", (today - one_day, today - one_day)), (DAY_THIS, "today", (today, today)),
                            (WEEK_LAST, "last week", (monday - 7 * one_day, monday - one_day)), (WEEK_THIS, "this week", (monday, today)),
                            (MONTH_THIS, "this month", (today.replace(day=1), today))):
        m = rx.search(q)
        if m:
            found.append((m.start(), PeriodMention(label, *span)))
    m = MONTH_LAST.search(q)
    if m:
        prev_last = today.replace(day=1) - one_day
        found.append((m.start(), PeriodMention("last month", prev_last.replace(day=1), prev_last)))
    rest = MONTH_LAST.sub(lambda x: " " * len(x.group(0)), MONTH_THIS.sub(lambda x: " " * len(x.group(0)), q))
    for m in MONTH.finditer(rest):
        mo, yr = int(m.group(1)), m.group(2)
        if 1 <= mo <= 12:
            y = int(yr) if yr else _implied_year(today, month=mo)
            found.append((m.start(), PeriodMention(f"month {mo}/{y}", *_month_bounds(y, mo))))
    for m in QUARTER.finditer(rest):
        qn, yr = int(m.group(1)), m.group(2)
        y = int(yr) if yr else _implied_year(today, quarter=qn)
        found.append((m.start(), PeriodMention(f"quarter {qn}/{y}", date(y, (qn - 1) * 3 + 1, 1), _month_bounds(y, qn * 3)[1])))
    found.sort(key=lambda x: x[0])
    return [p for _, p in found]


# ---------------------------------------------------------------------------------------------- all of it
@dataclass
class QuestionFacts:
    names: list[Name] = field(default_factory=list)
    reading: Reading | None = None
    periods: list[PeriodMention] = field(default_factory=list)

    def prompt(self) -> str:
        """The section the spec call reads: facts the code is sure of, to use instead of guessing."""
        lines = ["DATA MATCHES (phrases of the question that are values in the shop's data; use these exact spellings):"]
        for n in self.names:
            where = "; ".join(f"`{t}`.`{c}` = {v!r}" for t, c, v in n.hits[:3]) + (" ..." if len(n.hits) > 3 else "")
            lines.append(f"- \"{n.phrase}\" ({n.how}) -> {where}")
        if not self.names:
            lines.append("- (none: the question names no value of the data)")
        lines.append("PERIODS NAMED IN THE QUESTION (rulebook dates; a comparison or a window may need more or other periods):")
        lines += [f"- {p.label}: {p.start} .. {p.end}" for p in self.periods] or ["- (no calendar phrase)"]
        r = self.reading
        if r is not None:
            lines.append(f"ANSWER TYPE READ FROM THE WORDING: {r.answer_type}" + (f", k = {r.k}" if r.k else "")
                         + (f" (rule: {r.rule}; reliable)" if r.sure else " (default reading, decide yourself)"))
        return "\n".join(lines)

    def as_log(self) -> dict:
        return {"names": [{"phrase": n.phrase, "how": n.how, "hits": n.hits[:3]} for n in self.names],
                "reading": self.reading.__dict__ if self.reading else None,
                "periods": [{"label": p.label, "start": str(p.start), "end": str(p.end)} for p in self.periods]}


def read_question(question: str, as_of: str, tables: dict) -> QuestionFacts:
    names = find_names(question, tables)
    return QuestionFacts(names, read_type(question, names), mentioned_periods(question, as_of))
