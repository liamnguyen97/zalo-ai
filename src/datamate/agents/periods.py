"""PERIODS IN CODE: the calendar phrases the rulebook (§2) defines exactly, computed from the question and `as_of_date`.

The Explainer is a 9B model: on dev300 "tuần trước / tuần rồi" came out as Tuesday-to-Monday or an 8-day window in 3 of 13
week questions (62% correct against 86% elsewhere), "tháng 6/2025" as 06-01..06-02, and "cùng kỳ tháng trước" as the same dates
as the main period. These phrases mean the same in every shop, so the dates are computed here and the model's periods are
replaced only when this module is sure: exactly one calendar phrase in the question and exactly one period in the spec.
Anything else (comparisons, balances "đến hết ngày X", N-day windows, Tết) is left to the model and the other repairs.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from datamate.text import fold_text

WEEK_LAST = re.compile(r"\btuan (?:truoc|roi|qua|trc|vua roi|vua qua)\b")
WEEK_THIS = re.compile(r"\btuan nay\b")
MONTH_LAST = re.compile(r"\bthang (?:truoc|roi|qua|trc|vua roi)\b")
MONTH_THIS = re.compile(r"\bthang nay\b")
MONTH = re.compile(r"\b(?:thang|th|t) ?(\d{1,2}) ?(?:/|-|nam)? ?(20\d{2})?(?!\d)")
QUARTER = re.compile(r"\b(?:quy|q) ?([1-4]) ?(?:/|-|nam)? ?(20\d{2})?(?!\d)")
# questions whose period is not "the whole named span": balances, windows, comparisons
SKIP = re.compile(r"\bden het\b|\btinh den\b|\btu dau\b|\bso voi\b|\bcung ky\b|\bnhung\b|\bden thang\b|\btu thang\b|\bden ngay\b|"
                  r"\btu ngay\b|\bke tu\b|\bngay gan nhat\b|\b\d+ ngay\b|\btet\b|\blien truoc\b|\btruoc do\b")


def _month_bounds(y: int, m: int) -> tuple[date, date]:
    first = date(y, m, 1)
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return first, nxt - timedelta(days=1)


def _implied_year(today: date, month: int | None = None, quarter: int | None = None) -> int:
    """§2: a month / quarter written without a year is this year's unless it lies after today, then last year's."""
    start_month = month if month else (quarter - 1) * 3 + 1
    return today.year if date(today.year, start_month, 1) <= today else today.year - 1


def calendar_period(question: str, as_of: str) -> tuple[date, date, str] | None:
    """(start, end, phrase) the question names with one calendar phrase, or None when there is none, more than one, or the
    question is a comparison / balance / window (see SKIP)."""
    q = fold_text(question or "")
    if SKIP.search(q):
        return None
    today = date.fromisoformat(as_of[:10])
    monday = today - timedelta(days=today.weekday())
    found: list[tuple[date, date, str]] = []
    if WEEK_LAST.search(q):
        found.append((monday - timedelta(days=7), monday - timedelta(days=1), "last week"))
    if WEEK_THIS.search(q):
        found.append((monday, today, "this week"))
    if MONTH_LAST.search(q):
        prev_last = today.replace(day=1) - timedelta(days=1)
        found.append((prev_last.replace(day=1), prev_last, "last month"))
    if MONTH_THIS.search(q):
        found.append((today.replace(day=1), today, "this month"))
    rest = MONTH_LAST.sub(" ", MONTH_THIS.sub(" ", q))
    for m in MONTH.finditer(rest):
        mo, yr = int(m.group(1)), m.group(2)
        if 1 <= mo <= 12:
            y = int(yr) if yr else _implied_year(today, month=mo)
            found.append((*_month_bounds(y, mo), f"month {mo}/{y}"))
    for m in QUARTER.finditer(rest):
        qn, yr = int(m.group(1)), m.group(2)
        y = int(yr) if yr else _implied_year(today, quarter=qn)
        first = date(y, (qn - 1) * 3 + 1, 1)
        found.append((first, _month_bounds(y, qn * 3)[1], f"quarter {qn}/{y}"))
    spans = {(a, b) for a, b, _ in found}
    return found[0] if len(spans) == 1 else None


def fix_calendar_period(periods: list, question: str | None, as_of: str, notes: list) -> None:
    """Replace the only period of a spec by the rulebook's dates when the question names exactly one calendar phrase."""
    if len(periods) != 1 or not question:
        return
    got = calendar_period(question, as_of)
    if got is None:
        return
    start, end, phrase = got
    p = periods[0]
    if str(p.get("start"))[:10] == str(start) and str(p.get("end"))[:10] == str(end):
        return
    notes.append(f"harness: period {p.get('label')!r} {p.get('start')}..{p.get('end')} -> {start}..{end} (rulebook §2: {phrase})")
    p["start"], p["end"], p["role"] = str(start), str(end), "main"
