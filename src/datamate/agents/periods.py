"""PERIODS IN CODE: the calendar phrases the rulebook (§2) defines exactly, computed from the question and `as_of_date`.

The Explainer is a 9B model: on dev300 "tuần trước / tuần rồi" came out as Tuesday-to-Monday or an 8-day window in 3 of 13
week questions (62% correct against 86% elsewhere), "tháng 6/2025" as 06-01..06-02, and "cùng kỳ tháng trước" as the same dates
as the main period. These phrases mean the same in every shop, so the dates are computed here and the model's periods are
replaced only when this module is sure: exactly one calendar phrase in the question and exactly one period in the spec.
Anything else (comparisons, balances "đến hết ngày X", N-day windows, Tết) is left to the model and the other repairs.

`scan` reads every calendar phrase of a (folded) question once; `calendar_period` (forcing a period) and
`agents/question.mentioned_periods` (the PERIODS section of the spec prompt, the coverage check) both use it, so they cannot
disagree about what the question says. What it reads:
  day / week / month   hôm qua, hôm nay, tuần trước, tuần này, tháng trước, tháng này
  month                tháng 3, th3/2026, T3-26, tháng 3 năm 2026, 03/2026, 3/26, "tháng 5, 6, 7/2025" (the year ends the list)
  quarter              quý 4, Q4/2025, q1/26
  year                 năm 2025 (a past year), năm ngoái (the only phrase of the question)
  to date              từ đầu năm / tháng / quý [tới nay | tới chừ | đến hết hôm nay]  (§2: from the 1st to the end of today)
A year written with 2 digits ("3/25") is read only inside as_of.year-3 .. as_of.year, never before "ngày", never in a full date
(15/3/25) and, without the prefix tháng / th / t, only after "/" and only when it cannot be a day/month (the second number > 12).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from datamate.text import norm

DAY_LAST = re.compile(r"\b(?:hom qua|bua qua|hqua|ngay hom qua)\b")       # §2 "hôm qua" = the day before as_of_date
DAY_THIS = re.compile(r"\b(?:hom nay|bua nay|hnay|ngay hom nay)\b")
WEEK_LAST = re.compile(r"\btuan (?:truoc|roi|qua|trc|vua roi|vua qua)\b")
WEEK_THIS = re.compile(r"\btuan nay\b")
MONTH_LAST = re.compile(r"\bthang (?:truoc|roi|qua|trc|vua roi)\b")       # "thangs" (a telex slip) is undone by text.norm
MONTH_THIS = re.compile(r"\bthang nay\b")
# "tháng sau cao hơn tháng trước": the pair of a chain of months, not the month before this one
CHAIN = re.compile(r"\bthang sau\b|\bthang ke tiep\b|\blien tuc\b|\bmoi thang\b|\btung thang\b")
_PFX = r"(?:thang|th|t)"
# a year after the month: 4 digits (a separator or "năm" may stand between), or 2 digits after "/" or "-" and not before "ngày"
MONTH = re.compile(rf"\b{_PFX} ?(\d{{1,2}})(?: ?(?:/|-|nam)? ?(20\d{{2}})| ?[/-] ?(\d{{2}})(?![\d/])(?! ?ngay\b))?(?!\d)")
MONTH_LIST = re.compile(rf"\b{_PFX} ?(\d{{1,2}}(?: (?:va |hoac )?\d{{1,2}})+)(?: ?[/-] ?(20\d{{2}}|\d{{2}})| nam (20\d{{2}}))(?![\d/])")
QUARTER = re.compile(r"\b(?:quy|q) ?([1-4])(?: ?(?:/|-|nam)? ?(20\d{2})| ?[/-] ?(\d{2})(?![\d/])(?! ?ngay\b))?(?!\d)")
# no "tháng" in front: 03/2026, 6-2025 (a 4-digit year) and 3/26 (2 digits, > 12: it cannot be a day/month). Not part of a longer date
MONTH_YEAR = re.compile(r"(?<![\d/.\-])(?<!quy )(?<!qui )(0?[1-9]|1[0-2]) ?[/-] ?(20\d{2})(?![\d/\-])")
MONTH_YY = re.compile(r"(?<![\d/.\-])(?<!quy )(?<!qui )([1-9]|1[0-2])/(\d{2})(?![\d/\-])(?! ?ngay\b)")
# "6 năm 2025" is a month (or a quarter, a day) of that year, not the year
YEAR = re.compile(r"(?<!\d )\bnam (20\d{2})\b")
YEAR_LAST = re.compile(r"\bnam (?:ngoai|truoc|roi)\b")
# not "từ đầu tháng 5", "từ đầu năm 2025" (a number follows) and not "từ đầu năm đến hết tháng 5" (it ends somewhere else)
TO_DATE = re.compile(r"\btu dau (nam|thang|quy)(?: nay)?(?: (?:toi|den|cho den)(?: het)? (?:hom nay|bua nay|hnay|nay|chu|bay gio|hien tai)\b)?"
                     r"(?! ?(?:nam )?\d)(?! (?:toi|den|cho den)\b)")
# a question that compares: its year / to-date phrase may be one side of the comparison, so it is not "the" period
VERSUS = re.compile(r"\bso voi\b|\bso vs\b|\bvs\b|\bso sanh\b|\bcung ky\b|\bcung (?:thang|quy|nam|tuan|ngay)\b|\bhon\b|\bkem\b|\bchenh\b|\btang\b|\bgiam\b")
BALANCE = re.compile(r"\bcuoi ngay\b|\bhet hom nay\b|\bhet ngay\b|\bton\b|\bcon no\b|\bdang no\b|\bcon lai\b")
YEAR_WINDOW = 3          # a 2-digit year is read as 20yy only when it is within this many years before as_of's year
# questions whose period is not "the whole named span": balances, windows, comparisons
SKIP_COMPARE = re.compile(r"\bso voi\b|\bcung ky\b|\bnhung\b")
SKIP_BOUND = re.compile(r"\bden het\b|\btinh den\b|\btu dau\b|\bden thang\b|\btu thang\b|\bden ngay\b|"
                        r"\btu ngay\b|\bke tu\b|\bngay gan nhat\b|\b\d+ ngay\b|\btet\b|\blien truoc\b|\btruoc do\b|"
                        # balances at the end of a day (stock, debt): from the first day of the data to that day, not that day only
                        r"\bcuoi ngay\b|\bhet hom nay\b|\bhet ngay\b|\bton\b|\bcon no\b|\bdang no\b|\bcon lai\b")
SKIP = re.compile(SKIP_COMPARE.pattern + "|" + SKIP_BOUND.pattern)
# where a spec period may be wider than the phrase it covers ("đến hết tháng 5", "từ tháng 3 đến tháng 5", "7 ngày"): `covered` is not strict
RANGE_OR_BALANCE = re.compile(SKIP_BOUND.pattern + r"|\btu\b.*\b(?:den|toi)\b")


def _month_bounds(y: int, m: int) -> tuple[date, date]:
    first = date(y, m, 1)
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return first, nxt - timedelta(days=1)


def _implied_year(today: date, month: int | None = None, quarter: int | None = None) -> int:
    """§2: a month / quarter written without a year is this year's unless it lies after today, then last year's."""
    start_month = month if month else (quarter - 1) * 3 + 1
    return today.year if date(today.year, start_month, 1) <= today else today.year - 1


def _year2(yy: str, today: date) -> int | None:
    """A year written with 2 digits ("25" -> 2025) when it is a plausible one for this question, else None."""
    y = 2000 + int(yy)
    return y if today.year - YEAR_WINDOW <= y <= today.year else None


@dataclass(frozen=True)
class Hit:
    pos: int
    label: str
    start: date
    end: date
    kind: str            # day | week | month | quarter | year | to_date


def _blank(q: str, m: re.Match) -> str:
    return q[:m.start()] + " " * (m.end() - m.start()) + q[m.end():]


def _month_hit(m: re.Match, mo: int, y: int | None, today: date) -> Hit | None:
    if not 1 <= mo <= 12:
        return None
    y = y if y is not None else _implied_year(today, month=mo)
    return Hit(m.start(), f"month {mo}/{y}", *_month_bounds(y, mo), "month")


def scan(q: str, today: date) -> tuple[list[Hit], str]:
    """(calendar phrases of the folded question `q` in reading order, `q` with the to-date phrases taken out). The to-date phrase
    ("từ đầu năm tới hết hôm nay") is taken out so that its "hết hôm nay" is not read as a balance or as a second period."""
    hits: list[Hit] = []
    rest = q
    for m in TO_DATE.finditer(q):
        what = m.group(1)
        if what == "nam":
            lo = date(today.year, 1, 1)
        elif what == "thang":
            lo = today.replace(day=1)
        else:
            lo = date(today.year, 3 * ((today.month - 1) // 3) + 1, 1)
        hits.append(Hit(m.start(), f"{what} to date", lo, today, "to_date"))
        rest = _blank(rest, m)
    work = rest
    monday = today - timedelta(days=today.weekday())
    one_day = timedelta(days=1)
    for rx, label, span in ((DAY_LAST, "yesterday", (today - one_day, today - one_day)), (DAY_THIS, "today", (today, today)),
                            (WEEK_LAST, "last week", (monday - 7 * one_day, monday - one_day)), (WEEK_THIS, "this week", (monday, today)),
                            (MONTH_THIS, "this month", (today.replace(day=1), today))):
        m = rx.search(work)
        if m:
            hits.append(Hit(m.start(), label, *span, "day" if label in ("yesterday", "today") else "week" if "week" in label else "month"))
    m = MONTH_LAST.search(work)
    if m and not CHAIN.search(work):
        prev_last = today.replace(day=1) - one_day
        hits.append(Hit(m.start(), "last month", prev_last.replace(day=1), prev_last, "month"))
    # blank the relative months so that "thang nay" / "thang truoc" are not read again below
    work = MONTH_LAST.sub(lambda x: " " * len(x.group(0)), MONTH_THIS.sub(lambda x: " " * len(x.group(0)), work))
    # "tháng 5, 6, 7/2025": one year for the whole list
    for m in list(MONTH_LIST.finditer(work)):
        months = [int(x) for x in re.findall(r"\d{1,2}", m.group(1))]
        y = int(m.group(3)) if m.group(3) else (int(m.group(2)) if len(m.group(2)) == 4 else _year2(m.group(2), today))
        if y is None or not all(1 <= x <= 12 for x in months) or len(set(months)) != len(months):
            continue
        hits += [h for mo in months if (h := _month_hit(m, mo, y, today))]
        work = _blank(work, m)
    for m in list(MONTH.finditer(work)):
        mo, y = int(m.group(1)), (int(m.group(2)) if m.group(2) else _year2(m.group(3), today) if m.group(3) else None)
        if h := _month_hit(m, mo, y, today):
            hits.append(h)
    work = MONTH.sub(lambda x: " " * len(x.group(0)) if 1 <= int(x.group(1)) <= 12 else x.group(0), work)
    for m in QUARTER.finditer(work):
        qn = int(m.group(1))
        y = int(m.group(2)) if m.group(2) else _year2(m.group(3), today) if m.group(3) else None
        y = y if y is not None else _implied_year(today, quarter=qn)
        hits.append(Hit(m.start(), f"quarter {qn}/{y}", date(y, (qn - 1) * 3 + 1, 1), _month_bounds(y, qn * 3)[1], "quarter"))
    work = QUARTER.sub(lambda x: " " * len(x.group(0)), work)
    for m in MONTH_YEAR.finditer(work):
        if h := _month_hit(m, int(m.group(1)), int(m.group(2)), today):
            hits.append(h)
    work = MONTH_YEAR.sub(lambda x: " " * len(x.group(0)), work)
    for m in MONTH_YY.finditer(work):
        y = _year2(m.group(2), today)
        if y is not None and (h := _month_hit(m, int(m.group(1)), y, today)):
            hits.append(h)
    # a whole year: only when it is the one phrase of a question that does not compare (it may be one side of a comparison)
    if not hits and not VERSUS.search(rest):
        m = YEAR.search(work)
        if m and int(m.group(1)) < today.year:       # this year's "năm 2026" is a year to date or a whole year: left to the model
            hits.append(Hit(m.start(), f"year {m.group(1)}", date(int(m.group(1)), 1, 1), date(int(m.group(1)), 12, 31), "year"))
        elif (m := YEAR_LAST.search(work)):
            hits.append(Hit(m.start(), f"year {today.year - 1}", date(today.year - 1, 1, 1), date(today.year - 1, 12, 31), "year"))
    named = [h for h in hits if h.kind in ("month", "quarter") and h.label not in ("last month", "this month")]
    if named:
        # "tháng 11/2025 hơn hay kém tháng trước": the month before THAT month, not the one before today's (kept when it is the same month)
        hits = [h for h in hits if h.label != "last month" or any((n.start, n.end) == (h.start, h.end) for n in named)]
    return sorted(hits, key=lambda h: h.pos), rest


def calendar_period(question: str, as_of: str) -> tuple[date, date, str] | None:
    """(start, end, phrase) the question names with one calendar phrase, or None when there is none, more than one, or the
    question is a comparison / balance / window (see SKIP)."""
    hits, rest = scan(norm(question), date.fromisoformat(as_of[:10]))
    if SKIP.search(rest):
        return None
    spans = {(h.start, h.end) for h in hits}
    return (hits[0].start, hits[0].end, hits[0].label) if len(spans) == 1 else None


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


def unmatched_periods(question: str, named: list, spans: list[tuple[str, str]]) -> list[tuple[object, bool]]:
    """[(mention, strict)] for the calendar phrases of the question (`named`: objects with .label .start .end) that none of `spans`
    ((start, end) ISO dates) matches: the spec stage's coverage check and the audit of the periods handed to the Coder both ask this.
    `strict`: the phrase had to be EQUALLED (see `covered`), else containing it was enough."""
    fq = norm(question)
    strict = not RANGE_OR_BALANCE.search(fq)
    ranges = [(p.start, p.end) for p in named]
    out = []
    for p in named:
        if p.label == "last month" and re.search(r"\bcung ky\b", fq) and any(a == str(p.start) and b <= str(p.end) for a, b in spans):
            continue    # "từ đầu tháng này đến hôm nay so với cùng kỳ tháng trước": the same days of last month (repair_spec cut it short)
        if not covered((p.start, p.end), spans, ranges, strict):
            out.append((p, strict))
    return out


def covered(mention: tuple[date, date], spec_spans: list[tuple[str, str]], mentions: list[tuple[date, date]], strict: bool) -> bool:
    """Is a period the question names covered by a period of the spec? Not strict: a spec period that contains it. Strict: one that
    EQUALS it, or one that is exactly the union of >= 2 contiguous named periods ("3 tháng 10, 11, 12" as one period Oct..Dec).
    A spec Q1 contains T1 but is not T1: "T1/2026 so với T12/2025" answered over a quarter was passing the containment check."""
    lo, hi = (str(mention[0]), str(mention[1]))
    for a, b in spec_spans:
        if not (a <= lo and hi <= b):
            continue
        if not strict or (a, b) == (lo, hi):
            return True
        inside = sorted((s, e) for s, e in ((str(x), str(y)) for x, y in mentions) if a <= s and e <= b)
        if len(inside) >= 2 and inside[0][0] == a and inside[-1][1] == b and all(
                date.fromisoformat(n[0]) - date.fromisoformat(p[1]) == timedelta(days=1) for p, n in zip(inside, inside[1:])):
            return True
    return False
