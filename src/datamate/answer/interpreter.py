"""INTERPRETER stage: turn the runner's raw Python value into a typed answer, and judge whether to give it at all.

Typing: the spec's expected `answer_type` decides number vs percent and resolves ambiguous shapes; the value's own
shape decides the rest. Coercions fix format slips seen in the runs: a Timestamp or "2026-04-29 00:00:00" for a date,
`{"type": "percent", "answer": -39.96}` wrappers, single-element lists, a month written in words ("Tháng 3/2026" ->
'2026-03', rulebook §9), a count returned as the list of what was counted (`as_count`).

Judging: scoring is +1 / -1 / 0 (abstain), so an answer is worth giving only when it is more likely right than wrong.
`issues` lists signs of a failed computation that do not depend on any business's data layout:
    empty       empty list/table/string, NaN, inf
    type        the value cannot have the expected type (e.g. a list for a "how many" question)
    walk_in     a walk-in customer label ranked as a customer (rulebook §3: walk-ins are not customers)
    zero        a 0 for a number/percent question: no data is "unanswerable", not 0 (rulebook §7), and a 0 is far more
                often a filter that matched nothing (4 of 2,462 numeric training answers are 0)
    sentinel    a 999999 placeholder from the data (unknown stock), not a measured value
    risky_type  a table without named columns, with ragged or no rows (a well-formed table is accepted and left to the judge)
    table_rows  a top-k table whose row count is not k
    outside_period  the top day / month of an amount lies outside every period the spec asks about
The pipeline abstains when any issue is found (DATAMATE_GATE=1, the default).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from datamate.text import fold

TYPES = ("number", "percent", "string", "date", "boolean", "list", "table")
YES = {"true", "có", "co", "yes", "đúng", "dung", "phải", "phai"}
NO = {"false", "không", "khong", "no", "sai"}
DATE_STR = re.compile(r"\s*(\d{4})(?:\.0)?-(\d{1,2})(?:\.0)?(?:-(\d{1,2})(?:\.0)?)?(?:[ T]00:00(?::00(?:\.0+)?)?)?\s*")
WALK_IN = re.compile(r"^\s*(?:kh[áa]ch\s*)?(?:l[ẻe]|v[ãa]ng\s*lai)\s*$|^\s*kl\s*$|^\s*walk[\s-]*in\s*$|^\s*kh[áa]ch\s*$", re.I)

AMOUNTS = ("quantity", "money", "orders", "count")   # spec measures whose top day / month must lie in the asked period


@dataclass
class Interpretation:
    kind: str
    raw: Any
    issues: list[str] = field(default_factory=list)


def _is_rows(raw: Any) -> bool:
    return isinstance(raw, (list, tuple)) and len(raw) > 0 and all(isinstance(r, (list, tuple, dict)) for r in raw)


def _heuristic(raw: Any) -> str | None:
    if isinstance(raw, (bool, np.bool_)):
        return "boolean"
    if isinstance(raw, (pd.Timestamp, datetime, date)):
        return "date"
    if isinstance(raw, pd.DataFrame) or (isinstance(raw, dict) and {"columns", "rows"} <= raw.keys()) or _is_rows(raw):
        return "table"
    if isinstance(raw, dict) and "type" in raw:  # e.g. {"type": "unanswerable", ...}; formatter handles it
        return "string"
    if isinstance(raw, (list, tuple, set, pd.Series)):
        return "list"
    if isinstance(raw, str):
        return "date" if DATE_STR.fullmatch(raw) else "string"
    return None  # a number: number vs percent is a semantic call


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _unwrap(raw: Any) -> tuple[Any, str | None]:
    """{"type": "percent", "answer": x} -> (x, "percent"); a 1-row Series/1-element list stays for later."""
    if isinstance(raw, dict) and "answer" in raw and str(raw.get("type", "")).lower() in TYPES:
        return raw["answer"], str(raw["type"]).lower()
    return raw, None


def _empty(raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, float) and not math.isfinite(raw):
        return True
    if isinstance(raw, str):
        return not raw.strip()
    if isinstance(raw, (list, tuple, set, pd.Series)):
        return len(raw) == 0
    if isinstance(raw, pd.DataFrame):
        return raw.empty
    if isinstance(raw, dict) and "rows" in raw:
        return not raw["rows"]
    return False


def _names(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [x for x in raw if isinstance(x, str)] + [r[0] for r in raw if isinstance(r, (list, tuple)) and r and isinstance(r[0], str)]
    if isinstance(raw, dict) and "rows" in raw:
        return [r[0] for r in raw["rows"] if r and isinstance(r[0], str)]
    if isinstance(raw, pd.DataFrame) and len(raw.columns):
        return [x for x in raw.iloc[:, 0].tolist() if isinstance(x, str)]
    return []


def is_unanswerable(raw: Any) -> bool:
    if isinstance(raw, str):
        return raw.strip().upper().startswith("UNANSWERABLE")
    return isinstance(raw, dict) and str(raw.get("type", "")).lower() == "unanswerable"


def as_count(raw: Any) -> Any:
    """A sample for a spec that counts objects, as one number. The code may list the objects (the rewrite said "liệt kê
    tên") or return {'count': 2, 'item_names': [...]}: holdout6 r7 "mấy mặt hàng ế" got 2, a dict and a list of names
    from three samples, which then disagreed. Anything else is returned unchanged. Applied to each sample before the vote."""
    if isinstance(raw, (list, tuple, set)) and raw and all(isinstance(r, str) for r in raw):
        return len(set(raw))
    if isinstance(raw, dict):
        nums = [v for v in raw.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(nums) == 1:
            return nums[0]
    return raw


MONTH = re.compile(r"(?:thang|th|t)\s*0?(\d{1,2})(?:\s*(?:/|-|nam)\s*(\d{4}|\d{2}))?|0?(\d{1,2})\s*[/-]\s*(\d{4})")


def month_answer(raw: Any, periods) -> str | None:
    """'YYYY-MM' when the answer names one month in words ("Tháng 3/2026", "tháng 5 năm 2025", "T2", "05/2025"), else None.

    Rulebook §9 wants a month answer as a date 'YYYY-MM' ("tháng 9/2025" is listed as wrong); public run: 6 answers were
    the text "Tháng 2" / "tháng 7/2025" typed string. A missing year comes from the spec period that covers that month;
    when no single year fits, None."""
    if not isinstance(raw, str):
        return None
    m = MONTH.fullmatch(fold(raw).strip())
    if not m:
        return None
    month, year = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
    month = int(month)
    if not 1 <= month <= 12:
        return None
    if year:
        year = int(year) + (2000 if len(year) == 2 else 0)
    else:
        years = set()
        for p in periods or []:                      # every year whose month `month` lies in a period's span
            a, b = str(p.get("start", ""))[:7], str(p.get("end", "") or p.get("start", ""))[:7]
            if re.fullmatch(r"\d{4}-\d{2}", a) and re.fullmatch(r"\d{4}-\d{2}", b):
                years |= {y for y in range(int(a[:4]), int(b[:4]) + 1) if a <= f"{y:04d}-{month:02d}" <= b}
        if len(years) != 1:
            return None
        year = years.pop()
    return f"{year:04d}-{month:02d}"


def in_periods(day: str, periods) -> bool:
    """A 'YYYY-MM-DD' or 'YYYY-MM' answer falls in one of the spec periods (True when there are none to check)."""
    spans = [(str(p.get("start", ""))[:10], str(p.get("end", ""))[:10]) for p in periods or [] if p.get("start") and p.get("end")]
    if not spans or not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", day):
        return True
    return any(a[:len(day)] <= day <= b[:len(day)] for a, b in spans)


def _table_issues(raw: Any, spec) -> list[str]:
    """A table is judged by its structure instead of refused outright (0 of 6 tables were answered on dev300 a1): named
    columns, every row as wide as the header, at least one row, and k rows when the question asks for the top k."""
    if isinstance(raw, pd.DataFrame):
        cols, rows = list(raw.columns), raw.values.tolist()
    elif isinstance(raw, dict) and {"columns", "rows"} <= raw.keys() and isinstance(raw["rows"], (list, tuple)):
        cols, rows = list(raw["columns"] or []), list(raw["rows"])
    else:
        return ["risky_type"]       # bare rows without column names: the columns cannot be checked
    if not cols or not rows or any(not isinstance(r, (list, tuple)) or len(r) != len(cols) for r in rows):
        return ["risky_type"]
    k = getattr(spec, "k", None)
    if k and len(rows) != k:
        return ["table_rows"]
    return []


PERIOD_LABEL = re.compile(r"^\s*(?:\d{4}-\d{1,2}(?:-\d{1,2})?|(?:th[aá]ng|t|q|qu[yý]|tu[aầ]n|ng[aà]y)\s*\d{1,2}(?:\s*[/-]\s*\d{2,4})?)\s*$", re.I)


def _list_from_pairs(raw: Any, spec) -> list | None:
    """A list answer returned as [name, amount] rows: the half the question asks for, or None when that is not clear.

    dev300 q_33e8f7506387 ("top 5 hàng bán chạy"): the first turn returned the right 5 names with their quantities, the check
    refused the table, and the Coder rebuilt the ranking by hand and got it wrong (abstain). Every list answer of the training
    gold is names only (106) or numbers only (87), never pairs. Names when the spec asks for a top-k or for names; numbers when
    the first column labels periods ("doanh thu từng tháng": ['2025-04', 123]); otherwise the check stays."""
    rows = [r for r in raw if isinstance(r, (list, tuple))] if isinstance(raw, (list, tuple)) else []
    if not rows or len(rows) != len(raw) or any(len(r) != 2 or not isinstance(r[0], str) or not _is_num(r[1]) for r in rows):
        return None
    if all(PERIOD_LABEL.match(r[0]) for r in rows):
        return [r[1] for r in rows]
    if getattr(spec, "k", None) or getattr(spec, "measure", None) == "name":
        return [r[0] for r in rows]
    return None


def interpret(raw: Any, expected: str | None = None, spec=None) -> Interpretation:
    """Type `raw` against the spec's expected answer type and list what looks like a failed computation."""
    if is_unanswerable(raw):
        return Interpretation("unanswerable", raw)
    raw, hinted = _unwrap(raw)
    expected = expected if expected in TYPES else hinted
    issues: list[str] = []

    # shape fixes that do not change meaning
    if isinstance(raw, pd.Series) and len(raw) == 1 and expected not in ("list", "table"):
        raw = raw.iloc[0]
    if isinstance(raw, (list, tuple)) and len(raw) == 1 and expected not in ("list", "table") and not isinstance(raw[0], (list, tuple, dict)):
        raw = raw[0]
    if isinstance(raw, np.generic):
        raw = raw.item()

    if _empty(raw):
        return Interpretation(expected or "string", raw, ["empty"])

    if isinstance(raw, str) and expected in ("number", "percent"):
        try:
            raw = float(raw.replace(",", "").replace("%", "").strip())
        except ValueError:
            pass
    if isinstance(raw, str) and expected == "boolean" and raw.strip().lower() in YES | NO:
        raw = raw.strip().lower() in YES

    shape = _heuristic(raw)
    if shape is None:  # numeric
        kind = expected if expected in ("number", "percent") else "number"
        if expected is not None and expected not in ("number", "percent"):
            issues.append("type")
    elif expected is None or shape == expected:
        kind = shape
    elif expected == "string" and shape == "date":  # "tháng nào ...": a month/day value is a date, never a name
        kind = "date"
    elif expected == "date" and shape == "string":
        try:
            raw, kind = pd.to_datetime(raw, dayfirst=True).date().isoformat(), "date"
        except (ValueError, TypeError):
            kind = shape
            issues.append("type")
    elif expected == "list" and shape == "table" and _is_rows(raw) and all(len(r) == 1 for r in raw if isinstance(r, (list, tuple))):
        raw, kind = [r[0] for r in raw], "list"
    elif expected == "list" and shape == "table" and (half := _list_from_pairs(raw, spec)) is not None:
        raw, kind = half, "list"
    else:  # e.g. a list for a "how many" question, a table for a list of numbers: the model answered something else
        kind = shape
        issues.append("type")

    if kind in ("string", "list", "table") and any(WALK_IN.match(n) for n in _names(raw)):
        issues.append("walk_in")
    if kind in ("number", "percent") and _is_num(raw) and float(raw) == 0:
        issues.append("zero")
    if kind in ("number", "percent") and _is_num(raw) and re.fullmatch(r"9{6,}", str(abs(int(float(raw))))):
        issues.append("sentinel")      # 999999999 is a placeholder in the data (unknown stock), not a measured value
    if kind == "table":
        issues += _table_issues(raw, spec)
    periods = getattr(spec, "periods", None)
    month = month_answer(raw, periods) if kind in ("string", "date") else None
    if month:
        kind, raw = "date", month
        issues = [x for x in issues if x != "type"]
    if (kind == "date" and isinstance(raw, str) and getattr(spec, "measure", None) in AMOUNTS
            and not in_periods(raw, periods)):
        # "T9/2025 ngày nào doanh thu cao nhất?" answered 2025-08-12 (public run): the top day / month of an amount
        # lies inside the asked period. Expiry / due dates are measure 'date' and are not checked.
        issues.append("outside_period")
    return Interpretation(kind, raw, issues)
