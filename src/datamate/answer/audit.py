"""AUDIT: checks of a turn's code and value that need no model, run beside the judge (ledger #12).

Measured on the dev300 v3.5 logs (scripts/review_counterfactual.py) as stand-ins for what the LLM review caught there:
- entity_absent: a value the question names (a spec entity) is in none of the code's string constants. F-strings and print()
  arguments do not count: they echo the name, they do not filter by it. It fired on q_0f6233b739e9 ('natri|clor' for
  'Natri clorid') and q_12bcf28ac9f0 (staff code NV001 typed in for 'Bùi Dũng Lan'); on the first value of the 300 questions it
  fired 6 times, 3 on wrong values and 3 on right ones.
- profit_bound: a profit or loss larger in size than the shop's whole revenue in the period (q_40de33fe6cdc: -2.29 bn against
  314.5 m of revenue in January 2025, gold -62 m).
- period_mismatch: a calendar phrase of the question that no period handed to the Coder (`period_start` / `period_end`, `compare_*`, the
  brief's other periods) matches (the spec stage's coverage check, repeated on the final periods; error analysis, section "Review").
  The Coder cannot change them: the Explainer is asked once more (only while a re-run is left).
- date_literal: a full date written in the answering block that lies outside every period of the question and of the spec, so the
  block filters on dates of its own instead of `period_start` / `period_end`. A heuristic (balances, ranges and windows are exempt):
  logged in shadow, not enforced unless it is left out of DATAMATE_AUDIT_SKIP.
Firing is not fixing: in v3.5 those answers were fixed by a re-run of the Explainer, not by the Coder alone.
DATAMATE_AUDIT: off | shadow (the default: logged with the turn, decides nothing) | on (an issue the Evaluator sends back,
said once like engine_mismatch). Each was seen on one or two dev300 questions: measure them in shadow before turning them on.
"""
from __future__ import annotations

import ast
import numbers
import re
from datetime import date, timedelta

from datamate.agents.periods import RANGE_OR_BALANCE, unmatched_periods
from datamate.text import fold_text, norm

# profit questions: "lãi" / "lỗ" need their diacritics ("lai" is also "lại", "lo" also "lô"); folded, only unambiguous phrases
PROFIT = re.compile(r"lãi|lợi nhuận|(?<!\w)lỗ(?!\w)", re.I)
PROFIT_FOLDED = re.compile(r"\bloi nhuan\b|\blai (?:rong|gop)\b|\btien lai\b|\blai lo\b")


def string_literals(code: str) -> str:
    """The code's string constants, folded and joined; f-strings and print(...) arguments left out. "" when it does not parse."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    skip: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.JoinedStr) or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"):
            skip |= {id(x) for x in ast.walk(n)}
    return fold_text(" | ".join(n.value for n in ast.walk(tree)
                                if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip))


def entity_absent(spec, codes: list[str]) -> list[str]:
    """The spec's entity values that no string constant of the turn's code blocks contains (compared folded). Each block is
    parsed alone: one block with a syntax error must not hide the others' strings."""
    codes = [c for c in codes if c and c.strip()]
    if not codes:
        return []
    lits = " | ".join(string_literals(c) for c in codes)   # the sandbox holds no entity variable: a name used is a literal
    out = []
    for e in getattr(spec, "entities", None) or []:
        v = str(e.get("value") or "").strip() if isinstance(e, dict) else ""
        if len(fold_text(v)) >= 2 and fold_text(v) not in lits:
            out.append(v)
    return out


def is_profit_question(question: str) -> bool:
    return bool(PROFIT.search(question or "")) or bool(PROFIT_FOLDED.search(fold_text(question or "")))


def profit_bound(question: str, kind: str, value, variables: dict) -> str:
    """A sentence when a profit / loss is larger in size than the period's revenue (`sales_metrics`), else ""."""
    if kind != "number" or not isinstance(value, numbers.Real) or isinstance(value, bool) or not is_profit_question(question):
        return ""
    sm, lo, hi = variables.get("sales_metrics"), variables.get("period_start"), variables.get("period_end")
    if sm is None or lo is None or hi is None:
        return ""
    try:
        revenue = float(sm(str(lo.date()), str(hi.date()))["revenue"])
    except Exception:  # noqa: BLE001 - no revenue to compare with: nothing to say
        return ""
    if revenue <= 0 or abs(float(value)) <= revenue:
        return ""
    return (f"a profit / loss of {float(value):,.0f} is larger in size than the shop's whole revenue in the period "
            f"({revenue:,.0f}, `sales_metrics`).")


def period_mismatch(question: str, named: list, periods: list) -> str:
    """A sentence when a calendar phrase of the question (`named`) is matched by none of the periods handed to the Coder (`periods`:
    objects with .start .end .role), else "". Same rule as the spec stage (agents/periods.unmatched_periods): equal, or containing
    where the question is a balance / range / window."""
    spans = [(str(p.start)[:10], str(p.end)[:10]) for p in periods]
    bad = unmatched_periods(question, named, spans) if named else []
    if not bad:
        return ""
    asked = "; ".join(f"{m.label} {m.start}..{m.end}" for m, _ in bad)
    given = "; ".join(f"{getattr(p, 'role', 'main')} {p.start}..{p.end}" for p in periods) or "none"
    return f"the question names {asked}, but the periods handed to the Coder (`period_start`/`period_end`, `compare_*`) are: {given}."


FULL_DATE = re.compile(r"(?<![\d/.\-])(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)|(?<![\d/.\-])(\d{1,2})/(\d{1,2})/(\d{4})(?![\d/])")
DATE_CALLS = {"Timestamp", "datetime", "date"}


def dates_in(text: str) -> set[date]:
    """Full dates written in a text: 2025-03-01, 2025-3-1, 01/03/2025 (day first, as the questions write them)."""
    out = set()
    for m in FULL_DATE.finditer(text or ""):
        y, mo, d = (m.group(1), m.group(2), m.group(3)) if m.group(1) else (m.group(6), m.group(5), m.group(4))
        try:
            out.add(date(int(y), int(mo), int(d)))
        except ValueError:
            pass
    return out


def code_dates(code: str) -> set[date]:
    """Full dates the code writes: in string constants ('2025-03-01', '2025-03-01 00:00:00') and in pd.Timestamp(2025, 3, 1) /
    datetime(2025, 3, 1) calls. F-strings and print(...) arguments only echo, as in string_literals. Empty when it does not parse."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    skip: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.JoinedStr) or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"):
            skip |= {id(x) for x in ast.walk(n)}
    out: set[date] = set()
    for n in ast.walk(tree):
        if id(n) in skip:
            continue
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out |= dates_in(n.value)
        elif isinstance(n, ast.Call):
            f = n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else ""
            args = [a.value for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, int) and not isinstance(a.value, bool)]
            if f in DATE_CALLS and len(args) >= 3 and len(args) == len(n.args):
                try:
                    out.add(date(*args[:3]))
                except ValueError:
                    pass
    return out


def date_literal(question: str, named: list, periods: list, code: str, as_of: str) -> str:
    """A sentence when the answering block writes a full date outside every period of the question and the spec (±1 day: `< end + 1`),
    else "". Exempt: a question that is a balance, a range or a window (RANGE_OR_BALANCE: its bounds are wide on purpose), dates the
    question itself writes, dates more than 3 years before or 1 year after as_of (sentinels like 1900-01-01 / 2100-01-01), and a
    question with no period at all (nothing to compare with)."""
    if not code or RANGE_OR_BALANCE.search(norm(question)):
        return ""
    one = timedelta(days=1)
    allowed = [(p.start - one, p.end + one) for p in [*periods, *named]] + [(d - one, d + one) for d in dates_in(question)]
    if not allowed or not (periods or named):
        return ""
    today = date.fromisoformat(as_of[:10])
    lo, hi = today - timedelta(days=3 * 366), today + timedelta(days=366)
    stray = sorted(d for d in code_dates(code) if lo <= d <= hi and not any(a <= d <= b for a, b in allowed))
    if not stray:
        return ""
    ranges = "; ".join(sorted({f"{p.start}..{p.end}" for p in [*periods, *named]}))
    return (f"the code writes the date(s) {', '.join(str(d) for d in stray[:4])}, which lie outside every period of the question "
            f"({ranges}).")


def audit_turn(question: str, spec, kind: str, value, codes: list[str], variables: dict,
               named: list | None = None, periods: list | None = None, as_of: str | None = None) -> dict:
    """{check: detail} for the checks that fire on this turn (`codes`: its code blocks, the last one is the answering block; `named`:
    the question's calendar phrases, `periods`: the periods handed to the Coder); {} when none does."""
    out: dict = {}
    try:
        if missing := entity_absent(spec, codes):
            out["entity_absent"] = missing
        if bound := profit_bound(question, kind, value, variables):
            out["profit_bound"] = bound
        if named is not None and periods is not None:
            if why := period_mismatch(question, named, periods):
                out["period_mismatch"] = why
            if as_of and codes and (stray := date_literal(question, named, periods, codes[-1], as_of)):
                out["date_literal"] = stray
    except Exception:  # noqa: BLE001 - an audit never fails the turn
        return out
    return out
