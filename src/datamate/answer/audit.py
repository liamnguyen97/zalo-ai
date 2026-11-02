"""AUDIT: checks of a turn's code and value that need no model, run beside the judge (ledger #12).

Measured on the dev300 v3.5 logs (scripts/review_counterfactual.py) as stand-ins for what the LLM review caught there:
- entity_absent: a value the question names (a spec entity) is in none of the code's string constants. F-strings and print()
  arguments do not count: they echo the name, they do not filter by it. It fired on q_0f6233b739e9 ('natri|clor' for
  'Natri clorid') and q_12bcf28ac9f0 (staff code NV001 typed in for 'Bùi Dũng Lan'); on the first value of the 300 questions it
  fired 6 times, 3 on wrong values and 3 on right ones.
- profit_bound: a profit or loss larger in size than the shop's whole revenue in the period (q_40de33fe6cdc: -2.29 bn against
  314.5 m of revenue in January 2025, gold -62 m).
Firing is not fixing: in v3.5 those answers were fixed by a re-run of the Explainer, not by the Coder alone.
DATAMATE_AUDIT: off | shadow (the default: logged with the turn, decides nothing) | on (an issue the Evaluator sends back,
said once like engine_mismatch). Each was seen on one or two dev300 questions: measure them in shadow before turning them on.
"""
from __future__ import annotations

import ast
import numbers
import re

from datamate.text import fold_text

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


def audit_turn(question: str, spec, kind: str, value, codes: list[str], variables: dict) -> dict:
    """{check: detail} for the checks that fire on this turn (`codes`: its code blocks); {} when none does."""
    out: dict = {}
    try:
        if missing := entity_absent(spec, codes):
            out["entity_absent"] = missing
        if bound := profit_bound(question, kind, value, variables):
            out["profit_bound"] = bound
    except Exception:  # noqa: BLE001 - an audit never fails the turn
        return out
    return out
