"""VERIFIED FACTS: counts computed in code for this question's periods and values, shown to the Coder before OUTPUT.

The Coder spends blocks printing what the data holds (date range, rows after each filter), and a filter that matches nothing
looks like a real 0: on dev300 q_b445c50f6b5e the morning revenue came out 0 in a month that had morning orders, the Coder
concluded UNANSWERABLE and was bounced, then stalled (ledger #9). These counts are cheap and exact, so the Coder starts from them:
- the data range of `sales_orders` (order date);
- per period (the same `period_*` / `compare_*` the sandbox holds): orders placed by `status_std`, and how many are valid;
- valid orders per `shift` when the question names a shift; valid orders per value of a spec entity that is an order column;
- valid sales lines of each spec item in the main period.
Counts only, never an amount: an amount would be a ready answer that ignores the question's other conditions.
Any failure leaves the section out.
"""
from __future__ import annotations

import re

import pandas as pd

from datamate.agents.question import norm
from datamate.text import fold

SHIFT_WORDS = re.compile(r"\bca (?:sang|chieu|toi|dem)\b|\b(?:theo|tung|moi) ca\b")
MAX_LINES = 10


def _day(x) -> str:
    return str(pd.Timestamp(x).date())


def verified_facts(question: str, spec, variables: dict) -> str:
    try:
        lines = _facts(question, spec, variables)
    except Exception:  # noqa: BLE001 - facts are a help, never a reason to fail the question
        return ""
    if not lines:
        return ""
    return ("# VERIFIED FACTS (counted by the system from the data; counts, not the answer)\n" + "\n".join(lines[:MAX_LINES]) + "\n\n")


def _facts(question: str, spec, variables: dict) -> list[str]:
    so = variables.get("sales_orders")
    if so is None or not len(so) or not {"date", "valid"} <= set(so.columns):
        return []
    d = so["date"]
    out = [f"- `sales_orders`: {len(so)} orders, order dates {_day(d.min())}..{_day(d.max())}."]
    asks_shift = "shift" in so.columns and bool(SHIFT_WORDS.search(norm(question)))
    entities = [e for e in (getattr(spec, "entities", None) or []) if isinstance(e, dict)]
    periods = [("main period", variables.get("period_start"), variables.get("period_end")),
               ("comparison period", variables.get("compare_start"), variables.get("compare_end"))]
    for role, lo, hi in periods:
        if lo is None or hi is None:
            continue
        w = so[(d >= lo) & (d <= hi)]
        status = ""
        if "status_std" in so.columns and len(w):
            vc = w["status_std"].value_counts()
            status = " (" + ", ".join(f"{k} {int(n)}" for k, n in vc.items()) + ")"
        valid = w[w["valid"].astype(bool)]
        out.append(f"- {role} {_day(lo)}..{_day(hi)}: {len(w)} orders placed{status}, {len(valid)} valid."
                   + ("" if len(w) else " No order in this period: no data is not 0 (rulebook §7)."))
        if asks_shift and len(valid):
            sc = valid["shift"].fillna("(outside both shifts)").value_counts()
            out.append("  valid orders by `sales_orders['shift']`: " + ", ".join(f"{k} {int(n)}" for k, n in sc.items()))
        for e in entities:
            col, val = e.get("column"), e.get("value")
            if col in so.columns and col not in ("date", "valid") and val is not None:
                n = int((valid[col].astype(str).map(fold) == fold(val)).sum())
                out.append(f"  valid orders with `{col}` == {val!r}: {n}")
        if role == "main period":
            out += _item_lines(variables.get("item_lines"), entities, lo, hi)
    return out


def _item_lines(il, entities: list[dict], lo, hi) -> list[str]:
    if il is None or not {"key", "date", "valid"} <= set(il.columns):
        return []
    out = []
    for e in entities:
        if e.get("role") != "item" or not e.get("value"):
            continue
        k = fold(e["value"])
        m = (il["key"] == k) & (il["date"] >= lo) & (il["date"] <= hi) & il["valid"].astype(bool)
        n_all = int((il["key"] == k).sum())
        out.append(f"  valid sales lines of item {e['value']!r} in the main period: {int(m.sum())} "
                   f"(in {il.loc[m, 'order'].nunique() if 'order' in il.columns else '?'} orders; {n_all} lines in all the data)")
    return out
