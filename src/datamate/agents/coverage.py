"""COVERAGE: unanswerable decisions the data settles without a model (rulebook §7).

§7: "no data" is not 0: a period lying entirely outside the dates the shop's data covers, or information the shop does not
record, makes the question unanswerable. v2 on dev300 got 7 of 12 gold-unanswerable questions right; the misses were
`46e6ca` (the period starts before the data does: answered 0), `d18944` (every order of the month is a walk-in, so no
customer exists to rank: three rejected turns, then abstain) and questions needing a column that does not exist. These checks
use only the resolved period and the shop's tables, so they hold for any shop.
"""
from __future__ import annotations

import pandas as pd

AMOUNT_MEASURES = ("money", "orders", "quantity", "count", "ratio")


def period_outside_data(spec, variables: dict) -> str | None:
    """A reason when the main period lies entirely outside the dates of `sales_orders` AND of every table the spec uses; None
    when there is any doubt (no orders table, no period, a non-amount question, or another table has rows in the period)."""
    so, lo, hi = variables.get("sales_orders"), variables.get("period_start"), variables.get("period_end")
    if so is None or lo is None or hi is None or getattr(spec, "measure", None) not in AMOUNT_MEASURES:
        return None
    if "date" not in so.columns or so["date"].isna().all():
        return None
    first, last = so["date"].min(), so["date"].max()
    if not (hi < first or lo > last):
        return None
    tables = variables.get("tables") or {}
    for t in (getattr(spec, "tables", None) or {}):
        df = tables.get(t)
        if df is None:
            continue
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]) and ((df[c] >= lo) & (df[c] <= hi)).any():
                return None
    return (f"The period {lo.date()}..{hi.date()} lies entirely outside the dates the shop's data covers "
            f"({first.date()}..{last.date()}): no data is not 0 (rulebook §7).")


def no_identified_customers(variables: dict) -> str | None:
    """A reason when the main period has valid orders and every one is a walk-in: the shop does not record who bought, so a
    question about customers cannot be answered (rulebook §7). None when anything else is true."""
    so, lo, hi = variables.get("sales_orders"), variables.get("period_start"), variables.get("period_end")
    if so is None or lo is None or hi is None or not {"date", "valid", "walk_in"} <= set(so.columns):
        return None
    d = so[(so["date"] >= lo) & (so["date"] <= hi) & so["valid"]]
    if len(d) == 0 or not d["walk_in"].all():
        return None
    return (f"All {len(d)} valid orders between {lo.date()} and {hi.date()} are walk-in orders: the shop records no customer "
            "for them, so a question about customers cannot be answered (rulebook §7).")
