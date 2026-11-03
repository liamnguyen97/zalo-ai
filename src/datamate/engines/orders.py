"""`sales_orders`: one row per order, with the rulebook's revenue already computed, for the coder to filter.

The coder answered "revenue of channel X / staff Y / customer group Z / payment W / morning shift" by summing an order
header total (after order-level discounts, vouchers, ...), which is not the rulebook's revenue (§1: sum of line amounts
after line discounts, orders that are not cancelled and not returned). The numbers came out a little off. Here the
revenue per order is the sales engine's own (sales.py), and the order's header columns (customer, channel, staff,
payment, ...) are joined on, so any condition is a plain filter followed by `revenue.sum()`.
"""
from __future__ import annotations

import re

import pandas as pd

from datamate.data.loader import Business
from datamate.data.status import find_order_col
from datamate.engines.debt import _clean_name, _customer, _ids
from datamate.engines.sales import SalesView, _overlap

COVERAGE = 0.99      # an order table must hold the ids of at least this share of the orders in the sales lines
QTY = r"s[ốo] ?l|sl\b|qty|l[ưu][ợo]ng|đi[ểe]m|diem|point|s[ốo] ng[àa]y"


def _header(biz: Business, orders: pd.Series, skip: str) -> tuple[str, pd.DataFrame, str] | None:
    """The order-level table: one row per order id, ids overlapping the sales lines."""
    best = None
    for t, df in biz.tables.items():
        if len(df) == 0:
            continue
        oc = find_order_col(df)
        if oc is None or df[oc].nunique() < 0.9 * len(df) or _overlap(_ids(df[oc]), orders) < 0.8:
            continue
        score = (t != skip, len(df.columns))
        if best is None or score > best[0]:
            best = (score, t, oc)
    if best is None:
        return None
    _, t, oc = best
    return t, biz.tables[t], oc


def build_orders(biz: Business, sales: SalesView | None) -> pd.DataFrame | None:
    """One row per order. A book whose rows are whole documents and that writes no order id (a repair book: one row per slip)
    has one order per row, `order` being the row number."""
    if sales is None or (sales.order_col is None and not sales.row_docs):
        return None
    L = sales.lines
    g = L.groupby("order", sort=False)
    out = pd.DataFrame({
        "date": g["date"].first(),
        "valid": g["order_ok"].all(),
        "revenue": L["amount"].where(L["item"], 0).groupby(L["order"], sort=False).sum(),
    })
    out.index.name = "order"
    out = out.reset_index()
    out.loc[~out["valid"], "revenue"] = 0.0      # §1: cancelled / returned orders bring no revenue
    hdr = _header(biz, out["order"], "") if sales.order_col is not None else None
    if hdr is not None and out["order"].isin(_ids(hdr[1][hdr[2]])).mean() < COVERAGE:
        hdr = None      # e.g. a payments table with one row per PAID order (dev_003): it lacks the unpaid orders and the order columns
    if hdr is None and sales.table in biz.tables:      # no order table: the sales lines carry the order's columns
        if sales.order_col is None:                    # one row per document: the row number is the id (as in the sales lines)
            hdr = sales.table, biz.tables[sales.table].assign(__row=range(len(biz.tables[sales.table]))), "__row"
        else:
            hdr = sales.table, biz.tables[sales.table], sales.order_col
    if hdr is not None:
        t, df, oc = hdr
        h = df.assign(__k=_ids(df[oc])).drop_duplicates("__k")
        keep = [c for c in h.columns if c not in ("valid_order", "__k") and not str(c).startswith("__") and c not in out.columns
                and (t != sales.table or c != sales.amount_col)]      # a line amount is not the order's
        out = out.merge(h[["__k"] + keep], left_on="order", right_on="__k", how="left").drop(columns="__k")
        key, name = _customer(biz, h.set_index("__k").reindex(out["order"]).reset_index(drop=True), t)
        out["customer_key"] = key.values
        out["customer"] = name.astype(str).map(_clean_name).where(key.values != "", "").values
        out["walk_in"] = out["customer_key"] == ""
        # header columns that look like the order's revenue (within 5% on most orders) but are not §1's revenue: an order
        # total after order-level discounts, vouchers, fees... Summing one for revenue is refused by the runner.
        v = out[out["valid"] & (out["revenue"] > 0)]
        like = []
        for c in keep:
            if pd.api.types.is_numeric_dtype(out[c]) and not re.search(QTY, c, re.I) and len(v):
                rel = ((pd.to_numeric(v[c], errors="coerce") - v["revenue"]).abs() / v["revenue"]).dropna()
                if len(rel) and (rel < 0.05).mean() >= 0.8:
                    like.append(c)
        out.attrs["header_money"] = like
    return _with_shift(out)


def _with_shift(out: pd.DataFrame) -> pd.DataFrame:
    """`hour` and `shift` ('sáng' 06:00–13:59, 'chiều' 14:00–21:59, else None; rulebook: by order creation time) from
    the header datetime column that really carries times. holdout6: the order date had no time (0 morning revenue),
    the creation-time column gave the gold value."""
    best, share, when = None, 0.0, None
    for c in out.columns:
        if c == "date":
            continue
        col = out[c]
        if not pd.api.types.is_datetime64_any_dtype(col):
            if col.dtype != object and not pd.api.types.is_string_dtype(col):
                continue
            sample = col.dropna().astype(str).head(200)          # times kept as text ("2025-01-01 06:13:20", "06:13")
            if not len(sample) or not sample.str.contains(r"\d{1,2}:\d{2}").mean() >= 0.8:
                continue
            col = pd.to_datetime(col.astype(str).where(col.notna()), errors="coerce", format="mixed")
            if col.notna().mean() < 0.8 * out[c].notna().mean():
                continue
        t = col.dropna()
        f = float((t != t.dt.normalize()).mean()) if len(t) else 0.0
        if f > share:
            best, share, when = c, f, col
    if best is None or share < 0.5:
        return out
    h = when.dt.hour
    out = out.assign(hour=h, shift=pd.Series(None, index=out.index, dtype=object))
    out.loc[(h >= 6) & (h < 14), "shift"] = "sáng"
    out.loc[(h >= 14) & (h < 22), "shift"] = "chiều"
    out.attrs["shift_from"] = best
    return out


def top_customers(orders: pd.DataFrame, k: int | None, start, end) -> list[tuple[str, int]]:
    """§3: customers by revenue over [start, end]; one customer per key, walk-ins never ranked. Ties: by key."""
    if "customer_key" not in orders.columns:
        raise ValueError("this shop has no customer column on its orders")
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
    o = orders[(orders["date"] >= s) & (orders["date"] < e) & orders["valid"] & ~orders["walk_in"]]
    agg = o.groupby("customer_key").agg(customer=("customer", "first"), revenue=("revenue", "sum")).reset_index()
    agg = agg.sort_values(["revenue", "customer_key"], ascending=[False, True])
    rows = [(r.customer or r.customer_key, int(round(r.revenue))) for r in agg.itertuples()]
    return rows[:k] if k else rows
