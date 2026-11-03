"""ONBOARDING REPORT: what the engines detected in one shop and whether it passes checks that need no gold answers.

The engines find their columns with name patterns and README sentences tuned on 30 shops. On the 14 unseen public-test shops
the orders engine was missing in 4 (29%, against 10% on the training shops) and the debt engine in 12; a wrong detection does
not raise, it returns a confident wrong number. This report lists, per shop, the tables and columns each engine chose
(provenance) and invariants that hold for any shop. `trusted` is False when a critical one fails: the engine reference
(engines/reference.py) is then not shown to the Coder, so a mis-detected shop cannot push a wrong "exact" number.

Critical invariants: an order-id column exists; dates parse; the order status is mapped; the order headers and the sales lines
agree on how many orders there are. Informational: share of walk-in orders, README roles vs heuristics, unit conversion share.
"""
from __future__ import annotations

import re

import pandas as pd


def _check(name: str, ok: bool, detail: str = "", critical: bool = True) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail, "critical": critical}


def onboard(biz, sales, items, debt, orders, purchases=None) -> dict:
    """Detection summary and invariant results for one shop. Never raises."""
    rep: dict = {"tables": {t: [len(d), len(d.columns)] for t, d in biz.tables.items()}, "engines": {}, "checks": []}
    ck = rep["checks"]
    rep["engines"] = {"sales": sales is not None, "items": items is not None, "debt": debt is not None, "orders": orders is not None,
                      "purchases": purchases is not None}
    try:
        if sales is None:
            ck.append(_check("sales_built", False, "no sales table recognized"))
        else:
            lines = sales.lines
            rep["sales"] = {"table": sales.table, "order_col": sales.order_col, "date_col": sales.date_col,
                            "amount_col": sales.amount_col, "source": sales.source, "rows": len(lines)}
            ledger = bool(re.search(r"không (?:ghi|có) (?:mã|số) (?:đơn|phiếu|hoá đơn|hóa đơn)", biz.readme or "", re.I))
            rows = getattr(sales, "row_docs", False)
            ck.append(_check("order_col_found", sales.order_col is not None or ledger or rows,
                             f"order column: {sales.order_col}" if sales.order_col else
                             ("one row per document: each row of the book is one order (no order id)" if rows else
                              "ledger: the README says the shop records no order id (each row is one record)" if ledger else
                              "no order id column found and the README does not say the shop has none")))
            nat = float(lines["date"].isna().mean())
            span = (lines["date"].max() - lines["date"].min()).days if lines["date"].notna().any() else 0
            ck.append(_check("dates_parse", nat < 0.01 and span >= 28, f"{nat:.1%} unparsed dates, span {span} days"))
            counted = lines[lines["item"] & lines["order_ok"]]
            neg = float((counted["amount"] < 0).mean()) if len(counted) else 0.0       # voucher rows are already excluded (item False)
            ck.append(_check("amounts_sane", neg < 0.01 and float(counted["amount"].median()) > 0, f"{neg:.1%} negative, median {counted['amount'].median():,.0f}"))
            status = [(t, d) for t, d in biz.tables.items() if "status_std" in d.columns]
            unmapped = max((float(d["status_std"].isna().mean()) for _, d in status), default=0.0)
            ck.append(_check("status_mapped", unmapped < 0.01, f"{unmapped:.1%} of rows have an unmapped status" if status else "no status column"))
            if orders is not None:
                n_lines = lines["order"].nunique()
                gap = abs(len(orders) - n_lines) / max(len(orders), n_lines, 1)
                ck.append(_check("headers_match_lines", gap < 0.05, f"{len(orders)} header orders vs {n_lines} order ids in the lines"))
                rep["orders"] = {"rows": len(orders), "valid_share": round(float(orders["valid"].mean()), 3),
                                 "walk_in_share": round(float(orders["walk_in"].mean()), 3) if "walk_in" in orders else None}
                ck.append(_check("walk_in_plausible", True, f"walk-in share {rep['orders']['walk_in_share']}", critical=False))
            else:
                ck.append(_check("orders_built", False, "no order-level table (sales_orders) could be built", critical=False))
    except Exception as e:  # noqa: BLE001 - a report must never break loading
        ck.append(_check("report_error", False, f"{type(e).__name__}: {e}"[:120]))
    try:
        if items is not None:
            ln = items.lines
            conv = float((ln["qty"] != ln["qty_base"]).mean())
            nan = float(ln["qty_base"].isna().mean())
            rep["items"] = {"table": items.table, "note": items.note, "lines": len(ln), "converted_share": round(conv, 3)}
            ck.append(_check("qty_base_filled", nan < 0.01, f"{nan:.1%} lines without a base quantity", critical=False))
        mentions_debt = bool(re.search(r"công nợ", biz.readme or "", re.I))
        ck.append(_check("debt_matches_readme", debt is not None or not mentions_debt,
                         "README mentions debt but no debt engine" if mentions_debt and debt is None else "ok", critical=False))
    except Exception as e:  # noqa: BLE001
        ck.append(_check("report_error", False, f"{type(e).__name__}: {e}"[:120], critical=False))
    try:
        mentions = bool(re.search(r"phiếu nhập|nhập hàng|giá nhập", biz.readme or "", re.I))
        if purchases is not None:
            rep["purchases"] = dict(purchases.cols, layout=purchases.layout)
        ck.append(_check("purchases_match_readme", purchases is not None or not mentions,
                         "README mentions purchases but no purchase engine" if mentions and purchases is None else "ok", critical=False))
    except Exception as e:  # noqa: BLE001
        ck.append(_check("report_error", False, f"{type(e).__name__}: {e}"[:120], critical=False))
    rep["trusted"] = all(c["ok"] for c in ck if c["critical"]) and sales is not None
    return rep


def summary(rep: dict) -> str:
    bad = [f"{c['name']} ({c['detail']})" for c in rep["checks"] if not c["ok"]]
    eng = ",".join(k for k, v in rep["engines"].items() if v) or "none"
    return f"trusted={rep['trusted']} engines={eng}" + (f" FAILED: {'; '.join(bad)}" if bad else "")
