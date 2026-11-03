"""The rulebook engines as functions the coder can CALL (they no longer answer questions themselves).

The engines compute the standard numbers correctly (valid orders only, order date, quantities in base units), but as
answerers they had to guess which question they were looking at, and that routing needed a growing set of wording rules
tuned to the training questions. Now the coder reads the question, decides, and calls an engine when it fits: the
answer is always the coder's code. Misuse is loud: an unknown item name raises an error that lists the closest names,
instead of silently returning 0.
"""
from __future__ import annotations

import difflib
from typing import Callable

import pandas as pd

from datamate.engines.debt import DebtView
from datamate.engines.items import SIZE, ItemView, fold
from datamate.engines.orders import top_customers as _top_customers
from datamate.engines.purchases import PurchaseView, closest as _closest_items
from datamate.engines.sales import SalesView


def _ts(x) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()


def _num(x: float):
    return int(round(x)) if abs(x - round(x)) < 1e-9 else round(float(x), 4)


def build_tools(sales: SalesView | None, items: ItemView | None, debt: DebtView | None = None,
                orders: pd.DataFrame | None = None, purchases: PurchaseView | None = None, lots=None) -> dict[str, Callable]:
    """Functions for the coder sandbox. Dates: 'YYYY-MM-DD' strings or Timestamps, both ends included."""
    tools: dict[str, Callable] = {}

    if sales is not None:
        def sales_metrics(start, end) -> dict:
            """Whole-shop {'revenue', 'orders', 'avg_order'} for valid orders (no cancelled/returned) placed in [start, end]."""
            m = sales.metrics(_ts(start), _ts(end))
            return {"revenue": m["revenue"], "orders": m["orders"], "avg_order": m["avg_order"]}
        tools["sales_metrics"] = sales_metrics

    if items is not None:
        keys = set(items.lines.loc[items.lines["is_item"], "key"].dropna().astype(str))

        def _key(item: str) -> str:
            k = fold(item)
            k = items.aliases.get(k, k)          # a short form from the shop's abbreviation sheet -> its full name
            if k in keys:
                return k
            names = [items.display.get(c, c) for c in difflib.get_close_matches(k, sorted(keys), n=5, cutoff=0.5)]
            raise ValueError(f"no item named {item!r}; closest names in the data: {names}. Use find_items(...) to search.")

        def find_items(text: str, n: int = 10) -> list[str]:
            """Item names (exact data spelling) containing `text` (case/diacritics ignored); closest names if none contain it."""
            t = fold(text)
            hit = sorted({k for k in keys if t and t in k} | {v for a, v in items.aliases.items() if t and t in a and v in keys})
            if not hit:
                hit = difflib.get_close_matches(t, sorted(keys), n=n, cutoff=0.4)
            return [items.display.get(k, k) for k in hit[:n]]

        def item_qty(item: str, start, end):
            """Quantity sold in BASE units of `item` over [start, end], valid orders only."""
            return _num(items.qty(_key(item), _ts(start), _ts(end)))

        def item_revenue(item: str, start, end) -> int:
            """Revenue (line amounts) of `item` over [start, end], valid orders only."""
            return items.revenue(_key(item), _ts(start), _ts(end))

        def item_base_unit(item: str):
            return items.base_unit(_key(item))

        def top_items(k: int, start, end, by: str = "qty") -> list:
            """[(name, value)] best sellers over [start, end]; by='qty' (base units) or 'amount' (revenue). Ties: by item code."""
            if by not in ("qty", "amount"):
                raise ValueError("by must be 'qty' or 'amount'")
            return [(name, _num(v)) for name, v in items.top(int(k), _ts(start), _ts(end), by=by)]

        def item_price_on(item: str, day):
            """Unit price per base unit of `item` on orders placed on `day` (None if it was not sold that day)."""
            return items.price_on(_key(item), _ts(day))

        def item_qty_change(item: str, day, n: int) -> dict:
            """{'before', 'after', 'pct'}: quantity in the n days from `day` (included) vs the n days right before."""
            return items.change(_key(item), _ts(day), int(n))

        sizes = sorted({m.group(1).upper() for m in items.lines["item"].dropna().astype(str).map(SIZE.search) if m})

        def size_qty(size: str, start, end):
            """Quantity of drinks/items of size `size` ('M', 'L', ...) over [start, end]."""
            if str(size).strip().upper() not in sizes:      # loud, like an unknown item: an unknown size is not 0 drinks
                raise ValueError(f"no item of size {size!r}; the sizes in the item names are {sizes}")
            return _num(items.size_qty(size, _ts(start), _ts(end)))

        _key_of = _key
        tools.update(find_items=find_items, item_qty=item_qty, item_revenue=item_revenue, item_base_unit=item_base_unit,
                     top_items=top_items, item_price_on=item_price_on, item_qty_change=item_qty_change, size_qty=size_qty)

    if debt is not None:  # rulebook §4, see engines/debt.py
        def debt_total(day, older_than_days=None) -> int:
            """Money customers still owe at the end of `day` (orders placed more than `older_than_days` days before, if given)."""
            return debt.total_owed(_ts(day), None if older_than_days is None else int(older_than_days))

        def debt_by_customer(day, k=None) -> list:
            """[(customer name, owed)] at the end of `day`, largest first; walk-in orders have no row."""
            rows = debt.by_customer(_ts(day))
            return rows[: int(k)] if k else rows

        def debt_customers(day) -> int:
            """Number of customers who still owe money at the end of `day`."""
            return len(debt.by_customer(_ts(day)))

        def debt_aging(day, edges=(7, 15)) -> list:
            """Owed money by age since the order date: [0..7, 8..15, >15] days for the default edges."""
            return debt.aging(_ts(day), tuple(int(e) for e in edges))

        def debt_repaid(start, end, customer=None) -> int:
            """Debt repayments dated in [start, end] (payments after the purchase day, credit orders), optionally one customer."""
            return debt.repaid(_ts(start), _ts(end), customer)

        def debt_new(start, end, customer=None) -> int:
            """New debt from credit orders placed in [start, end] (unpaid at the end of the purchase day), optionally one customer."""
            return debt.new_debt(_ts(start), _ts(end), customer)

        tools.update(debt_total=debt_total, debt_by_customer=debt_by_customer, debt_customers=debt_customers, debt_aging=debt_aging)
        if getattr(debt, "flows", True):     # a shop that records only each order's balance at the end of its data has no flows
            tools.update(debt_repaid=debt_repaid, debt_new=debt_new)

    if purchases is not None:  # rulebook §5, see engines/purchases.py
        pkeys = set(purchases.lines["key"].astype(str))

        def _pkey(item: str) -> str:
            k = fold(item)
            if k not in pkeys:
                raise ValueError(f"no purchased item named {item!r}; closest names in the purchase data: {_closest_items(purchases, k)}")
            return k

        def purchase_total(start, end) -> int:
            """Money of the purchase slips dated in [start, end] (cancelled slips left out): `tổng tiền nhập hàng`."""
            return int(round(purchases.total(_ts(start), _ts(end))))

        def purchase_avg_price(item: str, start, end, unit=None):
            """Average import price of `item` per `unit` over the slips dated in [start, end], WEIGHTED by the quantity bought
            (not the mean of the prices), in the unit of the purchase line. `unit` (e.g. 'thùng') keeps the lines in that unit;
            without it the item must have been bought in one unit, otherwise the units are listed. None when nothing was bought."""
            k, s, e = _pkey(item), _ts(start), _ts(end)
            if unit is None:
                us = purchases.units(k, s, e)
                if len(us) > 1:
                    raise ValueError(f"{item!r} was bought in several units in this period {us}: pass unit=...")
            v = purchases.avg_price(k, s, e, unit)
            return None if v is None else int(round(v))

        tools.update(purchase_total=purchase_total, purchase_avg_price=purchase_avg_price)
        if purchases.lines["supplier"].notna().any():
            def top_supplier(start, end, k=1) -> list:
                """[(supplier, money)] of the slips dated in [start, end], largest first (`nhà cung cấp nhập nhiều nhất`: the first)."""
                return [(n, int(round(v))) for n, v in purchases.top_suppliers(_ts(start), _ts(end), int(k))]
            tools["top_supplier"] = top_supplier

    if orders is not None and {"date", "status_std"} <= set(orders.columns):   # rulebook §1 "Tỉ lệ đơn huỷ trong kỳ"
        def cancel_rate(start, end):
            """% of the orders PLACED in [start, end] that are cancelled, 2 decimals. The denominator is EVERY order of the
            period, every status (completed, shipping, cancelled, returned); returned orders are not cancelled. None when no
            order was placed in the period. Train gold: 135 of 135 questions with a clear period."""
            s, e = _ts(start), _ts(end) + pd.Timedelta(days=1)
            w = orders[(orders["date"] >= s) & (orders["date"] < e)]
            if not len(w):
                return None
            return round(float((w["status_std"] == "cancelled").sum()) / len(w) * 100, 2)
        tools["cancel_rate"] = cancel_rate

    if lots is not None and items is not None:   # rulebook §6 (pharmacies), see engines/lots.py
        def expiring_lots(day, n) -> list:
            """[(item, lot, expiry 'YYYY-MM-DD', remaining base qty, money at the lot's own import price)] of the lots still holding
            stock at the end of `day` whose expiry date is in [day, day + n] (§6), earliest expiry first, then item code."""
            e = lots.expiring(_ts(day), int(n)).sort_values(["expiry", "code", "lot"])
            return [(lots.names.get(r.item_key, r.item_key), r.lot, str(r.expiry.date()), _num(r.remaining), _num(round(r.value, 2)))
                    for r in e.itertuples()]

        def expiring_items(day, n) -> list:
            """Item names with a lot still in stock at the end of `day` expiring in [day, day + n] (§6), by earliest expiry, then code."""
            e = lots.expiring(_ts(day), int(n)).sort_values(["expiry", "code"])
            return list(dict.fromkeys(lots.names.get(k, k) for k in e["item_key"]))

        def expiring_value(day, n) -> int:
            """§6 "tiền vốn": money of the stock left at the end of `day` in lots expiring in [day, day + n], each at its own import
            price per base unit."""
            return int(round(lots.expiring(_ts(day), int(n))["value"].sum()))

        def nearest_expiry(item: str, day):
            """§6: the expiry date ('YYYY-MM-DD') of the item's lot that expires first among those still in stock and not expired at
            `day`; None when no such lot."""
            d = lots.nearest(_key_of(item), _ts(day))
            return None if d is None else str(pd.Timestamp(d).date())

        tools.update(expiring_lots=expiring_lots, expiring_items=expiring_items, expiring_value=expiring_value,
                     nearest_expiry=nearest_expiry)

    # rulebook §3, see engines/orders.py; not offered when no order names a customer (a repair book: every order is a walk-in)
    if orders is not None and "customer_key" in orders.columns and bool((orders["customer_key"].astype(str) != "").any()):
        def top_customers(k=None, start=None, end=None) -> list:
            """[(customer, revenue)] for valid orders placed in [start, end], largest first; one customer per customer
            code, walk-in orders never ranked (§3). Ties: by customer code."""
            lo = start if start is not None else orders["date"].min()
            hi = end if end is not None else orders["date"].max()
            return _top_customers(orders, None if k is None else int(k), lo, hi)
        tools["top_customers"] = top_customers
    return tools


TOOL_NAMES = ("sales_metrics", "find_items", "item_qty", "item_revenue", "item_base_unit", "top_items", "item_price_on",
              "item_qty_change", "size_qty", "debt_total", "debt_by_customer", "debt_customers", "debt_aging", "debt_repaid",
              "debt_new", "top_customers", "purchase_total", "purchase_avg_price", "top_supplier", "cancel_rate", "sales_orders",
              "expiring_lots", "expiring_items", "expiring_value", "nearest_expiry")

# Engines that hard-code one rulebook row keep that row's defining words here. The split's rulebook may differ from the
# training one (the public / private test rulebooks are not in the repo): when the words are gone, the engine is switched off
# and the Coder computes from the tables, instead of an "exact" function silently applying an old rule.
RULE_ANCHORS = {
    "cancel_rate": ("phan tram don huy trong moi don dat trong ky", "moi trang thai"),
}


def rule_gated(rulebook: str) -> set[str]:
    """Engine functions whose rulebook row (RULE_ANCHORS) is not in this rulebook: not offered to the Coder."""
    from datamate.text import fold_text
    text = fold_text(rulebook or "")
    return {name for name, words in RULE_ANCHORS.items() if not all(w in text for w in words)}
