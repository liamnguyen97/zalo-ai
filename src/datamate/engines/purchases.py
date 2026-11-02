"""PURCHASES engine (rulebook §5): money and prices of goods bought from suppliers, in code.

Sales, items and debt had engines; purchases did not, and 97 of the 3,000 training questions (3.2%, 13 of 30 shops) are about
them: "tổng tiền nhập hàng tháng này", "giá nhập bình quân một thùng X trong tháng 1/2025". Left to a 9B model they went wrong in
the same way: train q_00a6e23cbacf divided the purchase slip's TOTAL (`Cần trả NCC`, every item on the slip) by one item's
quantity (7,892,815 against the gold 200,556).

Rulebook §5: purchases in a period = slips whose DATE is in the period, cancelled slips left out (a first-of-data stock slip is a
slip like any other); the average import price of an item in a month = weighted by the quantity bought, NOT the mean of the
prices, in the unit of the line ("theo đơn vị nhập").

Two layouts, both found from the data (data/structure.py, values only; the README only says which tables are purchases):
  slips  a slip table (date, supplier, status, total) + a lines table (item, unit, qty, unit price): the slip total must equal
         sum(qty x price) on >= 95% of slips, which is also what tells the lines' quantity and price columns;
  flat   one table, one row per line, the slip's fields repeated: qty x price must equal an amount column on >= 95% of rows.
The item column is the text column whose values are item codes or names of the sales lines (>= 80%). Anything not found makes the
engine absent for that shop (no guess), and the onboarding report says so.
"""
from __future__ import annotations

import difflib
import itertools
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from datamate.data.status import STATUS_COL, classify
from datamate.text import fold

PURCHASE_NAME = re.compile(r"nh[ậa]p|nhap|purchase|import|receiv", re.I)
PURCHASE_NOTE = re.compile(r"gi[áa] nh[ậa]p|ti[ềe]n nh[ậa]p|phi[ếe]u nh[ậa]p|nh[ậa]p h[àa]ng", re.I)
SUPPLIER_COL = re.compile(r"ncc|nhacungcap|supplier|vendor|nhacc|doitac")      # on a folded name without separators
MATCH = 0.95          # share of slips / rows where the amounts must agree
ITEM_COVER = 0.8      # share of a column's values that must be item codes / names of the sales lines


@dataclass
class PurchaseView:
    lines: pd.DataFrame          # slip, date, valid, key, item, unit, qty, price, amount, supplier (None when unknown)
    layout: str                  # "slips" | "flat"
    tables: list[str]
    cols: dict                   # provenance: the columns chosen, for the onboarding report
    display: dict[str, str] = field(default_factory=dict)      # key -> item name as in the data
    supplier_names: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ selection
    def _sel(self, start, end) -> pd.DataFrame:
        d = self.lines
        m = d["valid"] & (d["date"] >= pd.Timestamp(start)) & (d["date"] < pd.Timestamp(end) + pd.Timedelta(days=1))
        return d[m]

    def total(self, start, end) -> float:
        """Money bought in [start, end]: sum of the slips' line amounts (= the slips' totals), cancelled slips left out."""
        return float(self._sel(start, end)["amount"].sum())

    def units(self, key: str, start, end) -> dict[str, float]:
        s = self._sel(start, end)
        s = s[s["key"] == key]
        return {str(u): float(q) for u, q in s.groupby("unit")["qty"].sum().items()}

    def avg_price(self, key: str, start, end, unit: str | None = None) -> float | None:
        """Sum(qty x price) / sum(qty) over the item's lines in the period, in `unit` (all lines when None)."""
        s = self._sel(start, end)
        s = s[s["key"] == key]
        if unit is not None:
            s = s[s["unit"].map(lambda u: fold(str(u))) == fold(unit)]
        q = float(s["qty"].sum())
        return float(s["amount"].sum()) / q if q else None

    def top_suppliers(self, start, end, k: int | None = None) -> list[tuple[str, float]]:
        s = self._sel(start, end)
        s = s[s["supplier"].notna()]
        g = s.groupby("supplier")["amount"].sum().reset_index()
        g["name"] = g["supplier"].map(lambda v: self.supplier_names.get(str(v), str(v)))
        g = g.sort_values(["amount", "supplier"], ascending=[False, True])
        rows = [(r.name, float(r.amount)) for r in g.itertuples()]
        return rows[:k] if k else rows


# ---------------------------------------------------------------------------------------------- finders
def _cn(c) -> str:
    """Column name folded to letters and digits: 'Mã NCC', 'MaNhaCungCap' and 'nha_cung_cap' compare alike."""
    return re.sub(r"[\W_]+", "", fold(str(c)))


def _valid(df: pd.DataFrame) -> np.ndarray:
    """Slips that count: a slip whose status column says cancelled (or returned) does not (rulebook §5). Only labels the status
    rules recognise as cancelled are left out: 'Đã nhập hàng' / 'received' or a missing status count. (`add_status_std` is
    not used: it keeps a status column only when >= 90% of its labels are recognised, and 'Đã nhập hàng' is not one.)"""
    ok = np.ones(len(df), bool)
    for c in df.columns:
        if df[c].dtype.kind in "iufbM" or not STATUS_COL.search(str(c)):
            continue
        vals = df[c].dropna().astype(str).unique()
        bad = {v for v in vals if len(vals) <= 40 and classify(v) in ("cancelled", "returned")}
        if bad:
            ok &= ~df[c].astype(str).isin(bad).to_numpy()
    return ok


def _date_col(df: pd.DataFrame, hi) -> str | None:
    """The slip date: the datetime column (>= 80% filled) that ends earliest and not after the end of the data. An expiry date
    or a due date runs into the future; a slip date does not."""
    best = None
    for c in df.columns:
        s = df[c]
        if not pd.api.types.is_datetime64_any_dtype(s) or s.notna().mean() < 0.8:
            continue
        if hi is not None and s.max() > hi:
            continue
        if best is None or s.max() < best[0]:
            best = (s.max(), c)
    return best[1] if best else None


def _num_cols(df: pd.DataFrame, skip: set) -> list[str]:
    return [c for c in df.columns if c not in skip and df[c].dtype.kind in "if" and df[c].notna().mean() > 0.5][:8]


def _qty_price_amount(df: pd.DataFrame, skip: set) -> tuple[str, str, str] | None:
    """(qty, price, amount) with qty x price = amount on >= 95% of the non-zero rows of a flat table. The product does not say
    which factor is which: the one with the smaller median is the quantity."""
    cols = _num_cols(df, skip)
    best = None
    for x, y in itertools.combinations(cols, 2):
        prod = df[x] * df[y]
        for a in cols:
            if a in (x, y):
                continue
            m = (df[a].fillna(0) != 0) & prod.notna()
            if m.sum() < 20:
                continue
            share = float(np.isclose(prod[m], df.loc[m, a], rtol=0.005, atol=1.0).mean())
            if share >= MATCH and (best is None or share > best[0]):
                q, p = (x, y) if df[x].median() <= df[y].median() else (y, x)
                best = (share, q, p, a)
    return best[1:] if best else None


def _item_col(df: pd.DataFrame, skip: set, codes: set, keys: set) -> tuple[str, str] | None:
    """(column, "code" | "name"): the text column whose values are the sales lines' item codes or names."""
    best = None
    for c in df.columns:
        if c in skip or df[c].dtype.kind in "iufbM" or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        v = df[c].dropna().astype(str)
        if len(v) < 20:
            continue
        u = pd.Series(v.unique())
        for kind, hit in (("code", u.isin(codes).mean()), ("name", u.map(fold).isin(keys).mean())):
            if hit >= ITEM_COVER and (best is None or hit > best[0]):
                best = (hit, c, kind)
    return (best[1], best[2]) if best else None


def _unit_col(df: pd.DataFrame, skip: set, units: set) -> str | None:
    """The unit of the line: among the text columns with a few distinct values (<= 40) filled on most rows, the one whose values
    are units the shop SELLS in (the sales lines' units, >= 50% of them); else the one with the most distinct values. biz_105: the
    supplier column has more distinct values than the unit column and won the second rule."""
    cands = []
    for c in df.columns:
        if c in skip or df[c].dtype.kind in "iufbM" or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        s = df[c].dropna().astype(str)
        if len(s) >= 0.9 * len(df) and 1 <= s.nunique() <= 40:
            cands.append((c, s.nunique(), pd.Series(s.unique()).map(fold).isin(units).mean()))
    sold = [x for x in cands if x[2] >= 0.5]
    pool = sold or [x for x in cands if x[1] >= 2 and not SUPPLIER_COL.search(_cn(x[0]))]
    return max(pool, key=lambda x: (x[2], x[1]))[0] if pool else None


def _supplier_col(df: pd.DataFrame, skip: set) -> str | None:
    """A text column named like a supplier with 2..100 distinct values (a name regex AND the value shape: a code or a name)."""
    for c in df.columns:
        if c in skip or df[c].dtype.kind in "iufbM" or not SUPPLIER_COL.search(_cn(c)):
            continue
        s = df[c].dropna().astype(str)
        if len(s) >= 0.9 * len(df) and 2 <= s.nunique() <= min(100, 0.25 * len(df)):
            return c
    return None


def _names_for(values: pd.Series, tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    """code -> name for a column of codes: a table with a unique column holding >= 90% of the codes and a mostly-unique text
    column of multi-word values next to it."""
    u = set(values.dropna().astype(str))
    for t, df in tables.items():
        for c in df.columns:
            if df[c].dtype.kind in "iufbM":
                continue
            col = df[c].dropna().astype(str)
            if col.nunique() != len(col) or len(u & set(col)) < 0.9 * len(u):
                continue
            for n in df.columns:
                if n == c or df[n].dtype.kind in "iufbM":
                    continue
                s = df[n].dropna().astype(str)
                if len(s) >= 0.9 * len(df) and s.nunique() >= 0.9 * len(s) and s.str.contains(" ").mean() > 0.5:
                    return dict(zip(df[c].astype(str), df[n].astype(str)))
    return {}


# ---------------------------------------------------------------------------------------------- build
def build_purchases(biz, items, structure) -> PurchaseView | None:
    """The purchase lines of a shop with their slip date / status / supplier, or None when they cannot be found for sure."""
    if items is None or structure is None:
        return None
    T = biz.tables
    il = items.lines
    codes = set(il["code"].dropna().astype(str)) if "code" in il.columns else set()
    keys = set(il["key"].dropna().astype(str))
    units = {fold(str(u)) for col in ("unit", "base_unit") if col in il.columns for u in il[col].dropna().unique()}
    hi = il["date"].max() + pd.Timedelta(days=60) if il["date"].notna().any() else None
    sales_t = items.table
    named = {t for t in T if t != sales_t and (PURCHASE_NAME.search(t) or any(
        PURCHASE_NOTE.search(x) for x in structure.notes.get(t, {}).values()))}
    code_to_key = {}
    if "code" in il.columns:
        sub = il.dropna(subset=["code", "key"]).drop_duplicates("code")
        code_to_key = dict(zip(sub["code"].astype(str), sub["key"].astype(str)))
    display = {k: v for k, v in items.display.items()}

    found = None
    # --- layout "slips": slip table + lines table, the total tells qty and price
    for t in structure.totals:
        if "×" not in t.expr or t.lines not in named or t.lines == sales_t:
            continue
        refs = re.findall(r"`([^`]+)`", t.expr)
        link = next((l for l in structure.links if l.head == t.table and l.lines == t.lines), None)
        H, L = T[t.table], T[t.lines]
        if link is None or len(refs) != 2 or not all(r in L.columns for r in refs):
            continue
        dcol = _date_col(H, hi)
        ic = _item_col(L, {link.line_key, *refs}, codes, keys)
        if dcol is None or ic is None:
            continue
        q, p = refs
        unit = _unit_col(L, {link.line_key, ic[0], q, p}, units)
        hv = H.drop_duplicates(subset=[link.key]).copy()
        hv["_valid"] = _valid(H.drop_duplicates(subset=[link.key]))
        sup = _supplier_col(hv, {link.key, dcol, "_valid"})
        h = hv.set_index(hv[link.key].astype(str))
        slip = L[link.line_key].astype(str)
        out = pd.DataFrame({"slip": slip.to_numpy(), "date": slip.map(h[dcol]).to_numpy(), "valid": slip.map(h["_valid"]).fillna(True).to_numpy(bool),
                            "qty": L[q].astype(float).to_numpy(), "price": L[p].astype(float).to_numpy(),
                            "unit": (L[unit].astype(str) if unit else pd.Series("", index=L.index)).to_numpy(),
                            "supplier": (slip.map(h[sup]).to_numpy() if sup else None)})
        raw = L[ic[0]].astype(str)
        out["key"] = (raw.map(code_to_key) if ic[1] == "code" else raw.map(fold)).to_numpy()
        found = ("slips", out, [t.table, t.lines], {"slips": t.table, "lines": t.lines, "slip_key": link.key, "date": dcol, "item": ic[0],
                                                     "qty": q, "price": p, "unit": unit, "supplier": sup, "total": t.column,
                                                     "total_matches": t.share}, hv, sup)
        break
    # --- layout "flat": one row per line
    if found is None:
        for t in sorted(named):
            L = T[t]
            dcol = _date_col(L, hi)
            if dcol is None or len(L) < 20:
                continue
            qpa = _qty_price_amount(L, {dcol})
            if qpa is None:
                continue
            q, p, a = qpa
            ic = _item_col(L, {dcol, q, p, a}, codes, keys)
            if ic is None:
                continue
            doc = structure.doc_level.get(t, (None, []))[0]
            unit = _unit_col(L, {dcol, ic[0], doc}, units)
            sup = _supplier_col(L, {dcol, ic[0], doc, unit})
            raw = L[ic[0]].astype(str)
            out = pd.DataFrame({"slip": (L[doc].astype(str) if doc else pd.Series(L.index.astype(str), index=L.index)).to_numpy(),
                                "date": L[dcol].to_numpy(), "valid": _valid(L), "qty": L[q].astype(float).to_numpy(),
                                "price": L[p].astype(float).to_numpy(), "unit": (L[unit].astype(str) if unit else pd.Series("", index=L.index)).to_numpy(),
                                "supplier": L[sup].to_numpy() if sup else None})
            out["key"] = (raw.map(code_to_key) if ic[1] == "code" else raw.map(fold)).to_numpy()
            found = ("flat", out, [t], {"table": t, "slip_key": doc, "date": dcol, "item": ic[0], "qty": q, "price": p, "amount": a,
                                        "unit": unit, "supplier": sup}, L, sup)
            break
    if found is None:
        return None
    layout, out, tabs, cols, src, sup = found
    out["amount"] = out["qty"] * out["price"]
    out = out[out["date"].notna() & out["key"].notna()].reset_index(drop=True)
    if out.empty:
        return None
    names = {}
    if sup is not None:
        names = _names_for(out["supplier"], {t: d for t, d in T.items() if t not in tabs})
    cols["rows"] = len(out)
    cols["valid_share"] = round(float(out["valid"].mean()), 3)
    return PurchaseView(out, layout, tabs, cols, display, names)


def closest(view: PurchaseView, key: str, n: int = 5) -> list[str]:
    ks = sorted(set(view.lines["key"].astype(str)))
    return [view.display.get(k, k) for k in difflib.get_close_matches(key, ks, n=n, cutoff=0.5)]
