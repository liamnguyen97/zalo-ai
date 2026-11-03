"""Deterministic sales view: revenue / number of orders / average per order for a period.

The rulebook defines these precisely (revenue = sum of line amounts after line discounts, no VAT, no delivery fee,
only orders that are not cancelled and not returned, by order date; an order with any returned item is a returned
order). Left to the model they were the most frequent source of wrong answers (summary rows summed in, lines
counted instead of orders, an invented de-duplication, ...). Here they are computed in code, from the cleaned
tables, and validated against the training answers.

The view is built from what the README declares (see data/readme.py) and falls back to column-name heuristics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from datamate.answer.formatter import round_half_up
from datamate.data.loader import Business, _base_name, _safe
from datamate.data.readme import roles_by_section, roles_for_tables, row_documents
from datamate.data.status import find_order_col
from datamate.text import nodia

AMT_OK = re.compile(
    r"ti[ềe]n h[àa]ng|tien_?hang|th[àa]nh ?ti[ềe]n|thanh_?tien|^amount$|t\.ti[ềe]n|^ti[ềe]n( b[áa]n)?$|^t[ổo]ng$|line_?total|^tt$"
    r"|^t[ổo]ng ti[ềe]n$|tr[ịi] gi[áa]|gi[áa] tr[ịi]", re.I)
AMT_BAD = re.compile(
    r"t[ổo]ng (bill|phi[ếe]u|[đd][ơo]n|c[ộo]ng|thanh to[áa]n|sau)|kh[áa]ch|c[òo]n|gi[ảa]m|chi[ếe]t|thu[ếe]|ph[íi]|ship|v[ốo]n|l[ãa]i"
    r"|gi[áa] (b[áa]n|nh[ậa]p|v[ốo]n)|[đd][ơo]n gi[áa]|s[ốo] l|t[ốo]ng d[òo]ng|tt d[òo]ng", re.I)
DATE_PREF = re.compile(r"ng[àa]y (b[áa]n|[đd][ặa]t|giao d[ịi]ch|xu[ấa]t|l[ậa]p|mua)|^ng[àa]y$|th[ờo]i gian|ngay|date", re.I)
NOT_SALES = re.compile(
    r"nhap|tonkho|baocaoton|^ton|giaohang|sogiao|hanno|sohan|banggia|danhmuc|quydoi|khachhang$|sanpham$|thuoc$|chiphi|soquy|nhacungcap"
    r"|lohang|thekho|trahang|lichsugia|donvitinh|dvt|phieugiao|thanhtoan$|nhanvien|chinhanh|ghitat")
IS_SALES = re.compile(r"ban|hoadon|donhang|chitiet|sales|order|bill")
# Rows that sit among an order's lines but are not goods: voucher / loyalty-points redemption (negative) and delivery fee.
# Rulebook: order-level vouchers are not deducted and delivery fees are not revenue. ("Phí phục vụ" stays: it IS revenue.)
NON_ITEM = re.compile(r"^\s*(?:voucher|d[ùu]ng\s*\d+\s*[đd]i[ểe]m|ph[íi]\s*(?:ship|giao|v[ậa]n chuy[ểe]n|giao h[àa]ng))\b", re.I)


def _nm(n: str) -> str:
    return re.sub(r"[_\s]+", "", nodia(n).lower())


@dataclass
class SalesView:
    lines: pd.DataFrame  # order, date, amount, valid
    table: str
    amount_col: str
    order_col: str | None
    date_col: str
    source: str  # "readme" | "heuristic"
    row_docs: bool = False  # no order id and each row is a whole document (a repair slip): one row = one order

    def metrics(self, start, end) -> dict:
        d = self.lines
        m = (d["date"] >= pd.Timestamp(start)) & (d["date"] < pd.Timestamp(end) + pd.Timedelta(days=1)) & d["valid"]
        s = d[m]
        n = int(s["order"].nunique())
        rev = float(s["amount"].sum())
        # §3: the average order value is whole đồng, half a đồng rounded UP (Python's round would round half to even)
        return {"revenue": int(round(rev)), "orders": n, "avg_order": round_half_up(rev / n) if n else None, "lines": int(len(s))}


# a column that makes a row one LINE of a document: an item name or a quantity (normalised names, as in engines/items.py)
LINE_COL = re.compile(r"^(ten)?(hang|hanghoa|sanpham|mon|sp|mathang|goi)(hoa|ban)?$|^noidung$|^(soluong|sl|slg|sluong)")


def is_row_documents(readme: str, df: pd.DataFrame) -> bool:
    """A table without order ids whose rows are whole documents: the README says so ("mỗi dòng một lượt xe (một phiếu)"), or no
    column makes a row a line of something (no item, no quantity). The shops whose README says "mỗi dòng một mặt hàng" have item and
    quantity columns: their rows stay lines (biz_107 / biz_113 / biz_126 have no orders at all)."""
    if row_documents(readme):
        return True
    return not any(LINE_COL.search(re.sub(r"[\W_]+", "", nodia(str(c)).lower())) for c in df.columns)


def _pick_amount(df: pd.DataFrame) -> str | None:
    cands = [c for c in df.columns if df[c].dtype.kind in "if" and AMT_OK.search(c) and not AMT_BAD.search(c)]
    for pat in ("th[àa]nh ?ti[ềe]n", "amount", r"t\.ti[ềe]n", "thanh_?tien", "tr[ịi] gi[áa]", "^ti[ềe]n( b[áa]n)?$", "^t[ổo]ng$", "ti[ềe]n h[àa]ng"):
        for c in cands:
            if re.search(pat, c, re.I):
                return c
    return cands[0] if cands else None


def _pick_date(df: pd.DataFrame) -> str | None:
    dc = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    for c in dc:
        if DATE_PREF.search(c):
            return c
    return dc[0] if dc else None


def returned_orders(tables: dict[str, pd.DataFrame]) -> set[str]:
    """Orders with a return recorded in a side table (a returned-quantity column > 0, or a returns table)."""
    out: set[str] = set()
    for n, df in tables.items():
        oc = find_order_col(df)
        if oc is None:
            continue
        for c in df.columns:
            if df[c].dtype.kind in "if" and re.search(r"tr[ảa]", c, re.I) and re.search(r"s[ốo] l|sl|qty|l[ưu][ợo]ng", c, re.I):
                out |= set(df.loc[df[c] > 0, oc].dropna().astype(str))
        if re.search(r"tr[ảa]_?h[àa]ng|tra_?hang|return", n, re.I):
            out |= set(df[oc].dropna().astype(str))
    return out


def _table_roles(biz: Business) -> dict[str, dict[str, str]]:
    """Map README sections ({file stem, sheet} -> roles) onto the loader's table names."""
    out: dict[str, dict[str, str]] = {}
    for (stem, sheet), roles in roles_by_section(biz.readme).items():
        key = _safe(stem) if not sheet else f"{_safe(stem)}__{_safe(sheet)}"
        for t, df in biz.tables.items():
            if t == key or t == _base_name(key) or _base_name(t) == _base_name(key):
                out.setdefault(t, {}).update({r: c for r, c in roles.items() if c in df.columns})
    # tables the file-name match missed: match README sections by the columns they document
    by_cols = roles_for_tables(biz.readme, {t: list(df.columns) for t, df in biz.tables.items() if not out.get(t)})
    for t, roles in by_cols.items():
        out[t] = roles
    return {t: r for t, r in out.items() if r}


def _overlap(a: pd.Series, b: pd.Series) -> float:
    va, vb = set(a.dropna().astype(str).head(3000)), set(b.dropna().astype(str).head(3000))
    return len(va & vb) / min(len(va), len(vb)) if va and vb else 0.0


def enrich_from_orders(df: pd.DataFrame, table: str, order_key: str | None, dcol: str | None,
                       tables: dict[str, pd.DataFrame], roles: dict[str, dict[str, str]]) -> tuple[pd.DataFrame, str | None]:
    """Left-join date and/or status (`valid_order`) from the order-level table when the line table lacks them."""
    if not order_key or (dcol is not None and "valid_order" in df.columns):
        return df, dcol
    for n2, d2 in tables.items():
        if n2 == table:
            continue
        oc2 = roles.get(n2, {}).get("order_id") or find_order_col(d2)
        dc2 = roles.get(n2, {}).get("order_date") or _pick_date(d2)
        if not oc2 or len(d2) == 0 or d2[oc2].nunique() < 0.9 * len(d2) or _overlap(d2[oc2], df[order_key]) < 0.8:
            continue
        need_date = dcol is None and dc2
        take = [oc2] + ([dc2] if need_date else []) + (["valid_order"] if "valid_order" in d2.columns and "valid_order" not in df.columns else [])
        add = d2[take].drop_duplicates(oc2).rename(columns={dc2: "__odate"} if need_date else {})
        df = df.merge(add, left_on=order_key, right_on=oc2, how="left", suffixes=("", "__o"))
        if need_date:
            dcol = "__odate"
        break
    return df, dcol


def build_sales(biz: Business) -> SalesView | None:
    tables = biz.tables
    roles = _table_roles(biz)
    best = None
    for n, df in tables.items():
        r = roles.get(n, {})
        amt, source = r.get("line_amount"), "readme"
        if amt is None:
            if NOT_SALES.search(_nm(n)):
                continue
            amt, source = _pick_amount(df), "heuristic"
        if amt is None or (source == "heuristic" and NOT_SALES.search(_nm(n))):
            continue
        oc = r.get("order_id") or find_order_col(df)
        score = (source == "readme", bool(IS_SALES.search(_nm(n))), oc is not None, len(df))
        if best is None or score > best[0]:
            best = (score, n, oc, amt, source)
    if best is None:
        return None
    _, n, oc, amt, source = best
    df = tables[n].copy()
    row_docs = False
    if oc is None:  # ledger without order ids: every row is one record (= one order when the rows are documents)
        df["__row"] = range(len(df))
        oc = "__row"
        order_key = None
        row_docs = is_row_documents(biz.readme, tables[n])
    else:
        order_key = oc
    dcol = roles.get(n, {}).get("order_date") or _pick_date(df)
    df, dcol = enrich_from_orders(df, n, order_key, dcol, tables, roles)
    if dcol is None:
        return None
    valid = df["valid_order"].fillna(True) if "valid_order" in df.columns else pd.Series(True, index=df.index)
    valid = valid & ~df[oc].astype(str).isin(returned_orders(tables))
    non_item = pd.Series(False, index=df.index)
    for c in df.columns[:14]:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            non_item |= df[c].astype(str).str.match(NON_ITEM)
    # §1: a shop-wide voucher / order-level discount is not subtracted from revenue. Such rows are recorded as extra lines with a
    # negative amount and no item (dev_013: 27% of the rows, -10,000 each, no text for NON_ITEM to match): not goods lines
    non_item |= (df[amt] < 0).fillna(False)
    lines = pd.DataFrame({"order": df[oc].astype(str).values, "date": df[dcol].values, "amount": df[amt].values,
                          "valid": (valid & ~non_item).values, "order_ok": valid.values, "item": (~non_item).values})
    return SalesView(lines, n, amt, order_key, dcol, source, row_docs)
