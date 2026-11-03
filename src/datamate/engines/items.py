"""Item-level view: base-unit quantities, revenue per item, rankings, item price, size M/L, price-change windows.

Rulebook (§3, §6, §10): quantities are counted in the item's *base unit* (a line sold in a large unit, e.g. a
"thùng" of 24 "lon", counts 24); "bán chạy" ranks by that quantity over orders that are neither cancelled nor
returned; "Phí phục vụ" is not an item; a cafe's size M/L is any item whose name ends in "(M)" / "(L)".
In the dev-300 run the model summed raw `Số lượng` (units mixed), filtered on the line's unit, or searched names
without diacritics; here it is computed in code and validated against the training answers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from datamate.data.loader import Business
from datamate.data.readme import parse_conversions
from datamate.data.status import find_order_col
from datamate.engines.sales import (
    IS_SALES,
    NON_ITEM,
    NOT_SALES,
    _nm,
    _overlap,
    _pick_amount,
    _pick_date,
    _table_roles,
    enrich_from_orders,
    returned_orders,
)
from datamate.text import fold, nodia

# Column-name patterns run on a normalised name (no diacritics / spaces / punctuation): "Số lượng quy đổi" -> "soluongquydoi",
# "SLQuyDoi" -> "slquydoi", "Quy ra ĐV lẻ" -> "quyradvle". README roles take precedence; these are the fallback.
ITEM_NAME = re.compile(r"^(ten)?(hang|sanpham|mon|sp|goi)(hoa|ban)?$|^noidung$|^mathang$|chitiethang|^tengoi$")
ITEM_CODE = re.compile(r"^(ma|id)(mat)?(hang|sanpham|sku|sp|hh|hhoa)(hoa|ban)?$|^sku$|^idsp$|^mahh$")
QTY_BASE = re.compile(r"quydoi|quyra|coban|^le$|soluongle|slle|dvle|slco")
NOT_QTY = re.compile(r"^gia|price|tien|thanh|^tri|von|phi|giam|thue|tile")  # a price / amount is never a base quantity
QTY = re.compile(r"^(soluong|sl|slg|sluong)(ban|hang|sanpham)?$")
UNIT = re.compile(r"^(donvi|donvitinh|dvt|dv|dvtinh|unit|dvi)$")
UNIT_PRICE = re.compile(r"^(dongia|dongiaban|gia|gia1dv|giale|giaban|giabanle|price|giadonvi|gianiemyet|gia1donvi|dongiad|giasanpham)$")
def cn(c: str) -> str:
    return re.sub(r"[\W_]+", "", nodia(str(c)).lower())


SERVICE_FEE = re.compile(r"^\s*ph[íi]\s*ph[ụu]c\s*v[ụu]\b", re.I)
SIZE = re.compile(r"\(\s*([ml])\s*\)\s*$", re.I)




def _num(s: pd.Series) -> pd.Series:
    """Quantities may be text such as '2 đĩa' or '1,5 kg'."""
    if s.dtype.kind in "if":
        return s.astype(float)
    t = s.astype(str).str.extract(r"^\s*(-?\d+(?:[.,]\d+)?)")[0].str.replace(",", ".", regex=False)
    return pd.to_numeric(t, errors="coerce")


@dataclass
class ItemView:
    lines: pd.DataFrame  # order, date, valid, item, key, code, qty, qty_base, unit, base_unit, amount, unit_price, is_item
    table: str
    note: str
    aliases: dict = field(default_factory=dict)   # folded short form -> folded full name (an abbreviation sheet, data/aliases.py)

    def _sel(self, start, end, *, valid: bool = True) -> pd.DataFrame:
        d = self.lines
        m = d["is_item"] & (d["date"] >= pd.Timestamp(start)) & (d["date"] < pd.Timestamp(end) + pd.Timedelta(days=1))
        return d[m & d["valid"]] if valid else d[m]

    def qty(self, item: str, start, end) -> float:
        s = self._sel(start, end)
        return float(s.loc[s["key"] == fold(item), "qty_base"].sum())

    def revenue(self, item: str, start, end) -> int:
        s = self._sel(start, end)
        return int(round(s.loc[s["key"] == fold(item), "amount"].sum()))

    def base_unit(self, item: str) -> str | None:
        u = self.lines.loc[self.lines["key"] == fold(item), "base_unit"].dropna()
        return u.iloc[0] if len(u) else None

    def top(self, k: int, start, end, by: str = "qty") -> list[tuple[str, float]]:
        s = self._sel(start, end)
        col = "qty_base" if by == "qty" else "amount"
        g = s.groupby("key").agg(v=(col, "sum"), code=("code", "first")).reset_index()
        g["name"] = g["key"].map(self.display).fillna(g["key"])
        g["code"] = g["code"].where(g["code"].notna() & (g["code"] != "None"), g["name"]).astype(str)
        g = g.sort_values(["v", "code"], ascending=[False, True])  # equal values: ascending by code
        return [(r.name, float(r.v)) for r in g.head(k).itertuples()]

    @property
    def display(self) -> dict[str, str]:
        """key -> the spelling to show: the most frequent one among those with diacritics (the data has dirty variants)."""
        if not hasattr(self, "_display"):
            d = self.lines
            vc = d.groupby(["key", "item"]).size().reset_index(name="n")
            vc["clean"] = vc["item"].astype(str).map(lambda x: x == re.sub(r"\s+", " ", x).strip() and x != x.upper() and x != nodia(x))
            vc = vc.sort_values(["key", "clean", "n"], ascending=[True, False, False])
            self._display = dict(zip(vc.drop_duplicates("key")["key"], vc.drop_duplicates("key")["item"]))
        return self._display

    def size_qty(self, size: str, start, end) -> float:
        s = self._sel(start, end)
        m = s["item"].astype(str).map(lambda x: (SIZE.search(x).group(1).lower() == size.lower()) if SIZE.search(x) else False)
        return float(s.loc[m, "qty_base"].sum())

    def price_on(self, item: str, day) -> float | None:
        """Unit price per base unit, before the line discount, on the lines of orders placed that day (all statuses)."""
        d = self.lines
        s = d[(d["key"] == fold(item)) & (d["date"].dt.normalize() == pd.Timestamp(day).normalize())]
        s = s[(s["qty"] > 0) & (s["qty_base"] > 0) & s["unit_price"].notna()]
        if s.empty:
            return None
        per_base = (s["unit_price"] * s["qty"] / s["qty_base"]).round(2)
        return float(per_base.mode().iloc[0])

    def change(self, item: str, day, n: int) -> dict:
        """Quantity in the n days from `day` (day included) vs the n days right before it."""
        day = pd.Timestamp(day)
        after = self.qty(item, day, day + pd.Timedelta(days=n - 1))
        before = self.qty(item, day - pd.Timedelta(days=n), day - pd.Timedelta(days=1))
        return {"before": before, "after": after, "pct": round((after - before) / before * 100, 2) if before else None}


def _conversions(biz: Business, roles: dict[str, dict[str, str]]) -> tuple[dict[tuple[str, str], float], dict[str, str]]:
    """({(item, large unit): ratio}, {item: base unit}) from conversion tables and README conversion lines."""
    conv: dict[tuple[str, str], float] = {}
    base: dict[str, str] = {}
    for t, df in biz.tables.items():
        r = roles.get(t, {})
        if {"item_name", "large_unit", "conv_ratio"} <= r.keys():
            for row in df[[r["item_name"], r["large_unit"], r["conv_ratio"]] + ([r["base_unit"]] if "base_unit" in r else [])].itertuples(index=False):
                try:
                    conv[(fold(row[0]), fold(row[1]))] = float(row[2])
                except (TypeError, ValueError):
                    continue
                if len(row) > 3 and pd.notna(row[3]):
                    base[fold(row[0])] = str(row[3])
        elif {"item_name", "base_unit"} <= r.keys():  # catalog: item -> base unit
            for a, b in df[[r["item_name"], r["base_unit"]]].dropna().itertuples(index=False):
                base.setdefault(fold(a), str(b))
    for name, big, ratio, small in parse_conversions(biz.readme):
        conv.setdefault((fold(name), fold(big)), ratio)
        base.setdefault(fold(name), small)
    return conv, base


def build_items(biz: Business) -> ItemView | None:
    tables, roles = biz.tables, _table_roles(biz)
    best = None
    for n, df in tables.items():
        r = roles.get(n, {})
        if NOT_SALES.search(_nm(n)) and "qty" not in r and "qty_base" not in r:
            continue
        item = r.get("item_name") or next((c for c in df.columns if ITEM_NAME.search(cn(c)) and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))), None)
        code = r.get("item_code") or next((c for c in df.columns if ITEM_CODE.search(cn(c))), None)
        qb = r.get("qty_base") or next((c for c in df.columns if QTY_BASE.search(cn(c)) and not NOT_QTY.search(cn(c)) and df[c].dtype.kind in "if"), None)
        q = r.get("qty") or next((c for c in df.columns if QTY.search(cn(c))), None)
        if not (item or code) or not (qb or q) or NOT_SALES.search(_nm(n)):
            continue
        score = (bool(IS_SALES.search(_nm(n))), _pick_amount(df) is not None, find_order_col(df) is not None, len(df))
        if best is None or score > best[0]:
            best = (score, n, item, code, qb, q)
    if best is None:
        return None
    _, n, item, code, qb, q = best
    df = tables[n].copy()
    r = roles.get(n, {})
    # item names written short in places: count by the full name the abbreviation sheet gives (data/aliases.py)
    exp = next((e for e in getattr(biz, "expansions", []) if e.table == n and e.column == item and e.full in df.columns), None)
    if exp is not None:
        item = exp.full
    oc = r.get("order_id") or find_order_col(df)
    order_key = oc
    if oc is None:
        df["__row"] = range(len(df))
        oc = "__row"
    dcol = r.get("order_date") or _pick_date(df)
    df, dcol = enrich_from_orders(df, n, order_key, dcol, tables, roles)
    if dcol is None:
        return None

    # item names: from a catalog when the line table only has codes (best-overlapping small table with a name column)
    if item is None:
        best_cat = None
        for n2, d2 in tables.items():
            if n2 == n or len(d2) >= 20000:
                continue
            nc = roles.get(n2, {}).get("item_name") or next((c for c in d2.columns if ITEM_NAME.search(cn(c)) and not ITEM_CODE.search(cn(c))), None)
            if nc is None:
                continue
            for cc in d2.columns:
                if cc == nc or d2[cc].dtype.kind in "fb":
                    continue
                ov = _overlap(df[code], d2[cc])
                if ov >= 0.8 and (best_cat is None or ov > best_cat[0]):
                    best_cat = (ov, cc, nc, n2)
        if best_cat is not None:
            _, cc, nc, n2 = best_cat
            df = df.merge(tables[n2][[cc, nc]].drop_duplicates(cc).rename(columns={nc: "__name", cc: "__code"}), left_on=code, right_on="__code", how="left")
            item = "__name"
        else:
            return None

    conv, base_map = _conversions(biz, roles)
    qty = _num(df[q]) if q else pd.Series(float("nan"), index=df.index)
    unit_col = r.get("unit") or next((c for c in df.columns if UNIT.search(cn(c))), None)
    names = df[item].astype(str)
    key = names.map(fold)
    if qb:
        qty_base = df[qb].astype(float)
        if q is not None:
            qty = qty.fillna(qty_base)
    else:
        ratio = pd.Series(1.0, index=df.index)
        if unit_col and conv:
            ratio = pd.Series([conv.get((k, fold(u)), 1.0) for k, u in zip(key, df[unit_col].astype(str))], index=df.index)
        qty_base = qty * ratio
    # base unit: catalog / conversion table / README, else the line unit that needs no conversion
    base_unit = key.map(base_map)
    if unit_col:
        plain = (qty_base - qty).abs() < 1e-9
        mode = df.loc[plain].groupby(key[plain])[unit_col].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None)
        base_unit = base_unit.fillna(key.map(mode))
    amt = r.get("line_amount") or _pick_amount(df)
    price = r.get("unit_price") or next((c for c in df.columns if UNIT_PRICE.search(cn(c)) and df[c].dtype.kind in "if"), None)

    valid = df["valid_order"].fillna(True) if "valid_order" in df.columns else pd.Series(True, index=df.index)
    valid = valid & ~df[oc].astype(str).isin(returned_orders(tables))
    text_cols = [c for c in df.columns[:14] if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    non_item = pd.Series(False, index=df.index)
    for c in text_cols:
        non_item |= df[c].astype(str).str.match(NON_ITEM)
    is_item = ~non_item & ~names.str.match(SERVICE_FEE) & df[item].notna() & names.str.strip().ne("") & ~names.isin(["nan", "<NA>", "None"])
    lines = pd.DataFrame({
        "order": df[oc].astype(str).values, "date": df[dcol].values, "valid": valid.values, "item": names.values, "key": key.values,
        "code": df[code].astype(str).values if code else None, "qty": qty.values, "qty_base": qty_base.values,
        "unit": df[unit_col].values if unit_col else None, "base_unit": base_unit.values,
        "amount": df[amt].values if amt else float("nan"), "unit_price": df[price].values if price else float("nan"),
        "is_item": is_item.values,
    })
    how = "conversion column" if qb else ("conversion table/README" if conv else "no conversion needed")
    aliases = {k: fold(v) for k, v in exp.to_full.items() if k != fold(v)} if exp is not None else {}
    return ItemView(lines, n, f"table `{n}`, item name = `{item}`, quantity = `{qb or q}` ({how})", aliases)


def _codes_from_names(lines: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """Sales lines that carry only a product NAME (no code column) get `code` from the product table, matched on the
    folded name. Names in sales lines are spelled differently from the product list (case, accents, spacing: biz_121 had
    82 of 3072 lines), so a hand-written `merge(on=name)` loses those lines and the drug / group flags after it."""
    if "key" not in lines.columns or lines["code"].notna().mean() >= 0.8:
        return lines
    keys = set(lines["key"].dropna())
    if not keys:
        return lines
    for df in tables.values():
        for name_col in df.columns:
            if not pd.api.types.is_string_dtype(df[name_col]):
                continue
            folded = df[name_col].dropna().astype(str).map(fold)
            if folded.empty or not folded.is_unique or len(keys & set(folded)) < 0.8 * len(keys):
                continue
            code_cols = [c for c in df.columns if c != name_col and df[c].notna().all() and df[c].astype(str).is_unique]
            if not code_cols:
                continue
            code_col = next((c for c in code_cols if re.search(r"(^|\b)(ma|mã|sku|code|id)\b", fold(c))), code_cols[0])
            by_key = pd.Series(df.loc[folded.index, code_col].astype(str).values, index=folded.values)
            return lines.assign(code=lines["key"].map(by_key))
    return lines


def with_item_attrs(lines: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """`lines` plus every column of the product tables, joined on item code. A product table is any table with a
    unique column holding >= 80% of the codes sold (product list, drug list, stock report): its other columns
    (group, prescription flag, category...) become line columns, so "doanh thu nhóm bia" is a filter on item_lines.
    On holdout6 the coder abstained on 3 such questions; this join gives the gold value for all 3."""
    if lines is None or "code" not in lines.columns:
        return lines
    lines = _codes_from_names(lines, tables)
    codes = set(lines["code"].dropna().astype(str))
    out = lines
    if not codes:
        return out
    for df in tables.values():
        for col in df.columns:
            v = df[col].dropna().astype(str)
            if len(v) == len(df) and v.is_unique and len(codes & set(v)) >= 0.8 * len(codes):
                extra = [c for c in df.columns if c != col and c not in out.columns]
                if extra:
                    key = out["code"].astype(str)
                    out = out.assign(**{c: key.map(pd.Series(df[c].values, index=v.values)) for c in extra})
                break
    return out
