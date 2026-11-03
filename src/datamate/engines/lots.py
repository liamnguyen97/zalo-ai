"""Lots and expiry dates (rulebook §6, pharmacies): what each lot still holds on a day, and which lots expire in the next N days.

§6: "Hàng hết hạn trong N ngày tới: tính tại hôm nay D: lô còn tồn có hạn dùng từ D đến hết ngày D + N. Tiền vốn là giá trị phần còn
tồn của lô, theo giá nhập của chính lô đó quy về đơn vị cơ bản"; "Lô gần hết hạn nhất của một thuốc: trong các lô còn tồn chưa quá hạn
tại hôm nay, lô có hạn sớm nhất". README: "mỗi dòng nhập là một lô ... Một dòng bán có thể lấy từ nhiều lô. Hàng khách trả về lại lô
đã bán". Left to the Coder these were three joins (lots, sales by lot, purchase slips) and a remaining quantity; done once here.

Found by shape, not by table name:
  - LOTS: a table with a unique lot column, an expiry-date column and a quantity, and an item column the sales lines know;
  - SOLD BY LOT: a table whose lot column holds the lots' ids and whose order column holds the sales orders' ids, with the quantity
    taken from each lot (and the quantity returned to it);
  - the lot's import date: its own date, or its slip's date (a slip table keyed by the lots' slip id);
  - the lot's money: from purchase lines that name the lot (quantity x price, or their amount), else the lots table's price.
Remaining on day D = imported by D - taken by orders placed by D (cancelled orders take nothing) + returned to the lot.
This engine is CASE-SPECIFIC (one kind of shop keeps lots); it is offered only where all the tables above are found.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from datamate.data.loader import Business
from datamate.data.status import find_order_col
from datamate.text import fold, fold_text

LOT = re.compile(r"(?:^|\b)(?:ma |so )?lo(?: hang)?$|\blot\b|\bbatch\b|\bma lo\b|\bso lo\b")
EXPIRY = re.compile(r"\bhsd\b|\bhan su dung\b|\bhan dung\b|\bexpir|\bhet han\b")
BASE_QTY = re.compile(r"quy doi|co ban|\ble\b|ra le|base")
QTY = re.compile(r"\bsl\b|so luong|\bslg\b|qty|quantity")
RETURNED = re.compile(r"\btra\b")
PRICE = re.compile(r"gia nhap|don gia")
AMOUNT = re.compile(r"thanh tien|tien nhap")
IMPORT_DATE = re.compile(r"ngay nhap|nhap kho|ngay")
SLIP = re.compile(r"phieu|\bpn\b|so phieu")


def _w(c) -> str:
    t = str(c)
    t = "".join(" " + ch if i and ch.isupper() and t[i - 1].islower() else ch for i, ch in enumerate(t))
    return fold_text(t.replace("_", " "))


def _dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.normalize()
    return pd.to_datetime(s.astype(str).str.slice(0, 10).where(s.notna()), errors="coerce", dayfirst=True, format="mixed").dt.normalize()


def _num_col(df: pd.DataFrame, want: re.Pattern, prefer: re.Pattern | None = None, avoid: re.Pattern | None = None) -> str | None:
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and want.search(_w(c)) and not (avoid and avoid.search(_w(c)))]
    if prefer is not None:
        cols = sorted(cols, key=lambda c: not prefer.search(_w(c)))
    return cols[0] if cols else None


def _overlap(a: pd.Series, b: set) -> float:
    v = set(a.dropna().astype(str).head(5000))
    return len(v & b) / len(v) if v else 0.0


@dataclass
class LotView:
    lots: pd.DataFrame   # lot, item_key, imported, expiry, qty, per_base (money per base unit)
    taken: pd.DataFrame  # lot, odate, qty, back
    names: dict          # item key -> display name
    codes: dict          # item key -> code (for ties)
    source: str

    def remaining(self, day) -> pd.DataFrame:
        """Lots with what they still hold at the end of `day` (base units)."""
        d = pd.Timestamp(day).normalize()
        L = self.lots[self.lots["imported"].isna() | (self.lots["imported"] <= d)]
        t = self.taken[self.taken["odate"] <= d]
        out = t.groupby("lot")["qty"].sum()
        back = t.groupby("lot")["back"].sum()
        rem = L["qty"] - L["lot"].map(out).fillna(0) + L["lot"].map(back).fillna(0)
        return L.assign(remaining=rem.values)

    def expiring(self, day, n: int) -> pd.DataFrame:
        """§6: lots still holding stock at `day` whose expiry date is in [day, day + n]."""
        d = pd.Timestamp(day).normalize()
        r = self.remaining(d)
        r = r[(r["remaining"] > 0) & (r["expiry"] >= d) & (r["expiry"] <= d + pd.Timedelta(days=int(n)))]
        return r.assign(value=r["remaining"] * r["per_base"], code=r["item_key"].map(self.codes).fillna(r["item_key"]))

    def nearest(self, key: str, day) -> pd.Timestamp | None:
        d = pd.Timestamp(day).normalize()
        r = self.remaining(d)
        r = r[(r["item_key"] == key) & (r["remaining"] > 0) & (r["expiry"] >= d)]
        return None if r.empty else r["expiry"].min()


def build_lots(biz: Business, items, orders: pd.DataFrame | None) -> LotView | None:
    if items is None or orders is None or not {"order", "date"} <= set(orders.columns):
        return None
    il = items.lines
    codes = set(il["code"].dropna().astype(str)) - {"None", "nan"}
    keys = set(il["key"].dropna().astype(str))
    tables = biz.tables
    best = None
    for t, df in tables.items():
        lot = next((c for c in df.columns if LOT.search(_w(c)) and df[c].notna().mean() > 0.95 and df[c].astype(str).is_unique), None)
        exp = next((c for c in df.columns if EXPIRY.search(_w(c))), None)
        if lot is None or exp is None or _dates(df[exp]).notna().mean() < 0.9:
            continue
        item = next((c for c in df.columns if c not in (lot, exp) and codes and _overlap(df[c], codes) >= 0.8), None)
        by_name = item is None
        if by_name:
            item = next((c for c in df.columns if c not in (lot, exp) and _overlap(df[c].astype(str).map(fold), keys) >= 0.8), None)
        qty = _num_col(df, re.compile(BASE_QTY.pattern + "|" + QTY.pattern), BASE_QTY, re.compile(r"gia|tien"))
        if item is None or qty is None:
            continue
        # a lots table counts in base units ("SL nhập quy đổi"); purchase lines that also carry lot and expiry count in import units
        score = (bool(BASE_QTY.search(_w(qty))), len(df))
        if best is None or score > best[0]:
            best = (score, t, lot, exp, item, qty, by_name)
    if best is None:
        return None
    _, t, lot, exp, item, qty, by_name = best
    L = tables[t]
    lot_ids = set(L[lot].astype(str))
    order_ids = set(orders["order"].astype(str))
    sold = None
    for t2, df in tables.items():
        if t2 == t:
            continue
        lc = next((c for c in df.columns if _overlap(df[c], lot_ids) >= 0.8 and df[c].astype(str).duplicated().any()), None)
        oc = find_order_col(df)
        if lc is None or oc is None or _overlap(df[oc], order_ids) < 0.8:
            continue
        q = _num_col(df, re.compile(BASE_QTY.pattern + "|" + QTY.pattern), BASE_QTY, RETURNED)
        if q is None:
            continue
        back = next((c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and RETURNED.search(_w(c))), None)
        sold = (t2, lc, oc, q, back)
        break
    if sold is None:
        return None
    t2, lc, oc, q, back = sold
    S = tables[t2]
    o = orders.assign(order=orders["order"].astype(str)).drop_duplicates("order").set_index("order")
    status = o["status_std"] if "status_std" in o.columns else pd.Series("completed", index=o.index)
    oid = S[oc].astype(str)
    taken = pd.DataFrame({"lot": S[lc].astype(str).values, "odate": oid.map(o["date"]).dt.normalize().values,
                          "qty": pd.to_numeric(S[q], errors="coerce").fillna(0).values,
                          "back": (pd.to_numeric(S[back], errors="coerce").fillna(0) if back else pd.Series(0.0, index=S.index)).values,
                          "cancelled": oid.map(status).eq("cancelled").values})
    taken = taken[~taken["cancelled"]].drop(columns="cancelled")
    # the lot's import date: its own date column, else its slip's date
    imported = pd.Series(pd.NaT, index=L.index)
    own = next((c for c in L.columns if c != exp and pd.api.types.is_datetime64_any_dtype(L[c]) and IMPORT_DATE.search(_w(c))), None)
    if own is not None:
        imported = _dates(L[own])
    else:
        slip = next((c for c in L.columns if SLIP.search(_w(c)) and c not in (lot, item)), None)
        if slip is not None:
            for df in tables.values():
                kc = next((c for c in df.columns if df[c].astype(str).is_unique and _overlap(L[slip], set(df[c].astype(str))) >= 0.9), None)
                dc = next((c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) and IMPORT_DATE.search(_w(c))), None) if kc else None
                if kc and dc:
                    imported = L[slip].astype(str).map(df.assign(__k=df[kc].astype(str)).set_index("__k")[dc]).dt.normalize()
                    break
    # the lot's money: purchase lines naming the lot, else the lots table's own price (per base unit)
    base = pd.to_numeric(L[qty], errors="coerce")
    money = None
    for t3, df in tables.items():
        if t3 in (t, t2):
            continue
        lc3 = next((c for c in df.columns if _overlap(df[c], lot_ids) >= 0.8), None)
        if lc3 is None:
            continue
        amt = _num_col(df, AMOUNT)
        if amt is not None:
            m = pd.to_numeric(df[amt], errors="coerce")
        else:
            p, n3 = _num_col(df, PRICE), _num_col(df, QTY, None, re.compile(r"gia|tien"))
            if p is None or n3 is None:
                continue
            m = pd.to_numeric(df[p], errors="coerce") * pd.to_numeric(df[n3], errors="coerce")
        money = L[lot].astype(str).map(m.groupby(df[lc3].astype(str)).sum())
        break
    if money is not None and money.notna().mean() > 0.9:
        per_base = money / base
    else:
        price = _num_col(L, PRICE)
        per_base = pd.to_numeric(L[price], errors="coerce") if price else pd.Series(float("nan"), index=L.index)
    if by_name:
        item_key = L[item].astype(str).map(fold)
    else:
        code_key = il.dropna(subset=["code"]).assign(code=lambda x: x["code"].astype(str)).drop_duplicates("code").set_index("code")["key"]
        item_key = L[item].astype(str).map(code_key)
    lots = pd.DataFrame({"lot": L[lot].astype(str).values, "item_key": item_key.values, "imported": imported.values,
                         "expiry": _dates(L[exp]).values, "qty": base.values, "per_base": per_base.values})
    code_of = il.dropna(subset=["code"]).drop_duplicates("key").set_index("key")["code"].astype(str).to_dict()
    return LotView(lots.dropna(subset=["expiry", "item_key"]), taken, items.display, code_of,
                   f"lots `{t}` (lot `{lot}`, expiry `{exp}`, base qty `{qty}`), taken by orders in `{t2}` (`{q}`"
                   + (f", returned `{back}`" if back else "") + ")")
