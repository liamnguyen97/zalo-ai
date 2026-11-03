"""Customer debt (rulebook §4) computed in code, as functions the coder can call.

Debt questions were the coder's worst topic (0 of 25 right when measured): §4 has several exact rules that a model
rewrites differently each time (payments made on the day of purchase are not debt repayments; returned orders still
count; COD and marketplace orders are never debt; age is counted from the order date; walk-in orders count in the total
but have no row per customer).

Layout: the README says where debt lives, in one standard sentence of its "Quy ước" section:
    **Công nợ**: mỗi lần khách trả tiền cho một đơn là một dòng, **<pay col>** là số tiền trả lần đó;
    **<total col>** là số khách phải trả cho cả đơn.
(columns may also be written `table.column`). The orders table holds <total col>, the payments table <pay col>; ids,
dates, customer and payment method are found from the data (value overlap, column types), not from table names.
Without that sentence there is no debt view, and the coder works from the tables alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from datamate.answer.formatter import _clean_name
from datamate.data.loader import Business
from datamate.data.readme import roles_for_tables
from datamate.data.status import PAYMENT_COL, find_order_col
from datamate.text import fold

DEBT_LINE = re.compile(r"\*\*C[ôo]ng n[ợo]\*\*\s*:(.*)$", re.I | re.M)
NAMED = re.compile(r"\*\*([^*]+)\*\*|`([^`]+)`")
CREDIT = {"ghi no", "debt", "cong no", "no", "tra sau", "mua chiu", "credit"}
NOT_DEBT_CHANNEL = re.compile(r"cod|thu ho|shopee|lazada|tiki|tiktok|\bsan\b|tmdt|ecommerce|marketplace")
METHOD_COL = PAYMENT_COL   # one definition with data/status.py (it also reads "Thanh toán bằng"); the README role `payment_method` comes first
CHANNEL_COL = re.compile(r"k[êe]nh|ngu[ồo]n|b[áa]n qua|lo[ạa]i b[áa]n|channel|kenhbanhang|source", re.I)
NAME_COL = re.compile(r"t[êe]n|name|kh[áa]ch h[àa]ng|^kh$|tengoi", re.I)
DATE_COL = re.compile(r"th[ờo]i|ng[àa]y|date|time|gi[ờo]", re.I)
ITEM_HDR = re.compile(r"h[àa]ng h[óo]a|t[êe]n h[àa]ng|m[ặa]t h[àa]ng|t[êe]n m[óo]n|\\bm[óo]n\\b|s[ảa]n ph[ẩa]m|\\bsp\\b|item|product|d[ịi]ch v[ụu]|barcode", re.I)
CUSTOMER_HINT = re.compile(r"kh[áa]ch|khach|\bkh\b|m[ãa] ?kh|customer|client|b[êe]n mua|ng[ườu]+i mua|buyer", re.I)
WALK_IN = {"khach le", "le", "kl", "khach vang lai", "vang lai", "walk-in", "walk in", ""}


def _ids(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _day(s: pd.Series) -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, errors="coerce")
    return s.dt.normalize()


def _date_col(df: pd.DataFrame, avoid: tuple[str, ...] = ()) -> str | None:
    cands = [c for c in df.columns if c not in avoid and (pd.api.types.is_datetime64_any_dtype(df[c]) or DATE_COL.search(c))]
    best = None
    for c in cands:
        ok = _day(df[c]).notna().mean()
        if ok > 0.9 and (best is None or ok > best[0]):
            best = (ok, c)
    return best[1] if best else None


def _overlap(a: pd.Series, b: pd.Series) -> float:
    va, vb = set(_ids(a.dropna()).head(5000)), set(_ids(b.dropna()))
    return len(va & vb) / len(va) if va else 0.0


def _text_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_datetime64_any_dtype(df[c]) and df[c].dtype.kind not in "iufb"]


def _named_cols(readme: str) -> tuple[str, str] | None:
    m = DEBT_LINE.search(readme or "")
    if not m:
        return None
    names = [(a or b).split(".")[-1].strip() for a, b in NAMED.findall(m.group(1))]
    return (names[0], names[1]) if len(names) >= 2 else None


@dataclass
class DebtView:
    orders: pd.DataFrame    # order, day, total, credit, debt_ok, cust, name
    payments: pd.DataFrame  # order, day, amount
    source: str
    flows = True            # dated payments exist: repayments / new debt of a period can be computed

    # ---- balances ------------------------------------------------------------------------------------------------
    def balances(self, day) -> pd.DataFrame:
        """Orders still owing at the end of `day`: order, day, cust, name, owed (> 0)."""
        d = pd.Timestamp(day).normalize()
        o = self.orders[self.orders["debt_ok"] & (self.orders["day"] <= d)]
        paid = self.payments[self.payments["day"] <= d].groupby("order")["amount"].sum()
        owed = o["total"] - o["order"].map(paid).fillna(0)
        out = o.assign(owed=owed)
        return out[out["owed"] > 0.5]

    def total_owed(self, day, older_than_days: int | None = None) -> int:
        b = self.balances(day)
        if older_than_days is not None:
            b = b[(pd.Timestamp(day).normalize() - b["day"]).dt.days > older_than_days]
        return int(round(b["owed"].sum()))

    def by_customer(self, day) -> list[tuple[str, int]]:
        b = self.balances(day)
        b = b[b["cust"] != ""]
        g = b.groupby("cust").agg(owed=("owed", "sum"), name=("name", "first")).sort_values("owed", ascending=False, kind="stable")
        return [(n, int(round(v))) for n, v in zip(g["name"], g["owed"])]

    def aging(self, day, edges=(7, 15)) -> list[int]:
        """Owed money by age bucket: [0..e1], [e1+1..e2], ..., > last edge (age = days since the order date)."""
        b = self.balances(day)
        age = (pd.Timestamp(day).normalize() - b["day"]).dt.days
        out, lo = [], 0
        for e in edges:
            out.append(int(round(b.loc[(age >= lo) & (age <= e), "owed"].sum())))
            lo = e + 1
        out.append(int(round(b.loc[age >= lo, "owed"].sum())))
        return out

    # ---- flows in a period ----------------------------------------------------------------------------------------
    def _credit(self, customer: str | None) -> pd.DataFrame:
        o = self.orders[self.orders["credit"] & self.orders["debt_ok"]]
        if customer is not None:
            o = o[o["cust"] == self._cust_key(customer)]
        return o

    def repaid(self, start, end, customer: str | None = None) -> int:
        """Debt repayments: payments AFTER the purchase day, dated in [start, end], on credit orders (any order date)."""
        o = self._credit(customer)
        p = self.payments.merge(o[["order", "day"]].rename(columns={"day": "oday"}), on="order")
        p = p[(p["day"] > p["oday"]) & (p["day"] >= pd.Timestamp(start)) & (p["day"] <= pd.Timestamp(end))]
        return int(round(p["amount"].sum()))

    def new_debt(self, start, end, customer: str | None = None) -> int:
        """New debt: unpaid part, at the end of the purchase day, of credit orders placed in [start, end]."""
        o = self._credit(customer)
        o = o[(o["day"] >= pd.Timestamp(start)) & (o["day"] <= pd.Timestamp(end))]
        same = self.payments.merge(o[["order", "day"]], on=["order", "day"]).groupby("order")["amount"].sum()
        return int(round((o["total"] - o["order"].map(same).fillna(0)).clip(lower=0).sum()))

    def _cust_key(self, customer: str) -> str:
        c = fold(customer)
        o = self.orders
        hit = o.loc[(o["cust"].map(fold) == c) | (o["name"].map(fold) == c), "cust"]
        if hit.empty:
            names = sorted(set(o.loc[o["cust"] != "", "name"]))
            close = [n for n in names if c and c in fold(n)][:5]
            raise ValueError(f"no customer named {customer!r}; similar names in the data: {close}")
        return hit.iloc[0]


PERSONISH = re.compile(r"^[^\W\d_]+(?:\s+[^\W\d_]+)+")
PERSON_FULL = re.compile(r"[^\W\d_]+(?:\s+[^\W\d_]+){1,5}")
CUST_HDR = re.compile(r"kh[áa]ch|customer|^kh\b|tengoi", re.I)


def _name_col(df: pd.DataFrame, avoid: str) -> str | None:
    """The column holding customer names, judged by content (headers vary: `TenGoi`, `Khách lẻ/sỉ`, ...): mostly
    distinct values that start with two or more words, not codes, phones or addresses."""
    best = None
    for c in _text_cols(df):
        if c == avoid or ITEM_HDR.search(str(c)):      # biz_130: the item-name column of the sales lines passed for people's names
            continue
        v = df[c].dropna().astype(str).str.strip()
        if len(v) == 0:
            continue
        score = v.str.match(PERSONISH).mean() * min(1.0, v.nunique() / max(1, len(v)) / 0.5)
        if score <= 0.6:
            # one row per ORDER: repeat customers make names look non-unique. Judge the distinct names instead: people's
            # names are 2-6 words of letters (phone tails removed), and there are many of them.
            dv = pd.Series(v[~v.map(fold).isin(WALK_IN)].unique()).map(_clean_name)
            if len(dv) >= 20 and len(dv) >= 0.002 * len(v):
                score = dv.str.fullmatch(PERSON_FULL).mean() * (1.0 if CUST_HDR.search(c) else 0.9)
        score += 0.05 if NAME_COL.search(c) else 0
        if score > 0.6 and (best is None or score > best[0]):
            best = (score, c)
    return best[1] if best else None


def _declared_code(biz: Business, octable: str, orders: pd.DataFrame) -> str | None:
    """The column of the order table that the README documents as the customer code ("mã khách (trống nếu khách lẻ)"), when the orders
    have it with at least two codes. dev_007 / dev_008 record the code on the order itself (190 / 210 customers); keyed by name and phone
    they had 767 / 844 "customers", and "Th5 mấy khách mua từ 9 đơn" was 2 instead of 62."""
    df = biz.tables.get(octable)
    if df is None:
        return None
    col = roles_for_tables(biz.readme, {octable: list(df.columns)}).get(octable, {}).get("customer_code")
    if col is None or col not in orders.columns:
        return None
    return col if _ids(orders[col].dropna()).replace("", pd.NA).dropna().nunique() >= 2 else None


# "không ghi tên khách", "không ghi khách": the shop's README says its book records no customer (dev_012's repair book; without this
# the column of jobs done, `Làm gì`, passed for 3,188 customers' names)
NO_CUSTOMER = re.compile(r"kh[ôo]ng (?:ghi|c[óo]|l[ưu]u) (?:t[êe]n )?kh[áa]ch(?! (?:l[ẻe]|h[àa]ng l[ẻe]))", re.I)


def _customer(biz: Business, orders: pd.DataFrame, octable: str) -> tuple[pd.Series, pd.Series]:
    """(customer key, display name) per order row. Shops with customer codes: one customer per code, name from the
    customer list (or the order's name column). Shops without codes: the name is the key, spelling-folded (§3). A shop whose
    README says it records no customer has none: every order is a walk-in."""
    if NO_CUSTOMER.search(biz.readme or ""):
        blank = pd.Series("", index=orders.index)
        return blank, blank
    texts = _text_cols(orders)
    declared = _declared_code(biz, octable, orders)
    if declared is not None:
        texts = [declared] + [c for c in texts if c != declared]      # the README's column first
    code_col, name_map = None, {}

    def search(need_hint: bool):
        for t, df in biz.tables.items():
            if t == octable or len(df) == 0 or len(df) > 50_000:
                continue
            for kc in _text_cols(df):
                vals = _ids(df[kc].dropna())
                if vals.nunique() < max(20, 0.9 * len(vals)):
                    continue
                for c in texts:
                    if need_hint and not (CUSTOMER_HINT.search(str(c)) or CUSTOMER_HINT.search(str(kc)) or CUSTOMER_HINT.search(t)):
                        continue
                    if (orders[c].nunique() >= 2 and orders[c].nunique() < 0.5 * len(orders) + 1
                            and _overlap(orders[c].dropna().head(5000), df[kc]) > 0.8):
                        nc = _name_col(df, kc)
                        if nc is not None:
                            return c, dict(zip(_ids(df[kc]), df[nc].astype(str).str.strip()))
        return None, {}
    # value overlap alone also matches a product barcode column to the product list: dev_003 had every "customer" a product, and
    # biz_130 / dev_007 / dev_008 (shops that record no customer) got 60 "customers" that were the 60 products, so dev300
    # 05a8b3 (gold unanswerable) was answered 98.33. Only a pair whose column / key / table name speaks of customers counts;
    # without one the shop has no customer codes (the order's own name column is tried next) and its orders are walk-ins
    code_col, name_map = search(True)
    if declared is not None and code_col != declared:
        code_col, name_map = declared, {}          # no customer list keyed by it: the names come from the order's own name column
    own_name = _name_col(orders.head(20000), code_col or "")
    if code_col is not None:
        key = _ids(orders[code_col]).where(orders[code_col].notna(), "")
        key = key.where(~key.map(fold).isin(WALK_IN | {"nan", "none"}), "")
        name = key.map(name_map)
        if own_name is not None:
            name = name.fillna(orders[own_name].astype(str).str.strip())
        return key, name.fillna(key)
    if own_name is not None:
        raw = orders[own_name].fillna("").astype(str).str.strip()
        key = raw.map(fold)
        key = key.where(~key.isin(WALK_IN | {"nan", "none"}), "")
        return key, raw
    blank = pd.Series("", index=orders.index)
    return blank, blank


def build_debt(biz: Business) -> DebtView | None:
    named = _named_cols(biz.readme)
    if named is None:
        return None
    pay_col, total_col = named
    ot = next((t for t, d in biz.tables.items() if total_col in d.columns), None)
    pt = next((t for t, d in biz.tables.items() if pay_col in d.columns and t != ot), None)
    if ot is None or pt is None:
        return None
    o, p = biz.tables[ot], biz.tables[pt]
    oc = find_order_col(o)
    if oc is None:
        return None
    o = o.assign(**{oc: _ids(o[oc])}).drop_duplicates(oc)        # line-level order tables: one row per order
    link = max(_text_cols(p), key=lambda c: _overlap(p[c], o[oc]), default=None)
    if link is None or _overlap(p[link], o[oc]) < 0.8:
        return None
    odate, pdate = _date_col(o), _date_col(p)
    if odate is None or pdate is None:
        return None
    declared = roles_for_tables(biz.readme, {ot: list(o.columns)}).get(ot, {}).get("payment_method")
    texts = _text_cols(o)
    method = declared if declared in texts else next((c for c in texts if METHOD_COL.search(c)), None)
    channel = next((c for c in _text_cols(o) if CHANNEL_COL.search(c)), None)
    mfold = o[method].map(fold) if method else pd.Series("", index=o.index)
    cfold = o[channel].map(fold) if channel else pd.Series("", index=o.index)
    status = o["status_std"] if "status_std" in o.columns else pd.Series("completed", index=o.index)
    cust, name = _customer(biz, o, ot)
    orders = pd.DataFrame({
        "order": o[oc].values, "day": _day(o[odate]).values, "total": pd.to_numeric(o[total_col], errors="coerce").fillna(0).values,
        "credit": mfold.isin(CREDIT).values,
        # §4: cancelled orders are not debt (returned ones still are); COD and marketplace orders are never debt
        "debt_ok": ((status != "cancelled") & ~mfold.str.contains(NOT_DEBT_CHANNEL) & ~cfold.str.contains(NOT_DEBT_CHANNEL)).values,
        "cust": cust.values, "name": name.astype(str).map(_clean_name).values})   # names without phone tails
    payments = pd.DataFrame({"order": _ids(p[link]).values, "day": _day(p[pdate]).values,
                             "amount": pd.to_numeric(p[pay_col], errors="coerce").fillna(0).values})
    return DebtView(orders.dropna(subset=["day"]), payments.dropna(subset=["day"]),
                    f"đơn `{ot}` (tổng phải trả `{total_col}`, ngày `{odate}`), phiếu thu `{pt}` (tiền `{pay_col}`, ngày `{pdate}`)")


# ---------------------------------------------------------------------------------------------- debt recorded on the order itself
# Shops with no payment records keep what each order still owes on the order: a column (`Còn nợ`, `Nợ`: "số khách còn nợ của đơn",
# dev_007 / dev_009 and 4 training shops) or the order's note ("đơn ghi nợ thì ghi khách đã đưa / còn thiếu bao nhiêu", dev_008:
# "ghi nợ 1.038.500đ, đã trả đủ", "ck 913 nghìn, thiếu lại 456.500đ", "ghi sổ 443 nghìn"). The README says it is the state at the end of
# the data. So the balance is known on the last day of the data (or later) and not before it: `balances` refuses an earlier day, and
# there is no repayment / new-debt flow (no dated payment exists).
OWED_DESC = re.compile(r"^(?:s[ốo] (?:ti[ềe]n )?)?(?:kh[áa]ch )?c[òo]n n[ợo]\b|^n[ợo] c[òo]n l[ạa]i\b|^d[ưu] n[ợo]\b", re.I)
NOTE_DEBT_DESC = re.compile(r"ghi n[ợo].*(?:c[òo]n thi[ếe]u|đ[ãa] đ[ưu]a|c[òo]n n[ợo])|(?:c[òo]n thi[ếe]u|c[òo]n n[ợo]).*ghi ch[úu]", re.I)
_AMOUNT = r"(\d+(?:[.,]\d{3})+(?:\s*(?:đ|d|vnd|dong))?|\d+\s*tr\s*\d*|\d+(?:[.,]\d+)?\s*(?:k|nghin|ngan|tr|trieu|đ|d)\b|\d{4,})"
PAID_FULL = re.compile(r"(?:da )?(?:tra|ck|chuyen khoan|thanh toan|dua) (?:du|het|xong|ca)\b|\bdu roi\b|\bhet roi\b")
STILL_OWED = re.compile(rf"(?:con thieu|thieu lai|con no|no lai|thieu|con lai)\s*{_AMOUNT}")
OWED_AMOUNT = re.compile(rf"(?:ghi no|no|ghi so|chiu)\s*{_AMOUNT}")
PAID_PART = re.compile(rf"(?:dua truoc|tra truoc|dua|tra|ck|chuyen khoan)\s*{_AMOUNT}")
WHOLE_ORDER = re.compile(r"\bno ca don\b|\bghi so ca don\b|\bchua tra\b(?!.*\d)|^\s*(?:ghi no|ghi so|no)\s*$")


def _money(text: str) -> float | None:
    from datamate.data.loader import MONEY_K, MONEY_TR, MONEY_TR_DEC
    t = re.sub(r"\s*(?:đ|d|vnd|dong)\s*$", "", text.strip())
    t = re.sub(r"\s*(?:nghin|ngan)$", "k", t)
    t = re.sub(r"\s*trieu$", "tr", t)
    for pat, mul in ((MONEY_TR_DEC, 1e6), (MONEY_K, 1e3)):
        if (m := pat.fullmatch(t)) and not re.search(r"tr\s*\d", t):
            return float(m.group(1).replace(",", ".")) * mul
    if (m := MONEY_TR.fullmatch(t)):
        return float(f"{m.group(1)}.{m.group(2) or '0'}") * 1e6
    digits = re.sub(r"[.,]", "", t)
    return float(digits) if digits.isdigit() else None


def owed_from_note(note, total: float) -> float | None:
    """What the order still owes, read from its note (folded); None when the note says nothing about it."""
    if note is None or (isinstance(note, float) and pd.isna(note)):
        return None
    t = fold(str(note))
    if m := STILL_OWED.search(t):
        return _money(m.group(1))
    if PAID_FULL.search(t):
        return 0.0
    if m := OWED_AMOUNT.search(t):
        return _money(m.group(1))
    if re.search(r"con lai (?:ghi so|ghi no|no)", t) and (m := PAID_PART.search(t)) and (paid := _money(m.group(1))) is not None:
        return max(total - paid, 0.0)
    if WHOLE_ORDER.search(t):
        return float(total)
    return None


class SnapshotDebt(DebtView):
    """Debt known only as each order's state at the end of the data (`last`)."""
    flows = False

    def __init__(self, orders: pd.DataFrame, source: str, last: pd.Timestamp):
        super().__init__(orders, pd.DataFrame({"order": [], "day": pd.to_datetime([]), "amount": []}), source)
        self.last = pd.Timestamp(last).normalize()

    def balances(self, day) -> pd.DataFrame:
        if pd.Timestamp(day).normalize() < self.last:
            raise ValueError(f"this shop records what each order still owes only as of the end of its data ({self.last.date()}); "
                             f"the balance on {pd.Timestamp(day).date()} is not recorded")
        return super().balances(day)

    def repaid(self, start, end, customer=None) -> int:
        raise ValueError("this shop records no dated payments: repayments in a period are not recorded")

    def new_debt(self, start, end, customer=None) -> int:
        raise ValueError("this shop records no dated payments: new debt in a period is not recorded")


def _declared(biz: Business, pattern: re.Pattern) -> tuple[str, str] | None:
    """(table, column) whose README description matches `pattern`."""
    from datamate.data.readme import parse_column_docs, roles_for_tables as _rft  # noqa: F401 - same module family
    docs = [d for d in parse_column_docs(biz.readme) if pattern.search(d.desc)]
    for d in docs:
        for t, df in biz.tables.items():
            if d.column in df.columns:
                return t, d.column
    return None


def build_snapshot_debt(biz: Business) -> DebtView | None:
    """The debt view of a shop that writes what each order owes on the order (a column, or the note), as of the end of its data."""
    found, from_note = _declared(biz, OWED_DESC), False
    if found is None:
        found, from_note = _declared(biz, NOTE_DEBT_DESC), True
    if found is None:
        return None
    t, col = found
    o = biz.tables[t]
    oc = find_order_col(o)
    if oc is None:
        return None
    o = o.assign(**{oc: _ids(o[oc])}).drop_duplicates(oc)
    roles = roles_for_tables(biz.readme, {t: list(o.columns)}).get(t, {})
    odate = roles.get("order_date") if roles.get("order_date") in o.columns else _date_col(o)
    if odate is None:
        return None
    total_col = next((c for c in o.columns if re.search(r"t[ổo]ng (?:ti[ềe]n|thanh to[áa]n|c[ộo]ng)|tong tien|ph[ảa]i tr[ảa]|c[ầa]n tr[ảa]", str(c), re.I)
                      and pd.api.types.is_numeric_dtype(o[c])), None)
    total = pd.to_numeric(o[total_col], errors="coerce").fillna(0) if total_col else pd.Series(0.0, index=o.index)
    if from_note:
        owed = pd.Series([owed_from_note(n, tt) for n, tt in zip(o[col], total)], index=o.index, dtype="float64").fillna(0)
    else:
        owed = pd.to_numeric(o[col], errors="coerce").fillna(0)
    if not (owed > 0).any():
        return None
    declared = roles.get("payment_method")
    texts = _text_cols(o)
    method = declared if declared in texts else next((c for c in texts if METHOD_COL.search(c)), None)
    channel = next((c for c in texts if CHANNEL_COL.search(c)), None)
    mfold = o[method].map(fold) if method else pd.Series("", index=o.index)
    cfold = o[channel].map(fold) if channel else pd.Series("", index=o.index)
    status = o["status_std"] if "status_std" in o.columns else pd.Series("completed", index=o.index)
    cust, name = _customer(biz, o, t)
    days = _day(o[odate])
    orders = pd.DataFrame({
        "order": o[oc].values, "day": days.values, "total": owed.values, "credit": (owed > 0).values,
        "debt_ok": ((status != "cancelled") & ~mfold.str.contains(NOT_DEBT_CHANNEL) & ~cfold.str.contains(NOT_DEBT_CHANNEL)).values,
        "cust": cust.values, "name": name.astype(str).map(_clean_name).values})
    where = f"ghi chú `{col}`" if from_note else f"cột `{col}`"
    return SnapshotDebt(orders.dropna(subset=["day"]), f"nợ còn lại của mỗi đơn ở {where} của `{t}` (tình trạng tính đến hết ngày "
                        f"cuối dữ liệu, {days.max().date()}; ngày `{odate}`)", days.max())
