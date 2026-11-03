"""Canonical order status: map each business's status labels onto the rulebook's four states.

The rulebook says every order has exactly one of: hoàn thành / huỷ / trả hàng / đang giao, but the
files spell them in dozens of ways ("Đã giao", "đã lấy", "ok", "Đổi trả", "Đang vận chuyển",
"completed", "cancelled", ...). In the dev-300 run the coder kept only the first label it saw
("completed"/"đã xong") and silently dropped in-delivery orders. `status_std` fixes that once, in code:

    completed, shipping  -> count as sales (rulebook: "Đơn đang giao vẫn tính")
    cancelled, returned  -> excluded (a partly returned order is a returned order: the whole order goes)
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# "khách trả" is a return ("khách trả 1 món") but also a PAYMENT: "khách trả 300.000, còn thiếu 200.000", "khách trả trước", "ghi nợ, khách trả sau",
# "khách trả tiền mặt", "ghi sổ, khách trả hết rồi". Those are told by what follows: a when / how / how much, not a thing given back
# when / how much (the error analysis's list): "trả trước / sau / hết / đủ / góp", "trả 300.000 / 500k / 1tr5"
_PAID_WHEN = r"(?:truoc|sau|tien|het|no|gop|du|them|roi|luon|xong|ngay|bang)\b"
_PAID_AMOUNT = r"(?:\d{1,3}(?:[.,]\d{3})+|\d{4,}|\d+(?:[.,]\d+)?\s*(?:k|d|tr|trieu|nghin|ngan|dong|vnd)\d*)\b"
# how: the payment methods. The shop's own labels come from its data (`payment_labels`: the values of its payment-method column);
# these are the methods of the rulebook every shop has. The lexicon's "ck" / "tm" are written in full before classifying.
PAYMENT_METHODS = ("tien mat", "chuyen khoan", "vi dien tu", "the", "cod", "ghi no")
# a payment-method column by its name (the README role `payment_method` declares the others: dev_008's `Thanh toán bằng`)
PAYMENT_COL = re.compile(r"ph[ưu][ơo]ng th[ứu]c|pttt|httt|h[ìi]nh th[ứu]c|tr[ảa] b[ằa]ng|thanh to[áa]n b[ằa]ng|lo[ạa]i tt|phuongthuc|payment",
                         re.I)


def _customer_returns(labels=()) -> str:
    how = "|".join(re.escape(x) for x in sorted({*PAYMENT_METHODS, *labels}, key=len, reverse=True))
    return rf"khach (?:mang )?tra\b(?! (?:{_PAID_WHEN}|{_PAID_AMOUNT}|(?:{how})\b))"


CUSTOMER_RETURNS = _customer_returns()
_RETURNED = r"tra hang|tra lai|hoan hang|hoan tra|doi tra|hang tra ve|returned?\b|{customer}|tra ve"

# Order matters: the first matching rule wins. Patterns run on lowercase, diacritic-free text.
RULES: list[tuple[str, str]] = [
    ("returned", _RETURNED.format(customer=CUSTOMER_RETURNS)),
    ("cancelled", r"\bhuy|cancel|khong giao duoc|giao that bai"),
    ("shipping", r"dang (?:giao|van chuyen|gui|ship|xu ly)|cho giao|chua giao|shipping|in transit|dang giao dich"),
    ("completed", r"hoan thanh|\bxong\b|da giao|da lay|completed|thanh cong|\bok\b|success|da nhan|da thanh toan"),
]
PRIORITY = {"returned": 3, "cancelled": 2, "shipping": 1, "completed": 0}  # order-level: the worst state wins
STATUS_COL = re.compile(
    r"tr[ạa]ng[ _]?th[áa]i|status|k[ếe]t qu[ảa]|ti[ếe]n [đd][ộo]|t[ìi]nh[ _]?tr[ạa]ng|x[ửu] l[ýy]|ghi[ _]?ch[úu]|note", re.I)
NOTE_COL = re.compile(r"ghi ch[úu]|note", re.I)  # free-text notes: cancel/ship are noted there, blank = completed
# Order-id column: a TEXT column whose name says "order/invoice/voucher code". Money columns such as "Tổng bill"
# must not match (they once did, and orders got grouped by equal totals).
ORDER_STRONG = re.compile(
    r"(m[ãa]|ma|s[ốo]|so)\s*(đ[ơo]n|don|h\.?\s*[óo]a|hoa|phi[ếe]u|phieu|bill|h[đd]|hd|ct\b|ch[ứu]ng t[ừu]|giao d[ịi]ch|gd\b)"
    r"|(đ[ơo]n|don|h[đd]|hd|phi[ếe]u|phieu|bill)\s*(s[ốo]|so|id|no)\b"        # "Đơn số", "HĐ số", "Bill no"
    r"|^h[đd]$|^order(\s*id)?$|order\s*(id|no|number|code)|invoice\s*(no|id|number)", re.I)
ORDER_WEAK = re.compile(r"order|invoice|bill|phi[ếe]u", re.I)


def find_order_col(df: pd.DataFrame) -> str | None:
    text = [c for c in df.columns if not pd.api.types.is_datetime64_any_dtype(df[c]) and df[c].dtype.kind not in "iufb"]
    for pat in (ORDER_STRONG, ORDER_WEAK):
        for c in text:
            if pat.search(c):
                return c
    return None


def _nodia(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")


def rules_for(payment_labels=()) -> list[tuple[str, str]]:
    """RULES with the shop's payment labels among the payment words after "khách trả": "khách trả momo" is a payment in a shop that
    takes MoMo, "khách trả 2 cái" stays a return."""
    labels = tuple(sorted({x for x in payment_labels if x}))
    if not labels:
        return RULES
    return [("returned", _RETURNED.format(customer=_customer_returns(labels)))] + RULES[1:]


def payment_labels(tables: dict, roles: dict | None = None) -> set[str]:
    """The shop's payment methods as its data writes them (folded): the values of each payment-method column (by name, or declared by
    the README role `payment_method`) that has at most 40 labels. "Thu hộ (COD)" gives "thu ho cod", "thu ho" and "cod"."""
    out: set[str] = set()
    for t, df in tables.items():
        declared = (roles or {}).get(t, {}).get("payment_method")
        for c in df.columns:
            if (c != declared and not PAYMENT_COL.search(str(c))) or df[c].dtype.kind in "iufbM":
                continue
            u = df[c].dropna().astype(str).unique()
            if not 1 <= len(u) <= 40:
                continue
            for v in u:
                f = _nodia(v).strip()
                out.add(" ".join(re.sub(r"[^\w ]+", " ", f).split()))
                out |= {" ".join(part.split()) for part in re.split(r"[()/,;]", f) if part.strip()}
    return {x for x in out if len(x) >= 2}


def classify(value, rules: list[tuple[str, str]] | None = None) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    from datamate import lexicon      # "khách trả ck" -> "khách trả chuyen khoan": the lexicon's abbreviations, written in full
    t = lexicon.apply_folded(_nodia(value))
    for cls, pat in rules or RULES:
        if re.search(pat, t):
            return cls
    return None


def _order_runs(df: pd.DataFrame) -> pd.Series | None:
    oc = find_order_col(df)
    if oc is None:
        return None
    ids = df[oc].ffill()
    if ids.isna().mean() > 0.05 or ids.nunique() < 2:
        return None
    return (ids != ids.shift()).cumsum()


def add_status_std(df: pd.DataFrame, prefer: str | None = None, payment: set[str] | None = None) -> pd.DataFrame:
    """Add `status_std` (completed/shipping/cancelled/returned) when the table has a status column.

    `prefer`: the column the README documents as "trạng thái đơn". It is used even when its name matches no status
    pattern (`TT đơn`), as long as most of its labels are recognised.
    """
    best = None
    rules = rules_for(payment or ())
    if prefer in df.columns and df[prefer].dtype.kind not in "iufM":
        vc = df[prefer].dropna().astype(str).value_counts()
        mapping = {v: classify(v, rules) for v in vc.index}
        if len(vc) and len(vc) <= 40 and sum(n for v, n in vc.items() if mapping[v]) / vc.sum() >= 0.5:
            best = ((2.0, 0), prefer, mapping, bool(NOTE_COL.search(prefer)))
    for col in df.columns:
        if best is not None and best[1] == prefer:
            break
        if not STATUS_COL.search(col) or pd.api.types.is_datetime64_any_dtype(df[col]) or df[col].dtype.kind in "iuf":
            continue
        vc = df[col].dropna().astype(str).value_counts()
        if vc.empty or len(vc) > 40:
            continue
        mapping = {v: classify(v, rules) for v in vc.index}
        note = bool(NOTE_COL.search(col))
        hit = sum(n for v, n in vc.items() if mapping[v])
        non_completed = sum(n for v, n in vc.items() if mapping[v] and mapping[v] != "completed")
        if note:
            if non_completed == 0 or hit / vc.sum() > 0.9:  # a notes column that is mostly statuses is a status column
                if non_completed == 0:
                    continue
            cov = 1.0
        else:
            cov = hit / vc.sum()
            if cov < 0.9:
                continue
        score = (cov, -len(vc))
        if best is None or score > best[0]:
            best = (score, col, mapping, note)
    if best is None:
        return df
    _, col, mapping, note = best
    std = df[col].map(lambda v: mapping.get(str(v)) if pd.notna(v) else None)
    if note:
        std = std.fillna("completed")
    runs = _order_runs(df)
    if runs is not None:  # order-level: any returned/cancelled/shipping line decides the whole order
        pr = std.map(PRIORITY)
        worst = pr.groupby(runs).transform("max")
        inv = {v: k for k, v in PRIORITY.items()}
        std = worst.map(lambda x: inv.get(x) if pd.notna(x) else None)
    df = df.copy()
    df["status_std"] = std
    df["valid_order"] = ~std.isin(["cancelled", "returned"])  # sale-counting orders: completed + shipping (unknown = valid)
    df.attrs["status_source"] = col
    df.attrs["status_map"] = {k: v for k, v in mapping.items() if v}
    return df
