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

# Order matters: the first matching rule wins. Patterns run on lowercase, diacritic-free text.
RULES: list[tuple[str, str]] = [
    ("returned", r"tra hang|tra lai|hoan hang|hoan tra|doi tra|hang tra ve|returned?\b|khach (?:mang )?tra\b|tra ve"),
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


def classify(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    t = _nodia(value)
    for cls, pat in RULES:
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


def add_status_std(df: pd.DataFrame, prefer: str | None = None) -> pd.DataFrame:
    """Add `status_std` (completed/shipping/cancelled/returned) when the table has a status column.

    `prefer`: the column the README documents as "trạng thái đơn". It is used even when its name matches no status
    pattern (`TT đơn`), as long as most of its labels are recognised.
    """
    best = None
    if prefer in df.columns and df[prefer].dtype.kind not in "iufM":
        vc = df[prefer].dropna().astype(str).value_counts()
        mapping = {v: classify(v) for v in vc.index}
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
        mapping = {v: classify(v) for v in vc.index}
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
