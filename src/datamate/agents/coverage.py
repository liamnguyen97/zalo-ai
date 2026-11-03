"""COVERAGE: unanswerable decisions the data settles without a model (rulebook §7).

§7: "no data" is not 0: a period lying entirely outside the dates the shop's data covers, or information the shop does not
record, makes the question unanswerable. v2 on dev300 got 7 of 12 gold-unanswerable questions right; the misses were
`46e6ca` (the period starts before the data does: answered 0), `d18944` (every order of the month is a walk-in, so no
customer exists to rank: three rejected turns, then abstain) and questions needing a column that does not exist. These checks
use only the resolved period and the shop's tables, so they hold for any shop.
"""
from __future__ import annotations

import pandas as pd

AMOUNT_MEASURES = ("money", "orders", "quantity", "count", "ratio")


def period_outside_data(spec, variables: dict) -> str | None:
    """A reason when the main period lies entirely outside the dates of `sales_orders` AND of every table the spec uses; None
    when there is any doubt (no orders table, no period, a non-amount question, or another table has rows in the period)."""
    so, lo, hi = variables.get("sales_orders"), variables.get("period_start"), variables.get("period_end")
    if so is None or lo is None or hi is None or getattr(spec, "measure", None) not in AMOUNT_MEASURES:
        return None
    if "date" not in so.columns or so["date"].isna().all():
        return None
    first, last = so["date"].min(), so["date"].max()
    if not (hi < first or lo > last):
        return None
    tables = variables.get("tables") or {}
    for t in (getattr(spec, "tables", None) or {}):
        df = tables.get(t)
        if df is None:
            continue
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]) and ((df[c] >= lo) & (df[c] <= hi)).any():
                return None
    return (f"The period {lo.date()}..{hi.date()} lies entirely outside the dates the shop's data covers "
            f"({first.date()}..{last.date()}): no data is not 0 (rulebook §7).")


def no_identified_customers(variables: dict) -> str | None:
    """A reason when the main period has valid orders and every one is a walk-in: the shop does not record who bought, so a
    question about customers cannot be answered (rulebook §7). None when anything else is true."""
    so, lo, hi = variables.get("sales_orders"), variables.get("period_start"), variables.get("period_end")
    if so is None or lo is None or hi is None or not {"date", "valid", "walk_in"} <= set(so.columns):
        return None
    d = so[(so["date"] >= lo) & (so["date"] <= hi) & so["valid"]]
    if len(d) == 0 or not d["walk_in"].all():
        return None
    return (f"All {len(d)} valid orders between {lo.date()} and {hi.date()} are walk-in orders: the shop records no customer "
            "for them, so a question about customers cannot be answered (rulebook §7).")


# ---------------------------------------------------------------------------------------------- information the shop records
# §7: "shop không ghi loại thông tin câu hỏi cần" is unanswerable, not 0. On dev300 the system answered "khách nữ chiếm bao nhiêu %
# doanh thu" (no gender anywhere in the shop's data) with 13.14 and "% khách mua từ 8 đơn" with 100.0 in a shop that records no
# customer. Each NEED is a kind of information (a field every kind of shop may or may not keep), the words a question asks it with
# (read on text.norm of the question), and the evidence that the shop records it: a column whose name or README description says
# so, an engine built on it, or (for a label the question names) the label among the data's values. A need is reported only when
# the question asks it AND no evidence exists; `direct` needs are answered UNANSWERABLE in code, the others are told to the Coder as
# a fact, chosen by their precision on the 3,000 training questions (scripts/coverage_report.py).
import re as _re
import unicodedata
from dataclasses import dataclass as _dataclass

from datamate.data.readme import parse_column_docs
from datamate.text import fold_text, norm


@_dataclass(frozen=True)
class Need:
    name: str
    asks: str            # regex on norm(question)
    cols: str            # regex on fold_text(column name) / fold_text(README description of a column)
    tables: str = ""     # regex on fold_text(table name): a table of that kind is evidence too
    engine: str = ""     # a BusinessContext engine attribute that is evidence (debt, purchases)
    direct: bool = True  # answered UNANSWERABLE in code; False = told to the Coder as a fact (a need with a measured false fire)
    what: str = ""       # how the fact names it


NEEDS = (
    # not "đàn ông": folded it is "dan ong", as in "Keo dán ống PVC" (2 training questions); not "khách năm nay / năm 2025" (the
    # year, folded "nam"; the accented question is read in `_asked` too)
    Need("gender", r"\bkhach (?:hang )?(?:nu|nam)\b(?! (?:nay|ngoai|truoc|qua|roi|toi|sau|\d))|\bgioi tinh\b|\b(?:nam|nu) gioi\b"
                   r"|\bphai (?:nam|nu)\b|\bphu nu\b",
         r"\bgioi tinh\b|\bgender\b|\bsex\b|\bphai\b|\bnam nu\b", what="customers' gender"),
    # not "phường": folded it is a given name ("Giang Phương Thu"). "khách đường X" is read on the accented question (`_asked`):
    # folded, "khách Dương Minh Hà" (a surname) is the same text
    Need("address", r"\b(?:o|tai|nha o|song o) duong\b|\bdia chi\b|\bkhu vuc\b",
         r"\bdia chi\b|\baddress\b|\bduong\b|\bphuong\b|\bkhu vuc\b|\bquan huyen\b|\btinh thanh\b", what="addresses / streets"),
    Need("stock", r"\bton kho\b|\btrong kho\b|\bkho con\b|\bcon ton\b|\bhang ton\b|\bton cuoi\b|\bton dau\b",
         r"\bton\b|\bkho\b|\bstock\b|\binventory\b|\bnhap\b", r"ton|kho|stock|inventory|nhap", "purchases",
         what="stock (no stock table, no purchases)"),
    # what was bought in, not a plan to ("đang tính nhập thêm máy cũ, tháng này bán ra được bao nhiêu": a sales question)
    Need("purchase", r"\bgia nhap\b|\btien nhap\b|\bnhap hang\b(?! them)|\bphieu nhap\b|\bnha cung cap\b|\bncc\b"
                     r"|\bnhap (?:vao |ve )?(?:bao nhieu|may|bn|binh quan|tong cong)\b",
         r"\bnhap\b|\bncc\b|\bnha cung cap\b|\bpurchase\b|\bsupplier\b", r"nhap|ncc|nhacungcap|purchase|supplier", "purchases",
         what="purchases (no purchase slips, no import price)"),
    Need("cost", r"\blai gop\b|\bgia von\b|\bloi nhuan\b|\btien von\b",
         r"\bvon\b|\bcost\b|\blai gop\b|\bloi nhuan\b|\bgia nhap\b", r"nhap|purchase", "purchases", what="cost prices"),
    Need("debt", r"\bcong no\b|\bcon no\b|\bghi no\b|\btra no\b|\bthu no\b|\bmua chiu\b|\bkhach no\b|\bcon thieu\b|\bno (?:nhieu|bao nhieu|qua han)\b",
         r"\bno\b|\bcong no\b|\bcon thieu\b|\bda tra\b|\btra truoc\b|\bthanh toan\b|\bghi so\b|\bkhach tra\b|\bphai thu\b|\bgop\b",
         r"\bno\b|congno|sono|thanhtoan|phieuthu|gop", "debt", what="debt (nothing owed or paid later is recorded)"),
    Need("expiry", r"\bhan dung\b|\bhet han\b|\bhan su dung\b|\bhsd\b|\bdate\b",
         r"\bhan dung\b|\bhan su dung\b|\bhsd\b|\bhet han\b|\bexpir|\bhan\b", what="expiry dates"),
    # who bought: a question counting / ranking / filtering customers, in a shop whose orders name no customer at all (every order a
    # walk-in). Not "khách lẻ" / "vãng lai" (walk-in sales are recorded), not "khách trả" (what was paid)
    # not "xe khách quen": a repair shop knows its regulars by plate (rulebook "Khách quen, xe tới nhiều lần")
    Need("customers", r"(?<!luot )\b(?:bao nhieu|may|bn|so|nhung|cac|moi|tung|top \d+|\d+|%) khach\b(?! (?:le|vang lai|tra|dua|quen))"
                      r"|(?<!xe )\bkhach (?:hang )?(?:nao|quen|than thiet|vip|si|mua nhieu|chi nhieu|da mua)\b|\bai (?:mua|chi) (?:nhieu|it)",
         r"(?!)", what="customers (every order is a walk-in: no customer is named)"),
)


def _words(name) -> str:
    """A column / table name as folded words: `HanSuDung` -> "han su dung", `trang_thai_kh` -> "trang thai kh"."""
    t = str(name)
    t = "".join(" " + c if i and c.isupper() and t[i - 1].islower() else c for i, c in enumerate(t))
    return fold_text(t.replace("_", " "))


def _column_texts(ctx) -> list[str]:
    """Every column name (folded) of every table, and every README column description."""
    out = []
    for t, df in (getattr(ctx, "tables", None) or {}).items():
        out += [_words(c) for c in df.columns if c not in ("source_sheet", "status_std", "valid_order") and df[c].notna().any()]
    out += [fold_text(d.desc) for d in parse_column_docs(getattr(ctx, "readme", "") or "")]
    return out


def _table_names(ctx) -> list[str]:
    return [_re.sub(r"[\W_]+", "", fold_text(t)) for t in (getattr(ctx, "tables", None) or {})]


def _gender_values(ctx) -> bool:
    """A column whose labels are genders (Nam / Nữ / male / female) whatever its name."""
    for df in (getattr(ctx, "tables", None) or {}).values():
        for c in df.columns:
            if df[c].dtype != object and not pd.api.types.is_string_dtype(df[c]):
                continue
            u = {fold_text(v) for v in df[c].dropna().astype(str).unique()[:50]}
            if 2 <= len(u) <= 6 and u <= {"nam", "nu", "male", "female", "m", "f", "khac", "khong ro", ""}:
                return True
    return False


def _named_customers(ctx) -> bool:
    """Some order names its customer (True when the shop has no `sales_orders` to tell: not judged)."""
    o = getattr(ctx, "orders", None)
    if o is None or "customer_key" not in o.columns:
        return True
    return bool((o["customer_key"].astype(str) != "").any())


def has_information(need: Need, ctx) -> bool:
    if need.name == "customers":
        return _named_customers(ctx)
    cols = _column_texts(ctx)
    if any(_re.search(need.cols, c) for c in cols):
        return True
    if need.tables and any(_re.search(need.tables, t) for t in _table_names(ctx)):
        return True
    if need.engine and getattr(ctx, need.engine, None) is not None:
        return True
    return need.name == "gender" and _gender_values(ctx)


# the accented question: "khách (ở) đường X" is a street ("Dương" is a surname), "khách năm ..." is the year, not a gender
STREET_RAW = _re.compile(r"\bkhách(?: hàng)?(?: ở| tại)? đường\b")
YEAR_RAW = _re.compile(r"\bkhách(?: hàng)? năm\b")
AFTER_KHACH = _re.compile(r"\bkhach (?:hang )?(?:o |tai )?(\w+ \w+)")


def _person_prefixes(ctx) -> set[str]:
    """The first two words (folded) of the people's names the shop records: "khách Dương Minh Hà" names a customer."""
    out: set[str] = set()
    o = getattr(ctx, "orders", None)
    if o is not None and "customer" in o.columns:
        for v in o["customer"].dropna().astype(str).unique()[:20000]:
            w = fold_text(v).split()
            if len(w) >= 2:
                out.add(" ".join(w[:2]))
    return out


def _asked(need: Need, question: str, q: str, names) -> bool:
    raw = unicodedata.normalize("NFC", (question or "").lower())
    accented = raw != fold_text(raw)
    hit = bool(_re.search(need.asks, q))
    if need.name == "address" and not hit:
        # "khách đường X": written with accents, or with none at all ("khach duong ly thuong kiet") and X not a customer's name
        hit = bool(STREET_RAW.search(raw)) or (not accented and bool(_re.search(r"\bkhach (?:hang )?duong\b", q)))
    if need.name == "gender" and hit and YEAR_RAW.search(raw) and not _re.search(r"\bkhách(?: hàng)? (?:nữ|nam)\b", raw):
        hit = False
    street_written = need.name == "address" and bool(STREET_RAW.search(raw))     # "đường" with its accents: a street, not "Dương"
    if hit and need.name in ("address", "gender") and not street_written:
        m = AFTER_KHACH.search(q)
        if m and m.group(1) in names():          # "khách Dương Minh Hà", "khách Nam Châu ...": a customer of the shop
            hit = False
    return hit


def missing_information(question: str, ctx) -> list[Need]:
    """The kinds of information the question asks for that the shop's data does not record ([] when in doubt or on failure)."""
    try:
        q = norm(question)
        cache: dict = {}
        names = lambda: cache.setdefault("n", _person_prefixes(ctx))   # noqa: E731 - read only when a need could fire
        return [n for n in NEEDS if _asked(n, question, q, names) and not has_information(n, ctx)]
    except Exception:  # noqa: BLE001 - a coverage check never fails a question
        return []


def coverage_fact(needs: list[Need]) -> str:
    if not needs:
        return ""
    return "".join(f"- The shop's data records no {n.what}: no column, README column or table of that kind. A question that needs it is "
                   "UNANSWERABLE (rulebook §7), unless you find a column that records it (print the column names).\n" for n in needs)
