"""Column descriptor: compact, deterministic schema card per business (cached).

What the dev-300 failures taught us to put on the card:
- ALL values (with counts) of low-cardinality columns, so the coder sees every status/unit/channel label
  instead of guessing from 3 examples;
- the `status_std` mapping (raw label -> completed/shipping/cancelled/returned);
- join keys between tables, found from overlapping values (keys often have different names:
  `Mã chứng từ` in the order table vs `Ma don` in the line table).
"""
from __future__ import annotations

import re

import pandas as pd

from datamate.data.loader import Business
from datamate.data.status import find_order_col
from datamate.data.structure import Structure, card_lines

LOW_CARD = 20  # list every value (with counts) up to this many distinct values
ID_NAME = re.compile(r"\bm[ãa]\b|\bid\b|code|barcode|s[ốo] [đd]i[ệe]n tho[ạa]i|sdt|phone", re.I)


def _clip(v, n: int = 32) -> str:
    v = str(v)
    return v if len(v) <= n else v[: n - 1] + "…"


def describe_column(s: pd.Series, name: str = "", max_examples: int = 3) -> str:
    n = len(s)
    nn = int(s.notna().sum())
    base = f"dtype={s.dtype}, non-null={nn}/{n}"
    if pd.api.types.is_datetime64_any_dtype(s):
        return f"{base}, range={s.min().date() if nn else None}..{s.max().date() if nn else None}"
    vc = s.dropna().value_counts()
    is_id = bool(ID_NAME.search(name))
    if pd.api.types.is_numeric_dtype(s):
        if not is_id and 0 < len(vc) <= 8:
            return f"{base}, values: " + ", ".join(f"{_clip(k)}({v})" for k, v in vc.items())
        return f"{base}, min={s.min()}, max={s.max()}"
    if not is_id and 0 < len(vc) <= LOW_CARD:
        return f"{base}, ALL values: " + ", ".join(f"{_clip(k)}({v})" for k, v in vc.items())
    ex = ", ".join(repr(_clip(v, 30)) for v in s.dropna().astype(str).unique()[:max_examples])
    return f"{base}, distinct={len(vc)}, e.g. {ex}"


ROUND_COL = re.compile(r"l[ưuượ]+t|[đd][ợo]t|l[ầa]n\b|chuy[ếe]n|round|batch|trip", re.I)


def _round_cols(df: pd.DataFrame, order_col: str) -> list[str]:
    """Small integer columns named like "lượt / đợt / lần / chuyến" that number repeated events inside one order."""
    out = []
    for c in df.columns:
        if c == order_col or not ROUND_COL.search(str(c)) or df[c].dtype.kind not in "iuf":
            continue
        v = df[c].dropna()
        if len(v) and v.nunique() <= 20 and (v % 1 == 0).all() and v.min() >= 0:
            out.append(c)
    return out


def describe_table(name: str, df: pd.DataFrame, columns: list[str] | None = None, structure: Structure | None = None) -> str:
    """`structure` (data/structure.py) adds what the values prove about the table: parent / child tables and the columns
    that live only on the parent, document totals, per-document columns of a flat table, README definitions, aliases."""
    cols = [c for c in (columns or df.columns) if c in df.columns]
    extra, notes = card_lines(name, structure)
    lines = [f"Table `{name}` ({len(df)} rows)"] + extra
    lines += [f"  - `{c}`: {describe_column(df[c], c)}" + (f" — {notes[c]}" if c in notes else "") for c in cols]
    oc = find_order_col(df)
    if oc is not None and len(df):
        if df[oc].nunique() < 0.9 * len(df):
            lines.append(f"  * Each row is an ORDER LINE (order id: `{oc}`, {df[oc].nunique()} orders): COUNT ORDERS = `{oc}`.nunique(), never len().")
            for rc in _round_cols(df, oc):  # e.g. `Lượt giao`: one delivery round = one (order, round) pair, not one row
                lines.append(f"  * `{rc}` numbers the rounds (lượt/đợt) inside one order: COUNT ROUNDS = number of distinct (`{oc}`, `{rc}`) pairs "
                             f"(drop_duplicates), never len() (each round has several order lines).")
        else:
            lines.append(f"  * Each row is one order (order id: `{oc}`).")
    src, mp = df.attrs.get("status_source"), df.attrs.get("status_map")
    if src and mp and "status_std" in df.columns:
        by = {}
        for raw, cls in mp.items():
            by.setdefault(cls, []).append(raw)
        lines.append("  * `valid_order` = True when the order is not cancelled/returned (i.e. completed or shipping: the orders that COUNT toward revenue / order counts).")
        lines.append(f"  * `status_std` is derived from `{src}`: " + "; ".join(f"{cls} ← {', '.join(map(repr, raws))}" for cls, raws in by.items()))
    return "\n".join(lines)


def _key_values(s: pd.Series, cap: int = 5000) -> set[str]:
    u = s.dropna().astype(str).unique()
    return set(u[:cap]) if len(u) else set()


def describe_relations(tables: dict[str, pd.DataFrame], max_lines: int = 25, skip: set | None = None) -> str:
    """Join hints: column pairs from different tables whose values (largely) coincide. `skip`: pairs
    {(table, column), (table, column)} already written on the cards as parent / child (data/structure.py)."""
    keys: list[tuple[str, str, set[str]]] = []
    for t, df in tables.items():
        for c in df.columns:
            s = df[c]
            if pd.api.types.is_datetime64_any_dtype(s) or s.dtype.kind in "fb" or c == "source_sheet" or c == "status_std":
                continue
            vals = _key_values(s)
            nun = s.nunique(dropna=True)
            if len(vals) >= 5 and (ID_NAME.search(c) or nun >= 0.5 * max(s.notna().sum(), 1) or nun <= 5000) and nun >= 5:
                # id-like (named so, or mostly unique) or categorical with enough distinct values to be a key
                if ID_NAME.search(c) or nun >= 0.3 * max(s.notna().sum(), 1) or re.search(r"t[êe]n|name|h[àa]ng|kh[áa]ch", c, re.I):
                    keys.append((t, c, vals))
    out = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (ta, ca, va), (tb, cb, vb) = keys[i], keys[j]
            if ta == tb:
                continue
            inter = len(va & vb)
            cover = inter / min(len(va), len(vb))
            if skip and frozenset({(ta, ca), (tb, cb)}) in skip:
                continue
            if inter >= 5 and cover >= 0.8:
                out.append((cover, f"- `{ta}`.`{ca}` ↔ `{tb}`.`{cb}` (match {cover:.0%})"))
    out.sort(key=lambda x: -x[0])
    seen, lines = set(), []
    for _, line in out:
        if line not in seen:
            seen.add(line)
            lines.append(line)
    if not lines:
        return ""
    return "RELATIONS BETWEEN TABLES (columns whose values coincide, usable as join keys; column names may differ):\n" + "\n".join(lines[:max_lines])


def describe_business(biz: Business, structure: Structure | None = None) -> dict[str, str]:
    out = {name: describe_table(name, df, structure=structure) for name, df in biz.tables.items()}
    if biz.notes:
        out["__notes__"] = "DATA CLEANING NOTES:\n" + "\n".join(f"- {n}" for n in biz.notes)
    skip = {frozenset({(l.head, l.key), (l.lines, l.line_key)}) for l in structure.links} if structure else None
    rel = describe_relations(biz.tables, skip=skip)
    if rel:
        out["__relations__"] = rel
    return out
