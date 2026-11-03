"""ABBREVIATION SHEETS: a small table that writes each short item name in full (and gives its category), found by its shape.

A shop owner's book writes items short (`Sạc 25W SS`, `IP16 128 đen`) and keeps a sheet that spells them out (dev_011 / dev_014:
`Ghi tắt` = short form, `Tên hàng` = full name, `Loại` = máy mới / máy cũ / phụ kiện; 103 public questions). The loader read that
sheet as one more table: item names stayed short, a question naming the full item matched nothing, and there was no category.

Found from the values, never from the sheet's name: a table of at most MAX_ROWS rows with a short-form column K and a longer, name-like
column F such that the item-name column of another table is mostly (>= COVER, folded: case, accents and spacing ignored) written in
K or F, and a real part of it (>= SHORT_SHARE) in K only. A product catalogue does not pass: its codes are not what the item-name
column holds, and its group column is not a name with as many distinct values. The book's column is not overwritten (the README
tells the Coder to look the short forms up in the sheet, and a lookup of full names would give nothing): a column with the full
name and the sheet's category columns are added next to it, and the items engine counts by the full name.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

import pandas as pd

from datamate.text import fold, given_name, nodia

MAX_ROWS = 2000
COVER = 0.8          # share of the item-name rows written in the sheet's short or full form
SHORT_SHARE = 0.05   # ... and in its short form only
MAX_CATEGORY = 50    # a sheet column with at most this many labels is a category (Loại), joined too
SKIP = {"source_sheet", "status_std", "valid_order"}
ITEM_COL = re.compile(r"^(ten)?(hang|hanghoa|sanpham|mon|sp|mathang|goi|thuoc)(hoa|ban)?$|^noidung$|^tengoi$")
NAME_COL = re.compile(r"ten|name|hang|day ?du")     # the full-name column: a name, not a description or a note
ID_COL = re.compile(r"\bm[ãa]\b|\bma\b|code|\bid\b|barcode|sku", re.I)


def _cn(c) -> str:
    return re.sub(r"[\W_]+", "", nodia(str(c)).lower())


def _text(s: pd.Series) -> bool:
    return (s.dtype == object or pd.api.types.is_string_dtype(s)) and s.notna().any()


@dataclass
class Expansion:
    table: str                    # the book whose item names are written short
    column: str                   # its item-name column (left as it is)
    full: str                     # the added column: the full name
    sheet: str                    # the abbreviation sheet
    short_col: str
    full_col: str
    categories: list[str] = field(default_factory=list)   # added category columns (names in `table`)
    to_full: dict[str, str] = field(default_factory=dict)  # folded short or full form -> full name (the sheet's spelling)
    share: float = 0.0
    short_share: float = 0.0
    example: str = ""             # one short form as the sheet writes it

    def note(self) -> str:
        ex = self.example
        cats = "".join(f", `{c}` its category" for c in self.categories)
        return (f"`{self.table}`.`{self.column}` writes items short in places{f' ({ex!r})' if ex else ''}: `{self.full}` is the full "
                f"name from `{self.sheet}` (`{self.short_col}` -> `{self.full_col}`){cats}. Filter, group and name items by "
                f"`{self.full}` (item_lines already does).")


def _item_cols(df: pd.DataFrame, declared: str | None) -> list[str]:
    out = [declared] if declared in df.columns else []
    out += [c for c in df.columns if c not in out and c not in SKIP and ITEM_COL.search(_cn(c)) and _text(df[c])]
    return [c for c in out if _text(df[c])]


def find_expansions(tables: dict[str, pd.DataFrame], roles: dict[str, dict[str, str]] | None = None) -> list[Expansion]:
    """Every (book column, abbreviation sheet) pair that passes the shape test, the best sheet per book column."""
    sheets = []
    for a, A in tables.items():
        if not 3 <= len(A) <= MAX_ROWS:
            continue
        texts = [c for c in A.columns if c not in SKIP and _text(A[c])]
        for k, f in itertools.permutations(texts, 2):
            if ID_COL.search(str(f)) or not NAME_COL.search(_cn(f)):
                continue
            pairs = A[[k, f]].dropna()
            if len(pairs) < 3:
                continue
            kf, ff = pairs[k].astype(str).map(fold), pairs[f].astype(str).map(fold)
            if not kf.is_unique or (kf == ff).mean() > 0.1 or ff.nunique() < 0.8 * kf.nunique():
                continue
            if ff.str.len().mean() <= kf.str.len().mean():
                continue
            sheets.append((a, k, f, pairs))
    best: dict[tuple[str, str], Expansion] = {}
    for a, k, f, pairs in sheets:
        short = dict(zip(pairs[k].astype(str).map(fold), pairs[f].astype(str).str.strip()))
        full = {fold(v): v for v in pairs[f].astype(str).str.strip()}
        only_short = set(short) - set(full)
        for t, T in tables.items():
            if t == a or len(T) <= len(pairs):
                continue
            for c in _item_cols(T, (roles or {}).get(t, {}).get("item_name")):
                v = T[c].dropna().astype(str).map(fold)
                if len(v) < 20:
                    continue
                share, sshare = float(v.isin(set(short) | set(full)).mean()), float(v.isin(only_short).mean())
                if share < COVER or sshare < SHORT_SHARE:
                    continue
                if (t, c) in best and (best[(t, c)].share, best[(t, c)].short_share) >= (share, sshare):
                    continue
                ex = next((str(x) for x, y in zip(pairs[k], pairs[f]) if fold(x) in only_short), "")
                best[(t, c)] = Expansion(t, c, f"{c} (tên đầy đủ)", a, k, f, to_full={**full, **short}, share=round(share, 3),
                                         short_share=round(sshare, 3), example=ex)
    return list(best.values())


def expand_abbreviations(tables: dict[str, pd.DataFrame], roles: dict[str, dict[str, str]] | None = None
                         ) -> tuple[dict[str, pd.DataFrame], list[Expansion]]:
    """The tables with the full-name and category columns added; never raises (a failure leaves the tables as they were)."""
    try:
        found = find_expansions(tables, roles)
    except Exception:  # noqa: BLE001 - a missed sheet costs less than a broken load
        return tables, []
    out = dict(tables)
    for e in found:
        T, A = out[e.table], tables[e.sheet]
        key = T[e.column].astype(str).map(fold).where(T[e.column].notna())
        full_name = key.map(e.to_full)
        add = {e.full: full_name.where(full_name.notna(), T[e.column])}
        by_full = A.assign(__f=A[e.full_col].astype(str).map(fold)).drop_duplicates("__f").set_index("__f")
        for c in A.columns:
            if c in (e.short_col, e.full_col) or c in SKIP or not _text(A[c]) or not 2 <= A[c].nunique() <= MAX_CATEGORY:
                continue
            name = c if c not in T.columns else f"{c} ({e.sheet.split('__')[-1]})"
            add[name] = full_name.map(lambda x: fold(x) if isinstance(x, str) else None).map(by_full[c])
            e.categories.append(name)
        cols = list(T.columns)
        at = cols.index(e.column) + 1
        T = T.copy()
        for i, (name, values) in enumerate(add.items()):
            T.insert(at + i, name, values.values)
        out[e.table] = T
    return out, found


# ---------------------------------------------------------------------------------------------- nicknames of staff
MAX_NICKNAMES = 60   # a staff / technician column has few distinct spellings
MAX_ROSTER = 500
NICK_SHARE = 0.9     # share of the column's rows whose name (title removed) is ONE word that is the given name of ONE roster person


@dataclass
class Nicknames:
    table: str
    column: str
    full: str              # the added column: the roster's full name
    roster: str
    roster_col: str
    share: float

    def note(self) -> str:
        return (f"`{self.table}`.`{self.column}` names people by their given name, often with a title (\"A.Đức\", \"a Đức\", \"Đức\"): "
                f"`{self.full}` is the person's full name from `{self.roster}`.`{self.roster_col}` (matched on the given name, which is "
                f"unique there). Count, group and join people by `{self.full}`.")


def _person_col(df: pd.DataFrame) -> list[str]:
    """Columns of full people's names: two or more words of letters, mostly distinct."""
    out = []
    for c in df.columns:
        if c in SKIP or not _text(df[c]) or ID_COL.search(str(c)):
            continue
        v = df[c].dropna().astype(str).str.strip()
        if len(v) < 2 or v.nunique() < 0.9 * len(v):
            continue
        if v.str.fullmatch(r"[^\W\d_]+(?:\s+[^\W\d_]+){1,4}").mean() >= 0.9:
            out.append(c)
    return out


def find_nicknames(tables: dict[str, pd.DataFrame]) -> list[Nicknames]:
    """A column that writes people by given name (with or without a title) and a roster table whose full names have those given
    names, each once. dev_012: `Thợ` = "A.Đức" / "a Đức" / "Đức", `DanhSachTho.Họ tên` = "Lương Chi Đức"."""
    rosters = []
    for r, R in tables.items():
        if not 2 <= len(R) <= MAX_ROSTER:
            continue
        for rc in _person_col(R):
            last = R[rc].dropna().astype(str).map(lambda x: fold(x).split()[-1])
            counts = last.value_counts()
            unique = {k: v for k, v in zip(last, R[rc].dropna().astype(str).str.strip()) if counts[k] == 1}
            if len(unique) >= 2:
                rosters.append((r, rc, unique))
    out = []
    for t, T in tables.items():
        for c in T.columns:
            if c in SKIP or not _text(T[c]) or ID_COL.search(str(c)):
                continue
            vc = T[c].dropna().astype(str).value_counts()
            if not 2 <= len(vc) <= MAX_NICKNAMES or vc.sum() < 20:
                continue
            given = {v: given_name(v) for v in vc.index}
            if sum(n for v, n in vc.items() if len(given[v].split()) == 1) < NICK_SHARE * vc.sum():
                continue
            best = None
            for r, rc, unique in rosters:
                if r == t:
                    continue
                share = sum(n for v, n in vc.items() if given[v] in unique) / vc.sum()
                if share >= NICK_SHARE and (best is None or share > best[0]):
                    best = (share, r, rc, unique)
            if best is not None:
                out.append(Nicknames(t, c, f"{c} (họ tên)", best[1], best[2], round(best[0], 3)))
                out[-1]._map = {v: best[3].get(given[v]) for v in vc.index}   # noqa: SLF001 - used once, by expand_nicknames
    return out


def expand_nicknames(tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[Nicknames]]:
    """The tables with each nickname column's full-name column added next to it; never raises."""
    try:
        found = find_nicknames(tables)
    except Exception:  # noqa: BLE001
        return tables, []
    out = dict(tables)
    for n in found:
        T = out[n.table].copy()
        T.insert(list(T.columns).index(n.column) + 1, n.full, T[n.column].astype(str).map(n._map).where(T[n.column].notna()).values)
        out[n.table] = T
    return out, found
