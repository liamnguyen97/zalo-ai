"""TABLE STRUCTURE: which rows are documents and which are their lines, measured on the data (no column-name rules).

A shop's export splits a paper document (an order, an invoice, a purchase slip) in one of two ways:

  A. two tables: one row per DOCUMENT (date, supplier/customer, document total) and one row per LINE (item, qty, unit
     price), joined on the document id (17 of the 30 training shops, e.g. `DanhSachNhapHang` + `DanhSachChiTietNhapHang`);
  B. one flat table: one row per line, the document's own fields (date, status, customer) repeated on every line of it
     (13 of 30, e.g. `ban_hang.csv`).

A 9B model reads the two cards of layout A as two unrelated tables: on train q_00a6e23cbacf it looked for the date in the
lines table and divided the purchase slip's TOTAL (`Cần trả NCC`, every item of the slip) by one item's quantity (7.9M
per carton instead of 200,556). This module finds, from the values alone:

  - `DocLink`   document table -> lines table, with the columns that exist only on the document side;
  - `DocTotal`  a document column that equals the sum of its lines (a whole-document amount);
  - `doc_level` in a flat table, the columns that are the same on every line of a document;
  - `notes`     README sentences that define a column, attached to the table that has it;
  - `aliases`   names the README uses for a table (`DanhSachNhapHang`) -> the loader's key (`DuLieuBanHang__DanhSachNhapHang`).

Everything is reported only when the evidence is strong (unique ids, ~all ids shared, few child rows per parent, a flat
table's document id stored in runs, totals matching on >= 95% of documents with a non-zero value): a wrong structure on a
card misleads more than no structure, so when unsure nothing is said and the card stays as it was.
`DATAMATE_STRUCTURE_CARDS=0` turns it off (A/B runs).
"""
from __future__ import annotations

import itertools
import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

SKIP = {"source_sheet", "status_std", "valid_order"}
MIN_DOCS = 20            # documents needed before any claim
MAX_LINES_PER_DOC = 50   # a master table (60 products, 83k sales lines: 1,388 per "document") is not a document table
CONTIGUOUS = 1.10        # runs of the id / distinct ids: the lines of one document are stored together
TOTAL_SHARE = 0.95       # share of documents (non-zero) whose value equals the sum of their lines
MAX_NUM_LINE_COLS = 10   # numeric line columns tried (and their pairwise products)


@dataclass
class DocLink:
    head: str                 # one row per document
    key: str                  # document id in `head`
    lines: str                # several rows per document
    line_key: str             # document id in `lines`
    docs: int
    lines_per_doc: float
    head_only: list[str] = field(default_factory=list)   # document columns the lines table does not have


@dataclass
class DocTotal:
    table: str
    column: str
    lines: str
    expr: str                 # "`Số lượng` × `Đơn giá`" or "`Thành tiền`"
    share: float
    docs: int


@dataclass
class Structure:
    links: list[DocLink] = field(default_factory=list)
    totals: list[DocTotal] = field(default_factory=list)
    doc_level: dict[str, tuple[str, list[str]]] = field(default_factory=dict)   # flat table -> (doc id, per-document columns)
    notes: dict[str, dict[str, str]] = field(default_factory=dict)              # table -> {column: README sentence}
    aliases: dict[str, str] = field(default_factory=dict)                       # README / short name -> table key

    def summary(self) -> dict:
        """Compact, JSON-able record for the onboarding report."""
        return {"links": [f"{l.head}.{l.key} -> {l.lines}.{l.line_key} ({l.lines_per_doc}/doc)" for l in self.links],
                "totals": [f"{t.table}.{t.column} = sum {t.expr} of {t.lines} ({t.share:.0%} of {t.docs})" for t in self.totals],
                "doc_level": {t: f"{k}: {cols}" for t, (k, cols) in self.doc_level.items()},
                "notes": sum(len(v) for v in self.notes.values()), "aliases": self.aliases}

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------------------------- table names
def _safe(name: str) -> str:
    """The loader's own sanitising of file / sheet names (data/loader._safe)."""
    return re.sub(r"\W+", "_", name, flags=re.UNICODE).strip("_")


def _stem(name: str) -> str:
    return re.sub(r"\.(xlsx|xls|csv|sqlite|db)$", "", str(name).strip(), flags=re.I)


def resolve_table(name: str, tables) -> str | None:
    """The table key a name refers to: itself, or the ONE key whose sheet / file part is that name.

    The loader keys a sheet as `<file>__<sheet>` (and a sqlite table as `<file>__<table>`), while the README and the model
    call it by the sheet name: biz_109's README lists `DanhSachNhapHang`, the key is `DuLieuBanHang__DanhSachNhapHang`.
    None when the name matches no table or more than one."""
    if name in tables:
        return name
    s = _safe(_stem(name)).lower()
    if not s:
        return None
    hits = [t for t in tables if t.lower() == s or t.lower().endswith("__" + s) or t.lower().split("__")[0] == s]
    hits = list(dict.fromkeys(hits))
    if len(hits) == 1:
        return hits[0]
    exact_sheet = [t for t in hits if t.lower().endswith("__" + s)]
    return exact_sheet[0] if len(exact_sheet) == 1 else None


def readme_aliases(readme: str, tables) -> dict[str, str]:
    """Every name written in backticks in the README that is not a key but resolves to exactly one table."""
    out = {}
    for name in re.findall(r"`([^`\n]{2,80})`", readme or ""):
        name = name.split(".")[0] if re.fullmatch(r"[\w ]+\.[\w ]+", name) and not re.search(r"\.(xlsx|xls|csv|sqlite|db)$", name, re.I) else name
        stem = _stem(name)
        if stem in tables or stem in out:
            continue
        t = resolve_table(stem, tables)
        if t is not None and t != stem:
            out[stem] = t
    return out


# ---------------------------------------------------------------------------------------------- ids
def _id_like(s: pd.Series) -> bool:
    """Columns that can hold a document id: text or integers, never dates, flags or decimals."""
    if pd.api.types.is_datetime64_any_dtype(s) or s.dtype.kind in "bfmM":
        return False
    return s.notna().any()


def _contiguity(s: pd.Series) -> float:
    """Runs of equal consecutive values / distinct values: 1.0 when each id's rows are stored together."""
    v = s.dropna().astype(str)
    if v.empty:
        return float("inf")
    runs = int((v != v.shift()).sum())
    return runs / max(v.nunique(), 1)


def find_links(tables: dict[str, pd.DataFrame]) -> list[DocLink]:
    """Parent table -> child table pairs, from the ids alone.

    The id is unique in the parent, repeats in the child, almost every child row points to a known parent and a parent has
    few child rows (a 60-product master over 83k sales lines is not reported: 1,388 rows per product). Documents and their
    lines are the case that matters, but a product master and its units / lots / purchase lines has the same shape and the
    same join, so the card words it as parent / child. Storage order is NOT required: biz_122 stores its purchase lines
    sorted by product, not by slip."""
    uniq, rep = [], []
    for t, df in tables.items():
        n = len(df)
        if n < MIN_DOCS:
            continue
        for c in df.columns:
            if c in SKIP or not _id_like(df[c]):
                continue
            s = df[c]
            nn, nu = int(s.notna().sum()), int(s.nunique())
            if nu < MIN_DOCS:
                continue
            if nn >= 0.99 * n and nu == nn:
                uniq.append((t, c, set(s.dropna().astype(str))))
            elif nn >= 0.9 * n and nu < nn:
                rep.append((t, c, nn, nu))
    out = []
    for (ht, hc, hv), (lt, lc, lnn, lnu) in itertools.product(uniq, rep):
        if ht == lt:
            continue
        lvals = set(tables[lt][lc].dropna().astype(str).unique())
        inter = len(hv & lvals)
        if inter < MIN_DOCS or inter < 0.95 * len(lvals) or inter < 0.5 * len(hv):
            continue
        per = lnn / max(inter, 1)
        if per > MAX_LINES_PER_DOC:
            continue
        head_only = [c for c in tables[ht].columns if c not in tables[lt].columns and c != hc and c not in SKIP]
        out.append(DocLink(ht, hc, lt, lc, inter, round(per, 1), head_only))
    # one document id per (head, lines) pair: keep the best-covered key
    best: dict[tuple[str, str], DocLink] = {}
    for l in out:
        k = (l.head, l.lines)
        if k not in best or l.docs > best[k].docs:
            best[k] = l
    return list(best.values())


# ---------------------------------------------------------------------------------------------- totals
def _num_cols(df: pd.DataFrame, skip: set) -> list[str]:
    return [c for c in df.columns if c not in skip and c not in SKIP and df[c].dtype.kind in "if" and df[c].notna().any()]


def find_totals(tables: dict[str, pd.DataFrame], links: list[DocLink]) -> list[DocTotal]:
    """Document columns equal to the sum of a line column (or of a product of two, qty x unit price) over the document's
    lines. Documents where both sides are 0 do not count (a 0 = 0 match proves nothing: `Còn nợ` = sum `Giảm giá`)."""
    out = []
    for l in links:
        H = tables[l.head].drop_duplicates(subset=[l.key])
        L = tables[l.lines]
        hcols = [c for c in _num_cols(H, {l.key}) if H[c].nunique() > 1]
        lcols = _num_cols(L, {l.line_key})[:MAX_NUM_LINE_COLS]
        if not hcols or not lcols:
            continue
        key = L[l.line_key].astype(str)
        cands: dict[str, pd.Series] = {f"`{c}`": L[c].fillna(0).groupby(key).sum() for c in lcols}
        for a, b in itertools.combinations(lcols, 2):
            cands[f"`{a}` × `{b}`"] = (L[a].fillna(0) * L[b].fillna(0)).groupby(key).sum()
        hv = H.set_index(H[l.key].astype(str))
        for hc in hcols:
            best = None
            for expr, agg in cands.items():
                j = pd.concat([hv[hc], agg], axis=1, join="inner").dropna()
                j = j[(j.iloc[:, 0] != 0) | (j.iloc[:, 1] != 0)]
                if len(j) < MIN_DOCS:
                    continue
                share = float(np.isclose(j.iloc[:, 0], j.iloc[:, 1], rtol=0.005, atol=1.0).mean())
                if best is None or share > best[0]:
                    best = (share, expr, len(j))
            if best and best[0] >= TOTAL_SHARE:
                out.append(DocTotal(l.head, hc, l.lines, best[1], round(best[0], 3), best[2]))
    return out


# ---------------------------------------------------------------------------------------------- flat tables
TIME_TEXT = re.compile(r"^\s*(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}:\d{2})")


def _time_text(s: pd.Series) -> bool:
    """Dates / times stored as text (`Giờ` = '2025-02-01 06:00:25'): repeated and stored together like a document id,
    but they are not one."""
    v = s.dropna().astype(str).head(200)
    return len(v) > 0 and v.str.match(TIME_TEXT).mean() > 0.8


def _same_within(df: pd.DataFrame, doc: str, col: str, mode) -> bool | None:
    """True when `col` never changes inside a document. Judged only on documents where it is not the common value: a
    returned quantity that is 0 on almost every line is 'constant' everywhere except where it matters (the lot line that
    was returned). None when too few documents show anything."""
    s = df[col]
    informative = (s.notna() & (s != mode)).groupby(df[doc]).any()
    keys = informative[informative].index
    if len(keys) < 10:
        return None
    return float(s.groupby(df[doc]).nunique(dropna=True).loc[keys].le(1).mean()) >= 0.95


def find_doc_level(tables: dict[str, pd.DataFrame], links: list[DocLink]) -> dict[str, tuple[str, list[str]]]:
    """Flat tables (layout B): the document id (repeated, its lines stored together) and the columns that never change
    within a document. Parent tables (one row per id) are skipped: biz_123 `don_hang` has one row per order, and its shift
    id `MaCaLam` would otherwise read as a document of orders."""
    out = {}
    heads = {l.head for l in links}
    for t, df in tables.items():
        if len(df) < 2 * MIN_DOCS or t in heads:
            continue
        best = None
        for c in df.columns:
            if c in SKIP or not _id_like(df[c]) or _time_text(df[c]):
                continue
            s = df[c]
            nn, nu = int(s.notna().sum()), int(s.nunique())
            if nu < MIN_DOCS or nu >= nn or nn < 0.9 * len(df) or nu / nn > 0.9 or _contiguity(s) > CONTIGUOUS:
                continue
            if nn / nu > MAX_LINES_PER_DOC:
                continue
            if best is None or nu > best[1]:
                best = (c, nu)
        if best is None:
            continue
        doc = best[0]
        sub = df[df[doc].notna()]
        sizes = sub.groupby(doc)[doc].transform("size")
        multi = sub[sizes > 1]
        if multi[doc].nunique() < MIN_DOCS:
            continue
        multi = multi.head(60000)          # enough documents to judge; keeps big shops fast
        cols = []
        for c in df.columns:
            if c == doc or c in SKIP or df[c].nunique() <= 1 or df[c].notna().mean() < 0.5:
                continue
            mode = multi[c].mode(dropna=True)
            if mode.empty:
                continue
            if _same_within(multi, doc, c, mode.iloc[0]):
                cols.append(c)
        if cols:
            out[t] = (doc, cols)
    return out


# ---------------------------------------------------------------------------------------------- README notes
BULLET = re.compile(r"^\s*-\s+(?:\*\*([^*]+)\*\*\s*:\s*)?(.+)$")
TERM = re.compile(r"\*\*([^*]{1,60})\*\*|`([^`]{1,80})`")


def readme_notes(readme: str, tables: dict[str, pd.DataFrame], links: list[DocLink]) -> dict[str, dict[str, str]]:
    """README sentences that define a column ("**Cần trả NCC** là tổng tiền của cả phiếu nhập"), per table.

    A clause names a column in bold (`**Col**`) or as `table.Col`. When the column exists in several tables (`Đơn giá` is in
    the sales lines AND the purchase lines), the table is the one whose document link connects it to another column of
    the same bullet (the purchase sentence also names `Cần trả NCC`); still ambiguous -> the note is left out."""
    if not readme:
        return {}
    linked: dict[str, set[str]] = {}
    for l in links:
        linked.setdefault(l.head, set()).add(l.lines)
        linked.setdefault(l.lines, set()).add(l.head)
    has = {t: {str(c) for c in df.columns} for t, df in tables.items()}
    out: dict[str, dict[str, str]] = {}
    for raw in readme.splitlines():
        m = BULLET.match(raw)
        if not m:
            continue
        body = m.group(2)
        refs = []                                 # (table or None, column)
        for b, q in TERM.findall(body):
            if b:
                refs.append((None, b.strip()))
            elif "." in q and not re.search(r"\.(xlsx|xls|csv|sqlite|db)$", q, re.I):
                tq, cq = q.split(".", 1)
                refs.append((resolve_table(tq, tables), cq.strip()))
        cols_here = {c for _, c in refs}
        for clause in re.split(r";\s*|\.\s+(?=\S)", body):
            terms = [(t, c) for t, c in refs if f"**{c}**" in clause or (t and f".{c}`" in clause)]
            if not terms:
                continue
            t0, col = terms[0]
            if t0 is not None:
                cands = [t0] if col in has.get(t0, set()) else []
            else:
                cands = [t for t, cs in has.items() if col in cs]
                if len(cands) > 1:
                    others = cols_here - {col}
                    near = [t for t in cands if any(o in has[u] for u in ({t} | linked.get(t, set())) for o in others)]
                    cands = near
            if len(cands) != 1:
                continue
            text = re.sub(r"\*\*|`", "", clause).strip().rstrip(".;")
            if len(text) > 8:
                out.setdefault(cands[0], {}).setdefault(col, text[:200])
    return out


# ---------------------------------------------------------------------------------------------- all
def detect_structure(tables: dict[str, pd.DataFrame], readme: str = "") -> Structure:
    """Never raises: a failure leaves that part empty (the cards then look as they did before)."""
    st = Structure()
    try:
        st.links = find_links(tables)
    except Exception:  # noqa: BLE001
        st.links = []
    for name, fn in (("totals", lambda: find_totals(tables, st.links)), ("doc_level", lambda: find_doc_level(tables, st.links)),
                     ("notes", lambda: readme_notes(readme, tables, st.links)), ("aliases", lambda: readme_aliases(readme, tables))):
        try:
            setattr(st, name, fn())
        except Exception:  # noqa: BLE001
            pass
    return st


def card_lines(table: str, st: Structure | None) -> tuple[list[str], dict[str, str]]:
    """(lines to append under a table card, {column: suffix for that column's line})."""
    if st is None:
        return [], {}
    lines, col = [], {}
    total_heads = {(t.table, t.lines) for t in st.totals}
    # documents first (a total proves it), then the fewest child rows per parent; at most 3 children and 2 parents per card
    order = sorted(st.links, key=lambda l: ((l.head, l.lines) not in total_heads, l.lines_per_doc))
    for l in [l for l in order if l.head == table][:2]:
        lines.append(f"  * PARENT of `{l.lines}` (`{l.lines}`.`{l.line_key}` = `{l.key}`, ~{l.lines_per_doc} rows each)")
    for l in [l for l in order if l.lines == table][:2]:
        only = ", ".join(f"`{c}`" for c in l.head_only[:5]) + (", ..." if len(l.head_only) > 5 else "")
        lines.append(f"  * CHILD of `{l.head}` (`{l.line_key}` = `{l.head}`.`{l.key}`)"
                     + (f"; only the parent has {only}" if only else ""))
    for t in st.totals:
        if t.table == table and t.column not in col:
            col[t.column] = (f"DOCUMENT TOTAL = sum of {t.expr} of its `{t.lines}` rows: whole-document amount, "
                             "never divide it by one item's quantity")
    if table in st.doc_level:
        doc, cols = st.doc_level[table]
        for c in cols:
            col.setdefault(c, f"PER `{doc}` (same on all its rows)")
    for c, text in st.notes.get(table, {}).items():
        col[c] = (col[c] + "; " if c in col else "") + f'README: "{text}"'
    names = [a for a, t in st.aliases.items() if t == table]
    if names:
        lines.insert(0, "  * README name: " + ", ".join(f"`{a}`" for a in names))
    return lines, col
