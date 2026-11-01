"""Generic per-business loader: .csv / .xlsx / .sqlite -> {name: DataFrame}.

Handles the common export quirks seen in the training data: title rows above the header,
one sheet per month (same columns -> concatenated with ``source_sheet``), numbers written as
"65,000.0000" / "1.250.000đ", mixed day-first dates (incl. Excel serials, "1 thg 1 2025", "d/m"
with the year taken from the sheet/file name), blank order-level cells on continuation rows of
an order (forward-filled within the order), and tables split by month/file/part (merged into
one table with ``source_sheet``). De-duplication of export rows and unit conversion are still
left to the coder + rulebook.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from datamate.data.readme import roles_for_tables
from datamate.data.status import add_status_std, find_order_col

DATE_HINT = re.compile(r"ng[àa]y|date|h[ạa]n|th[ờo]i ?gian|time", re.I)
NUMDAYS_HINT = re.compile(r"s[ốo] ng[àa]y|ng[àa]y n[ợo]|ng[àa]y c[ôo]ng|tu[ổo]i n[ợo]", re.I)  # day *counts*, not dates
FILL_HINT = re.compile(
    r"ng[àa]y|date|th[ờo]i ?gian|kh[áa]ch|customer|tr[ạa]ng th[áa]i|status|k[ếe]t qu[ảa]|ti[ếe]n [đd][ộo]|ghi ch[úu]|note"
    r"|k[êe]nh|channel|nh[âa]n vi[êe]n|thu ng[âa]n|h[ìi]nh th[ứu]c|thanh to[áa]n|payment|[đd][ốo]i t[áa]c", re.I)
ID_HINT = re.compile(r"\bm[ãa]\b|barcode|code|s[ốo] [đd]i[ệe]n tho[ạa]i|sdt|phone|\bid\b", re.I)
MONEY_TR = re.compile(r"\s*(\d+)\s*tr\s*(\d*)\s*", re.I)          # 1tr440 = 1,440,000 ; 2tr = 2,000,000
MONEY_TR_DEC = re.compile(r"\s*(\d+(?:[.,]\d+)?)\s*(?:tr|tri[ệe]u)\s*", re.I)  # "1,19tr" = 1,190,000 ; "2 triệu" = 2,000,000
MONEY_K = re.compile(r"\s*(\d+(?:[.,]\d+)?)\s*k\s*", re.I)          # 55k = 55,000
VI_DATE = re.compile(r"\s*(\d{1,2})\s*(?:thg|th[áa]ng)\s*(\d{1,2})\s*,?\s*(\d{4})\s*", re.I)  # 1 thg 1 2025
DMY = re.compile(r"\s*(\d{1,2})[/.\-](\d{1,2})(?:[/.\-](\d{2,4}))?\s*$")
# Excel exports carry daily/monthly subtotal rows ("Tổng ngày 31/01", "Cộng ngày 31/01/2026", "Tổng tháng 1/2026",
# a bare "Cộng"): no order id, amount = the total. Summing them double-counts (seen: one day = 1.4 billion).
# Deliberately specific: customer surnames such as "Tong Ngoc Chi" must survive.
SUMMARY_ROW = re.compile(
    r"^\s*(?:t[ổo]ng|c[ộo]ng)\s+(?:ng[àa]y|th[áa]ng)\b|^\s*t[ổo]ng\s+\d{1,2}\s*/\s*\d{1,2}|^\s*(?:t[ổo]ng\s*c[ộo]ng|t[ổo]ng|c[ộo]ng)\s*$",
    re.I)
NUM_CLEAN = re.compile(r"[^\d,.\-]")


@dataclass
class Business:
    business_id: str
    readme: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _detect_header(raw: pd.DataFrame, scan: int = 15) -> int:
    """Index of the first row that looks like a header: wide, mostly strings, followed by data."""
    width = raw.shape[1]
    for i in range(min(scan, len(raw))):
        row = raw.iloc[i]
        filled = row.notna().sum()
        strs = sum(isinstance(v, str) for v in row)
        if filled >= max(2, 0.6 * width) and strs >= 0.8 * filled:
            return i
    return 0


def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def _to_number(s: pd.Series) -> pd.Series | None:
    if not _is_text(s):
        return None
    nonnull = s.dropna().astype(str)
    if nonnull.empty:
        return None
    sample = nonnull.head(500)

    def parse(v: str) -> float | None:
        m = MONEY_TR_DEC.fullmatch(v)
        if m:
            return float(m.group(1).replace(",", ".")) * 1_000_000
        m = MONEY_TR.fullmatch(v)
        if m:
            return float(f"{m.group(1)}.{m.group(2) or '0'}") * 1_000_000
        m = MONEY_K.fullmatch(v)
        if m:
            return float(m.group(1).replace(",", ".")) * 1_000
        v = NUM_CLEAN.sub("", v.strip())
        if not v or not re.search(r"\d", v):
            return None
        if v.count(",") and v.count("."):
            # the last separator is the decimal one
            dec = "," if v.rfind(",") > v.rfind(".") else "."
            v = v.replace("," if dec == "." else ".", "").replace(dec, ".")
        elif v.count(",") > 1 or (v.count(",") == 1 and len(v.split(",")[1]) == 3):
            v = v.replace(",", "")
        elif v.count(".") > 1 or (v.count(".") == 1 and len(v.split(".")[1]) == 3 and len(v.split(".")[0]) <= 3):
            v = v.replace(".", "")
        else:
            v = v.replace(",", ".")
        try:
            return float(v)
        except ValueError:
            return None

    # only treat as numeric if values are number-like (not "1 cuộn", names, ids)
    pure = sample.str.fullmatch(r"\s*-?[\d.,\s]+(?:đ|₫|vnd|VND|d)?\s*") | sample.str.fullmatch(MONEY_TR_DEC.pattern, flags=re.I) | sample.str.fullmatch(MONEY_TR.pattern, flags=re.I) | sample.str.fullmatch(MONEY_K.pattern, flags=re.I)
    if pure.mean() < 0.9:
        return None
    # parse each distinct value once (columns are highly repetitive)
    lut = {u: parse(str(u)) for u in s.dropna().unique()}
    return s.map(lut).astype("float64")


def _parse_date_value(v, year_hint: int | None):
    """One cell -> Timestamp/NaT. Handles Excel serials, '1 thg 1 2025', 'd/m' (year from sheet), d/m/yyyy, ISO."""
    if isinstance(v, (pd.Timestamp,)) or pd.isna(v):
        return v if isinstance(v, pd.Timestamp) else pd.NaT
    if hasattr(v, "year") and hasattr(v, "month"):
        return pd.Timestamp(v)
    t = str(v).strip()
    if re.fullmatch(r"\d{4,6}(\.0+)?", t) and 20000 <= float(t) <= 70000:  # Excel serial date
        return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(float(t)))
    m = VI_DATE.fullmatch(t) or DMY.fullmatch(t)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        y = int(y) if y else year_hint
        y = y + 2000 if y is not None and y < 100 else y
        if y is None:
            return pd.NaT
        try:
            return pd.Timestamp(year=y, month=int(mo), day=int(d))
        except ValueError:
            return pd.NaT
    try:
        return pd.to_datetime(t, errors="raise", format="ISO8601")
    except (ValueError, TypeError):
        return pd.NaT


def _parse_dates(s: pd.Series, year_hint: int | None) -> pd.Series:
    lut = {u: _parse_date_value(u, year_hint) for u in s.dropna().unique()}
    return pd.to_datetime(s.map(lut))


def _year_of(*names: str) -> int | None:
    for n in names:
        m = re.search(r"20\d{2}", str(n))
        if m:
            return int(m.group(0))
    return None


def drop_summary_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    mask = pd.Series(False, index=df.index)
    for c in df.columns:  # every text column: in dev_007 the label sits in the 13th column
        s = df[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            mask |= s.astype(str).str.match(SUMMARY_ROW)
    n = int(mask.sum())
    return (df[~mask] if n else df), n


def _clean(df: pd.DataFrame, year_hint: int | None = None) -> pd.DataFrame:
    df = df.dropna(how="all").dropna(axis=1, how="all").copy()
    df, dropped = drop_summary_rows(df)
    df.attrs["summary_rows_dropped"] = dropped
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    for col in df.columns:
        s = df[col]
        if not _is_text(s):
            continue
        s = s.astype(object)
        if ID_HINT.search(col):  # codes/barcodes/phones: keep as text, never as floats
            df[col] = s.map(lambda v: str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() else v)
            continue
        if DATE_HINT.search(col) and not NUMDAYS_HINT.search(col):
            # date-named column: try dates first (it may hold Excel serials that look numeric)
            parsed = _parse_dates(s, year_hint)
            if s.notna().sum() and parsed.notna().sum() / s.notna().sum() >= 0.9:
                df[col] = parsed
                continue
        num = _to_number(s)  # "Số ngày được nợ" holds 30/60, not dates
        if num is not None:
            df[col] = num
    return df.reset_index(drop=True)


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            raw = pd.read_csv(path, header=None, dtype=object, encoding=enc, on_bad_lines="skip")
            break
        except UnicodeDecodeError:
            continue
    h = _detect_header(raw)
    out = raw.iloc[h + 1 :].copy()
    out.columns = raw.iloc[h].fillna("unnamed").astype(str)
    return out


def _read_xlsx(path: Path) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    out = {}
    for name, raw in sheets.items():
        if raw.empty:
            continue
        h = _detect_header(raw)
        df = raw.iloc[h + 1 :].copy()
        df.columns = raw.iloc[h].fillna("unnamed").astype(str)
        out[name] = df
    return out


def _safe(name: str) -> str:
    return re.sub(r"\W+", "_", name, flags=re.UNICODE).strip("_")


_PERIOD = re.compile(r"(?<![a-zA-Z])(?:thang|th[áa]ng|th|t|q|quy|nam|part|phan)?[_\- ]?\d{1,4}(?![a-zA-Z])", re.I)


def _base_name(key: str) -> str:
    """'BanHang_Thang01_2025' / 'SoBanHang__T02_2025' / 'ChiTietHoaDon_1' -> stable base name."""
    return re.sub(r"_+", "_", _PERIOD.sub("", key)).strip("_") or key


def _merge_split_tables(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Concatenate tables that share a base name and (nearly) the same columns: one sheet per
    month, one file per year, parts of a big export. ``source_sheet`` records the origin."""
    groups: dict[str, list[str]] = {}
    for key in frames:
        groups.setdefault(_base_name(key), []).append(key)
    out: dict[str, pd.DataFrame] = {}
    for base, keys in groups.items():
        if len(keys) == 1:
            out[keys[0]] = frames[keys[0]]
            continue
        ref = set(frames[keys[0]].columns)
        same = [k for k in keys if len(ref & set(frames[k].columns)) / len(ref | set(frames[k].columns)) >= 0.8]
        rest = [k for k in keys if k not in same]
        if len(same) > 1:
            out[base] = pd.concat([frames[k].assign(source_sheet=k) if "source_sheet" not in frames[k] else frames[k] for k in same],
                                  ignore_index=True)
        else:
            rest = keys
        for k in rest:
            out[k] = frames[k]
    return _merge_across_files(out)


def _cols(df: pd.DataFrame) -> set[str]:
    return set(df.columns) - {"source_sheet"}


def _merge_across_files(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Second pass: one table split over several workbooks whose sheets are keyed differently, e.g.
    `BanHang_TongHop_2025.xlsx` (5 sheets -> `..._2025__BanHang`) and `BanHang_TongHop_2026.xlsx` (one sheet ->
    `..._2026`). Same file family (file stem without period tokens) and nearly the same columns = one table."""
    fam: dict[str, list[str]] = {}
    for k in frames:
        fam.setdefault(_base_name(k.split("__")[0]), []).append(k)
    out, done = {}, set()
    for keys in fam.values():
        for i, a in enumerate(keys):
            if a in done:
                continue
            group = [a]
            for b in keys[i + 1:]:
                if b in done or a.split("__")[0] == b.split("__")[0]:  # same workbook: different sheets stay apart
                    continue
                ca, cb = _cols(frames[a]), _cols(frames[b])
                if ca and len(ca & cb) / len(ca | cb) >= 0.8:
                    group.append(b)
            done.update(group)
            if len(group) == 1:
                out[a] = frames[a]
                continue
            parts = [frames[k] if "source_sheet" in frames[k] else frames[k].assign(source_sheet=k) for k in group]
            name = next((k for k in group if "__" in k), group[0])  # keep the sheet name: `<file>__<sheet>`
            out[_base_name(name.split("__")[0]) + ("__" + name.split("__", 1)[1] if "__" in name else "")] = pd.concat(parts, ignore_index=True)
    return out


def _fill_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Rulebook §8: blank order-level cells on later rows of an order equal the first row's."""
    oc = find_order_col(df)
    if oc is None:
        return df
    ids = df[oc].ffill()
    if ids.isna().mean() > 0.05 or ids.nunique() < 2:
        return df
    run = (ids != ids.shift()).cumsum()  # consecutive rows of one order (ids may repeat across months)
    for col in df.columns:
        if col == oc or not FILL_HINT.search(col) or df[col].dtype.kind in "iuf":
            continue
        if df[col].isna().any():
            df[col] = df[col].groupby(run).ffill()
    return df


def drop_consecutive_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rulebook §8: two lines of the same order identical in every cell = export error, count once.

    Only a row identical to the row right before it is dropped. Measured on the training businesses: where
    deduplication fixed the totals, every duplicate was adjacent; where it hurt, none was (those are genuine
    repeated lines of an order), so a blanket drop_duplicates() would be wrong.
    """
    oc = find_order_col(df)
    if oc is None or len(df) < 2 or df[oc].nunique() >= 0.9 * len(df):  # only line-level tables
        return df, 0
    cols = [c for c in df.columns if c not in ("source_sheet", "status_std", "valid_order")]
    same = (df[cols].eq(df[cols].shift()) | (df[cols].isna() & df[cols].shift().isna())).all(axis=1) & df[oc].notna()
    n = int(same.sum())
    return (df[~same].reset_index(drop=True) if n else df), n


def load_business(biz_dir: str | Path) -> Business:
    biz_dir = Path(biz_dir)
    readme_path = biz_dir / "README.md"
    biz = Business(biz_dir.name, readme_path.read_text(encoding="utf-8") if readme_path.exists() else "")

    frames: dict[str, pd.DataFrame] = {}
    dropped: dict[str, int] = {}
    for path in sorted(biz_dir.iterdir()):
        stem = _safe(path.stem)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frames[stem] = _clean(_read_csv(path), _year_of(path.stem))
            dropped[_base_name(stem)] = dropped.get(_base_name(stem), 0) + frames[stem].attrs.get("summary_rows_dropped", 0)
        elif suffix in (".xlsx", ".xlsm", ".xls"):
            sheets = _read_xlsx(path)
            for sheet, raw in sheets.items():
                df = _clean(raw, _year_of(sheet, path.stem))
                if not df.empty:
                    key = stem if len(sheets) == 1 else f"{stem}__{_safe(sheet)}"
                    frames[key] = df
                    dropped[_base_name(key)] = dropped.get(_base_name(key), 0) + df.attrs.get("summary_rows_dropped", 0)
        elif suffix in (".sqlite", ".db", ".sqlite3"):
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                names = [r[0] for r in con.execute("select name from sqlite_master where type in ('table','view')")]
                for n in names:
                    frames[f"{stem}__{_safe(n)}"] = _clean(pd.read_sql_query(f'select * from "{n}"', con))
            finally:
                con.close()
    biz.tables = {}
    merged = _merge_split_tables(frames)
    declared = roles_for_tables(biz.readme, {k: list(v.columns) for k, v in merged.items()})
    for k, v in merged.items():
        v, n_dup = drop_consecutive_duplicates(_fill_groups(v))
        if n_dup:
            dropped[f"{k}:dup"] = n_dup
        biz.tables[k] = add_status_std(v, prefer=declared.get(k, {}).get("order_status"))
    biz.notes = [
        (f"`{k[:-4]}`: removed {n} adjacent duplicate rows (an export error; counted once by the rulebook)" if k.endswith(":dup")
         else f"`{k}`: removed {n} summary rows (Tổng ngày/Cộng ngày/Tổng tháng...) because they are not orders")
        for k, n in dropped.items() if n
    ]
    return biz
