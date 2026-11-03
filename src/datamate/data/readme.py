"""Read the column documentation that most businesses ship in README.md.

    ## Cột trong `chi_tiet_don.csv`            (or:  ## Cột trong `DuLieuXuat.xlsx`, sheet `DonHang_Lo`)
    - **Tiền bán**: thành tiền sau giảm

The descriptions are standardised across businesses ("thành tiền sau giảm", "mã đơn", "ngày bán",
"trạng thái đơn", ...), so they tell us what a column *is* far more reliably than its name
(`Tiền bán`, `Trị giá`, `T.Tiền`, `Amount` are all the line amount).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnDoc:
    file_stem: str
    sheet: str | None
    column: str
    desc: str


HEAD = re.compile(r"^##\s+Cột trong\s+`([^`]+)`(?:\s*,\s*sheet\s+`([^`]+)`)?", re.I)
ITEM = re.compile(r"^-\s+\*\*(.+?)\*\*\s*:\s*(.+)$")

ROLES: list[tuple[str, re.Pattern]] = [
    ("line_amount", re.compile(r"^th[àa]nh ti[ềe]n (?:sau gi[ảa]m|c[ủu]a d[òo]ng)|^th[àa]nh ti[ềe]n$", re.I)),
    ("order_id", re.compile(r"^m[ãa] đ[ơo]n$", re.I)),
    # "số thứ tự đơn (hoá đơn) trong sổ": a running number written on the first line of each order (dev_011 `STT`)
    ("order_no", re.compile(r"^s[ốo] th[ứu] t[ựu] (?:c[ủu]a )?(?:đ[ơo]n|h[óo]a đ[ơo]n|ho[áa] đ[ơo]n|phi[ếe]u)", re.I)),
    ("order_date", re.compile(r"^ng[àa]y b[áa]n\b", re.I)),
    ("order_status", re.compile(r"^tr[ạa]ng th[áa]i đ[ơo]n$", re.I)),
    ("qty_base", re.compile(r"đ[ãa] đ[ổo]i ra đ[ơo]n v[ịi] c[ơo] b[ảa]n|s[ốo] l[ưu][ợo]ng.*c[ơo] b[ảa]n", re.I)),
    ("item_name", re.compile(r"^t[êe]n h[àa]ng\b", re.I)),
    ("item_code", re.compile(r"^m[ãa] h[àa]ng$", re.I)),
    ("qty", re.compile(r"^s[ốo] l[ưu][ợo]ng$", re.I)),
    ("unit", re.compile(r"^đ[ơo]n v[ịi] t[íi]nh$", re.I)),
    ("unit_price", re.compile(r"^đ[ơo]n gi[áa]$", re.I)),
    ("base_unit", re.compile(r"^đ[ơo]n v[ịi] c[ơo] b[ảa]n$", re.I)),
    ("item_group", re.compile(r"^nh[óo]m h[àa]ng$", re.I)),
    # "mã khách (trống nếu khách lẻ)": the column is called `Mã người mua`, `Mã khách mua`, `Mã thẻ`..., the description is the standard one
    ("customer_code", re.compile(r"^m[ãa] kh[áa]ch\b", re.I)),
    # "hình thức thanh toán của đơn: cash tiền mặt, transfer chuyển khoản, ..." (dev_008's column is `Thanh toán bằng`)
    ("payment_method", re.compile(r"^(?:h[ìi]nh th[ứu]c|ph[ưu][ơo]ng th[ứu]c) thanh to[áa]n\b", re.I)),
    ("conv_ratio", re.compile(r"^h[ệe] s[ốo]\b", re.I)),
    ("large_unit", re.compile(r"^đ[ơo]n v[ịi] l[ớo]n\b|^đ[ơo]n v[ịi] quy đ[ổo]i", re.I)),
]

# "**Đơn**: các dòng cùng **STT** là một đơn" (dev_014): the column whose equal values make one order
SAME_ORDER = re.compile(r"c[áa]c d[òo]ng (?:c[óo] )?c[ùu]ng \*\*([^*]{1,40})\*\* (?:l[àa]|thu[ộo]c) (?:m[ộo]t|1) "
                        r"(?:đ[ơo]n|h[óo]a đ[ơo]n|ho[áa] đ[ơo]n|phi[ếe]u)", re.I)
# "mỗi dòng một lượt xe vào sửa (một phiếu)", "mỗi dòng là một đơn": each row is a whole document. Not "mỗi dòng một mặt hàng ...",
# which is one LINE of a document (biz_107 / dev_011)
ROW_DOCUMENT = re.compile(r"m[ỗo]i d[òo]ng (?:l[àa] )?(?:m[ộo]t|1) (?:l[ưu][ợo]t|phi[ếe]u|đ[ơo]n|h[óo]a đ[ơo]n|ho[áa] đ[ơo]n|giao d[ịi]ch)\b",
                          re.I)


def same_order_cols(readme: str) -> set[str]:
    """Columns the README says group rows into one order ("các dòng cùng **STT** là một đơn")."""
    return {m.group(1).strip() for m in SAME_ORDER.finditer(readme or "")}


def row_documents(readme: str) -> bool:
    """The README says each row is one whole document (an order, a repair slip)."""
    return bool(ROW_DOCUMENT.search(readme or ""))


# README sentences such as "- Bia Tiger lon: 1 thùng = 24 lon" (conversion list written in the Quy ước section)
CONV_LINE = re.compile(r"^\s*-\s*(.+?):\s*1\s+([^=]+?)\s*=\s*([\d.,]+)\s*(.+?)\s*$")


def parse_conversions(readme: str) -> list[tuple[str, str, float, str]]:
    """[(item name, large unit, ratio, base unit)] from README conversion lines (cleaned of '4m3'-style tails)."""
    out = []
    for line in readme.splitlines():
        m = CONV_LINE.match(line)
        if m:
            try:
                out.append((m.group(1).strip(), m.group(2).strip(), float(m.group(3).replace(",", ".")), m.group(4).strip()))
            except ValueError:
                pass
    return out


def parse_column_docs(readme: str) -> list[ColumnDoc]:
    out, cur = [], None
    for line in readme.splitlines():
        h = HEAD.match(line.strip())
        if h:
            cur = (re.sub(r"\.[A-Za-z0-9]+$", "", h.group(1)).strip(), h.group(2))
            continue
        if line.startswith("## "):
            cur = None
            continue
        m = ITEM.match(line.strip())
        if m and cur:
            out.append(ColumnDoc(cur[0], cur[1], m.group(1).strip(), m.group(2).strip()))
    return out


def roles_by_section(readme: str) -> dict[tuple[str, str | None], dict[str, str]]:
    """{(file_stem, sheet): {role: column}} using the README's own descriptions."""
    res: dict[tuple[str, str | None], dict[str, str]] = {}
    for d in parse_column_docs(readme):
        for role, pat in ROLES:
            if pat.search(d.desc):
                res.setdefault((d.file_stem, d.sheet), {}).setdefault(role, d.column)
    return res


def _cn(c: str) -> str:
    return re.sub(r"\s+", " ", str(c)).strip().lower()


def roles_for_tables(readme: str, columns: dict[str, list[str]], min_cover: float = 0.6) -> dict[str, dict[str, str]]:
    """{table: {role: column}}: each loaded table takes the README section whose documented columns it contains.

    Matching by columns, not by file name: README headings use placeholders (`BanHang_T<tháng>_<năm>.xlsx`), name a
    workbook whose single sheet the loader keyed differently, or describe tables the loader merged. On the public
    split, name matching missed most sections and the engines fell back to guessed columns.
    """
    sections: dict[tuple[str, str | None], list[ColumnDoc]] = {}
    for d in parse_column_docs(readme):
        sections.setdefault((d.file_stem, d.sheet), []).append(d)
    out: dict[str, dict[str, str]] = {}
    for table, cols in columns.items():
        actual = {_cn(c): c for c in cols}
        best = None
        for docs in sections.values():
            names = {_cn(d.column) for d in docs}
            matched = len(names & actual.keys())
            cover = matched / len(names) if names else 0.0
            if cover >= min_cover and (best is None or (matched, cover) > best[0]):
                best = ((matched, cover), docs)
        if best is None:
            continue
        roles: dict[str, str] = {}
        for d in best[1]:
            col = actual.get(_cn(d.column))
            if col is None:
                continue
            for role, pat in ROLES:
                if pat.search(d.desc):
                    roles.setdefault(role, col)
        if roles:
            out[table] = roles
    return out
