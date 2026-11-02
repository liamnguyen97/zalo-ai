"""GUIDELINE: the rulebook rows that decide THIS question, repeated right before the Coder's OUTPUT section, each with the
sandbox column or engine function that applies it in this shop.

The Coder already gets the whole rulebook at the top of its prompt (it stays there: a missed row here is not a lost rule), and
the spec's `rule_refs` rows in the middle. On dev300 q_b445c50f6b5e ("doanh thu ca chiều so với ca sáng") the rule (§2 shifts)
and the way to apply it (`sales_orders['shift']`, `revenue`) were both in the prompt, ~10k tokens above where the model starts
writing and apart from each other, and the Coder still filtered the raw status text and summed the order total column (ledger #9).

Selection is code, from the rulebook's own words (no word list per industry, the principle of agents/question.py):
- each table row's label ("Ca sáng, ca chiều", "Tỉ lệ đơn huỷ trong kỳ") is split into phrases; placeholders of the label
  (X, N, M/Y, <đơn vị>) match a number / a single letter / one word;
- a phrase of 1-2 words must appear as written in the question (folded, telex slips undone, "%" read as "tỉ lệ"); a phrase of 3+
  content words must appear within a short window in any order ("bao nhiêu % đơn bị huỷ" matches "Tỉ lệ đơn huỷ trong kỳ");
- the time section's rows (§2 "Thời gian": "Tháng M/Y", "hôm qua", "N ngày gần nhất"...) are period definitions the Coder never
  applies (it gets the dates as `period_*` / `compare_*`), so they are left out unless they say how to compute (shifts, "quá N",
  balances "đến hết ngày X"); on the 3,000 train questions they fired on 10-11% each and crowded out the rows that matter;
- a row whose match lies inside the match of a more specific row is left out ("bán được bao nhiêu tiền" is the money row, not
  "bán được bao nhiêu <đơn vị>");
- the spec's `rule_refs` rows come first (all kept), then the matched rows, more specific (longer phrase) first, up to `limit`
  rows in all (DATAMATE_GUIDELINE_ROWS, default 4; 0 turns the section off and the spec's rows stay where they were).
Plus up to two shop reminders from the data itself (the order total columns that are not revenue, the status column).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from datamate.agents.question import norm

PH_NUM, PH_WORD = "zzph", "zzunit"      # label placeholders after normalising: X / N / M/Y ..., <đơn vị>
STOP = {"trong", "ky", "cua", "cac", "theo", "la", "va", "mot", "moi", "nhung", "duoc"}
TIME_SECTION = "thoi gian"            # the rulebook section of period definitions (§2)
SYNONYMS = ((r"\bty le\b", "ti le"), (r"\bphan tram\b", "ti le"), (r"\bty trong\b", "ti trong"))


def _norm(text: str) -> str:
    """The comparison form of a question or a label: folded, telex slips undone, "%" / "phần trăm" / "tỷ lệ" -> "ti le"."""
    s = norm(re.sub(r"%", " phần trăm ", text or ""))
    for rx, to in SYNONYMS:
        s = re.sub(rx, to, s)
    return s


@dataclass
class Phrase:
    tokens: tuple[str, ...]          # normalised words, placeholders as PH_NUM / PH_WORD
    content: tuple[str, ...]         # the words that carry meaning (no placeholder, no STOP word)

    def match(self, q: str, qtok: list[str]) -> tuple[int, int] | None:
        """(start, end) character span in the normalised question, or None."""
        if len(self.content) >= 3 and PH_WORD not in self.tokens:   # any order, inside a window a little longer than the phrase
            need, n = set(self.content), len(self.content) + 3
            for i in range(len(qtok)):
                if need <= set(qtok[i:i + n]):
                    hit = [j for j in range(i, min(i + n, len(qtok))) if qtok[j] in need]
                    a = len(" ".join(qtok[:hit[0]])) + (1 if hit[0] else 0)
                    return a, a + len(" ".join(qtok[hit[0]:hit[-1] + 1]))
            return None
        parts = []
        for t in self.tokens:
            parts.append(r"(?:\S+ )?(?:\S*\d\S*|[a-z])" if t == PH_NUM else r"\S+" if t == PH_WORD else re.escape(t))
        m = re.search(r"(?<!\S)" + " ".join(parts) + r"(?!\S)", q)
        return (m.start(), m.end()) if m else None


@dataclass
class Row:
    section: str
    label: str
    text: str                        # the row's rule, as written (bold marks removed)
    phrases: list[Phrase] = field(default_factory=list)

    @property
    def line(self) -> str:
        return f"{self.label}: {self.text}"


def _phrases(label: str) -> list[Phrase]:
    out = []
    for part in re.split(r"[,;]", label):
        part = re.sub(r"<[^>]*>", f" {PH_WORD} ", part)
        part = re.sub(r"(?<![\w])[A-Z](?:\s*[/–-]\s*[A-Z])?(?![\w])", f" {PH_NUM} ", part)
        tokens = tuple(_norm(part).split())
        content = tuple(t for t in tokens if t not in (PH_NUM, PH_WORD) and t not in STOP)
        if not content or (len(content) == 1 and len(content[0]) < 4):
            continue
        out.append(Phrase(tokens, content))
    return out


def parse_rows(rulebook: str) -> list[Row]:
    """The two-column table rows of the rulebook ("| label | rule |"), with their section; header rows skipped."""
    lines, rows, section = (rulebook or "").splitlines(), [], ""
    for i, line in enumerate(lines):
        if line.startswith("#"):
            section = line.lstrip("#").strip()
            continue
        if not line.startswith("|") or re.match(r"^\|\s*-{3}", line):
            continue
        if i + 1 < len(lines) and re.match(r"^\|\s*-{3}", lines[i + 1]):
            continue                                     # header row
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 2 or not cells[0]:
            continue
        label, text = (c.replace("**", "") for c in cells)
        rows.append(Row(section, label, text, _phrases(label)))
    return rows


# How each row is applied in the sandbox: (needed function or None, text with it, text without it). Keyed by the start of the
# normalised label; a row without an entry is shown as written. A function is named only when this shop has it.
REVENUE = "`sales_orders['revenue']` (already §1) or `sales_metrics(start, end)`; never an order total column"
POINTERS: dict[str, tuple[str | None, str, str]] = {
    "ti le don huy trong ky": ("cancel_rate", "`cancel_rate(period_start, period_end)`",
                               "cancelled = `status_std == 'cancelled'` over EVERY order placed in the period (`sales_orders`, every "
                               "status), x 100, 2 decimals"),
    "don bi huy": (None, "`status_std == 'cancelled'` (on `sales_orders` too); returned = `status_std == 'returned'`; `~valid` is both", ""),
    "so don$": (None, "`sales_orders[sales_orders['valid'] & <condition>]['order'].nunique()`", ""),
    "doanh thu mot don": (None, REVENUE, ""),
    "ban duoc bao nhieu tien": (None, REVENUE, ""),
    "doanh thu theo hinh thuc thanh toan": (None, "filter `sales_orders` (valid) by the payment column written on the order, sum `revenue`", ""),
    "doanh thu theo nhom khach": (None, "`sales_orders` (valid) by the customer-group column; `walk_in` orders are in no group; a share "
                                        "divides by the whole shop's revenue (`sales_metrics`, walk-ins included)", ""),
    "ti le thay doi": (None, "(current - previous) / previous x 100, negative for a decrease, 2 decimals", ""),
    "chenh lech": (None, "asked period - compared period (negative when less), in the unit of the measure", ""),
    "ca sang": (None, "`sales_orders['shift']` ('sáng' | 'chiều', from the order creation time): revenue = "
                      "`sales_orders[sales_orders['valid'] & (sales_orders['shift'] == 'sáng')]['revenue'].sum()`; chiều vs sáng = "
                      "(chiều - sáng) / sáng x 100", ""),
    "qua zzph": (None, "'quá / hơn / trên N' -> `> N`, 'dưới N' -> `< N`, 'từ N trở lên' -> `>= N`", ""),
    "den het ngay": (None, "a balance: everything from the first day of the data to the END of that day, not that day only", ""),
    "ban chay": ("top_items", "`top_items(k, start, end, by='qty')`",
                 "rank `item_lines[item_lines['valid'] & item_lines['is_item']]` by the sum of `qty_base`"),
    f"ban duoc bao nhieu {PH_WORD}": (None, "`item_lines` (valid) with `base_unit` equal to that unit, sum `qty_base`", ""),
    "mat hang$": (None, "`item_lines[item_lines['is_item']]` (service-fee lines are not items but stay in revenue)", ""),
    "xac dinh khach": ("top_customers", "`top_customers(k, start, end)` (walk-ins removed)",
                       "group `sales_orders[valid & ~walk_in]` by `customer_key`"),
    "nhan vien ban gioi nhat": (None, "group `sales_orders[valid]` by the staff column and sum `revenue` (not the order count)", ""),
    "gia tri trung binh mot don": ("sales_metrics", "`sales_metrics(start, end)['avg_order']`", "revenue / number of valid orders, whole đồng"),
    "no mot don tai ngay": ("debt_total", "`debt_total(day)`", ""),
    "no theo khach": ("debt_by_customer", "`debt_by_customer(day)`", ""),
    "tuoi no": ("debt_aging", "`debt_aging(day)`", ""),
    "no them": ("debt_new", "`debt_new(start, end)` / `debt_repaid(start, end)`", ""),
    "tien nhap hang trong ky": ("purchase_total", "`purchase_total(start, end)`; the supplier: `top_supplier(start, end)`", ""),
    "gia nhap binh quan": ("purchase_avg_price", "`purchase_avg_price(item, start, end, unit)`", ""),
    "size": ("size_qty", "`size_qty(size, start, end)`", ""),
    "don vi co ban": (None, "`qty_base` / `base_unit` of `item_lines`; `item_qty` returns base units (a bigger unit: convert with the README)", ""),
}
KEYS = sorted(POINTERS, key=len, reverse=True)


COLUMNS = {"status_std", "shift", "walk_in", "customer_key", "revenue", "valid", "is_item", "qty_base", "base_unit", "order"}


def pointer(row: Row, have: set[str], columns: set[str] | None = None) -> str:
    """How to apply the row here ("" when there is no entry). A key ending in "$" must equal the whole label. `columns`: the
    columns of this shop's `sales_orders` / `item_lines`; a pointer naming one it lacks is not shown (a shop whose orders carry
    no creation time has no `shift`: pointing there would only cost a KeyError block). None = not checked."""
    lab = " ".join(_norm_label(row.label))
    key = next((k for k in KEYS if (lab == k[:-1] if k.endswith("$") else lab == k or lab.startswith(k + " "))), None)
    if key is None:
        return ""
    fn, with_fn, without = POINTERS[key]
    text = with_fn if fn is None or fn in have else without
    if columns is not None:
        named = {w for span in re.findall(r"`([^`]*)`", text) for w in re.findall(r"\w+", span) if w in COLUMNS}
        if named - set(columns):
            return ""
    return text


def _norm_label(label: str) -> list[str]:
    label = re.sub(r"<[^>]*>", f" {PH_WORD} ", label)
    label = re.sub(r"(?<![\w])[A-Z](?:\s*[/–-]\s*[A-Z])?(?![\w])", f" {PH_NUM} ", label)
    return _norm(label).split()


@dataclass
class Entry:
    label: str
    text: str
    how: str = ""
    source: str = "match"            # spec (the Explainer's rule_refs) | match (code) | shop (reminder from the data)


def select(question: str, as_of: str, rows: list[Row], spec_rules: list[str] | None = None, have=(), sales_orders=None,
           measure: str | None = None, limit: int = 4, columns: set[str] | None = None) -> list[Entry]:
    """The GUIDELINE entries for one question (see the module doc). `as_of` is kept for callers; periods are not re-read here."""
    have, q = set(have or ()), _norm(question)
    qtok = q.split()
    by_line = {_norm(r.line): r for r in rows}
    out: list[Entry] = []
    seen: set[str] = set()

    def add(row: Row | None, text: str, source: str) -> None:
        key = row.label if row else _norm(text)
        if key in seen:
            return
        seen.add(key)
        out.append(Entry(row.label, row.text, pointer(row, have, columns), source) if row else Entry("", text, "", source))

    by_label = {_norm(r.label): r for r in rows}
    for line in spec_rules or []:                        # the Explainer's rule_refs rows (rulebook lines as written)
        line = str(line).strip()
        if line.startswith("#"):
            continue                                     # a section heading says nothing by itself (the rulebook is in CONTEXT)
        cells = [c.strip() for c in line.strip("|").split("|")]
        as_row = by_line.get(_norm(f"{cells[0]}: {cells[1]}")) if len(cells) == 2 else None
        if as_row is None:                               # "| label | ... |", or "1. Doanh thu một đơn: tổng ..." (section number, colon)
            label = re.sub(r"^\s*§?\d+(?:\.\d+)*\.?\s+", "", cells[0].replace("**", "")).split(":")[0]
            as_row = by_label.get(_norm(label))
        add(as_row, line.replace("**", ""), "spec")
    scored = []
    for i, row in enumerate(rows):
        if TIME_SECTION in _norm(row.section) and not pointer(row, have):
            continue                                     # a period definition: the Coder gets the dates
        best = None
        for ph in row.phrases:
            span = ph.match(q, qtok)
            if span is not None and (best is None or len(ph.content) > best[0]):
                best = (len(ph.content), span)
        if best:
            scored.append((-best[0], i, best[1], row))
    taken: list[tuple[int, int]] = []
    for _, _, span, row in sorted(scored, key=lambda x: (x[0], x[1])):
        if any(a <= span[0] and span[1] <= b for a, b in taken):
            continue                                     # inside the match of a more specific row
        taken.append(span)
        add(row, row.line, "match")
    spec = sum(1 for e in out if e.source == "spec")     # the spec's rows are all kept; the matched rows fill up to `limit`
    out = out[:max(limit, spec)]
    out += shop_reminders(q, sales_orders, measure)
    return out


def shop_reminders(q: str, sales_orders, measure: str | None) -> list[Entry]:
    """Up to two facts of THIS shop the guard would otherwise refuse after the fact (ledger #9)."""
    out = []
    if sales_orders is None:
        return out
    money = list(getattr(sales_orders, "attrs", {}).get("header_money", ()) or ())
    if money and measure in ("money", "ratio"):
        cols = ", ".join(f"`{c}`" for c in money[:3])
        out.append(Entry("Doanh thu (cửa hàng này)", f"the order columns {cols} are the money WRITTEN on the order (after order "
                         "discounts / fees), not revenue by §1", "`sales_orders['revenue']`", "shop"))
    if "status_std" in getattr(sales_orders, "columns", ()) and re.search(r"\bhuy\b|\btra hang\b|\bhoan\b", q):
        out.append(Entry("Trạng thái đơn (cửa hàng này)", "status labels are spelled many ways in the raw column",
                         "`status_std` ('completed' | 'shipping' | 'cancelled' | 'returned'); `~valid` = cancelled AND returned", "shop"))
    return out


def render(entries: list[Entry]) -> str:
    if not entries:
        return ""
    lines = ["# RULES FOR THIS QUESTION (rulebook rows as written, and how to apply them in this sandbox; the full rulebook is "
             "in CONTEXT)"]
    for e in entries:
        head = f"- {e.label}: {e.text}" if e.label else f"- {e.text}"
        lines.append(head + (f"\n  -> {e.how}" if e.how else ""))
    return "\n".join(lines) + "\n\n"
