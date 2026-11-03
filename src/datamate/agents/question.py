"""QUESTION READING IN CODE: what the question names, asks for and when, read before any model call.

The Explainer used to spend model calls on this: a turn asking for lookups (6 of 30 turns on the first local runs did only
that), an independent "blind" model reading of the answer type, and a clarity-check model reading of what the spec lost or
added. Three things in a question can be read without a model, in a way that does not depend on the kind of shop:

  names    the phrases of the question that are values of the shop's own data (item, group, customer, unit, channel...).
           No word list: the shop's values are the dictionary, so a new kind of shop brings its own.
  type, k  the answer type from Vietnamese question grammar ("bao nhiêu %", "ngày nào", "mặt hàng nào", "có ... không",
           "top 3"). Grammar words, not shop words. On the 2,930 answerable training questions: 95.2% right overall and
           98-100% right whenever a specific rule fires (only the default "number" is weaker, 93.7%); k right on 136 of 142.
  periods  calendar phrases the rulebook defines (agents/periods.py), listed for the model.

`sure` marks a reading precise enough to override the model; the default reading never does.
Telex typing left in a word ("naof", "nhaats", "khoong") is undone first (from zalo-challenge `norm_q`).
"""
from __future__ import annotations

import re
import threading
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from datamate.agents.periods import BALANCE, scan
from datamate.engines.items import UNIT as UNIT_COL
from datamate.engines.items import cn
from datamate.lexicon import ENTRIES as _LEXICON
from datamate.lexicon import glossary
from datamate.text import SYLLABLE, fold_text, norm  # noqa: F401 - norm is re-exported: facts.py and guideline.py import it from here

# words the readers know: the lexicon's standard forms, and the abbreviations agents/periods.py reads itself ("th 3/25", "so vs")
LEXICON_WORDS = {w for _, std, _ in _LEXICON for w in fold_text(std).split()} | {"th", "vs"}

# ---------------------------------------------------------------------------------------------- normalising
# `norm` (telex slips undone) lives in text.py: agents/periods.py reads the question with it too


# Question grammar and time phrases: removed before names are looked up, so "bao nhiêu" does not match the unit "bao",
# "phần trăm" the unit "phần" or "từ tháng" the customer "Từ Thắng Hải" (all three seen on train questions).
GRAMMAR = re.compile(
    r"\b(?:bao nhieu|bao nhiu|bnhiu|bn|phan tram|ti le|ty le|ti trong|ty trong|la gi|la bao nhieu|cho hoi|xem giup|coi giup|"
    r"nhieu nhat|it nhat|cao nhat|thap nhat|lon nhat|nho nhat|tong cong|tat ca|cua hang|so voi|cung ky|tang hay giam|"
    r"co phai|hay khong|phai khong|dung khong|"
    # the measures and objects the question is about are not values of the data: "dthu hả" -> "doanh thu ha" matched a customer
    # "Thu Hà", "nhân viên" a unit "viên", "đơn hàng" the value "Trạng thái đơn hàng" (the lexicon writes "dthu", "nv", "đh" in full)
    r"doanh thu|don hang|nhan vien)\b")
TIME = re.compile(
    r"\b(?:tu|den|toi|qua|truoc|sau|trong|cuoi|dau|het)? ?(?:ngay|thang|quy|tuan|nam|hom|bua|th|t|q)\s*\d{1,2}(?:\s*[/-]\s*\d{1,4})*\b|"
    r"\b\d{1,2}\s*/\s*\d{1,2}(?:\s*/\s*\d{2,4})?\b|"
    r"\b(?:tu|den|toi|qua) (?:ngay|thang|quy|tuan|nam|hom|dau|cuoi|het)\b|"
    r"\b(?:hom|bua|tuan|thang|nam|quy) (?:nay|qua|truoc|roi|sau|ngoai|kia)\b|\b\d+ ngay\b|\bhnay\b|\bhqua\b")


# ---------------------------------------------------------------------------------------------- names
@dataclass
class Name:
    phrase: str                       # as folded in the question
    how: str                          # "exact" | "inside one value"
    hits: list[tuple[str, str, str]]  # (table, column, value as stored)

    @property
    def value(self) -> str:
        return self.hits[0][2]

    @property
    def words(self) -> int:
        return len(self.phrase.split())


_INDEX: dict[int, dict] = {}
_LOCK = threading.Lock()
MAX_DISTINCT = 20000


def name_index(tables: dict) -> dict:
    """folded value -> {(table, column, value)} over the text columns of a shop (built once per shop, 1-15 s)."""
    key = id(tables)
    with _LOCK:
        if key in _INDEX:
            return _INDEX[key]
        idx: dict = defaultdict(set)
        for t, df in tables.items():
            for c in df.columns:
                s = df[c]
                if s.dtype.kind in "iufbM" or str(s.dtype).startswith("datetime") or c in ("source_sheet", "status_std"):
                    continue
                u = s.dropna().astype(str).unique()
                if len(u) > MAX_DISTINCT:
                    continue
                head = norm(str(c))
                for v in u:
                    f = norm(v)
                    # a value equal to its own column name is a header row inside the data ("Tên khách hàng" in `Tên khách hàng`)
                    if len(f) >= 3 and re.search(r"[a-z]", f) and f != head:
                        idx[f].add((t, str(c), v))
        _INDEX[key] = dict(idx)
        return _INDEX[key]


_VOCAB: dict[int, set] = {}
_UNITS: dict[int, frozenset] = {}


def shop_units(tables: dict) -> frozenset:
    """The units the shop sells in (values of its unit columns, `ĐVT` / `Đơn vị tính`..., same columns as engines/items.py), folded:
    "bao nhiêu kg / đĩa / viên / lon" is a count in this shop's units (ticket T4.1; train: 126 such questions, all numbers)."""
    key = id(tables)
    if key not in _UNITS:
        out = set()
        for df in tables.values():
            for c in df.columns:
                if UNIT_COL.match(cn(str(c))) and df[c].dtype.kind not in "iufbM":
                    for v in df[c].dropna().astype(str).unique()[:500]:
                        f = norm(v)
                        if f.replace(" ", "").isalpha() and 1 <= len(f.split()) <= 2 and len(f) <= 12:
                            out.add(f)
        _UNITS[key] = frozenset(out)
    return _UNITS[key]


def unknown_words(question: str, tables: dict) -> list[str]:
    """Words of the question the code cannot place: not a Vietnamese syllable (text.SYLLABLE), not a word of the shop's data, not a
    standard form of the lexicon. Logged per question (ticket T3.1): new spellings show in the run logs ("o đờ" was found only when
    a question went wrong). A teencode word that is a syllable ("kon") is not caught here: scripts/oov_discovery.py ranks those."""
    key = id(tables)
    if key not in _VOCAB:
        _VOCAB[key] = {w for f in name_index(tables) for w in f.split()}
    known = _VOCAB[key] | LEXICON_WORDS
    out = []
    for w in norm(question).split():
        if w.isalpha() and len(w) >= 2 and not SYLLABLE.match(w) and w not in known and w not in out:
            out.append(w)
    return out


INSIDE_MIN_WORDS = 3   # a run of words found INSIDE a longer value needs this many words


def find_names(question: str, tables: dict, max_n: int = 8) -> list[Name]:
    """Longest phrases of the question (grammar and time phrases removed) that are a whole data value, or a run of >= 3
    words covering >= half of exactly one data value ("rang muối 200g" -> "Đậu phộng rang muối 200g").

    Two words inside a longer value were wrong on 29 of the 30 such matches of the train + public questions: common pairs are part
    of people's names and status labels ("bình quân" -> customer "Nghiêm Bình Quân", "trung bình" -> "Triệu Trung Bình", "là những" ->
    "La Nhung Phong", "nhập hàng" -> "Đã nhập hàng", "điện thoại" -> the product "Dây đeo điện thoại"), and a name the spec then
    "lost" became an issue the model had to fix. A two-word value written whole ("Bánh tiramisu") is still an exact match.
    The price, on purpose: two words naming one product inside its longer name ("mì omachi" -> "Mì Omachi sườn hầm", "cát xây" -> "Cát
    xây tô") are no longer matched in code; the spec call still sees the schema and finds them itself."""
    idx = name_index(tables)
    text = TIME.sub(" | ", GRAMMAR.sub(" | ", norm(question)))
    out: list[Name] = []
    for chunk in text.split("|"):
        toks = chunk.split()
        taken = [False] * len(toks)
        for n in range(min(max_n, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                if any(taken[i:i + n]):
                    continue
                ph = " ".join(toks[i:i + n])
                hits, how = idx.get(ph), "exact"
                if not hits and n >= INSIDE_MIN_WORDS:
                    inside = [f for f in idx if f" {ph} " in f" {f} " and n >= 0.5 * len(f.split())]
                    if len({v for f in inside for (_, _, v) in idx[f]}) == 1:
                        hits, how = idx[inside[0]], "inside one value"
                if hits:
                    taken[i:i + n] = [True] * n
                    out.append(Name(ph, how, sorted(hits)))
    return out


# ---------------------------------------------------------------------------------------------- type and k
NUM_WORDS = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}
OBJ = r"(?:mat hang|hang|san pham|sp|mon|thuoc|khach(?: hang)?|nhan vien|nv|nha cung cap|ncc|kenh|nhom|loai|size|xe|tho|lo|ngay|thang)"
PERCENT = re.compile(r"%|phan tram|\bti le\b|\bty le\b|\bti trong\b|\bty trong\b|chiem (?:bao nhieu|bn|bnhiu|may|nhieu)")
DATE = re.compile(r"\b(?:ngay|hom|bua|thang|tuan|nam|quy) nao\b|\bkhi nao\b|\bluc nao\b|\bhan (?:su dung|dung) (?:nao|gan nhat|som nhat)")
BOOL = re.compile(r"\b(?:co|da|duoc|vuot|cao hon|thap hon|nhieu hon|it hon|tang|giam)\b.*\b(?:khong|ko|k|hong)\s*$|"
                  r"\bphai khong\b|\bdung khong\b|\bhay khong\b|\bhay chua\b|\bchua\s*$")
WHICH = re.compile(r"\b(?:mat hang|hang|san pham|sp|mon|thuoc|khach(?: hang)?|nhan vien|nv|nha cung cap|ncc|kenh|nhom|loai|size|xe|tho|ca|lo)"
                   r"\b[^?]{0,40}?\bnao\b|\bai\b")
WHAT = re.compile(r"\bla (?:nhung )?(?:gi|mon gi|hang gi|ai|nhung (?:mon|hang|mat hang|san pham|sp) nao)\b|\bnhat\b[^?]{0,30}\bla gi\b|\btop ?\d")
# "top 3", "3 sp", "ba khách": the number is k. Not a number that follows "/" or "-" ("tháng 1/26 sp nào" is not 26 products) and, for a
# time unit, not a window ("7 ngày gần nhất", "trong 90 ngày tới", "trên 15 ngày", "0 7 ngay": ages of a debt): see `_k`
TOP_N = re.compile(r"\btop ?(\d{1,2})\b")
N_OBJ = re.compile(r"(?<![\d/.,\-])\b(\d{1,2}|" + "|".join(NUM_WORDS) + r") (" + OBJ + r")\b")
WINDOW_BEFORE = {"tren", "duoi", "qua", "hon", "trong", "sau", "moi", "cach", "khoang", "it", "den", "toi", "tu", "vong", "ke", "gan"}
WINDOW_AFTER = re.compile(r" (?:qua|toi|gan|ke tu|lien truoc|truoc|dau|cuoi|sau|vua|roi|tiep|nua|lai)\b")
LISTED_AFTER = re.compile(r" (?:\d|lien nhau|lien tiep)")      # "3 tháng 7, 8, 9", "3 tháng liền nhau": the count of the months asked for
LIST = re.compile(r"\bnhung\b|\bcac\b.*\bnao\b|\bliet ke\b|\bke ten\b|\bdanh sach\b|\btheo thu tu\b|\blan luot\b|\btung\b.*\bnao\b")
TABLE = re.compile(r"\bkem\b|\bva so (?:no|tien|luong)\b|\bva (?:doanh thu|so tien|tien no|so no)\b|\bvoi so\b|"
                   r"\b(?:moi|tung) (?:khach|mat hang|nguoi|ngui|tho|nhan vien|nv)\b.*\b(?:bao nhieu|bao nhiu|bnhiu|bn|nhieu|nhiu)\b")
# a table asked for outside "which ... ?": "cho bảng 3 khách còn nợ nhiều nhất", "liệt kê 3 khách nợ nhiều nhất, kèm số tiền còn nợ".
# "bảng" is read on the written text: folded, it is also "bằng" ("trả bằng chuyển khoản")
BANG = re.compile(r"(?:^|[,.;:!?]\s*|\b(?:cho|lập|làm|kẻ|vẽ|xin|xuất|dạng|thành|cái|một|1)\s+)bảng\b|"
                  r"\b(?:cho|lap|xin|cai|mot)\s+bang\b(?!\s+(?:chuyen|tien mat|tm|cod|vi|the|ck|tien)\b)")
KEM_SO = re.compile(r"\bkem (?:theo )?(?:so )?(?:tien|no|luong|doanh thu)\b")
_COUNTED = r"(?:mat hang|hang|san pham|sp|mon|loai|khach|nguoi|don|lan|ngay|ly|ca|xe|phieu)"
# "bao nhiêu / mấy <object>": a count, which keeps "which" from reading the question as a name. Not a bare "nhiều <object>": in a
# question with "nào" / "ai" it qualifies the name asked for ("ai mua nhiều lần nhất", "khách nào mua nhìu đơn nhứt"): train has 11,
# all a name, which this count used to block. "nhiêu" alone (= "bao nhiêu") is written in full by the lexicon first
COUNT = re.compile(rf"\b(?:bao nhieu|may|bn|bnhiu) {_COUNTED}\b")
# "bao nhiêu / mấy <object>" COUNTS (train: 304 of 307 are numbers; the 3 others ask two stock quantities "mấy xe 4m3 chẵn, mấy m3 lẻ"). Not the
# bare "nhiều" of COUNT: "ai mua nhiều đơn nhất" is a person
COUNT_SURE = re.compile(r"\b(?:bao nhieu|bao nhiu|may|bn|bnhiu) (?:mat hang|hang|san pham|sp|mon|loai|khach|nguoi|don|lan|ngay|ly|ca|xe|phieu)\b")
ASKS = re.compile(r"\b(?:bao nhieu|bao nhiu|may|bn|bnhiu)\b")
LIST_NUMBERS = re.compile(r"\b(?:tung thang|moi thang|tung ngay|lan luot|tung quy)\b")
# C6: two named groups and "mỗi <group>" ("dthu các đơn trả bằng cod và các đơn trả bằng tm mỗi loại bao nhiêu"): one number each
# (train: 7 of 7 lists, read as one number before). Not "mỗi loại" alone: "số ly size M, tính mỗi loại đồ uống" is one total (9 of 9)
EACH_OF = re.compile(r"\bva\b.*\bmoi (?:ben|cai|loai|kenh|ca|nhom)\b")


@dataclass
class Reading:
    answer_type: str
    k: int | None
    rule: str
    sure: bool                         # a specific rule fired (98-100% precise on train); the default never overrides


def _k(s: str) -> int | None:
    """The first "top N" / "N <object>" of the folded question that really counts elements (see N_OBJ)."""
    cands = [(m.start(), m, True) for m in TOP_N.finditer(s)] + [(m.start(), m, False) for m in N_OBJ.finditer(s)]
    for _, m, is_top in sorted(cands, key=lambda c: c[0]):
        if not is_top and m.group(2) in ("ngay", "thang"):
            after, before = s[m.end():], s[:m.start()].split()
            prev = before[-1] if before else ""
            if prev.isdigit():
                continue                                   # "0 7 ngay 8 15 ngay": the ends of age ranges
            if not (m.group(2) == "thang" and LISTED_AFTER.match(after)) and (prev in WINDOW_BEFORE or WINDOW_AFTER.match(after)):
                continue                                   # a window of time, not a number of elements
        v = m.group(1)
        return int(v) if v.isdigit() else NUM_WORDS.get(v)
    return None


def _asks_units(s: str, units: frozenset) -> bool:
    return any(re.search(rf"\b(?:bao nhieu|may|bn|bnhiu) {re.escape(u)}\b", s) for u in units)


def read_type(question: str, names: list[Name] | None = None, units: frozenset | None = None) -> Reading:
    """Answer type and k from the question's grammar. Item names of >= 2 words are removed first ('Otrivin 0,1% xịt mũi')."""
    s = norm(question)
    for n in sorted((n for n in names or [] if n.words >= 2), key=lambda n: -len(n.phrase)):
        s = s.replace(n.phrase, " ")
    s = re.sub(r"\s+", " ", s).strip()
    k = _k(s)
    raw = unicodedata.normalize("NFC", (question or "").lower())     # "bảng" is told from "bằng" on the written text (BANG)
    asked_table = bool(BANG.search(raw))
    if PERCENT.search(s):
        return Reading("percent", k, "percent words", True)
    if DATE.search(s):
        return Reading("date", k, "which day / month", True)
    if LIST_NUMBERS.search(s):
        return Reading("list", k, "one number per period", True)
    if (WHICH.search(s) or WHAT.search(s)) and not COUNT.search(s):
        if TABLE.search(s) or asked_table:
            return Reading("table", k, "which, with amounts", True)
        if (k and k > 1) or LIST.search(s):
            return Reading("list", k, "which, several", True)
        return Reading("string", k, "which, one", True)
    if asked_table or (KEM_SO.search(s) and ((k and k > 1) or LIST.search(s))):
        return Reading("table", k, "a table asked for", True)
    if BOOL.search(s):
        return Reading("boolean", k, "yes / no", True)
    if EACH_OF.search(s):
        return Reading("list", k, "one number per group", True)
    if len(ASKS.findall(s)) == 1 and (COUNT_SURE.search(s) or (units and _asks_units(s, units))):
        return Reading("number", k, "how many", True)
    return Reading("number", k, "default", False)


# ---------------------------------------------------------------------------------------------- periods
@dataclass
class PeriodMention:
    label: str
    start: date
    end: date


def mentioned_periods(question: str, as_of: str) -> list[PeriodMention]:
    """Calendar phrases in reading order, with the rulebook's dates (§2: a month without a year is this year's unless it
    lies after today). Windows, balances and comparisons are not resolved here: they stay the model's (and repair_spec's).
    The reading is agents/periods.scan, the one calendar_period forces periods with."""
    q = norm(question)
    hits, rest = scan(q, date.fromisoformat(as_of[:10]))
    if BALANCE.search(rest):       # a balance "cuối ngày hôm nay" runs from the start of the data: not named as a one-day period
        hits = [h for h in hits if h.kind not in ("day", "to_date")]
    return [PeriodMention(h.label, h.start, h.end) for h in hits]


# ---------------------------------------------------------------------------------------------- all of it
@dataclass
class QuestionFacts:
    names: list[Name] = field(default_factory=list)
    reading: Reading | None = None
    periods: list[PeriodMention] = field(default_factory=list)
    glossary: list[tuple[str, str]] = field(default_factory=list)    # datamate/lexicon.py: (word as written, meaning)
    unknown: list[str] = field(default_factory=list)                 # unknown_words: logged only

    def prompt(self) -> str:
        """The section the spec call reads: facts the code is sure of, to use instead of guessing."""
        lines = ["DATA MATCHES (phrases of the question that are values in the shop's data; use these exact spellings):"]
        for n in self.names:
            where = "; ".join(f"`{t}`.`{c}` = {v!r}" for t, c, v in n.hits[:3]) + (" ..." if len(n.hits) > 3 else "")
            lines.append(f"- \"{n.phrase}\" ({n.how}) -> {where}")
        if not self.names:
            lines.append("- (none: the question names no value of the data)")
        lines.append("PERIODS NAMED IN THE QUESTION (rulebook dates; a comparison or a window may need more or other periods):")
        lines += [f"- {p.label}: {p.start} .. {p.end}" for p in self.periods] or ["- (no calendar phrase)"]
        if self.glossary:
            lines.append("WORDS OF THE QUESTION (shop owners' spellings): " + "; ".join(f'"{w}" = {g}' for w, g in self.glossary))
        r = self.reading
        if r is not None:
            lines.append(f"ANSWER TYPE READ FROM THE WORDING: {r.answer_type}" + (f", k = {r.k}" if r.k else "")
                         + (f" (rule: {r.rule}; reliable)" if r.sure else " (default reading, decide yourself)"))
        return "\n".join(lines)

    def as_log(self) -> dict:
        return {"names": [{"phrase": n.phrase, "how": n.how, "hits": n.hits[:3]} for n in self.names],
                "reading": self.reading.__dict__ if self.reading else None,
                "periods": [{"label": p.label, "start": str(p.start), "end": str(p.end)} for p in self.periods],
                **({"glossary": [list(x) for x in self.glossary]} if self.glossary else {}),
                **({"unknown_words": self.unknown} if self.unknown else {})}


def read_question(question: str, as_of: str, tables: dict) -> QuestionFacts:
    names = find_names(question, tables)
    return QuestionFacts(names, read_type(question, names, shop_units(tables)), mentioned_periods(question, as_of), glossary(question),
                         unknown_words(question, tables))
