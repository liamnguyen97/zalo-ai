"""Text helpers shared by the data layer, the agents and the coder's sandbox.

Two comparison forms exist on purpose (they are not interchangeable):
- `fold`: no accents, lowercase, single spaces, punctuation KEPT. Item keys in `item_lines.key` are built with it and the
  coder compares names with `key == fold(name)`, so names like "Ống PVC D50 (4m)" keep their punctuation.
- `fold_text`: like `fold` but punctuation becomes a space. Used to match text a user typed against data values and
  against the question (find_values, evidence quotes), where punctuation is noise.
"""
from __future__ import annotations

import re
import unicodedata


def nodia(s: str) -> str:
    """Remove Vietnamese diacritics (đ -> d), keep case."""
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")


def fold(s) -> str:
    return re.sub(r"\s+", " ", nodia(str(s)).lower()).strip()


def fold_text(s) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s./%-]", " ", s)).strip()


# a title written before a person's name in a shop book: "A.Đức", "a Đức", "chị Lan", "c.Hoa", "e Tâm", "cô Ba", "chú Tư"
HONORIFIC = re.compile(r"^\s*(?:a|anh|c|ch|chi|e|em|co|chu|bac|ong|ba|thay|di|o|cau|mo|sep|bs|ks)\s*[.\s]\s*(?=\S)", re.I)


def given_name(name) -> str:
    """The folded name with a leading title removed ("A.Đức" -> "duc", "chị Lan" -> "lan"). A name that is only one word stays as it is
    ("Anh", "An"): the title is removed only when a name follows it."""
    f = fold(name)
    rest = HONORIFIC.sub("", f, count=1)
    return rest if rest and rest != f else f


# Telex typing left in a word ("naof", "nhaats", "khoong", "quys") is undone (from zalo-challenge `norm_q`).
FINALS = re.compile(r"(?:[aeiouy]|c|ch|m|n|ng|nh|p|t)$")
# a folded Vietnamese syllable: initial consonant, 1-3 vowels, final consonant. A trailing tone letter (s f r x j) of a longer token is
# a telex slip when what is left is a syllable and the token is not: "thangs" (44 questions), "khachs", "nhieuf", "thoaij"; while
# "thanks", "class", "plus" are left alone
SYLLABLE = re.compile(r"^(?:ngh|ng|gh|gi|kh|nh|ph|qu|th|tr|ch|[bcdghklmnprstvx])?[aeiouy]{1,3}(?:ng|nh|ch|[cmnpt])?$")


def _telex(tok: str) -> str:
    if not tok.isalpha() or len(tok) < 2:
        return tok
    t = re.sub(r"dd", "d", tok)
    t = re.sub(r"(aa|ee|oo)", lambda m: m.group(1)[0], t)
    t = re.sub(r"(a|o|u)w", r"\1", t)
    if t[-1] in "sfrxj" and FINALS.search(t[:-1]) and (t != tok or len(t) <= 5 or (SYLLABLE.match(t[:-1]) and not SYLLABLE.match(t))):
        t = t[:-1]
    return t


TONES = {"s": "́", "f": "̀", "r": "̉", "x": "̃", "j": "̣"}     # sắc huyền hỏi ngã nặng
MODIFIED = set("ăâêôơư")
VOWELS = set("aăâeêioôơuưy")


def telex_decode(tok: str) -> str:
    """Telex typing to Vietnamese letters: "cuwar" -> "cửa", "tieenf" -> "tiền", "ddos" -> "đó", "nhoms" -> "nhóm". The input method's
    rules (dd aa ee oo aw ow uw, a final tone letter s f r x j), not a word list, so it holds for any word. `_telex` decides whether a
    word holds a slip at all; this only writes it out (agents/question.written_out)."""
    t = tok.lower()
    tone = None
    if len(t) >= 3 and t[-1] in TONES and re.search(r"[aeiouy]", t[:-1]):
        tone, t = TONES[t[-1]], t[:-1]
    t = t.replace("dd", "đ")
    for a, b in (("uow", "ươ"), ("aa", "â"), ("ee", "ê"), ("oo", "ô"), ("aw", "ă"), ("ow", "ơ"), ("uw", "ư")):
        t = t.replace(a, b)
    if tone:
        start = 2 if t.startswith(("qu", "gi")) and len(t) > 2 and t[2] in VOWELS else 0
        idx = [i for i in range(start, len(t)) if t[i] in VOWELS]
        if idx:
            run = [idx[-1]]                                   # the last run of vowels carries the tone
            while len(run) < len(idx) and idx[-len(run) - 1] == run[0] - 1:
                run.insert(0, idx[-len(run) - 1])
            mod = [i for i in run if t[i] in MODIFIED]
            if mod:
                pos = mod[-1]                                 # a vowel with a hat or a horn: "tiền", "người"
            elif run[-1] < len(t) - 1 or len(run) == 1:
                pos = run[-1]                                 # a final consonant follows: the last vowel ("khách", "tháng")
            else:
                pos = run[-2] if len(run) == 2 else run[1]    # open syllable: "nào", "hai"; three vowels: the middle one
            t = t[: pos + 1] + tone + t[pos + 1:]
    out = unicodedata.normalize("NFC", t)
    return out[0].upper() + out[1:] if tok[:1].isupper() else out


def norm(text: str) -> str:
    """Folded (no accents, lower case, punctuation as spaces), telex slips undone, the shop owners' spellings in their standard form
    (datamate/lexicon.py: "đh" -> "don hang", "bnhiu" -> "bao nhieu"; accented variants are read before folding).

    The lexicon is read BEFORE the telex rule and once more after it (for a word the rule repaired): the rule takes a final s/f/r/x/j
    for a tone letter, so "order" became "orde" and "oder" "ode", and those lexicon entries never matched (dev300 q_f19c4e273491
    "cuwar hàng có bn order ddos": the type reading fell back to the default instead of "how many")."""
    from datamate import lexicon      # imported here: lexicon imports text for its glossary
    low = lexicon.apply_accented(unicodedata.normalize("NFC", (text or "").lower()))
    words = lexicon.apply_folded(fold_text(low)).split()
    return lexicon.apply_folded(" ".join(_telex(w) for w in words))
