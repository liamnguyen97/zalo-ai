"""Words of the questions that no reader knows yet, ranked by how many questions use them (ticket T3.1).

usage: python scripts/oov_discovery.py <split_dir> [<split_dir> ...] [--min 3] [--top 80] [--no-data]

A word is known when it is in the rulebook, a README, a column name, a value of the shop's data (skip with --no-data: faster, but
product names then show up), or a standard form of the lexicon (src/datamate/lexicon.py). What is left is mostly shop owners'
spellings ("j", "wan", "kon", "đh"), typos and product names missing from the data. Read the list, add the real variants to the
lexicon as one line each, with a test line in tests/test_question_reading.py, and check the train reading stays where it is.
Rule 6: use the public split only to count, never to tune a rule on its answers.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datamate.agents.question import LEXICON_WORDS, name_index  # noqa: E402
from datamate.data.loader import load_business  # noqa: E402
from datamate.text import SYLLABLE, norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("splits", nargs="+")
ap.add_argument("--min", type=int, default=3, help="questions a word must appear in")
ap.add_argument("--top", type=int, default=80)
ap.add_argument("--no-data", action="store_true", help="do not load the shops' data (faster; product names are then unknown)")
a = ap.parse_args()

known = set(LEXICON_WORDS)
uses: dict[str, list[str]] = collections.defaultdict(list)
for split in map(Path, a.splits):
    if (split / "rulebook.md").exists():
        known |= set(norm((split / "rulebook.md").read_text(encoding="utf-8")).split())
    by_shop = collections.defaultdict(list)
    for line in open(split / "questions.jsonl", encoding="utf-8"):
        if line.strip():
            q = json.loads(line)
            by_shop[q["business_id"]].append(q["question"])
    for shop, questions in sorted(by_shop.items()):
        d = split / "data" / shop
        if (d / "README.md").exists():
            known |= set(norm((d / "README.md").read_text(encoding="utf-8")).split())
        shop_words = set()
        if not a.no_data and d.exists():
            tables = load_business(d).tables
            shop_words = {w for f in name_index(tables) for w in f.split()} | {w for df in tables.values() for c in df.columns for w in norm(str(c)).split()}
        for q in questions:
            for w in set(norm(q).split()):
                if w.isalpha() and w not in shop_words:
                    uses[w].append(q)
rows = [(w, qs) for w, qs in uses.items() if w not in known and len(qs) >= a.min]
rows.sort(key=lambda x: -len(x[1]))
print(f"{len(rows)} words in >= {a.min} questions that no reader knows (syllable = a well-formed Vietnamese syllable)\n")
for w, qs in rows[: a.top]:
    print(f"{len(qs):>5}  {w:<16} {'syllable' if SYLLABLE.match(w) else '        '}  e.g. {qs[0][:90]}")
