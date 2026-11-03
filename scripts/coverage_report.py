"""Precision of the information-coverage needs (agents/coverage.py NEEDS) on a split with gold answers.

For each need: how many questions ask it, and how many ask it in a shop that does not record it (fires), split by gold type. A need
may answer UNANSWERABLE in code (`direct`) only with no fire on a gold-answerable question. Uses the gate's own functions.

    uv run python scripts/coverage_report.py train
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _shop(job):
    d, questions = job
    from datamate.agents.coverage import NEEDS, _asked, _person_prefixes, missing_information
    from datamate.context import BusinessContext
    from datamate.text import norm
    ctx = BusinessContext.build(d)
    cache: dict = {}
    names = lambda: cache.setdefault("n", _person_prefixes(ctx))   # noqa: E731
    out = []
    for q in questions:
        asked = [n.name for n in NEEDS if _asked(n, q["question"], norm(q["question"]), names)]
        fired = [n.name for n in missing_information(q["question"], ctx)]
        out.append((q, asked, fired))
    return out


def main(split: str) -> None:
    from datamate.agents.coverage import NEEDS
    root = Path(split)
    qs = [json.loads(l) for l in open(root / "questions.jsonl", encoding="utf-8")]
    gold = {}
    if (root / "answers.jsonl").exists():
        gold = {a["id"]: a for a in map(json.loads, open(root / "answers.jsonl", encoding="utf-8"))}
    by_shop = defaultdict(list)
    for q in qs:
        by_shop[q["business_id"]].append(q)
    fires, asked = defaultdict(Counter), defaultdict(Counter)
    with ProcessPoolExecutor(8) as ex:
        for rows in ex.map(_shop, [(str(root / "data" / b), v) for b, v in sorted(by_shop.items())]):
            for q, a, f in rows:
                g = gold.get(q["id"], {}).get("type", "?")
                g = "unanswerable" if g == "unanswerable" else ("answerable" if g != "?" else "?")
                for n in a:
                    asked[n][g] += 1
                for n in f:
                    fires[n][g] += 1
                    print(f"FIRE {n:9} gold={g:12} {q['business_id']} {q['question'][:110]}")
    print()
    for n in NEEDS:
        print(f"{n.name:9} asked {dict(asked[n.name])}  fires {dict(fires[n.name])}  direct={n.direct}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
