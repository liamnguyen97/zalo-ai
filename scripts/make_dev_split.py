"""Build a stratified dev split (questions + symlinks) from the unzipped training split.

usage: python scripts/make_dev_split.py <train_dir> <out_dir> [n=300] [seed=42]
Stratified by answer type (rare types get a floor) and spread round-robin over businesses.
Writes <out_dir>/{questions.jsonl,answers.jsonl,data,rulebook.md,calendar.csv} and ids to docs/dev300_ids.txt.
"""
import collections
import json
import os
import random
import sys

train, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
rnd = random.Random(seed)

qs = [json.loads(l) for l in open(f"{train}/questions.jsonl", encoding="utf-8") if l.strip()]
gold = {d["id"]: d for d in map(json.loads, open(f"{train}/answers.jsonl", encoding="utf-8"))}
by_type = collections.defaultdict(list)
for q in qs:
    by_type[gold[q["id"]]["type"]].append(q)

FLOOR = {"unanswerable": 12, "string": 12, "list": 12, "date": 12, "boolean": 6, "table": 6}
quota = {t: max(round(n * len(v) / len(qs)), FLOOR.get(t, 0)) for t, v in by_type.items()}
excess = sum(quota.values()) - n
quota["number"] -= max(excess, 0)  # keep the total at n (number is the biggest bucket)

picked = []
for t, k in quota.items():
    pools = collections.defaultdict(list)
    for q in by_type[t]:
        pools[q["business_id"]].append(q)
    for v in pools.values():
        rnd.shuffle(v)
    order = sorted(pools)
    rnd.shuffle(order)
    got = []
    while len(got) < min(k, len(by_type[t])):  # round-robin over businesses
        for b in order:
            if pools[b] and len(got) < k:
                got.append(pools[b].pop())
    picked += got

os.makedirs(out, exist_ok=True)
for name in ("data", "rulebook.md", "calendar.csv"):
    dst = f"{out}/{name}"
    if not os.path.lexists(dst):
        os.symlink(os.path.abspath(f"{train}/{name}"), dst)
with open(f"{out}/questions.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in picked)
with open(f"{out}/answers.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps(gold[q["id"]], ensure_ascii=False) + "\n" for q in picked)
os.makedirs("docs", exist_ok=True)
open("docs/dev300_ids.txt", "w").write("\n".join(q["id"] for q in picked) + "\n")
print(len(picked), "questions;", dict(collections.Counter(gold[q["id"]]["type"] for q in picked)),
      "|", len({q["business_id"] for q in picked}), "businesses")
