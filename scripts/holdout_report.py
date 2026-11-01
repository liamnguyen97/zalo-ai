"""Score a run on a split with gold answers and explain every wrong / abstain (spec, turns, Evaluator verdicts).

usage: python scripts/holdout_report.py <split_dir> <run_name> [baseline_run_name] [--detail]
  e.g. python scripts/holdout_report.py splits/holdout6 h6-agents-r7 h6-classic-r1 --detail
Runs are read from result/runs/*_<run_name> (the latest such folder).
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datamate.cli import _infra_failure  # noqa: E402
from datamate.eval.scorer import is_correct  # noqa: E402

detail = "--detail" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
SPLIT, RUN, BASE = args[0], args[1], (args[2] if len(args) > 2 else None)
gold = {g["id"]: g for g in map(json.loads, open(f"{SPLIT}/answers.jsonl", encoding="utf-8"))}


def load(name):
    runs = sorted(glob.glob(f"result/runs/*_{name}"))
    if not runs:
        return {}
    return {d["id"]: d for d in (json.loads(f.read_text()) for f in Path(runs[-1], "logs").glob("*.json"))}


def grade(d, i):
    a = d["answer"]
    return "A" if a.get("type") == "abstain" else ("C" if is_correct(a, gold[i]) else "W")


def row(name, L, ids):
    r = [grade(L[i], i) for i in ids]
    n = len(ids) or 1
    print(f"  {name:<12} n={len(ids):3} correct {r.count('C'):3} wrong {r.count('W'):3} abstain {r.count('A'):3} "
          f"PA {(r.count('C') - r.count('W')) / n:.3f}  infra {sum(_infra_failure(L[i]) for i in ids)}")


A, C = load(RUN), (load(BASE) if BASE else {})
print(f"done: {RUN} {len(A)}/{len(gold)}" + (f", {BASE} {len(C)}/{len(gold)}" if BASE else ""))
row("run", A, list(A))
if C:
    row("baseline", C, list(C))
    both = sorted(set(A) & set(C))
    print(f"same {len(both)} questions:")
    row("run", A, both)
    row("baseline", C, both)


def cause(d, i):
    sp = (d.get("explainer") or {}).get("spec") or {}
    g = gold[i]
    if _infra_failure(d):
        return "infra: model server"
    if d.get("error"):
        return "pipeline crash"
    if grade(d, i) == "A":
        turns = d.get("turns") or []
        if not turns:
            return "no turn ran"
        decides = [e for e in d.get("orchestrator", []) if "decide" in e]
        last = turns[-1]["verdict"]
        if decides and decides[-1]["action"] == "abstain":
            return f"orchestrator abstained after {len(turns)} turn(s) (last: {last['source']})"
        return f"rejected {len(turns)} turn(s): " + " / ".join(t["verdict"]["source"] for t in turns)
    # wrong
    if g["type"] == "unanswerable":
        return "answered an unanswerable question"
    if d["answer"].get("type") == "unanswerable":
        return "said unanswerable, gold has a value"
    if sp.get("answer_type") and sp.get("answer_type") != g["type"]:
        return f"spec type {sp.get('answer_type')} != gold {g['type']}"
    return "wrong value (same type)"


if A:
    c = Counter()
    stat = Counter(((d.get("explainer") or {}).get("spec") or {}).get("status") for d in A.values())
    for i, d in A.items():
        if grade(d, i) != "C":
            c[(grade(d, i), cause(d, i))] += 1
    print("spec status:", dict(stat))
    print("wrong / abstain causes:")
    for (g_, k), n in c.most_common():
        print(f"  {g_} {n:3}  {k}")
    if detail:
        for i, d in sorted(A.items()):
            if grade(d, i) == "C":
                continue
            sp = (d.get("explainer") or {}).get("spec") or {}
            print(f"\n[{grade(d, i)}] {i} {d['question'][:110]}  (as_of {d['as_of_date']}; classic={grade(C[i], i) if i in C else '-'})")
            print(f"   gold {gold[i]['type']}: {json.dumps(gold[i].get('answer'), ensure_ascii=False)[:90]}")
            print(f"   got  {json.dumps(d['answer'].get('answer'), ensure_ascii=False)[:90]}")
            print(f"   spec {sp.get('status')} type={sp.get('answer_type')} | {str(sp.get('rewritten_question'))[:150]}")
            print(f"   periods {[(p.get('start'), p.get('end'), p.get('role')) for p in sp.get('periods', [])]} entities {[e.get('value') for e in sp.get('entities', [])]}")
            print(f"   cause: {cause(d, i)} | turns {[(str(t.get('raw'))[:30], t['verdict']['decision']) for t in d.get('turns', [])]}")
