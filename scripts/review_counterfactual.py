"""What the LLM review / judge is worth, measured on a finished run with exact counterfactuals (ledger #12). No model, no GPU.

Reads run logs in the per-turn agent format (`turns[].agents[]` with explainer / coder / interpreter / formatter / checks / review,
the dev300 v3.5 logs). Every variant walks the SAME trajectory as the real run and only stops earlier, submitting a value the run
had already computed, so its outcome is known exactly:

  actual         the run as it was
  no_review      submit the first value that passes the checks in code (no judge at all)
  unans_direct   a value UNANSWERABLE that passed the checks is submitted without review
  trust_repeat   a value the review rejected and the Coder returned unchanged is submitted (A2)
  engine_skip    a value `confirms` with the engine reference is submitted without review (A3)
  all            the last three together
Also: review calls on engine-confirmed turns, and the entity check (B1: a value the question names absent from the string
literals of the turn's code, f-strings and print() excluded) on the first value that passed the checks.

usage: python scripts/review_counterfactual.py <logs_dir> [--gold train/answers.jsonl]
"""
import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datamate.engines.reference import Reference, confirms  # noqa: E402
from datamate.eval.scorer import is_correct, load_jsonl  # noqa: E402
from datamate.text import fold_text  # noqa: E402


def outcome(line, g):
    if not line or line.get("type") == "abstain":
        return "abstain"
    return "right" if is_correct(line, g) else "wrong"


def parse_ref(text):
    head, sep, value = str(text or "").partition(") -> ")
    if not sep:
        return None
    try:
        return Reference({"function": head.split("(", 1)[0], "args": {}}, ast.literal_eval(value))
    except (ValueError, SyntaxError):
        return None


def literals(code: str) -> str:
    """The code's string constants, folded, without f-strings and print(...) arguments (they echo names, they do not filter)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    skip = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.JoinedStr) or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"):
            skip |= {id(x) for x in ast.walk(n)}
    return fold_text(" | ".join(n.value for n in ast.walk(tree)
                                if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip))


def turns_of(log):
    brief, out = None, []
    for t in log["turns"]:
        rec = {"turn": t["turn"], "code": [], "line": None, "passed": False, "review": None, "ref": None}
        for a in t["agents"]:
            o = a.get("output") if isinstance(a.get("output"), dict) else {}
            if a["agent"] == "explainer" and o.get("brief"):
                brief = o["brief"]
            elif a["agent"] == "coder":
                rec["code"].append(o.get("code") or "")
            elif a["agent"] == "interpreter":
                rec["ref"] = (a.get("input") or {}).get("engine_reference")
            elif a["agent"] == "formatter":
                rec["line"] = o
            elif a["agent"] == "checks":
                rec["passed"] = o.get("decision") == "pass"
            elif a["agent"] == "review":
                rec["review"] = o
        rec["brief"] = brief
        if t["turn"] >= 1:
            out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("--gold", default="train/answers.jsonl")
    a = ap.parse_args()
    gold = load_jsonl(a.gold)
    names = ("actual", "no_review", "unans_direct", "trust_repeat", "engine_skip", "all")
    res, changed = {v: Counter() for v in names}, {v: [] for v in names}
    a3, b1, b1_hits, n = Counter(), Counter(), [], 0
    for f in sorted(Path(a.logs).glob("*.json")):
        log = json.loads(f.read_text())
        g = gold.get(log["id"])
        if g is None:
            continue
        n += 1
        fin, turns = outcome(log.get("answer"), g), turns_of(log)

        def unans(r):
            return r["line"].get("type") == "unanswerable"

        def engine(r):
            ref = parse_ref(r["ref"])
            return ref is not None and not unans(r) and confirms(ref, r["line"].get("answer"))

        def walk(stop):
            rejected = []
            for r in turns:
                if not r["passed"] or not r["line"]:
                    continue
                if stop(r, rejected):
                    return outcome(r["line"], g)
                rv = r["review"] or {}
                if rv.get("action") in ("code", "explain") and not rv.get("ignored"):
                    rejected.append(r["line"])
            return fin

        def repeat(r, rej):
            return not unans(r) and r["line"] in rej

        out = {"actual": fin, "no_review": walk(lambda r, rej: True), "unans_direct": walk(lambda r, rej: unans(r)),
               "trust_repeat": walk(repeat), "engine_skip": walk(lambda r, rej: engine(r)),
               "all": walk(lambda r, rej: unans(r) or repeat(r, rej) or engine(r))}
        for v, o in out.items():
            res[v][o] += 1
            if o != fin:
                changed[v].append(f"{log['id']} {fin}->{o}")
        for r in turns:
            if r["review"] is not None and r["line"] and engine(r):
                a3[(r["review"].get("action"), outcome(r["line"], g))] += 1
        first = next((r for r in turns if r["passed"] and r["line"] and not unans(r)), None)
        if first and first["brief"]:
            lits = literals("\n".join(first["code"]))
            miss = [e["value"] for e in first["brief"].get("entities") or []
                    if isinstance(e, dict) and str(e.get("value") or "").strip() and fold_text(str(e["value"])) not in lits]
            b1[(bool(miss), outcome(first["line"], g))] += 1
            if miss:
                b1_hits.append(f"{log['id']} {outcome(first['line'], g)} {miss}")
    for v in names:
        c = res[v]
        print(f"{v:13s} right {c['right']:3d} wrong {c['wrong']:3d} abstain {c['abstain']:2d}  PA {(c['right'] - c['wrong']) / n:.3f}"
              + (f"  changed: {', '.join(changed[v])}" if changed[v] else ""))
    print(f"\nreview calls on engine-confirmed turns (action, value): {dict(a3)}")
    print(f"entity check B1 on the first value passing the checks (fired, value): {dict(b1)}")
    for h in b1_hits:
        print("  ", h)


if __name__ == "__main__":
    main()
