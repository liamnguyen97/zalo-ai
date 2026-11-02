"""P4 / P5 report (ledger #12): what the judge and the orchestrator's decision did on finished runs, and what the rule table, the
trusted repeat, the engine skip and the code audits do (or would have done) on the same turns. Reads the logs only: no model, no GPU.

usage: python scripts/p45_report.py <run_dir> [<run_dir> ...] [--gold answers.jsonl] [--detail]
The decision metric for keeping the judge is the first block: PA as run vs PA without the judge (the first value that passed
the checks in code submitted; exact, it stops the same trajectory earlier). Logs in the older per-turn agent format (dev300 v3.5)
go to scripts/review_counterfactual.py instead.
The gold defaults to <split>/answers.jsonl of each run's run.json. Older logs (before ledger #12) are replayed with the live rules:
the rule table on their turns; a repeat is a later turn, still passing the checks and not UNANSWERABLE, with the same submission
line as a turn the judge rejected; an engine agreement is `confirms` on the logged reference / hint (`fn(args) -> value`) and the
turn's answer. A reference whose value cannot be parsed back (cut at 400 characters, numpy reprs) counts as not confirmed, and
the number of such turns is printed. Newer logs record `engine_confirms` themselves.
"""
import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datamate.agents.orchestrator import decide_by_rule  # noqa: E402
from datamate.engines.reference import Reference, confirms  # noqa: E402
from datamate.eval.scorer import is_correct, load_jsonl  # noqa: E402

TURN = re.compile(r"turn (\d+)/")


def parse_ref(text: str) -> Reference | None:
    """A logged `Reference.text` back to a Reference; None when the value is not a Python literal (cut, numpy repr)."""
    head, sep, value = str(text).partition(") -> ")
    if not sep:
        return None
    try:
        return Reference({"function": head.split("(", 1)[0], "args": {}}, ast.literal_eval(value))
    except (ValueError, SyntaxError):
        return None


def replay_confirms(orch: list, routing: list, turn: dict) -> tuple[bool, int]:
    """(would `confirms` hold for this turn, how many of its references could not be parsed)."""
    sub = turn.get("submit") or {}
    if not sub or sub.get("type") == "unanswerable" or turn.get("issues"):
        return False, 0
    texts = [e["engine_reference"] for e in orch if "engine_reference" in e and e.get("turn") == turn["turn"]]
    texts += [e["engine_hint"] for e in orch if "engine_hint" in e and e.get("turn") == turn["turn"]]
    texts += [r["hint"] for r in routing if r.get("turn") == turn["turn"]] if not texts else []
    refs = [parse_ref(t) for t in texts]
    return any(r is not None and confirms(r, sub.get("answer")) for r in refs), sum(r is None for r in refs)


def right(sub, gold):
    return None if not sub else bool(is_correct(sub, gold))


def label(r):
    return {True: "right", False: "wrong", None: "no value"}[r]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--gold")
    ap.add_argument("--detail", action="store_true", help="list every turn behind the counts")
    a = ap.parse_args()
    verdicts, final, decide, rule_vs = Counter(), Counter(), Counter(), Counter()
    repeat, engine, seconds, calls = Counter(), Counter(), Counter(), Counter()
    audit, no_judge, quoted = Counter(), Counter(), Counter()
    changed = []
    detail, n, unparsed = [], 0, 0
    for run in map(Path, a.runs):
        gold_path = a.gold or Path(json.loads((run / "run.json").read_text()).get("split", "")) / "answers.jsonl"
        gold = load_jsonl(gold_path)
        for f in sorted((run / "logs").glob("*.json")):
            log = json.loads(f.read_text())
            g = gold.get(log["id"])
            if g is None:
                continue
            n += 1
            ans = log.get("answer") or {}
            final["abstain" if ans.get("type") == "abstain" else label(right(ans, g))] += 1
            for c in log.get("llm_calls") or []:
                if c.get("stage") in ("judge", "orchestrator"):
                    seconds[c["stage"]] += c.get("s", 0)
                    calls[c["stage"]] += 1
            turns, orch = log.get("turns") or [], log.get("orchestrator") or []
            new_log = any("engine_confirms" in e or "rule_would" in e or e.get("by") == "rule" and "after_evaluator" in str(e.get("decide"))
                          for e in orch)
            # without the judge: the first value that passed the checks in code is submitted (exact: same trajectory, stops earlier)
            fin = "abstain" if ans.get("type") == "abstain" else label(right(ans, g))
            first = next((t for t in turns if t["verdict"]["source"] not in ("checks", "no_answer") and t.get("submit")), None)
            nj = label(right(first["submit"], g)) if first else fin
            no_judge[nj] += 1
            if nj != fin:
                changed.append(f"{log['id']} {fin} -> {nj} without the judge")
            for t in turns:
                v, r = t["verdict"], right(t.get("submit"), g)
                verdicts[(v["source"], v["decision"], label(r))] += 1
                if v["source"] == "judge" and v["decision"] == "revise":
                    quoted[(v.get("found_in") or "? (older log)", label(r))] += 1
                for check in t.get("audit") or {}:          # answer/audit.py, logged in shadow mode too
                    audit[(check, label(r))] += 1
                    if a.detail:
                        detail.append(f"audit   {log['id']} t{t['turn']} {check} value {label(r)}: {str(t['audit'][check])[:120]}")
                # engine agreement: recorded (new logs) or approximated (old logs)
                if new_log:
                    agree = any(e.get("engine_confirms") and e.get("turn") == t["turn"] for e in orch)
                else:
                    agree, bad = replay_confirms(orch, log.get("routing") or [], t)
                    unparsed += bad
                if agree and v["source"] in ("judge", "judge_ignored", "engine_agrees"):
                    engine[(v["source"], v["decision"], label(r))] += 1
                    if a.detail:
                        detail.append(f"engine  {log['id']} t{t['turn']} {v['source']}/{v['decision']} value {label(r)}")
            # a value the judge rejected, returned unchanged later
            for i, t in enumerate(turns):
                sub = t.get("submit") or {}
                if t["verdict"]["source"] == "repeat_after_veto" or (
                        sub and sub.get("type") != "unanswerable" and not t.get("issues")
                        and any(p["verdict"]["source"] == "judge" and p["verdict"]["decision"] == "revise"
                                and p.get("submit") == sub for p in turns[:i])):
                    repeat[(t["verdict"]["source"], label(right(t.get("submit"), g)))] += 1
                    if a.detail:
                        detail.append(f"repeat  {log['id']} t{t['turn']} {t['verdict']['source']} value {label(right(t.get('submit'), g))}")
            # decisions after a rejected turn
            for e in orch:
                if not str(e.get("decide", "")).startswith("after_evaluator"):
                    continue
                m = TURN.search(e["decide"])
                k = int(m.group(1)) if m else len(turns)
                rule = e.get("rule_would") or (e["action"] if e.get("by") == "rule" else decide_by_rule(turns[:k], e["allowed"]).action)
                decide[(e["by"], e["action"])] += 1
                rule_vs[("same" if rule == e["action"] else f"{e['action']} -> rule {rule}")] += 1
                if a.detail and rule != e["action"]:
                    detail.append(f"decide  {log['id']} after t{k}: {e['by']} {e['action']}, rule {rule}; final "
                                  f"{'abstain' if ans.get('type') == 'abstain' else label(right(ans, g))}; {str(e.get('reason'))[:100]}")

    pa = lambda c: (c["right"] - c["wrong"]) / n if n else 0.0
    print(f"questions {n}")
    print(f"  actual          {dict(final)}  PA {pa(final):.3f}")
    print(f"  without judge   {dict(no_judge)}  PA {pa(no_judge):.3f}   (first value that passed the checks in code, submitted)")
    for c in changed:
        print("    ", c)
    print("\nverdicts by turn (source, decision, was the turn's value right?):")
    for k, v in sorted(verdicts.items()):
        print(f"  {v:4d}  {k}")
    print("\njudge revise verdicts by where the quoted evidence was found (where, was the turn's value right?):")
    for k, v in sorted(quoted.items()):
        print(f"  {v:4d}  {k}")
    print("\njudge / engine on turns whose value equals the engine's (skipped by DATAMATE_SKIP_JUDGE_ON_ENGINE):")
    for k, v in sorted(engine.items()):
        print(f"  {v:4d}  {k}")
    if unparsed:
        print(f"  ({unparsed} logged engine values could not be parsed back: counted as not confirmed)")
    print("\nvalue the judge rejected, returned unchanged (accepted by DATAMATE_TRUST_REPEAT):")
    for k, v in sorted(repeat.items()):
        print(f"  {v:4d}  {k}")
    print("\ncode audits that fired (check, was the turn's value right?) - with DATAMATE_AUDIT=shadow nothing was sent back:")
    for k, v in sorted(audit.items()):
        print(f"  {v:4d}  {k}")
    print("\ndecisions after a rejected turn (by, action):", dict(decide))
    print("model vs rule table:", dict(rule_vs))
    if calls:
        print("\nmodel time in P4/P5:", {s: f"{calls[s]} calls, {seconds[s]:.0f} s" for s in calls})
    if detail:
        print("\n" + "\n".join(detail))


if __name__ == "__main__":
    main()
