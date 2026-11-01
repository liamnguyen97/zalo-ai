"""Render per-question logs (result/logs/*.json) into readable Markdown traces.

usage: python scripts/trace_report.py <logs_dir> <answers.jsonl|-> <out.md>      # one file
       python scripts/trace_report.py <logs_dir> <answers.jsonl|-> <out_dir>/    # one file per business + INDEX.md
"""
import collections
import glob
import json
import os
import sys

logs_dir, gold_path, out = sys.argv[1:4]
gold = {}
if gold_path != "-":
    gold = {d["id"]: d for d in map(json.loads, open(gold_path, encoding="utf-8"))}


def fence(text, lang=""):
    return f"```{lang}\n{str(text).rstrip()}\n```"


def _spec_lines(sp: dict) -> list[str]:
    keep = ("rewritten_question", "answer_type", "measure", "k", "periods", "entities", "answerable", "assumptions", "status")
    return [f"- **{k}**: `{json.dumps(sp.get(k), ensure_ascii=False)[:300]}`" for k in keep if sp.get(k) not in (None, "", [], {})]


def _turns(turns: list) -> list[str]:
    lines = []
    for t in turns:
        v = t.get("verdict") or {}
        lines.append(f"\n### 3. Turn {t['turn']}: Coder ⇄ Runner, {len(t.get('steps', []))} block(s)")
        for i, s in enumerate(t.get("steps") or [], 1):
            if s.get("code"):
                lines += [f"Block {i} — code", fence(s["code"], "python")]
            lines += [f"Block {i} — runner", fence((s.get("error") or s.get("output") or "(no output)")[:800])]
        lines += [f"**Interpreter:** type `{t.get('type')}`, issues `{t.get('issues') or []}` · **Formatter:** "
                  f"`{json.dumps(t.get('submit'), ensure_ascii=False)[:200]}`",
                  f"**Evaluator:** {v.get('decision')} ({v.get('source')}, fix in {v.get('target')})"]
        if v.get("decision") == "revise":
            lines.append(fence(str(v.get("feedback", ""))[:1200]))
    return lines


def render(d):
    g = gold.get(d["id"])
    u = d.get("usage", {})
    ref = f" — gold: `{json.dumps(g.get('answer'), ensure_ascii=False)[:120]}` ({g['type']})" if g else ""
    sel = (d.get("column_selector") or {}).get("selection") or {}
    ex = d.get("explainer") or {}
    lines = [f"\n---\n## {d['id']}: {d['question']}",
             f"*{d.get('elapsed_s')}s · {u.get('calls', '?')} model calls · {u.get('input_tokens', 0)} in / "
             f"{u.get('output_tokens', 0)} out tokens*{ref}\n",
             "### 1. Column selector — tables/columns picked",
             "\n".join(f"- `{t}`: {', '.join(c)}" for t, c in sel.items()) or "- (none)",
             f"\n### 2. Explainer — spec after {len(ex.get('turns', []))} turn(s)", *_spec_lines(ex.get("spec") or {}),
             f"- independent reading: `{json.dumps(d.get('explainer_blind_type'), ensure_ascii=False)}`",
             "\n**Plan given to the coder**", fence(d.get("plan", ""))]
    decides = [e for e in d.get("orchestrator", []) if "decide" in e]
    if decides:
        lines += ["\n**Orchestrator decisions:**"] + [f"- {e['decide']}: **{e['action']}** ({e['by']}) {e.get('message') or ''}"
                                                       for e in decides]
    lines += _turns(d.get("turns") or [])
    lines += ["\n### 4. Submission line", fence(json.dumps(d["answer"], ensure_ascii=False), "json")]
    if d.get("error"):
        lines.append(f"**Pipeline error (→ abstain):** `{d['error']}`")
    return lines


HEAD = ["Orchestrator loop: **Column selector → Explainer (spec + plan) → [decide] → turns of Coder⇄Runner → Interpreter → "
        "Formatter → Evaluator (accept / revise → [decide]) → submit or abstain**\n"]
logs = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(f"{logs_dir}/*.json"))]

if out.endswith("/"):
    os.makedirs(out, exist_ok=True)
    by_biz = collections.defaultdict(list)
    for d in logs:
        by_biz[d.get("business_id", "unknown")].append(d)
    index = ["# Trace index\n", "| business | questions | answered | unanswerable | abstain | avg s | file |", "|---|---|---|---|---|---|---|"]
    for biz, ds in sorted(by_biz.items()):
        lines = [f"# Trace: {biz} ({len(ds)} questions)\n"] + HEAD
        for d in ds:
            lines += render(d)
        open(os.path.join(out, f"{biz}.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        t = collections.Counter(d["answer"]["type"] for d in ds)
        ans = len(ds) - t["abstain"] - t["unanswerable"]
        index.append(f"| {biz} | {len(ds)} | {ans} | {t['unanswerable']} | {t['abstain']} | "
                     f"{sum(d.get('elapsed_s') or 0 for d in ds) / len(ds):.0f} | [{biz}.md]({biz}.md) |")
    open(os.path.join(out, "INDEX.md"), "w", encoding="utf-8").write("\n".join(index) + "\n")
    print("wrote", len(by_biz), "business traces ->", out)
else:
    lines = ["# Trace: how each question flows through the pipeline\n"] + HEAD
    for d in logs:
        lines += render(d)
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote", out)
