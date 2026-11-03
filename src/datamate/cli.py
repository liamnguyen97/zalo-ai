"""datamate run   --split-dir <split> [--name public-baseline] [--backend vllm|openrouter]   -> result/runs/<datetime>_<name>/
   datamate run   --split-dir <split> --out submission.jsonl                                 -> legacy: explicit output file
   datamate score --pred submission.jsonl --gold answers.jsonl"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from datamate.config import Settings
from datamate import lexicon
from datamate.eval.runs import SubmissionWriter, code_fingerprint, new_run_dir, summarize, write_run
from datamate.eval.scorer import score
from datamate.pipeline import Pipeline, Split
from datamate.sandbox.memory import apply_limit

# smolagents wraps a failed request inside the coder as "AgentGenerationError: Error in generating model output:
# Connection error." (seen: 18 dev300 abstains after an SSH tunnel drop that resume did not redo)
INFRA_ERRORS = ("APIStatusError", "APIConnectionError", "APITimeoutError", "Error code: 402", "Error code: 429", "Error code: 5",
                "ConnectError", "ReadTimeout", "RemoteProtocolError", "Connection error", "Request timed out",
                "the run's time budget is used up")      # never started (pipeline.py): a resumed run answers it


def _infra_failure(log: dict) -> bool:
    """A question that abstained because the model API failed (credits, rate limit, timeout), not because the model failed."""
    e = (log.get("error") or "") + (log.get("coder_error") or "")
    return log.get("answer", {}).get("type") == "abstain" and any(k in e for k in INFRA_ERRORS)


def _run(args) -> None:
    if args.backend:
        os.environ["DATAMATE_BACKEND"] = args.backend
    settings = Settings()
    cap = apply_limit()
    if cap:
        print(f"memory cap {cap:.0f} GB (DATAMATE_MEM_LIMIT_GB)")
    if args.workers:
        settings.workers = args.workers
    split = Split.open(args.split_dir)
    questions = [json.loads(l) for l in open(split.root / "questions.jsonl", encoding="utf-8") if l.strip()]
    if args.business:
        questions = [q for q in questions if q["business_id"] in args.business]
    if args.limit:
        questions = questions[: args.limit]

    started = datetime.now()
    if args.out:  # legacy: explicit file, logs next to it
        run_dir, out = Path(args.out).parent, Path(args.out)
        name = args.name or out.parent.name
    elif args.resume_run:  # continue an interrupted run in its own folder (finished questions are reused)
        run_dir = Path(args.resume_run)
        out, name = run_dir / "submission.jsonl", args.name or run_dir.name.split("_", 1)[-1]
    else:
        name = args.name or f"{settings.backend}-{settings.model_id.split('/')[-1]}"
        run_dir = new_run_dir(args.runs_dir, name, started)
        out = run_dir / "submission.jsonl"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"run folder: {run_dir}")
    t_start = time.time()
    pipe = Pipeline(split, settings)
    pipe.set_run_budget(settings.run_deadline_s)      # the run's clock starts here, shop loading included

    # warm the per-business cache serially (file parsing is not worth racing)
    for b in sorted({q["business_id"] for q in questions}):
        pipe.business(b)

    def work(q):
        path = log_dir / f"{q['id']}.json"
        if path.exists() and not args.no_resume:  # resume: reuse finished questions, don't pay twice
            prev = json.loads(path.read_text(encoding="utf-8"))
            if not _infra_failure(prev):  # ...but redo the ones that only failed because the API did (402, 429, timeouts)
                return prev["answer"]
        line, log = pipe.answer(q)
        path.write_text(json.dumps(log, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return line

    # submission.jsonl is rewritten after EVERY finished question (eval/runs.py): a run that is killed or cut by the time limit
    # (README Rule 8) still has all its finished answers on disk
    writer = SubmissionWriter(out, [q["id"] for q in questions], keep_existing=bool(args.resume_run))
    with ThreadPoolExecutor(settings.workers) as ex:
        futures = {ex.submit(work, q): q for q in questions}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{settings.backend}:{settings.model_id}"):
            q = futures[fut]
            try:
                line = fut.result()
            except Exception as e:  # noqa: BLE001 - work() should not raise (Pipeline.answer turns failures into abstain); one bad question must not lose the file
                print(f"{q['id']}: {type(e).__name__}: {e}", file=sys.stderr)
                line = {"id": q["id"], "type": "abstain"}
            writer.add(line)
    print(f"wrote {len(writer)} answers -> {out}")

    gold = split.root / "answers.jsonl"
    meta = {
        "name": name, "started_at": started.isoformat(timespec="seconds"), "finished_at": datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - t_start), "split": str(split.root), "code": code_fingerprint(),
        "backend": settings.backend, "model": settings.model_id,
        "settings": {"workers": settings.workers, "temperature": settings.temperature, "thinking": settings.thinking,
                     "max_turns": settings.max_turns, "coder_steps": settings.coder_steps,
                     "explainer_reruns": settings.explainer_reruns, "check_temperature": settings.check_temperature,
                     "gate": settings.gate, "plan_thinking": settings.plan_thinking,
                     "thinking_max_tokens": settings.thinking_max_tokens,
                     "explainer_calls": settings.explainer_calls, "explainer_checks": settings.explainer_checks,
                     "providers": settings.openrouter_providers,
                     "structure_cards": os.getenv("DATAMATE_STRUCTURE_CARDS", "1"),
                     "purchase_engine": os.getenv("DATAMATE_PURCHASE_ENGINE", "1"),
                     "lexicon": lexicon.ENABLED, "glossary": lexicon.GLOSSARY,   # the values in effect
                     "p2_mode": settings.p2_mode, "plan_mode": settings.plan_mode, "spec_max_tokens": settings.spec_max_tokens,
                     "written_out": settings.written_out, "spec_conditions": settings.spec_conditions,
                     "shared_prefix": settings.shared_prefix, "guideline_rows": settings.guideline_rows,
                     "verified_facts": settings.verified_facts, "coverage_gate": settings.coverage_gate, "decide_mode": settings.decide_mode,
                     "skip_judge_on_engine": settings.skip_judge_on_engine, "trust_repeat": settings.trust_repeat,
                     "judge_unanswerable": settings.judge_unanswerable, "judge_facts": settings.judge_facts,
                     "judge_periods": settings.judge_periods, "judge_max_tokens": settings.judge_max_tokens, "judge_format": settings.judge_format, "judge_evidence": settings.judge_evidence,
                     "audit_mode": settings.audit_mode, "audit_skip": list(settings.audit_skip), "enforce_hints": list(settings.enforce_hints), "question_deadline_s": settings.question_deadline_s,
                     "run_deadline_s": settings.run_deadline_s, "run_quick_s": settings.run_quick_s,
                     "deadline_fallback": settings.deadline_fallback},
        "filters": {"limit": args.limit, "business": args.business},
        **summarize(log_dir, out, gold if gold.exists() else None),
    }
    write_run(run_dir, meta, Path(args.runs_dir) if not args.out else run_dir)
    print(json.dumps({k: meta[k] for k in ("n_questions", "answer_types", "cost_usd", "seconds_per_question", "score") if k in meta}, ensure_ascii=False, indent=2))


def _score(args) -> None:
    print(json.dumps(score(args.pred, args.gold), ensure_ascii=False, indent=2))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="datamate")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="answer a split's questions")
    r.add_argument("--split-dir", required=True)
    r.add_argument("--name", help="run name (default: backend-model); the folder is <datetime>_<name>")
    r.add_argument("--runs-dir", default="result/runs", help="where run folders are created")
    r.add_argument("--resume-run", help="existing run folder to continue (finished questions are reused)")
    r.add_argument("--out", help="legacy: write the submission to this file (logs go next to it)")
    r.add_argument("--backend", choices=["vllm", "openrouter", "ollama"])
    r.add_argument("--limit", type=int)
    r.add_argument("--business", nargs="*")
    r.add_argument("--workers", type=int)
    r.add_argument("--no-resume", action="store_true", help="re-answer questions that already have a log")
    r.set_defaults(fn=_run)
    s = sub.add_parser("score", help="compute Penalized Accuracy")
    s.add_argument("--pred", required=True)
    s.add_argument("--gold", required=True)
    s.set_defaults(fn=_score)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
