"""datamate run   --split-dir <split> [--name public-baseline] [--backend vllm|openrouter]   -> result/runs/<datetime>_<name>/
   datamate run   --split-dir <split> --out submission.jsonl                                 -> legacy: explicit output file
   datamate score --pred submission.jsonl --gold answers.jsonl"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from datamate.config import Settings
from datamate.eval.runs import code_fingerprint, new_run_dir, summarize, write_run
from datamate.eval.scorer import score
from datamate.pipeline import Pipeline, Split
from datamate.sandbox.memory import apply_limit

# smolagents wraps a failed request inside the coder as "AgentGenerationError: Error in generating model output:
# Connection error." (seen: 18 dev300 abstains after an SSH tunnel drop that resume did not redo)
INFRA_ERRORS = ("APIStatusError", "APIConnectionError", "APITimeoutError", "Error code: 402", "Error code: 429", "Error code: 5",
                "ConnectError", "ReadTimeout", "RemoteProtocolError", "Connection error", "Request timed out")


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

    with ThreadPoolExecutor(settings.workers) as ex:
        lines = list(tqdm(ex.map(work, questions), total=len(questions), desc=f"{settings.backend}:{settings.model_id}"))

    with open(out, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"wrote {len(lines)} answers -> {out}")

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
                     "structure_cards": os.getenv("DATAMATE_STRUCTURE_CARDS", "1")},
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
