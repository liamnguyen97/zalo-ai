"""Run tracking: one folder per run, named by creation time and run name, so runs can be compared later.

    result/runs/20261027-013512_public-baseline/
        submission.jsonl      the answers (the file you upload)
        logs/<question id>.json   per-question trace (plan, code, observations, answer, tokens, cost, time)
        run.json              settings, code fingerprint, counts, token/cost/time totals, score (if answers exist)
    result/runs/index.jsonl   one summary line per run (append-only)
    result/runs/latest        symlink to the most recent run
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path

from datamate.eval.scorer import score

SRC = Path(__file__).resolve().parents[1]     # the datamate package


def code_fingerprint() -> str:
    """Short hash of the source files: there is no git here, this tells two runs' code apart."""
    h = hashlib.sha1()
    for p in sorted(SRC.rglob("*.py")):
        h.update(p.relative_to(SRC).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:10]


class SubmissionWriter:
    """`submission.jsonl` kept complete up to the last finished question (README Rule 8: when a run is cut by the time limit or
    killed, only the lines already in the file are scored, the rest count as abstain).

    Every `add` rewrites the whole file to a temporary file next to it, fsyncs it and swaps it in with `os.replace`: the file on
    disk is always a whole set of valid lines in the questions' order, never a torn last line, and a question has exactly one
    line (appending would give a duplicate id when a question is redone). A question missing from the file is an abstain for the
    scorer (README: "Questions missing from the file are scored as abstain"), so unfinished ones are not written as such.
    A thousand rewrites of a thousand short lines cost about a second in all.
    """

    def __init__(self, path: str | Path, ids: list[str], keep_existing: bool = False):
        self.path, self.rank = Path(path), {qid: n for n, qid in enumerate(ids)}
        self.lines: dict[str, dict] = {}
        self._lock = threading.Lock()
        if keep_existing and self.path.exists():     # resuming: lines of an earlier attempt stay until their question is done again
            for raw in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(d, dict) and d.get("id") in self.rank:
                    self.lines[d["id"]] = d

    def add(self, line: dict) -> None:
        with self._lock:
            self.lines[line["id"]] = line
            self._write()

    def _write(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for qid in sorted(self.lines, key=self.rank.__getitem__):
                f.write(json.dumps(self.lines[qid], ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self.lines)


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "run"


def new_run_dir(base: str | Path, name: str, now: datetime | None = None) -> Path:
    now = now or datetime.now()
    d = Path(base) / f"{now:%Y%m%d-%H%M%S}_{slug(name)}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def summarize(logs_dir: Path, submission: Path | None = None, gold: Path | None = None) -> dict:
    logs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(logs_dir).glob("*.json"))]
    n = len(logs)
    types = collections.Counter(d["answer"]["type"] for d in logs)
    u = lambda k: sum(d.get("usage", {}).get(k, 0) for d in logs)  # noqa: E731
    el = sorted(d.get("elapsed_s", 0) for d in logs)
    out = {
        "n_questions": n, "answer_types": dict(types),
        "tokens": {"input": u("input_tokens"), "output": u("output_tokens"), "cached": u("cached_tokens"), "model_calls": u("calls")},
        "cost_usd": round(u("cost_usd"), 4), "cost_per_question_usd": round(u("cost_usd") / n, 5) if n else None,
        "seconds_per_question": {"mean": round(sum(el) / n, 1), "median": el[n // 2], "p90": el[int(n * 0.9)], "max": el[-1]} if n else None,
        "abstain": types.get("abstain", 0), "unanswerable": types.get("unanswerable", 0),
    }
    if gold and submission and Path(gold).exists() and Path(submission).exists():
        s = score(submission, gold, {d["id"] for d in logs})
        out["score"] = {k: s[k] for k in ("n", "correct", "wrong", "abstain", "PA", "accuracy")}
    return out


def write_run(run_dir: Path, meta: dict, base: Path | None = None) -> None:
    run_dir = Path(run_dir)
    (run_dir / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    base = Path(base) if base else run_dir.parent
    entry = {"run": run_dir.name, **{k: meta.get(k) for k in ("name", "started_at", "split", "n_questions", "code", "model", "backend", "cost_usd")},
             "PA": (meta.get("score") or {}).get("PA"), "abstain": meta.get("abstain"), "unanswerable": meta.get("unanswerable")}
    with open(base / "index.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    latest = base / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)
