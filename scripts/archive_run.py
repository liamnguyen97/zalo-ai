"""Copy an existing result folder (logs/ + submission.jsonl) into result/runs/ under the new naming scheme.

usage: python scripts/archive_run.py <old_result_dir> <name> [--gold answers.jsonl] [--split DIR]
The run's datetime is taken from its first log file (when it was created); code fingerprint is unknown for old runs.
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datamate.eval.runs import new_run_dir, summarize, write_run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("old")
ap.add_argument("name")
ap.add_argument("--gold")
ap.add_argument("--split", default="")
ap.add_argument("--model", default="qwen/qwen3.5-9b")
ap.add_argument("--runs-dir", default="result/runs")
ap.add_argument("--note", default="")
a = ap.parse_args()
old = Path(a.old)
logs = sorted((old / "logs").glob("*.json"))
started = datetime.fromtimestamp(min(p.stat().st_mtime for p in logs))
run = new_run_dir(a.runs_dir, a.name, started)
shutil.copytree(old / "logs", run / "logs")
if (old / "submission.jsonl").exists():
    shutil.copy2(old / "submission.jsonl", run / "submission.jsonl")
meta = {"name": a.name, "started_at": started.isoformat(timespec="seconds"), "split": a.split, "code": "n/a (archived)",
        "backend": "openrouter", "model": a.model, "archived_from": str(old), "note": a.note,
        **summarize(run / "logs", run / "submission.jsonl", Path(a.gold) if a.gold else None)}
write_run(run, meta, Path(a.runs_dir))
print(run, "| PA:", (meta.get("score") or {}).get("PA"), "| n:", meta["n_questions"], "| cost $", meta["cost_usd"])
