"""Build evaluation splits (questions + answers + symlinked data) without touching the source split.

usage:
  python scripts/subsplit.py holdout <src_split> <out> <n_per_biz> <seed> <biz,...> --exclude <questions.jsonl>...
      questions of the given businesses never used in the excluded splits (held-out evaluation)
  python scripts/subsplit.py engine-routed <src_split> <run_dir> <out>
  python scripts/subsplit.py ids <src_split> <out> <ids.txt>
      a fixed list of question ids, e.g. docs/holdout6_ids.txt (150 questions, 6 businesses)
      the questions a finished run answered with an engine (to re-run them coder-only)
"""
import json
import random
import sys
from pathlib import Path


def _write(src: Path, out: Path, keep: set[str]) -> int:
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in ("questions.jsonl", "answers.jsonl"):
        if not (src / name).exists():
            continue
        rows = [l for l in open(src / name, encoding="utf-8") if json.loads(l)["id"] in keep]
        (out / name).write_text("".join(rows), encoding="utf-8")
        n = max(n, len(rows))
    for name in ("calendar.csv", "data", "rulebook.md"):
        t = out / name
        if not t.exists() and (src / name).exists():
            t.symlink_to((src / name).resolve())
    return n


def main() -> None:
    mode, src = sys.argv[1], Path(sys.argv[2])
    if mode == "holdout":
        out, per, seed, bizs = Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), set(sys.argv[6].split(","))
        excl = set()
        for p in sys.argv[sys.argv.index("--exclude") + 1:]:
            excl |= {json.loads(l)["id"] for l in open(p, encoding="utf-8")}
        qs = [json.loads(l) for l in open(src / "questions.jsonl", encoding="utf-8")]
        rng, keep = random.Random(seed), set()
        for b in sorted(bizs):
            pool = sorted(q["id"] for q in qs if q["business_id"] == b and q["id"] not in excl)
            rng.shuffle(pool)
            keep |= set(pool[:per])
    elif mode == "ids":
        out = Path(sys.argv[3])
        keep = {l.strip() for l in open(sys.argv[4], encoding="utf-8") if l.strip()}
    else:
        run, out = Path(sys.argv[3]), Path(sys.argv[4])
        keep = {json.loads(f.read_text(encoding="utf-8"))["id"] for f in (run / "logs").glob("*.json")
                if json.loads(f.read_text(encoding="utf-8")).get("route")}
    print(f"{_write(src, out, keep)} questions -> {out}")


if __name__ == "__main__":
    main()
