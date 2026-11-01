"""Onboarding report for every shop under a data directory: detected engines/columns and invariants, no gold needed.

usage: .venv/bin/python scripts/onboard_report.py data/public_test/data data/train/data [--json out.jsonl]
Prints one line per shop (trusted / engines / failed checks), the provenance of the sales engine and the table structure the
schema cards will state (data/structure.py: parent/child tables, document totals, flat tables). Run it on any new split
before a run: a shop where `trusted=False` or an engine is missing is where the engines cannot be relied on, and a total or
flat-table column that looks wrong is a card that would mislead the model.
"""
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def one(path: str) -> dict:
    from datamate.context import BusinessContext
    from datamate.data.onboarding import summary
    try:
        ctx = BusinessContext.build(path)
        rep = ctx.onboarding
        return {"biz": Path(path).name, "summary": summary(rep), "report": rep}
    except Exception as e:  # noqa: BLE001
        return {"biz": Path(path).name, "summary": f"LOAD ERROR {type(e).__name__}: {str(e)[:100]}", "report": None}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if out:
        args = [a for a in args if a != out]
    dirs = [d for a in args for d in sorted(Path(a).iterdir()) if d.is_dir()]
    with ProcessPoolExecutor(3) as ex:
        rows = list(ex.map(one, map(str, dirs)))
    for r in rows:
        s = (r["report"] or {}).get("sales", {})
        print(f"{r['biz']:8} {r['summary'][:230]}" + (f"\n         sales: {s.get('table')} order={s.get('order_col')} date={s.get('date_col')} amount={s.get('amount_col')} via {s.get('source')}" if s else ""))
        st = (r["report"] or {}).get("structure")
        if st:   # data/structure.py: what the schema cards will claim about this shop; check it on shops never seen before
            print(f"         structure: {len(st['links'])} parent/child, {len(st['totals'])} document totals, "
                  f"{len(st['doc_level'])} flat tables, {st['notes']} README notes, {len(st['aliases'])} aliases")
            for x in st["totals"]:
                print(f"           total  {x}")
            for t, x in st["doc_level"].items():
                print(f"           flat   {t}: {x[:150]}")
    print(f"\n{sum('trusted=True' in r['summary'] for r in rows)}/{len(rows)} shops trusted")
    if out:
        Path(out).write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows), encoding="utf-8")
