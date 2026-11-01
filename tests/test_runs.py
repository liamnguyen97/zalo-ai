import json
from datetime import datetime

from datamate.eval.runs import code_fingerprint, new_run_dir, slug, summarize, write_run


def test_run_folder_name_and_summary(tmp_path):
    d = new_run_dir(tmp_path, "public baseline/v2", datetime(2026, 10, 27, 1, 35, 12))
    assert d.name == "20261027-013512_public-baseline-v2"
    logs = d / "logs"
    logs.mkdir()
    for i, (typ, route, cost) in enumerate([("number", "store_revenue", 0.0), ("abstain", None, 0.01), ("unanswerable", None, 0.02)]):
        (logs / f"q{i}.json").write_text(json.dumps({
            "id": f"q{i}", "answer": {"id": f"q{i}", "type": typ}, "route": route, "elapsed_s": 1.0 + i,
            "usage": {"input_tokens": 100, "output_tokens": 10, "cached_tokens": 20, "calls": 2, "cost_usd": cost}}), encoding="utf-8")
    s = summarize(logs)
    assert s["n_questions"] == 3 and s["abstain"] == 1 and s["unanswerable"] == 1
    assert s["tokens"]["cached"] == 60 and s["cost_usd"] == 0.03
    write_run(d, {"name": "x", "started_at": "t", **s})
    assert (tmp_path / "latest").is_symlink() and (tmp_path / "latest").resolve() == d.resolve()
    assert json.loads((tmp_path / "index.jsonl").read_text().splitlines()[0])["run"] == d.name
    assert len(code_fingerprint()) == 10 and slug("a b/c") == "a-b-c"


def test_resume_retries_only_api_failures():
    from datamate.cli import _infra_failure
    assert _infra_failure({"answer": {"type": "abstain"}, "error": "RuntimeError: AgentGenerationError: Error code: 402 - credits"})
    assert _infra_failure({"answer": {"type": "abstain"}, "coder_error": "APIStatusError: Error code: 429"})
    assert not _infra_failure({"answer": {"type": "abstain"}, "error": "RuntimeError: no final_answer within max_steps"})  # a model failure: keep
    assert not _infra_failure({"answer": {"type": "number", "answer": 1}, "error": None})
