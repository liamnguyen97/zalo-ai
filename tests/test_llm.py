"""Per-model thinking, the chat helper, and SystemExit from generated code (no model calls)."""
from types import SimpleNamespace

import pytest

import datamate.pipeline as pl
from datamate.config import Settings
from datamate.llm import build_model, chat


@pytest.mark.parametrize("backend", ["vllm", "openrouter"])
def test_thinking_is_per_model(backend, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    s = Settings(backend=backend, thinking=False, max_tokens=4096, thinking_max_tokens=12288)
    plain, think = build_model(s), build_model(s, thinking=True, max_tokens=s.thinking_max_tokens)

    def on(m):
        eb = m.kwargs.get("extra_body") or {}
        return eb.get("chat_template_kwargs", {}).get("enable_thinking", eb.get("reasoning", {}).get("enabled"))

    assert on(plain) is False and on(think) is True
    assert think.kwargs.get("max_tokens") == 12288 and plain.kwargs.get("max_tokens") == 4096


def test_ollama_thinking_goes_through_reasoning_effort():
    # Ollama's OpenAI endpoint ignores chat_template_kwargs and `think`; only reasoning_effort switches thinking
    s = Settings(backend="ollama", thinking=False, ollama_model="m", ollama_base_url="http://localhost:11434/v1")
    plain, think = build_model(s), build_model(s, thinking=True, max_tokens=12288)
    assert plain.kwargs["extra_body"] == {"reasoning_effort": "none"}
    assert think.kwargs["extra_body"] == {"reasoning_effort": "high"} and think.kwargs["max_tokens"] == 12288
    assert s.model_id == "m"


def test_pipeline_gives_thinking_only_to_the_plan():
    s = Settings(backend="vllm", plan_thinking=True)
    split = SimpleNamespace(rulebook="r", calendar=None)
    p = pl.Pipeline(split, s)
    think = lambda m: m.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"]  # noqa: E731
    o = p.orchestrator
    assert think(o.explainer.plan_model) is True                              # the plan thinks
    assert think(o.explainer.model) is False and think(o.explainer.check_model) is False
    assert not think(o.coder.model) and not think(o.evaluator.model) and not think(o.model)   # coder, judge, orchestrator never


def test_chat_drops_inline_reasoning():
    m = SimpleNamespace(generate=lambda msgs: SimpleNamespace(content="<think>long trace</think>\n{\"a\": 1}"))
    assert chat(m, "s", "u") == '{"a": 1}'


def test_missing_content_is_empty_and_the_plan_is_left_out():
    # holdout6 r7: 88 of 150 plans reached the coder as the text "None" (thinking spent the whole budget)
    from datamate.agents.coder import build_task
    from datamate.agents.explainer import Spec
    m = SimpleNamespace(generate=lambda msgs: SimpleNamespace(content=None))
    assert chat(m, "s", "u") == ""
    st = SimpleNamespace(ctx=SimpleNamespace(readme=""), schema="S", as_of="2026-05-31", question="q", brief=Spec(), plan="")
    assert "PLAN" not in build_task(st, "R") and "# PLAN (suggested steps)\n1." in build_task(SimpleNamespace(**{**vars(st), "plan": "1."}), "R")


def test_system_exit_in_generated_code_does_not_end_the_run():
    from smolagents.local_python_executor import LocalPythonExecutor

    from datamate.sandbox.guard import GuardedExecutor
    inner = LocalPythonExecutor(additional_authorized_imports=[])
    inner.send_tools({})
    ex = GuardedExecutor(inner)
    with pytest.raises(ValueError, match="answer = "):
        ex("raise SystemExit('unanswerable')")


def test_answer_survives_system_exit(monkeypatch):
    p = object.__new__(pl.Pipeline)
    p.split = SimpleNamespace(root=SimpleNamespace(name="x"))
    p.business = lambda b: (_ for _ in ()).throw(SystemExit(0))
    line, log = p.answer({"id": "q1", "business_id": "b", "question": "q", "as_of_date": "2026-01-01"})
    assert line == {"id": "q1", "type": "abstain"} and "SystemExit" in log["error"]
