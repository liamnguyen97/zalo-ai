"""Model factory: one smolagents model, switchable between self-hosted vLLM and OpenRouter.

Prompt caching: on OpenRouter, qwen3.5-9b is cached automatically (provider-side prefix cache);
`cache_control` markers have no effect for this model. What matters is (1) routing to a provider
that bills cache reads (see Settings.openrouter_providers) and (2) putting stable text first in every
prompt (system prompt, rulebook, README, then per-question content). Every call's cached tokens and
real cost are recorded so the benefit is measured, not guessed.
"""
from __future__ import annotations

import re
import threading
import time

from smolagents import OpenAIModel

from datamate.config import Settings

_usage = threading.local()  # each question runs in one worker thread

MIN_CALL_S = 5.0            # a model call is never given less than this once the budget still has time left


class DeadlineExceeded(Exception):
    """The question's time budget ran out (DATAMATE_QUESTION_DEADLINE_S, or the run's: DATAMATE_RUN_DEADLINE_S). Raised before a
    model call is made, never in the middle of one: a thread cannot be interrupted, so every call is also given a request
    timeout of the time that is left (UsageModel.generate)."""


def set_deadline(at: float | None) -> None:
    """The time (time.monotonic()) this thread's question must be answered by; None = no limit."""
    _usage.deadline = at


def time_left() -> float | None:
    at = getattr(_usage, "deadline", None)
    return None if at is None else at - time.monotonic()


def check_deadline(what: str = "") -> None:
    left = time_left()
    if left is not None and left <= 0:
        raise DeadlineExceeded(f"the time budget of the question is used up{f' (before {what})' if what else ''}")


def set_session(session_id: str | None) -> None:
    """Sticky-routing key for this thread's requests (OpenRouter `x-session-id`): follow-up requests of the
    same business go to the same provider, which keeps that provider's prefix cache warm."""
    _usage.session = session_id


def reset_usage() -> None:
    _usage.calls = _usage.in_tok = _usage.out_tok = _usage.cached = 0
    _usage.cost = 0.0
    _usage.call_log = []


def get_call_log() -> list[dict]:
    """One record per model call of this question: stage, seconds, tokens in / cached / out, whether the output cap was hit.
    The first local runs had 5 of 201 calls write ~4,000 tokens (the cap) and nothing said which stage they came from."""
    return list(getattr(_usage, "call_log", []))


def get_usage() -> dict:
    return {
        "calls": getattr(_usage, "calls", 0),
        "input_tokens": getattr(_usage, "in_tok", 0),
        "output_tokens": getattr(_usage, "out_tok", 0),
        "cached_tokens": getattr(_usage, "cached", 0),
        "cost_usd": round(getattr(_usage, "cost", 0.0), 6),
    }


def _record(raw) -> None:
    u = getattr(raw, "usage", None)
    if u is None:
        return
    details = getattr(u, "prompt_tokens_details", None)
    extra = getattr(u, "model_extra", None) or {}
    _usage.calls = getattr(_usage, "calls", 0) + 1
    _usage.in_tok = getattr(_usage, "in_tok", 0) + (getattr(u, "prompt_tokens", 0) or 0)
    _usage.out_tok = getattr(_usage, "out_tok", 0) + (getattr(u, "completion_tokens", 0) or 0)
    _usage.cached = getattr(_usage, "cached", 0) + ((getattr(details, "cached_tokens", 0) or 0) if details else 0)
    _usage.cost = getattr(_usage, "cost", 0.0) + float(extra.get("cost") or 0.0)


class UsageModel(OpenAIModel):
    """OpenAIModel that records tokens / cached tokens / cost of *every* call, agent or not."""

    sticky_sessions = False

    def generate(self, *args, **kwargs):
        sid = getattr(_usage, "session", None)
        if self.sticky_sessions and sid:
            kwargs.setdefault("extra_headers", {"x-session-id": sid[:256]})
        left = time_left()
        if left is not None:      # a request that outlives the question's budget (600 s by default) is cut at the budget
            kwargs["timeout"] = max(MIN_CALL_S, left)
        msg = super().generate(*args, **kwargs)
        _record(getattr(msg, "raw", None))
        return msg


def build_model(settings: Settings, *, temperature: float | None = None, seed: int | None = None,
                thinking: bool | None = None, max_tokens: int | None = None) -> OpenAIModel:
    think = settings.thinking if thinking is None else thinking
    if settings.backend == "openrouter":
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (DATAMATE_BACKEND=openrouter)")
        base_url, api_key = settings.openrouter_base_url, settings.openrouter_api_key
        extra_body = {"reasoning": {"enabled": think}, "usage": {"include": True}}  # usage.include -> real cost
        if settings.openrouter_providers:
            extra_body["provider"] = {"order": settings.openrouter_providers, "allow_fallbacks": True}
    elif settings.backend == "vllm":
        base_url, api_key = settings.vllm_base_url, settings.vllm_api_key
        extra_body = {"chat_template_kwargs": {"enable_thinking": think}}
    elif settings.backend == "ollama":
        base_url, api_key = settings.ollama_base_url, "ollama"
        # Ollama's OpenAI endpoint ignores chat_template_kwargs, and `think: false` is ignored too (the model still thinks
        # and `content` comes back empty when max_tokens runs out inside the trace). `reasoning_effort` is what switches it.
        extra_body = {"reasoning_effort": "high" if think else "none"}
    else:
        raise ValueError(f"unknown backend {settings.backend!r}; use 'vllm', 'openrouter' or 'ollama'")

    UsageModel.sticky_sessions = settings.backend == "openrouter"
    extra = {} if seed is None else {"seed": seed}  # fixed seed: sampled runs stay reproducible between reruns
    return UsageModel(
        model_id=settings.model_id,
        api_base=base_url,
        api_key=api_key,
        temperature=settings.temperature if temperature is None else temperature,
        max_tokens=settings.max_tokens if max_tokens is None else max_tokens,
        extra_body=extra_body,
        **extra,
        # a request must never hang for ever (after the machine slept, connections went stale and 8 workers froze)
        client_kwargs={"timeout": settings.request_timeout, "max_retries": settings.max_retries},
    )


def chat(model: OpenAIModel, system: str, user: str, temperature: float | None = None, *, stage: str = "other",
         max_tokens: int | None = None, context: str | None = None) -> str:
    """Single-shot completion used by every stage. Missing content is "": a thinking call that spends its whole budget on
    reasoning returns no content, which str() used to turn into the text "None" (88 of 150 holdout6 r7 plans reached the
    Coder as "KẾ HOẠCH: None"). `stage` names the call in the per-question call log; `max_tokens` caps a stage whose
    answer is short (a JSON verdict), so a runaway generation stops early instead of filling the model's whole budget.
    `context` (agents/shared.py) is put before the stage's system prompt."""
    if context:   # agents/shared.py: the shop's context first, identical in every step, so the prefix cache serves it
        from datamate.agents.shared import with_context
        system = with_context(context, system)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    kw = {}
    if temperature is not None:
        kw["temperature"] = temperature
    if max_tokens is not None and isinstance(model, OpenAIModel):      # scripted test models take no generation options
        kw["max_tokens"] = max_tokens
    check_deadline(stage)
    before, t0 = get_usage(), time.time()
    try:
        content = model.generate(messages, **kw).content
    except Exception as e:  # noqa: BLE001 - a request cut by the budget's timeout is the budget running out, not a model failure
        left = time_left()
        if left is not None and left <= MIN_CALL_S:
            raise DeadlineExceeded(f"a model call ({stage}) was still running when the time budget ended: {type(e).__name__}") from e
        raise
    after = get_usage()
    out_tok = after["output_tokens"] - before["output_tokens"]
    defaults = getattr(model, "kwargs", None)
    cap = kw.get("max_tokens") or (defaults.get("max_tokens") if isinstance(defaults, dict) else None)
    if not hasattr(_usage, "call_log"):
        _usage.call_log = []
    _usage.call_log.append({"stage": stage, "s": round(time.time() - t0, 2), "in": after["input_tokens"] - before["input_tokens"],
                            "cached": after["cached_tokens"] - before["cached_tokens"], "out": out_tok,
                            "capped": bool(cap) and out_tok >= int(cap) - 2})
    text = content if isinstance(content, str) else "" if content is None else str(content)
    # a server without a reasoning parser returns the trace inline; no caller wants it
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
