"""Runtime settings, read from environment variables (and a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(int(default))).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # "vllm" = self-hosted OpenAI-compatible server, "openrouter" = hosted API (dev only,
    # the competition forbids network access at inference).
    backend: str = field(default_factory=lambda: os.getenv("DATAMATE_BACKEND", "vllm"))

    vllm_base_url: str = field(default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    vllm_model: str = field(default_factory=lambda: os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-9B"))
    vllm_api_key: str = field(default_factory=lambda: os.getenv("VLLM_API_KEY", "EMPTY"))

    openrouter_base_url: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    openrouter_model: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", "qwen/qwen3.5-9b"))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))

    # "ollama" = a local Ollama server (its OpenAI-compatible endpoint), for development on a laptop. The model is the
    # derived one from scripts/ollama/Modelfile (bigger context, presence_penalty 0), created by scripts/ollama_run.sh.
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3.5-datamate"))

    # Optional preferred OpenRouter providers, in order (fallbacks stay allowed). Empty = OpenRouter's default
    # routing (fast, ~70-100 tok/s). "Darkbloom" is the cheapest and the only qwen3.5-9b host that bills a
    # cache-read discount, but it is ~5-8x slower and serves fp4 weights; Parasail/DeepInfra (bf16) rate-limit
    # at ~6 concurrent requests.
    openrouter_providers: list[str] = field(
        default_factory=lambda: [p.strip() for p in os.getenv("OPENROUTER_PROVIDERS", "").split(",") if p.strip()]
    )

    temperature: float = field(default_factory=lambda: float(os.getenv("DATAMATE_TEMPERATURE", "0.0")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("DATAMATE_MAX_TOKENS", "4096")))
    # Qwen3.5 can emit long reasoning traces; off by default to keep latency down.
    thinking: bool = field(default_factory=lambda: _flag("DATAMATE_THINKING", False))

    # Agent loop (agents/orchestrator.py): evaluated turns before abstaining, Coder <-> Runner blocks per turn, and how
    # often the orchestrator may send the question back to the Explainer.
    max_turns: int = field(default_factory=lambda: int(os.getenv("DATAMATE_MAX_TURNS", "3")))
    coder_steps: int = field(default_factory=lambda: int(os.getenv("DATAMATE_CODER_STEPS", os.getenv("DATAMATE_MAX_STEPS", "6"))))
    explainer_reruns: int = field(default_factory=lambda: int(os.getenv("DATAMATE_EXPLAINER_RERUNS", "1")))
    # temperature of the Explainer's independent type / measure reading (a second, differently sampled reading)
    check_temperature: float = field(default_factory=lambda: float(os.getenv("DATAMATE_CHECK_TEMPERATURE", "0.5")))

    # Checks in code before the Evaluator's judge: a value that looks like a failed computation (empty, wrong type, 0,
    # walk-in customer ranked, placeholder, day outside the period) sends the turn back. See answer/evaluator.py.
    gate: bool = field(default_factory=lambda: _flag("DATAMATE_GATE", True))

    # Per-request timeout (s). A self-hosted server queues requests under load, so it gets a longer default.
    # API retries per request (exponential backoff, ~8 s cap). A dropped SSH tunnel takes ~30 s to come back; 2 retries
    # turned every in-flight question into an abstain, 8 ride it out.
    max_retries: int = field(default_factory=lambda: int(os.getenv("DATAMATE_MAX_RETRIES", "8")))
    request_timeout: float = field(default_factory=lambda: float(os.getenv(
        "DATAMATE_TIMEOUT", "90" if os.getenv("DATAMATE_BACKEND", "vllm") == "openrouter" else "600")))

    # Explainer agent's call budget (agents/explainer.py): model calls in its loop, and clarity checks among them.
    explainer_calls: int = field(default_factory=lambda: int(os.getenv("DATAMATE_EXPLAINER_CALLS", "5")))
    explainer_checks: int = field(default_factory=lambda: int(os.getenv("DATAMATE_EXPLAINER_CHECKS", "2")))

    # Thinking for the plan call only. Most wrong answers are misreads of the question (scope, revenue column, per-unit
    # vs total), which the plan decides. The multi-step coder stays without thinking: its traces would pile up in the
    # context. The thinking call gets its own output cap.
    plan_thinking: bool = field(default_factory=lambda: _flag("DATAMATE_PLAN_THINKING", True))
    thinking_max_tokens: int = field(default_factory=lambda: int(os.getenv("DATAMATE_THINKING_MAX_TOKENS", "12288")))

    # P2 (understanding the question). "fast": no Column Selector (every table's card is in the prompt), the question read in code
    # (agents/question.py), ONE spec call, checks in code, at most one fix call, no orchestrator decision after the spec.
    # "legacy": Column Selector -> Explainer loop (lookups, blind reading, clarity checks) -> plan -> orchestrator decision.
    p2_mode: str = field(default_factory=lambda: os.getenv("DATAMATE_P2", "fast"))
    # Where the plan for the Coder comes from: "spec" = a `method` field of the spec call (no extra call), "call" = the separate
    # plan call, "none" = no plan (the Coder plans in its first block).
    plan_mode: str = field(default_factory=lambda: os.getenv("DATAMATE_PLAN_MODE", "spec"))
    spec_max_tokens: int = field(default_factory=lambda: int(os.getenv("DATAMATE_SPEC_MAX_TOKENS", "3000")))
    # Every call opens with the same shop context (rulebook, README, schema, functions): the prefix cache serves it across
    # steps (agents/shared.py). OFF by default: with it the Coder needed 7 and 12 calls on dev300 q_b445c50f6b5e (one abstain)
    # against 1 and 2 without (ledger #9); to be retried with the Coder's key instructions repeated after the question.
    shared_prefix: bool = field(default_factory=lambda: _flag("DATAMATE_SHARED_PREFIX", False))

    # Coder phase (agents/guideline.py, agents/facts.py): the rulebook rows that decide the question, repeated right before the
    # Coder's OUTPUT with the column / function that applies them (0 = off: the spec's rows stay in CALCULATION RULES), and the
    # counts computed from the data for the question's periods (VERIFIED FACTS).
    guideline_rows: int = field(default_factory=lambda: int(os.getenv("DATAMATE_GUIDELINE_ROWS", "4")))
    verified_facts: bool = field(default_factory=lambda: _flag("DATAMATE_VERIFIED_FACTS", True))

    workers: int = field(default_factory=lambda: int(os.getenv("DATAMATE_WORKERS", "4")))

    @property
    def model_id(self) -> str:
        return {"openrouter": self.openrouter_model, "ollama": self.ollama_model}.get(self.backend, self.vllm_model)
