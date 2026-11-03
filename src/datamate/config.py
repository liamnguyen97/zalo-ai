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
    # The spec call also reads the question with its shorthand written out by code (agents/question.written_out: lexicon words in
    # full with accents, telex typing decoded), next to the original question, which stays the authority. And every condition the
    # spec filters by quotes the question's own words (Spec.conditions): a condition whose quote is not in the question goes back to
    # the fix call (the spec added "prescription" to "doanh thu thuốc hoạt chất Vitamin C", dev300 q_83273acf61f7). 0 = off (A/B).
    written_out: bool = field(default_factory=lambda: _flag("DATAMATE_WRITTEN_OUT", True))
    spec_conditions: bool = field(default_factory=lambda: _flag("DATAMATE_SPEC_CONDITIONS", True))
    # Every call opens with the same shop context (rulebook, README, schema, functions): the prefix cache serves it across
    # steps (agents/shared.py). OFF by default: with it the Coder needed 7 and 12 calls on dev300 q_b445c50f6b5e (one abstain)
    # against 1 and 2 without (ledger #9); to be retried with the Coder's key instructions repeated after the question.
    shared_prefix: bool = field(default_factory=lambda: _flag("DATAMATE_SHARED_PREFIX", False))

    # Coder phase (agents/guideline.py, agents/facts.py): the rulebook rows that decide the question, repeated right before the
    # Coder's OUTPUT with the column / function that applies them (0 = off: the spec's rows stay in CALCULATION RULES), and the
    # counts computed from the data for the question's periods (VERIFIED FACTS).
    guideline_rows: int = field(default_factory=lambda: int(os.getenv("DATAMATE_GUIDELINE_ROWS", "4")))
    verified_facts: bool = field(default_factory=lambda: _flag("DATAMATE_VERIFIED_FACTS", True))
    # §7 information coverage (agents/coverage.missing_information): a question asking for a kind of information the shop does not
    # record (gender, address, stock, purchases, cost, debt, expiry, named customers). fact (default) = a line in the Coder's VERIFIED
    # FACTS, the Coder and the judge decide; on = UNANSWERABLE in code before any model call for the needs measured precise (train: 28
    # fires, all on gold-unanswerable questions; unseen shops may name their columns otherwise); off = not run.
    coverage_gate: str = field(default_factory=lambda: os.getenv("DATAMATE_COVERAGE_GATE", "fact"))

    # P4 / P5 (ledger #12). After a rejected turn: "rule" = a rule table on the verdicts picks coder / explainer / abstain (no model
    # call), "model" = the orchestrator model picks, as before (the rule's choice is logged as `rule_would`). No judge call when the
    # value equals an exact engine function's: OFF by default (dev300: -1 question for ~0.4% of the time). A value the judge
    # rejected and the Coder returned unchanged is accepted. An UNANSWERABLE the code concluded is submitted without the judge
    # (dev300: PA 0.840 -> 0.857 on the same trajectories). The judge reads the VERIFIED FACTS and the engine value. The code audits
    # (answer/audit.py: entity named by the question absent from the code, profit larger than revenue): off | shadow | on.
    decide_mode: str = field(default_factory=lambda: os.getenv("DATAMATE_DECIDE", "rule"))
    skip_judge_on_engine: bool = field(default_factory=lambda: _flag("DATAMATE_SKIP_JUDGE_ON_ENGINE", False))
    trust_repeat: bool = field(default_factory=lambda: _flag("DATAMATE_TRUST_REPEAT", True))
    judge_unanswerable: bool = field(default_factory=lambda: _flag("DATAMATE_JUDGE_UNANSWERABLE", False))
    judge_facts: bool = field(default_factory=lambda: _flag("DATAMATE_JUDGE_FACTS", True))
    # The judge also reads the periods code computed (the dates the question names by the rulebook, the spec's periods, the period
    # issues the spec stage left open), so it checks the Coder's dates against them instead of guessing them. OFF until measured.
    judge_periods: bool = field(default_factory=lambda: _flag("DATAMATE_JUDGE_PERIODS", False))
    # The judge's output cap. 800 cut 8 of 287 dev300 judge replies before their closing brace: read as "unreadable" = accept. A reply
    # averages ~180 tokens, so a higher cap changes only the replies that were cut.
    judge_max_tokens: int = field(default_factory=lambda: int(os.getenv("DATAMATE_JUDGE_MAX_TOKENS", "1500")))
    # v2: the judge writes verdict, target and evidence before problem / fix, and a reply cut by the cap is read up to the cut (its
    # complete fields; a revise still needs its whole verbatim evidence). v1: the earlier prompt and parser (A/B).
    judge_format: str = field(default_factory=lambda: os.getenv("DATAMATE_JUDGE_FORMAT", "v2"))
    # fragments: a judge revise also stands when its evidence, split at "..." and stripped of labels, has a piece verbatim in the
    # code-computed FACTS / ENGINE VALUE (dev300: 2 right diagnoses on wrong answers were dropped). strict: the whole quote, as before.
    judge_evidence: str = field(default_factory=lambda: os.getenv("DATAMATE_JUDGE_EVIDENCE", "fragments"))
    audit_mode: str = field(default_factory=lambda: os.getenv("DATAMATE_AUDIT", "shadow"))
    # With DATAMATE_AUDIT=on, the audits listed here are still only logged (comma list: entity_absent, profit_bound, period_mismatch,
    # date_literal). date_literal is a heuristic on the Coder's own dates: log it before enforcing it.
    # Engine hints (engines/routing.py) the Coder is sent back on, once, when its value differs ("hint_mismatch", like engine_mismatch):
    # only functions whose hint precision was measured high (dev300: item_qty 16/16). Never the whole-shop sales_metrics (64/91).
    enforce_hints: tuple = field(default_factory=lambda: tuple(x.strip() for x in os.getenv("DATAMATE_ENFORCE_HINTS", "item_qty").split(",") if x.strip()))
    audit_skip: tuple = field(default_factory=lambda: tuple(x.strip() for x in os.getenv("DATAMATE_AUDIT_SKIP", "date_literal").split(",") if x.strip()))

    # Time budget (error-analysis P0). Rule 7: the whole private set within 6 h; Rule 8: if the run is cut, only the lines already
    # in submission.jsonl are scored. A question that runs past `question_deadline_s` (p90 794 s, max 2218 s on public v3.5) stops
    # before its next model call, and every request is cut at the time left. What it submits then: "checked" = the latest value
    # that passed the checks in code (even if the judge had rejected it), "abstain" = nothing. `run_deadline_s` (0 = no limit) is
    # the whole run's: with `run_quick_s` or less left, a question starts with one turn and no Explainer re-run; once it has passed,
    # the questions still waiting are abstained without a model call.
    question_deadline_s: float = field(default_factory=lambda: float(os.getenv("DATAMATE_QUESTION_DEADLINE_S", "900")))
    run_deadline_s: float = field(default_factory=lambda: float(os.getenv("DATAMATE_RUN_DEADLINE_S", "0")))
    run_quick_s: float = field(default_factory=lambda: float(os.getenv("DATAMATE_RUN_QUICK_S", "1800")))
    deadline_fallback: str = field(default_factory=lambda: os.getenv("DATAMATE_DEADLINE_FALLBACK", "checked"))

    workers: int = field(default_factory=lambda: int(os.getenv("DATAMATE_WORKERS", "4")))

    @property
    def model_id(self) -> str:
        return {"openrouter": self.openrouter_model, "ollama": self.ollama_model}.get(self.backend, self.vllm_model)
