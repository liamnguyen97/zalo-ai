"""SHARED PROMPT PREFIX: the shop's context, byte-identical at the start of every model call of a question.

Each stage used to open with its own system prompt (Explainer, Coder, judge, orchestrator), so the server's prefix cache matched
nothing past the first token when the stage changed: on the first local runs 42% of the prompt tokens had to be read again
(~33% of the model time). Now every call starts with this block (rulebook, README, the shop's schema cards, its engine
functions), then the stage's own instructions, then the question's part. With the full schema (fast P2) the block is also the
same for every question of a shop. OFF by default (`DATAMATE_SHARED_PREFIX=1` turns it on): on dev300 q_b445c50f6b5e the Coder
needed 7 and 12 calls with it (one abstain) against 1 and 2 without, its instructions coming after ~10k tokens of context
(ledger #9).
"""
from __future__ import annotations

from datamate.agents.explainer import ENGINE_NOTE

SEPARATOR = "\n\n========================================= YOUR ROLE IN THIS STEP =========================================\n"


def shared_context(rulebook: str, readme: str, schema: str, functions: list[str] | None) -> str:
    have = ", ".join(sorted(functions or [])) or "(none)"
    return ("# SHOP CONTEXT (identical in every step)\n"
            f"## RULEBOOK\n{rulebook}\n\n## SHOP README\n{(readme or '')[:6000]}\n\n## SCHEMA\n{schema}\n\n"
            f"## ENGINE FUNCTIONS\n{ENGINE_NOTE}\nAVAILABLE FUNCTIONS FOR THIS SHOP: {have}")


def with_context(context: str | None, system: str) -> str:
    """The system message: the shared context first, the stage's instructions after it."""
    return context + SEPARATOR + system if context else system
