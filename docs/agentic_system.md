# Agentic system: orchestrator + sub-agents

```
question ─► ORCHESTRATOR (main agent, agents/orchestrator.py)
              │ column_selector ─► explainer ─► [decide: coder | explainer once]
              │
              │  turn 1..3 ───────────────────────────────────────────────────────────────────┐
              │   coder ⇄ runner   (≤ 6 blocks: the Coder writes one block, the Runner runs it, │
              │                     output / error goes back, until a block sets `answer`)       │
              │   interpreter ─► formatter ─► evaluator                                          │
              │        accept / unanswerable ─► submit                                           │
              │        revise ─► [decide: coder (with the reason) | explainer (once) | abstain] ─┘
              │
              └─► after 3 rejected turns: abstain (0 points; a wrong answer is -1)
```

| Sub-agent | Module | Kind | Input → output |
|---|---|---|---|
| Column Selector | `agents/column_selector.py` | 1 model call + data check | question, README, table cards → tables/columns to read first |
| Explainer | `agents/explainer.py` | model loop (lookups, data checks, independent reading, clarity check) + plan | question → checked `Spec` + plan; re-run with the orchestrator's feedback |
| Coder | `agents/coder.py` | 1 model call per block | task (context, spec, plan, output contract) + this turn's blocks and results + earlier feedback → one code block |
| Runner | `sandbox/runner.py` | code | code block → printed output / error, `answer` (state kept within a turn; guarded sandbox) |
| Interpreter | `answer/interpreter.py` | code | value → type + doubts (empty, type, zero, walk-in, placeholder, table, day outside the period) |
| Formatter | `answer/formatter.py` | code | typed value → submission line |
| Evaluator | `answer/evaluator.py` | checks in code, then 1 judge call | turn → accept / revise (who fixes: coder or explainer, why) / unanswerable |

## The orchestrator and its guardrails

The orchestrator is a model at the two points where there is a real choice; everywhere else the next step is fixed and
taken by code (no model call to say "now run the code"):

| Decision point | Allowed actions | Default |
|---|---|---|
| after the Explainer | `coder`, `explainer` (while a re-run is left) | `coder` |
| after a rejected turn (turns left) | `coder`, `abstain`, `explainer` (while a re-run is left) | the Evaluator's `target` |

It sees the question, the spec summary and every earlier turn (answer, verdict, reason) and replies
`{"action", "message", "reason"}`; the message is passed to the sub-agent it calls. Guardrails in code: only the
allowed actions are offered; any other reply falls back to the default (logged as `by: default`); Explainer re-runs
(`DATAMATE_EXPLAINER_RERUNS`, 1), turns (`DATAMATE_MAX_TURNS`, 3) and blocks per turn (`DATAMATE_CODER_STEPS`, 6) are
capped. It replies in JSON rather than native tool calls: the vLLM server runs without a tool-call parser.

## The Evaluator

1. Checks in code, no model call: no `answer` within the blocks; the Interpreter's doubts; a value the Formatter cannot
   format. Each sends the turn back with a concrete fix (the old retry hints).
2. Judge (one call): question, output contract, last code and its output, the answer. `revise` must quote evidence that
   is in the question or the code/output, otherwise it is ignored; an unreadable reply accepts (the checks passed). A judge
   that thinks the data is insufficient while the code computed a value sends it back to the Coder to check, rather
   than dropping it.

The Runner sends the first `UNANSWERABLE` of a turn back once, asking for evidence (the old unanswerable gate).

## What changed from the voting pipeline

- No vote: one Coder per turn at temperature 0 (reproducible, rule 8), corrected by the Evaluator instead of agreed by
  2 of 3 samples. Removed `answer/vote.py`, `answer/retry.py`, the smolagents `CodeAgent` (the Coder ⇄ Runner loop is
  now the orchestrator's), `DATAMATE_SAMPLES`, `DATAMATE_MIN_AGREE`, `DATAMATE_RETRY*`.
- The Explainer's independent type reading keeps its own sampled model (`DATAMATE_CHECK_TEMPERATURE`, 0.5) so it is
  not the same greedy call as the Explainer's own reading.
- Logs: `log["orchestrator"]` (every sub-agent call with its time, every decision with `by: model | default | rule`),
  `log["turns"]` (blocks, value, type, doubts, submission line, verdict). `scripts/trace_report.py` and
  `scripts/holdout_report.py` read them.

## Not yet measured

Only offline: 12 loop tests with a scripted model and the real sandbox, and the 6-question smoke test on real holdout6
data (6/6). Before trusting it: score holdout6 (and compare with r6 0.927 / r7 0.867), check how often the judge
rejects right answers (`source: judge` on questions the old pipeline got right), and the time per question (a rejected
turn costs a full Coder ⇄ Runner round).
