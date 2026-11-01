# Harness review (agents harness, version 7)

Scope: the harness that produced holdout6 r6/r7 and the public run, with the later format fixes (commit 793c503).
Branch `refactor/agents-only` removes everything else.

## 1. Workflow

```
question ─► Column Selector ─► Explainer agent ─────────────────────────────► Coder ⇄ Runner ─► Interpreter ─► Formatter
            1 LLM call         loop: lookups → SPEC → data check →             up to 3 samples,    type + doubts    exact
            + name check       independent type reading → clarity check;       2 must agree        (gate)           format
                               then a numbered PLAN                                 ▲                  │
                                                                                    └── one RETRY ◄────┘ (reason + last code)
```

Model calls per question: 1 selector + 2–5 explainer turns + 1–2 type readings + 1–2 clarity checks + 1 plan + 2–3
coder samples (each up to 6 steps), and once more for a retry. Median on holdout6 r7 (16 workers on one 5090): 10
calls and 310 s per question.

**Findings**

| # | Finding | Status |
|---|---|---|
| W1 | **The plan reaches the coder as the text "None" for ~60% of questions** (88/150 on r6 and r7, 316/621 on public). The plan call runs in thinking mode and returns no content when reasoning uses the whole 12k-token budget; `chat()` turned the missing content into `"None"`. Those questions also take **about twice as long** (median 349 s vs 178 s on r7; 516 s vs 234 s on public) and use ~8k more output tokens, for nothing. | Bug fixed (empty plan is left out). **Speed fix not applied**: needs a holdout run (see Recommendations) |
| W2 | The agents pipeline inherited the classic `Pipeline` and built unused stages on every run (rule-based selector, old explainer, investigator, risk gate, fast-path router). | Removed: one `Pipeline` class |
| W3 | With no expected type and a numeric value, the interpreter asked the model "number or percent?" with an **empty question**. | Fixed: defaults to `number` without a call |
| W4 | Run time is set by GPU memory, not CPU. On one 5090, 16 workers fill the KV cache; 24–32 cause preemptions and slow the run (public run: 1.9 → 2.7 questions/min going from 24 to 16 workers). | Documented in docs/box_runs.md |

## 2. Agent system

| Agent | Input | Output | Checks the harness applies |
|---|---|---|---|
| Column Selector (`agents/column_selector.py`) | question, README, all table cards | tables / columns to read first | unknown names fed back once with the closest real names; nothing is hidden from the coder |
| Explainer (`agents/explainer.py`) | question, rulebook, README, focused schema, calendar facts | `Spec` + plan | every field checked against the data (`repair_spec`); entities snapped to real spellings; entities named in the question cannot be dropped; repeated "not in the data" errors settled; independent type/measure reading (two readings must agree before a measure is overruled); clarity issues kept only with a quote from the question |
| Coder (`agents/coder.py` + `sandbox/`) | task = rulebook, README, schema, spec, plan | `answer` | guarded sandbox: no re-assigning inputs, no pasted constants, no SystemExit; hints for missing columns that live on `sales_orders` / `item_lines`; raw status-text filters and order-header money sums refused once |
| Vote (`answer/vote.py`) | 2–3 coder samples | one value or abstain | values compared in canonical form (dates, rounding, a single name-value row = the name, counted lists = the count) |

**Findings**

| # | Finding | Status |
|---|---|---|
| A1 | The check model for the independent reading is coder sample model 1 (temperature 0.5). The reading is meant to be independent, but at 0.5 it is also noisy; two readings must agree before a measure is overruled, which keeps this safe. | Kept |
| A2 | 79 of 621 public answers abstain: 31 coder out of steps, 24 samples disagree, 24 gate. Out-of-steps and disagreement are the biggest remaining losses. | Next work |
| A3 | Two `fold` helpers behaved differently (one strips punctuation). Merging them would change matching. | Kept both, named and documented in `text.py` |

## 3. Prompts and instructions

| # | Finding | Status |
|---|---|---|
| P1 | The retry and the SystemExit message told the coder to "call `final_answer(...)`", while the coder instructions say never call it and assign `answer`. | Fixed: both now say `answer = ...` |
| P2 | Explainer prompt did not say how to read "T3/2025" (two public questions read it as Q3). | Added in 793c503 |
| P3 | Rulebook §7 (a property no column records is unanswerable) was missing from the Explainer and coder prompts; ~4–5 public questions answered "friendliest staff" style questions with a name. | Added in 793c503 (prompt hint; only one training example) |
| P4 | Stable text comes first in every prompt (rulebook, README, schema), the question last, so vLLM's prefix cache serves the long part across questions of a business. | Kept |
| P6 | The Explainer's spec reached the Coder as a raw JSON dump: the answer contract was buried mid-way, periods were not tied to the sandbox variables (a third period had none), and the Explainer's process notes ("như lần trước", "theo hướng dẫn sửa lỗi") were mixed with harness notes. | Coder task now framed BỐI CẢNH → CÂU HỎI → NHIỆM VỤ → ĐẦU VÀO (periods ↔ variables, exact values with the filter) → QUY TẮC → LƯU Ý (harness vs Explainer) → KẾ HOẠCH → ĐẦU RA (type format from §9 + measure source, last). Coder instructions regrouped under headers (same rules). Explainer told to write assumptions as interpretation choices only |
| P5 | The coder budget says "tối đa 6 bước" while the retry round has 8 steps. Harmless. | Kept |

## 4. Output format

All 621 public answers and all 150 r7 answers pass `scripts/check_format.py` (type names, shapes, 2 decimals, dates
`YYYY-MM-DD` / `YYYY-MM`, list / table layout). Content-level rules found by reading the answers:

| # | Rule (evaluation table, rulebook §9) | Status |
|---|---|---|
| F1 | A month answer is `date` `"YYYY-MM"`; `"tháng 9/2025"` is wrong. Six public answers were month text typed `string`. | Fixed (interpreter converts; root cause in the Explainer fixed) |
| F2 | The `type` must equal the reference's type; a percent typed `number` is wrong. | Independent reading now always runs (fixes the 2 public cases) |
| F3 | "How many items ..." is one number, not the list of names. | Count readings override a names-list spec; counted lists are turned into a count before the vote |
| F4 | Money is whole VND; percent and other non-integers 2 decimals; names exactly as in the data, without phone numbers or labels. | Already handled by `answer/formatter.py` |

## Layout

```
src/datamate/
  cli.py  config.py  pipeline.py  context.py  llm.py  text.py
  data/     loader, readme, status, descriptor          tables in, schema cards out
  engines/  sales, items, debt, orders, functions       exact rulebook figures; functions the coder calls
  agents/   column_selector, explainer, lookups, coder  the LLM stages (+ the Explainer's lookup tools)
  sandbox/  guard, variables                            guarded execution of the coder's code, its inputs
  answer/   vote, interpreter, retry, formatter         from raw values to the submission line
  eval/     scorer, runs                                Penalized Accuracy, run folders
```

Renamed on the way: `engine_tools` → `engines/functions`, `col_selector` → `agents/column_selector`,
`explainer_agent` → `agents/explainer`, `tools` → `agents/lookups`, `consistency` → `answer/vote`, and `runner` split
into `sandbox/guard` (the guarded executor) and `sandbox/variables` (the coder's inputs).

## Removed (branch `refactor/agents-only`)

Modules: `harness.py` (merged into `pipeline.py`), `explainer.py`, `investigator.py`, `selector.py`, `risk.py` +
`risk_topics.json`, `router.py`, `intent.py`, `prompts_en.py`, `entities.py`, `normalize.py`. Settings: fast path,
rule selector, language, risk gate, investigator, harness switch, thinking stages (now `DATAMATE_PLAN_THINKING`).
Classic-only code inside kept modules (reference tables, `items_reference`, the answerability veto, English
messages, `SalesView.describe`, `Spec.from_dict`). Scripts that replayed or validated the classic stages. Source:
6,070 → 4,234 lines. Tests: 129 pass; an offline smoke test ran the real pipeline on 6 holdout6 questions with real
data and scripted model replies (6/6 correct).

## Recommendations (need a model run)

1. **Plan without thinking** (`DATAMATE_PLAN_THINKING=0`) or no plan at all. Expected to cut time per question by
   about 40% (W1). Accuracy was the same with and without a plan on r6 (95% vs 95%), but the questions with a plan were
   chosen by the model finishing its reasoning, so score holdout6 before switching.
2. Out-of-steps (31 public abstains): give the coder a ready-made `sales_orders` / `item_lines` example per question
   type in the plan, or raise `max_steps` only for the retry.
3. Run holdout6 2–3 times per change and average: one run moves by ±0.03 PA on its own.
