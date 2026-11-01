# DataMate agentic harness (orchestrator + sub-agents, Qwen3.5-9B)

A main agent (`agents/orchestrator.py`) runs the sub-agents in an agent loop with up to 3 evaluated turns; see
`docs/agentic_system.md`. `pipeline.py` builds the agents, `context.py` holds the shared state.
Package layout (`src/datamate/`):

```
cli.py  config.py  pipeline.py  context.py  llm.py  text.py
data/     loader.py readme.py status.py descriptor.py          tables in, schema cards out
engines/  sales.py items.py debt.py orders.py functions.py      exact rulebook figures; functions the coder calls
agents/   orchestrator.py column_selector.py explainer.py lookups.py coder.py   the main agent and the LLM sub-agents
sandbox/  runner.py guard.py variables.py memory.py             the Runner sub-agent, its guard, the coder's inputs, memory cap
answer/   interpreter.py formatter.py evaluator.py              Interpreter, Formatter and Evaluator sub-agents
eval/     scorer.py runs.py                                     Penalized Accuracy, run folders
```

| Diagram box | Code | Notes |
|---|---|---|
| Table data | `data/loader.py` → `context.BusinessContext` | csv/xlsx/sqlite → DataFrames; header detection, month sheets and year files merged, VND/date coercion, group-row forward-fill, `status_std` |
| Columns → Column descriptor | `data/descriptor.py` | deterministic schema card (dtype, all values of low-cardinality columns, join hints); cached per business |
| Engines (exact by the rulebook) | `engines/` → `engines/functions.py` | functions the coder may call (`sales_metrics`, `item_qty`, `top_customers`, `debt_*`...), plus `sales_orders` (one row per order, §1 revenue, shift) and `item_lines` (base units, product attributes joined by code, or by folded name when lines carry no code) |
| Column selector | `agents/column_selector.py` | one LLM call picks tables/columns; unknown names fed back once; nothing is hidden from the coder |
| Explainer | `agents/explainer.py` | LLM loop: lookups (`agents/lookups.py`) → SPEC checked field by field against the data (plus three rulebook period fixes in code: "N ngày gần nhất", month-to-date vs previous month, "N days vs the N days before" get equal windows; a top-k spec typed string becomes a list) → independent type/measure reading → clarity check with quoted evidence; may name one `engine_call`; then a numbered plan |
| Orchestrator | `agents/orchestrator.py` | main agent: decides after the spec and after a rejected turn (coder / explainer / abstain); guardrails in code; up to `DATAMATE_MAX_TURNS` turns |
| Coder ⇄ Runner (feedback) | `agents/coder.py`, `sandbox/runner.py` | one code block per Coder call, run in the guarded sandbox (refuses fabricated data, moved periods, pasted answers, unknown `tables[...]` names; rewrites the blocked `pd.api.types.*`; a repeated block is not re-run and the next call is sampled; `sales_orders` / `item_lines` are also keys of `tables`; an UNANSWERABLE for a question judged answerable is bounced once more when the data has valid orders in the period); up to `DATAMATE_CODER_STEPS` blocks per turn; the process has a memory cap (`DATAMATE_MEM_LIMIT_GB`, 16): a runaway merge raises MemoryError into the block |
| Interpreter | `answer/interpreter.py` | types the value with the spec's answer type (month in words → `YYYY-MM`, counted lists → a count); lists doubts: empty, type, zero, walk-in, placeholder, malformed table / wrong top-k row count (a well-formed table passes), day outside the period (`DATAMATE_GATE`) |
| Evaluator | `answer/evaluator.py` | checks in code (including `engine_mismatch`: the answer differs from the engine reference, said once), then a judge that must quote evidence: accept / revise (coder or explainer) / unanswerable. The judge sees head and tail of the code and the English spec |
| Engine reference | `engines/reference.py` | the Explainer's `engine_call` is validated, run, shown to the Coder and compared with its answer; a Coder that still disagrees after one retry is trusted |
| Formatter | `answer/formatter.py` | pure code, exact submission format (`scripts/check_format.py` checks a run) |
| (extra) | `eval/scorer.py`, `eval/runs.py` | local clone of the official scorer + Penalized Accuracy; run folders and summaries |

Any failure or doubt becomes `abstain` (0 points) rather than a wrong answer (−1).

**Version 2 ("English agents", 2026-10-31).** The questions, rulebook, README and data stay Vietnamese, but every
instruction prompt, the Explainer's spec text (`rewritten_question`, `answer_form`, `assumptions`), the plan, the Coder's task,
the Evaluator's feedback and the sandbox hints are English; names, columns, unit words and every `quote` / `evidence` stay
verbatim Vietnamese (quotes are checked as substrings of the question). Version 1 (Vietnamese prompts) is commit `a5480ce`.
Also in v2, from the failure analysis of v1: the sandbox rewrites `pd.api.types.*` (blocked, 41 failed blocks) to provided
helpers and refuses unknown `tables[...]` names with the real ones (a name differing only in case/diacritics is fixed in
place); a repeated Coder block is not re-run and the next call is sampled (`REPEAT_TEMPERATURE`); a well-formed table is no
longer refused (`risky_type` now means malformed, `table_rows` a wrong top-k count); `repair_spec` gives "N days vs the N days
before" equal windows (the spec had 7 for 14, 4 for 7, 29 for 30); the Explainer may name one `engine_call`
(`engines/reference.py`); `sales_orders` / `item_lines` are keys of `tables`; the UNANSWERABLE coverage bounce.
Follow-up commit `24e0ce1` (top-k string → list; `~valid` also counts returned orders, use `status_std`) came after the run
below and is NOT measured.

## Baseline results v2 (this version, 2026-10-31)

Qwen3.5-9B bf16 on one RTX 5090 (vast.ai, South Korea, vLLM v0.31.0, 10.5 GB KV cache), `WORKERS=16`, temperature 0, thinking
off except the plan call. Code at `44864c9`, one uninterrupted run, 82 minutes.

**dev300, full run (`dev300-en-b1`, 300 questions): PA 0.707**
(255 correct, 43 wrong, 2 abstain, accuracy 0.850, 0 infrastructure failures, 0 OOM kills).

| Same 300 questions | correct | wrong | abstain | PA |
|---|---|---|---|---|
| **v2, English agents (this version)** | **255** | **43** | **2** | **0.707** |
| v1 + re-run of its failures (upper estimate, mixed code) | 242 | 44 | 14 | 0.660 |
| v1, Vietnamese prompts (`dev300-agents-a1`) | 214 | 59 | 27 | 0.517 |
| voting pipeline, `result/dev300` | 221 | 54 | 25 | 0.557 |

The fairer gain is +0.047 over the v1+a2 estimate; +0.19 over v1 includes fixes v1 had not had. dev300 is made of the 30
training businesses every heuristic was tuned on and some v2 fixes were chosen from its failures: it does not predict the
leaderboard (public test −0.006 for an earlier version). Run holdout6 and the public test before trusting the gain.

By gold answer type (correct / wrong / abstain), v2 vs v1: number 151/22/0 vs 133/35/5, percent 49/9/0 vs 39/16/3, list 15/4/0
vs 10/6/3, date 12/2/0 vs 12/1/1, string 11/0/1 vs 10/0/2, boolean 5/1/0 vs 5/0/1, **table 5/1/0 vs 0/0/6**,
unanswerable 7/4/1 vs 5/1/6.

Transitions from v1: wrong→correct 37, abstain→correct 18, wrong→wrong 22, **correct→wrong 13**, **abstain→wrong 8**,
correct→abstain 1, abstain→abstain 1.

Cost and speed: mean 245 s per question (median 258, p90 352, max 768) vs 280 s in v1; 75k input and 14k output tokens per
question (v1: 86k / 15k). 267 questions finished in 1 turn, 25 in 2, 8 in 3 (v1: 248 / 17 / 35). Wall-clock 82 minutes for 300
= 16.5 s per question, so about 4.6 h for 1000 questions on one 5090 (v1: ~7.8 h); the 6 h limit now fits one GPU with 1.4 h
margin on this host (hosts differ; keep the second replica as the safety plan).

Engine reference: the Explainer named an engine call on 112 questions; 107 of them are correct (v1 got 82 of the same 112,
but these are standard question types, so the comparison is confounded). `engine_mismatch` fired on 6 questions, 5 ended
correct. Spec status: clear 256, unchecked 36, fallback 8 (v1: 269 / 22 / 9).

**Regressions (correct in v1, not in v2): 14.** 2 involved the engine reference: `q_019ec9` (Explainer typed a top-3 as a
comma-separated string; fixed in `24e0ce1`) and `q_48c295` (stock question: the Explainer picked `item_qty`, the sales
quantity, the engine returned 7 against the gold 22 and the Coder trusted it). `q_1b04aa` (`~valid` counted returned orders,
77 vs gold 60; fixed in `24e0ce1`). The rest are not engine-related: average order value (3, off by 0.1-2%), morning/afternoon
shift revenue (2), size M/L cup counts (2, large misses), Nếp cái kg, a Thép Pomina revenue, a "who bought most" abstain,
a boolean with an empty-period spec.

Remaining after v2: 36 wrong values of the right type, 4 answered unanswerable questions (the gold is unanswerable),
1 table typed where the gold is a list, 1 string typed where the gold is a list, 1 said unanswerable with a gold value,
2 abstains.

## v2.1: the same agents plus the later checks (`dev300-en-b2`, commit `e1726c3`, 2026-11-01)

One full dev300 run, 16 workers, one 5090, 90 minutes (~$1.00): **PA 0.713** (252 correct, 38 wrong, 10 abstain, 0 infrastructure
failures, 0 OOM kills) against v2 `b1` 0.707 (255/43/2). Mean 252 s per question (b1 245 s); 271 questions in 1 turn, 17 in 2, 11 in 3.
Added since `44864c9`: top-k string → list; `status_std` guidance; rulebook calendar phrases computed in code; unanswerable
checks (period outside the data, walk-in-only customer ranking, `missing_fields`); Coder guards (item-attribute merge, `~valid`);
the onboarding report gating the engine reference; engine fixes found on the unseen shops (`Đơn số` order column, payments table
is not an order header, customers need a customer-named pair, item-name columns are not people, voucher lines out of revenue).

**The +0.006 is inside the noise.** The 259 questions touched by none of the new mechanisms changed anyway: 14 correct→wrong,
11 wrong→correct, 4 correct→abstain, 1 abstain→correct (temperature 0, but batching and the Coder's path vary). One run of each cannot
show a gain smaller than about ±0.05 PA. Judge a change by the questions where its mechanism fired, not by the total:

| Mechanism | Questions it touched | b1→b2 on those |
|---|---|---|
| Periods in code (spec periods differ in 25) | `33e8f7`, `95827d`, `e5f51c`, `f0565c`, `e2ea7a`, `019ec9` | 6 wrong→correct, 1 correct→wrong (`3fe3f0`, an Explainer variation), 3 wrong→wrong, 14 unchanged |
| Coverage shortcut (period outside the data) | `46e6ca` (gold unanswerable) | wrong→correct; fired once in 300 |
| Item-attribute merge guard | 26 | 5 wrong→correct, 2 correct→wrong, 1 correct→abstain, 15 unchanged, 3 still wrong |
| Repeated-block guard | 16 | 3 wrong→correct, 1 correct→wrong, 3 correct→abstain, 2 wrong→abstain: neutral, but 5 of the 9 new abstains are "no answer ×3" after a repeat |
| `missing_fields` | 6 | 4 gold-unanswerable (3 correct, 1 abstain), 2 answerable (both still correct): no harm |
| Gold-unanswerable questions (12) | | b1 7 correct/4 wrong/1 abstain → b2 8/2/2 |

The 9 new abstains (b1 correct or wrong): 3 customer-ranking questions where the Coder grouped `sales_orders` by hand and ranked
the walk-ins instead of calling `top_customers` (b1 called it; the engine data is unchanged), 5 "no answer ×3" Coder stalls
(4 after a repeat-block bounce), 1 gold-unanswerable question answered by abstaining (0 instead of 1 point).
Spec status: clear 245, unchecked 50, fallback 5 (b1: 256 / 36 / 8): the longer Explainer prompt (`engine_call`, `missing_fields`)
leaves more specs unchecked. Engine reference used on 110 questions, 104 correct.

Engine detection on the 14 unseen public shops (no gold; `scripts/onboard_report.py`): before the fixes 36 of 44 shops were
trusted and `dev_003` (129 public questions) had no orders or debt engine; after them `dev_003` has all four, `dev_007` and
`dev_008` have real customers (40% walk-in) instead of products, `biz_130` correctly has none. These fixes cannot show on dev300.

**Engine hint (commit after `f7299cf`, not yet run on the model).** The harness derives an advisory engine call from the checked spec
(`engines/routing.py`): `item_qty` for one item + quantity, `top_items` for top-k by quantity, `top_customers` for a money
ranking of customers; shown to the Coder with its value and the base unit, logged as `routing[].agrees`, never a verdict. Offline
against the dev300 gold with the b2 specs and the real engines: 29 hints, 27 of 28 scorable right (`item_qty` 16/16 where the Coder
got 9/16, `top_items` 9/10 against 8/10): 8 potential gains, 0 harm. Two shapes were dropped because they were worse than the Coder:
whole-shop `sales_metrics` (64/91 right against the Coder's 88/91: the spec cannot see conditions such as cancelled, channel or
shift) and `item_revenue` (11/17 against 16/17). Every wrong `item_qty` hint came from a spec whose "English rewrite" was the original
question copied back (the Explainer's fallback), so a hint needs a usable rewrite that is not about stock, price, cost or revenue.
The Coder's instructions also gained a RECIPES section (customer thresholds, cancelled vs returned, shift base, average order, nearest
expiry lot, stock vs sales, base units), each reproduced on the gold.

Next measurement: repeat the run (or the touched subsets) to shrink the noise, then holdout6 / the public test.

## v2.2: engine hints and recipes (`dev300-en-b3`, commit `bb07e92`, 2026-11-01)

| run | correct | wrong | abstain | PA |
|---|---|---|---|---|
| b2 (v2.1) | 252 | 38 | 10 | 0.713 |
| **b3 (v2.2)** | **259** | **36** | **5** | **0.743** |

Versus b2: 24 questions improved, 17 got worse (net +7), inside the ±12 run-to-run noise, so the total is not a proven gain.
Abstains fell 10 -> 5. Spec status: 256 clear / 37 unchecked / 7 fallback.

Advisory engine hint (`item_qty` only; `top_items` and `top_customers` never fired): shown on 13 questions, agreed with the
final answer on 12 (all correct; 5 of them were wrong in b2). One miss: Nếp cái hoa vàng, 2025-05, hint 32, b2 abstained,
b3 answered wrong. Hinted questions: about 5 gains against 1 loss.
Recipes (rulebook definitions in the Coder prompt): effect not separated from noise.
Remaining failures: 30 of 36 wrong are "wrong value, same type". Run cost about $0.81 (5090, 1.64 h).

## v2.3: structure cards (2026-11-01, NOT measured yet)

Found on a local Ollama run (train q_00a6e23cbacf, biz_109, "giá nhập bình quân một thùng Mì Omachi"): the model divided the
purchase slip's TOTAL (`Cần trả NCC`, every item on the slip) by one item's quantity (7,892,815 against the gold 200,556), after
the Explainer had looked for the slip date in the slip LINES table and called the tables by their README names, which are not
the loader's keys. All the needed facts were in the prompt; none was on the card of the table it belongs to.

`data/structure.py` measures, from the values only (no column-name rules): parent / child tables (unique id in one table,
repeated in another, ~all ids shared, <= 50 child rows per parent), document totals (a parent column equal to the sum of a child
column or of a product of two, on >= 95% of documents, 0 = 0 not counted), per-document columns of flat tables, README sentences
that define a column (attached to the table the sentence is about), and README table names. On the 30 training shops: 15 have
parent / child pairs, 16 flat tables in 14 shops, 30 document totals in 11 shops (5 single-table shops get nothing new); `Cần trả NCC` (or `TongGiaTriNhap`) = sum qty x unit price is found on all 7 shops
with purchase slips, including biz_123 whose README bold names match no column. The cards state it (`PARENT of`, `CHILD of ...;
only the parent has ...`, `DOCUMENT TOTAL ...`, `README: "..."`, `README name:`), +16% card text over the 30 shops; the
relation lines a parent / child line already states are dropped.

Also: README table names resolve to the real key in the Explainer lookups, `repair_spec`, the Column Selector and the sandbox;
`find_values` searches the other tables when the asked one has no such value; `repair_spec` moves a column put on the wrong
table to the table linked to it (the Omachi spec now passes with a note instead of an error); a table-name fix in the sandbox
guard now reaches the sandbox (before, a case / diacritics fix was made on the parsed tree only and the original text ran).

Off switch for A/B: `DATAMATE_STRUCTURE_CARDS=0`. Prepared measurement: `splits/purchase12` (12 purchase questions, one per
shop, `docs/purchase12_ids.txt`), `SPLIT=purchase12 DATAMATE_STRUCTURE_CARDS=0|1 scripts/ollama_run.sh <name>`. Not run on the
public shops yet: `scripts/onboard_report.py data/public_test/data` now prints what the cards will claim per shop.

## Results of v1 (Vietnamese prompts, 2026-10-30)

Qwen3.5-9B bf16 on one RTX 5090 (vast.ai, vLLM v0.31.0, 10.5 GB KV cache), `WORKERS=16`, temperature 0, thinking off
except the plan call. Code at `0359787` for the first 68 questions, `c82985e` for the rest (see "Run notes").

**dev300, first full run (`dev300-agents-a1`, 300 questions, v1): PA 0.517**
(214 correct, 59 wrong, 27 abstain, accuracy 0.713, 0 infrastructure failures).

| Same 300 questions | correct | wrong | abstain | PA |
|---|---|---|---|---|
| agentic system v1 | 214 | 59 | 27 | 0.517 |
| voting pipeline, `result/dev300` | 221 | 54 | 25 | 0.557 |
| voting pipeline v3, `result/dev300_v3` | 215 | 59 | 26 | 0.520 |

The agentic system is level with the voting pipeline, not better. dev300 is made of the 30 training businesses every
heuristic was tuned on: it does not predict the leaderboard (public test −0.006 for an earlier version). Judge changes by
whether they are generic, and check `docs/public_traces/`.

By gold answer type (correct / wrong / abstain): number 133/35/5, percent 39/16/3, list 10/6/3, date 12/1/1,
string 10/0/2, boolean 5/0/1, unanswerable 5/1/6, **table 0/0/6**.

Cost and speed: mean 280 s per question (median 284, p90 399, max 778) with 16 questions in flight, about 28 s of
wall-clock per question, 86k input and 15k output tokens per question. 248 questions finished in 1 turn, 17 in 2, 35 used
all 3 (most of those abstained). About 28 s/question means ~7.8 h for 1000 questions on one 5090: the 6 h limit needs
both GPUs (not load-balanced yet).

**Re-run of the 86 failed questions on the fixed code (`dev300-failed-a2`, finished, code `9b34b9d`, not `adc212b`):**
28 correct, 44 wrong, 14 abstain (mean 295 s per question). Transitions from the first run: wrong→correct 17,
abstain→correct 11, wrong→wrong 40, abstain→abstain 12, abstain→wrong 4, wrong→abstain 2. So 28 of 86 failures (33%) were
recovered, and 4 abstains became wrong answers (−4).

**Combined estimate: PA 0.660** (242 correct, 44 wrong, 14 abstain, accuracy 0.807): the 214 first-run correct answers kept
as they were, plus the a2 re-run of the failures. This is an upper estimate: the correct answers were not re-run on the
fixed code, so regressions are unmeasured, and the fixes were chosen by looking at exactly these failures (dev300 is also
the set every heuristic was tuned on). A full dev300 re-run on the final code is needed for a clean number.

Still failing after a2: tables 6 (debt top-N and aging, all abstain), customer ranking with walk-in 4, top-N ordering 4,
quantity/unit totals ~8, near-miss revenue 5 (`57cf71` and `6e3896` need `adc212b`), "unanswerable" calls 5, period/week
misreads (`cadff2` returned 0).

## Failure analysis of the first run (v1)

Abstains (27): table answers rejected by the whole-table check (5: debt top-N and aging; 6 table golds in all, 0 answered);
walk-in "khách lẻ" ranked first (4); the coder never set `answer` (9); judge rejecting a correct value (1); genuine zero
rejected (1); other (7).
Wrong (59): near-miss money totals (~10), period misreads (~6), top-N in the wrong order or with wrong items (~5), quantity
and unit totals (~8), unanswerable said but gold has a value (7), type misreads (3), answered an unanswerable question (1).

Root causes found and fixed in v1 (all generic, none business-specific):
1. **Judge saw code cut at 1500 characters** and called the cut "incomplete", rejecting a correct value three times.
   Now head + tail with a marker, 6000 characters, and the judge is told the code ran in full.
2. **Guard refused valid code**: `else: answer = "UNANSWERABLE…"` made `answer` a constant, so the closing `answer = answer`
   was refused as a pasted literal (identical block refused up to 6 times). Only top-level assignments define constants now;
   self-assignments are dropped.
3. **A genuine 0 was rejected by the zero check** (gold 0.0). A zero that comes back identical after being sent back is accepted.
4. **Period misreads**: "30 ngày gần nhất" came out as 31 days; month-to-date was compared with the whole previous month
   instead of the same number of days. Both are corrected from the rulebook in `repair_spec`.
5. **Product flag lost on a name merge** (biz_121: 82 of 3072 sales lines spell names differently from the product list;
   exact-name `merge` + `fillna(False)` counted prescription drugs as non-prescription, off by 7 000 and 5 500 đ).
   `item_lines` now joins attributes through the folded name when lines carry no code; the coder is told to filter
   `item_lines` instead of merging by name. Reproduces the gold exactly for both questions.
6. **Memory**: see Run notes.

Not explained by engine use: money answers that used an engine were correct 80% of the time, raw columns 82%.

## Run notes (what went wrong on the box)

- The harness was **OOM-killed** three times (cgroup limit 31.4 GB, `memory.events` `oom_kill`): once together with the vLLM
  engine (server down, 500s), twice the harness alone. Steady state is fine (anon 5.9 GB), so the cause is a spike from
  model-written pandas code running in-process. Added an address-space cap (`sandbox/memory.py`, 16 GB, Linux only) that
  turns it into a MemoryError reported to the coder. No kill since.
- First 68 questions ran on the uncapped code; the judge fix and the cap arrived mid-run, so a1 mixes code versions.
- `.vast_instance.json` can point at a destroyed instance (`status` still prints it); `vastai show instances` is the truth.
  Move the file away before `up`.
- After the engine died, `box_run.sh` waited for vLLM and resumed; infrastructure failures are re-run on resume.
- v2 run (`dev300-en-b1`): no OOM kill (anon memory ~7.5 GB, harness 1.6 GB RSS, load 0.5 on 16 cores: the run is GPU-bound,
  vLLM 16 running / 0 waiting / 0 preemptions, KV cache ~50%). Renting: vast offers vanish within seconds and "no longer
  available" costs nothing; the first box (Vietnam) was still loading after 16 min and was replaced. Skip China-hosted offers
  (model download), loop through offers. `box_deploy.sh` only builds splits/holdout6: build dev300 on the box with
  `python3 scripts/subsplit.py ids data/train splits/dev300 docs/dev300_ids.txt`. Cost ~$0.80 for the run (box $0.50-0.53/h).
  Monitors written in a zsh shell must not use `set -- $var` (no word splitting): use `read`.

## Next steps (in order of expected gain)

1. **Measure `24e0ce1`** (top-k string → list, `status_std` for cancelled) on the 43 wrong + 14 regressed questions, then a full
   re-run; then **holdout6 and the public test**: dev300 over-states the leaderboard.
2. **Engine choice errors**: `q_48c295` (stock asked, `item_qty` sold quantity picked) shows the cost of a wrong `engine_call`.
   No stock engine exists; restrict `engine_call` when the question measures stock (or add `stock_on(item, day)`), and check
   the function's meaning against the spec's measure.
3. **Unanswerable**: 4 answered questions whose gold is unanswerable and 1 wrong unanswerable (gold has a value). Compare the
   period with the data's date range in code before answering a number.
4. **Regression families**: average order value over a week (off by 0.1-2%: one denominator rule is inconsistent),
   morning/afternoon shift revenue for last week, size M/L cup counts (large misses): reproduce with the engines
   (`sales_metrics`, `size_qty`) and compare.
5. **Cost / profit engines**: gross profit, net profit and average purchase price (4 v1 failures) have import tables
   (`DanhSachChiTietNhapHang`) but no engine and no rulebook guidance used by the coder.
6. **Customer threshold questions** ("share of customers with at least N orders": 3 wrong): my check of a `sales_orders` groupby
   by `customer_key` was inconclusive (a single group); find the right key before writing an engine.
7. **Quantity and unit totals** (kg, thùng, price-increase effects): the base-unit conversion is used inconsistently.
8. **Throughput**: 16.5 s/question wall-clock (4.6 h for 1000) on one 5090 fits the 6 h limit with margin; two replicas and a load
   balancer are still not done, and the Docker host's memory cap and `WORKERS` are unchecked.
9. Determinism (rule 8): batch-invariant mode untested on Qwen3.5; compare two runs of the same split.

## Setup

```bash
uv sync
cp .env.example .env   # then edit
```

## Backends

`DATAMATE_BACKEND` (or `--backend`) selects the model server; both speak the OpenAI API.

**OpenRouter (temporary, dev only — the competition forbids network access):**
```bash
export OPENROUTER_API_KEY=...   # OPENROUTER_MODEL defaults to qwen/qwen3.5-9b
uv run datamate run --backend openrouter --split-dir <split> --out result/submission.jsonl
```

**Self-hosted vLLM (what the final Docker image must use):**
```bash
GPUS=2 scripts/serve_vllm.sh          # replicas on :8000 and :8001 (one per GPU)
uv run datamate run --backend vllm --split-dir <split> --out result/submission.jsonl
```
Needs CUDA ≥ 12.8 for RTX 5090. `DATAMATE_THINKING=0` disables Qwen3.5's long reasoning traces.
Only one `VLLM_BASE_URL` is used by the client; put a load balancer in front for two replicas (not done yet).
Runs on a rented box: `docs/box_runs.md`.

## Usage

```bash
# split dir contains data/, rulebook.md, calendar.csv, questions.jsonl (+ answers.jsonl for train)
uv run datamate run --split-dir path/to/train --business biz_118 --limit 20 --out result/dev.jsonl
uv run datamate score --pred result/dev.jsonl --gold path/to/train/answers.jsonl
python3 scripts/subsplit.py ids data/train splits/dev300 docs/dev300_ids.txt
.venv/bin/python scripts/holdout_report.py splits/dev300 <run_name> [baseline_run_name] --detail
uv run pytest
```
`run` writes per-question logs (column selection, spec, plan, generated code, observations, answer, elapsed time) to
`result/runs/<datetime>_<name>/logs/`, as the Docker rules require, and prints PA when `answers.jsonl` exists.
