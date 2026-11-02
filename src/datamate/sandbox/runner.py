"""RUNNER sub-agent: executes the Coder's code in the guarded sandbox and reports what happened.

One `Runner` lives for one turn: variables defined by earlier code blocks of the turn stay (like a notebook), the
inputs (tables, periods, engine functions...) are protected, and every block goes through `GuardedExecutor`, which
refuses fabricated data, moved periods and pasted answers, and turns opaque errors into hints. A block that sets
`answer` ends the turn: the guard appends `final_answer(answer)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smolagents.default_tools import FinalAnswerTool
from smolagents.local_python_executor import LocalPythonExecutor

from datamate.sandbox.guard import PERIOD_VARS, PROTECTED, GuardedExecutor, GuardRefusal
from datamate.sandbox.memory import OOM_HINT, is_oom

AUTHORIZED_IMPORTS = ["pandas", "numpy", "datetime", "re", "math", "unicodedata", "collections", "statistics", "itertools"]
MAX_OUTPUT = 2000      # characters of printed output / error fed back to the Coder
SHOW_EVIDENCE = (
    "Before concluding UNANSWERABLE, print with code: the table and column names in `tables`, the min / max date of the "
    "relevant date columns, and the row count after filtering to the period in the question. If the data DOES cover the "
    "period and has the needed columns you must compute the answer. Answer UNANSWERABLE only when the period lies entirely "
    "outside the dates the data covers or no column/table holds the needed information.")


def _is_unanswerable(v) -> bool:
    return (isinstance(v, str) and v.strip().upper().startswith("UNANSWERABLE")) or (
        isinstance(v, dict) and str(v.get("type", "")).lower() == "unanswerable")


@dataclass
class RunResult:
    code: str
    output: str = ""          # what the code printed
    error: str | None = None  # refused by the guard, or raised while running
    answered: bool = False    # the block set `answer` (the turn's value is `value`)
    value: Any = None
    refused: bool = False     # the guard refused it before it ran (the same block may run when sent again)

    def feedback(self) -> str:
        """The text the Coder reads back for this block."""
        if self.error:
            return f"ERROR: {self.error[:MAX_OUTPUT]}"
        out = self.output.strip()[:MAX_OUTPUT] or "(printed nothing)"
        return out + (f"\nanswer = {self.value!r:.300}" if self.answered else "")

    def as_log(self) -> dict:
        return {"code": self.code, "output": self.output[:MAX_OUTPUT], "error": self.error,
                "answered": self.answered, "value": repr(self.value)[:300] if self.answered else None,
                **({"refused": True} if self.refused else {})}


class Runner:
    def __init__(self, variables: dict, answerable: bool = False):
        inner = LocalPythonExecutor(additional_authorized_imports=AUTHORIZED_IMPORTS)
        inner.send_tools({"final_answer": FinalAnswerTool()})
        inner.send_variables(dict(variables))
        protected = PROTECTED | {v for v in PERIOD_VARS if variables.get(v) is not None}
        self.ex = GuardedExecutor(inner, protected)
        self.ex.tables = variables.get("tables")
        self.ex.frames = {k: variables[k] for k in ("sales_orders", "item_lines") if variables.get(k) is not None}
        il = variables.get("item_lines")
        self.ex.sales_table = il.attrs.get("source_table") if il is not None else None
        so = variables.get("sales_orders")
        self.ex.order_money = set(so.attrs.get("header_money", ())) if so is not None else set()
        self.asked_evidence = False   # the first UNANSWERABLE of a turn is sent back once: show evidence first
        self.asked_coverage = False   # ... and once more when the data covers the period of a question judged answerable
        self.answerable = answerable  # the Explainer's reading
        self.sales_orders, self.span = variables.get("sales_orders"), (variables.get("period_start"), variables.get("period_end"))

    def run(self, code: str) -> RunResult:
        res = RunResult(code=code)
        try:
            out = self.ex(code)
        except GuardRefusal as e:   # refused before running: same text as before ("ValueError: ..."), flagged
            res.error, res.refused = f"ValueError: {e}", True
            return res
        except Exception as e:  # noqa: BLE001 - refused or failed code is feedback for the Coder, not a crash
            res.error = OOM_HINT if is_oom(e) else f"{type(e).__name__}: {e}"
            return res
        res.output = str(getattr(out, "logs", "") or "")
        if getattr(out, "is_final_answer", False):
            if _is_unanswerable(out.output) and not self.asked_evidence:
                self.asked_evidence = True
                res.error = SHOW_EVIDENCE
                return res
            if _is_unanswerable(out.output) and self.answerable and not self.asked_coverage:
                self.asked_coverage = True
                if facts := self.coverage():
                    res.error = facts
                    return res
            res.answered, res.value = True, out.output
        return res

    def coverage(self) -> str:
        """Facts that contradict an UNANSWERABLE: valid orders exist in the question's period. Empty when there is nothing to
        say (no resolved period, no orders table, or the period holds no orders: then the data really lacks it). dev300:
        8 wrong UNANSWERABLE answers, several where the coder had not found the orders table, not where data was missing."""
        so, (lo, hi) = self.sales_orders, self.span
        if so is None or lo is None or hi is None or not {"date", "valid"} <= set(so.columns):
            return ""
        n = int(so.loc[(so["date"] >= lo) & (so["date"] <= hi), "valid"].sum())
        if n == 0:
            return ""
        return (f"The data DOES cover this period: `sales_orders` has {n} valid orders between {lo.date()} and {hi.date()} "
                f"(data range {so['date'].min().date()}..{so['date'].max().date()}), and the explainer judged the question "
                "answerable. Compute the answer from `sales_orders` / `item_lines` / the engine functions. Answer UNANSWERABLE "
                "only if a column the question needs truly does not exist (print the column names first).")

    @property
    def dropped(self) -> tuple[int, int]:
        """(period re-definitions, pasted literal answers) the guard removed this turn."""
        return self.ex.dropped, self.ex.dropped_literals
