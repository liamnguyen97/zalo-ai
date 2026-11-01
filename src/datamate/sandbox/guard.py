"""RUNNER guard: wraps the smolagents sandbox in the CODER <-> RUNNER loop.

`GuardedExecutor` refuses code that fabricates data or hard-codes an answer, drops re-definitions of the given periods
and pasted literal answers, and turns errors into hints; its errors are fed back to the coder (the FEEDBACK arrow of the
diagram).
"""
from __future__ import annotations

import ast
import re

import difflib

from datamate.data.structure import resolve_table
from datamate.engines.functions import TOOL_NAMES
from datamate.text import fold
from datamate.sandbox.variables import DTYPE_HELPERS

# Real inputs of the sandbox. Re-assigning them is how the model fabricates data (seen: sample tables in the public
# run) or silently moves the period boundary (seen: `period_end = pd.Timestamp('2025-06-08')` dropped the last day's
# orders, which carry a time). as_of_date may be re-assigned: harmless, and models do it by habit.
PROTECTED = {"tables", "calendar", *TOOL_NAMES}
PERIOD_VARS = ("period_start", "period_end", "compare_start", "compare_end")  # protected only when resolved (not None)
CONST_FUNCS = {"round", "int", "float", "abs", "str"}  # wrappers that keep an expression constant


def _is_unanswerable_const(n: ast.AST) -> bool:
    if isinstance(n, ast.Constant):
        return isinstance(n.value, str) and n.value.upper().startswith("UNANSWERABLE")
    if isinstance(n, ast.Dict):  # {"type": "unanswerable", ...}
        return any(isinstance(v, ast.Constant) and isinstance(v.value, str) and v.value.lower() == "unanswerable" for v in n.values)
    return False


def is_const_expr(n: ast.AST, consts: set[str]) -> bool:
    """True if the expression is built only from literals (or names that hold such literals).

    A model cannot know a computed result before the code runs, so a constant answer is a guess
    (seen: the code printed 629820000, then `total = round(128470000.0)` and answered that).
    """
    if isinstance(n, ast.Constant):
        return True
    if isinstance(n, ast.Name):
        return n.id in consts
    if isinstance(n, ast.UnaryOp):
        return is_const_expr(n.operand, consts)
    if isinstance(n, ast.BinOp):
        return is_const_expr(n.left, consts) and is_const_expr(n.right, consts)
    if isinstance(n, (ast.List, ast.Tuple, ast.Set)):
        return all(is_const_expr(e, consts) for e in n.elts)
    if isinstance(n, ast.Dict):
        return all(k is None or is_const_expr(k, consts) for k in n.keys) and all(is_const_expr(v, consts) for v in n.values)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in CONST_FUNCS:
        return all(is_const_expr(a, consts) for a in n.args) and not n.keywords
    return False


class GuardedExecutor:
    """Wraps smolagents' sandbox. The harness, not the model, owns the final answer.

    - refuses code that re-assigns the protected inputs (fake sample data, moved period boundaries);
    - refuses a constant answer (literal, or a variable that only holds literals) in a block that also
      computes something; a bare block after the model has seen the value is allowed;
    - if the code sets `answer = ...` without calling `final_answer`, appends `final_answer(answer)`.
    """

    def __init__(self, inner, protected: set[str] | None = None):
        self._inner = inner
        self._protected = PROTECTED if protected is None else protected
        self.dropped = 0  # re-definitions of resolved periods removed (they are authoritative)
        self.dropped_literals = 0  # pasted literal answers removed after a computed `answer = <expr>`
        self._seen_text = ""  # everything earlier steps printed: a literal answer seen there is not a guess
        self.tables: dict | None = None  # for richer error messages
        self.frames: dict = {}           # sales_orders / item_lines: derived columns the raw tables lack
        self._consts: set[str] = set()  # names known to hold literals, kept across steps
        self.status_bounced = False  # raw status-text filter refused once already
        self.order_money: set[str] = set()  # order-header money columns (see engines/orders.py)
        self.money_bounced = 0
        self.merge_bounced = False  # a name-merge for product attributes refused once already
        self.sales_table: str | None = None  # key of the sales-lines table `item_lines` is built from (None: unknown)
        self.valid_bounced = False  # `~valid` read as "cancelled" refused once already

    def __call__(self, code_action: str):
        try:
            tree = ast.parse(code_action)
        except SyntaxError:
            return self._inner(code_action)  # let the sandbox report the syntax error as usual
        tree, rewritten = self._rewrite_dtype_helpers(tree)
        tree, dropped = self._drop_period_redefinitions(tree)
        tree, dropped_lit = self._drop_literal_after_computed_answer(tree)
        tree, noop = self._drop_self_assignments(tree)
        renamed = self._check_table_names(tree)
        if dropped or dropped_lit or noop or rewritten or renamed:
            self.dropped += dropped
            self.dropped_literals += dropped_lit
            code_action = ast.unparse(tree)
        self._check_protected(tree)
        self._check_raw_status_filter(tree, code_action)
        self._check_item_attribute_merge(tree)
        self._check_valid_as_cancelled(tree, code_action)
        self._check_header_money_sum(tree, code_action)
        consts = set(self._consts)
        body = tree.body
        top = {id(st) for st in body}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        # only an unconditional top-level literal makes a name constant: `else: answer = "UNANSWERABLE"`
                        # must not turn the later `answer = answer` (or `x = answer`) into a "pasted literal"
                        (consts.add if id(node) in top and is_const_expr(node.value, consts) else consts.discard)(t.id)
        trivial = all(
            isinstance(st, (ast.Import, ast.ImportFrom, ast.Pass))
            or (isinstance(st, ast.Assign) and is_const_expr(st.value, consts))
            or (isinstance(st, ast.Expr) and (isinstance(st.value, ast.Constant) or self._is_final_call(st.value)))
            for st in body
        )
        has_final, sets_answer = False, False
        for node in ast.walk(tree):
            if self._is_final_call(node):
                has_final = True
                if node.args and is_const_expr(node.args[0], consts) and not _is_unanswerable_const(node.args[0]) \
                        and not trivial and not self._was_printed(node.args[0]):
                    raise ValueError(self._msg())
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "answer" for t in node.targets):
                sets_answer = True
                # A literal inside a branch (`if not mask.any(): answer = 0`) is a fallback, not a guess: only the
                # unconditional top-level assignment is refused (agents run: 3 of 4 abstains were this).
                if node in body and is_const_expr(node.value, consts) and not _is_unanswerable_const(node.value) \
                        and not trivial and not self._was_printed(node.value):
                    raise ValueError(self._msg())
        if sets_answer and not has_final:
            code_action = code_action.rstrip() + "\nfinal_answer(answer)\n"
        self._consts = consts
        try:
            res = self._inner(code_action)
            self._seen_text += f"\n{getattr(res, 'logs', '')}\n{getattr(res, 'output', '')}"
            return res
        except SystemExit as e:
            # `raise SystemExit` escapes the sandbox and, through the thread pool, silently ended whole runs (exit 0,
            # no summary: dev300b A at 222/300, B' at 248, target R1 at 46, R3 at 69). Turn it into a step error.
            raise ValueError(f"Do not stop the program (SystemExit {e.code!r}): assign `answer = <value>`, or "
                             "`answer = \"UNANSWERABLE: <reason>\"` when the data is insufficient.") from None
        except Exception as e:  # noqa: BLE001 - re-raised with a more useful message for the feedback loop
            better = self.explain_error(str(e), code_action)
            if better == str(e):
                raise
            try:
                err = type(e)(better)
            except Exception:  # noqa: BLE001 - exception type with an unusual constructor
                raise e from None
            raise err from None

    @staticmethod
    def _rewrite_dtype_helpers(tree: ast.Module) -> tuple[ast.Module, int]:
        """`pd.api.types.is_datetime64_any_dtype(x)` -> `is_datetime64_any_dtype(x)`: the sandbox refuses `pd.api` but provides
        the helpers as variables."""
        class Rewrite(ast.NodeTransformer):
            n = 0

            def visit_Attribute(self, node: ast.Attribute):
                self.generic_visit(node)
                v = node.value
                if (node.attr in DTYPE_HELPERS and isinstance(v, ast.Attribute) and v.attr == "types"
                        and isinstance(v.value, ast.Attribute) and v.value.attr == "api"
                        and isinstance(v.value.value, ast.Name) and v.value.value.id in ("pd", "pandas")):
                    self.n += 1
                    return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
                return node

        r = Rewrite()
        tree = r.visit(tree)
        return ast.fix_missing_locations(tree), r.n

    def _check_table_names(self, tree: ast.AST) -> int:
        """`tables['x']` with a name that is not a table: a spelling that only differs in case / diacritics, or the README's
        name for it, is fixed in place (returns how many, so the caller re-renders the code: before, a fix made on the
        tree alone never reached the sandbox); any other name is refused with the real names (56 failed blocks on dev300
        a1 guessed table names)."""
        if not self.tables:
            return 0
        fixed = 0
        by_fold = {}
        for t in self.tables:
            by_fold.setdefault(fold(t), []).append(t)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "tables"
                    and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)
                    and node.slice.value not in self.tables):
                name = node.slice.value
                same = by_fold.get(fold(name), [])
                if len(same) == 1:
                    node.slice.value, fixed = same[0], fixed + 1
                    continue
                real = resolve_table(name, self.tables)   # the README's name: `DanhSachNhapHang` for `DuLieuBanHang__DanhSachNhapHang`
                if real is not None:
                    node.slice.value, fixed = real, fixed + 1
                    continue
                close = difflib.get_close_matches(name, list(self.tables), n=3, cutoff=0.3)
                raise ValueError(f"There is no table {name!r}. The tables are exactly: {list(self.tables)}; closest: {close}.")
        return fixed

    @staticmethod
    def _is_final_call(n: ast.AST) -> bool:
        return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "final_answer"

    @staticmethod
    def _msg() -> str:
        return (
            "`answer` must be COMPUTED from the data, not a pasted number/string (also not via a constant variable such as "
            "`x = round(123.0)`): you do not know the result before the code runs. Assign `answer = <expression using "
            "tables[...] / sales_orders / item_lines>`; if you already SAW the value in an earlier step's output, send a block "
            "that contains only `answer = <the value you saw>`."
        )

    def _drop_period_redefinitions(self, tree: ast.Module) -> tuple[ast.Module, int]:
        """Remove top-level `period_start = ...` style statements for periods the harness resolved.

        Measured on dev300: the coder re-defined them 135 times, almost always to the same dates at midnight. Refusing
        cost a step each time and was the main reason runs ran out of steps; silently keeping the resolved values is
        both correct and free.
        """
        periods = {v for v in PERIOD_VARS if v in self._protected}
        if not periods:
            return tree, 0
        keep, n = [], 0
        for st in tree.body:
            if isinstance(st, ast.Assign) and all(isinstance(t, ast.Name) and t.id in periods for t in st.targets):
                n += 1
                continue
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Tuple) \
                    and all(isinstance(e, ast.Name) and e.id in periods for e in st.targets[0].elts):
                n += 1
                continue
            keep.append(st)
        tree.body = keep or [ast.Pass()]
        return tree, n

    @staticmethod
    def _drop_self_assignments(tree: ast.Module) -> tuple[ast.Module, int]:
        """`answer = answer` (the usual closing line) does nothing: remove it instead of judging it."""
        keep = [st for st in tree.body if not (isinstance(st, ast.Assign) and isinstance(st.value, ast.Name)
                and all(isinstance(t, ast.Name) and t.id == st.value.id for t in st.targets))]
        n = len(tree.body) - len(keep)
        if n:
            tree.body = keep or [ast.Pass()]
        return tree, n

    def _drop_literal_after_computed_answer(self, tree: ast.Module) -> tuple[ast.Module, int]:
        """`answer = revenue` then `answer = 230034300` in one block: keep the computed one, drop the pasted literal
        (seen 69 times on the public test; refusing the whole block cost a step each time)."""
        computed, keep, n = False, [], 0
        for st in tree.body:
            is_ans = isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "answer" for t in st.targets)
            if is_ans and not is_const_expr(st.value, set()):
                computed = True
            elif is_ans and computed and not _is_unanswerable_const(st.value):
                n += 1
                continue
            keep.append(st)
        tree.body = keep or [ast.Pass()]
        return tree, n

    def _was_printed(self, node: ast.AST) -> bool:
        """A literal answer the code already printed in an earlier step was read, not guessed."""
        try:
            v = ast.literal_eval(node)
        except (ValueError, SyntaxError, TypeError):
            return False
        vals = v if isinstance(v, (list, tuple)) else [v]
        if not vals or not self._seen_text:
            return False
        seen = self._seen_text

        def shown(x) -> bool:
            if isinstance(x, bool) or x is None:
                return str(x) in seen
            if isinstance(x, (int, float)):
                forms = {str(x), f"{x:,}", f"{float(x):.2f}", str(float(x))}
                if float(x).is_integer():
                    forms |= {str(int(x)), f"{int(x):,}", f"{int(x)}.0"}
                return any(f in seen for f in forms)
            return str(x) in seen

        return all(shown(x) for x in vals)

    def explain_error(self, msg: str, code: str) -> str:
        """smolagents reports `Could not index <whole DataFrame>` and hides the cause: add the facts needed to fix it."""
        if re.search(r"Invalid comparison|Cannot compare", msg):
            return msg + ("\nHINT: date columns are datetime64: compare with pd.Timestamp(...) (or period_start / period_end), "
                          "not datetime.date or a string.")
        m = re.search(r"The variable `(\w+)` is not defined", msg)
        if m and m.group(1) in ("unicodedata", "re", "math", "datetime", "collections", "itertools", "statistics", "numpy", "np", "pd"):
            return msg + f"\nHINT: add `import {m.group(1)}` at the top of the block (every block must import for itself)."
        if self.tables is None or not re.search(r"Could not index|KeyError|not in index", msg):
            return msg
        named = [t for t in self.tables if f"'{t}'" in code or f'"{t}"' in code]
        hints = [f"table `{t}` has columns: {list(self.tables[t].columns)}" for t in named[:3]]
        wrong = [w for w in re.findall(r"tables\[['\"]([^'\"]+)['\"]\]", code) if w not in self.tables]
        if wrong:
            hints.append(f"no table {wrong}; tables = {list(self.tables)} (sales_orders and item_lines are also available as variables)")
        # holdout6 r6: the coder read `shift` from the raw orders table; only `sales_orders` has it (and the §1 revenue)
        raw_cols = {c for t in named for c in self.tables[t].columns}
        for name, df in self.frames.items():
            missing = [c for c in dict.fromkeys(re.findall(r"\[['\"]([^'\"]+)['\"]\]", code))
                       if c in df.columns and c not in raw_cols and c not in self.tables]
            if missing and named:
                hints.append(f"column {missing} exists only in the variable `{name}` (not in tables[...]): use `{name}` "
                             f"(columns: {list(df.columns)[:40]})")
        return msg + ("\nHINT:\n- " + "\n- ".join(hints) if hints else "")

    STR_FILTERS = {"contains", "isin", "startswith", "endswith", "match", "fullmatch", "eq", "ne"}

    def _check_raw_status_filter(self, tree: ast.AST, code: str) -> None:
        """Refuse (once) code that filters a raw status column by typed text when `status_std` exists.

        Raw labels are spelled many ways ("hủy" vs "Huỷ đơn" differ in where the accent sits): seen 24 cancelled orders
        instead of 28. `status_std` already maps every label. Refused only once: a second attempt runs as written."""
        if self.status_bounced or not self.tables or "status_std" in code or "valid_order" in code:
            return
        status = {}
        for t, df in self.tables.items():
            src = getattr(df, "attrs", {}).get("status_source")
            if src and "status_std" in df.columns:
                status[src] = (t, df.attrs.get("status_map", {}))

        def raw_col(n: ast.AST) -> str | None:
            for x in ast.walk(n):
                if isinstance(x, ast.Subscript) and isinstance(x.slice, ast.Constant) and x.slice.value in status:
                    return x.slice.value
            return None

        def has_str(nodes) -> bool:
            return any(isinstance(x, ast.Constant) and isinstance(x.value, str) for n in nodes for x in ast.walk(n))

        for node in ast.walk(tree):
            col = None
            if isinstance(node, ast.Compare):
                ops = [node.left, *node.comparators]
                if has_str(ops):
                    col = raw_col(node)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in self.STR_FILTERS:
                if has_str([*node.args, *[k.value for k in node.keywords]]):
                    col = raw_col(node.func.value)
            if col:
                self.status_bounced = True
                t, mp = status[col]
                by: dict[str, list[str]] = {}
                for raw, cls in mp.items():
                    by.setdefault(cls, []).append(raw)
                mapping = "; ".join(f"{cls} ← {', '.join(map(repr, raws))}" for cls, raws in by.items())
                raise ValueError(
                    f"Do not filter the raw status column `{col}` by typed text (labels are spelled many ways and rows get missed). "
                    f"Use the `status_std` column of table `{t}` (completed/shipping/cancelled/returned) or `valid_order`. "
                    f"Mapping: {mapping}. If the question separates labels inside one class (e.g. a return reason), filter "
                    f"`{col}`.isin([...]) with EXACTLY the labels above. "
                    "Only count what the question asks: 'huỷ' = cancelled only, 'trả hàng' = returned only."
                )

    ITEM_BASE = {"order", "date", "valid", "item", "key", "code", "qty", "qty_base", "unit", "base_unit", "amount", "unit_price", "is_item"}

    def _merge_involves_sales(self, tree: ast.AST) -> bool:
        """Is one side of a `.merge(...)` made from the sales lines (`item_lines`, `sales_orders`, or `tables[<the sales-lines
        table>]`)? The refusal below is about sales lines gaining product attributes. biz_109 q_00a6e23cbacf merged the
        PURCHASE lines with the purchase slips (the code also read 'Tên hàng' / 'ĐVT' from the product table) and lost a
        block (77 s). The side's origin is followed through plain assignments (`d = tables['x']`, `d2 = d[d['a'] == 1]`).
        When the sales table is unknown, assume yes (the old rule)."""
        if not self.sales_table:
            return True
        prov: dict[str, set] = {}

        def refs(e: ast.AST) -> set:
            out: set = set()
            for n in ast.walk(e):
                if isinstance(n, ast.Name):
                    out |= prov.get(n.id, {n.id})
                elif (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "tables"
                        and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)):
                    out.add(n.slice.value)
            return out

        for n in sorted((n for n in ast.walk(tree) if isinstance(n, ast.Assign)), key=lambda n: n.lineno):
            r = refs(n.value)
            for t in n.targets:
                if isinstance(t, ast.Name):
                    prov[t.id] = r
        sales = {self.sales_table, "item_lines", "sales_orders"}
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "merge"):
                continue
            if isinstance(n.func.value, ast.Name) and n.func.value.id in ("pd", "pandas"):      # pd.merge(a, b, ...)
                ops = n.args[:2] + [k.value for k in n.keywords if k.arg in ("left", "right")]
            else:                                                                               # a.merge(b, ...)
                ops = [n.func.value] + n.args[:1] + [k.value for k in n.keywords if k.arg == "right"]
            if any(refs(o) & sales for o in ops):
                return True
        return False

    def _check_item_attribute_merge(self, tree: ast.AST) -> None:
        """Refuse (once) a `merge` that brings product attributes (group, prescription flag...) into the sales lines when
        `item_lines` already has them, joined on the normalized name. dev300 57cf71 / 6e3896: the Coder merged by exact name
        and lost lines whose spelling differs (off by 1.7M and 5M). A second attempt runs as written."""
        lines = self.frames.get("item_lines")
        if self.merge_bounced or lines is None:
            return
        attrs = set(map(str, lines.columns)) - self.ITEM_BASE
        merges = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "merge" for n in ast.walk(tree))
        uses_lines = any(isinstance(n, ast.Name) and n.id == "item_lines" for n in ast.walk(tree)) or any(
            isinstance(n, ast.Constant) and n.value == "item_lines" for n in ast.walk(tree))
        hit = sorted({n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in attrs})
        if merges and hit and not uses_lines and self._merge_involves_sales(tree):
            self.merge_bounced = True
            raise ValueError(
                f"Do not merge the sales table with the product table yourself: names that differ in case, accents or spacing lose "
                f"their attributes. The variable `item_lines` already has the product columns {hit} joined on the normalized name; "
                "filter and sum `item_lines` (`amount`, `qty_base`) instead. If you really need this merge, send it again.")

    def _check_valid_as_cancelled(self, tree: ast.AST, code: str) -> None:
        """Warn (once) when `~x['valid']` / `x['valid'] == False` is used without `status_std`: `valid` is False for cancelled
        AND returned orders (77 instead of 60 on dev300 1b04aa)."""
        if self.valid_bounced or "status_std" in code or self.frames.get("sales_orders") is None:
            return

        def is_valid_ref(n: ast.AST) -> bool:
            return (isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value == "valid") or (
                isinstance(n, ast.Attribute) and n.attr == "valid")
        for n in ast.walk(tree):
            neg = isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Invert) and is_valid_ref(n.operand)
            eq = (isinstance(n, ast.Compare) and is_valid_ref(n.left) and len(n.ops) == 1 and isinstance(n.ops[0], ast.Eq)
                  and isinstance(n.comparators[0], ast.Constant) and n.comparators[0].value is False)
            if neg or eq:
                self.valid_bounced = True
                raise ValueError(
                    "`valid` is False for cancelled AND returned orders. If the question asks for cancelled orders use "
                    "`status_std == 'cancelled'`, for returned `status_std == 'returned'`. If it really means cancelled or returned "
                    "together, send the same block again.")

    def _check_header_money_sum(self, tree: ast.AST, code: str) -> None:
        """Refuse (twice) code that adds up an order-header column that looks like revenue (order total after discount,
        ...; see engines/orders.py) when the rulebook revenue is at hand in `sales_orders`. Seen on holdout6: revenue by staff / channel / customer group /
        day off by 30k-2.6M because the header total is not §1's sum of line amounts. A second attempt runs as written:
        the column may be what the question asks (money paid, still owed)."""
        if self.money_bounced >= 2 or not self.order_money or "sales_orders" in code or "sales_metrics" in code:
            return
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("sum", "mean")):
                continue
            cols = [x.value for x in ast.walk(node.func.value) if isinstance(x, ast.Constant) and x.value in self.order_money]
            if cols:
                self.money_bounced += 1
                raise ValueError(
                    f"Column `{cols[0]}` is the money written on the ORDER, NOT revenue by rulebook §1 (sum of line amounts, "
                    "cancelled/returned orders excluded, delivery fee excluded). Revenue / average order (also by staff, channel, "
                    "customer group, payment method, day, shift): use `sales_orders[sales_orders['valid'] & <condition>]['revenue']` "
                    "(every column of the order table is in `sales_orders`) or `sales_metrics(start, end)`.")

    def _check_protected(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            elif isinstance(node, (ast.For, ast.comprehension)):
                targets = [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name) and n.id in self._protected:
                        raise ValueError(
                            f"Do not re-assign `{n.id}`: it is real data / a resolved period that is already provided. Do not make up "
                            "sample data or reset the period boundaries; use the provided variable directly (pick another variable name if you need one)."
                        )

    def __getattr__(self, name):
        return getattr(self._inner, name)
