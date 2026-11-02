"""Rule-based interceptor registry (R1, docs/27).

The pre-LLM interpretation pipeline fires a long sequence of ``_try_*``
interceptors; historically each was a hand-written ``x = await _try_...();
if x is not None: return x`` block inside a 998-line function, so the firing
ORDER lived in code position and a new NL feature meant editing that giant
function.  This module makes the order DATA: an ordered list of
``Interceptor`` entries, run by a single loop that short-circuits on the first
non-None result.  A new interceptor is then "write a function + add one list
entry", never "edit the dispatcher".

Interceptors keep their existing ``(engine, loop, lang, text, registered, …)``
signatures; the registry adapts them via a small ``run`` callable per entry, so
migrating a batch touches only the call site (the functions themselves are
unchanged).  ``InterceptorContext`` carries everything any interceptor might
need; ``text`` is mutable so a rewriter (e.g. the notation-candidate resolver)
can narrow it for the interceptors that follow.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from notebook.engine_history_mixin import label_names_internal_scaffold


@dataclass
class InterceptorContext:
    engine: Any
    loop: Any
    lang: str
    text: str                       # the flowing raw text; a rewriter MAY reassign it
    #: The same request BEFORE 2a-sym rewrote the researcher's symbols into
    #: E-IDs for the LLM's benefit.  Read it wherever a rule needs a NAME the
    #: researcher wrote rather than a reference to an expression — the two have
    #: opposite requirements and one string cannot serve both (measured
    #: 2026-08-16: `solve E1 and E2 and E3 for a, b, c` sent
    #: `solve([…], [a, b, E3])` because `c` was also an equation's left side).
    #: Empty means "same as text".
    registered: list
    alias_free_text: str = ""
    tokens: list = field(default_factory=list)   # req_tokens (notation candidates)
    supervisor: Any = None                        # LLM supervisor (change_vars)
    req: Any = None                               # InterpretRequest (rule short-circuits)
    rule_order: Any = None                        # differentiation order pre-extracted at 2e
    interactive: bool = True                      # whether to offer confirmation dialogs
    route: Optional[str] = None                   # set to the interceptor that WON


@dataclass(frozen=True)
class Interceptor:
    """One registered pre-LLM interceptor.

    ``run`` receives the context and returns the interpreter response dict to
    short-circuit on, or None to fall through.  ``note`` records the
    ordering rationale that used to live in an inline comment — so the
    *reason* an interceptor sits where it does is data, not lore.
    """
    name: str
    run: Callable[[InterceptorContext], Awaitable[Optional[dict]]]
    note: str = ""


def _attribute_new_steps(engine: Any, before: int, name: str) -> None:
    """Stamp every step *engine* gained since *before* with the interceptor *name*.

    Done here, in the one function all five registries go through, rather than
    in the 24 interceptors — an attribution each of them had to remember to
    write would be wrong the first time one forgot.  Steps are only appended,
    so the tail of the (insertion-ordered) dict is exactly what this call made.

    Interceptors that fall THROUGH are stamped too: an interceptor may record a
    step and still return None, and that step is no less its doing.
    """
    steps = getattr(engine, "steps", None)
    if not isinstance(steps, dict) or len(steps) <= before:
        return
    for step_id in list(steps)[before:]:
        step = steps[step_id]
        if getattr(step, "route", None) is None:
            step.route = name


def _label_the_result_card(engine, resp: dict, before: int) -> None:
    """Give an operation's response the label of the step it produced.

    The timeline draws a card for every ``registered_expressions`` entry and
    then, only if none of them already showed ``expr_id``, one for ``data``
    itself (``notebook_main.js``).  The label it renders comes from whichever
    object it drew — so an interceptor that appends its own result and does not
    put ``operation_label`` on that entry produces an UNLABELLED card, however
    carefully it filled the response.

    Measured 2026-08-27: 28 of the 189 corpus lines that reach an operation drew
    no labelled card at all, across eight different routes — `simplify` via
    both `evaluate_rhs` and `both_sides`, `build_inverse_transform`,
    `taylor_expand`, `expand_neighbor_sum`, `fork_scalar_handoff`,
    `solve_ode_general` and the two `pair_bracket` brackets.  Each had answered
    the same question ("what does an operation return?") for itself, and eight
    of them had answered it incompletely.

    Answering it HERE is the same move the route attribution above already
    made: every interceptor's response passes through this function, so the
    label is filled once instead of in each handler.

    **The step comes from what the interceptor just DID, not from the
    response.**  Reading ``resp["step_id"]`` was the first attempt and it fixed
    only 3 of the 28: most of these handlers never put a ``step_id`` in their
    response at all (measured — ``evaluate_rhs`` and ``both_sides`` return
    ``step_id: None``).  ``_attribute_new_steps`` above already identifies the
    steps this interceptor added, for exactly the same reason, so the label is
    read from the last of them that names an operation.

    Nothing is invented: a response that names no operation, or an interceptor
    that added no operation step, is left alone (a registration draws the
    expression itself; "register" is not an operation label).
    """
    if not resp.get("operation") or resp.get("status") != "completed":
        return
    label = resp.get("operation_label")
    if not label:
        steps = getattr(engine, "steps", None)
        if not isinstance(steps, dict):
            return
        for step_id in reversed(list(steps)[before:]):
            step = steps[step_id]
            if getattr(step, "operation", None) and getattr(step, "visible_action", None):
                label = step.visible_action
                break
        if not label or label_names_internal_scaffold(label):
            # A label naming a key the researcher has never seen is worse than
            # none — it asks them what `_T1` is.  H-8's both-sides path registers
            # a scaffold and simplifies it, so its operation step reads
            # `_T1 を簡約する`; measured 2026-08-27, that spelling is already in
            # 40 published labels (open_items SCAFFOLD-LABEL).  Filling it onto a
            # card would put it on screen too, so this stops here and the three
            # `both_sides` lines stay in the ratchet until that is fixed.
            return
        resp["operation_label"] = label
    # The result may have been appended to `registered_expressions`, in which
    # case the UI draws THAT entry and skips `data` — so the label has to be on
    # it.  Only the entry that IS the result: the others are further answers to
    # the same operation (solve returns every root), not further operations.
    for entry in (resp.get("registered_expressions") or []):
        if entry.get("expr_id") == resp.get("expr_id"):
            entry.setdefault("operation_label", label)


def _carry_result_warnings(engine, resp: dict) -> None:
    """操作の結果に付いた警告を、応答（と描かれるカード）まで運ぶ。

    CLAUDE.md の規則は「警告は画面に配線するか、足さないか」。数えると
    （2026-09-05）**操作の結果を返す応答は18箇所あり、`warnings` を渡していたのは
    1箇所だけ**だった——付けた警告はほぼどこにも描かれない。

    答えを18箇所に配るのではなく、`operation_label` と同じ場所に1つ置く
    （七問目と同じ形、同じ理由: 全応答がここを通る）。
    """
    expr = getattr(engine, "expressions", {}).get(resp.get("expr_id"))
    warns = list(getattr(expr, "warnings", None) or [])
    if not warns:
        return
    existing = list(resp.get("warnings") or [])
    resp["warnings"] = existing + [w for w in warns if w not in existing]
    for entry in (resp.get("registered_expressions") or []):
        if entry.get("expr_id") == resp.get("expr_id"):
            entry.setdefault("warnings", resp["warnings"])


async def run_interceptors(ctx: InterceptorContext,
                           interceptors: "list[Interceptor]") -> "dict | None":
    """Fire each interceptor in list order; the FIRST non-None result wins and
    is returned.  Order is the list, not code position.

    The winner's name is recorded — on ``ctx.route``, on the response dict, and
    on every step the interceptor produced — so that WHICH route claimed an
    input is answerable afterwards.  The firing order was already data here;
    the attribution was not, and that asymmetry is what let a defect be filed
    against a path nobody takes (handoff §2.2).
    """
    steps = getattr(ctx.engine, "steps", None)
    for it in interceptors:
        before = len(steps) if isinstance(steps, dict) else 0
        result = await it.run(ctx)
        _attribute_new_steps(ctx.engine, before, it.name)
        if result is not None:
            ctx.route = it.name
            if isinstance(result, dict):
                result.setdefault("route", it.name)
                _label_the_result_card(ctx.engine, result, before)
                # ラベルと警告は**独立**——ラベルが決められないとき
                # （足場名しか無いなど）に警告まで落とさないよう、別に呼ぶ。
                _carry_result_warnings(ctx.engine, result)
            return result
    return None
