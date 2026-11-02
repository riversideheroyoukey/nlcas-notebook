r"""A sum written WITHOUT a range — giving it one the CAS can hold.

`\sum_i x_i` is an ordinary thing to write, and QuantumAlgebra takes it as-is
(`∑(:i, …)` binds an index and asks for nothing else).  Maxima's `sum` needs
four arguments, so such a sum could not be handed to it at all: the generator
could only produce the two-argument form, which Maxima rejects outright
(open_items B-10 ②).

The answer is a PLACEHOLDER range, and the whole design is about who shares one:

* **Never two expressions.**  Measured 2026-08-08 with the pair allocated per
  COMMAND instead: a separately registered `\sum_i x_i` minus another came back
  as **0** — Maxima identifies sums whose ranges are spelled the same, so a
  shared placeholder asserts an equality nobody told us.  The pair is therefore
  allocated ONCE, at registration, and travels in the IR (`IndexSum.
  placeholder_bounds`).

  The IR is NOT persisted (`Expression.ir_ast` is `exclude=True`), so a reloaded
  session has to REBUILD it from the canonical — and the rebuild has to arrive
  at the same pair, or a result the CAS already returned stops matching its own
  source.  A forward-only counter cannot do that: it makes pairs unique, not
  reproducible.  `BoundLedger` below remembers them instead, keyed on the
  expression and the written index name (open_items B-10 ②-c).

* **Always within one expression.**  The same `i` written twice in one
  expression is ONE binding: `\sum_i x_i + \sum_i y_i` must not end up with two
  different ranges.  The allocation is keyed on the index NAME for that reason.

  Which is why `fill` runs BEFORE `alphatize_indices`, through the
  `before_alphatize` hook on `BaseSemanticResolver.resolve` (B-10 ②-a, closed
  2026-08-08).  Alphatisation renamed a dummy that collided with a name already
  seen, so the second `\sum_i` became `\sum_j` — and the allocation, running
  after it, saw two bindings where the researcher had written one and handed out
  two ranges.  Sharing a RANGE is not sharing a binding, and only the written
  name can say which sums share one; that ordering still stands.

  **The parallel rename itself was withdrawn on 2026-08-17**
  (EXPR-REF-PROVENANCE, cause A).  It was defended here as "not the defect: it
  IS alpha-conversion, and the two sums really are separate bindings" — true,
  and it does not follow that renaming is REQUIRED.  Alpha-conversion prevents
  CAPTURE, and a parallel binder is never in scope inside its sibling.  The
  rename was not free: Maxima's noun forms do not alpha-convert, so
  `\sum_i x_i - \sum_i x_i` did not reduce to 0 even though both terms carried
  the same placeholder pair.  Nested binders are still renamed.

The names carry their own meaning (`nbSumLo3` / `nbSumHi3`), so the display
strips them by a text rule and nothing has to be recorded beside the
expression.  They stay in the canonical: that is what the CAS returned, and
removing them there would send a two-argument sum next time.

The CAS view is DERIVED per call from the stored IR (`cas_view` below), never
persisted — the canonical is the researcher's own writing and carries no range,
which is the same rule `to_backend_code` follows.
"""
from __future__ import annotations

from notebook.models.ir import IndexSum
from notebook.runtime.constrained_sum import is_relational

#: Attributes an IR node may hold children in.  Walked generically so a node
#: type added later is not silently skipped.
_CHILD_LISTS = ("args", "indices", "dummy_indices")
_CHILD_NODES = ("body", "expr", "base", "exp", "func", "lhs", "rhs",
                "a", "b", "left", "right", "sub", "script")


def _walk(node, visit, seen=None) -> None:
    seen = set() if seen is None else seen
    if node is None or id(node) in seen:
        return
    seen.add(id(node))
    visit(node)
    for attr in _CHILD_LISTS:
        for child in (getattr(node, attr, None) or ()):
            _walk(child, visit, seen)
    for attr in _CHILD_NODES:
        child = getattr(node, attr, None)
        if child is not None and not isinstance(child, (str, int, float)):
            _walk(child, visit, seen)


class BoundLedger:
    """Who gets which placeholder pair — the one place that decides.

    Two pieces of state, both persisted with the session:

    * a COUNTER that only ever goes forward, so a pair handed out now is never
      one an existing expression already carries in its canonical;
    * a MAP from (expression, WRITTEN index name) to the pair it was given, so
      the same question asked a second time gets the same answer.

    The counter alone cannot do the second job, and the second job is not
    optional: `Expression.ir_ast` is `exclude=True`, so a reloaded session has
    to REBUILD the IR from the canonical, and a rebuild that invents new names
    disagrees with every result the CAS already returned (B-10 ②-c).
    """

    def __init__(self, counter: int = 0, pairs: "dict | None" = None):
        self.counter = int(counter)
        self.pairs: dict[str, dict[str, list[str]]] = pairs if pairs is not None else {}

    def alloc(self, expr_id: str, index_name: str) -> "tuple[str, str]":
        remembered = self.pairs.get(expr_id, {}).get(index_name)
        if remembered:
            return (remembered[0], remembered[1])
        self.counter += 1
        # No underscore in the name: Maxima's tex1 turns a trailing digit into a
        # subscript either way (`{\it nbSumLo}_{3}`), and an underscore would add
        # a second one, making the display harder to strip than it needs to be
        # (measured 2026-08-08).
        pair = (f"nbSumLo{self.counter}", f"nbSumHi{self.counter}")
        self.pairs.setdefault(expr_id, {})[index_name] = [pair[0], pair[1]]
        return pair

    def as_saved(self) -> dict:
        return {"sum_bound_counter": self.counter, "sum_bound_pairs": self.pairs}

    @classmethod
    def from_saved(cls, data: dict) -> "BoundLedger":
        """An older file has neither key and loads EMPTY, which is the safe
        direction: its sums are handed fresh pairs on first use, and a fresh
        pair is never one another sum already holds — so nothing is claimed
        equal that the researcher did not say was equal."""
        return cls(
            counter=int(data.get("sum_bound_counter", 0)),
            pairs={
                str(eid): {str(n): [str(lo), str(hi)] for n, (lo, hi) in p.items()}
                for eid, p in (data.get("sum_bound_pairs") or {}).items()
            },
        )


def fill(ir, alloc) -> None:
    """Give every range-less sum in *ir* a placeholder range, in place.

    *alloc* is the session's allocator: ``alloc(index_name) -> (lo, hi)``.  It
    is called once per index NAME, so two sums over the same written index share
    a range and two expressions never do.  A range the researcher DID write is
    left alone.
    """
    per_name: dict = {}

    def _pair(name: str) -> list:
        if name not in per_name:
            per_name[name] = list(alloc(name))
        return list(per_name[name])

    def visit(node) -> None:
        if not isinstance(node, IndexSum) or node.placeholder_bounds:
            return
        _pair_sum = (is_relational(node.constraint)
                     and len(node.dummy_indices) == 2)
        # A constrained pair sum needs BOTH ends of its one shared range, so a
        # HALF-written range still wants a placeholder for the end that was
        # left out.  `\sum_{i<j}^{N}` is the ordinary spelling and states only
        # the upper end; treated all-or-nothing (which is right for every other
        # sum, where a written range is written whole) it got no placeholder,
        # so the lower end was missing and the whole form stayed refused.
        if node.lower is not None and node.upper is not None:
            return
        if not _pair_sum and not (node.lower is None and node.upper is None):
            return
        if _pair_sum:
            # The two indices of `\sum_{i \ne j}` range over the SAME set — that
            # is what the notation says, and the rewrite in `constrained_sum`
            # is only valid for one shared range.  One pair each would have
            # made `Σ_{i≠j}` a sum over two unrelated ranges whose "diagonal"
            # `i = j` means nothing.
            node.placeholder_bounds = [_pair(node.dummy_indices[0].name)] * 2
        else:
            node.placeholder_bounds = [_pair(i.name)
                                       for i in node.dummy_indices]

    _walk(ir, visit)


def has_placeholder(ir) -> bool:
    """True if any sum in *ir* carries a placeholder range."""
    found = []

    def visit(node) -> None:
        if isinstance(node, IndexSum) and node.placeholder_bounds:
            found.append(True)

    _walk(ir, visit)
    return bool(found)


def cas_view_of(eid: str, expr, generate, rebuild=None) -> "str | None":
    """The CAS view of ONE expression, or None if it carries no placeholder.

    The single decision both callers make — the operation path, which swaps the
    whole `expr_lookup` before a command is built, and the H-2 expression-
    reference path, which splices one expression's text into another
    (`ui.interpret_common._expand_expr_ids`).

    They used to answer it differently, and the difference was visible in the
    same session (EXPR-REF-PROVENANCE cause B, measured 2026-08-17)::

        combine_expressions(E1, E2, "-")   →  0
        「E1 - E2」                          →  'sum(x(j),j,nbSumLo2,nbSumHi2)
                                                - 'sum(x(i),i,nbSumLo1,nbSumHi1)

    E2 is E1's own simplification and already carries E1's pair — a result
    inherits its source's range by construction.  What E1 could not do was
    carry its OWN: its canonical is the researcher's writing (`\\sum_i*x_i`,
    no range at all), so spliced as text it re-parsed inside the new
    expression and `BoundLedger.alloc` handed it a fresh pair.  The identity of
    the expression the text came from was simply dropped at the splice.

    Two expressions that merely both lack a range still get different pairs:
    the ledger is keyed per expression, so nothing here asserts an equality
    nobody stated.  What it stops asserting is an INequality between an
    expression and itself.

    A failure returns None: the canonical is still the researcher's writing,
    and the caller's own validation decides whether that is usable — the same
    stance `math_operand` takes.
    """
    from notebook.runtime.shared_parser import UnknownNotationError

    if expr is None:
        return None
    ir = expr.ir_ast
    try:
        if ir is None and rebuild is not None and "sum" in (expr.canonical or ""):
            ir = rebuild(eid, expr.canonical)
        if ir is None or not has_placeholder(ir):
            return None
        return generate(ir) or None
    except (UnknownNotationError, ValueError, AttributeError, TypeError):
        return None


def cas_view(expr_lookup: dict, expressions: dict, generate, rebuild=None) -> dict:
    """Replace the CAS view of every expression whose IR carries a placeholder.

    *generate* turns an IR into backend code.  *rebuild* ``(eid, canonical) ->
    ir`` is used when an expression has NO IR, which is what a reloaded session
    looks like (B-10 ②-c) — it is asked only for a canonical that could contain
    a sum at all, so an ordinary reloaded expression costs one substring test.
    """
    out = None
    for eid in list(expr_lookup):
        code = cas_view_of(eid, expressions.get(eid), generate, rebuild)
        if code:
            out = dict(expr_lookup) if out is None else out
            out[eid] = code
    return out if out is not None else expr_lookup
