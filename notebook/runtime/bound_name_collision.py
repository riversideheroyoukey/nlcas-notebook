r"""A bound index that reuses a name the expression already uses for something
else.

``\sum_{i=n}^{I} \sum_{n>m}^{N} V`` binds ``n`` in the inner sum while ``n`` is
also the outer sum's lower LIMIT.  Both readings are legitimate writing —
a researcher may well have meant it — so this is **not** an error and nothing is
refused.  It is also not something the system can decide: whether the reuse is
deliberate or a slip is in the researcher's head, not in the text.

So it is reported, and that is all.  The one thing that must not happen is
silence, because the two occurrences resolve to different things (one is bound
by a Σ, one is not) and the expression computes without complaint either way.

Kept in its own module for the same reason as `subscript_policy`: this is a
place further discriminators will be added, and it must not grow inside the
resolver.
"""
from __future__ import annotations

from notebook.models.ir import Expr, IndexSum, IndexSymbol, Symbol


def _names_in(node, *, skip: "set[int] | None" = None) -> "set[str]":
    """Every Symbol / IndexSymbol name appearing under *node*.

    *skip* holds ``id()`` of subtrees to leave out — used to ask "does this name
    occur ANYWHERE ELSE", which is the whole question here.
    """
    found: set[str] = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur is None or (skip and id(cur) in skip):
            continue
        if isinstance(cur, (Symbol, IndexSymbol)):
            found.add(cur.name)
        if isinstance(cur, Expr):
            for field in type(cur).model_fields:
                if field == "simplification_log":
                    continue
                value = getattr(cur, field, None)
                if isinstance(value, Expr):
                    stack.append(value)
                elif isinstance(value, (list, tuple)):
                    stack.extend(v for v in value if isinstance(v, Expr))
    return found


def _iter_sums(node):
    stack = [node]
    while stack:
        cur = stack.pop()
        if not isinstance(cur, Expr):
            continue
        if isinstance(cur, IndexSum):
            yield cur
        for field in type(cur).model_fields:
            if field == "simplification_log":
                continue
            value = getattr(cur, field, None)
            if isinstance(value, Expr):
                stack.append(value)
            elif isinstance(value, (list, tuple)):
                stack.extend(v for v in value if isinstance(v, Expr))


def find_collisions(ir, symbol_context=None) -> "list[tuple[str, str]]":
    r"""Bound names that are also used for something else.

    Returns ``[(name, reason_key), …]`` — ``reason_key`` names WHICH other use
    was found, because the two are different mistakes:

    * ``bound_name.also_free`` — the name occurs OUTSIDE this sum as well.
      ``\sum_{i=n}^{I} \sum_{n>m}^{N} V``: `n` is the outer sum's lower limit
      and the inner sum's dummy.
    * ``bound_name.also_declared`` — a quantity of that name is declared (`r` a
      radius, then `\sum_r`).  Measured 2026-08-08 with GENERATED dummies,
      which is why generated names now always carry a subscript; a name the
      RESEARCHER writes is their own choice, and only reported.

    The comparison skips the SUM NODE, not just its body: a dummy naturally
    appears in `dummy_indices`, and skipping only the body left that occurrence
    in view — every name looked like a collision, so subtracting the dummies
    back out hid the real ones too (both wrong, measured 2026-08-10).

    Many potential collisions never reach here: the dummy-renaming step already
    moves a dummy off a name the expression uses (`n \sum_{n=1}^{N} x_n` binds
    `j`).  What survives is what that step cannot see.
    """
    if ir is None:
        return []
    declared = set()
    if symbol_context is not None:
        registry = getattr(symbol_context, "registry", None)
        if isinstance(registry, dict):
            declared = set(registry)

    out: "list[tuple[str, str]]" = []
    seen: set[str] = set()
    for s in _iter_sums(ir):
        outside = _names_in(ir, skip={id(s)})
        limits = _names_in(s.lower) | _names_in(s.upper)
        for idx in s.dummy_indices:
            name = getattr(idx, "name", None)
            if not name or name in seen:
                continue
            if name in outside or name in limits:
                out.append((name, "bound_name.also_free"))
                seen.add(name)
            elif name in declared:
                out.append((name, "bound_name.also_declared"))
                seen.add(name)
    return out
