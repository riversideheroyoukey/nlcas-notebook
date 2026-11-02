r"""A shadowed name is not a witness that the CAS will expand it.

`lhs_may_match` refuses a substitution whose lhs occurs nowhere in the source,
because Maxima answers that case with the source unchanged and the engine records
success.  It exempts CAS-BOUND names (H-1 P2 sends `H : c_1 + y`), on the
reasoning that Maxima expands `H` before `subst()` runs, so the text is not a
reliable witness.

`shadow_own_definition` and `_build_substitute`'s own `block([H], …)` wrap break
exactly that reasoning for the names they shadow: inside the block `H` is
unbound, so it expands to nothing.

The two only met once the substitute builder stopped cutting its source equation
down to the right-hand side (2026-08-14).  Passing the equation WHOLE puts its
own left symbol into the source text — and that symbol is P2-bound by
construction, since it is an H-1 definition.  Measured before the fix:

    \sum_{i=1}^{3} c_i = u a + v b
    H = c_1 + y
    「E1 を E2 に代入して」   → completed, canonical `y+c(1)`

The sum is not in `H` and cannot be — the refusal is the correct answer, and it
had become "success, nothing changed" for EVERY substitution into a named
equation, which is the whole class this guard exists to catch.

Pinned at the guard rather than only end-to-end: the end-to-end test that caught
this reads the interceptor's `=` split, so a change in routing would take the
coverage with it.
"""
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.maxima_subst_guard import lhs_may_match


def _ctx_with_definition(name: str, eid: str = "E2") -> SymbolContext:
    ctx = SymbolContext()
    ctx.register_quantity(name, kind="scalar", defining_expr_id=eid)
    return ctx


def test_a_bound_name_still_exempts_when_nothing_shadows_it():
    """The exemption itself is unchanged — this is what it is for."""
    ctx = _ctx_with_definition("H")
    assert lhs_may_match("H + K", ["x"], ctx) is True


def test_the_shadowed_name_does_not_exempt():
    ctx = _ctx_with_definition("H")
    assert lhs_may_match("H=c(1) + y", ["sum(c(i), i, 1, 3)"], ctx,
                         shadowed={"H"}) is False


def test_shadowing_one_name_leaves_the_others_witnesses():
    """Narrow: only the shadowed name loses its exemption."""
    ctx = _ctx_with_definition("H")
    ctx.register_quantity("K", kind="scalar", defining_expr_id="E3")
    assert lhs_may_match("H=K + y", ["sum(c(i), i, 1, 3)"], ctx,
                         shadowed={"H"}) is True


def test_a_present_lhs_is_unaffected_by_shadowing():
    """The first check (the lhs is literally there) answers before the loop."""
    ctx = _ctx_with_definition("H")
    assert lhs_may_match("H=c(1) + y", ["c(1)"], ctx, shadowed={"H"}) is True
