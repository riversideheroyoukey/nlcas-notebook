"""The guards that keep ``subst``/``solve`` honest about P2 bindings.

Two of them live here, for one reason: both answer "is what the CAS holds for
this name going to interfere with what the researcher asked?".
:func:`lhs_may_match` refuses a substitution that cannot match;
:func:`shadow_own_definition` removes the binding that would make an operation
on a definition vacuous.  Both are pure functions of NOTEBOOK state — neither
queries the CAS — so both are decidable line by line as input is processed.

Can this substitution do anything at all?  (docs/32 Phase 1a)

Maxima's ``subst()`` on an ATOMIC lhs is purely SYNTACTIC: it rewrites a
subexpression only where that subexpression appears verbatim in the internal
form.  So an lhs that occurs in no spelling in the source cannot match — and
the CAS does not report that.  It returns the expression unchanged, the engine
records ``success``, and the researcher carries forward a value that was never
substituted (open_items AUDIT-SUBST-INDEXED: ``subst(V(i)=5, a*V_i + b)``).

:func:`lhs_may_match` is the predicate that lets the builder refuse instead.
It is deliberately CONSERVATIVE — it answers "may match" whenever the source
text is not a reliable witness — because a false alarm would block a
substitution that works, which is worse than missing one that does not.

Two situations make the text unreliable, and both are exempted:

* **CAS-bound symbols.**  H-1 P2 sends ``H : J*x*y`` to Maxima, so ``subst(x=2,
  H + K)`` succeeds even though ``x`` is absent from the source text — Maxima
  expands ``H`` before ``subst()`` runs.
* **Algebraic matching.**  A COMPOUND lhs goes through ``ratsubst`` (H-10),
  whose match is algebraic rather than syntactic, so a sign-flipped or
  reordered lhs legitimately matches without appearing verbatim.  That case
  never reaches this predicate — the caller applies it only to an atomic lhs.

Split out of ``maxima_backend_handler`` to keep that file under its hygiene
ceiling, the same way ``maxima_derivative_latex`` was (tests/test_code_hygiene).
"""
from __future__ import annotations

import re

from notebook.runtime.expr_ref_fields import is_expr_ref
from notebook.runtime.shared_parser import split_relation


def _squash(s: str) -> str:
    """Drop all whitespace: the source may carry the spacing that
    ``ensure_explicit_mult`` introduces (``a * V(i) + b``) while the generated
    lhs variant does not."""
    return re.sub(r"\s+", "", s)


def lhs_may_match(src: str, lhs_variants: "list[str]", ctx,
                  shadowed: "set[str] | None" = None) -> bool:
    """True when substituting one of *lhs_variants* into *src* can do something.

    *ctx* is the session :class:`SymbolContext`; a quantity carrying a
    ``defining_expr_id`` is bound in the CAS (see the module docstring).

    *shadowed* names the symbols the caller wraps in ``block([...], ...)``.  The
    CAS-bound exemption above rests on "Maxima expands this name before subst()
    runs" — and a shadow is exactly what makes that false, so a shadowed name is
    not a witness.  Measured: the source is now passed WHOLE (`H = c_1 + y`
    rather than its right side), so the equation's own left symbol appears in the
    text and, being H-1 P2-bound, exempted every substitution into any named
    equation — `\\sum_{i=1}^3 c_i` "substituted" into `H` completed with nothing
    changed, which is the silent success this guard exists to refuse.

    The caller passes EVERY name it shadows, not only the one that exposed this.
    Measured over the full suite with a probe logging where the two rules
    disagree: the narrow set (the newly-shadowed source LHS alone) and the full
    one never once differ.  So the choice is about keeping ONE rule rather than
    about a behaviour the tests can see — silence here is not evidence that the
    narrow rule is safe, only that no test input reaches the difference.

    Membership needs no "is this an identifier?" test: the names compared against
    it are ``ctx.registry`` keys, which are symbol names by construction, so a
    caller's non-identifier (`'diff(q,t,2)` stripped of its parentheses) can
    never match one.
    """
    squashed_src = _squash(src)
    if any(_squash(v) in squashed_src for v in lhs_variants if v):
        return True
    for name, prop in ctx.registry.items():
        if shadowed and name in shadowed:
            continue
        if not getattr(prop.quantity, "defining_expr_id", None):
            continue
        if re.search(r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])',
                     src):
            return True
    return False


def shadow_own_definition(ctx, eq_ref: str, eq_str: str, command: str) -> str:
    r"""Wrap *command* in ``block([S], …)`` when the equation being solved IS
    the H-1 expression that DEFINES its own left-hand symbol.

    P2 binds an H-1 definition CAS-side (`S : 14*x`), which is what makes a
    later `simplify(S)` expand — and which makes solving the DEFINITION for a
    variable on its right vacuous: `solve(S = 14*x, x)` answers `all`, since
    both sides are the same object.  Measured 2026-08-14; before the
    lowering work the bind happened to FAIL on such inputs, so this was
    invisible (open_items P2-BIND-NOT-LOWERED).

    A local `block` binding shadows the global one for this call only: no
    `kill`, no restore step, nothing to leak if the call raises — unlike the
    kill/restore the P2 bind itself uses.  Measured:
    `block([S], solve(S = 14*x, x))` → `[x = S/14]`, and `S` is still `14*x`
    afterwards, so P2 keeps its value.

    NARROW ON PURPOSE.  An equation that merely MENTIONS a defined symbol must
    still expand it — shadowing there answers `[]` (measured).  The two are
    told apart by asking whether THIS expression is the one that defined the
    symbol (`Quantity.defining_expr_id`), not by the shape of the left side.

    DECIDABLE LINE BY LINE, and deliberately so:

    * it is a pure function of NOTEBOOK state (`defining_expr_id` plus the
      equation text) and never queries the CAS.  In particular it does not
      need to know whether the P2 bind actually TOOK: shadowing a symbol that
      was never bound is a no-op (measured on an integral definition, whose
      bind fails because Maxima wants sign(s)).  Had the answer depended on
      CAS state, the decision would not have been reproducible.
    * no lookahead — it reads only what earlier lines established.

    One ORDER DEPENDENCE is real and inherited from H-1/P2, not introduced
    here: redefining the symbol moves `defining_expr_id`, so re-solving the
    OLD definition afterwards is no longer shadowed (measured).  That is the
    right answer — after `S = y` the CAS's `S` IS `y`, so `S = 14*x` solved
    for `x` should give `y/14`, not `S/14`, which would name a symbol whose
    meaning has since changed.
    """
    if not (ctx and is_expr_ref(eq_ref.strip())):
        return command
    rel = split_relation(eq_str)
    if rel is None or "=" not in rel[1]:
        return command
    name = rel[0].strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return command
    qty = ctx.get_quantity(name)
    if qty is None or getattr(qty, "defining_expr_id", None) != eq_ref.strip():
        return command
    return f"block([{name}], {command})"


def shadow_coordinate_variables(coord_vars, command: str) -> str:
    r"""Wrap *command* in ``block([x, y, z], …)`` so the coordinate variables the
    ∇ operators bind are read as COORDINATES, whatever the notebook has bound
    those names to.

    ``nb_curl_x(Ax,Ay,Az) := diff(Az,y) - diff(Ay,z)`` spells the coordinate
    variables INSIDE the function, so H-1's P2 binding reaches them: after
    ``y = x^3`` the call means "differentiate by ``x^3``", Maxima refuses, and
    what the researcher gets is a raw ``Maxima evaluation error`` — the CAS's
    own words for a question they never asked (Core Principle #1, open_items
    COORD-NAME-BOUND).  Measured 2026-09-17: ``\nabla \times \vec{A}``,
    ``\vec{A} の回転を求めて`` and ``the curl of \vec{A}`` all fell, in both
    languages, and so did divergence, gradient and the scalar Laplacian.

    THE RESEARCHER NEVER WROTE THE COORDINATE.  ``∇×`` binds x, y and z the way
    ``∑`` binds its index and ``∫`` its measure — the name comes from the
    session's coordinate system, not from the sentence — so a global binding on
    it is a name COLLISION, not a value the operator should use.  Shadowing is
    what says that, and it is Maxima's own mechanism for it.

    REFUSING THE DEFINITION INSTEAD WAS MEASURED AND REJECTED.  Of 1994 scenario
    input lines, 4 define a coordinate name, and they are ordinary writing: an
    LP objective ``z = c_1 x + c_2 y`` and ``y = x^3``.  "Write it down first,
    refuse at the point of use" — and here there is nothing to refuse at the
    point of use either, because the curl of a field is the curl of that field.

    UNCONDITIONAL, following `_build_solve_system`, which shadows its solve-for
    variables for this same P2 binding and records the reason: shadowing a name
    that is NOT bound is a no-op, so no discriminator is needed.  Measured here
    too — with nothing bound, ``[nb_curl_x(A_x,A_y,A_z),…]`` and the wrapped
    form both answer ``[x,-y,0]`` for ``A = (0,0,xy)`` — and the notebook-state
    predicate that a discriminator would have to use (`defining_expr_id`) is not
    equivalent to "the CAS holds a value": a solve result carries one and never
    binds (measured, and already recorded in `shadow_own_definition`).

    ``depends()`` SURVIVES IT, which is what makes this safe for a field that
    was declared rather than given components: with ``depends(A_z,[x,y,z])`` the
    wrapped and unwrapped calls both answer ``'diff(A_z,y,1)-'diff(A_y,z,1)``,
    byte for byte (measured).

    DO NOT APPLY THIS TO A VARIABLE THE RESEARCHER WROTE.  ``diff(f, r)`` after
    ``r = \sqrt{x^2+y^2+z^2}`` fails the same way, and shadowing it answers
    **0** where the answer is ``-1/r^2`` — because P2 evaluates on assignment,
    so ``f = 1/r`` was stored as ``1/sqrt(x^2+y^2+z^2)`` and the symbol ``r`` is
    no longer in it (measured).  That family trades a loud error for a silent
    wrong and needs a translated refusal instead, not a shadow (open_items
    DEFINED-NAME-AS-VARIABLE).

    NO EMPTY-LIST BRANCH, and that is measured rather than overlooked: the names
    come from a coordinate system, which always has three, and ``block([], …)``
    is valid Maxima and a no-op anyway (measured).  A guard written for that case
    survived mutation because nothing can reach it — so it was deleted rather
    than pinned.
    """
    names = [str(v).strip() for v in coord_vars if str(v).strip()]
    return f"block([{', '.join(names)}], {command})"


#: The coordinate-system functions that DIFFERENTIATE, and so bind x, y, z.
#: `nb_dot` / `nb_cross` are coordinate-free and are deliberately absent.
_COORDINATE_FUNCTION_CALL = re.compile(r"\bnb_(?:curl_\w+|grad|div|laplacian_scalar)\(")


def shadow_differentiated_operand(coord_vars, operand: str) -> str:
    r"""Shadow the coordinates for ONE operand of a vector product, and only when
    that operand was built by a ∇ operation.

    Reading `\nabla \times \vec{A}` as an operand of a dot or cross product put
    `nb_curl_*` inside `nb_dot(...)` — a NEW exit for the COORD-NAME-BOUND defect,
    which had been counted as having one exit only because this path did not
    exist yet.  After `y = x^3` the product answered a raw
    `Maxima evaluation error` again.

    **The wrap is per OPERAND, never the whole command.**  The product itself
    binds nothing: in `(∇×A)·(x, y, z)` the second operand's `y` is the
    researcher's, and it must see their binding.  Measured with `A = (0,0,xy)`
    and `y : x^3` — wrapping the ∇ operand alone gives `x^2 - x^3*y`, wrapping the
    whole `nb_dot` gives `x^2 - y^2`, a different value.  With nothing bound both
    agree (`x - y` for the operand `[1,1,1]`), which is why no discriminator is
    needed beyond "was this operand differentiated?" (`shadow_coordinate_variables`).
    """
    if not _COORDINATE_FUNCTION_CALL.search(operand):
        return operand
    return shadow_coordinate_variables(coord_vars, operand)

# `shadow_coordinate_variables` has a THIRD family of callers since 2026-09-17: the
# DECLARATIONS that name their variables of dependence — `depends([B_x,…],[x,y,z])`
# for a vector or scalar field, `depends([q],[t])` for a time-dependent quantity.
# Found by the regression scenario's replay, not by the ticket: `y = x^3` then
# `B をベクトル場とする` answered a raw `Maxima evaluation error` in both languages,
# and `t = 2` then `q を時間依存とする` was WORSE — the CAS call failed while the
# notebook answered "declared", `confirmed`, so every later d/dt of `q` would see no
# dependence.  A dependence declared inside the block PERSISTS outside it
# (`block([x, y, z], depends(C_x,[x,y,z]))` leaves `dependencies` = `[C_x(x,y,z)]`,
# measured), and a saved session is unaffected either way: declarations are
# replayed BEFORE the P2 bindings on load (`NotebookEngine.start`).
