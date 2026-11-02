r"""Opening a vector expression into its components — `k·x` → `k_x x_x + …`.

`\int \frac{d^3k}{(2\pi)^3} e^{i k\cdot x} \tilde f(k)` is how a Fourier
transform is written, and nothing about it can be handed to a one-variable CAS
as it stands.  The researcher's own answer (2026-08-08) is to open the dimension
explicitly — `\int d^3k → \int dk_x dk_y dk_z` — and that is what this module
does.  It is a GENERAL rewrite, not an integral feature: the measure is only one
of the places a vector has to become components.

**Opening the measure alone is a trap**, which is why all three happen together.
Measured: with the integrand left as written, Maxima reads `f(k)` as a constant
in `k_x` and returns `k_x k_y k_z · f(k)` — the same silent wrong answer the
unopened form gave, wearing a different face.  So a vector is opened wherever it
carries meaning:

* the volume measure — one integral per component, and the range the researcher
  wrote applies to EVERY one of them (that is what `\int d^3k` over a region
  means, unlike `\int_0^1 dx dy` where the range decorates one ∫ sign);
* a dot product — `k·x` → `k_x x_x + k_y x_y + k_z x_z`;
* a function OF the vector — `f(k)` → `f(k_x, k_y, k_z)`, without which the CAS
  would treat it as a constant under each component integral.

**A vector left standing after all that is refused**, by name.  It is the one
remaining way to reach the CAS with an unopened vector under a component
integral, and it would be silent.

The DIMENSION is the researcher's, taken from the measure they wrote (`d^2k` in
a two-dimensional problem opens two components, not three).  With no measure it
is the session's coordinate system.  Component NAMES follow the same convention
`define_vector_field` already uses — cartesian gives `k_x, k_y, k_z` — so the
names that appear are the ones the rest of the notebook would have produced.
"""
from __future__ import annotations

from notebook.models.ir import (
    Add, DotProduct, Expr, FunctionCall, Integral, Mul, Number, Symbol,
)

#: Attributes an IR node may hold children in.  Walked generically so a node
#: type added later is rebuilt rather than silently skipped.
_CHILD_LISTS = ("args", "indices", "dummy_indices")
_CHILD_NODES = ("body", "expr", "base", "exponent", "func", "lhs", "rhs",
                "a", "b", "left", "right", "sub", "script", "var",
                "lower", "upper", "x", "y")


class UnopenedVector(Exception):
    """A declared vector survived the rewrite, so the CAS must not see it."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


def _rewrite(node: Expr, fn) -> Expr:
    """Depth-first rebuild.  *fn* returns a replacement, or None to recurse.

    Rebuilt with ``model_copy`` so a field this module knows nothing about
    survives — the mistake that made ``measure_dim`` arrive as ``None`` and
    turned a volume integral into a one-variable one (measured 2026-08-08).
    """
    replaced = fn(node)
    if replaced is not None:
        return replaced
    updates = {}
    for attr in _CHILD_LISTS:
        values = getattr(node, attr, None)
        if values:
            updates[attr] = [_rewrite(c, fn) for c in values]
    for attr in _CHILD_NODES:
        child = getattr(node, attr, None)
        if child is not None and isinstance(child, Expr):
            updates[attr] = _rewrite(child, fn)
    return node.model_copy(update=updates) if updates else node


def is_vector(name: str, context, also: "set[str] | None" = None) -> bool:
    r"""Is *name* a vector here?

    *also* carries the vectors the EXPRESSION declares rather than the session:
    writing `\int d^3x` says `x` is a three-dimensional vector, and saying it in
    the measure is saying it.  Without this the measure opened while `f(x)` did
    not, and `\int d^3x f(x)` came back as ``f(x)·x_x·x_y·x_z`` — the exact
    silent wrong answer this module exists to prevent, reached by forgetting a
    declaration (measured 2026-08-08).  It is honoured for the rewrite only; the
    session's declarations are not edited behind the researcher's back.
    """
    if also and name in also:
        return True
    quantity = context.get_quantity(name) if context is not None else None
    if quantity is None:
        return False
    return getattr(quantity.math_type, "algebra", None) == "vector"


def component_names(base: str, dim: int, suffixes) -> list[str]:
    """`k`, 3, ("x","y","z") → ``["k_x", "k_y", "k_z"]``.

    Falls back to numeric suffixes when the coordinate system does not offer
    enough of them, so a dimension the researcher wrote is never silently
    reduced to the three a coordinate system happens to name.
    """
    if dim <= len(suffixes):
        return [f"{base}_{s}" for s in suffixes[:dim]]
    return [f"{base}_{i}" for i in range(1, dim + 1)]


def expand(ir: Expr, context, suffixes, default_dim: int = 3,
           infinity: str = "inf") -> "tuple[Expr, list[str]]":
    """Open every vector in *ir* into components.  Returns (new IR, names used).

    Raises :class:`UnopenedVector` when a declared vector is still standing
    afterwards — see the module docstring for why that cannot be let through.
    """
    dim = _written_dimension(ir) or default_dim
    declared_by_measure = _measure_vars(ir)
    used: list[str] = []

    def names_for(base: str) -> list[str]:
        names = component_names(base, dim, suffixes)
        for n in names:
            if n not in used:
                used.append(n)
        return names

    def fn(node: Expr) -> "Expr | None":
        if isinstance(node, Integral) and node.measure_dim:
            names = names_for(node.var.name)
            out = _rewrite(node.expr, fn)
            # The range decorates the WHOLE volume, so every component carries
            # it — `\int_{-\infty}^{\infty} d^3k` is over all of k-space, not
            # over one axis with the other two left open.
            #
            # Written NOWHERE, it is all space.  A volume measure is not an
            # antiderivative — nobody writes `\int d^3x` for one — and reading
            # it as one produced `π^{3/2} erf(x_x) erf(x_y) erf(x_z)/8` for the
            # Gaussian: an answer with the integration variables still in it,
            # close enough to the real one to skim past.  Filling this in is
            # only safe because the result is HELD (see `hold_integrals`): the
            # range is the notation's meaning, evaluating it is a separate
            # request, and over an arbitrary basis it can never be evaluated
            # at all.
            lower, upper = node.lower, node.upper
            if lower is None and upper is None:
                lower = Mul(args=[Number(value=-1), Symbol(name=infinity)])
                upper = Symbol(name=infinity)
            for name in reversed(names):
                out = Integral(expr=out, var=Symbol(name=name),
                               lower=lower, upper=upper)
            return out

        if isinstance(node, DotProduct):
            left, right = node.left, node.right
            if (isinstance(left, Symbol) and isinstance(right, Symbol)
                    and is_vector(left.name, context, declared_by_measure)
                    and is_vector(right.name, context, declared_by_measure)):
                a, b = names_for(left.name), names_for(right.name)
                return Add(args=[Mul(args=[Symbol(name=p), Symbol(name=q)])
                                 for p, q in zip(a, b)])
            return None

        if isinstance(node, FunctionCall):
            opened: list[Expr] = []
            changed = False
            for arg in node.args:
                if isinstance(arg, Symbol) and is_vector(arg.name, context,
                                                         declared_by_measure):
                    opened.extend(Symbol(name=n) for n in names_for(arg.name))
                    changed = True
                else:
                    opened.append(_rewrite(arg, fn))
            if changed:
                return node.model_copy(update={"args": opened})
            return None

        return None

    out = _rewrite(ir, fn)
    _reject_unopened(out, context, declared_by_measure)
    return out, used


def _measure_vars(ir: Expr) -> "set[str]":
    r"""The vectors the expression itself declares — `\int d^3x` declares `x`."""
    names: set[str] = set()

    def visit(node: Expr) -> None:
        if isinstance(node, Integral) and node.measure_dim:
            names.add(node.var.name)

    _walk(ir, visit)
    return names


def _written_dimension(ir: Expr) -> "int | None":
    """The dimension the researcher WROTE, from the volume measure if there is one."""
    found: list[int] = []

    def visit(node: Expr) -> None:
        if isinstance(node, Integral) and node.measure_dim:
            found.append(int(node.measure_dim))

    _walk(ir, visit)
    return found[0] if found else None


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
        if isinstance(child, Expr):
            _walk(child, visit, seen)


def _bare_vectors(ir: Expr, context, also=None) -> list[str]:
    r"""Declared vectors standing on their OWN — a component does not count.

    `x_1` is an `IndexedSymbol` over the declared vector `x`, which is already a
    component reference; reading its base as an unopened vector refused
    `\int d^3k x_1` — perfectly good componentwise notation (caught by this
    module's own tests, 2026-08-08).
    """
    from notebook.models.ir import IndexedSymbol
    component_bases: set[int] = set()
    bare: list[str] = []

    def note(node: Expr) -> None:
        if isinstance(node, IndexedSymbol) and isinstance(node.base, Symbol):
            component_bases.add(id(node.base))

    def visit(node: Expr) -> None:
        if (isinstance(node, Symbol) and id(node) not in component_bases
                and is_vector(node.name, context, also)):
            bare.append(node.name)

    _walk(ir, note)
    _walk(ir, visit)
    return bare


def has_openable(ir: Expr, context) -> bool:
    """Is there anything here to open — a volume measure, or a bare vector?"""
    if ir is None:
        return False
    return bool(_written_dimension(ir)) or bool(_bare_vectors(ir, context))



def _reject_unopened(ir: Expr, context, also=None) -> None:
    left = _bare_vectors(ir, context, also)
    if left:
        raise UnopenedVector(left[0])
