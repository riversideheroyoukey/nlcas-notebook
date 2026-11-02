from __future__ import annotations
from typing import List, Optional, Tuple, Union
from pydantic import BaseModel, Field

# Ensure we can forward reference IR nodes
class Expr(BaseModel):
    """Base class for all Intermediate Representation expressions."""
    simplification_log: List[str] = Field(default_factory=list, exclude=True)
    
    def is_commutative_with(self, other: Expr, context: 'SymbolContext') -> bool: # type: ignore
        """Check if this expression commutes with another expression."""
        # By default, conservative assumption
        return False

class Symbol(Expr):
    """A mathematical symbol (variable, constant, function name)."""
    name: str

    def is_commutative_with(self, other: Expr, context: 'SymbolContext') -> bool: # type: ignore
        if isinstance(other, Symbol):
            return context.is_commutative(self.name, other.name)
        # For complex expressions, need to check all components, but for now:
        return False

class Number(Expr):
    """A scalar number (integer, float, etc)."""
    value: Union[int, float]

    def is_commutative_with(self, other: Expr, context: 'SymbolContext') -> bool: # type: ignore
        # Numbers commute with everything
        return True

class Add(Expr):
    """Addition of multiple expressions: A + B + ..."""
    args: List[Expr] = Field(default_factory=list)

class Mul(Expr):
    """Multiplication of multiple expressions: A * B * ...
    Order matters for non-commutative symbols.
    """
    args: List[Expr] = Field(default_factory=list)

class NonCommutativeMul(Expr):
    """Multiplication of multiple non-commutative expressions: A * B * ...
    Order is strictly preserved and must not be sorted.
    """
    args: List[Expr] = Field(default_factory=list)

class Pow(Expr):
    """Power: base^exponent"""
    base: Expr
    exponent: Expr

class Decorated(Expr):
    """An expression with a LaTeX decoration accent.

    Produced by SharedParser when it recognises a canonical identifier
    that was created by LatexConverter._decorations(), e.g. ``H_hat`` →
    ``Decorated(base=Symbol("H"), decoration="hat")``.

    CodeGenerators convert this back to the canonical suffix form
    (``H_hat``).  Display bridges may render it as ``\\hat{H}``.

    Valid *decoration* values: ``"hat"``, ``"tilde"``, ``"bar"``,
    ``"dot"``, ``"ddot"``, ``"check"``, ``"breve"``, ``"acute"``,
    ``"grave"``.
    """
    base: Expr
    decoration: str

class Adjoint(Expr):
    """Hermitian adjoint (dagger) of an expression."""
    expr: Expr

class Conjugate(Expr):
    """Complex conjugate of an expression."""
    expr: Expr

class Transpose(Expr):
    """Transpose of an expression."""
    expr: Expr

class DotProduct(Expr):
    """Dot product of two vectors: A · B"""
    left: Expr
    right: Expr

class CrossProduct(Expr):
    """Cross product of two vectors: A × B"""
    left: Expr
    right: Expr

class FunctionCall(Expr):
    """A function applied to arguments: f(x, y)

    ``spaced`` records that whitespace separated the callee from the opening
    paren in the surface text (``f (x)`` vs ``f(x)``).  The SemanticResolver uses
    it to disambiguate: ``f (x)`` is a product ``f·(x)`` unless ``f`` is a known
    or declared function — a decision that needs the SymbolContext, so the parser
    only records the signal and leaves the choice to the resolver.
    """
    func: Expr
    args: List[Expr] = Field(default_factory=list)
    spaced: bool = False

class Derivative(Expr):
    """Derivative of an expression with respect to variables."""
    expr: Expr
    vars: List[Tuple[Symbol, int]] = Field(default_factory=list) # e.g., [(x, 1), (y, 2)] for d^3 / dx dy^2
    is_partial: bool = False

class Integral(Expr):
    """Integral of an expression."""
    expr: Expr
    var: Symbol
    lower: Optional[Expr] = None
    upper: Optional[Expr] = None
    #: Dimension of a VOLUME measure — ``\int d^3k`` sets 3 and leaves ``var``
    #: the vector ``k``.  There is no d-dimensional integral to hand a CAS, so
    #: this form is refused at code generation (loudly, by name) and made
    #: concrete by expanding the vector into its components.  ``None`` is the
    #: ordinary one-variable integral, which is every integral written before
    #: this field existed.  Additive and optional.
    measure_dim: Optional[int] = None

class Limit(Expr):
    """The value *expr* approaches as *var* → *point*.

    Field names are the OPERATION's own (`expr`/`var`/`point`/`direction`), so
    the notation and the operation speak one vocabulary — the same quantity
    reached two ways must not need a translation table.

    ``direction`` is ``"plus"`` / ``"minus"`` for a one-sided limit and ``None``
    for the two-sided one.  It is not decoration: measured against a live
    Maxima, `limit(abs(x)/x, x, 0)` is `ind` while the two sides are `1` and
    `-1`, so dropping it answers a different question (LIMIT-ONE-SIDED).
    """
    expr: Expr
    var: Symbol
    point: Expr
    direction: Optional[str] = None

class Gradient(Expr):
    """Gradient of a scalar field: ∇f"""
    expr: Expr

class Divergence(Expr):
    """Divergence of a vector field: ∇ · F"""
    expr: Expr

class Curl(Expr):
    """Curl of a vector field: ∇ × F"""
    expr: Expr

class Laplacian(Expr):
    """Laplacian of a field: ∇²f or Δf"""
    expr: Expr

class Commutator(Expr):
    """Commutator [A, B] = AB - BA"""
    a: Expr
    b: Expr

class Anticommutator(Expr):
    """Anticommutator {A, B} = AB + BA"""
    a: Expr
    b: Expr

class Bra(Expr):
    """Dirac Bra <psi|"""
    label: Expr

class Ket(Expr):
    """Dirac Ket |psi>"""
    label: Expr

class Equation(Expr):
    """A relation LHS <op> RHS. op is one of "=", "<=", ">=", "<", ">"."""
    lhs: Expr
    rhs: Expr
    op: str = "="

class CasOperation(Expr):
    """A generic CAS operation like expand, factor, solve.
    Used by the Execution Layer to represent commands before compilation.
    """
    operation: str
    args: dict[str, Union[Expr, list, str, int, float]] = Field(default_factory=dict)

class OpaqueExpr(Expr):
    """An expression from a backend without a structured parser.

    Stores the backend's human-readable text (e.g. Julia's string(result))
    so that IR consumers have a non-None node to inspect, even when the
    algebraic structure cannot be decomposed into Add/Mul/… nodes.
    """
    text: str


class SurfaceExpr(Expr):
    """Base class for expressions that represent pure syntax (Surface AST)."""
    pass

class PrimeExpr(SurfaceExpr):
    """Prime notation: A' (e.g. f' or a'). Represents syntactic prime."""
    expr: Expr

class Juxtaposition(SurfaceExpr):
    """Implicit multiplication or juxtaposition: A B or 2 x."""
    left: Expr
    right: Expr


class BracketExpr(SurfaceExpr):
    """Generic bracket expression: [A, B] or {A, B}."""
    args: List[Expr] = Field(default_factory=list)
    kind: str  # "square" for [], "curly" for {}


class Subscript(SurfaceExpr):
    """Syntactic representation of a subscript: base_sub

    *bracketed* records that the subscript was written with the backends' own
    index brackets (``f[i](x)``) rather than LaTeX's ``_``.  Only the SPELLING
    is recorded here; what it means stays with ``subscript_policy``, which uses
    it to skip the single-index name fold — that fold answers an ambiguity the
    ``_`` spelling has (``C_i(t)`` vs ``p_x(x)``) and the bracket does not.
    """
    base: Expr
    sub: Expr
    bracketed: bool = False

class Superscript(SurfaceExpr):
    """Syntactic representation of a superscript: base^super"""
    base: Expr
    script: Expr

class Assignment(SurfaceExpr):
    """An assignment or definition operation: LHS := RHS or similar.
    Used for define_function or define_boson_operator.
    """
    lhs: Expr
    rhs: Expr
    is_operator_def: bool = False
    operator_kind: Optional[str] = None  # "boson" or "fermion"


class IndexSymbol(Expr):
    """A mathematical subscript index symbol (e.g., i, j)."""
    name: str
    is_up: bool = False
    domain: Optional[str] = None
    range: Optional[str] = None

class IndexedSymbol(Expr):
    """An object (scalar, matrix, tensor, operator) with subscript indices (e.g., x_i, A_{ij})."""
    base: Expr
    indices: List[IndexSymbol] = Field(default_factory=list)

class KroneckerDelta(Expr):
    """Kronecker delta δ_ij for discrete indices."""
    left: Expr
    right: Expr

class DiracDelta(Expr):
    """Dirac delta δ(x - y) for continuous fields."""
    x: Expr
    y: Expr

class IndexSum(Expr):
    r"""Discrete summation over dummy indices: Σ_j expr.

    ``lower`` / ``upper`` mirror :class:`Integral`'s and apply to the SINGLE
    dummy of a ``\sum_{i=1}^{N}``.  They are additive and optional: every
    construction site that does not write a range leaves them ``None``, and a
    saved ``ir_ast`` from before they existed loads unchanged.

    They are deliberately NOT per-index.  A multi-index sum is written without
    ranges in practice (``\sum_{m,n} h_{m,n}`` — over all modes), and an upper
    bound written as a superscript over several indices cannot be attributed to
    one of them without guessing.  What to do about a sum with no range at all
    is a separate decision, not a default to be filled in here (a range is a
    physics choice; see the "No hardcoded modeling choices" rule).
    """
    body: Expr
    dummy_indices: List[IndexSymbol] = Field(default_factory=list)
    lower: Optional[Expr] = None
    upper: Optional[Expr] = None
    #: ``"sum"`` (Σ) or ``"product"`` (Π).  The two differ ONLY in the operator
    #: applied to the terms: same dummies, same ranges, same placeholder pairs,
    #: same binding of the indices, same display bridge.  A separate IR node
    #: would have meant auditing all sixteen files that walk this one — and the
    #: `measure_dim` episode showed what that costs (five walkers rebuilt
    #: ``Integral`` and silently dropped a field added later).
    #:
    #: Additive with a default, so every existing construction site and every
    #: saved expression keeps meaning Σ.  The one site that REBUILDS an
    #: ``IndexSum`` was moved to ``model_copy`` for exactly that reason.
    op: str = "sum"
    #: The CAS's NOUN form (`'sum(...)`, quoted): "do not evaluate this".  A
    #: result comes back quoted when the CAS could not evaluate the sum, and
    #: re-emitting it unquoted is a DIFFERENT request.  Additive with a default,
    #: like `op` above — an expression the researcher wrote is not a noun.
    noun: bool = False
    #: A CONSTRAINT written among the indices, as the researcher wrote it:
    #: ``i \ne j``, ``i < j``, ``\langle i,j \rangle`` (nearest neighbours).
    #: Read so the expression can be WRITTEN and SEEN; what it means to the CAS
    #: is refused with a reason at code generation — the same path
    #: ``measure_dim`` takes for ``\int d^3k``.  Before it was read, the
    #: condition became an index and the sum ran over a dummy named ``ne``.
    constraint: Optional[str] = None
    #: Placeholder range NAMES, one ``[lo, hi]`` per dummy, for a sum the
    #: researcher wrote WITHOUT a range.  Allocated once at registration and
    #: carried here, so every sum gets its own pair and keeps it for as long as
    #: the expression is loaded — two sums are never claimed to share a range
    #: just because they were both written without one (open_items B-10 ②).
    #: ``Expression.ir_ast`` is ``exclude=True`` and does NOT survive a save, so
    #: a reloaded session rebuilds this from the canonical and the session's
    #: ``BoundLedger`` — same pairs, same CAS string (B-10 ②-c).  Additive and
    #: optional: an older IR has none and lowers as before.
    placeholder_bounds: List[List[str]] = Field(default_factory=list)


