"""Theory canonical → numpy expression string (N-1 lowering).

Follows the 3-layer compiler architecture: SharedParser (Surface AST) →
MaximaSemanticResolver (Semantic IR) → NumpyExprGenerator (numpy code).
Unsupported constructs raise UnknownNotationError — the numeric layer must
never silently mistranslate a theory expression (same fail-fast rule as
R-3; a wrong numeric formula with plausible output is the worst failure
mode this system knows).
"""
from __future__ import annotations

from notebook.models.ir import (
    BracketExpr, Expr, Symbol, Number, Add, Mul, NonCommutativeMul, Pow,
    FunctionCall,
)
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.expr_generator import ExprGenerator
from notebook.runtime.shared_parser import SharedParser, UnknownNotationError

# CAS function name → numpy spelling.  Whitelist — anything else fails loudly.
_NUMPY_FUNCS: dict[str, str] = {
    "sin": "np.sin", "cos": "np.cos", "tan": "np.tan",
    "asin": "np.arcsin", "acos": "np.arccos", "atan": "np.arctan",
    "atan2": "np.arctan2",
    "sinh": "np.sinh", "cosh": "np.cosh", "tanh": "np.tanh",
    "exp": "np.exp", "log": "np.log", "sqrt": "np.sqrt",
    "abs": "np.abs", "floor": "np.floor", "round": "np.round",
    "signum": "np.sign", "sign": "np.sign",
}

_CONSTANTS: dict[str, str] = {
    "%pi": "np.pi", "pi": "np.pi", "%e": "np.e", "%i": "1j", "im": "1j",
}


class NumpyExprGenerator(ExprGenerator):
    """Renders a resolved scalar IR as a numpy-vectorizable Python expression."""

    def generate(self, ir_ast: Expr) -> str:
        if isinstance(ir_ast, Symbol):
            if ir_ast.name in _CONSTANTS:
                return _CONSTANTS[ir_ast.name]
            return ir_ast.name
        if isinstance(ir_ast, Pow):
            base = self.generate(ir_ast.base)
            exp = self.generate(ir_ast.exponent)
            return f"({base})**({exp})"
        if isinstance(ir_ast, FunctionCall):
            if isinstance(ir_ast.func, Symbol):
                fname = ir_ast.func.name
                if fname in _NUMPY_FUNCS:
                    args = ", ".join(self.generate(a) for a in ir_ast.args)
                    return f"{_NUMPY_FUNCS[fname]}({args})"
            raise UnknownNotationError(
                "numeric.unsupported_function",
                func=getattr(ir_ast.func, "name", str(type(ir_ast.func).__name__)))
        if isinstance(ir_ast, (Add, Mul, NonCommutativeMul, Number)):
            return super().generate(ir_ast)
        # Everything else (IndexedSymbol, IndexSum, Derivative, Integral,
        # Commutator, matrices, …) is out of the phase-1 numeric scope.
        raise UnknownNotationError(
            "numeric.unsupported_construct", node=type(ir_ast).__name__)


def _collect_free(node: Expr, free: "set[str]") -> None:
    """Free symbol names of a resolved IR (np/1j spellings excluded)."""
    if isinstance(node, Symbol):
        if node.name not in _CONSTANTS:
            free.add(node.name)
        return
    if isinstance(node, FunctionCall):
        for a in node.args:
            _collect_free(a, free)
        return
    val = getattr(node, "args", None)
    if isinstance(val, (list, tuple)):
        for item in val:
            if isinstance(item, Expr):
                _collect_free(item, free)
    for field in ("base", "exponent", "left", "right", "expr", "lhs", "rhs"):
        val = getattr(node, field, None)
        if isinstance(val, Expr):
            _collect_free(val, free)


def canonical_to_numpy(canonical: str,
                       context: "SymbolContext | None" = None) -> "tuple[str, set[str]]":
    """Convert a theory canonical to (numpy expression, free symbol names).

    Free symbols are what the generated code must bind — either as the sweep
    variable or via fixed_values.  np/1j spellings are not counted as free.
    """
    from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver

    ctx = context or SymbolContext()
    surface = SharedParser().parse(canonical)
    if surface is None:
        raise UnknownNotationError("numeric.unsupported_construct", node="empty")
    resolved = MaximaSemanticResolver().resolve(surface, ctx)
    expr_str = NumpyExprGenerator().generate(resolved)

    free: set[str] = set()
    _collect_free(resolved, free)
    return expr_str, free


def is_matrix_canonical(canonical: str) -> bool:
    """True for a Maxima matrix literal canonical (``matrix([...],[...])``)."""
    return (canonical or "").lstrip().startswith("matrix(")


def canonical_matrix_to_numpy(canonical: str,
                              context: "SymbolContext | None" = None,
                              ) -> "tuple[list[list[str]], set[str], tuple[int, int]]":
    """Lower a Maxima matrix canonical to per-cell numpy expressions.

    The decomposition happens on the SURFACE AST — a matrix row is a
    ``BracketExpr``, and handing it to the semantic resolver would turn a
    2-element row into a Commutator (the bracket-normalization rule that is
    right for operator algebra and wrong here).  Each CELL is an ordinary
    scalar expression, so cell-level resolution + generation reuses the
    whitelist lowering unchanged.

    Returns (rows of numpy-expression strings, free symbols, (m, n)).
    """
    from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver

    ctx = context or SymbolContext()
    surface = SharedParser().parse(canonical)
    if not (isinstance(surface, FunctionCall)
            and isinstance(surface.func, Symbol)
            and surface.func.name == "matrix"):
        raise UnknownNotationError("numeric.unsupported_construct",
                                   node=type(surface).__name__)

    resolver = MaximaSemanticResolver()
    generator = NumpyExprGenerator()
    rows: list[list[str]] = []
    free: set[str] = set()
    ncols: "int | None" = None
    for row in surface.args:
        if not isinstance(row, BracketExpr):
            raise UnknownNotationError("numeric.unsupported_construct",
                                       node=type(row).__name__)
        cells: list[str] = []
        for cell in row.args:
            resolved = resolver.resolve(cell, ctx)
            cells.append(generator.generate(resolved))
            _collect_free(resolved, free)
        if ncols is None:
            ncols = len(cells)
        elif len(cells) != ncols:
            raise UnknownNotationError("numeric.matrix_ragged")
        rows.append(cells)
    if not rows or not ncols:
        raise UnknownNotationError("numeric.unsupported_construct",
                                   node="empty matrix")
    return rows, free, (len(rows), ncols)
