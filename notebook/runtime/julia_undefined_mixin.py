"""Julia backend mixin: undefined-symbol detection.

Extracted from JuliaBackendHandler (AUD-4 stage 2b).  Decides which symbols in
an IR are undefined (UI warning) and which block CAS execution (multi-char
physics parameters).  Mixed into JuliaBackendHandler; the known-name tables are
class attributes carried along with the methods.
"""
from __future__ import annotations


class JuliaUndefinedSymbolMixin:
    """Undefined-symbol detection helpers mixed into JuliaBackendHandler."""

    # ------------------------------------------------------------------
    # Undefined-symbol detection
    # ------------------------------------------------------------------

    # Names that are always "known" — math builtins, constants, Julia keywords.
    _KNOWN_SYMBOL_NAMES: frozenset[str] = frozenset({
        # math functions (shared)
        "sin", "cos", "tan", "cot", "sec", "csc",
        "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh",
        "exp", "log", "sqrt", "abs",
        "floor", "ceiling", "round", "sign",
        "max", "min", "gcd", "lcm",
        "diff", "integrate", "limit", "sum",
        # Julia / QuantumAlgebra builtins
        "normal_form", "comm", "acomm", "vacExpVal", "heisenberg_eom",
        "expand", "simplify", "factor", "real", "imag",
        "param", "params",
        # constants
        "pi", "Inf", "hbar", "im", "true", "false",
        "nabla", "delta",
        # Spin projections.  `\uparrow` reaches here as the canonical name
        # `uparrow` (see spin_label_canon), and a projection is an index VALUE —
        # the same kind of thing as the site number `1` — not a physics
        # parameter to declare.  Without this the researcher was told to define
        # "uparrow, downarrow" before computing (measured 2026-08-22).
        "uparrow", "downarrow",
        # Greek letters that are conventional standalone constants
        "alpha", "beta", "gamma", "epsilon", "theta", "lambda",
        "mu", "nu", "rho", "sigma", "tau", "phi", "chi", "psi", "omega",
        # uppercase (capital) versions
        "Gamma", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi", "Omega",
        # LaTeX decorator commands (appear as superscripts, never physics params)
        "ast", "star", "dagger", "dag", "prime",
        "mathrm", "text", "mathbf", "mathit",
        "hat", "tilde", "bar", "vec", "dot",
        # canonical expression syntax keywords (SharedParser produces FunctionCall for these)
        "IndexSum", "matrix",
        # internal
        "_nb_eval_index_sum", "_nb_eval_continuous_integral",
    })

    def find_undefined_symbols(self, ir_ast: "Expr | None", context: "SymbolContext") -> list[str]:
        """Scan *ir_ast* for symbols not registered in *context*.

        Skips: math builtins, summation dummy indices, LHS of equations,
        operator-adjoints (c' is derived from c), and ``_nb_*`` internals.
        Returns bare symbol names, e.g. ``["xi", "Delta"]``.
        """
        if ir_ast is None:
            return []
        result: list[str] = []
        seen: set[str] = set()
        self._scan_undefined(ir_ast, context, frozenset(), result, seen)
        return result

    def _scan_undefined(self, node, context, dummy_set: frozenset, result: list, seen: set) -> None:
        from notebook.models.ir import (
            Symbol, IndexedSymbol, IndexSymbol, IndexSum, Equation,
            Add, Mul, NonCommutativeMul, Pow, Adjoint, Conjugate, Transpose,
            FunctionCall, Commutator, Anticommutator, KroneckerDelta,
        )
        # Subscript is a surface-AST node produced by SharedParser for "A_x" notation.
        # If base+sub forms a registered symbol (e.g. "Sx" from "S"+"x"), skip entirely.
        try:
            from notebook.models.ir import Subscript as _Subscript
            if isinstance(node, _Subscript):
                base = node.base
                sub  = node.sub
                if isinstance(base, Symbol) and isinstance(sub, Symbol):
                    bname = base.name.rstrip("'†")
                    sname = sub.name
                    if len(sname) == 1 and context.get(bname + sname) is not None:
                        return  # e.g. S_x where Sx is registered
                    # Underscore-joined full name registered as a whole (e.g.
                    # "alpha_mkd" from a forward-transform LHS) — the subscript is
                    # part of the name, not an independent symbol, so don't descend
                    # into it (which would flag "mkd" as undefined and block).
                    if context.get(f"{bname}_{sname}") is not None:
                        return
                # Fall through to generic recursion for other subscripts
        except (ImportError, AttributeError):
            pass
        if node is None:
            return
        if isinstance(node, Equation):
            # LHS is being defined — scan only RHS
            self._scan_undefined(node.rhs, context, dummy_set, result, seen)
            return
        if isinstance(node, IndexSum):
            new_dummy = dummy_set | {
                idx.name for idx in node.dummy_indices if hasattr(idx, "name")
            }
            self._scan_undefined(node.body, context, new_dummy, result, seen)
            return
        if isinstance(node, IndexedSymbol):
            base = node.base
            if isinstance(base, Symbol):
                bname = base.name.rstrip("'†")
                # Reconstruct full name e.g. "H_0" and check that before the bare base
                idx_parts = []
                for idx in (node.indices or []):
                    if isinstance(idx, IndexSymbol):
                        idx_parts.append(idx.name)
                    elif isinstance(idx, Symbol):
                        idx_parts.append(idx.name)
                if idx_parts:
                    full_name = "_".join([bname] + idx_parts)
                    # Spin-component shortcut: if base+index concatenated (e.g. "Sx")
                    # is already a registered symbol, the subscript notation "S_x" is
                    # just an alternate spelling of the component → skip entirely.
                    if (len(idx_parts) == 1 and len(idx_parts[0]) == 1
                            and context.get(bname + idx_parts[0]) is not None):
                        return  # e.g. S_x where Sx is registered → no warning
                    self._check_name(full_name, context, dummy_set, result, seen)
                else:
                    self._check_name(bname, context, dummy_set, result, seen)
            # Don't recurse into indices — they are dummy/summation variables
            self._scan_undefined(base, context, dummy_set, result, seen)
            return
        if isinstance(node, Symbol):
            self._check_name(node.name.rstrip("'†"), context, dummy_set, result, seen)
            return
        if isinstance(node, (IndexSymbol,)):
            return  # always a summation/index variable
        if isinstance(node, KroneckerDelta):
            return  # built-in
        # Generic recursion over pydantic/dataclass children
        for field_name in getattr(type(node), "model_fields", {}):
            val = getattr(node, field_name, None)
            if val is None:
                continue
            if getattr(type(val), "model_fields", None) is not None or hasattr(val, "__dataclass_fields__"):
                self._scan_undefined(val, context, dummy_set, result, seen)
            elif isinstance(val, list):
                for item in val:
                    if getattr(type(item), "model_fields", None) is not None or hasattr(item, "__dataclass_fields__"):
                        self._scan_undefined(item, context, dummy_set, result, seen)

    # QuantumAlgebra exports these single-char names as operator constructors into
    # Main scope.  They behave like multi-char physics parameters — must be
    # explicitly declared before use, otherwise Julia raises a cryptic MethodError.
    _QA_BUILTIN_OP_NAMES: frozenset[str] = frozenset({"a", "f"})

    def blocks_on_undefined_symbols(self, ir_ast: "Expr | None", context: "SymbolContext") -> list[str]:
        """Block CAS execution for multi-character physics symbols that are undefined.

        Single-letter symbols (x, y, z, ...) are treated as free algebraic
        variables — Julia auto-declares them via @variables on UndefVarError.
        Multi-character symbols (xi, Delta, omega_n, ...) must be declared
        explicitly before use in a quantum session.

        Exception: 'a' and 'f' are exported by QuantumAlgebra as boson/fermion
        operator constructors into Main scope and must be declared (e.g. via
        define_boson_operator) before algebraic use, even though len == 1.
        """
        blocking = []
        for s in self.find_undefined_symbols(ir_ast, context):
            # Strip subscript and trailing digits to get the base name.
            # "xi_k" → "xi" (len 2, blocked), "a0" → "a" (len 1, QA builtin → blocked).
            base = s.split("_")[0].rstrip("0123456789")
            if len(base) > 1 or base in self._QA_BUILTIN_OP_NAMES:
                blocking.append(s)
        return blocking

    def _check_name(self, name: str, context, dummy_set: frozenset, result: list, seen: set) -> None:
        if (not name
                or name in self._KNOWN_SYMBOL_NAMES
                or name in ("i", "im")              # imaginary unit constants
                # Electrical-engineering convention: `j` is √-1 (not an index).
                or (name == "j"
                    and "electrical_engineering" in getattr(context, "domains", []))
                or name in dummy_set
                or name.startswith("_nb_")
                or name.startswith(":")
                or context.get(name) is not None
                or name in seen):
            return
        seen.add(name)
        result.append(name)

