"""Maxima-specific implementation of BackendHandler."""
from __future__ import annotations

import re

from notebook.runtime.shared_parser import split_relation, strip_lhs

from notebook.models.quantity import Quantity
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.backend_handler import BackendHandler, BuildResult, MATH_FUNC_NAMES
from notebook.runtime.maxima_game_mixin import MaximaGameMixin
from notebook.runtime.maxima_signal_mixin import MaximaSignalMixin
from notebook.runtime import maxima_error_classify
from notebook.runtime.maxima_constants import (
    constant_to_latex, quote_aggregates)
from notebook.runtime.maxima_subst_guard import (
    lhs_may_match, shadow_coordinate_variables, shadow_differentiated_operand,
    shadow_own_definition)
from notebook.runtime.vector_operand import (
    cross_products_as_calls, expression_components, is_bare_name, listify,
    not_a_vector_reason, three_entries, unwrap_vector_mark)
from notebook.runtime.vector_op_types import infer_algebra, value_lookup
from notebook.runtime.expr_lowering import LoweringRefused, lower_expr_lookup
from notebook.runtime.maxima_code_generator import template_product_operator

def user_form(expr_lookup: dict, ref: str) -> str:
    r"""研究者が書いた綴りのほう——**型を読むときはこちら**。

    外積は `~` のままでは CAS が評価できないので、`expr_lookup` の側は
    `nb_cross(...)` の呼び出しへ差し替えられていることがある（`vector_cas_view`）。
    その綴りからは型が読めない（`~` は `CrossProduct` 節点、落とした後は未知の
    関数呼び出し）ので、差し替えた側が `__user__<eid>` に残した綴りを先に見る。
    約束は `lower_expr_lookup` と同じで、**読む側を1箇所に**置く: 成分の読みと
    ノルムが別々に綴ると、ノルムだけが「ベクトルにも行列にも見えません」で断る
    （実測 2026-09-19、まさにそれを踏んだ）。
    """
    return expr_lookup.get(f"__user__{ref}") or expr_lookup.get(ref, ref)


# ---------------------------------------------------------------------------
# Maxima-specific constants and helpers
# ---------------------------------------------------------------------------

# 'diff(name(var),var,n) or 'diff(name,var,n) pattern in Maxima canonical output.
# The optional group 2 captures the explicit argument (e.g. "t" in q(t));
# it is absent when q was declared via depends([q],[t]) and Maxima writes q bare.
# Fields carrying identifier NAMES, passed through unparsed (a subscripted name
# like "k_B" would otherwise lower to "k(B)").  One set, shared with the
# validator and Julia's builder — see notebook/runtime/expr_ref_fields.py.
from notebook.runtime.expr_ref_fields import (
    IDENTIFIER_KEYS as _IDENTIFIER_KEYS, is_expr_ref)


def _is_row_or_column_matrix(code: str) -> bool:
    """`matrix(...)` が1行または1列か。**括弧の深さで数える**——行の中の
    `matrix([1,2],[3,4])` を素朴に `,` で割ると列数を数え違える。"""
    inner = code.strip()
    if not inner.startswith("matrix(") or not inner.endswith(")"):
        return False
    inner = inner[len("matrix("):-1]
    rows, depth, cur = [], 0, []
    for ch in inner:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            rows.append("".join(cur)); cur = []
            continue
        cur.append(ch)
    if cur:
        rows.append("".join(cur))
    rows = [r.strip() for r in rows if r.strip()]
    if not rows:
        return False
    if len(rows) == 1:                     # 1行
        return True
    def _width(row: str) -> int:
        row = row.strip()
        if row.startswith("[") and row.endswith("]"):
            row = row[1:-1]
        d, n = 0, 1
        for ch in row:
            if ch in "([{":
                d += 1
            elif ch in ")]}":
                d -= 1
            elif ch == "," and d == 0:
                n += 1
        return n if row.strip() else 0
    return all(_width(r) == 1 for r in rows)   # 1列

_UNEVAL_DIFF_RE = re.compile(r"'diff\((\w+)(?:\((\w+)\))?,(\w+),(\d+)\)")

# Strips Maxima 'diff(…) subexpressions from a canonical before variable scanning
_DIFF_STRIP_RE = re.compile(r"'diff\([^)]*\)")


def _paren_balanced(inner: str) -> bool:
    depth = 0
    for ch in inner:
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                return False
            depth -= 1
    return depth == 0


_OVER_RE = re.compile(
    r'\{\{((?:[^{}]|\{[^{}]*\})*)\}\\over\{((?:[^{}]|\{[^{}]*\})*)\}\}'
)


def _over_to_frac(latex: str) -> str:
    """Convert Maxima's {{A}\\over{B}} fraction notation to \\frac{A}{B}."""
    prev = None
    while prev != latex:
        prev = latex
        latex = _OVER_RE.sub(r'\\frac{\1}{\2}', latex)
    return latex


def _strip_outer_parens(latex: str) -> str:
    """Remove cosmetic outer parentheses added by Maxima's tex1()."""
    s = latex.strip()
    for prefix, suffix in ((r"\left(", r"\right)"), ("(", ")")):
        if s.startswith(prefix) and s.endswith(suffix):
            inner = s[len(prefix): len(s) - len(suffix)]
            if _paren_balanced(inner):
                return inner.strip()
        neg_prefix = "-" + prefix
        if s.startswith(neg_prefix) and s.endswith(suffix):
            inner = s[len(neg_prefix): len(s) - len(suffix)]
            if _paren_balanced(inner):
                return "-" + inner.strip()
    return s


def _apply_dot_notation_to_latex(
    canonical: str, raw_latex: str, registry: SymbolContext
) -> str:
    """Replace Maxima Leibniz LaTeX for unevaluated diffs with dot/ddot notation."""
    result = raw_latex
    for m in _UNEVAL_DIFF_RE.finditer(canonical):
        func_name = m.group(1)
        func_arg  = m.group(2)   # None when bare symbol form: 'diff(q,t,1)
        diff_var  = m.group(3)
        order     = int(m.group(4))

        # For function form 'diff(q(t),t,1): func_arg must equal diff_var.
        # For bare-symbol form 'diff(q,t,1): func_arg is None — skip this check.
        if func_arg is not None and func_arg != diff_var:
            continue

        qty = registry.get_quantity(func_name)
        if (qty is None
                or not qty.time_dependent
                or qty.notation != "dot"
                or qty.time_parameter != diff_var):
            continue

        base = qty.latex_name or func_name
        if order == 1:
            dot_latex = rf"\dot{{{base}}}"
        elif order == 2:
            dot_latex = rf"\ddot{{{base}}}"
        elif order == 3:
            dot_latex = rf"\dddot{{{base}}}"
        else:
            dot_latex = rf"{{{base}}}^{{({order})}}"

        v = re.escape(diff_var)
        nm = re.escape(func_name)
        if order == 1:
            frac_pat = r"\{\{d\}\\over\{d\\," + v + r"\}\}"
        else:
            n = str(order)
            frac_pat = r"\{\{d\^" + n + r"\}\\over\{d\\," + v + r"\^" + n + r"\}\}"

        # Pattern for explicit-arg form: {{d}\over{d\,t}}\,q\left(t\right)
        leibniz_pat = frac_pat + r"\\," + nm + r"\\left\(" + v + r"\\right\)"
        result = re.sub(r"\\left\(" + leibniz_pat + r"\\right\)", lambda _: dot_latex, result)
        result = re.sub(leibniz_pat, lambda _: dot_latex, result)

        # Pattern for bare-symbol form: {{d}\over{d\,t}}\,q  (no \left(t\right))
        # tex1() wraps this in \left(...\right) when inside a product:
        #   l^2*m*omega^2*sin(q)*'diff(q,t,1)
        #   → ...sin q\,\left({{d}\over{d\,t}}\,q\right)
        leibniz_pat_bare = frac_pat + r"\\," + nm
        result = re.sub(r"\\left\(" + leibniz_pat_bare + r"\\right\)", lambda _: dot_latex, result)
        result = re.sub(leibniz_pat_bare, lambda _: dot_latex, result)

    return result


# ---------------------------------------------------------------------------
# Coordinate system definitions for vector calculus
# ---------------------------------------------------------------------------

# Component suffixes per coordinate system
# "custom" uses numeric suffixes (1/2/3) matching nb_curl_1/2/3 in Lamé-generated functions
_COORD_COMPONENTS: dict[str, tuple[str, str, str]] = {
    "cartesian":   ("x",     "y",      "z"),
    "spherical":   ("r",     "theta",  "phi"),
    "cylindrical": ("r",     "phi",    "z"),
    "custom":      ("1",     "2",      "3"),
}

# Coordinate variables per system
_COORD_VARS: dict[str, tuple[str, str, str]] = {
    "cartesian":   ("x",     "y",     "z"),
    "spherical":   ("r",     "theta", "phi"),
    "cylindrical": ("r",     "phi",   "z"),
}

# LaTeX display names for components
_COORD_COMPONENT_LATEX: dict[str, tuple[str, str, str]] = {
    "cartesian":   ("x",           "y",            "z"),
    "spherical":   ("r",           r"\theta",       r"\phi"),
    "cylindrical": ("r",           r"\phi",         "z"),
}

def _make_nb_functions_cartesian() -> list[str]:
    """Return Maxima commands that define nb_div, nb_curl_*, nb_grad, nb_laplacian_scalar
    for Cartesian coordinates."""
    return [
        "nb_div(Ax, Ay, Az) := diff(Ax,x) + diff(Ay,y) + diff(Az,z)$",
        "nb_curl_x(Ax, Ay, Az) := diff(Az,y) - diff(Ay,z)$",
        "nb_curl_y(Ax, Ay, Az) := diff(Ax,z) - diff(Az,x)$",
        "nb_curl_z(Ax, Ay, Az) := diff(Ay,x) - diff(Ax,y)$",
        "nb_grad(f) := [diff(f,x), diff(f,y), diff(f,z)]$",
        "nb_laplacian_scalar(f) := diff(f,x,2) + diff(f,y,2) + diff(f,z,2)$",
    ]


def _make_nb_functions_spherical() -> list[str]:
    return [
        ("nb_div(Ar, Atheta, Aphi) :="
         " diff(r^2*Ar,r)/r^2"
         " + diff(sin(theta)*Atheta,theta)/(r*sin(theta))"
         " + diff(Aphi,phi)/(r*sin(theta))$"),
        ("nb_curl_r(Ar, Atheta, Aphi) :="
         " (diff(sin(theta)*Aphi,theta) - diff(Atheta,phi))/(r*sin(theta))$"),
        ("nb_curl_theta(Ar, Atheta, Aphi) :="
         " (diff(Ar,phi)/(sin(theta)) - diff(r*Aphi,r))/r$"),
        ("nb_curl_phi(Ar, Atheta, Aphi) :="
         " (diff(r*Atheta,r) - diff(Ar,theta))/r$"),
        ("nb_grad(f) :="
         " [diff(f,r), diff(f,theta)/r, diff(f,phi)/(r*sin(theta))]$"),
        ("nb_laplacian_scalar(f) :="
         " diff(r^2*diff(f,r),r)/r^2"
         " + diff(sin(theta)*diff(f,theta),theta)/(r^2*sin(theta))"
         " + diff(f,phi,2)/(r^2*sin(theta)^2)$"),
    ]


def _make_nb_functions_cylindrical() -> list[str]:
    return [
        "nb_div(Ar, Aphi, Az) := diff(r*Ar,r)/r + diff(Aphi,phi)/r + diff(Az,z)$",
        "nb_curl_r(Ar, Aphi, Az) := diff(Az,phi)/r - diff(Aphi,z)$",
        "nb_curl_phi(Ar, Aphi, Az) := diff(Ar,z) - diff(Az,r)$",
        "nb_curl_z(Ar, Aphi, Az) := (diff(r*Aphi,r) - diff(Ar,phi))/r$",
        "nb_grad(f) := [diff(f,r), diff(f,phi)/r, diff(f,z)]$",
        ("nb_laplacian_scalar(f) :="
         " diff(r*diff(f,r),r)/r + diff(f,phi,2)/r^2 + diff(f,z,2)$"),
    ]


_COORD_NB_FUNCTIONS: dict[str, list[str]] = {
    "cartesian":   _make_nb_functions_cartesian(),
    "spherical":   _make_nb_functions_spherical(),
    "cylindrical": _make_nb_functions_cylindrical(),
}

# nb_cross / nb_dot are coordinate-system independent (pure algebraic)
_NB_CROSS_DOT_FUNCTIONS: list[str] = [
    "nb_cross(A, B) := [A[2]*B[3]-A[3]*B[2], A[3]*B[1]-A[1]*B[3], A[1]*B[2]-A[2]*B[1]]$",
    "nb_dot(A, B) := A[1]*B[1] + A[2]*B[2] + A[3]*B[3]$",
]


def _make_nb_functions_from_lame(
    coords: tuple[str, str, str],
    lame: tuple[str, str, str],
) -> list[str]:
    """Generate nb_div/curl/grad/laplacian from Lamé coefficients."""
    q1, q2, q3 = coords
    h1, h2, h3 = lame
    return [
        (f"nb_div(A1, A2, A3) :="
         f" (diff({h2}*{h3}*A1,{q1})"
         f" + diff({h1}*{h3}*A2,{q2})"
         f" + diff({h1}*{h2}*A3,{q3}))/({h1}*{h2}*{h3})$"),
        (f"nb_curl_1(A1, A2, A3) :="
         f" (diff({h3}*A3,{q2}) - diff({h2}*A2,{q3}))/({h2}*{h3})$"),
        (f"nb_curl_2(A1, A2, A3) :="
         f" (diff({h1}*A1,{q3}) - diff({h3}*A3,{q1}))/({h1}*{h3})$"),
        (f"nb_curl_3(A1, A2, A3) :="
         f" (diff({h2}*A2,{q1}) - diff({h1}*A1,{q2}))/({h1}*{h2})$"),
        (f"nb_grad(f) :="
         f" [diff(f,{q1})/{h1}, diff(f,{q2})/{h2}, diff(f,{q3})/{h3}]$"),
        (f"nb_laplacian_scalar(f) :="
         f" (diff({h2}*{h3}/{h1}*diff(f,{q1}),{q1})"
         f" + diff({h1}*{h3}/{h2}*diff(f,{q2}),{q2})"
         f" + diff({h1}*{h2}/{h3}*diff(f,{q3}),{q3}))/({h1}*{h2}*{h3})$"),
    ]


# ---------------------------------------------------------------------------
# MaximaBackendHandler
# ---------------------------------------------------------------------------

# Linear-algebra operations that act on a matrix/expression VALUE.  When their
# source is an H-1 equation ("A = matrix(...)"), build_command operates on the
# RHS (the value), not the whole equation.  Equation-consuming ops (use_relation,
# solve, substitute) are NOT in this set and keep the full "LHS = RHS" form.
_MATRIX_VALUE_OPS: frozenset[str] = frozenset({
    "eigenvalues", "eigenvectors", "inverse_matrix", "determinant",
    "transpose", "trace", "matrix_multiply", "characteristic_polynomial",
    "diagonalize", "diagonal_matrix",
})


#: 研究者が言った微分の種別。エンジンが `_derivative_kind` として渡す
#: （`_preprocess_args` が `_is_*` を剥がすのは検証器に見せないためで、
#: ビルダに隠すためではない——隠れていた結果 ∂ が d を返していた）。
def _derivative_kind(args: dict) -> str:
    return str(args.get("_derivative_kind") or "")


def time_dep_derivative(name: str, t: str, order: int) -> str:
    r"""How a DECLARED time-dependent quantity's derivative is spelled for Maxima.

    `depends([q],[t])` is exactly the mechanism that makes `q` a function of `t`
    **without writing `q(t)` in the expressions**, so the noun form carries the
    bare name.  One helper because two places spell it and they must agree:
    `_prepare_expr_lookup`'s map (which rewrites `q_dot` inside the EXPRESSION)
    and `wrt_qdot_canonical` (which spells the differentiation VARIABLE).

    They did not agree.  The variable was `'diff(q(t),t,1)` — a third spelling
    that matched neither the atom nor the map's output — so `∂/∂q̇` came back as
    **0, `completed`**: the canonical momentum of the researcher's own
    Lagrangian.  And the answer depended on a DECLARATION made lines earlier —
    the same sentence on the same expression gave `m r^2` when the quantity was
    not declared time-dependent (measured 2026-09-25, DOT-DISPLAY-IS-AN-ATOM)::

        diff(m*r^2*'diff(theta,t,1), 'diff(theta,t,1), 1)     →  m*r^2
        diff(m*r^2*'diff(theta,t,1), 'diff(theta(t),t,1), 1)  →  0
    """
    return f"'diff({name},{t},{order})"


class MaximaBackendHandler(BackendHandler, MaximaGameMixin, MaximaSignalMixin):
    """Backend handler for Maxima sessions."""

    #: スキーマの ``supported_backends`` での自分の綴り。
    _SCHEMA_BACKEND = "maxima"

    #: いま設置されている ``nb_*`` が束縛している座標変数。表から引かず**設置時に
    #: 記録する**——カスタム座標系の名前は研究者のもので、表は知らない。
    _nb_coord_vars: tuple = ("x", "y", "z")

    def __init__(self, builder: "MaximaCommandBuilder") -> None:  # type: ignore[name-defined]
        self._builder = builder
        from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
        if self._builder is not None:
            self._builder._compiler = MaximaCodeGenerator()
        self._context: "SymbolContext | None" = None

    # Sign qualifier ("positive"/"negative"/"nonneg"/"nonpos") → Maxima
    # comparison operator against 0, used to build the assume() condition.
    _SIGN_TO_MAXIMA_OP: dict = {
        "positive": ">", "negative": "<", "nonneg": ">=", "nonpos": "<=",
    }

    # Error signatures live in maxima_error_classify (see its docstring).
    def cas_holds_value(self, adapter, name: str) -> bool:
        """`values` は Maxima 自身が持つ「`:` で束縛された名前」の一覧であり、
        P2 が使うのがまさにその `:` である（実測 2026-09-19:
        `y : x^3` のあと `member('y, values)` は `true`、素の `u` は `false`）。

        **名前を評価しない**訊き方を選んだ——`is(y # 'y)` でも答えは出るが、
        値が大きい式だと丸ごと読み出すことになる。
        """
        if not re.fullmatch(r"[A-Za-z_]\w*", name or ""):
            return False
        result = adapter.execute(f"member('{name}, values)")
        return bool(result.ok) and (result.value or "").strip() == "true"

    classify_cas_error = staticmethod(maxima_error_classify.classify)
    cas_error_args = staticmethod(maxima_error_classify.message_args)

    #: Maxima holds and withdraws facts with these two.  Named here rather than
    #: built in ``engine.add_assumption``, which is the backend-INDEPENDENT
    #: layer and had been assembling ``assume(...)`` itself (open_items BP-5).
    _ASSUME_FN = "assume"
    _FORGET_FN = "forget"

    def assumption_canonical(self, canonical: str) -> str:
        r"""``a # b`` → ``notequal(a, b)`` — Maxima's assume spells it that way.

        Measured 2026-08-11: ``assume(a # b)`` FAILS, while
        ``assume(notequal(a,b))`` registers the fact and makes
        ``is(equal(a,b))`` answer false.  So "a is not equal to b" is a
        statement this CAS can hold — it simply cannot be written the way the
        expression language writes it.  Everything else is passed through.

        The split uses ``find_top_level_relation``, the one function that
        defines where a relation splits, rather than a second scan of its own.
        """
        from notebook.runtime.shared_parser import find_top_level_relation
        rel = find_top_level_relation(canonical)
        if not rel or rel[1] != "#":
            return canonical
        idx = rel[0]
        lhs, rhs = canonical[:idx].strip(), canonical[idx + 1:].strip()
        if not lhs or not rhs:
            return canonical
        return f"notequal({lhs}, {rhs})"

    def assume_sign(self, engine, sym_name: str, sign: str, kind: "str | None" = None) -> None:
        """Send ``assume(sym_name OP 0)`` for a scalar parameter's sign qualifier.

        The canonical sent to Maxima uses the bare ascii name (e.g. ``gamma``);
        the display kept for the UI/TeX export uses the symbol's LaTeX form
        (e.g. ``\\gamma``) via the same lookup the rest of the app uses, so a
        TeX-declared symbol's assumption tile doesn't fall back to its
        internal ascii identifier.
        """
        op = self._SIGN_TO_MAXIMA_OP.get(sign)
        if op is None:
            return
        condition = f"{sym_name} {op} 0"
        display = f"{self._canonical_name_to_latex(sym_name)} {op} 0"
        engine.add_assumption(condition, display)

    #: Number domain → the Maxima feature that states it.  `natural` is a
    #: positive integer, and Maxima has no single feature for that: the
    #: integer half is declared here and the sign half is `assume_sign`'s, so
    #: the two together say it.
    _DOMAIN_TO_MAXIMA_FEATURE = {
        "real": "real", "complex": "complex",
        "integer": "integer", "natural": "integer",
    }

    def declare_domain(self, engine, sym_name: str, domain: str) -> None:
        """Send ``declare(name, <feature>)`` for a scalar's number domain.

        Measured 2026-08-16: the declaration coexists with the `declare(name,
        scalar)` the definition already sends (both features stay true), and it
        changes answers — `conjugate(z)*z` stops collapsing to `z^2`, and
        `sin(n*%pi)` becomes `0`.
        """
        feature = self._DOMAIN_TO_MAXIMA_FEATURE.get((domain or "").strip())
        if not feature or not sym_name:
            return
        try:
            engine._adapter.execute(f"declare({sym_name}, {feature})")
        except Exception:
            # A declaration the CAS refuses must not fail the definition step:
            # the symbol is registered either way, and the researcher sees the
            # acceptance.  (What must NOT happen is the silent OPPOSITE — the
            # acceptance shown while nothing is sent, which is this defect.)
            pass

    # -- Coordinate system initialization ------------------------------------

    def coordinate_init_commands(self, coord_system: str) -> list[str]:
        """Return Maxima commands that define nb_div/curl/grad/laplacian for
        the given built-in coordinate system, plus coordinate-independent
        nb_cross / nb_dot.  Returns nb_cross/nb_dot alone for unknown names."""
        cmds = list(_COORD_NB_FUNCTIONS.get(coord_system, []))
        cmds.extend(_NB_CROSS_DOT_FUNCTIONS)
        return cmds

    def coordinate_init_commands_from_lame(
        self,
        coords: tuple[str, str, str],
        lame: tuple[str, str, str],
    ) -> list[str]:
        """Generate nb_* functions from user-supplied Lamé coefficients."""
        return _make_nb_functions_from_lame(coords, lame)

    def reserved_coord_vars(self, coord_system: str) -> tuple[str, str, str]:
        """Return the coordinate variable names for the given system."""
        return _COORD_VARS.get(coord_system, ("x", "y", "z"))

    def coord_component_suffixes(self, coord_system: str) -> tuple[str, str, str]:
        """Return the component suffix triple for the given system."""
        return _COORD_COMPONENTS.get(coord_system, ("x", "y", "z"))

    @property
    def binds_symbol_definitions(self) -> bool:
        """Maxima evaluates assignments (``H : RHS``) immediately, so symbol
        definitions are bound into CAS state at definition time."""
        return True

    def init_coordinate_system(self, adapter: object, session: object) -> None:
        """Send coordinate-system ``nb_*`` function definitions to Maxima.

        Driven by ``session.coordinate_system`` (defaults to ``cartesian``).
        For a ``custom`` system, builds ``nb_*`` from the user-supplied Lamé
        coefficients; falls back to cartesian when those are incomplete.
        """
        import logging
        if getattr(session, "coordinate_system", None) is None:
            logging.getLogger(__name__).warning(
                "座標系が指定されていません。デカルト座標 (cartesian) を使用します。"
                " Coordinate system not specified; falling back to cartesian."
            )
        coord_sys = getattr(session, "coordinate_system", None) or "cartesian"
        if coord_sys == "custom":
            cv = getattr(session, "custom_coord_vars", None)
            cl = getattr(session, "custom_coord_lame", None)
            if cv and cl and len(cv) == 3 and len(cl) == 3:
                cmds = self.coordinate_init_commands_from_lame(cv, cl)
                cmds = list(cmds) + list(_NB_CROSS_DOT_FUNCTIONS)
                self._nb_coord_vars = tuple(cv)
            else:
                logging.getLogger(__name__).warning(
                    "カスタム座標系の設定が不完全です。デカルトにフォールバックします。"
                )
                cmds = self.coordinate_init_commands("cartesian")
                self._nb_coord_vars = self.reserved_coord_vars("cartesian")
        else:
            cmds = self.coordinate_init_commands(coord_sys)
            self._nb_coord_vars = self.reserved_coord_vars(coord_sys)
        for cmd in cmds:
            adapter.execute(cmd)

    # -- define_vector_field: build command interceptor ----------------------

    def build_define_vector_field(
        self,
        name: str,
        coord_system: str,
        natural_vars: list[str],
    ) -> str:
        """Build a Maxima ``depends`` command for a vector field declaration."""
        suffixes = self.coord_component_suffixes(coord_system)
        components = [f"{name}_{s}" for s in suffixes]
        vars_str = "[" + ",".join(natural_vars) + "]"
        comp_list = "[" + ",".join(components) + "]"
        # 依存変数も宣言が束縛する名前（COORD-NAME-BOUND の3つ目の出口、理由は subst_guard）
        return shadow_coordinate_variables(natural_vars, f"depends({comp_list},{vars_str})")

    def vector_field_latex(self, name: str) -> str:
        """Return the LaTeX display string for a vector field symbol."""
        return rf"\boldsymbol{{{name}}}"

    # -- Recursive component resolver ----------------------------------------

    def replay_session_declarations(self, adapter: object, commands: list[str]) -> None:
        """Re-execute Maxima declaration commands after session load.

        For Maxima, only `depends(...)` commands need to be replayed — the
        `nb_*` function definitions are handled by `_init_coordinate_system`.
        Multi-line commands (from `define_coordinate_system`) are split and
        executed line-by-line.
        """
        for cmd in commands:
            for line in cmd.strip().splitlines():
                line = line.strip()
                if line:
                    adapter.execute(line)  # type: ignore[attr-defined]

    # Patterns that match nested vector-op expressions in the `expr` argument.
    # Supported forms:
    #   curl(X)          divergence(X)    gradient(X)    laplacian(X)
    _VEC_OP_RE = re.compile(
        r'^(?:'
        r'(curl|rot)\((.+)\)'
        r'|(div|divergence)\((.+)\)'
        r'|(grad|gradient)\((.+)\)'
        r'|(laplacian|scalar_laplacian)\((.+)\)'
        r'|(cross(?:_product)?)\((.+),(.+)\)'
        r')$',
        re.IGNORECASE,
    )

    # Matches a Maxima list of exactly 3 elements: [e1, e2, e3]
    # Used to recognise intermediate vector results from cross_product / curl etc.
    _MAXIMA_LIST3_RE = re.compile(
        r'^\s*\[\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*\]\s*$'
    )

    #: `vec(...)` の包みの中身が適用形かどうか（包みは `vector_operand.unwrap_vector_mark`）。
    _APPLIED_NAME_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\((.*)\)", re.S)

    def _build_vec_components(
        self,
        expr: str,
        coord_sys: str,
        expr_lookup: dict,
        depth: int = 0,
    ) -> tuple[str, str, str]:
        """Recursively resolve *expr* into a (c1, c2, c3) component triple.

        Raises
        ------
        TypeError
            When the expression has scalar type but a vector is required
            (e.g. ``curl(divergence(A))`` — divergence returns a scalar).
        RecursionError
            When nesting depth exceeds 8.

        Supported recursive forms
        -------------------------
        - Named vector_field registered in SymbolContext → components directly
        - ``curl(X)``       → (nb_curl_?(…), nb_curl_?(…), nb_curl_?(…))
        - ``gradient(X)``   → (nb_grad(X)[1], nb_grad(X)[2], nb_grad(X)[3])
        """
        from notebook.runtime.i18n import t as _t_be
        if depth > 8:
            raise RecursionError(_t_be("vector.nesting_too_deep", self.lang))

        # --- Expression-ID lookup ---
        # **ベクトルの成分は「値」の側にある。** H-1 で名前つきに登録した式は、
        # `expr_lookup` の中で `LHS = RHS` の形に復元されている（`solve_equation`
        # がその形を要るため——外すと `y = x^2-1` を `x^2-1 = 0` として解く
        # silent-wrong になる、実測 2026-09-10）。等式の左辺は**名前**なので、
        # そのまま下の「名前つきベクトル場」の枝に落ちると
        # `nb_dot([a = [1, 2, 3]_x, …])` という綴りを CAS に渡してしまい、
        # 生の `Maxima evaluation error` が研究者に返っていた（起票 NORM-OPERATION
        # の測定中に発覚。`dot_product` と `cross_product` の両方）。
        # `strip_lhs` は「等式なら右辺、そうでなければそのまま」の唯一の答えで、
        # 括弧を数えるので `\sum_{i=1}^{3}` の `=` では切らない。
        # **型は利用者書式から読む。** `~` は `CrossProduct` 節点で「ベクトル」と
        # 読めるが、CAS 綴りへ落とすと未知の関数呼び出しになり読めない。だから
        # ベクトルの CAS 表示を入れた側が `__user__<eid>` に残した綴りを先に見る
        # （約束は `lower_expr_lookup` と同じ）。
        raw = strip_lhs(user_form(expr_lookup, expr))

        # --- Maxima list literal [e1, e2, e3] — intermediate vector result ---
        # This handles results from cross_product, curl, gradient stored as E-IDs.
        m_list = self._MAXIMA_LIST3_RE.match(raw)
        if m_list:
            return (m_list.group(1).strip(),
                    m_list.group(2).strip(),
                    m_list.group(3).strip())

        # --- SymbolContext lookup (named vector field) ---
        if self._context:
            qty = self._context.get_quantity(raw)
            if qty is not None:
                if qty.kind == "vector_field" and qty.components and len(qty.components) >= 3:
                    return (qty.components[0], qty.components[1], qty.components[2])
                if qty.kind in ("scalar_field", "scalar", "function"):
                    raise TypeError(_t_be("vector.operand_is_scalar", self.lang, operand=raw))

        # --- Nested vector-op pattern ---
        m = self._VEC_OP_RE.match(raw.strip())
        if m:
            if m.group(1):      # curl(X) → vector output ✓
                inner = m.group(2).strip()
                inner_comps = self._build_vec_components(inner, coord_sys, expr_lookup, depth + 1)
                c1, c2, c3 = inner_comps
                sfxs = _COORD_COMPONENTS.get(coord_sys, ("x", "y", "z"))
                s1, s2, s3 = sfxs
                if coord_sys == "cartesian":
                    return (
                        f"nb_curl_x({c1},{c2},{c3})",
                        f"nb_curl_y({c1},{c2},{c3})",
                        f"nb_curl_z({c1},{c2},{c3})",
                    )
                return (
                    f"nb_curl_{s1}({c1},{c2},{c3})",
                    f"nb_curl_{s2}({c1},{c2},{c3})",
                    f"nb_curl_{s3}({c1},{c2},{c3})",
                )

            elif m.group(3):    # divergence(X) → scalar output ✗ for vector context
                raise TypeError(_t_be("vector.operand_is_scalar", self.lang,
                                      operand=rf"\nabla \cdot {m.group(4).strip()}"))

            elif m.group(5):    # gradient(X) → vector output ✓
                inner = m.group(6).strip()
                # Verify inner is scalar
                scalar_expr = self._build_scalar_expr(inner, coord_sys, expr_lookup, depth + 1)
                # gradient components: nb_grad(f)[1..3]  (Maxima 1-indexed)
                return (
                    f"nb_grad({scalar_expr})[1]",
                    f"nb_grad({scalar_expr})[2]",
                    f"nb_grad({scalar_expr})[3]",
                )

            elif m.group(7):    # laplacian(X) → scalar output ✗ for vector context
                raise TypeError(_t_be("vector.operand_is_scalar", self.lang,
                                      operand=rf"\nabla^2 {m.group(8).strip()}"))

        # --- Last resort: derive component names from bare field name ---
        # (field not in SymbolContext — assume it is a vector field)
        # Raise if the name looks like a coordinate variable (typo guard).
        all_coord_vars: set[str] = set()
        for vs in _COORD_VARS.values():
            all_coord_vars.update(vs)
        if raw in all_coord_vars:
            raise TypeError(_t_be("vector.coordinate_name_as_field", self.lang, name=raw))
        sfxs = _COORD_COMPONENTS.get(coord_sys, ("x", "y", "z"))
        # **`vec(...)` はこの系がベクトルを包む綴りであって、成分の名前ではない。**
        # 包みを外さずに接尾辞を付けると `vec(A)_x` という綴りが CAS へ渡り、
        # 生の `Maxima evaluation error` が研究者に返る（実測 2026-09-11。
        # `\vec{A}` と書いて回転を求めるだけで踏む）。
        inner = unwrap_vector_mark(raw)
        if inner is not None:
            # **適用された場は、成分に何を渡すかが書かれていない。**
            # `\vec A(t)` の成分を `A_x(t)` と読むと、座標で微分する ∇× と ∇· は
            # **0 を返す**（実測、Maxima）——研究者がベクトルポテンシャルのつもりで
            # 書いた式に、静止した答えが `completed` で返ることになる。座標を
            # 補うのは物理の選択（CLAUDE.md「問題依存の物理の選択を焼き込まない」）
            # なので、ここは推測せずに断り、**実測で通る道**を名指す。
            _m_applied = self._APPLIED_NAME_RE.fullmatch(inner)
            if _m_applied:
                from notebook.runtime.i18n import t as _t_be
                raise TypeError(_t_be("vector.applied_field_components", self.lang,
                                      name=_m_applied.group(1),
                                      args=_m_applied.group(2)))
            return self._build_vec_components(inner, coord_sys, expr_lookup, depth + 1)
        # 名前でないものに成分名を**発明しない**（VEC-BINARY-OPERAND-WHITESPACE）。
        # 読み方と測定は `vector_operand` に在る——ノルムが既に答えていた問い。
        if not is_bare_name(raw):
            _low = self.to_backend_code(raw)
            if _low.ok and _low.code.strip() != raw.strip():
                return self._build_vec_components(_low.code, coord_sys, expr_lookup, depth + 1)
            _vo = value_lookup(self._context, expr_lookup.get)
            _entries = three_entries(raw) or expression_components(
                raw, self._context, _vo, to_cas=self.vector_expression_code)
            if _entries is not None:
                return _entries
            raise TypeError(_t_be(not_a_vector_reason(raw, self._context, _vo), self.lang,
                                  operand=raw.strip()))
        return (f"{raw}_{sfxs[0]}", f"{raw}_{sfxs[1]}", f"{raw}_{sfxs[2]}")

    def vector_expression_code(self, text: str) -> "str | None":
        r"""*text* の CAS 綴り——**外積は `nb_cross` の呼び出しになる**。

        Maxima に外積は無く、生成器が書く `((v) ~ (w))` はリストの上でも
        **評価できない**（実測 2026-09-18、生の `Maxima evaluation error`）ので、
        この系は毎セッション `nb_cross` を定義している。置き換える場所は
        **構文木**が決める（`vector_operand.cross_products_as_calls`）——文字列の
        上では `v ~ w + r` が `nb_cross(v, w) + r` か `nb_cross(v, w + r)` か
        決まらない。生成器そのものを変えない理由と、一般の CAS 操作で外積が
        生の CAS エラーになる件は CLAUDE.md と open_items
        CROSS-PRODUCT-IN-GENERIC-OP に在る。
        """
        from notebook.runtime.shared_parser import UnknownNotationError
        try:
            tree = self.parse_ir(text)
            if tree is None:
                return None
            return self.generate_backend_code(cross_products_as_calls(tree, "nb_cross"))
        except (UnknownNotationError, ValueError, AttributeError, TypeError):
            return None

    def vector_cas_view(self, text: str, value_of) -> "str | None":
        r"""外積を含む式を、CAS が評価できる綴りへ——`nb_cross` × 成分のリスト。

        Maxima の `~` は評価できない（実測: リストでも原子でも生の
        `Maxima evaluation error`）ので、外積を含む式に一般の操作をかけると
        `簡約`・`展開`・`因数分解`・`微分`・`積分`・`代入` がすべて生の CAS エラーを
        返していた（実測 2026-09-18、日英とも、H-1 の値でも宣言された場でも）。
        ∇ の3操作だけが通っていたのは、そちらが成分の読みを通るからである。

        **2つそろって初めて正しい**: 構文木から `nb_cross` へ書き換え
        （`vector_expression_code`）、そして**ベクトルを成分のリストへ開く**
        （`listify`）。開かずに当てると、宣言された場の `nb_cross(A, B)` が
        `A[2]*B[3]-…` を返す——研究者が書いていない名前で、生のエラーを
        silent-wrong と取り替えることになる（実測）。開けなければ None を返し、
        呼び出し側は研究者の綴りのままにする。
        """
        if "~" not in (text or ""):
            return None
        code = self.vector_expression_code(text)
        if not code or "~" in code:
            return None
        return listify(code, self._context, value_of)

    def _build_scalar_expr(
        self,
        expr: str,
        coord_sys: str,
        expr_lookup: dict,
        depth: int = 0,
    ) -> str:
        """Resolve *expr* to a Maxima scalar expression.

        Raises
        ------
        TypeError
            When the expression has vector type but a scalar is required
            (e.g. ``gradient(curl(A))`` — curl returns a vector).
        RecursionError
            When nesting depth exceeds 8.

        Supports:
        - Bare scalar-field name registered in SymbolContext
        - Bare name not in SymbolContext (pass through as-is)
        - ``divergence(X)`` / ``div(X)``  (scalar result)
        - ``laplacian(X)``                (scalar result)
        """
        from notebook.runtime.i18n import t as _t_be
        if depth > 8:
            raise RecursionError(_t_be("vector.nesting_too_deep", self.lang))

        # 値は右辺にある——ベクトル側（`_build_vec_components`）が NORM-OPERATION で
        # 入れた読み方。こちらに無く、`h = div(A)` の勾配が等式のまま `nb_grad` に渡り
        # `[0 = 0, 0 = 0, 0 = 0]` を返した（NESTED-VEC-OP-LATEX で型推論を直して露出）。
        raw = strip_lhs(expr_lookup.get(expr, expr))

        # SymbolContext: vector_field → type error
        if self._context:
            qty = self._context.get_quantity(raw)
            if qty is not None:
                if qty.kind == "vector_field":
                    raise TypeError(_t_be("vector.operand_is_vector", self.lang, operand=raw))
                if qty.kind in ("scalar_field", "scalar", "function"):
                    return raw

        # Nested op
        m = self._VEC_OP_RE.match(raw.strip())
        if m:
            if m.group(1):      # curl(X) → vector ✗
                raise TypeError(_t_be("vector.operand_is_vector", self.lang,
                                      operand=rf"\nabla \times {m.group(2).strip()}"))
            elif m.group(3):    # divergence(X) → scalar ✓
                inner = m.group(4).strip()
                inner_comps = self._build_vec_components(inner, coord_sys, expr_lookup, depth + 1)
                c1, c2, c3 = inner_comps
                return f"nb_div({c1},{c2},{c3})"
            elif m.group(5):    # gradient(X) → vector ✗
                raise TypeError(_t_be("vector.operand_is_vector", self.lang,
                                      operand=rf"\nabla {m.group(6).strip()}"))
            elif m.group(7):    # laplacian(X) → scalar ✓
                inner = m.group(8).strip()
                inner_scalar = self._build_scalar_expr(inner, coord_sys, expr_lookup, depth + 1)
                return f"nb_laplacian_scalar({inner_scalar})"

        # Fall through: treat as scalar name
        return raw

    # -- Class-level helpers (used by engine for Maxima-only paths) ----------

    def re_evaluate_expr(
        self,
        canonical: str,
        substitutions: "dict[str, str]",
        expr_lookup: "dict[str, str]",
        adapter: "object | None" = None,
        sympy_canonical: "str | None" = None,
    ) -> "str | None":
        """Re-evaluate *canonical* using Maxima's subst() function.

        Strategy: build ``subst([lhs=rhs, ...], canonical)`` and execute it.

        Example:
            canonical = "H_0 + H_1"
            substitutions = {"H_0": "p^2/(2*m)", "H_1": "-k*(x-a)^2"}
            → subst([H_0=p^2/(2*m), H_1=-k*(x-a)^2], H_0 + H_1)
            → p^2/(2*m) - k*(x-a)^2

        Returns:
            New canonical string, or None if adapter is unavailable or errors.
        """
        if adapter is None or not getattr(adapter, "is_running", False):
            return None
        if not substitutions:
            return canonical

        # Build subst list: [lhs=rhs, ...]
        pairs = ", ".join(
            f"{lhs}={rhs}" for lhs, rhs in substitutions.items()
        )
        cmd = f"subst([{pairs}], {canonical})"
        result = adapter.execute(cmd)
        if not result.ok:
            return None
        return result.value.strip() or None

    def uneval_diff_match(self, canonical: str) -> "re.Match[str] | None":
        return _UNEVAL_DIFF_RE.match(canonical)

    @staticmethod
    def uneval_diff_re() -> re.Pattern:
        """Return the compiled regex for unevaluated diff canonicals."""
        return _UNEVAL_DIFF_RE

    @staticmethod
    def maxima_leibniz_for_diff(
        func_name: str, func_arg: str, diff_var: str, order: int
    ) -> str:
        """Reconstruct Maxima's tex1() output for ``'diff(func(arg), var, n)``."""
        if order == 1:
            frac = r"{{d}\over{d\," + diff_var + r"}}"
        else:
            n = str(order)
            frac = r"{{d^" + n + r"}\over{d\," + diff_var + r"^" + n + r"}}"
        return frac + r"\," + func_name + r"\left(" + func_arg + r"\right)"

    # -- BackendHandler implementation ---------------------------------------

    # ------------------------------------------------------------------
    # solve_equation / solve_system helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eq_str_for_solve(eid_or_str: str, expr_lookup: dict) -> str:
        """Return the full equation string for use inside Maxima solve().

        For expression-ID references, looks up expr_lookup which already
        contains the full "LHS = RHS" form for H-1 expressions (restored by
        the engine's H-1 reconstruction step in run_operation).  Falls back
        to the raw string for non-ID inputs.
        """
        s = eid_or_str.strip()
        if is_expr_ref(s) and s in expr_lookup:
            return expr_lookup[s]
        return s

    def is_free_solve_const(self, value: str) -> bool:
        """Maxima emits ``%r1``, ``%r2``, … for unconstrained unknowns in an
        under-determined solve result."""
        return bool(re.fullmatch(r"%r\d+", value.strip()))

    def laplace_definition_command(self, f: str, t: str, s: str) -> str:
        """Build the DEFINITION integral for the Laplace transform of an explicit
        function: ``∫₀^∞ f(t) e^{-st} dt`` (no builtin ``laplace``).

        Convergence needs ``s>0`` (and positivity of any symbolic parameter, e.g.
        ``k`` in ``e^{-kt}``), otherwise Maxima asks an interactive sign question
        that would hang the persistent adapter.  These are assumed for the call and
        forgotten immediately after — no global assumption leaks."""
        params = sorted({x for x in re.findall(r"[A-Za-z_]\w*", f)
                         if x not in {t, s, "exp", "e"} and x not in MATH_FUNC_NAMES})
        al = ", ".join([f"{s}>0"] + [f"{p}>0" for p in params])
        return (f"block([_r], assume({al}), "
                f"_r: integrate(({f})*exp(-({s})*({t})), {t}, 0, inf), "
                f"forget({al}), _r)")

    @staticmethod
    def parse_solve_result(result_str: str) -> "list[tuple[str, str]]":
        """Parse Maxima's solve() result into (var, value) pairs.

        Handles a single solution set and **multiple solution sets** (a system
        with several roots / fixed points returns a list of sets):
          [x = 2]                          → [("x", "2")]
          [[x = 2, y = 1]]                 → [("x","2"), ("y","1")]
          [[x=0,y=0],[x=c/d,y=a/b]]        → [("x","0"),("y","0"),("x","c/d"),("y","a/b")]
          []                               → []
        """
        def _split_top(s: str) -> "list[str]":
            """Split on top-level commas, respecting () [] {} and skipping the
            Maxima thin-space token ``\\,`` so it is never a separator."""
            depth = 0
            parts: list[str] = []
            cur: list[str] = []
            i = 0
            while i < len(s):
                ch = s[i]
                if ch == "\\" and i + 1 < len(s) and s[i + 1] == ",":
                    cur.append(ch); cur.append(s[i + 1]); i += 2; continue
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                if ch == "," and depth == 0:
                    parts.append("".join(cur).strip()); cur = []
                else:
                    cur.append(ch)
                i += 1
            if "".join(cur).strip():
                parts.append("".join(cur).strip())
            return [p for p in parts if p]

        s = result_str.strip()
        if not s:
            return []
        # Strip exactly ONE outer bracket layer, then split into top-level parts.
        body = s[1:-1].strip() if (s.startswith("[") and s.endswith("]")) else s
        parts = _split_top(body)
        # Multiple solution sets: every part is itself a bracketed set → recurse.
        if parts and all(p.startswith("[") and p.endswith("]") for p in parts):
            pairs: list[tuple[str, str]] = []
            for p in parts:
                pairs.extend(MaximaBackendHandler.parse_solve_result(p))
            return pairs
        # Single set: parts are ``var = val`` items.
        pairs = []
        for part in parts:
            part = part.strip().lstrip("[").rstrip("]")
            # CAS OUTPUT, not researcher text: Maxima writes `var = val`, so the
            # `=` is always leftmost and the naive split was never wrong here
            # (measured 2026-08-13).  Routed through the one definition anyway —
            # a site that is safe only by a property of the CURRENT output format
            # is the shape this whole ratchet exists to stop.
            _rel = split_relation(part)
            if _rel is not None and "=" in _rel[1]:
                pairs.append((_rel[0], _rel[2]))
        return pairs

    def _prepare_expr_lookup(self, args: dict, expr_lookup: dict, ctx) -> dict:
        """Turn the stored canonicals into what the builders below consume:
        BACKEND CODE (docs/32 Phase 3).

        Two steps, and THE ORDER IS LOAD-BEARING:
          1. space-juxtaposition → explicit ``*`` (Maxima has no implicit
             multiplication; user expressions are stored with spaces);
          2. ``q_dot``/``q_ddot`` → the depends form ``'diff(q,t,N)``, outside
             Lagrangian mode where those are NOT independent atoms.

        Step 1 must precede step 2: a space in front of the noun form is
        mis-parsed downstream, so ``k 'diff(r,t,2)`` lowers to
        ``k' * diff(r, t, 2)`` — the quote is taken as a prime on ``k`` and the
        unevaluated derivative is silently lost.  With the explicit ``*`` in
        place (``k * 'diff(r,t,2)``) the IR round trip is a no-op on it.

        Neither patch is redundant with IR lowering, contrary to what the
        Phase 3 plan assumed (both measured 2026-07-30): step 2 is a SEMANTIC
        mapping the IR does not perform — lowering leaves ``theta_dot`` an atom.
        They stay.

        The IR lowering itself is NOT applied here.  It is applied by
        ``_build_substitute``, which is where a lowered argument meets the
        expression body inside one command; see that method for why a blanket
        lowering here is wrong.
        """
        # Replace <name>_dot / <name>_ddot symbols with diff(name(t),t,N) for
        # any time-dependent quantity in the context.  This ensures that
        # expressions entered with \dot{q} notation (stored as q_dot in the
        # canonical) are correctly recognized by Maxima when depends([q],[t])
        # is in effect — otherwise q_dot is an unrelated plain symbol.
        _time_dep_map: dict[str, str] = {}
        for _qty_name, _prop in ctx.registry.items():
            _qty = _prop.quantity
            if _qty.time_dependent:
                _t = _qty.time_parameter or "t"
                # Use Maxima depends form ('diff(q,t,N)) for consistency with
                # depends([q],[t]) declarations.  apply_time_dep_map uses the same
                # form so that T's derivatives and V's derivatives are the same atom.
                _time_dep_map[f"{_qty_name}_dot"]  = time_dep_derivative(_qty_name, _t, 1)
                _time_dep_map[f"{_qty_name}_ddot"] = time_dep_derivative(_qty_name, _t, 2)

        # Maxima has NO implicit multiplication: a space-juxtaposed canonical such
        # as "k r_ddot" or "m r^2 theta_dot" (how user expressions are stored) is
        # not parsed as a product and returns garbage when used verbatim as an
        # operation source.  Promote a space that sits BETWEEN two operands (a
        # juxtaposition-multiplication) to an explicit *.  Only the UNAMBIGUOUS
        # cases are touched:
        #   • operand-ending char (\w, ), ]) ↦ space ↦ symbol / number / 'diff noun
        #     ("k r_ddot", "2 x", "k 'diff(r,t,2)", "f(x) g")  → product
        #   • closing bracket (), ]) ↦ space ↦ "(" ("(a+b) (c+d)")  → product
        # A space between a SYMBOL and "(" ("f (x)") is left ALONE: whether it is a
        # product (f·(x)) or a function call (f(x)) cannot be decided without the
        # SymbolContext declarations, so that ambiguity is the SharedParser's job,
        # not this string patch.  Whitespace around a binary operator (a + b,
        # y = a*x) and subscripts (H_0) are likewise untouched, so — unlike a
        # SharedParser round-trip — this cannot corrupt H_0 into H(0).  Mode- and
        # operation-independent: fixes differentiate and substitute in both
        # Lagrangian and Newtonian sessions.
        def _ensure_kv(_k: str, _v) -> object:
            if _k in _IDENTIFIER_KEYS or not isinstance(_v, str):
                return _v
            return self.ensure_explicit_mult(_v)
        expr_lookup = {k: _ensure_kv(k, v) for k, v in expr_lookup.items()}

        # In analytical-mechanics (Lagrangian) mode q_dot/q_ddot are INDEPENDENT
        # atoms — do NOT collapse them to the function-derivative form here.
        if _time_dep_map and not self.is_lagrangian_mode():
            # Sort longest first to avoid partial replacement (e.g. q_ddot before q_dot).
            _sorted_keys = sorted(_time_dep_map, key=len, reverse=True)
            def _replace_dot_symbols(s: str) -> str:
                for _k in _sorted_keys:
                    s = re.sub(
                        r'(?<![A-Za-z0-9_])' + re.escape(_k) + r'(?![A-Za-z0-9_])',
                        _time_dep_map[_k], s
                    )
                return s
            expr_lookup = {k: _replace_dot_symbols(v) for k, v in expr_lookup.items()}

        return expr_lookup

    #: `\sum_{i=nbSumLo_{3}}^{nbSumHi_{3}}` — the placeholder range given to a
    #: sum the researcher wrote WITHOUT one.  Stripped for display: the range is
    #: not something they wrote, and showing it would put CAS bookkeeping in
    #: their notation (Core Principle #1).  It stays in the canonical, which is
    #: what the CAS returned and what the next command has to send back.
    _SUM_PLACEHOLDER_RANGE_RE = re.compile(
        r"=\s*nbSumLo_?\{?(\d+)\}?\s*\}\s*\^\s*\{\s*nbSumHi_?\{?\1\}?\s*\}")

    def strip_placeholder_sum_range(self, latex: str) -> str:
        r"""`\sum_{i=nbSumLo_{3}}^{nbSumHi_{3}}` → `\sum_{i}`.

        Paired on the NUMBER (the backreference), so two different placeholder
        ranges in one expression cannot be collapsed into each other.
        """
        return self._SUM_PLACEHOLDER_RANGE_RE.sub("}", latex or "")

    #: ベクトル解析の族——`build_command` の分岐がそのまま一覧である。
    _VECTOR_FAMILY = ("norm", "divergence", "curl", "gradient",
                      "scalar_laplacian", "dot_product", "cross_product")

    #: **宣言された対象を自分で扱う操作**（種別ごと。実測 7操作×4種別、理由と
    #: 測り方は `operand_guards.reject_declared_object_without_value`）。
    #:
    #: 「読む」だけでなく「**型エラーで断る**」も入れる——`curl` にスカラー場を
    #: 渡すと「'f' はスカラー場ですが、ベクトル場が必要です」という**より正確な
    #: 理由**を返すので、手前の guard がそれを「値がありません」に置き換えては
    #: いけない（スイートが落ちて気づいた、実測 2026-09-11）。値を与えても
    #: `curl` はスカラー場に対して正しくないので、条件は値ではない。
    #:
    #: **行列・テンソル・ベクトルには1つも無い**——`gradient`/`scalar_laplacian`
    #: はそこで型エラーも出さず `[0,0,0]`/`0` を返すので、guard が答える。
    _DECLARATION_HANDLED_BY_KIND = {
        "vector_field": _VECTOR_FAMILY,
        "scalar_field": _VECTOR_FAMILY + ("differentiate",),
    }

    #: そのうち**実際に計算できる**もの。断り文が案内するのはこちらである——
    #: 免除の一覧をそのまま見せると、`\nabla f` をベクトル場に勧めることになり、
    #: 「案内した回避策も効くか測る」に反する（実測 2026-09-11）。
    #: `differentiate` はスカラー場でだけ正直（`depends` つきで宣言されるので
    #: `'diff(f,x,1)`）。積分（`f*x`）にも総和（`2*f`）にも効かない。
    _DECLARATION_COMPUTING_BY_KIND = {
        "vector_field": ("norm", "divergence", "curl",
                         "dot_product", "cross_product"),
        "scalar_field": ("gradient", "scalar_laplacian", "differentiate"),
    }

    def declaration_reading_ops(self, kind=None, algebra=None):
        return self._DECLARATION_HANDLED_BY_KIND.get(kind, ())

    def declaration_computing_ops(self, kind=None, algebra=None):
        return self._DECLARATION_COMPUTING_BY_KIND.get(kind, ())

    def reads_declared_object(self, operation: str, kind=None, algebra=None) -> bool:
        return operation in self.declaration_reading_ops(kind, algebra)

    def build_command(self, operation: str, args: dict, expr_lookup: dict) -> BuildResult:
        from notebook.models.ir import CasOperation
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver


        parser = SharedParser()
        resolver = MaximaSemanticResolver()
        ctx = self._context or SymbolContext()

        try:
            expr_lookup = self._prepare_expr_lookup(args, expr_lookup, ctx)
        except LoweringRefused as _exc:
            from notebook.runtime.i18n import t as _t_be
            return BuildResult(
                ok=False, command="",
                error=_t_be("engine.cannot_lower_expression", self.lang,
                            expr_id=_exc.expr_id, reason=_exc.reason))

# solve_equation: solve({equation}, {var})
        if operation == 'solve_equation':
            return self._build_solve_equation(args, expr_lookup)

        # solve_ode_general: ode2(eq, depvar, indepvar) → the GENERAL solution of a
        # linear ODE, carrying its arbitrary constants (FREE-PARAM).  Maxima emits
        # %k1,%k2 (2nd order) or %c (1st order); they are renamed IN-CAS to the
        # engine-chosen free-parameter names so the output is already clean (no
        # %-artifacts to patch in latex).  Symbolic coefficients make ode2 ask
        # "<sym> zero or nonzero?" interactively — which would HANG the persistent
        # adapter — so every symbolic coefficient is assumed positive (physics
        # convention) around the call and forgotten immediately after, leaking no
        # global assumption.
        if operation == 'solve_ode_general':
            return self._build_solve_ode_general(args, expr_lookup, ctx)

        # solve_system: solve([eq1, eq2, ...], [var1, var2, ...])
        if operation == 'solve_system':
            return self._build_solve_system(args, expr_lookup)

        # build_inverse_transform: invert a linear basis change new = M·old by solving
        # the forward system for the OLD basis.  The new-basis symbols are bound to
        # their RHS in Maxima (H-1 P2 auto-substitute), which would turn each equation
        # into a 0=0 tautology; localizing them in a block() unbinds them so solve()
        # sees a proper N×N system.
        if operation == 'build_inverse_transform':
            return self._build_inverse_transform(args, expr_lookup)

# derive_metric_constraints: the non-operator metric-preservation condition
        # M G Mᵀ = G.  Build M by coeff-extracting each forward RHS over the old
        # basis, form the Gram matrix G from declared inner products, and return the
        # symmetric matrix ratsimp(M G Mᵀ − G); its non-zero entries are the
        # constraints (e.g. orthogonality cos²+sin²=1 when G = I).
        if operation == 'derive_metric_constraints':
            return self._build_derive_metric_constraints(args, expr_lookup)

# apply_canonical_constraints: reduce an expression under polynomial side
        # relations (e.g. u^2+v^2 = 1) so the constraint actually folds the result
        # (the Bogoliubov inverse denominator u^2+v^2 collapses to 1).  Plain subst
        # cannot use a polynomial relation; ratsubst is the right tool.
        if operation == 'apply_canonical_constraints':
            return self._build_apply_canonical_constraints(args, expr_lookup)

# extract_order: extract coefficient of var^n
        if operation == 'extract_order':
            return self._build_extract_order(args, expr_lookup)

        # truncate_series: keep terms up to var^order via taylor
        if operation == 'truncate_series':
            return self._build_truncate_series(args, expr_lookup)

        # declare_symmetry: tell Maxima about index symmetry of a 2-argument function.
        # Maxima supports: antisymmetric(f), symmetric(f), hermitian is not built-in.
        if operation == 'declare_symmetry':
            return self._build_declare_symmetry(args)

        # change_integration_variables (D-6, docs/open_items.md): multivariable change
        # of variables with the Jacobian determinant automatically inserted.  Produces
        # the new integrand f(transform) * det(jacobian(transform, new_vars)) which the
        # user then integrates over the new variables.  MUST be dispatched here, before
        # the generic per-arg loop below — that loop calls parser.parse() on each raw,
        # UNSPLIT string arg, and a `transform` value with 2+ variables written as full
        # equations (e.g. "x = r*cos(theta), y = r*sin(theta)") trips the chained-
        # relation guard in shared_parser.py before this operation's own comma-aware
        # splitting (_build_change_integration_variables's _split_top) ever runs.
        if operation == 'change_integration_variables':
            return self._build_change_integration_variables(args, expr_lookup, parser, resolver, ctx)


        # pure_nash_equilibria / mixed_strategy_indifference_eq (F-17): both
        # parse the payoff matrices' own canonical text structurally (row/
        # column counts) rather than relying on the generic per-arg loop.
        if operation == 'pure_nash_equilibria':
            return self._build_pure_nash_equilibria(args, expr_lookup)

        # convolve (F-8): substitutes into f/g via raw Maxima subst() calls
        # built from their own resolved canonical text.

        ir_args: dict = {}

        for k, v in args.items():
            if k in _IDENTIFIER_KEYS:
                # Pass identifier fields through as-is without parsing.
                ir_args[k] = v
            elif isinstance(v, str):
                raw_expr = (
                    expr_lookup[v]
                    if is_expr_ref(v) and v in expr_lookup
                    else v
                )
                # BP-9: an argument may arrive as a derivative DISPLAY form
                # (\frac{\partial phi}{\partial x}) — from an lhs_alias_map key,
                # a confirmation-dialog candidate or the LLM.  Parsed as a
                # fraction it either raises here or, worse, cancels silently to
                # phi/x.  Restoring it to 'diff(...) at the single place every
                # operation's string args pass through covers them all at once.
                raw_expr = self.restore_derivative_display_latex(raw_expr)
                surface_ast = parser.parse(raw_expr)
                # An H-1 source is reconstructed as "LHS = RHS".  Matrix/value
                # operations act on the value, so operate on the RHS rather than
                # feeding Maxima an equation (eigenvalues(A=matrix(...)) → error).
                # Only these ops strip — operations that legitimately consume an
                # equation (e.g. use_relation's relation_ref) keep the full form.
                from notebook.models.ir import Equation as _Equation
                if operation in _MATRIX_VALUE_OPS and isinstance(surface_ast, _Equation):
                    surface_ast = surface_ast.rhs
                if surface_ast is not None:
                    ir_args[k] = resolver.resolve(surface_ast, ctx)
                else:
                    ir_args[k] = v
            elif isinstance(v, list):
                ir_args[k] = []
                for item in v:
                    if isinstance(item, str):
                        raw_expr = (
                            expr_lookup[item]
                            if is_expr_ref(item) and item in expr_lookup
                            else item
                        )
                        surface_ast = parser.parse(raw_expr)
                        if surface_ast is not None:
                            ir_args[k].append(resolver.resolve(surface_ast, ctx))
                        else:
                            ir_args[k].append(item)
                    else:
                        ir_args[k].append(item)
            else:
                ir_args[k] = v

        # define_vector_field: declare component functions via depends()
        if operation == 'define_vector_field':
            return self._build_define_vector_field(args)

        # define_scalar_field: depends() for a scalar function
        if operation == 'define_scalar_field':
            return self._build_define_scalar_field(args)

        # define_coordinate_system: generate nb_* functions from Lamé coefficients
        if operation == 'define_coordinate_system':
            return self._build_define_coordinate_system(args)

        # cross_product / dot_product: coordinate-independent vector operations
        # poisson_bracket: {f,g} = Σ_i (∂f/∂q_i ∂g/∂p_i − ∂f/∂p_i ∂g/∂q_i) over the
        # session's declared canonical pairs (q_i, p_i).  Backend-native: built from
        # diff() and a sum, no special non-commutative algebra needed.
        if operation == 'poisson_bracket':
            return self._build_poisson_bracket(args, expr_lookup, ctx)

        if operation in ('cross_product', 'dot_product'):
            return self._build_cross_dot_product(operation, args, expr_lookup)

        if operation == 'norm':
            return self._build_norm(args, expr_lookup)

        # divergence / curl / gradient / scalar_laplacian
        # Uses _build_vec_components / _build_scalar_expr for recursive resolution,
        # so nested expressions like curl(curl(A)) or div(grad(f)) work without
        # registering intermediate results in SymbolContext.
        # TypeError / RecursionError from the resolvers are caught here and
        # returned as BuildResult(ok=False) with a user-facing message.
        if operation in ('divergence', 'curl', 'gradient', 'scalar_laplacian'):
            return self._build_vector_calculus(operation, args, expr_lookup)

        # change_variable: substitution in an integral (u-substitution)
        # Uses Maxima's changevar(expr, new_var - old_expr, new_var, old_var)
        # substitute: take expr verbatim from expr_lookup to avoid mangling
        # Maxima-specific notation ('diff, q(t), noun forms) through the
        # SharedParser → CodeGenerator round-trip.
        # lhs / rhs are still parsed through IR so that LaTeX inputs like
        # "\sin(q)" are correctly converted to Maxima form "sin(q)".
        if operation == 'substitute':
            _r = self._build_substitute(args, expr_lookup, parser, resolver, ctx)
            if _r is not None:
                return _r

        if operation == 'change_variable':
            return self._build_change_variable(args, expr_lookup, parser, resolver, ctx)

        # expand_neighbor_sum (F-1): 1D-chain nearest-neighbor bond sum.
        # The body (a scalar lattice term such as -J*s_i*s_j or a Coulomb/exchange
        # term in lattice-energy / quantum-chemistry work) is expanded into the
        # explicit per-site sum, then resolved with the Maxima generator.
        if operation == 'expand_neighbor_sum':
            return self._build_expand_neighbor_sum(args, parser, resolver, ctx)

        # differentiate w.r.t. a time-dependent quantity's coordinate:
        # When var=q and q is declared time-dependent, direct diff(L, q) triggers
        # Maxima's chain rule and produces spurious diff(q(t),q) terms from q̇=diff(q(t),t,1).
        # Solution: auto-wrap in block() that temporarily substitutes q̇→_nb_v_tmp and
        # q(t)→_nb_q_tmp, differentiates w.r.t. the plain temp var, then restores q(t).
        # This is transparent to the user — no manual substitute steps needed.
        if operation == 'differentiate':
            _r = self._build_differentiate(args, expr_lookup, ctx)
            if _r is not None:
                return _r

        # relabel_index: rename dummy summation index (no Jacobian)
        # For symmetric domains (e.g. BZ), Σ_{old} f(old) = Σ_{old} f(new) after
        # relabeling, so this is a direct subst with no extra factors.
        if operation == 'relabel_index':
            return self._build_relabel_index(args, expr_lookup)

        # MATMUL-TEMPLATE-STAR: the operand-joining templates spell `*`, which is
        # ELEMENTWISE on matrices.  Reason and measurement: `product_operator`.
        _tpl_op = template_product_operator(operation, ir_args, args.get("op"), ctx)
        if _tpl_op is not None:
            ir_args["op"] = _tpl_op

        build = self._builder.build(CasOperation(operation=operation, args=ir_args))
        if not build.ok:
            return BuildResult(ok=False, command="", error=build.error)

        command = build.command
        if operation == "define_function":
            command = re.sub(r'(?<!:)=', ':=', command, count=1)
        return BuildResult(ok=True, command=command)

    # ------------------------------------------------------------------
    # Per-operation builders (split from the build_command if-chain,
    # refactor Stage 4, 2026-07-04 — mirrors JuliaBackendHandler's
    # _build_* method structure).  Bodies are verbatim moves.
    # ------------------------------------------------------------------

    # solve_equation: solve({equation}, {var})
    def _solve_operand_to_backend(self, text: str) -> "tuple[str | None, str]":
        """Lower one solve operand, or explain why it cannot be lowered.

        ``solve_equation`` is dispatched BEFORE ``build_command``'s per-argument
        loop, so — unlike every other operation — its operands never met
        ``to_backend_code``.  Measured 2026-08-11: solving an equation whose
        canonical carries a bounded sum sent Maxima the LaTeX itself,

            solve(S = x \\sum_{i=1}^3*i^2, x)

        which it cannot read, so the adapter's own scratch name ``_res_`` came
        back and was REGISTERED as the result — a success card holding a
        variable name from the CAS bridge.

        BOTH operands go through the door, never one: lowering rewrites indexed
        names (``a_i`` → ``a(i)``, ``omega_0`` → ``omega(0)``), so an equation
        lowered against a raw variable would be solved for a symbol that no
        longer appears in it.  That asymmetry is the trap CLAUDE.md records for
        the other pre-loop builders ("lowering one side broke them").
        """
        from notebook.runtime.shared_parser import UnknownNotationError
        try:
            result = self.to_backend_code(text)
        except UnknownNotationError as exc:
            return None, getattr(exc, "i18n_key", str(exc))
        if result.status == "refused":
            return None, result.reason or ""
        return result.code, ""

    def _build_solve_equation(self, args, expr_lookup) -> BuildResult:
        eq_str  = self._eq_str_for_solve(str(args.get("equation", "")), expr_lookup)
        var_str = str(args.get("var", ""))
        if not eq_str or not var_str:
            return BuildResult(ok=False, command="",
                               error="solve_equation: missing equation or var")
        # The variable arrives as the researcher WROTE it — `G^{(HF)}_{n\sigma}
        # (\omega)`, not a canonical — while the equation arrives as a canonical
        # (that is what an Expression stores).  So the variable goes through the
        # LaTeX door FIRST, and both then go through the lowering door together.
        # The NL layer used to do this half with three regexes of its own, which
        # merged `G_{n\sigma}` into the single name `G_{nsigma}`: a variable that
        # appears nowhere in the equation, for which Maxima correctly reports
        # "no solution" (measured 2026-08-21).  `latex_to_canonical` is
        # idempotent on the plain spellings (`x`, `q_ddot`, `Az_z`), so a
        # variable that was already canonical is unchanged.
        var_str = self.latex_to_canonical(var_str) or var_str
        eq_str, _eq_why = self._solve_operand_to_backend(eq_str)
        var_str, _var_why = self._solve_operand_to_backend(var_str)
        if eq_str is None or var_str is None:
            return BuildResult(ok=False, command="",
                               error=f"solve_equation: {_eq_why or _var_why}")
        # Apply time-dep map so q_ddot → 'diff(q,t,2) when solving for derivatives.
        var_str = self.apply_time_dep_map(var_str)
        # Maxima's solve() cannot use unevaluated derivatives ('diff(q,t,N))
        # as solve variables — it returns [].  Work around by introducing a
        # temporary scalar variable _nb_dv, substituting it in for the
        # derivative, solving for _nb_dv, then substituting back.
        # Also handle the function form 'diff(q(t),t,N) — Maxima-output
        # expressions use q(t) as a function, not the depends form q.
        # If the equation contains 'diff(q(t),t,N) but var_str is the depends
        # form 'diff(q,t,N), upgrade var_str to the function form so subst
        # finds the right atom.
        if var_str.startswith("'diff("):
            _m_dep = re.match(r"'diff\((\w+),(\w+),(\d+)\)", var_str)
            if _m_dep:
                _func_form = f"'diff({_m_dep.group(1)}({_m_dep.group(2)}),{_m_dep.group(2)},{_m_dep.group(3)})"
                if _func_form in eq_str:
                    var_str = _func_form
        # Apply the same map to the equation for the depends form case.
        eq_str_dep = self.apply_time_dep_map(eq_str)
        if var_str.startswith("'diff(") or (
            var_str.startswith("diff(") and not var_str.startswith("diff_")
        ):
            return BuildResult(ok=True, command=shadow_own_definition(self._context,
                str(args.get("equation", "")), eq_str,
                f"block([_nb_dv],"
                f" _nb_sol: solve(subst({var_str} = _nb_dv, {eq_str_dep}), _nb_dv),"
                f" subst(_nb_dv = {var_str}, _nb_sol))"))
        return BuildResult(ok=True, command=shadow_own_definition(self._context,
            str(args.get("equation", "")), eq_str, f"solve({eq_str}, {var_str})"))

    # solve_ode_general: ode2(eq, depvar, indepvar) → the GENERAL solution of a
    # linear ODE, carrying its arbitrary constants (FREE-PARAM).  Maxima emits
    # %k1,%k2 (2nd order) or %c (1st order); they are renamed IN-CAS to the
    # engine-chosen free-parameter names so the output is already clean (no
    # %-artifacts to patch in latex).  Symbolic coefficients make ode2 ask
    # "<sym> zero or nonzero?" interactively — which would HANG the persistent
    # adapter — so every symbolic coefficient is assumed positive (physics
    # convention) around the call and forgotten immediately after, leaking no
    # global assumption.
    def _build_solve_ode_general(self, args, expr_lookup, ctx) -> BuildResult:
        eq_str = self._eq_str_for_solve(str(args.get("equation", "")), expr_lookup)
        eq_str = self.apply_time_dep_map(eq_str)
        dep = str(args.get("depvar", "")).strip()
        ind = str(args.get("indepvar", "t")).strip() or "t"
        if not eq_str or not dep:
            return BuildResult(ok=False, command="",
                               error="solve_ode_general: missing equation or depvar")
        names = args.get("const_names") or {}
        k1 = names.get("k1", "k_1"); k2 = names.get("k2", "k_2"); c0 = names.get("c", "C")
        renames = f"[%k1={k1}, %k2={k2}, %c={c0}]"
        _td = {n for n, p in ctx.registry.items()
               if p.quantity.time_dependent}
        _skip = {dep, ind, "diff"} | _td | set(MATH_FUNC_NAMES)
        params = sorted({s for s in re.findall(r"[A-Za-z_]\w*", eq_str)
                         if s not in _skip})
        assumes = "".join(f"assume({s}>0), " for s in params)
        forgets = "".join(f"forget({s}>0), " for s in params)
        return BuildResult(ok=True, command=(
            f"block([_nb_ode], {assumes}"
            f"_nb_ode: ode2({eq_str}, {dep}, {ind}), "
            f"{forgets}"
            f"subst({renames}, _nb_ode))"
        ))

    # solve_system: solve([eq1, eq2, ...], [var1, var2, ...])
    def _build_solve_system(self, args, expr_lookup) -> BuildResult:
        raw_eqs  = str(args.get("equations", ""))
        vars_str = str(args.get("vars", ""))
        if not raw_eqs or not vars_str:
            return BuildResult(ok=False, command="",
                               error="solve_system: missing equations or vars")
        eq_parts = [e.strip() for e in raw_eqs.split(",")]
        eq_strs  = [self._eq_str_for_solve(e, expr_lookup) for e in eq_parts]
        # Localize the solve-for variables.  A system routinely contains a
        # equation like `z = 2`, and H-1's P2 binds that symbol CAS-side — so
        # `solve([…], [x, y, z])` had `z` evaluate to 2 and Maxima refused the
        # whole call (measured 2026-08-16; the English spelling only appeared to
        # work because a second defect was replacing the variable name with an
        # expression id, and the two cancelled).  `block([…])` unbinds them for
        # this one call, which is the mechanism `_build_inverse_transform` below
        # already uses against the same P2 binding — it was simply never applied
        # here.  Shadowing an UNBOUND symbol is a no-op (measured: same answer),
        # so this needs no test for whether the binding exists.
        targets = [v.strip() for v in vars_str.split(",") if v.strip()]
        solve_cmd = f"solve([{', '.join(eq_strs)}], [{vars_str}])"
        if targets:
            solve_cmd = f"block([{', '.join(targets)}], {solve_cmd})"
        return BuildResult(ok=True, command=solve_cmd)

    # build_inverse_transform: invert a linear basis change new = M·old by solving
    # the forward system for the OLD basis.  The new-basis symbols are bound to
    # their RHS in Maxima (H-1 P2 auto-substitute), which would turn each equation
    # into a 0=0 tautology; localizing them in a block() unbinds them so solve()
    # sees a proper N×N system.
    def _build_inverse_transform(self, args, expr_lookup) -> BuildResult:
        raw_eqs  = str(args.get("equations", ""))
        vars_str = str(args.get("vars", ""))       # old basis (solve-for)
        new_basis = str(args.get("new_basis", "")).strip()  # LHS syms to localize
        if not raw_eqs or not vars_str:
            return BuildResult(ok=False, command="",
                               error="build_inverse_transform: missing equations or vars")
        eq_parts = [e.strip() for e in raw_eqs.split(",")]
        eq_strs  = [self._eq_str_for_solve(e, expr_lookup) for e in eq_parts]
        solve_cmd = f"solve([{', '.join(eq_strs)}], [{vars_str}])"
        if new_basis:
            return BuildResult(ok=True, command=f"block([{new_basis}], {solve_cmd})")
        return BuildResult(ok=True, command=solve_cmd)

    # derive_metric_constraints: the non-operator metric-preservation condition
    # M G Mᵀ = G.  Build M by coeff-extracting each forward RHS over the old
    # basis, form the Gram matrix G from declared inner products, and return the
    # symmetric matrix ratsimp(M G Mᵀ − G); its non-zero entries are the
    # constraints (e.g. orthogonality cos²+sin²=1 when G = I).
    def _build_derive_metric_constraints(self, args, expr_lookup) -> BuildResult:
        forward = str(args.get("forward", ""))
        basis   = str(args.get("basis", ""))
        gram    = str(args.get("gram", ""))
        if not forward or not basis or not gram:
            return BuildResult(ok=False, command="",
                               error="derive_metric_constraints: missing forward/basis/gram")
        bsyms = [b.strip() for b in basis.split(",") if b.strip()]
        rows = []
        for f in [x.strip() for x in forward.split(",") if x.strip()]:
            rhs = self._eq_str_for_solve(f, expr_lookup)
            # Goes into `coeff(rhs, b)` inside a matrix — a mis-split is CAS
            # CODE, not a display.  Measured 2026-08-13 with a forward equation
            # carrying a bounded sum on the left:
            #   matrix([coeff(1}^3*c_i = u*a + v*b, a), …])
            _rel = split_relation(rhs)
            if _rel is not None and "=" in _rel[1]:
                rhs = _rel[2]
            rows.append("[" + ", ".join(f"coeff({rhs}, {b})" for b in bsyms) + "]")
        mmat = "matrix(" + ", ".join(rows) + ")"
        gmat = "matrix(" + ", ".join(
            "[" + r.strip() + "]" for r in gram.split(";")) + ")"
        # trigsimp folds the Pythagorean identity so an orthonormal rotation
        # reduces to the zero matrix (no spurious cos²+sin²−1 constraint).
        cmd = (f"block([_M, _G], _M: {mmat}, _G: {gmat}, "
               f"trigsimp(ratsimp(_M . _G . transpose(_M) - _G)))")
        return BuildResult(ok=True, command=cmd)

    # apply_canonical_constraints: reduce an expression under polynomial side
    # relations (e.g. u^2+v^2 = 1) so the constraint actually folds the result
    # (the Bogoliubov inverse denominator u^2+v^2 collapses to 1).  Plain subst
    # cannot use a polynomial relation; ratsubst is the right tool.
    def _build_apply_canonical_constraints(self, args, expr_lookup) -> BuildResult:
        target = str(args.get("expr", ""))
        rels   = str(args.get("relations", ""))
        if not target or not rels:
            return BuildResult(ok=False, command="",
                               error="apply_canonical_constraints: missing expr or relations")
        cmd = self._eq_str_for_solve(target, expr_lookup)
        # Reduce the RHS only for an equation target; its LHS is bound to the RHS
        # by H-1 P2, so localize it in a block() to keep it symbolic (else Maxima
        # evaluates "F" and the equation degenerates to rhs = rhs).
        # `tgt_lhs` becomes a block-LOCAL SYMBOL and the relation's halves become
        # ratsubst's replacement and PATTERN.  Measured 2026-08-13, the naive
        # split produced both halves wrong at once:
        #   block([F], F = ratsubst(1}^3*c_i = u*a + v*b, \sum_{i, a^2 + b^2))
        # — a replacement that still carries an `=`, and raw LaTeX as the pattern.
        tgt_lhs = None
        _rel_t = split_relation(cmd)
        if _rel_t is not None and "=" in _rel_t[1]:
            tgt_lhs, cmd = _rel_t[0], _rel_t[2]

        # Everything that goes into the command as MATHEMATICS is lowered;
        # `tgt_lhs` is not, because it goes in as a NAME (the block-local symbol).
        # That split is the whole reason this builder could not simply be handed
        # to `to_backend_code` wholesale: CLAUDE.md records that widening the
        # lowering to every operation was measured wrong precisely because these
        # builders pair an operand with text that must stay raw.
        # Before this, the pattern reached Maxima as the researcher's own LaTeX
        # (`ratsubst(u*a + v*b, \sum_{i=1}^3*c_i, …)`), which Maxima cannot read
        # — open_items BUILDER-OPERAND-NOT-LOWERED.
        def _low(text: str) -> "str | None":
            _r = self.to_backend_code(text)
            return None if _r.status == "refused" else _r.code

        _c = _low(cmd)
        if _c is None:
            return BuildResult(ok=False, command="",
                               error="apply_canonical_constraints: cannot lower the target")
        cmd = _c
        for r in [x.strip() for x in rels.split(",") if x.strip()]:
            rel = self._eq_str_for_solve(r, expr_lookup)
            _rel_r = split_relation(rel)
            if _rel_r is None or "=" not in _rel_r[1]:
                continue
            _new, _pat = _low(_rel_r[2]), _low(_rel_r[0])
            if _new is None or _pat is None:
                return BuildResult(ok=False, command="",
                                   error="apply_canonical_constraints: cannot lower a relation")
            cmd = f"ratsubst({_new}, {_pat}, {cmd})"
        if tgt_lhs:
            # `block([X], …)` shadows an H-1 P2 binding so the equation does not
            # degenerate to `rhs = rhs`.  It can only shadow a SYMBOL — and the
            # target's left side is not always one.  Measured 2026-08-15:
            # `block([\sum_{i=1}^3*c_i], …)` came back as `_res_`, the adapter's
            # scratch name, because Maxima could not read the command at all.
            # Nothing is bound for a compound left side, so nothing needs
            # shadowing; it only has to be LOWERED like the rest of the maths.
            _lhs_low = _low(tgt_lhs)
            if _lhs_low is None:
                return BuildResult(ok=False, command="",
                                   error="apply_canonical_constraints: cannot lower the target's left side")
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", _lhs_low.strip()):
                cmd = f"block([{_lhs_low}], {_lhs_low} = {cmd})"
            else:
                cmd = f"{_lhs_low} = {cmd}"
        return BuildResult(ok=True, command=cmd)

    # extract_order: extract coefficient of var^n
    def _build_extract_order(self, args, expr_lookup) -> BuildResult:
        name  = str(args.get("expr", ""))
        var   = str(args.get("var", ""))
        order = int(args.get("order", 0))
        if not name or not var:
            return BuildResult(ok=False, command="",
                               error="extract_order: missing expr or var")
        raw = (expr_lookup[name] if is_expr_ref(name) and name in expr_lookup
               else name)
        return BuildResult(ok=True,
                           command=f"coeff(expand({raw}), {var}, {order})")

    # truncate_series: keep terms up to var^order via taylor
    def _build_truncate_series(self, args, expr_lookup) -> BuildResult:
        name  = str(args.get("expr", ""))
        var   = str(args.get("var", ""))
        order = int(args.get("order", 0))
        if not name or not var:
            return BuildResult(ok=False, command="",
                               error="truncate_series: missing expr or var")
        raw = (expr_lookup[name] if is_expr_ref(name) and name in expr_lookup
               else name)
        return BuildResult(ok=True,
                           command=f"expand(taylor({raw}, {var}, 0, {order}))")

    # declare_symmetry: tell Maxima about index symmetry of a 2-argument function.
    # Maxima supports: antisymmetric(f), symmetric(f); hermitian is not built-in
    # (symmetric is the closest available declaration).  Validation + dispatch
    # live in BackendHandler._build_declare_symmetry (base class).
    _SYMMETRY_COMMANDS = {
        "antisymmetric": "declare({name}, antisymmetric)",
        "symmetric":      "declare({name}, symmetric)",
        "hermitian":      "declare({name}, symmetric)",
    }

    #: 型（非スカラー）と対称性を**1つの declare で**宣言する形。Maxima の
    #: `declare` は (原子, 特性) の対を取るので `declare(A, nonscalar, symmetric)`
    #: は拒否され、`declare(A, nonscalar, A, symmetric)` が通る——実測 2026-09-05
    #: （前者は ok=False で、どちらの特性も立たなかった）。
    _SYMMETRY_COMMANDS_NONSCALAR = {
        "antisymmetric": "declare({name}, nonscalar, {name}, antisymmetric)",
        "symmetric":      "declare({name}, nonscalar, {name}, symmetric)",
        "hermitian":      "declare({name}, nonscalar, {name}, symmetric)",
    }

    # define_vector_field: declare component functions via depends()
    def _build_define_vector_field(self, args) -> BuildResult:
        field_name = str(args.get("name", "A"))
        coord_sys  = str(args.get("coordinate_system", "cartesian"))
        raw_vars   = args.get("natural_vars") or args.get("vars") or []
        if isinstance(raw_vars, str):
            raw_vars = [v.strip() for v in raw_vars.split(",") if v.strip()]
        if not raw_vars:
            raw_vars = list(_COORD_VARS.get(coord_sys, ("x", "y", "z")))
        cmd = self.build_define_vector_field(field_name, coord_sys, raw_vars)
        return BuildResult(ok=True, command=cmd)

    # define_scalar_field: depends() for a scalar function
    def _build_define_scalar_field(self, args) -> BuildResult:
        field_name = str(args.get("name", "f"))
        coord_sys  = str(args.get("coordinate_system", "cartesian"))
        raw_vars   = args.get("natural_vars") or args.get("vars") or []
        if isinstance(raw_vars, str):
            raw_vars = [v.strip() for v in raw_vars.split(",") if v.strip()]
        if not raw_vars:
            raw_vars = list(_COORD_VARS.get(coord_sys, ("x", "y", "z")))
        vars_str = "[" + ",".join(raw_vars) + "]"
        cmd = shadow_coordinate_variables(raw_vars, f"depends({field_name},{vars_str})")
        return BuildResult(ok=True, command=cmd)

    # define_coordinate_system: generate nb_* functions from Lamé coefficients
    def _build_define_coordinate_system(self, args) -> BuildResult:
        raw_vars = args.get("vars", "")
        raw_lame = args.get("lame", "")
        # Accept comma-separated strings or lists
        if isinstance(raw_vars, str):
            vars_list = [v.strip() for v in raw_vars.split(",") if v.strip()]
        else:
            vars_list = [str(v).strip() for v in raw_vars]
        if isinstance(raw_lame, str):
            lame_list = [h.strip() for h in raw_lame.split(",") if h.strip()]
        else:
            lame_list = [str(h).strip() for h in raw_lame]
        if len(vars_list) != 3 or len(lame_list) != 3:
            return BuildResult(
                ok=False, command="",
                error=(
                    "define_coordinate_system には座標変数 3 つとラメ係数 3 つが必要です。\n"
                    "define_coordinate_system requires exactly 3 coordinate variables "
                    "and 3 Lamé coefficients."
                ),
            )
        cmds = _make_nb_functions_from_lame(
            (vars_list[0], vars_list[1], vars_list[2]),
            (lame_list[0], lame_list[1], lame_list[2]),
        )
        self._nb_coord_vars = tuple(vars_list)
        cmds.extend(_NB_CROSS_DOT_FUNCTIONS)
        # Execute all commands; return the last one as the rendered_command
        # (the engine will store this for replay)
        combined = "\n".join(cmds)
        return BuildResult(ok=True, command=combined)

    # cross_product / dot_product: coordinate-independent vector operations
    # poisson_bracket: {f,g} = Σ_i (∂f/∂q_i ∂g/∂p_i − ∂f/∂p_i ∂g/∂q_i) over the
    # session's declared canonical pairs (q_i, p_i).  Backend-native: built from
    # diff() and a sum, no special non-commutative algebra needed.
    def _build_poisson_bracket(self, args, expr_lookup, ctx) -> BuildResult:
        pairs = list(ctx.canonical_pairs)
        if not pairs:
            return BuildResult(ok=False, command="", error="__no_canonical_pairs__")

        def _resolve_operand(_ref: str) -> str:
            # _ref may be an E-ID (→ expr_lookup canonical) or a literal expr.
            _v = expr_lookup.get(_ref, _ref)
            # H-1 equations reconstruct as "LHS = RHS"; the bracket acts on RHS.
            # This lands inside `diff((f), q)`, so a mis-split is CAS code:
            # measured 2026-08-13, `diff((1}^3*c_i = u*a + v*b),q)`.
            _rel = split_relation(_v)
            if _rel is not None and "=" in _rel[1]:
                _v = _rel[2]
            return _v.strip()

        f = _resolve_operand(str(args.get("a", "")))
        g = _resolve_operand(str(args.get("b", "")))
        if not f or not g:
            return BuildResult(ok=False, command="",
                               error="poisson_bracket requires 'a' and 'b' arguments")
        terms = [
            f"diff(({f}),{q})*diff(({g}),{p}) - diff(({f}),{p})*diff(({g}),{q})"
            for (q, p) in pairs
        ]
        return BuildResult(ok=True, command=f"ratsimp({' + '.join(terms)})")

    def _build_cross_dot_product(self, operation, args, expr_lookup) -> BuildResult:
        a_ref = str(args.get("a", ""))
        b_ref = str(args.get("b", ""))
        if not a_ref or not b_ref:
            return BuildResult(ok=False, command="",
                               error="cross_product/dot_product requires 'a' and 'b' arguments")
        coord_sys = str(args.get("coordinate_system") or "cartesian")
        try:
            a_comps = self._build_vec_components(a_ref, coord_sys, expr_lookup)
            b_comps = self._build_vec_components(b_ref, coord_sys, expr_lookup)
        except (TypeError, RecursionError) as exc:
            return BuildResult(ok=False, command="", error=str(exc))

        # ∇ で作ったオペランドだけを守る——内積自体は座標を束縛しない（`[x, y, z]` は
        # 研究者の束縛を受ける。命令全体を包むと値が変わる、実測）。
        a_list = shadow_differentiated_operand(self._nb_coord_vars, "[{},{},{}]".format(*a_comps))
        b_list = shadow_differentiated_operand(self._nb_coord_vars, "[{},{},{}]".format(*b_comps))
        if operation == "cross_product":
            cmd = f"nb_cross({a_list},{b_list})"
        else:
            cmd = f"nb_dot({a_list},{b_list})"
        return BuildResult(ok=True, command=cmd)

    #: 行列ノルムの種類 → Maxima の綴り。**既定は置かない**（理由と綴りの表は
    #: `i18n/nl_patterns.json` の `norm_kind`）。**Maxima 自身が `mat_norm(A,
    #: frobenius|1|inf)` を持ち**、具体でも記号でも値を返す（実測 2026-09-10）。
    #: スペクトルだけ `mat_norm` に無いので定義どおり √(AᵀA の最大固有値)。
    _MATRIX_NORM_COMMAND: dict[str, str] = {
        "frobenius":   "mat_norm({m}, frobenius)",
        "max_col_sum": "mat_norm({m}, 1)",
        "max_row_sum": "mat_norm({m}, inf)",
        "spectral":    "sqrt(lmax(first(eigenvalues(transpose({m}) . ({m})))))",
    }

    def _build_norm(self, args, expr_lookup) -> BuildResult:
        r"""`\|v\|` / `\|A\|` —— ベクトルは2ノルム、行列は**種類を書かせる**。

        非対称の理由（ベクトルの2ノルムは記法の慣用、行列の種類はモデルの選択）は
        `norm_kind` の `_comment`。値は等式の右辺から取る（`strip_lhs`）——H-1 の
        名前つきは `expr_lookup` の中で `LHS = RHS` に復元されており、左辺は名前。
        """
        from notebook.runtime.i18n import t as _t_be
        expr_ref = str(args.get("expr", ""))
        if not expr_ref:
            return BuildResult(ok=False, command="", error="Missing 'expr' argument")
        raw = strip_lhs(user_form(expr_lookup, expr_ref)).strip()
        kind = str(args.get("kind") or "").strip().lower() or None

        # **素の名前でも渡ってくる。** いまは別名走査が `expr` に入れられるのは
        # 式番号だけだが、宣言済みのベクトル場は名前で渡せる（`A`）し、
        # OPERAND-BOUNDARY の残件が解ければ H-1 の名前（`v`・`M`）も来る。
        # 名前は `Quantity.defining_expr_id` で自分の式を指しているので、
        # そこから値を取る——これが無いと、名前で渡された瞬間に「ベクトルにも
        # 行列にも見えません」と断ることになる（利用者の指摘で実測、2026-09-10）。
        if raw == expr_ref and self._context is not None:
            _q = self._context.get_quantity(expr_ref)
            _def = getattr(_q, "defining_expr_id", None) if _q is not None else None
            if _def and _def in expr_lookup:
                raw = strip_lhs(expr_lookup[_def]).strip()

        lowered = self.to_backend_code(raw)
        value = lowered.code if lowered.ok else raw
        looks_matrix = value.lstrip().startswith("matrix(")

        # **1行または1列の行列はベクトルである。** 研究者が普通に書くベクトルは
        # TeX の `\begin{pmatrix} 3 \\ 4 \end{pmatrix}` で、それは
        # `matrix([3],[4])` として届く——`[3, 4]` という CAS 綴りは平文入力箱の
        # 形にすぎない（利用者の指摘、2026-09-10。私の実装は最初これを行列として
        # 扱い、列ベクトルに種類を要求していた）。
        # **ここで選択はしていない**: 1行/1列なら Frobenius もスペクトルも2ノルムと
        # 一致するので、種類を書かなくても値は定まる。種類を書けばそれに従う
        # （列和 1ノルムは別の値なので、書けば違う答えが返る）。
        if looks_matrix and kind is None and _is_row_or_column_matrix(value):
            return BuildResult(ok=True, command=f"mat_norm({value}, frobenius)")

        if looks_matrix:
            if kind is None:
                return BuildResult(
                    ok=False, command="",
                    error=_t_be("norm.matrix_kind_required", self.lang),
                    reason_code="norm.matrix_kind_required")
            template = self._MATRIX_NORM_COMMAND.get(kind)
            if template is None:
                return BuildResult(
                    ok=False, command="",
                    error=_t_be("norm.unknown_matrix_kind", self.lang, kind=kind),
                    reason_code="norm.unknown_matrix_kind")
            return BuildResult(ok=True, command=template.format(m=value))

        # 素のリスト `[a, b, c]` は Maxima の `.` がそのまま内積を返すので、
        # 成分に開く必要が無い（**3成分に縛られない**——`nb_dot` は3成分固定で、
        # `[3,4]` のような2次元ベクトルはそこを通せない）。
        _is_vector = (value.lstrip().startswith("[")
                      or (self._context is not None
                          and getattr(self._context.get_quantity(raw), "kind", None)
                          == "vector_field"))
        # **種類の可否は、相手が何かを決めてから訊く。** 先に訊いていたとき、
        # 解決できない相手に種類を付けると「ベクトルに `frobenius` は無い」という
        # **的外れな断り**が返っていた（実測 2026-09-10）。
        if _is_vector and kind is not None and kind not in ("euclidean", "2"):
            return BuildResult(
                ok=False, command="",
                error=_t_be("norm.vector_kind_unsupported", self.lang, kind=kind),
                reason_code="norm.vector_kind_unsupported")

        if value.lstrip().startswith("["):
            return BuildResult(ok=True, command=f"sqrt(({value}) . ({value}))")

        # 宣言されたベクトル場は成分で持っているので、そちらの解決に任せる。
        _qty = self._context.get_quantity(raw) if self._context else None
        if _qty is not None and getattr(_qty, "kind", None) == "vector_field":
            try:
                c1, c2, c3 = self._build_vec_components(expr_ref, "cartesian", expr_lookup)
            except (TypeError, RecursionError) as exc:
                return BuildResult(ok=False, command="", error=str(exc))
            comps = f"[{c1},{c2},{c3}]"
            return BuildResult(ok=True, command=f"sqrt(nb_dot({comps},{comps}))")

        # **何のノルムか分からないものは、生の CAS 文言でなく理由で断る。**
        # ここに落ちる実例が1つある: `v = [3, 4]` である。**平坦な2要素の角括弧は
        # この記法では交換子**（CLAUDE.md「a flat two-element `[a, b]` really is
        # one in this notation」）なので `[3, 4]` は `3*4 - 4*3 = 0` に降りる
        # ——2成分ベクトルはこの綴りでは書けない（実測 2026-09-10）。
        if infer_algebra(value, self._context, value_lookup(self._context, expr_lookup.get)) == "vector":
            try:                          # `2 r` / `v + w` — the ∇ builders' component reading
                comps = "[{},{},{}]".format(*self._build_vec_components(expr_ref, "cartesian", expr_lookup))
                return BuildResult(ok=True, command=f"sqrt(nb_dot({comps},{comps}))")
            except (TypeError, RecursionError):
                pass
        return BuildResult(
            ok=False, command="",
            error=_t_be("norm.operand_kind_unknown", self.lang),
            reason_code="norm.operand_kind_unknown")

    # divergence / curl / gradient / scalar_laplacian
    # Uses _build_vec_components / _build_scalar_expr for recursive resolution,
    # so nested expressions like curl(curl(A)) or div(grad(f)) work without
    # registering intermediate results in SymbolContext.
    # TypeError / RecursionError from the resolvers are caught here and
    # returned as BuildResult(ok=False) with a user-facing message.
    def _build_vector_calculus(self, operation, args, expr_lookup) -> BuildResult:
        field_ref = str(args.get("expr", ""))
        if not field_ref:
            return BuildResult(ok=False, command="", error="Missing 'expr' argument")

        # Determine coordinate system: per-op override > field's own system > session default
        coord_sys = str(args.get("coordinate_system") or "cartesian")
        if self._context:
            raw_name = expr_lookup.get(field_ref, field_ref)
            qty = self._context.get_quantity(raw_name)
            if qty is not None and qty.coordinate_system:
                coord_sys = str(args.get("coordinate_system") or qty.coordinate_system)

        try:
            if operation == "divergence":
                c1, c2, c3 = self._build_vec_components(field_ref, coord_sys, expr_lookup)
                cmd = f"nb_div({c1},{c2},{c3})"

            elif operation == "curl":
                c1, c2, c3 = self._build_vec_components(field_ref, coord_sys, expr_lookup)
                sfxs = _COORD_COMPONENTS.get(coord_sys, ("x", "y", "z"))
                s1, s2, s3 = sfxs
                if coord_sys == "cartesian":
                    cmd = (f"[nb_curl_x({c1},{c2},{c3}),"
                           f"nb_curl_y({c1},{c2},{c3}),"
                           f"nb_curl_z({c1},{c2},{c3})]")
                else:
                    cmd = (f"[nb_curl_{s1}({c1},{c2},{c3}),"
                           f"nb_curl_{s2}({c1},{c2},{c3}),"
                           f"nb_curl_{s3}({c1},{c2},{c3})]")

            elif operation == "gradient":
                scalar_expr = self._build_scalar_expr(field_ref, coord_sys, expr_lookup)
                cmd = f"nb_grad({scalar_expr})"

            else:  # scalar_laplacian
                scalar_expr = self._build_scalar_expr(field_ref, coord_sys, expr_lookup)
                cmd = f"nb_laplacian_scalar({scalar_expr})"

        except (TypeError, RecursionError) as exc:
            return BuildResult(ok=False, command="", error=str(exc))

        # ∇ BINDS the coordinate variables, the way ∑ binds its index, and the
        # nb_* functions spell them inside themselves — so an H-1 P2 binding on
        # one of those names reaches them (COORD-NAME-BOUND).  Everything
        # measured, including why the differentiation variable must NOT get the
        # same treatment, is in `shadow_coordinate_variables`.
        return BuildResult(
            ok=True,
            command=shadow_coordinate_variables(self._nb_coord_vars, cmd))

    # change_variable: substitution in an integral (u-substitution)
    # Uses Maxima's changevar(expr, new_var - old_expr, new_var, old_var)
    # substitute: take expr verbatim from expr_lookup to avoid mangling
    # Maxima-specific notation ('diff, q(t), noun forms) through the
    # SharedParser → CodeGenerator round-trip.
    # lhs / rhs are still parsed through IR so that LaTeX inputs like
    # "\sin(q)" are correctly converted to Maxima form "sin(q)".
    @staticmethod
    def _is_compound_subst_lhs(lhs: str) -> bool:
        """True when *lhs* is a top-level sum/difference (``a+b``, ``-a-b-c``),
        the shape Maxima's syntactic ``subst`` cannot match (H-10).

        A leading unary sign is not an addition, and neither is a sign that
        follows an operator or an opening bracket (``%e^-x``, ``f(-x)``), so the
        scan requires a real left operand before the +/-.
        """
        depth = 0
        for i, ch in enumerate(lhs):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch in "+-" and depth == 0:
                prev = lhs[:i].rstrip()
                if prev and prev[-1] not in "+-*/^(,=":
                    return True
        return False

    def _build_substitute(self, args, expr_lookup, parser, resolver, ctx) -> "BuildResult | None":
        # docs/32 Phase 3.  This builder lowers its lhs/rhs through the IR and
        # took the expression BODY verbatim, so the two disagreed about what
        # `V_i` means and `subst(V(i)=5, a*V_i + b)` matched nothing — silently,
        # as a success (open_items AUDIT-SUBST-INDEXED).  Lowering the body here
        # makes both sides one language.
        #
        # Deliberately NOT done for every operation in _prepare_expr_lookup: a
        # blanket lowering was tried and MEASURED WRONG (2026-07-30).  The
        # builders dispatched before the generic per-argument loop pair the
        # source text with text that is NOT lowered — poisson_bracket with the
        # declared canonical pair names, convolve with its own subst() calls,
        # truncate_series/inverse_transform with variable and basis names — so
        # lowering one side only breaks them: the Poisson bracket of a p_r
        # source against a p_r pair name became 0, and convolve hung Maxima.
        # Those builders are self-consistent on raw text (the same reason
        # solve_equation always worked); making them uniform means lowering
        # their OTHER side too, per builder.  See docs/32 §8.
        try:
            expr_lookup = lower_expr_lookup(self, args, expr_lookup)
        except LoweringRefused as _exc:
            from notebook.runtime.i18n import t as _t_be
            return BuildResult(
                ok=False, command="",
                error=_t_be("engine.cannot_lower_expression", self.lang,
                            expr_id=_exc.expr_id, reason=_exc.reason))
        _sub_expr_ref = str(args.get("expr", ""))
        _sub_src = expr_lookup.get(_sub_expr_ref, _sub_expr_ref)
        # Normalize \dot{}/\ddot{} dot-notation to the atom form so the IR
        # below does not render it as the bogus Maxima function dot(theta).
        # Same reason for the derivative DISPLAY forms (BP-9): lhs/rhs go
        # straight to the parser here rather than through latex_to_canonical, so
        # a \frac{\partial phi}{\partial x} argument would reach the resolver as
        # a fraction and either raise or silently cancel to phi/x.
        def _norm_arg(v: str) -> str:
            return self.normalize_dot_notation(
                self.restore_derivative_display_latex(v))
        _sub_lhs_raw = _norm_arg(str(args.get("lhs", "")))
        _sub_rhs_raw = _norm_arg(str(args.get("rhs", "")))
        if _sub_src and _sub_lhs_raw and _sub_rhs_raw:
            from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
            _cg = MaximaCodeGenerator(ctx)
            # Parse lhs/rhs through IR for proper LaTeX→Maxima conversion.
            _lhs_surface = parser.parse(_sub_lhs_raw)
            _lhs_code = _cg.generate(resolver.resolve(_lhs_surface, ctx)) if _lhs_surface else _sub_lhs_raw
            _rhs_surface = parser.parse(_sub_rhs_raw)
            _rhs_code = _cg.generate(resolver.resolve(_rhs_surface, ctx)) if _rhs_surface else _sub_rhs_raw
            # Non-Lagrangian (Newtonian) sessions render every time derivative
            # in Maxima's depends form 'diff(q,t,N).  The IR above generates the
            # velocity/acceleration atoms (r_dot, theta_ddot); map them to the
            # depends form on the RHS so the substituted terms cancel against the
            # source expression's depends-form derivatives (EOM → 0).  In
            # Lagrangian mode q_dot/q_ddot stay independent atoms (no mapping).
            if not self.is_lagrangian_mode():
                _rhs_code = self.apply_time_dep_map(_rhs_code)
            # Build a q(t)-form variant of lhs_code for expressions produced by
            # diff/block where Maxima uses q(t) rather than bare q.
            _lhs_code_qt = _lhs_code
            for _td_name, _td_prop in ctx.registry.items():
                if _td_prop.quantity.time_dependent:
                    _td_t = _td_prop.quantity.time_parameter or "t"
                    # Do not rewrite a name that is the base of a derivative
                    # atom (theta in theta_ddot) — that would emit the invalid
                    # token theta(t)_ddot and break the whole subst.  The
                    # depends-form variant (_lhs_dep below) covers that case.
                    _lhs_code_qt = re.sub(
                        r'(?<![A-Za-z0-9_])' + re.escape(_td_name)
                        + r'(?!\s*\()(?!_d{1,3}ot)',
                        f"{_td_name}({_td_t})", _lhs_code_qt
                    )
            # Circular substitution guard: if the LHS symbol appears INSIDE A
            # DERIVATIVE in the RHS, the substitution is circular (e.g.
            # q → -'diff(q,t,2)/ω² from solving q̈+ω²q=0 for q). The multi-variant
            # loop below chains subst(bare_q=RHS, subst('diff(q,t,2)=RHS, expr))
            # — since RHS itself contains 'diff(q,t,2), each nested subst()
            # re-inserts a copy for the other to match again, diverging.
            # Scope: only a derivative-wrapped self-reference is dangerous. A
            # plain algebraic self-reference (L → n*L, a scaling/renaming
            # substitution) is NOT circular — Maxima's subst() is single-pass
            # and handles it correctly in one step (confirmed empirically: the
            # S-w practice-loop scenario's L→n*L substitution was wrongly
            # blocked here before this fix narrowed the check).
            _bare_lhs = re.sub(r'\([^)]*\)', '', _lhs_code).strip()  # strip (t)
            _rhs_diff_calls = re.findall(r"'diff\([^)]*\)", _rhs_code)
            _circular = _bare_lhs and any(
                re.search(r'(?<![A-Za-z0-9_])' + re.escape(_bare_lhs)
                         + r'(?![A-Za-z0-9_(])', _dc)
                for _dc in _rhs_diff_calls
            )
            if _circular:
                from notebook.runtime.i18n import t as _t_be
                _lang = self.lang
                return BuildResult(
                    ok=False,
                    command="",
                    error=_t_be("engine.circular_substitution", _lang,
                                lhs=_bare_lhs),
                )
            # The source used to be cut down to its RIGHT-HAND SIDE here, because
            # `subst` on a full `H = RHS` makes Maxima evaluate the bound `H` and
            # answer the tautology `RHS' = RHS'`.  That cut applied to EVERY
            # equation, and it threw away whatever was on the left: measured
            # 2026-08-14 on `L(s^2 Q - s q_0 - u_0) + R(s Q - q_0) + Q/C = V`,
            # the substitution's source became the bare `V`, so no target could
            # ever be found — and had the target also appeared on the right, it
            # would have substituted into HALF the equation and reported success.
            #
            # SHADOW instead of cut: a local `block([H], …)` stops Maxima
            # evaluating the left symbol, so the full equation is safe to pass.
            # Measured, all three cases come out right and the bound one comes
            # out BETTER than cutting did (the label survives):
            #
            #     bound `H`     block([H], subst(x=2, H = J*x*y))  → H = 2*J*y
            #                   (cut gave `2*J*y` — the label gone)
            #     unbound `W`   block([W], subst(x=2, W = J*x*y))  → W = 2*J*y
            #     compound lhs  subst(q_0=0, L*(s^2*Q - s*q_0) = V) → L*Q*s^2 = V
            #                   (cut gave `V` — everything gone)
            #
            # Shadowing a symbol that is not bound is a NO-OP, which is why this
            # needs no test for WHICH case it is.  That matters: the honest test
            # ("will the CAS evaluate this symbol?") is CAS state, and consulting
            # it would make the decision non-reproducible; the syntactic proxies
            # (monomial / "contains nothing evaluable") get the unbound bare
            # symbol wrong; and `defining_expr_id` is not equivalent either — a
            # solve result carries one but never binds (BP-7, measured).  Over-
            # firing a shadow costs nothing, so no discriminator is needed.
            _rel_s = split_relation(_sub_src)
            _shadow_lhs = ""
            if _rel_s is not None and "=" in _rel_s[1]:
                _cand = _rel_s[0].strip()
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", _cand):
                    _shadow_lhs = _cand
            # Apply substitution in both bare-q and q(t) forms so that it
            # matches regardless of whether the expression was produced by
            # depends([q],[t]) (bare q) or diff(q(t),t,N) paths (q(t) form).
            # Also generate the "bare" depends form of any derivative LHS:
            # 'diff(q(t),t,2) → 'diff(q,t,2) so both variants are tried.
            _all_lhs: list[str] = [_lhs_code]
            if _lhs_code_qt != _lhs_code:
                _all_lhs.append(_lhs_code_qt)
            # Depends-form of the LHS derivative atom (theta_ddot →
            # 'diff(theta,t,2)) so it matches a source expression produced by
            # differentiate, which emits the depends form.  Non-Lagrangian only.
            if not self.is_lagrangian_mode():
                _lhs_dep = self.apply_time_dep_map(_lhs_code)
                if _lhs_dep != _lhs_code and _lhs_dep not in _all_lhs:
                    _all_lhs.append(_lhs_dep)
            _bare_dep_m = re.match(r"'diff\((\w+)\((\w+)\),(\w+),(\d+)\)", _lhs_code)
            if _bare_dep_m:
                _bare_dep = (f"'diff({_bare_dep_m.group(1)},"
                             f"{_bare_dep_m.group(3)},{_bare_dep_m.group(4)})")
                if _bare_dep not in _all_lhs:
                    _all_lhs.append(_bare_dep)
            # H-10: Maxima's subst() is purely SYNTACTIC — it rewrites a
            # subexpression only where it appears verbatim in the internal form.
            # A COMPOUND lhs (a top-level sum, e.g. the Coulomb-gauge condition
            # ∂(∇·A)/∂t whose three terms sit inside the longer ∇·E sum) therefore
            # never matches: verified against live Maxima, subst(0,a+b,a+b+c)
            # returns a+b+c unchanged and the sign-flipped -a-b-c likewise, so the
            # operation silently did nothing.  ratsubst does the algebraic match
            # (-c, correctly) and is the right tool here — the same reasoning
            # _build_apply_canonical_constraints already uses for polynomial side
            # relations.  Deliberately restricted to the compound case: an ATOMIC
            # lhs keeps subst(), whose behaviour and un-rationalized output are
            # unchanged.  Note the argument order differs — subst takes the
            # old=new equation, ratsubst takes (new, old, expr).
            # A bounded `sum(...)`/`product(...)` on the LEFT is the one case
            # `subst` cannot see at all: Maxima evaluates the aggregate before
            # matching, so the pattern is already its terms.  Quoting BOTH sides
            # keeps them nouns, which is the only form where the match happens
            # (measured; see maxima_constants.quote_aggregates).  Applied only
            # when the lhs actually holds one — an ordinary substitution into an
            # expression that merely CONTAINS a sum keeps its existing behaviour,
            # because quoting there would change what the researcher gets back
            # (the sum would stop expanding) for no reason connected to the fix.
            if quote_aggregates(_lhs_code) != _lhs_code:
                _all_lhs = [quote_aggregates(v) for v in _all_lhs]
                _lhs_code = quote_aggregates(_lhs_code)
                _sub_src = quote_aggregates(_sub_src)
            _use_ratsubst = self._is_compound_subst_lhs(_lhs_code)
            # Silence guard (R5, docs/32 Phase 1a): an atomic lhs that occurs
            # nowhere in the source cannot match, and Maxima reports that as
            # success with nothing changed.  Refuse instead.  ratsubst is
            # exempt — its match is algebraic (H-10), not syntactic.
            _shadowed = {n for n in (_shadow_lhs, _bare_lhs) if n}
            if not _use_ratsubst and not lhs_may_match(
                    _sub_src, _all_lhs, ctx, shadowed=_shadowed):
                from notebook.runtime.i18n import t as _t_be
                return BuildResult(ok=False, command="", reason_code="lhs_not_found",
                                   error=_t_be("engine.substitute_lhs_not_found",
                                               self.lang, lhs=_sub_lhs_raw))
            _cmd = _sub_src
            for _lhs_variant in reversed(_all_lhs):
                _cmd = (f"ratsubst({_rhs_code},{_lhs_variant},{_cmd})"
                        if _use_ratsubst
                        else f"subst({_lhs_variant}={_rhs_code},{_cmd})")
            _cmd = f"ratsimp({_cmd})"
            # If the LHS symbol was itself H-1 P2-bound (`q_1: Q_1+Q_2$`, sent
            # when q_1 was registered), Maxima eagerly evaluates BOTH the
            # substitution rule's LHS and the source expression before subst()
            # ever runs — turning "subst(q_1=RHS, q_1^2+q_2^2)" into
            # "subst((Q_1+Q_2)=RHS, (Q_1+Q_2)^2+(Q_1-Q_2)^2)", which then
            # partially re-matches inside the already-substituted expression
            # (SUBST-P2-EAGER, found via chained multi-substitute regression
            # while fixing SUBST-DIGIT-SUB).  A local block() unbinds the
            # symbol for the duration of the call, exactly like
            # _build_inverse_transform already does for new-basis symbols.
            if re.fullmatch(r"[A-Za-z_]\w*", _bare_lhs):
                _cmd = f"block([{_bare_lhs}], {_cmd})"
            # The same shadow for the SOURCE equation's own left symbol, which
            # is what replaced cutting the equation down to its right-hand side
            # (see the comment above the `_shadow_lhs` computation).
            if _shadow_lhs:
                if _shadow_lhs != _bare_lhs:
                    _cmd = f"block([{_shadow_lhs}], {_cmd})"
                # …and the split moves to AFTER the substitution.  Cutting first
                # discarded whatever was on the left; cutting the RESULT keeps
                # this file's convention that an operation's canonical is the
                # VALUE while the label lives in the latex.  Only a BARE-symbol
                # left side has a "value" to take — where the content is on the
                # left (`L(s^2 Q - …) = V`) the equation IS the result, and
                # `_shadow_lhs` is empty there, so it is returned whole.
                _cmd = f"rhs({_cmd})"
            return BuildResult(ok=True, command=_cmd)

    def _integral_noun_code(self, text: str) -> "str | None":
        """*text* を**積分を名詞のまま**下ろした CAS 綴り。積分が無ければ `None`。

        `hold_integrals` は既に在る仕組みで、ここが2人目の利用者になる。
        """
        from notebook.runtime.shared_parser import UnknownNotationError
        try:
            ir = self.parse_ir(text)
            if ir is None:
                return None
            code = self.generate_backend_code(ir, hold_integrals=True)
        except (UnknownNotationError, ValueError, AttributeError, TypeError):
            return None
        return code if ("'integrate" in code or "'sum" in code) else None

    def _build_change_variable(self, args, expr_lookup, parser, resolver, ctx) -> BuildResult:
        old_var  = str(args.get("old_var",  "x"))
        new_expr_raw = args.get("new_expr", "u")
        new_var  = str(args.get("new_var")  or "u")
        expr_ref = args.get("expr", "")
        src_expr = expr_lookup.get(expr_ref, expr_ref)
        # **被演算子は研究者の書いたものなので、他と同じように下ろす**——そして
        # **積分は名詞のまま**下ろす。`changevar` は積分（やシグマ）の形にしか
        # 効かず、Maxima は `integrate(...)` を先に評価してしまうので、名詞に
        # しないと「評価済みの値」に当てることになり、必ず何も起きない
        # （実測 2026-09-19: `changevar('integrate(y^2,y), u-y, u, y)` は
        # `'integrate(u^2,u)`、`changevar(integrate(y^2,y), …)` は `y^3/3`）。
        # 下ろさずに渡していたので、`\int y^2 \, dy` は `\int*y^2*dy` のまま
        # CAS へ行き `dy*int*y^2` が**成功として**返っていた
        # （BUILDER-OPERAND-NOT-LOWERED、`_build_relabel_index` と同じ形）。
        _held = self._integral_noun_code(src_expr)
        if _held is not None:
            src_expr = _held
        else:
            # **積分が無ければ `changevar` にできることは何も無い。** ただの式に
            # 当てると式をそのまま返すので、研究者には「変換した」と読める成功が
            # 返っていた（実測 2026-09-19、`E1 の y を u に変数変換して` →
            # `z^2+y`、日英とも）。断って、通る道を名指す。
            from notebook.runtime.i18n import t as _t_cv
            return BuildResult(ok=False, command="", error=_t_cv(
                "change_variable.not_an_integral", self.lang, expr_id=expr_ref,
                old_var=old_var, new_var=new_var))
        # Resolve new_expr to Maxima string
        ne_surface = parser.parse(str(new_expr_raw))
        if ne_surface is not None:
            from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
            ne_code = MaximaCodeGenerator(ctx).generate(
                resolver.resolve(ne_surface, ctx)
            )
        else:
            ne_code = str(new_expr_raw)
        # changevar(expr, relation, new_var, old_var)
        # relation must equal zero at the substitution point
        cmd = f"changevar({src_expr}, ({ne_code})-({old_var}), {new_var}, {old_var})"
        return BuildResult(ok=True, command=cmd)

    # change_integration_variables: multivariable change of variables with the
    # Jacobian determinant automatically inserted.  Produces the new integrand
    #   f(transform) * det(jacobian(transform, new_vars))
    # which the user then integrates over the new variables.
    def _build_change_integration_variables(self, args, expr_lookup, parser, resolver, ctx) -> BuildResult:
        expr_ref  = args.get("expr", "")
        integrand = expr_lookup.get(expr_ref, expr_ref)

        _BRACKET_CLOSE = {"[": "]", "(": ")", "{": "}"}

        def _strip_outer_wrap(s: str) -> str:
            """Strip a single matching outer bracket pair that wraps the WHOLE
            string (e.g. "[x, y]" -> "x, y"), such as a plan step's args
            naturally supply for a var list.  Only strips when the opening
            bracket's matching close is the string's last character — "[x],[y]"
            (two separate wrapped items) is left untouched."""
            s = s.strip()
            if not s or s[0] not in _BRACKET_CLOSE or s[-1] != _BRACKET_CLOSE[s[0]]:
                return s
            depth = 0
            for i, ch in enumerate(s):
                if ch == s[0]:
                    depth += 1
                elif ch == _BRACKET_CLOSE[s[0]]:
                    depth -= 1
                    if depth == 0:
                        return s[1:-1].strip() if i == len(s) - 1 else s
            return s

        def _split_top(s: str) -> "list[str]":
            """Split on top-level commas (ignoring those inside (), [], {})."""
            parts, depth, buf = [], 0, []
            for ch in str(_strip_outer_wrap(s)):
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                if ch == "," and depth == 0:
                    parts.append("".join(buf).strip()); buf = []
                else:
                    buf.append(ch)
            if "".join(buf).strip():
                parts.append("".join(buf).strip())
            return [p for p in parts if p]

        old_vars  = _split_top(args.get("old_vars", ""))
        new_vars  = _split_top(args.get("new_vars", ""))
        transform_raw = _split_top(args.get("transform", ""))

        def _to_maxima(s: str) -> str:
            surf = parser.parse(s)
            if surf is None:
                return s
            resolved = resolver.resolve(surf, ctx)
            # A transform piece may be written as a bare RHS ("r*cos(theta)") or as a
            # full equation ("x = r*cos(theta)") — both a user and an LLM proposing a
            # plan step naturally reach for the latter.  Only the RHS is meaningful
            # here (old_vars already supplies the LHS name), so strip it uniformly.
            from notebook.models.ir import Equation as _Equation
            if isinstance(resolved, _Equation):
                resolved = resolved.rhs
            from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
            return MaximaCodeGenerator(ctx).generate(resolved)

        transform = [_to_maxima(t) for t in transform_raw]
        if not (old_vars and new_vars and transform
                and len(old_vars) == len(transform)):
            return BuildResult(
                ok=False, command="",
                error="change_integration_variables: old_vars and transform "
                      "must have equal length, and new_vars must be given")

        subst_list = ", ".join(f"{o} = {t}" for o, t in zip(old_vars, transform))
        jac_list   = ", ".join(transform)
        nv_list    = ", ".join(new_vars)
        # The transform went through the parser and generator above; the BODY
        # must go the same way, or the two sides of one `subst` disagree about
        # what the researcher wrote.  Measured 2026-08-20 on the S-aa Gaussian:
        # the canonical of an integral legitimately carries `\int` (the parser
        # reads it), and embedded raw it reached Maxima as the product
        # `int_ * -%e^-(r^2 cos(theta)^2) * inf^inf * dx` — the integral gone,
        # the Jacobian factor correct, and the step reported SUCCESS.  This is
        # the per-builder half of the docs/32 rule: a builder that lowers one
        # side has to lower the other.
        _low = self.to_backend_code(integrand)
        if _low.status == "refused":
            return BuildResult(ok=False, command="", error=_low.reason)
        cmd = (f"subst([{subst_list}], {_low.code}) "
               f"* determinant(jacobian([{jac_list}], [{nv_list}]))")
        return BuildResult(ok=True, command=cmd)

    # expand_neighbor_sum (F-1): 1D-chain nearest-neighbor bond sum.
    # The body (a scalar lattice term such as -J*s_i*s_j or a Coulomb/exchange
    # term in lattice-energy / quantum-chemistry work) is expanded into the
    # explicit per-site sum, then resolved with the Maxima generator.
    def _build_expand_neighbor_sum(self, args, parser, resolver, ctx) -> BuildResult:
        from notebook.runtime.backend_handler import build_neighbor_sum_terms
        sum_str, err = build_neighbor_sum_terms(
            args.get("body", ""), args.get("index1", ""), args.get("index2", ""),
            args.get("n_sites"), args.get("boundary", "open"),
            args.get("pair_constraint", "nearest_neighbor"),
            args.get("lattice_shape"),
            args.get("edges"),
            args.get("pin_site"),
        )
        if err:
            return BuildResult(ok=False, command="", error=err)
        surf = parser.parse(sum_str)
        if surf is None:
            return BuildResult(ok=False, command="",
                               error=f"expand_neighbor_sum: could not parse {sum_str!r}")
        from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
        return BuildResult(ok=True,
                           command=MaximaCodeGenerator(ctx).generate(resolver.resolve(surf, ctx)))

    # differentiate w.r.t. a time-dependent quantity's coordinate:
    # When var=q and q is declared time-dependent, direct diff(L, q) triggers
    # Maxima's chain rule and produces spurious diff(q(t),q) terms from q̇=diff(q(t),t,1).
    # Solution: auto-wrap in block() that temporarily substitutes q̇→_nb_v_tmp and
    # q(t)→_nb_q_tmp, differentiates w.r.t. the plain temp var, then restores q(t).
    # This is transparent to the user — no manual substitute steps needed.
    @staticmethod
    def _hold_atoms_and_diff(src: str, atoms: "list[str]", var: str,
                             order: int = 1) -> str:
        r"""座標を止めたまま *var* で微分する —— ∂f/∂*var*。

        **素の `diff(f,t)` では駄目である。** `declare_time_dependent` は
        `depends(q,t)` を送っており（実測: `dependencies` が `[q(t)]`）、Maxima は
        そこで連鎖律を適用する。だから `diff(f,t)` が返すのは **d**f/d*t* であって
        ∂f/∂*t* ではない——その一致こそが、偏微分を頼んだ研究者に全微分が
        `∂f/∂t` というラベルで返っていた仕組みそのものだった（実測 2026-09-06）。
        一時的に *t* 非依存の記号へ置換してから微分し、戻す。

        全微分の**最後の項**（明示的な時間依存 ∂f/∂t）と、偏微分の**全体**が、
        同じ式である。だから1箇所に置く——2箇所に書けば、片方だけが階数を
        知っている、というようなことが起きる。
        """
        if not atoms:
            return f"diff({src},{var},{order})"
        _fwd = ",".join(f"{a}=_nb_free_{a}" for a in atoms)
        _bwd = ",".join(f"_nb_free_{a}={a}" for a in atoms)
        return f"subst([{_bwd}],diff(subst([{_fwd}],{src}),{var},{order}))"

    def _build_differentiate(self, args, expr_lookup, ctx) -> "BuildResult | None":
        _var_str = str(args.get("var", ""))
        _expr_ref = str(args.get("expr", ""))
        _order = int(args.get("order", 1))
        # Analytical-mechanics (Lagrangian) mode: q, q̇ (q_dot), q̈ (q_ddot) are
        # INDEPENDENT atoms.  ∂/∂q̇ and ∂/∂q are plain diffs on those atoms; d/dt
        # (var = time parameter) is the total time derivative Σ_i Σ_k
        # ∂E/∂q_i^(k)·q_i^(k+1).  No block-substitution / function-form hacks.
        if self.is_lagrangian_mode():
            _src_l = expr_lookup.get(_expr_ref, _expr_ref)
            _coords = [n for n, p in ctx.registry.items()
                       if p.quantity.time_dependent]
            _tparams = {(ctx.get_quantity(c).time_parameter or "t") for c in _coords} \
                or {"t"}
            if _var_str in _tparams:                       # d/dt / ∂/∂t
                _atoms: list[str] = []
                for _c in _coords:
                    _atoms.extend([_c, f"{_c}_dot", f"{_c}_ddot",
                                   f"{_c}_dddot", f"{_c}_ddddot"])
                # **∂/∂t は「座標を止めて t で微分する」——それだけである。**
                # 直す前はこの枝が**変数だけ**で全微分を組んでおり、研究者が
                # 偏微分と書いても（旗はここまで届いていなかったので）連鎖項が
                # 付いた式が `∂L/∂t` というラベルで返っていた（実測 2026-09-06:
                # 駆動振動子 `L = q̇²/2 − q²/2 + F q cos(ωt)` の ∂L/∂t が
                # `−F q ω sin(ωt) + F q̇ cos(ωt) + …`、正しくは第1項だけ）。
                # `dH/dt = −∂L/∂t` はまさにこの量についての主張なので、
                # 解析力学で ∂L/∂t が要る場面の中心がこれである。
                # 階数もここでしか効かない——`2階偏微分` は全微分の1階を返していた。
                if _derivative_kind(args) == "partial":
                    return BuildResult(
                        ok=True,
                        command=self._hold_atoms_and_diff(_src_l, _atoms,
                                                          _var_str, _order))
                _terms: list[str] = []
                for _c in _coords:
                    _chain = [_c, f"{_c}_dot", f"{_c}_ddot",
                              f"{_c}_dddot", f"{_c}_ddddot"]
                    for _k in range(len(_chain) - 1):
                        _terms.append(f"diff({_src_l},{_chain[_k]})*{_chain[_k + 1]}")
                # **明示的な時間依存の項 ∂f/∂t** —— 連鎖律だけでは落ちる。
                # `df/dt = Σ_i Σ_k ∂f/∂q_i^(k)·q_i^(k+1) + ∂f/∂t` の最後の項で、
                # 式にもこの関数のコメントにも入っていなかった（実測 2026-09-06）:
                # `f = q^2 + t` の d/dt が `2 q q̇`（正しくは `2 q q̇ + 1`）、
                # 駆動振動子 `L = q̇²/2 − q²/2 + F q cos(ωt)` では
                # **`− F q ω sin(ωt)` が丸ごと欠落**して成功が返っていた。
                # 解析力学で dL/dt を使う中心的な場面がエネルギー定理
                # `dH/dt = −∂L/∂t` で、それは明示的時間依存そのものの主張である。
                #
                # **素の `diff(f,t)` では二重計上になる。** `declare_time_dependent`
                # は `depends(q,t)` を送っており（実測: `dependencies` が `[q(t)]`）、
                # Maxima はそこで連鎖項をもう一度出す。座標とドット鎖を一時的に
                # t 非依存の記号へ置換してから微分し、戻す——実測でこの形が
                # `f = q^2 + t` に対して `1`（＝明示項だけ）を返す。
                _terms.append(self._hold_atoms_and_diff(_src_l, _atoms, _var_str, 1))
                _cmd = ("ratsimp(" + "+".join(_terms) + ")") if _terms else _src_l
                return BuildResult(ok=True, command=_cmd)
            # ∂/∂q or ∂/∂q̇: plain diff on the independent atom (q̇ ⊥ q).
            _cmd = (f"diff({_src_l},{_var_str})" if _order == 1
                    else f"diff({_src_l},{_var_str},{_order})")
            return BuildResult(ok=True, command=_cmd)
        _var_qty = ctx.get_quantity(_var_str) if _var_str else None
        if (_var_qty is not None and _var_qty.time_dependent
                and _expr_ref in expr_lookup):
            _t = _var_qty.time_parameter or "t"
            _src = expr_lookup[_expr_ref]
            _qdot_pat = f"diff({_var_str}({_t}),{_t},1)"
            _qt_pat   = f"{_var_str}({_t})"
            # Only auto-wrap when the expression actually contains q(t) or q̇(t);
            # plain polynomial in q without depends-form doesn't need this.
            if _qdot_pat in _src or _qt_pat in _src:
                # Temp names that won't collide with user variables.
                _v_tmp  = "_nb_qdot_tmp"  # placeholder for q̇
                _qc_tmp = "_nb_qc_tmp"    # placeholder for q(t)
                # diff(_nb_s2, _nb_qc_tmp) for order=1,
                # diff(_nb_s2, _nb_qc_tmp, N) for order>1.
                _diff_call = (f"diff(_nb_s2, {_qc_tmp})" if _order == 1
                              else f"diff(_nb_s2, {_qc_tmp}, {_order})")
                # Replace both q(t) and bare q → _nb_qc_tmp.
                # Maxima may store q as the atom q (not q(t)) in expressions
                # like cos(q), so we need both substitutions.
                # Order: replace q(t) first, then bare q (q(t) ≠ q as atom).
                cmd = (
                    f"block([_nb_s1, _nb_s2, _nb_r],"
                    f" _nb_s1: subst({_v_tmp}, 'diff({_var_str}({_t}),{_t},1), {_src}),"
                    f" _nb_s2: subst({_qc_tmp}, {_var_str},"
                    f" subst({_qc_tmp}, {_qt_pat}, _nb_s1)),"
                    f" _nb_r: {_diff_call},"
                    # Revert BOTH placeholders: q(t) for the position temp and
                    # q̇ for the velocity temp.  Without restoring _nb_qdot_tmp the
                    # protected velocity leaks into ∂L/∂q for velocity-coupled
                    # Lagrangians (e.g. the double pendulum's q̇₁q̇₂cos term).
                    f" subst('diff({_var_str}({_t}),{_t},1), {_v_tmp},"
                    f" subst({_qt_pat}, {_qc_tmp}, _nb_r)))"
                )
                return BuildResult(ok=True, command=cmd)

    # relabel_index: rename dummy summation index (no Jacobian)
    # For symmetric domains (e.g. BZ), Σ_{old} f(old) = Σ_{old} f(new) after
    # relabeling, so this is a direct subst with no extra factors.
    def _build_relabel_index(self, args, expr_lookup) -> BuildResult:
        old_idx = str(args.get("old_index", "k"))
        new_idx = str(args.get("new_index", "-k"))
        expr_ref = args.get("expr", "")
        src_expr = expr_lookup.get(expr_ref, expr_ref)
        # The operand is the researcher's own writing, so it has to be lowered
        # like any other (it was not: measured 2026-08-15, `\sum_{i=1}^{3}*c_i`
        # reached Maxima verbatim, could not be read, and the ADAPTER'S SCRATCH
        # NAME `_res_` was registered as the result — BUILDER-OPERAND-NOT-LOWERED).
        _low = self.to_backend_code(src_expr)
        if _low.status == "refused":
            return BuildResult(ok=False, command="", error=_low.reason or "")
        _code = quote_aggregates(_low.code)
        # An index that is not in the expression cannot be renamed, and Maxima
        # says nothing: measured 2026-08-15, a name that occurred nowhere gave
        # the same successful answer as the real index (NOOP-REWRITE-FAMILY).
        # A plain word-boundary scan is enough here — unlike a substitution
        # pattern, a bound index is a single identifier and `_code` is CAS code,
        # so both sides are in one spelling.
        if not re.search(r"(?<![A-Za-z0-9_])" + re.escape(old_idx)
                         + r"(?![A-Za-z0-9_])", _code):
            from notebook.runtime.i18n import t as _t_ri
            return BuildResult(ok=False, command="", reason_code="lhs_not_found",
                               error=_t_ri("engine.relabel_index_not_found",
                                           self.lang, index=old_idx))
        cmd = f"subst({old_idx}={new_idx}, {_code})"
        return BuildResult(ok=True, command=cmd)

    def simplify_raw_command(self, canonical: str) -> "str | None":
        """Simplify an already-CAS-valid canonical string directly (no re-parse).
        Time-dependent placeholders (q_dot → 'diff(q,t,1)) are mapped first so the
        derivative atoms match; then ratsimp on the raw string."""
        # In Lagrangian mode keep q_dot/q_ddot as independent atoms.
        _c = canonical if self.is_lagrangian_mode() else self.apply_time_dep_map(canonical)
        return f"ratsimp({_c})"

    # Promote a space that sits BETWEEN two operands (juxtaposition-multiplication)
    # to an explicit *.  Only UNAMBIGUOUS cases are matched:
    #   • operand-ending char (\w, ), ]) ↦ space ↦ symbol / number / 'diff noun
    #   • closing bracket (), ]) ↦ space ↦ "("  ("(a+b) (c+d)")
    # A space between a SYMBOL and "(" ("f (x)") is NOT matched — product vs
    # function call needs the SymbolContext, so it is the SharedParser's call.
    # Whitespace around binary operators (a + b, y = a*x) and subscripts (H_0) are
    # untouched, so this cannot corrupt H_0 into H(0) the way a reparse would.
    _JUXTA_MULT_RE = re.compile(r"(?<=[\w\)\]])\s+(?=[\w'])|(?<=[\)\]])\s+(?=\()")

    def ensure_explicit_mult(self, s: str) -> str:
        if not s or " " not in s:
            return s
        return self._JUXTA_MULT_RE.sub("*", s)

    def apply_time_dep_map(self, s: str) -> str:
        """Replace q_dot / q_ddot placeholders with Maxima depends-form derivatives.

        Both this method and ``build_command``'s ``_time_dep_map`` use the same
        ``'diff(q,t,N)`` form (Maxima depends convention), since the session
        establishes ``depends([q],[t])`` at startup.  Uniform representation
        ensures diff(T,t) and diff(V,t) produce the same derivative atom so that
        algebraic cancellation works correctly (e.g. EOM substitution → 0).
        """
        ctx = self._context
        if ctx is None:
            return s
        _time_dep_map: dict[str, str] = {}
        for _qty_name, _prop in ctx.registry.items():
            _qty = _prop.quantity
            if _qty.time_dependent:
                _t = _qty.time_parameter or "t"
                # Use quoted depends form so T's derivative matches V's.
                _time_dep_map[f"{_qty_name}_dot"]  = time_dep_derivative(_qty_name, _t, 1)
                _time_dep_map[f"{_qty_name}_ddot"] = time_dep_derivative(_qty_name, _t, 2)
        if not _time_dep_map:
            return s
        _sorted_keys = sorted(_time_dep_map, key=len, reverse=True)
        for _k in _sorted_keys:
            s = re.sub(
                r'(?<![A-Za-z0-9_])' + re.escape(_k) + r'(?![A-Za-z0-9_])',
                _time_dep_map[_k], s
            )
        return s

    def wrap_command_for_storage(self, command: str, output_id: str | None, operation: str) -> str:
        return command

    def canonical_for_result(self, cas_value: str, output_id: str | None, operation: str) -> str:
        return cas_value

    def cas_repr(self, cas_value: str, canonical: str) -> str | None:
        return canonical

    def clean_void_result(self, value: str) -> str:
        """Suppress Maxima's CAS implementation artifacts from void operations.

        The Maxima adapter uses ``_res_`` as its internal result variable; the
        LaTeX channel may echo this name (or ``\\mathrm{true}``, ``done``) when
        a side-effect-only command is executed.  None of these are meaningful
        display values.
        """
        if value in ("nothing", "true", "done"):
            return ""
        # _res_ (or variants) leaked from the adapter's internal variable name
        if "_res_" in value or value.startswith("\\mathrm{true}"):
            return ""
        return value

    # get_void_op_display_latex is inherited from BackendHandler and handles
    # the four type-declaration ops (scalar/vector/matrix/tensor) via GREEK_CMD.
    # No Maxima-specific override is needed.

    # Maxima matrix-op names differ from Julia's (ctranspose / conjugate vs
    # adjoint / conj).  _restore_matrix_ops() in the base class uses this list.
    _MATRIX_OP_RESTORE = [
        ("ctranspose", r"{\1}^{\\dagger}"),
        ("conjugate",  r"{\1}^{\\ast}"),
        ("transpose",  r"{\1}^{T}"),
    ]

    # Maxima additionally treats "^" as arithmetic in _display_bridge_scalar_guard
    # (base class default omits it) — see BackendHandler._SCALAR_GUARD_ARITH_CHARS.
    _SCALAR_GUARD_ARITH_CHARS = ("+", "*", "/", "^", r"\left", r"\right")

    @staticmethod
    def _unwrap_italic_identifiers(latex: str) -> str:
        """Maxima renders bare multi-char identifiers as ``{\\it alpha\\_mk}`` (italic,
        escaped underscore).  Unwrap to a plain ``alpha_{mk}`` so the user never sees
        the backend's italic markup and the subscript matches the other backends."""
        def _restorable(nm: str) -> bool:
            """Does *nm* end in a suffix the DISPLAY RESTORE further down reads?

            `G_sup0` / `Sigma_psupHF` (label) and `c_hat` / `H_cal` (decoration)
            are NAMES whose suffix becomes a superscript or an accent later in
            this same pass.  Bracing or merging them here hides them from that
            restore, and the internal spelling reaches the screen.
            """
            from notebook.runtime.latex_converter import _split_decoration
            from notebook.runtime.script_label import split_label as _split_script_label
            return (_split_script_label(nm) is not None
                    or _split_decoration(nm)[0] is not None)

        def _norm(m: "re.Match") -> str:
            name = m.group(1).replace(r"\_", "_")
            tail = m.group(2)
            merged = name + tail if (tail is not None and "_" in name) else None
            if merged is not None and _restorable(merged):
                # `{\it G\_sup}_{0}` — Maxima split the LABEL's digit off, so the
                # two halves are one name (`G_sup0` = `G^0`).
                name, tail = merged, None
            elif _restorable(name):
                # `{\it G\_psupHF}_{n,\sigma}` — here the tail is a real
                # SUBSCRIPT (the indices of a subscripted function), and merging
                # it into the name destroyed both: the label restore read the
                # index as part of the label and published `G^{(HFn)},\sigma`
                # (measured 2026-08-21, on a solved Hartree–Fock Green's
                # function).  The name is left RAW so the restore can see its
                # suffix; the tail stays the subscript it was.
                pass
            elif merged is not None:
                # Maxima splits a trailing DIGIT run off an identifier it also
                # italicises (`x_ic1` → `{\it x\_ic}_{1}`).  Unwrapping the group
                # alone gives `x_{ic}_{1}` — a DOUBLE SUBSCRIPT, a hard LaTeX error,
                # so the exported .tex does not compile.  The halves are ONE name →
                # ONE subscript.  Measured on `x_ic1` / `k_B1` / `a_b2`; see
                # scenario_forced_oscillator_laplace_jp.md ②.  Without an underscore
                # there is no second subscript to collide with (`foo1` → `foo_{1}`).
                name, tail = merged, None
            out = name if _restorable(name) else re.sub(r"_([A-Za-z0-9]+)",
                                                        r"_{\1}", name)
            return out if tail is None else f"{out}_{{{tail}}}"
        return re.sub(r"\{\\it\s+([^{}]+)\}(?:_\{([^{}]*)\})?", _norm, latex)

    #: このノートの代数 → Maxima の feature。宣言テンプレート（backends/
    #: maxima.full.json）と同じ対応で、取り消しはその逆を撃つ。
    _ALGEBRA_FEATURE = {"scalar": "scalar", "vector": "nonscalar",
                        "matrix": "nonscalar", "tensor": "nonscalar"}

    def undeclare_command(self, name: str, algebra: str) -> "str | None":
        feature = self._ALGEBRA_FEATURE.get(algebra)
        return f"remove({name}, {feature})" if feature and name else None

    @classmethod
    def _restore_trace(cls, s: str) -> str:
        r"""``mattrace(X)`` → ``\mathrm{tr}\left(X\right)``.

        トレースは**具体行列なら値になる**が、記号行列（`M + N`）では noun の
        まま残る。そのとき `tex1` は `{\it mattrace}(...)` と書き、§0 がイタリック
        を外すので、**CAS の内部名がそのまま画面と公開 TeX に出ていた**
        （実測 2026-09-06: `mattrace\left(N+M\right)`）。`\it arcsin` と同じ形。

        戻す綴りは `\mathrm{tr}` —— **入力として読み直せる**（`upright_operator_spellings`
        が `\mathrm{tr}` を `mattrace` に戻す）ので往復が閉じる。研究者が
        `\text{tr}` と書いても同じ量なので、表示は1つに揃える。
        """
        return cls._restore_functional(
            s, "mattrace",
            lambda body, st: rf"\mathrm{{tr}}\left({body}\right)")

    #: Maxima の `tex1` は片側極限を **矢印の向き**で書く（`x\downarrow a` は
    #: 上から、`x\uparrow a` は下から）。研究者が書くのは `x \to a^{+}` なので、
    #: 戻す。両側極限は `x\rightarrow a` で、これは読める綴りなのでそのまま。
    _LIMIT_TEX1_SIDE = {r"\downarrow": "plus", r"\uparrow": "minus"}
    _LIMIT_TEX1_RE = re.compile(
        r"\\lim_\{(?P<var>.+?)\s*(?P<arrow>\\downarrow|\\uparrow)\s*(?P<point>.+?)\}")

    def _restore_limit_arrow(self, s: str) -> str:
        r"""`\lim_{t'\downarrow t}` → `\lim_{t' \to t^{+}}`.

        **What the CAS prints is not what the researcher wrote.** The side is
        the whole point of a one-sided limit, and left as an arrow direction it
        is a spelling nobody in this notebook types — the same class as
        `{\it arcsin}` and `\dot{theta}`.  The written form comes from the ONE
        place that says how a limit's subscript is spelled
        (`limit_subscript_latex`), which the canonical restore also uses.
        """
        from notebook.runtime.backend_handler import limit_subscript_latex
        return self._LIMIT_TEX1_RE.sub(
            lambda m: "\\lim_{" + limit_subscript_latex(
                m.group("var").strip(), m.group("point").strip(),
                self._LIMIT_TEX1_SIDE[m.group("arrow")]) + "}", s)

    def postprocess_result_latex(
        self,
        canonical: str,
        raw_latex: str,
        registry: SymbolContext,
        sum_indices: str = "",
    ) -> str:
        # Maxima's tex1 writes the sum with the index the command carried, so
        # there is nothing to name back here.
        # §0  Strip Maxima's bare {\it …} italic markup before any other pass so the
        # cleaned identifiers flow through the display bridge / decoration restore.
        raw_latex = self._unwrap_italic_identifiers(raw_latex)
        # §1  Shared restorations (kron_delta, matrix ops, IndexSum, display bridge)
        raw_latex = self._restore_kron_delta(raw_latex)
        raw_latex = self._restore_expectation(raw_latex)
        raw_latex = self._restore_time_order(raw_latex)
        raw_latex = self._restore_trace(raw_latex)
        raw_latex = self._restore_vector(raw_latex)
        # Maxima marks the NON-COMMUTATIVE power with angle brackets in tex1
        # (`A^^2` → `A^{\langle 2\rangle}`, `A^^-1` → `A^ {-\langle 1\rangle }`).
        # The researcher wrote `A^{2}` / `A^{-1}`; give it back.
        raw_latex = re.sub(
            r"\^\s*\{?\s*(-?)\s*\\langle\s*([^\\{}]*?)\s*\\rangle\s*\}?",
            lambda m: "^{" + m.group(1) + m.group(2).strip() + "}", raw_latex)
        raw_latex = self._restore_matrix_ops(raw_latex)
        raw_latex = self._restore_limit_arrow(raw_latex)
        raw_latex = self._restore_index_sum(raw_latex)
        raw_latex = self._apply_display_bridge(raw_latex, registry)

        # §2  Maxima-specific postprocessing
        raw_latex = _apply_dot_notation_to_latex(canonical, raw_latex, registry)
        # Strip time-argument from time-dependent quantities: q\left(t\right) → q
        # Maxima includes the (t) function-call syntax in output, but it's noise
        # in a physics notebook where q implicitly depends on t.
        for _qname, _prop in registry.registry.items():
            _qty = _prop.quantity
            # Only strip q(t) → q for dot-notation quantities.
            # Leibniz-notation quantities (e.g. p\left(t\right)) stay as-is.
            if _qty.time_dependent and getattr(_qty, "notation", None) == "dot":
                _t_par = _qty.time_parameter or "t"
                _base_lat = _qty.latex_name or _qname
                # Maxima LaTeX form: q\left(t\right)
                raw_latex = raw_latex.replace(
                    f"{_base_lat}\\left({_t_par}\\right)",
                    _base_lat,
                )
                # Plain form that may appear in intermediate strings: q(t)
                # Use lambda replacement to avoid backslash interpretation in repl.
                if not _base_lat.startswith("\\"):
                    _repl_val = _base_lat
                    raw_latex = re.sub(
                        r'(?<![A-Za-z_])' + re.escape(_base_lat)
                        + r'\(' + re.escape(_t_par) + r'\)',
                        lambda _, r=_repl_val: r, raw_latex
                    )
        raw_latex = _over_to_frac(raw_latex)
        # \mbox is a Maxima output artefact; KaTeX requires \text
        raw_latex = re.sub(r'\\mbox\{([^{}]*)\}', r'\\text{\1}', raw_latex)
        # Maxima wraps multi-char identifiers as \mbox{ {\it H\_hat} }.
        # Unwrap so that restore_decorations() can recognise the suffix.
        raw_latex = re.sub(
            r'\\mbox\s*\{\s*\{\\it\s+([^{}]+)\}\s*\}',
            lambda m: m.group(1).replace(r"\_", "_"),
            raw_latex,
        )
        # Lagrangian mode: q_dot / q_ddot are independent atoms; Maxima renders
        # them as q_{dot} / q_{ddot} (braced subscript) which restore_decorations
        # cannot recognise.  Map the braced subscript form to \dot / \ddot here.
        if self.is_lagrangian_mode():
            raw_latex = re.sub(r'([A-Za-z][A-Za-z0-9]*)_\{dddot\}', r'\\dddot{\1}', raw_latex)
            raw_latex = re.sub(r'([A-Za-z][A-Za-z0-9]*)_\{ddot\}',  r'\\ddot{\1}',  raw_latex)
            raw_latex = re.sub(r'([A-Za-z][A-Za-z0-9]*)_\{dot\}',   r'\\dot{\1}',   raw_latex)
        # B-2: restore decoration suffixes to LaTeX commands
        from notebook.runtime.latex_converter import restore_decorations
        raw_latex = restore_decorations(raw_latex)
        return self.strip_placeholder_sum_range(
            _strip_outer_parens(raw_latex.replace(r"\,", " ")))

    def localize_imaginary_unit(self, result, adapter, registry):
        """Electrical engineering writes the imaginary unit as ``j`` (the compute
        side maps ``j→%i``).  Swap ``%i→j`` in the result VALUE — an unambiguous
        token, unlike Maxima's rendered bare ``i`` which could be a subscript /
        superscript / exponent index — then re-``tex1`` the ``j`` form so the display
        is consistent and never touches a non-imaginary ``i``."""
        if not (getattr(result, "ok", False)
                and "%i" in (getattr(result, "value", "") or "")
                and "electrical_engineering" in getattr(registry, "domains", [])):
            return result
        result.value = result.value.replace("%i", "j")
        try:
            tx = adapter.execute(f"tex1({result.value})")
            if getattr(tx, "ok", False) and tx.value:
                result.latex = tx.value.strip()
        except Exception:
            pass
        return result

    def hold_aggregates(self, code: str) -> str:
        return quote_aggregates(code)

    def format_constant_for_latex(self, token: str) -> str:
        return constant_to_latex(token)

    def parse_quantity(self, canonical: str, defining_expr_id: str) -> Quantity | None:
        from notebook.runtime.maxima_ir_compiler import MaximaIRCompiler
        from notebook.models.ir import Equation, FunctionCall, Symbol

        compiler = MaximaIRCompiler()
        ir = compiler.parse(canonical)
        if not isinstance(ir, Equation):
            return None

        lhs = ir.lhs
        rhs = ir.rhs

        if isinstance(lhs, FunctionCall):
            if not isinstance(lhs.func, Symbol):
                return None
            name = lhs.func.name
            if name in MATH_FUNC_NAMES:
                return None
            func_args = [a.name for a in lhs.args if isinstance(a, Symbol)]
            return Quantity(
                name=name,
                kind="function",
                natural_vars=func_args,
                defining_expr_id=defining_expr_id,
            )

        if isinstance(lhs, Symbol):
            # Collect variable names directly from the IR.  The RHS here is a
            # SURFACE AST (no SemanticResolver ran), so it may still contain
            # Juxtaposition / curly BracketExpr nodes ("P = e^{j*w*t}") that
            # the base ExprGenerator cannot render — round-tripping through
            # generate() used to leak the pydantic repr into natural_vars.
            from notebook.runtime.maxima_semantic_resolver import _collect_symbols
            _names: set = set()
            _collect_symbols(rhs, _names)
            rhs_vars = sorted(n for n in _names if n not in MATH_FUNC_NAMES)
            return Quantity(
                name=lhs.name,
                kind="scalar",
                natural_vars=rhs_vars,
                defining_expr_id=defining_expr_id,
            )

        return None

    def canonical_variables(self, canonical: str) -> set[str]:
        # Bracket-aware: `"=" in text` read a bounded sum's own subscript as an
        # equation and dropped its head, so `\sum_{i=1}^N*a_i*x` reported the
        # variables of `1}^N*a_i*x` — the dummy `i` silently missing from a set
        # that decides, among other things, whether a derivative is partial.
        # Confirmed fed this shape by a source-level audit over the full suite
        # (2026-08-11); this call site is the busiest of them.
        text = strip_lhs(canonical)
        text = _DIFF_STRIP_RE.sub("", text)
        # Maxima escapes the quote INSIDE a name it returns, so the RESULT of
        # differentiating `y'^3` comes back as `3*y\'^2` — and with the
        # backslash standing between them the scan below sees `y`, not `y'`,
        # which is the very defect this method was just fixed for.  **A widening
        # opens a new exit; count them again**: measured 2026-09-24, the SECOND
        # derivative (`E2 を y' で微分して` on the previous result) still asked
        # "partial or total?".  The escape is the CAS's own spelling of the same
        # character, so it is removed before the name is read.
        text = text.replace("\\'", "'")
        # A TeX command is not a variable.  Only visible once the head stopped
        # being thrown away above: `\sum_{i=1}^3*i^2` then reported `sum_`
        # alongside `i`, because the identifier scan below reads `\sum` as a
        # name.  Commands are dropped, their braces left, so the INDICES inside
        # them still count.
        text = re.sub(r"\\[A-Za-z]+", " ", text)
        # **A PRIME IS PART OF THE NAME** — the parser says so (`f'(x)`, and
        # CLAUDE.md's "Math Input Conventions"), and `y'` / `t'` / `\sigma'` are
        # ordinary spellings here.  Without the `'*` this scan reported `y'^3`'s
        # variables as `{y}`, so `differentiate_is_ambiguous` saw a symbol that
        # is NOT the differentiation variable and asked "partial or total?"
        # about an expression with ONE symbol in it — in both languages
        # (measured 2026-09-24, PRIMED-VAR-NOT-A-VARIABLE).  Julia's own
        # `canonical_variables` deliberately does the opposite: there a prime is
        # the ADJOINT mark and `a'(k)` is the operator `a`, which is why it is a
        # different question answered in that handler.
        return {
            tok
            for tok in re.findall(r"[A-Za-z_]\w*'*", text)
            # a bare `_` is the subscript MARK left behind by the command strip
            # above, not a name
            if tok not in MATH_FUNC_NAMES and tok.strip("_")
        }

    def qty_name_from_origin(self, origin: str) -> str:
        from notebook.runtime.maxima_ir_compiler import MaximaIRCompiler
        from notebook.models.ir import FunctionCall, Symbol

        compiler = MaximaIRCompiler()
        ir = compiler.parse(origin)
        if isinstance(ir, FunctionCall) and isinstance(ir.func, Symbol):
            return ir.func.name
        elif isinstance(ir, Symbol):
            return ir.name
        return origin.strip()

    def is_lagrangian_mode(self) -> bool:
        """Analytical-mechanics (Lagrangian) mode: q and its time derivatives are
        independent variables.  Gated on the analytical_mechanics domain so that
        pure classical / general calculus keep the dependent d/dt = dq/dt."""
        ctx = self._context
        return bool(ctx is not None and "analytical_mechanics" in getattr(ctx, "domains", []))

    def wrt_qdot_canonical(self, wrt_name: str, t: str,
                           source_canonical: str = "") -> str:
        r"""How ``∂/∂q̇``'s variable is spelled — **the way the expression spells it**.

        In Lagrangian mode q̇ is an independent atom (``q_dot``).  Otherwise the
        quantity is a time derivative, and a session holds it in one of two
        spellings: the atom the researcher typed (``\dot{\theta}`` registers as
        ``theta_dot``) or the CAS's own ``'diff(theta,t,1)``.

        **The old answer was a THIRD spelling that matched neither**:
        ``'diff(theta(t),t,1)``.  `depends([theta],[t])` is exactly the mechanism
        that makes `theta` a function of t WITHOUT writing `theta(t)` in the
        expressions, so nothing in the session ever carried that form — and
        Maxima answers a derivative with respect to something absent with ``0``.
        Measured 2026-09-25 against a live CAS::

            diff(m*r^2*'diff(theta,t,1), 'diff(theta,t,1), 1)     →  m*r^2
            diff(m*r^2*'diff(theta,t,1), 'diff(theta(t),t,1), 1)  →  0
            diff(m*r^2*theta_dot,        theta_dot, 1)            →  m*r^2

        So the canonical momentum came back as **0, `completed`** — and the same
        sentence on the same expression answered `m r^2` when the quantity had
        NOT been declared time-dependent.  The declaration, made lines earlier,
        silently changed the answer.
        """
        if self.is_lagrangian_mode():
            return f"{wrt_name}_dot"
        # **The expression decides which of the two it is.**  A researcher who
        # types CAS text (`diff(q(t),t)`) keeps the function form; one who types
        # their own notation (`\dot{q}`) has `q_dot`, which
        # `_prepare_expr_lookup` maps to the BARE noun — so the variable has to
        # follow the same map or the CAS is asked about a quantity that is not
        # there (measured 2026-09-25: the canonical momentum came back as 0).
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(wrt_name)}\s*\(\s*"
                     rf"{re.escape(t)}\s*\)", source_canonical or ""):
            return f"'diff({wrt_name}({t}),{t},1)"
        return time_dep_derivative(wrt_name, t, 1)

    def declare_depends(self, adapter: object, var: str, depends_on: list[str]) -> None:
        deps = ",".join(depends_on)
        adapter.execute(shadow_coordinate_variables(depends_on, f"depends([{var}],[{deps}])"))  # type: ignore[union-attr]

    @property
    def supports_equation_source(self) -> bool:
        return True

    def to_time_function_form(self, canonical: str, names: "list[str]", t: str) -> str:
        """Rewrite each time-dependent symbol ``q`` → ``q(t)`` so Maxima's
        ``laplace`` treats it as a function (it ignores ``depends`` on a bare atom).
        Also rewrites it inside ``'diff(q,t,n)`` → ``'diff(q(t),t,n)``.  Skips names
        already in function form (``q(``)."""
        import re as _re_tf
        for name in names:
            canonical = _re_tf.sub(
                rf"(?<![A-Za-z0-9_\\]){_re_tf.escape(name)}(?![A-Za-z0-9_(])",
                f"{name}({t})", canonical)
        return canonical

    def normalize_bound(self, bound: str) -> str:
        r"""Maxima integration bound: π → %pi, ∞ → inf, and then the CONVERTER.

        A bound is ordinary written mathematics, and this used to answer that
        question a second time with string replacement — worse than the
        converter that already answers it.  Five spellings split (measured
        2026-08-16): `2\pi`/`n\pi`/`2\pi R` lost the implicit `*`, and
        `\frac{\pi}{2}`/`\sqrt{2}`/`\alpha` stayed LaTeX.  `2%pi` is not
        Maxima, so `\int_0^{2\pi} r \, d\theta` came back as `_res_`, the
        adapter's own scratch name, REGISTERED AS A SUCCESS.

        What only this method knows is the backend's spelling of π and ∞ and
        the unicode/bare forms a researcher types.  It does that much and hands
        the rest to :meth:`latex_to_canonical` rather than restating it.
        """
        import re as _re_nb
        from notebook.runtime.backend_handler import normalize_unicode_greek
        b = normalize_unicode_greek(str(bound))
        b = b.replace("∞", r"\infty")
        # Bare `pi` → `\pi` (NOT `%pi`) so the converter still sees a command and
        # puts the implicit `*` in: `2 pi` → `2 \pi` → `2*%pi`.  Already-Maxima
        # `%pi` is excluded by the lookbehind, as is `\pi` itself.
        #
        # A leading DIGIT RUN is taken with it (`2pi`), which is safe precisely
        # because no identifier may start with a digit — so a `pi` reached only
        # through digits is π and nothing else.  `\b` before the run is what
        # keeps `x2pi` and `k_pi` whole: neither has a word boundary there.
        # Without this the method was not idempotent (`2pi` → `2*pi` → `2*%pi`),
        # and it is called more than once on the same bound.
        b = _re_nb.sub(r"(?<![%\\])\b(\d+)?pi\b", r"\1\\pi", b)
        return self.latex_to_canonical(b)

    def restore_derivative_display_latex(self, s: str) -> str:
        """Rewrite every derivative DISPLAY form in *s* back to Maxima's noun
        form: ``\\frac{d^2}{d t d z} A_{z}`` → ``'diff(A_z,t,1,z,1)``,
        ``\\frac{\\partial phi}{\\partial x}`` → ``'diff(phi,x,1)``.

        Both the shapes Maxima's own tex1 produces and the ones
        ``lhs_formatting._format_provenance_lhs`` produces are covered; the
        grammar and the refuse-to-guess rules live in
        ``maxima_derivative_latex`` (BP-9 / H-10).

        The dot forms of order 3 and up (``\\dddot{q}``, ``{q}^{(4)}``) are
        restored FIRST, so that a fraction wrapping one is refused with a
        reason rather than read with the accent command as its operand.
        """
        from notebook.runtime.maxima_derivative_latex import (
            restore_derivative_display_latex as _restore,
            restore_dot_display_latex as _restore_dot)
        s = _restore_dot(s, self._latex_operand_to_canonical,
                         self._dot_notation_quantity)
        return _restore(s, self._latex_operand_to_canonical,
                        on_refused=self._warn_unreadable_derivative)

    def _dot_notation_quantity(self, latex_base: str) -> "tuple[str, str] | None":
        """``(canonical name, time parameter)`` for a displayed dot-notation base.

        The lookup accepts the LaTeX name as well as the canonical one because
        ``_apply_dot_notation_to_latex`` prints ``qty.latex_name or name`` — the
        inverse has to start from whatever it wrote.
        """
        ctx = getattr(self, "_context", None)
        if ctx is None or not latex_base:
            return None
        for name, prop in ctx.registry.items():
            qty = prop.quantity
            if not (qty.time_dependent and getattr(qty, "notation", None) == "dot"):
                continue
            if latex_base in (name, qty.latex_name or name):
                return name, qty.time_parameter or "t"
        return None

    def _warn_unreadable_derivative(self, fraction: str, reason_key: str) -> None:
        """A fraction that looks like a derivative but could not be read.

        Processing continues — the fraction stays an ordinary division, exactly
        as before this restorer existed — but the researcher is told, because a
        half-recognised derivative that quietly becomes a division is precisely
        the silent-wrong-answer failure BP-9 exists to stop.
        """
        import logging
        from notebook.runtime.i18n import t as _t_warn
        _lang = self.lang
        logging.getLogger(__name__).warning(
            _t_warn("notation.deriv_unreadable", _lang, frac=fraction,
                    reason=_t_warn(reason_key, _lang)))

    def _latex_operand_to_canonical(self, fragment: str) -> str:
        """Convert an operand / variable fragment for the derivative restorer.

        Goes straight to the converter rather than through
        ``latex_to_canonical``, which now calls the restorer itself — the
        fragment can never contain the outer derivative, so there is nothing to
        re-restore, and this keeps the two out of a mutual-recursion loop.
        """
        from notebook.runtime.maxima_latex_converter import MaximaLatexConverter
        return MaximaLatexConverter().to_canonical(fragment)

    def lhs_latex_to_canonical(self, lhs: str) -> str:
        """Convert LaTeX dot-notation LHS to Maxima derivative canonical form.

        ``\\ddot{q}`` → ``'diff(q(t),t,2)`` when q is time-dependent;
        ``\\frac{d^2}{d t d z} A_{z}`` → ``'diff(A_z,t,1,z,1)`` (H-10).
        """
        import re as _re_dlc
        # Base may be a plain name (x, q_1) or a LaTeX Greek command (\theta, \phi);
        # the latter's backslash is stripped so it resolves to the registered name.
        _m = _re_dlc.match(r"\\(d*dot)\{(\\?[A-Za-z]\w*)\}$", lhs.strip())
        if _m:
            _order_map = {"dot": 1, "ddot": 2, "dddot": 3}
            _order = _order_map.get(_m.group(1))
            _base  = _m.group(2).lstrip("\\")
            if _order is not None:
                _ctx = getattr(self, "_context", None)
                _prop = _ctx.get(_base) if _ctx else None
                _qty  = _prop.quantity if _prop else None
                _tpar = (getattr(_qty, "time_parameter", None) or "t") if _qty else "t"
                if _qty and getattr(_qty, "time_dependent", False):
                    return f"'diff({_base}({_tpar}),{_tpar},{_order})"
                return f"'diff({_base},{_tpar},{_order})"
        return self.restore_derivative_display_latex(lhs)

    def literal_derivative_canonical(self, dep: str, indep: str) -> "str | None":
        """D-7: literal-division derivative notation (``dy/dx``) → Maxima's
        unevaluated first-derivative noun form ``'diff(y,x)`` — the same form
        ``solve_ode``/``solve_ode_general`` (ode2) expect as their equation
        argument."""
        return f"'diff({dep},{indep})"

    # Dot-notation (\dot{X}, \ddot{X}, \dddot{X}) is a user-facing input form, not
    # CAS syntax.  Left in the canonical string it reaches the IR pipeline
    # (parse_ir → MaximaSemanticResolver → CodeGenerator) which renders \dot{theta}
    # as the bogus Maxima function dot(theta).  Normalize it to the atom form
    # (theta_dot / theta_ddot) at every entry point so register / differentiate /
    # substitute all share one representation; build_command's apply_time_dep_map
    # then maps the atom to the depends form 'diff(theta,t,N) for time-dependent
    # quantities (and Lagrangian mode keeps the atom).  Greek bases (\dot{\theta})
    # are handled by stripping the leading backslash of the captured base.
    _DOT_ATOM_RES = [
        (re.compile(r'\\dddot\{(\\?[A-Za-z]\w*)\}'), '_dddot'),
        (re.compile(r'\\ddot\{(\\?[A-Za-z]\w*)\}'),  '_ddot'),
        (re.compile(r'\\dot\{(\\?[A-Za-z]\w*)\}'),   '_dot'),
    ]

    def normalize_dot_notation(self, s: str) -> str:
        """Replace LaTeX dot-notation with the canonical atom form (no-op otherwise)."""
        if not s or '\\' not in s:
            return s
        for _re_dot, _suffix in self._DOT_ATOM_RES:
            s = _re_dot.sub(lambda m: m.group(1).lstrip('\\') + _suffix, s)
        return s

    def pre_process_canonical(self, canonical: str) -> str:
        # normalize_matrix_literal ("[2,1;1,2]" → matrix([2,1],[1,2])) is inherited
        # from BackendHandler; the ';' form is otherwise mis-parsed by Maxima as a
        # statement separator when bound via P2 ("A : …").
        return self.normalize_matrix_literal(self.normalize_dot_notation(canonical))

    def latex_to_canonical(self, latex: str) -> str:
        # BP-9: restore the derivative display forms FIRST.  Left to the generic
        # \frac{a}{b} → (a)/(b) rule they become silently wrong Maxima — `d` and
        # `\partial` degrade to ordinary symbols, so \frac{\partial phi}{\partial x}
        # cancels to phi/x with no error at all.  The restorer rewrites only what
        # parses unambiguously as a derivative; every other fraction is untouched
        # and still becomes a division.
        from notebook.runtime.latex_typography import normalize_command_aliases
        from notebook.runtime.maxima_latex_converter import MaximaLatexConverter
        # …but the size-hinted aliases must be made one spelling BEFORE the
        # restorer reads, or \dfrac{dy}{dx} is not recognised as a derivative
        # at all (it lowered to the product `dy * dx^-1`).
        return MaximaLatexConverter().to_canonical(
            self.restore_derivative_display_latex(
                normalize_command_aliases(latex)))

    def execute_multi_statement(self, adapter, operation: str, command: str):
        """``define_coordinate_system`` builds one ``nb_*`` function per LINE, so
        it must be fed to Maxima statement by statement (``execute_script``).
        Which operations do that is the handler's knowledge; the wire protocol
        and the result type are the adapter's (AUDIT-SHARED-CAS-LEAK)."""
        if operation != "define_coordinate_system" or "\n" not in command:
            return None
        return adapter.execute_script(command)

    def generate_backend_code(self, ir: "Expr", hold_integrals: bool = False) -> "str | None":
        """Render resolved IR as Maxima code (docs/32 Phase 0 hook).

        Uses the SAME generator as build_command's per-argument loop, so an
        expression body and an operation argument can no longer disagree about
        what ``V_i`` means (open_items AUDIT-SUBST-INDEXED).
        """
        from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
        return MaximaCodeGenerator(self._context or SymbolContext(),
                                   hold_integrals=hold_integrals).generate(ir)

    def parse_ir(self, canonical: str, before_alphatize=None) -> "Expr | None":
        # NOTE: UnknownNotationError from SharedParser propagates untranslated —
        # the engine boundary (NotebookEngine._translate_unknown_notation) owns
        # the translation because only it knows the session language.
        from notebook.runtime.maxima_ir_compiler import MaximaIRCompiler
        from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver, PrimeDerivativeError
        from notebook.runtime.i18n import t
        parsed = MaximaIRCompiler().parse(canonical)
        if parsed is not None and getattr(self, "_context", None) is not None:
            try:
                return MaximaSemanticResolver().resolve(parsed, self._context, before_alphatize=before_alphatize)
            except PrimeDerivativeError as exc:
                msg = t(exc.i18n_key, self.lang, **exc.fmt_kwargs)
                raise ValueError(msg) from exc
        return parsed

    # ------------------------------------------------------------------
    # Undefined symbol detection
    # ------------------------------------------------------------------

    _KNOWN_SYMBOL_NAMES: frozenset[str] = frozenset({
        # Maxima math functions
        "sin", "cos", "tan", "cot", "sec", "csc",
        "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "exp", "log", "sqrt", "abs", "signum",
        "floor", "ceiling", "round", "sign",
        "max", "min", "gcd", "lcm", "mod",
        "diff", "integrate", "limit", "sum", "product",
        "ratsimp", "expand", "factor", "simplify",
        "realpart", "imagpart", "conjugate", "transpose",
        "determinant", "invert", "eigenvalues",
        # Names this system itself emits as display-only functionals (R-9's
        # bracket functional, and the expectation bracket that lowers to the
        # same one).  Measured 2026-08-20: without them the ⚠ banner told the
        # researcher that `expectation` was undefined — a name they never
        # wrote, in a formula where they wrote `\langle A B \rangle` (and,
        # already, `E[X]`).
        "expectation", "variance",
        # Maxima constants
        "pi", "inf", "minf", "true", "false",
        # Greek letters (conventional standalone symbols)
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
        "eta", "theta", "iota", "kappa", "lambda", "mu",
        "nu", "xi", "rho", "sigma", "tau", "upsilon",
        "phi", "chi", "psi", "omega",
        # Uppercase
        "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi",
        "Sigma", "Phi", "Psi", "Omega",
    })

    def find_undefined_symbols(self, ir_ast: "Expr | None", context: "SymbolContext") -> list[str]:
        if ir_ast is None:
            return []
        result: list[str] = []
        seen: set[str] = set()
        self._scan_undefined(ir_ast, context, result, seen)
        return result

    def _scan_undefined(self, node, context, result: list, seen: set) -> None:
        from notebook.models.ir import (
            Symbol, Equation, Add, Mul, Pow, FunctionCall,
            Derivative, Integral, Subscript, Superscript, IndexedSymbol, IndexSymbol,
        )
        if node is None:
            return
        if isinstance(node, Equation):
            if node.op == "=":
                # LHS is being defined — scan only RHS
                self._scan_undefined(node.rhs, context, result, seen)
            else:
                # Inequality/relation constraint: neither side is a
                # definition, so both must be checked (e.g. both x_A and
                # x_B in "2*x_A + x_B <= 100").
                self._scan_undefined(node.lhs, context, result, seen)
                self._scan_undefined(node.rhs, context, result, seen)
            return
        if isinstance(node, IndexedSymbol):
            # Reconstruct full name e.g. IndexedSymbol(H, [0]) → "H_0"
            # and check the full name in SymbolContext before falling through to
            # base-symbol scanning (which would falsely report "H" as undefined).
            base = getattr(node.base, "name", None) if isinstance(node.base, Symbol) else None
            if base:
                idx_parts = []
                for idx in (node.indices or []):
                    if isinstance(idx, IndexSymbol):
                        idx_parts.append(idx.name)
                    elif isinstance(idx, Symbol):
                        idx_parts.append(idx.name)
                full_name = "_".join([base] + idx_parts) if idx_parts else base
                if context.get(full_name) is not None:
                    return  # registered — not undefined
                # full_name not registered; report it (not the bare base)
                if (full_name
                        and full_name not in self._KNOWN_SYMBOL_NAMES
                        and not full_name.startswith("%")
                        and full_name not in seen):
                    seen.add(full_name)
                    result.append(full_name)
                return
            # Fall through for unusual IndexedSymbol shapes
        if isinstance(node, Symbol):
            name = node.name.strip()
            # Single lowercase-letter symbols (x, t, i, j, …) are conventional
            # free variables in physics notation.  Maxima handles them as
            # symbolic atoms without declaration, so we never warn about them.
            # Uppercase single-letter symbols (K, H, V, …) are often meaningful
            # physics quantities and are still subject to undefined checks.
            if len(name) == 1 and name.islower():
                return
            if (name
                    and name not in self._KNOWN_SYMBOL_NAMES
                    and not name.startswith("%")
                    and context.get(name) is None
                    and name not in seen):
                seen.add(name)
                result.append(name)
            return
        for field_name in getattr(type(node), "model_fields", {}):
            val = getattr(node, field_name, None)
            if val is None:
                continue
            if getattr(type(val), "model_fields", None) is not None or hasattr(val, "__dataclass_fields__"):
                self._scan_undefined(val, context, result, seen)
            elif isinstance(val, list):
                for item in val:
                    if getattr(type(item), "model_fields", None) is not None or hasattr(item, "__dataclass_fields__"):
                        self._scan_undefined(item, context, result, seen)
