"""Julia-specific implementation of BackendHandler."""
from __future__ import annotations
import json
import logging
import re

from notebook.runtime.nl_patterns import nlp, nlp_any, nlp_search, pick_group


def _pick_suffixed(m, stem: str) -> "str | None":
    r"""The value of ``stem`` from whichever suffixed group matched.

    One language may spell a request several ways, and Python forbids repeating
    a group NAME across alternatives — so the alternatives are written
    ``opA``/``opB``/… and coalesced here.  Same shape as `subst_replace`'s
    ``_sr_pick``, which is the existing precedent in nl_patterns.json.
    """
    gd = m.groupdict()
    for suffix in "ABCDE":
        value = gd.get(stem + suffix)
        if value:
            return value.strip()
    return None
from pathlib import Path

_logger = logging.getLogger(__name__)

from notebook.models.quantity import Quantity
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.backend_handler import BackendHandler, BuildResult, GREEK_CMD
from notebook.runtime.shared_parser import split_relation
from notebook.runtime.i18n import t as _t

from notebook.runtime import julia_error_signatures as _JEC
from notebook.runtime.expr_ref_fields import (
    IDENTIFIER_KEYS as _IDENTIFIER_KEYS, is_expr_ref)
from notebook.runtime.julia_handler_common import (
    _GREEK_CMD,
    _DOMAIN_TO_JULIA_TYPE,
    _JULIA_VOID_OPS,
    _OP_BASE_NAME_RE,
    _collect_dummy_names,
    _relabel_index_in_canonical,
    _collect_scalar_fn_names,
    _adjoint_ir,
    _normalize_idx_for_sign,
    _is_neg_idx,
    _extract_pos_base,
    _subst_quantum_op,
    _split_top_level_commas,
    _EXPR_ID_RE,
    _JULIA_OP_RE,
    _JULIA_BUILTINS,
    _EXPRESSION_ARG_FIELDS,
)


from notebook.runtime.julia_sympy_mixin import JuliaSympyMixin
from notebook.runtime.julia_build_mixin import JuliaBuildMixin
from notebook.runtime.julia_undefined_mixin import JuliaUndefinedSymbolMixin
from notebook.runtime.julia_latex_mixin import JuliaLatexMixin


class _BuildCtx:
    """Shared context assembled once at the start of each ``build_command`` call.

    Holds the parser / resolver / symbol context / compiled IR args so that
    individual operation handlers don't need to repeat the setup boilerplate.
    """

    __slots__ = ("parser", "resolver", "ctx", "ir_args", "expr_lookup")

    def __init__(
        self,
        parser: "SharedParser",
        resolver: "JuliaSemanticResolver",
        ctx: "SymbolContext",
        ir_args: dict,
        expr_lookup: dict,
    ) -> None:
        self.parser = parser
        self.resolver = resolver
        self.ctx = ctx
        self.ir_args = ir_args
        self.expr_lookup = expr_lookup

    def nb_var_for_expr(self, expr_ref: str) -> "str | None":
        """Return the Julia source expression for *expr_ref*.

        If the canonical is already a Julia variable (``_nb_En``), returns it
        directly.  Otherwise parses ``input_canonical`` through the IR pipeline
        and returns an inline ``normal_form(…)`` call.
        """
        canonical = self.expr_lookup.get(expr_ref, "")
        if canonical.startswith("_nb_"):
            return canonical
        ic = self.expr_lookup.get(f"__input__{expr_ref}", canonical)
        if not ic:
            return None
        surf = self.parser.parse(ic)
        if surf is None:
            return None
        ir = self.resolver.resolve(surf, self.ctx)
        from notebook.models.ir import Equation as _EqIR
        if isinstance(ir, _EqIR):
            ir = ir.rhs
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator as _JCG
        _jcg = _JCG(context=self.ctx)
        _jcg.static_index_mode = True
        return f"normal_form({_jcg.generate(ir)})"

    def julia_expr_arg(self, expr_ref: str) -> str:
        """Return Julia source for *expr_ref* as a direct value or a string literal."""
        nb = self.nb_var_for_expr(expr_ref)
        if nb:
            return nb
        escaped = expr_ref.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'


#: Words that may follow a closing paren in an ENGLISH instruction without
#: being a factor.  Only the ones the NL patterns in this file actually use as
#: structure; adding a word here says "this is never a symbol name".
_EN_INSTRUCTION_WORDS = (
    "for", "and", "with", "into", "in", "to", "of", "by", "from", "where",
    "substitute", "using", "as", "on", "at", "then", "the",
    # H-4's "this index only" modifier reaches this rule as `Sz(i) only`, and
    # without the exclusion it became `Sz(i) * only` — after which the modifier
    # no longer sat against the call and was never seen.
    "only",
)
_EN_INSTRUCTION_JUXT_RE = __import__("re").compile(
    r"\)\s+(?!(?:" + "|".join(_EN_INSTRUCTION_WORDS) + r")\b)([A-Za-z])")


class JuliaBackendHandler(JuliaBuildMixin, JuliaSympyMixin, JuliaUndefinedSymbolMixin, JuliaLatexMixin, BackendHandler):
    """Backend handler for Julia (second-quantization) sessions."""

    #: スキーマの ``supported_backends`` での自分の綴り。
    _SCHEMA_BACKEND = "julia_full"

    #: Symbolics.jl has no `assume`; a symbol carries a TYPE and nothing else.
    #: `_ASSUME_FN` stays empty (inherited), so `assumption_command` refuses —
    #: this only makes the refusal say WHY, in this backend's own terms, and
    #: point at the handoff the way `julia_quexpr_not_differentiable` does.
    _ASSUME_UNSUPPORTED_KEY = "error.julia_assumption_unsupported"

    #: The fallback `_nb_eval_index_sum` path produces an expression that
    #: merely CARRIES a summation index and renders no Σ, so the caller still
    #: has to paste one on.  The `∑` path renders its own and is detected by
    #: content (`result_latex_has_sum`).
    needs_sum_prefix: bool = True

    # Julia uses adjoint/conj instead of Maxima's ctranspose/conjugate.
    _MATRIX_OP_RESTORE = [
        ("adjoint",   r"{\1}^{\\dagger}"),
        ("conj",      r"{\1}^{\\ast}"),
        ("transpose", r"{\1}^{T}"),
    ]

    # Latexify.jl renders a bare operator name wrapped: `b'(1)` comes back as
    # `\mathrm{b'}\left( 1 \right)`.  That is the CAS's spelling of the adjoint,
    # not the researcher's — measured 2026-08-04 leaking all the way into the
    # timeline for every solve / inverse-transform result (the shipped S-c
    # example showed `\mathrm{g'}\left( k, u \right)` beside a correctly
    # rendered `{c}_{k}^{\dagger}` from QuantumAlgebra's own latex).
    _DISPLAY_NAME_WRAPPERS = [r"\\mathrm"]

    # _display_bridge_scalar_guard: base-class default arithmetic-char set
    # (BackendHandler._SCALAR_GUARD_ARITH_CHARS) applies unchanged here.

    def __init__(self, builder: "JuliaCommandBuilder") -> None:  # type: ignore[name-defined]
        self._builder = builder
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator
        if self._builder is not None:
            self._builder._compiler = JuliaCodeGenerator()
        self._context: "SymbolContext | None" = None

    # ------------------------------------------------------------------
    # build_command: public entry point + per-operation handlers
    # ------------------------------------------------------------------

    def _make_build_ctx(self, args: dict, expr_lookup: dict) -> _BuildCtx:
        """Compile IR args and assemble a shared :class:`_BuildCtx`."""
        import re
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.models.ir import Equation as _Equation

        parser = SharedParser()
        resolver = JuliaSemanticResolver()
        ctx = self._context or SymbolContext()

        # Populate the code generator's input_lookup so that
        # _maybe_expand_commutator can look up original user canonicals.
        if (self._builder is not None
                and getattr(self._builder, "_compiler", None) is not None):
            _comp = self._builder._compiler
            _comp.input_lookup = {
                _key[len("__input__"):]: _val
                for _key, _val in expr_lookup.items()
                if _key.startswith("__input__") and isinstance(_val, str)
            }

        # Compile all string / list args through the IR pipeline.
        ir_args: dict = {}
        for k, v in args.items():
            if k in _IDENTIFIER_KEYS:
                # A NAME, not an expression.  This side had no such notion until
                # 2026-08-04, so `name: "k_B"` was parsed into `k(B)`.
                ir_args[k] = v
            elif isinstance(v, str):
                raw = (
                    expr_lookup[v]
                    if is_expr_ref(v) and v in expr_lookup
                    # **パース前の正規化はここに並ぶ。** 矢印はパーサが
                    # `\uparrow` を境界として読むので何も要らないが、整数スピンの
                    # 射影は素の符号なので、名前へ置き換えるのはパーサの手前で
                    # ある（`latex_to_canonical` と同じ規則・同じ位置づけ。この
                    # 経路は canonical を経ずに生の LaTeX を渡すので、両方に
                    # 通す必要がある——実測 2026-09-12、`a_{1,+}` が生のまま
                    # Julia に届いて `ParseError:` が研究者に返っていた）。
                    else self._projection_signs_to_names(
                        self._apply_spin_component_norm(v))
                )
                surf = parser.parse(raw)
                ir_args[k] = resolver.resolve(surf, ctx) if surf is not None else v
            elif isinstance(v, list):
                ir_args[k] = []
                for item in v:
                    if isinstance(item, str):
                        raw = (
                            expr_lookup[item]
                            if is_expr_ref(item) and item in expr_lookup
                            else self._apply_spin_component_norm(item)
                        )
                        surf = parser.parse(raw)
                        ir_args[k].append(
                            resolver.resolve(surf, ctx) if surf is not None else item
                        )
                    elif isinstance(item, dict):
                        # op_rules-style list-of-dicts (operator_substitute
                        # multi-rule): expand bare E-id string fields (e.g.
                        # "rhs": "E6") to their canonical text, the same way a
                        # top-level string arg is expanded above.  Without this,
                        # a multi-rule "rhs" naming an E-id (e.g. a build_
                        # inverse_transform result) is parsed as a literal
                        # unknown symbol named "E6" instead of its formula.
                        ir_args[k].append({
                            _dk: (expr_lookup[_dv]
                                  if is_expr_ref(_dv)
                                  and _dv in expr_lookup else _dv)
                            for _dk, _dv in item.items()
                        })
                    else:
                        ir_args[k].append(item)
            else:
                ir_args[k] = v

        # Unwrap equation forms (H = rhs) → rhs so QA operations receive QuExpr.
        for k in list(ir_args):
            if isinstance(ir_args[k], _Equation):
                ir_args[k] = ir_args[k].rhs

        return _BuildCtx(parser, resolver, ctx, ir_args, expr_lookup)

    def build_expr_restore_command(self, expr_id: str, expr: "Expression") -> str | None:
        """Build a Julia command that materializes _nb_{expr_id} for a registered expression.

        Called by the engine's auto-restore mechanism when a registered expression
        (one that was never CAS-derived) needs to exist as a Julia variable.
        Returns None when the expression cannot be parsed/generated.
        """
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator
        from notebook.models.ir import Equation as _EqIR

        input_canonical = getattr(expr, "input_canonical", None) or getattr(expr, "canonical", None)
        if not input_canonical or input_canonical.startswith("_nb_"):
            return None

        parser = SharedParser()
        resolver = JuliaSemanticResolver()
        ctx = self._context or SymbolContext()

        surface = parser.parse(input_canonical)
        if surface is None:
            return None
        ir = resolver.resolve(surface, ctx)
        if isinstance(ir, _EqIR):
            ir = ir.rhs

        jcg = JuliaCodeGenerator(context=ctx)
        jcg.static_index_mode = True
        julia_code = jcg.generate(ir)
        return f"_nb_{expr_id} = normal_form({julia_code})"

    def wrap_command_for_storage(self, command: str, output_id: str | None, operation: str) -> str:
        if output_id and operation not in _JULIA_VOID_OPS:
            nb_var = f"_nb_{output_id}"
            return f"{nb_var} = ({command})"
        return command

    def canonical_for_result(self, cas_value: str, output_id: str | None, operation: str) -> str:
        if output_id and operation not in _JULIA_VOID_OPS:
            return f"_nb_{output_id}"
        return cas_value

    def parse_quantity(self, canonical: str, defining_expr_id: str) -> Quantity | None:
        return None

    def canonical_variables(self, canonical: str) -> set[str]:
        if not canonical or canonical.startswith("_nb_"):
            return set()
        names = {m.group(1) for m in _JULIA_OP_RE.finditer(canonical)}
        return names - _JULIA_BUILTINS

    def qty_name_from_origin(self, origin: str) -> str:
        return origin.strip()

    def wrt_qdot_canonical(self, wrt_name: str, t: str,
                           source_canonical: str = "") -> str:
        return f"'diff({wrt_name}({t}),{t},1)"

    def declare_depends(self, adapter: object, var: str, depends_on: list[str]) -> None:
        pass

    @property
    def supports_equation_source(self) -> bool:
        return False

    def classify_cas_error(self, error: str) -> str | None:
        """Recognize QuantumAlgebra failures that stem from using an operator
        constructor without applying it to an index/site.

        Writing ``Sx`` (or ``c``) on its own leaves a ``QuExprConstructor`` —
        a function still waiting for its site/momentum argument — and
        multiplying it produces an opaque ``MethodError: no method matching
        *(::QuExpr, ::QuExprConstructor{…})``.  The user almost always meant a
        concrete component such as ``Sx_1`` or ``c(k)`` (A-11).
        """
        if not error:
            return None
        if "MethodError" in error and "QuExprConstructor" in error:
            return "error.operator_missing_index"
        if _JEC.UNDECLARED_CALL_RE.search(error):
            return "error.julia_undeclared_call"
        if _JEC.MULTICHAR_INDEX_RE.search(error):
            return "error.julia_multichar_index"
        if _JEC.QUEXPR_DIFF_RE.search(error):
            return "error.julia_quexpr_not_differentiable"
        return None

    def result_warnings(self, operation: str, adapter) -> "list[str]":
        """`pair_substitute` whose rules reached no term at all.

        The driver counts the matches — that is where the matching happens, and
        deciding it again here would be a second answer to one question — and
        leaves the verdict in `_nb_pair_no_match`.  Read back with one small
        query rather than folded into the command's value: the value IS the
        researcher's expression, and putting a flag in it would corrupt what
        they get.
        """
        _flags = {"pair_substitute": ("_nb_pair_no_match",
                                      "warning.julia_pair_substitute_no_match"),
                  "wick_contract": ("_nb_wick_no_match",
                                    "warning.julia_wick_no_match")}
        _entry = _flags.get(operation)
        if _entry is None or adapter is None:
            return []
        _var, _key = _entry
        try:
            r = adapter.execute(_var)
        except Exception:
            return []
        if r.ok and str(r.value).strip() == "true":
            return [_key]
        return []

    def cas_error_args(self, error: str) -> dict:
        """The offending name, so the message can say which symbol to declare."""
        m = _JEC.UNDECLARED_CALL_RE.search(error or "")
        if m:
            return {"name": m.group(1)}
        m = _JEC.MULTICHAR_INDEX_RE.search(error or "")
        if m:
            return {"index": m.group(1)}
        return {}

    # A-11 case B: when the whole session refers to a single, unambiguous site
    # (e.g. a one-spin Larmor model H = -γB Sz_1), complete a site-unspecified
    # spin operator (Sx → Sx_1) instead of erroring.  Stays inert with 0 or ≥2
    # sites (genuinely ambiguous in a multi-site lattice → the case-A error).
    completes_operator_sites = True

    def _spin_component_names(self) -> "set[str]":
        """Registered spin-component operator names (Sx, Sy, Sz, …)."""
        ctx = self._context
        if not ctx:
            return set()
        sfx = set(self._SPIN_COMP_SUFFIX.values())
        return {n for n in ctx.registry if len(n) > 1 and n[-1] in sfx}

    def spin_sites_in(self, canonical: str) -> "set[str]":
        sites: "set[str]" = set()
        for name in self._spin_component_names():
            for m in re.finditer(re.escape(name) + r"(?:\((\d+)\)|_(\d+))",
                                 canonical or ""):
                sites.add(m.group(1) or m.group(2))
        return sites

    def complete_bare_spin_sites(self, canonical: str, site: str) -> str:
        out = canonical or ""
        for name in self._spin_component_names():
            out = re.sub(re.escape(name) + r"(?![A-Za-z0-9_(])", f"{name}_{site}", out)
        return out

    # Matches operator^\dagger / operator^{\dagger} / operator^\dag (all forms)
    def pre_process_canonical(self, canonical: str) -> str:
        """Apply spin-arrow normalization so c(k, \\uparrow) → c(k, u) before parsing.

        Also calls _apply_spin_component_norm so that:
        - S^z_1 / S^{z}_1 are normalised to Sz_1 in the canonical form, and
        - the user's notation style is recorded in spin_display_styles on first use,
          even for direct formula registrations (H-1 equation-form input) that do not
          go through build_command / _make_build_ctx.
        """
        from notebook.runtime.spin_label_canon import canon_spin_arrows
        canonical = canon_spin_arrows(canonical)
        # "[2,1;1,2]" → matrix([2,1],[1,2]) (neutral form; JuliaCodeGenerator
        # renders it to the native [2 1; 1 2]).  Inherited from BackendHandler.
        canonical = self.normalize_matrix_literal(canonical)
        return self._apply_spin_component_norm(canonical)

    def replay_session_declarations(self, adapter: object, commands: list[str]) -> None:
        """Re-execute operator/variable declarations into a fresh Julia process.

        Called by NotebookEngine.start() after session load.  A failed
        declaration must not block the session from opening, so errors are
        caught — but logged (warning) so a broken replay leaves a trace rather
        than silently producing a half-declared session.
        """
        for cmd in commands:
            try:
                adapter.execute(cmd)
            except Exception:
                _logger.warning("failed to replay declaration command: %s",
                                cmd, exc_info=True)

    # treats_bracket_as_commutator: 既定が「このバックエンドに二項括弧の操作が
    # あるか」を訊くようになったので、`return True` のオーバーライドは不要に
    # なった（2026-09-06）。消したことで**行列リテラルの除外**（`;` を含む形）も
    # Julia 側に効くようになる——以前は無条件 True だった。

    # operands_provided: the `[X, Y]` / `{X, Y}` bracket-notation check now
    # lives in BackendHandler.operands_provided (shared, not Julia-specific).

    # ------------------------------------------------------------------
    # Nested commutator / anticommutator parsing helpers
    # ------------------------------------------------------------------



    def parse_ir(self, canonical: str, before_alphatize=None) -> "Expr | None":
        # Julia canonicals are variable references (_nb_E3), not parseable math.
        # Nothing is resolved here, so there is no IR to hand to the hook.
        return None

    def parse_for_undefined_scan(self, canonical: str) -> "Expr | None":
        r"""Resolve the canonical before the undefined-symbol scan reads it.

        `parse_ir` returns `None` here because a Julia canonical is usually the
        handle `_nb_E3`; the engine's fall-back therefore parses the canonical
        of a USER-INPUT expression instead. Left at the SURFACE AST, a binder is
        indistinguishable from a symbol — measured 2026-09-17,
        `\lim_{x \to 0} \frac{\sin x}{x}` registers with the canonical
        `\lim_{x,to,0} ((sin(x))/(x))` and the scan reported **「未定義の記号:
        lim, to」**: the researcher wrote neither, they are the two halves of
        the limit's own canonical spelling (`limit_subscript_latex`). The
        generator's `julia_limit` refusal — the one this codebase wrote for
        exactly this case, "書けて表示はできる、断るのは使用地点" — sits behind
        that guard and was never reached.

        **The resolver already answers correctly** (measured: both backends'
        resolvers turn that canonical into a `Limit` node), so the fix is to ask
        it rather than to add `lim`/`to` to `_KNOWN_SYMBOL_NAMES` — which would
        put an internal spelling into the table of names a researcher may use.

        Resolution is best-effort: a canonical this resolver cannot read falls
        back to the surface AST, which is what the engine used to do for
        everything.
        """
        from notebook.models.ir import Limit
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        parsed = SharedParser().parse(canonical)
        if parsed is None or getattr(self, "_context", None) is None:
            return parsed
        try:
            resolved = JuliaSemanticResolver().resolve(parsed, self._context)
        except Exception:                                   # noqa: BLE001
            # 読めない綴りでも走査は続ける——ここで例外を上げると、断りが
            # 「未定義の記号」から「解決器の内部エラー」に変わるだけである。
            return parsed
        # **解決したものを使うのは、根が極限のときだけ。** 広げ方を2回測って
        # 2回とも狭めた（2026-09-17）:
        #
        # | 使う範囲 | コーパス差分 | 壊れたもの |
        # |---|---|---|
        # | 解決したものを常に使う | **48行** | `\sum_n \varepsilon_n c'(n) c(n)` の `epsilon_n` が走査に見え、動いていたグリーン関数のシナリオが止まる |
        # | 根が `Limit`/`Integral`/`IndexSum` | **6行** | `\sum_\sigma a^\dagger_{1\sigma} a_{1\sigma}` で同じこと（`a_1_sigma`） |
        # | 根が `Limit` **だけ** | **0行** | —— |
        #
        # `Integral` と `IndexSum` は「束縛子だから同じはず」という**推測**で足した
        # ものだった。実測で要ったのは極限だけで、この系の規則どおり——**実在も
        # 実害も測れないものは足さない**。
        return resolved if isinstance(resolved, Limit) else parsed

    def missing_variable_expr_id(self, error: str) -> "str | None":
        """Julia reports a cleared ``_nb_E12`` as ``UndefVarError: '_nb_E12'``.

        Scaffolding (``_nb__T3``, INTERNAL-INTERMEDIATE-EID-GAP) is stored and
        lost the same way, so it is matched and restored the same way.
        """
        if "UndefVarError" not in error:
            return None
        _m = re.search(r"UndefVarError: ['\"`]_nb_(E\d+|_T\d+)['\"`]", error)
        return _m.group(1) if _m else None

    def is_opaque_handle(self, text: str) -> bool:
        """A ``_nb_<id>`` canonical is the name of a value living in the Julia
        session, not an expression — it must reach Julia verbatim
        (docs/32 Phase 0 hook)."""
        return text.strip().startswith("_nb_")

    _particles_cache: "list[dict] | None" = None

    @classmethod
    def _load_particles(cls) -> "list[dict]":
        # ONE table, loaded by the neutral module: the particle list is physics,
        # not this backend's knowledge, and the shared engine needs it too
        # (spin projections).  Keeping the load here made `engine.py` import a
        # BACKEND to learn what an electron is.
        from notebook.runtime.spin_value_set import load_particles
        return load_particles()

    def extra_pre_phase1(self, text: str, lang: str) -> "object | None":
        """Parse Julia-specific patterns before Phase 1, then fall back to shared patterns.

        Checks in order:
        1. Spin operator definition: 'S を スピン演算子 とする'
        2. Particle operator definition: 'X を 粒子 の 消滅/生成演算子 とする'
        3. Shared type declarations (scalar / vector / matrix / tensor) via super()

        Returns an InterpretResult on match, None to fall through to Phase 1.
        """
        from notebook.runtime.llm_supervisor import InterpretResult

        # --- 0-heis. Heisenberg equation of motion: dedicated rule-based
        # trigger, bypassing the generic alias-scan + positional-token-filler
        # (which was found to SHRED a 2-argument operator call like "c(k,u)"
        # on its internal comma when the surrounding text has no Japanese
        # particles to anchor tokenization — e.g. English "c(k,u) Heisenberg
        # equation E1" fed operator='c(k', hamiltonian='u)' to the op).
        #   JP: "c(k,u) の Heisenberg 方程式 E1" / "E1 と c(k,u) の Heisenberg 方程式"
        #   EN: "Heisenberg equation of c(k,u) for E1" /
        #       "Heisenberg equation of E1 for c(k,u)"
        # English: "Heisenberg equation of X for Y" — X and Y may appear in
        # either order (one is the E-id hamiltonian, the other the operator
        # expression), so classify by shape rather than fixing the order.
        # `E1 のもとでの a(1) のハイゼンベルク方程式` —— ハミルトニアンが先に来る
        # 語順で、**`のもと` がそれを名指す**。位置ではなく助詞が決めるので、
        # 名前付きグループで両言語とも同じ2欄を読む。直す前は専用パターンに
        # 当たらず**別名走査が位置で割り当て**、演算子とハミルトニアンが入れ替わって
        # `dH/dt = +i w a₁` という符号まで逆の式が `completed` で返っていた
        # （実測 2026-09-06。`dH/dt` は保存量なので本来 0 である）。
        for _l in ("ja", "en"):
            _hm = nlp("heisenberg_eom_under_first", _l).search(text)
            if _hm:
                return InterpretResult(
                    ok=True, operation="heisenberg_equation",
                    args={"operator": _hm.group("op"),
                          "hamiltonian": _hm.group("ham")},
                    confidence=1.0, source="rule", requires_confirmation=False,
                )
        _hm = nlp("heisenberg_eom_op_first", "ja").search(text)
        if _hm:
            return InterpretResult(
                ok=True, operation="heisenberg_equation",
                args={"operator": _hm.group(1), "hamiltonian": _hm.group(2)},
                confidence=1.0, source="rule", requires_confirmation=False,
            )
        _hm = nlp("heisenberg_eom_eid_first", "ja").search(text)
        if _hm:
            return InterpretResult(
                ok=True, operation="heisenberg_equation",
                args={"operator": _hm.group(2), "hamiltonian": _hm.group(1)},
                confidence=1.0, source="rule", requires_confirmation=False,
            )
        _hm = nlp("heisenberg_eom_verb_first", "en").search(text)
        if _hm:
            _a, _b = _hm.group(1), _hm.group(2)
            _ham, _op = (_a, _b) if re.fullmatch(r'E\d+', _a) else (_b, _a)
            return InterpretResult(
                ok=True, operation="heisenberg_equation",
                args={"operator": _op, "hamiltonian": _ham},
                confidence=1.0, source="rule", requires_confirmation=False,
            )

        # --- 0-pre. Normalize LaTeX subscript notation to canonical form ---
        # Converts:  c_{k} = u_{k} \alpha_{k} + v_{k} \beta^{\dagger}_{-k}
        #       to:  c(k) = u(k) * alpha(k) + v(k) * beta'(-k)
        # so that the operator-substitute patterns below can match.
        # The 0-pre normalizers (_normalize_latex_subscripts / _canon_spin) are
        # methods now — see below.  Apply normalization so LaTeX-subscript inputs
        # match the operator-substitute patterns.
        normalized = self._normalize_latex_subscripts(text)

        # Spin component notation: S^z(1) → Sz(1), S^x → Sx, etc.
        # Two orderings handled: S^z_1→S^z(1)→Sz(1), S_1^z→S(1)^z→Sz(1).
        normalized = self._apply_spin_component_norm(normalized)

        normalized, _exact_index = self._read_exact_index_modifier(normalized, lang)

        # --- 0. Operator substitution (Bogoliubov-style) ---
        # Route to operator_substitute when the LHS symbol is a registered
        # quantum operator in the SymbolContext, regardless of whether the
        # user wrote "演算子代入" or plain "代入".
        #
        # Catches:
        #   "E1 の c(k) に RHS を演算子代入して"  (explicit)
        #   "E1 の c(k) に RHS を代入して"        (plain 代入, operator detected)
        #   "E1 に c(k) = RHS を代入して"         (equation form)
        #   "operator_substitute E1 c(k) with RHS" (English)
        #
        # Pattern extracts: (expr_ref, op_name, formal_idx, rhs)
        # Two Japanese forms captured in one alternation:

        def _make_opsubst(expr, op_name, rhs, formal_index=None):
            args = {"expr": expr, "op_name": op_name, "rhs": rhs.strip(),
                    **({"exact_index": True} if _exact_index else {})}
            if formal_index:
                fi = formal_index.strip()
                if "," in fi:
                    # 2-index op: "k, ↑" → formal_index="k", spin_label="u"
                    parts = [p.strip() for p in fi.split(",", 1)]
                    args["formal_index"] = parts[0]
                    if len(parts) > 1 and parts[1]:
                        args["spin_label"] = self._canon_spin(parts[1])
                else:
                    args["formal_index"] = fi
            return InterpretResult(
                ok=True, operation="operator_substitute", args=args,
                confidence=1.0, source="rule", requires_confirmation=False,
            )

        # --- 0a-spin. Same operator, two different static (spin) indices ---
        # Catches: "E1のc(k,u)にRHS1を、c(-k,d)にRHS2を代入"
        # Both rules share the same base operator but differ in the static (spin) label.
        # Must be checked before PAIR (same-op-different-k) and MULTI (different-op).
        # English: "In E1, substitute RHS1 for op(idx1,spin1) and RHS2 for
        # op(idx2,spin2)".  English puts the RHS before the operator call
        # (opposite of the Japanese postposition order above), so this is a
        # separate pattern with named groups rather than a positionally
        # compatible alternation.
        # A stray "*" may have been inserted between the closing paren and
        # "and" by upstream implicit-multiplication normalization (which
        # treats an identifier-like word after ")" as a multiplied term) —
        # tolerate it defensively rather than depending on that pass being
        # English-aware.
        # --- 0a. 2つ組の演算子代入（日英とも1パターン、形が行き先を決める）---
        # 日本語はここが **3つ**あった（spin_pair / pair / multi）。英語は最初から
        # 1つで、そのコメントが理由を書いている——「英語は3つの形を同じ綴りで書く
        # ので、1つのパターンが読み、形が行き先を決める」。日本語がその設計に
        # 追いついていなかっただけで、3つに分ける理由は無い（2026-09-05）。
        #
        # **その英語側も 2つあった**（`multi` と `spin_pair`）。理由は綴りではなく
        # **添字の文字クラス**で、`multi[en]` が `[^,)]`——コンマで止まる——なので
        # `c(k,u)` のように添字を2つ持つ演算子が当たらず、`spin_pair` が
        # 「同じ演算子名・`substitute … for …` の綴りだけ」を拾って埋めていた。
        # 埋まらない綴りは下の単数形に落ち、**2つ目の代入が黙って捨てられて**
        # `completed` が返っていた（実測 2026-09-10、英語 3/6）:
        #
        #     In E1, substitute E2 for c(k,u) and E3 for d(-k,d)   ← 違う演算子
        #     In E1, replace c(k,u) with E2 and c(-k,d) with E3     ← replace 形
        #
        # 日本語の `multi[ja]` は最初から `[^)]` だった。クラスを揃えたら英語は
        # 6/6 になり、`spin_pair` は**1綴りも増やさなくなった**ので消した（実測）。
        for _lang in ("ja", "en"):
            _m2 = nlp("operator_substitute_multi", _lang).search(normalized)
            if _m2 is None:
                continue
            _r = self._two_rule_substitute_result(_m2, text, _exact_index)
            if _r is not None:
                return _r

        # Match against normalized text so LaTeX-subscript inputs work
        m = nlp("operator_substitute_single", "ja").search(normalized)
        if m:
            expr_ref  = m.group(1)
            op_name   = m.group(2)
            formal_idx = m.group(3).strip()
            rhs       = (m.group(4) or m.group(5) or "").strip()
            ctx = self._context
            prop = ctx.get(op_name) if ctx else None
            if prop and prop.quantity.kind == "operator":
                return _make_opsubst(expr_ref, op_name, rhs, formal_index=formal_idx)
            if "演算子代入" in text:
                return _make_opsubst(expr_ref, op_name, rhs, formal_index=formal_idx)

        # `operator_substitute_single[en]` は 2026-09-05 に削除した——綴りが
        # `operator_substitute E1 a(1) with E2`（操作名そのもの）で、シナリオ文書が
        # 自ら「不自然で CAS 構文のような綴り」と書いており、原則#1 に反する。
        # しかも実測で動いていなかった（**式として登録**されていた）。自然な形は
        # 下の `operator_substitute_in_form` で、そちらは動く。

        # Natural English single-clause form: "In E1, substitute mz for
        # Sz(2)" — mirrors the Japanese particle form above ("E1 の Sz(2) に
        # mz を代入して") but with English word order (RHS before the
        # operator call).  A trailing "* " artifact from upstream implicit-
        # multiplication insertion is tolerated defensively, same as the
        # spin double-substitution pattern above.
        m = nlp("operator_substitute_in_form", "en").search(normalized)
        if m:
            # `substitute Y for X(i)` と `replace X(i) with Y` で群の並びが違う
            # ——値の入っている枝を取る（`re` は同名グループの重複を禁じる）。
            _g = [x for x in m.groups() if x is not None]
            expr_ref, rhs, op_name, formal_idx = (
                (_g[0], _g[1].strip(), _g[2], _g[3].strip()) if len(_g) >= 4 and
                m.group(2) is not None else
                (_g[0], _g[3].strip(), _g[1], _g[2].strip()))
            ctx = self._context
            prop = ctx.get(op_name) if ctx else None
            if prop and prop.quantity.kind == "operator":
                return _make_opsubst(expr_ref, op_name, rhs, formal_index=formal_idx)

        # --- 0b. relabel_index: dummy sum-index relabeling (k → -k etc.) ---
        # Catches:
        #   "E1 の k を -k に置き換えて"
        #   "E1 の k を -k で指標変換して"
        #   "E1 で k → -k と置き換えて"
        #   "relabel k to -k in E1"
        # 日英で**同じ群名**なので、言語ごとに呼び分ける必要はない
        # （`nlp_search` は全言語を順に試す）。2026-09-10 まで位置引数で、
        # しかも並びが日英で逆だったため、ここは2回呼んで群番号を書き分けて
        # いた——`nl_patterns.py` の docstring が「語順の違いは名前付き群で
        # 吸収する」と書いている当の設計に、この鍵だけ従っていなかった。
        m = nlp_search("relabel_index", text)
        if m:
            return InterpretResult(
                ok=True, operation="relabel_index",
                args={"expr": m.group("eid"), "old_index": m.group("old"),
                      "new_index": m.group("new")},
                confidence=1.0, source="rule", requires_confirmation=False,
            )

        # --- 0b2. collect_by_operators ---
        # Catches:
        #   "E3の演算子でまとめて"  "E3を演算子係数でまとめる"
        #   "E3の同類項をまとめて"  "collect operators in E3"
        #   "collect like terms in E3"  "group E3 by operators"
        # **言語は直書きしない。** 直す前はここが `"ja"` 固定で、英語の綴りは
        # ja のパターンに埋め込まれていた——ja は E-id が先という日本語語順を
        # 要求するので、動詞が先に来る自然な英語は1つも当たらなかった（実測
        # 2026-09-07。上のコメントが挙げていた `collect operators in E3` すら）。
        for _l in ("ja", "en"):
            m = nlp("collect_operator_terms", _l).search(text)
            if m:
                return InterpretResult(
                    ok=True, operation="collect_by_operators",
                    args={"expr": pick_group(m, "eid")},
                    confidence=1.0, source="rule", requires_confirmation=False,
                )

        # --- 0c. Scalar parameter substitution ---
        # Catches:
        #   "E6 で ξ_{-k} に ξ_k を代入する"
        #   "E6 の ξ_{-k} を ξ_k に置き換えて"
        #   "E6 で xi(-k) を xi(k) に置き換え"
        m = nlp("scalar_substitute_op", "ja").search(normalized)
        if m:
            expr_ref = m.group(1)
            lhs_raw  = m.group(2).strip()
            rhs_raw  = m.group(3).strip()
            # Only route to substitute if lhs and rhs share the same base name
            # (avoid misrouting operator substitutions).
            # XStar (complex conjugate) → strip "Star" suffix for base comparison.
            lhs_base = re.sub(r'[_\(].*$', '', lhs_raw)
            rhs_base = re.sub(r'[_\(].*$', '', rhs_raw)
            lhs_base_norm = re.sub(r'Star$', '', lhs_base)
            rhs_base_norm = re.sub(r'Star$', '', rhs_base)
            if lhs_base == rhs_base or lhs_base_norm == rhs_base_norm:
                ctx = self._context
                prop = ctx.get(lhs_base_norm) if ctx else None
                if prop is None or prop.quantity.kind in ("scalar", "anonymous"):
                    return InterpretResult(
                        ok=True, operation="substitute",
                        args={"expr": expr_ref, "lhs": lhs_raw, "rhs": rhs_raw},
                        confidence=1.0, source="rule", requires_confirmation=False,
                    )

        # --- 1. Spin operator definition ---
        _spin_def = self._pre_spin_def(text)
        if _spin_def is not None:
            return _spin_def

        # --- 3. Particle operator definition (also the method's fallback to
        # super() when nothing Julia-specific matched).
        return self._pre_particle_def(text, lang)

    # ------------------------------------------------------------------
    # extra_pre_phase1 normalizers (R4/所見D, docs/27): the 0-pre helpers, promoted
    # from mid-method closures to methods so the families below read cleanly.
    # ------------------------------------------------------------------
    def _normalize_latex_subscripts(self, s: str) -> str:
        """0-pre. Normalize LaTeX subscript notation to canonical form, e.g.
        `c_{k} = u_{k} \\alpha_{k}` → `c(k) = u(k) * alpha(k)`, so the
        operator-substitute patterns can match."""
        # Spin arrows → the canonical arrow (before Greek mapping).
        # ↑/↓ and LaTeX \uparrow/\downarrow are unambiguously spin VALUES.
        from notebook.runtime.spin_label_canon import canon_spin_arrows
        s = canon_spin_arrows(s)
        # Unicode Greek letters → ASCII names (e.g. ξ → xi, Δ → Delta)
        _UNICODE_GREEK = {
            'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
            'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
            'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu',
            'ν': 'nu', 'ξ': 'xi', 'π': 'pi', 'ρ': 'rho',
            'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon', 'φ': 'phi',
            'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
            'Δ': 'Delta', 'Ξ': 'Xi', 'Π': 'Pi', 'Σ': 'Sigma',
            'Φ': 'Phi', 'Ψ': 'Psi', 'Ω': 'Omega',
        }
        for uc, name in _UNICODE_GREEK.items():
            s = s.replace(uc, name)
        _GREEK = {
            r'\alpha': 'alpha', r'\beta': 'beta', r'\gamma': 'gamma',
            r'\delta': 'delta', r'\epsilon': 'epsilon', r'\varepsilon': 'epsilon',
            r'\xi': 'xi', r'\eta': 'eta', r'\theta': 'theta',
            r'\kappa': 'kappa', r'\lambda': 'lambda', r'\mu': 'mu',
            r'\nu': 'nu', r'\rho': 'rho', r'\sigma': 'sigma',
            r'\tau': 'tau', r'\phi': 'phi', r'\chi': 'chi',
            r'\psi': 'psi', r'\omega': 'omega',
            r'\Delta': 'Delta', r'\Xi': 'Xi', r'\Pi': 'Pi',
        }
        for cmd, name in sorted(_GREEK.items(), key=lambda x: -len(x[0])):
            s = s.replace(cmd, name)
        # X^{\dagger} / X^{\dag} / X^† → X'
        s = re.sub(r"([A-Za-z][A-Za-z0-9_']*)\s*\^\{?\\(?:dagger|dag)\}?", r"\1'", s)
        s = re.sub(r"([A-Za-z][A-Za-z0-9_']*)\^[†]", r"\1'", s)
        # Complex conjugate: X^{*} / X^{\ast} / X^(\ast) / X^* → XStar
        # (Matches the internal Julia canonical where conj(X) is emitted as XStar)
        s = re.sub(r"([A-Za-z][A-Za-z0-9_]*)\s*\^\{?\\?(?:ast|\*)\}?", r"\1Star", s)
        s = re.sub(r"([A-Za-z][A-Za-z0-9_]*)\^?\(\s*\\?(?:ast|\*)\s*\)", r"\1Star", s)
        # Bare X* as conjugate when followed by Japanese particle or whitespace
        s = re.sub(r"([A-Za-z][A-Za-z0-9_]*)\*(?=[にをで\s]|$)", r"\1Star", s)
        # X_{idx} / X'_{idx} → X(idx)  (braces and bare single-char)
        s = re.sub(r"([A-Za-z][A-Za-z0-9_]*'?)\s*_\{([^}]+)\}", r"\1(\2)", s)
        # Negated bare subscript without braces: xi_-k → xi(-k)
        s = re.sub(r"([A-Za-z][A-Za-z0-9_]*'?)_(-[A-Za-z0-9]+)(?![A-Za-z0-9])", r"\1(\2)", s)
        # Bare single-char subscript: xi_k → xi(k)
        # Context-aware: if the compound name (e.g. mz) is registered as a
        # scalar in SymbolContext, treat the subscript as part of the name
        # (m_z → mz) rather than as a function argument (m_z → m(z)).
        # This preserves compound physics parameters like mz, nx, phi0.
        def _subscript_or_compound(m_re: "re.Match") -> str:
            base_prime = m_re.group(1)   # e.g. "m" or "xi'"
            sub        = m_re.group(2)   # e.g. "z" or "k"
            compound   = base_prime.rstrip("'") + sub
            prop = self._context.get(compound) if self._context else None
            if prop and prop.quantity.kind in ("scalar", "anonymous"):
                return base_prime + sub   # m_z → mz
            return base_prime + "(" + sub + ")"   # xi_k → xi(k)
        s = re.sub(
            r"([A-Za-z][A-Za-z0-9_]*'?)_([A-Za-z0-9](?![A-Za-z0-9]))",
            _subscript_or_compound, s,
        )
        # Implicit multiplication: adjacent terms "u(k) alpha(k)" → "u(k) * alpha(k)"
        #
        # This runs over the WHOLE natural-language instruction, not over an
        # expression, so an English structure word standing after a closing
        # paren was read as a factor: "substitute u_k a(k) for c(k,u) and …"
        # became "u(k) * a(k) * for c(k,u) * and …", and the RHS the pattern
        # then captured ended in a bare `*`.  That does not fail — measured
        # 2026-08-17, `u(k) * a(k) *` parses as a SYMBOL whose NAME is that
        # whole string, so the substitution's right-hand side silently stopped
        # being a product and became one opaque atom.  Japanese never hit it
        # because its particles are not `[A-Za-z]`.
        #
        # The exclusion is the same shape as `_separate_adjacent_commands`'s
        # `\left`/`\right` carve-out: these words are the INSTRUCTION'S syntax,
        # not factors, and a physics symbol is never spelled one of them.
        s = _EN_INSTRUCTION_JUXT_RE.sub(r") * \1", s)
        return s

    def _read_exact_index_modifier(self, text: str, lang: str) -> "tuple[str, bool]":
        r"""Strip a "this index ONLY" modifier, and say whether it was there (H-4).

        Returns ``(text_without_the_modifier, exact_index)``.

        ``operator_substitute`` reads a SYMBOLIC formal index as a template
        covering every index.  That is right for a Bogoliubov transform
        (``c(k) → u a(k) + v b(-k)`` means "for all k") and wrong for a
        mean-field decoupling on one site.  Measured 2026-08-17 against a live
        Julia session::

            H = J Sz_i Sz_j , substitute Sz(i) → Mz   →  H = Mz^2 J      (BOTH)
            ... the same, with exact_index=True       →  H = J Mz Sz_j

        **The filter already existed.**  ``exact_index`` was implemented end to
        end — schema field, builder, substituter — and the only thing missing
        was a way for a RESEARCHER to ask for it.  A CONCRETE index (``Sz(1)``,
        ``Sz(A)``) is auto-detected as exact; a symbolic one cannot be, because
        the text alone does not say whether ``i`` names a site or stands for any
        site.  So the researcher says which, and this is where they say it.

        The default is deliberately unchanged: both readings are legitimate and
        the notation does not choose between them.

        **The modifier must sit against the operator CALL, index and all.**
        ``Sz(i) だけ`` names one instance; a bare ``Sz だけ`` is about WHICH
        OPERATOR and says nothing about the index.  Stripping it here rather
        than threading it through each pattern keeps the (large) substitution
        regexes below reading exactly as they did.
        """
        rx = nlp("op_subst_exact_index", lang)
        if not rx.search(text):
            return text, False
        return rx.sub("", text), True

    def _two_rule_substitute_result(self, m, text: str, exact_index: bool):
        """2つ組の演算子代入 —— **形が行き先を決める**、日英で1つの答え。

        以前は日本語に3パターン（spin_pair / pair / multi）、英語に1パターン＋
        source 内の正規表現があり、同じ問い（「この2つの規則はどの操作か」）に
        4通りの答えがあった。振り分けの規則そのものは英語側が既に正しく書いて
        いたので、それをここに1つ置いて両言語が使う。

        - 同じ演算子で添字が `X` と `-X` …… Nambu の対（`operator_substitute_pair`
          という別の操作）。
        - 引数が `(添字, スピン)` …… `spin_label` を持つ2規則。
        - それ以外 …… 独立な2規則。
        """
        from notebook.runtime.llm_supervisor import InterpretResult
        # 群名は日英で同じ（`nl_patterns.py` の設計）。2026-09-10 まで日本語だけ
        # `a1` / `r1a` と綴っており、ここが**両方を試して**いた——同じ問いに
        # 2つの答えがある形で、綴りを揃えたら読みは1つになった。
        eid = pick_group(m, "eid")
        op1, op2 = pick_group(m, "op1"), pick_group(m, "op2")
        a1 = (pick_group(m, "idx1") or "").strip()
        a2 = (pick_group(m, "idx2") or "").strip()
        r1 = (pick_group(m, "rhs1") or "").strip()
        r2 = (pick_group(m, "rhs2") or "").strip()
        if not (eid and op1 and op2 and r1 and r2):
            return None
        ctx = self._context
        declared = any(
            (ctx.get(o) is not None
             and getattr(ctx.get(o).quantity, "kind", None) == "operator")
            for o in (op1.rstrip("'"), op2.rstrip("'"))) if ctx else False
        if not (declared or "演算子代入" in text):
            return None

        # 形① 同じ演算子の ±X → Nambu の対（それ自身が別の操作）
        if op1 == op2 and a2 == f"-{a1}":
            return InterpretResult(
                ok=True, operation="operator_substitute_pair",
                args={"expr": eid, "op_name": op1, "formal_index": a1,
                      "rhs": r1, "rhs_neg": r2},
                confidence=1.0, source="rule", requires_confirmation=False,
            )

        # 形② 引数が (添字, スピン) → spin_label を持つ2規則
        def _rule(op, arg, rhs):
            if "," in arg:
                idx, spin = (x.strip() for x in arg.split(",", 1))
                return {"op_name": op, "formal_index": idx,
                        "spin_label": self._canon_spin(spin), "rhs": rhs}
            return {"op_name": op, "formal_index": arg, "rhs": rhs}

        return InterpretResult(
            ok=True, operation="operator_substitute",
            args={
                "expr": eid,
                **({"exact_index": True} if exact_index else {}),
                "op_rules": [_rule(op1, a1, r1), _rule(op2, a2, r2)],
            },
            confidence=1.0, source="rule", requires_confirmation=False,
        )

    def _canon_spin(self, label: str) -> str:
        """Spin label → single-char QuIndex canonical (QuantumAlgebra.QuIndex needs
        single-char names, so multi-char labels from Greek normalization collapse
        here).  Applied after _normalize_latex_subscripts (so "↑"→"u" is done).
        The table is shared with JuliaCodeGenerator (spin_label_canon)."""
        from notebook.runtime.spin_label_canon import canon_spin_label
        canon = canon_spin_label(label)
        if canon:
            return canon
        return label[0] if label else label  # fallback: first char

    # ------------------------------------------------------------------
    # extra_pre_phase1 declaration families (R4/所見D, docs/27): each family is a
    # method returning an InterpretResult on match, or None to fall through.  The
    # dispatcher above tries them in the original firing order.
    # ------------------------------------------------------------------
    def _pre_spin_def(self, text: str) -> "object | None":
        """1. Spin operator definition: 'S を スピン演算子 とする'."""
        from notebook.runtime.backend_handler import read_spin_operator_declaration
        from notebook.runtime.llm_supervisor import InterpretResult
        # ONE reader, in the shared layer: how a DECLARATION SENTENCE is read is
        # not this backend's knowledge, and keeping it here meant a Maxima
        # session could not read it at all (measured 2026-08-22 — every spelling
        # fell through to `define_scalar_parameter`).
        _read = read_spin_operator_declaration(text)
        spin_op_expr = _read[0] if _read else None

        if spin_op_expr is not None:
            if not _EXPR_ID_RE.match(spin_op_expr):
                base_m = _OP_BASE_NAME_RE.match(spin_op_expr)
                if base_m:
                    base_name = base_m.group(1)
                    if len(base_name) > 1 and base_name[-1] in ('x', 'y', 'z', 'p', 'm'):
                        base_name = base_name[:-1]

                    return InterpretResult(
                        ok=True,
                        operation="define_spin_operator",
                        args={
                            "name": base_name,
                            "_op_expr_display": spin_op_expr,
                        },
                        confidence=1.0,
                        source="rule",
                        requires_confirmation=False,
                    )
        return None

    def _pre_particle_def(self, text: str, lang: str) -> "object | None":
        """3. Particle operator definition ('X を 粒子 の 消滅/生成演算子 とする'), the
        bare fermion/boson case, and the fall-through to the shared
        super().extra_pre_phase1 when nothing Julia-specific matched."""
        from notebook.runtime.llm_supervisor import InterpretResult
        op_expr: str | None = None
        particle_text: str | None = None
        op_type_key: str | None = None

        # ONE lookup instead of four branches tried in a fixed order.  The
        # spellings are language CONTENT and live in nl_patterns.json; the
        # word-order differences (Japanese puts the particle before the operator,
        # English after it, and the particle-FIRST English spelling puts it
        # between) are absorbed by named groups, so this reads the same three
        # fields whichever spelling matched.
        _pm = nlp("particle_operator_decl", lang).search(text)
        if _pm:
            op_expr = _pick_suffixed(_pm, "op")
            particle_text = _pick_suffixed(_pm, "particle")
            _kind = (_pick_suffixed(_pm, "type") or "")
            # **既定を置かない。** 直す前は `else: creation` で、向きの語彙を
            # 広げると下降演算子が生成演算子として宣言される（呼び名の表は
            # `notebook/runtime/particle_ladder.py`）。
            from notebook.runtime.particle_ladder import ladder_kind
            op_type_key = ladder_kind(_kind)
            if op_type_key is None:
                op_type_key = "creation" if "生成" in _kind else "annihilation"

        if op_expr is None or particle_text is None:
            # 粒子も統計も書かれていない「c を消滅演算子とする」。**どちらとも
            # 決めない**——書かれていないものを埋めるより、書かれていないと言う。
            # 直す前は型ラベルの既定に落ちて `define_scalar_parameter` を登録し
            # 成功を返していた（実測 2026-09-06、日英）。
            _bare_ladder = nlp("bare_ladder_op_decl", lang).search(text)
            if _bare_ladder and not nlp("bare_quantum_op_decl", lang).search(text):
                from notebook.runtime.i18n import t as _t
                return InterpretResult(
                    ok=False, operation=None, args={}, confidence=1.0,
                    source="rule",
                    error=_t("interpreter.ladder_op_statistics_missing", lang,
                             name=_bare_ladder.group("name")))
            # Bare fermion/boson (no particle name): "X をフェルミオン/ボゾン演算子とする"
            # Must intercept before super() which would misclassify as scalar parameter.
            m_bare = nlp("bare_quantum_op_decl", lang).search(text)
            if m_bare:
                op_name = m_bare.group("name")
                kind_text = m_bare.group("kind")
                # Both languages: the `kind` was captured by whichever
                # pattern matched, and the statistics is physics, not the
                # session's language (`fermion_kind_kw`).
                is_fermion = nlp_any("fermion_kind_kw", kind_text)
                operation = (
                    "define_fermion_operator" if is_fermion else "define_boson_operator"
                )
                op_disp = self._canonical_name_to_latex(op_name)
                return InterpretResult(
                    ok=True,
                    operation=operation,
                    args={
                        "name": op_name,
                        "_op_expr_display": op_disp,
                        "_op_kind": "fermion" if is_fermion else "boson",
                    },
                    confidence=1.0,
                    source="rule",
                    requires_confirmation=False,
                )
            return super().extra_pre_phase1(text, lang)

        # Expression IDs (E1, E7 …) are never valid operator names.
        if _EXPR_ID_RE.match(op_expr):
            return super().extra_pre_phase1(text, lang)

        matched: dict | None = None
        for p in self._load_particles():
            for names in p["names"].values():
                if any(name in particle_text for name in names):
                    matched = p
                    break
            if matched:
                break

        if matched is None:
            return super().extra_pre_phase1(text, lang)

        base_m = _OP_BASE_NAME_RE.match(op_expr)
        if not base_m:
            return super().extra_pre_phase1(text, lang)
        base_name = base_m.group(1)

        operation = (
            "define_boson_operator"
            if matched["statistics"] == "boson"
            else "define_fermion_operator"
        )
        lang_key = "ja" if lang == "ja" else "en"
        particle_display = (matched["names"].get(lang_key) or matched["names"].get("en") or [""])[0]

        return InterpretResult(
            ok=True,
            operation=operation,
            args={
                "name": base_name,
                "_op_expr_display": op_expr,
                "_particle_name": particle_display,
                # The researcher already WROTE which particle this is; the word
                # was used for the confirmation message and then discarded, so a
                # session knew `statistics=fermi` and nothing about the spin.
                # The key travels so the engine can record the projection set
                # (`spin_value_set`), which is what lets `\sum_\sigma` expand.
                "_particle_key": matched.get("key", ""),
                "_op_type_key": op_type_key or "creation",
            },
            confidence=1.0,
            source="rule",
            requires_confirmation=False,
        )

    def should_bypass_auto_register_formula(self, formula_str: str, op_text: str) -> bool:
        return bool(
            # BOTH languages, deliberately — unlike the READING above, which
            # runs when the session language is known.  This is a GUARD: it asks
            # "could this sentence be a declaration?", and the two mistakes are
            # not symmetric.  A false positive merely skips auto-registration; a
            # false NEGATIVE registers a declaration as a formula and reports
            # success, which is the NL-GRAMMAR ② defect (`Define k, m as
            # scalars`).  Narrowing it by language broke exactly that: a handler
            # with no engine bound answers `DEFAULT_LANG` ("en"), so
            # `の消滅演算子をbとする` stopped being recognised (caught by
            # test_llm_supervisor, NOT by the corpus — scenarios run each
            # language in its own session, so the crossing never occurs there).
            any(nlp("particle_def_op_text", _l).search(op_text)
                for _l in ("ja", "en"))
            or nlp_any("spin_def_op_text", op_text)
            or super().should_bypass_auto_register_formula(formula_str, op_text)
        )

    def format_void_op_message(self, operation: str, args: dict, lang: str) -> "str | None":
        """Return a Julia-specific confirmation message, or delegate to shared base.

        Symbol names are no longer embedded in the message text — they are
        rendered by the JS layer as KaTeX (latex field) with the plain name in
        parentheses (symbol_display field).
        """
        if operation == "define_spin_operator":
            return (self.spin_declaration_message(args, lang)
                    or _t("interpreter.spin_op_defined", lang))

        if args.get("_particle_name"):
            particle = args["_particle_name"]
            op_type_key = args.get("_op_type_key", "creation")
            op_type_str = _t(f"interpreter.op_type_{op_type_key}", lang)
            return _t("interpreter.particle_op_defined", lang,
                      particle=particle, op_type=op_type_str)

        if args.get("_op_kind") == "fermion":
            return _t("interpreter.fermion_op_defined", lang)
        if args.get("_op_kind") == "boson":
            return _t("interpreter.boson_op_defined", lang)

        # Shared type-declaration operations (scalar, vector, matrix, tensor)
        return super().format_void_op_message(operation, args, lang)

    _OPERATOR_DEF_OPS: frozenset[str] = frozenset({
        "define_fermion_operator", "define_boson_operator", "define_spin_operator",
    })

    def get_void_op_symbol_display(self, operation: str, args: dict) -> str:
        """Return the plain-text operator expression for the parenthetical display."""
        if operation in self._OPERATOR_DEF_OPS:
            return args.get("_op_expr_display") or args.get("name", "")
        return super().get_void_op_symbol_display(operation, args)

    def get_void_op_display_latex(self, operation: str, args: dict) -> str:
        """Return a LaTeX string to display next to the confirmation message.

        Type-declaration ops (scalar/vector/matrix/tensor) are handled by the
        shared base class.  This override additionally handles Julia-specific
        operator definitions, rendering the operator expression.
        """
        if operation in self._OPERATOR_DEF_OPS:
            display = args.get("_op_expr_display") or args.get("name", "")
            if display:
                return self.normalize_display_latex(display)

        # Shared type-declaration operations and default fallback
        return super().get_void_op_display_latex(operation, args)

    # ------------------------------------------------------------------
    # H-9: solve_equation / solve_system (Julia / Symbolics.jl)
    # ------------------------------------------------------------------

    @staticmethod
    def _eq_str_for_solve(eid_or_str: str, expr_lookup: dict) -> "tuple[str, str] | None":
        """Return (lhs, rhs) strings for a Symbolics.jl equation.

        For H-1 expressions, expr_lookup already has the full "LHS = RHS" form
        (reconstructed by the engine's H-1 restoration step).
        Returns None if the string cannot be split into an equation.
        """
        import re as _re
        s = (expr_lookup.get(eid_or_str, eid_or_str) if is_expr_ref(eid_or_str)
             else eid_or_str).strip()
        # Both halves are interpolated into a Julia STRING LITERAL
        # (`_nb_solve_equation("lhs", "rhs", var)`), so a mis-split does not just
        # produce wrong code — it produces unlexable code, and the researcher is
        # shown a complaint about an escape sequence.  Measured 2026-08-13 on
        # `\sum_{i=1}^{3} p_i = u*x + v*y`:
        #     _nb_solve_equation("\sum_{i", "1}^{3} p_i = u*x + v*y", "u")
        # → ParseError … invalid escape, because `\s` is not one.  Nothing in
        # that message points at the real cause.
        _rel = split_relation(s)
        if _rel is None or "=" not in _rel[1]:
            return None
        return _rel[0], _rel[2]

    def _to_julia_code(self, text: str) -> "str | None":
        """The researcher's writing → Julia code, for a builder that dispatches
        before `build_command`'s per-argument loop.

        NOT `to_backend_code`: that door is Maxima's.  On this side `parse_ir`
        returns None, so it answers `passthrough` and hands the raw text
        straight on — which is how a bounded sum reached a Julia STRING LITERAL
        as `\\sum_{i=1}^{3} p_i`, where `\\s` is an invalid escape, and the
        researcher got a character-level ParseError with no sign of the cause
        (open_items BUILDER-OPERAND-NOT-LOWERED).

        Returns None when the text cannot be represented, so the caller refuses
        with a reason rather than sending something unreadable.
        """
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator
        if self._context is None:
            # No SymbolContext bound — a bare handler, which happens only in unit
            # tests; the engine wires one at the same two points it binds the
            # language.  Resolution needs it (`context.get(name)`), so lowering
            # cannot even be attempted: pass the text through exactly as before
            # rather than refusing, which would fail an operation for a reason
            # that has nothing to do with what the researcher wrote.
            return text
        try:
            surf = SharedParser().parse(text)
            if surf is None:
                return None
            ir = JuliaSemanticResolver().resolve(surf, self._context)
            return JuliaCodeGenerator(context=self._context).generate(ir)
        except Exception:
            return None

    def _build_solve(self, operation: str, args: dict,
                     expr_lookup: dict) -> "BuildResult":
        """Build Symbolics.jl solve command.

        Single equation:
          let _sol = Symbolics.solve_for(lhs ~ rhs, var)
              "var = " * string(_sol)
          end

        System:
          let _sols = Symbolics.solve_for([lhs1 ~ rhs1, ...], [var1, ...])
              join(["var1", ...] .* " = " .* string.(_sols), ", ")
          end

        Result format (returned as string from Julia):
          single: "m = F/a"
          system: "c_k = u_k*alpha_k - v_k*alpha_mk, c_dag = ..."
        """
        from notebook.runtime.command_builder import BuildResult

        if operation == "solve_equation":
            eq_eid = str(args.get("equation", ""))
            var    = str(args.get("var", ""))
            if not eq_eid or not var:
                return BuildResult(ok=False, command="",
                                   error="solve_equation: missing equation or var")
            parts = self._eq_str_for_solve(eq_eid, expr_lookup)
            if parts is None:
                return BuildResult(ok=False, command="",
                                   error=f"solve_equation: cannot extract equation form from {eq_eid!r}")
            lhs, rhs = parts
            # Both halves are MATHEMATICS and are lowered; `var` is a NAME and is
            # not.  Before this the researcher's own LaTeX was interpolated into a
            # Julia string literal, where `\s` is an invalid escape — so the call
            # failed to LEX and the researcher saw a character-level ParseError
            # with no sign of the real cause (open_items BUILDER-OPERAND-NOT-LOWERED).
            _l, _r = self._to_julia_code(lhs), self._to_julia_code(rhs)
            if _l is None or _r is None:
                return BuildResult(ok=False, command="",
                                   error="solve_equation: cannot lower the equation")
            lhs, rhs = _l, _r
            # Uses _nb_solve_equation: tries Symbolics.solve_for (linear),
            # falls back to SymPy.solve for nonlinear (quadratic etc.)
            lhs_esc = lhs.replace('"', '\\"')
            rhs_esc = rhs.replace('"', '\\"')
            cmd = f'_nb_solve_equation("{lhs_esc}", "{rhs_esc}", "{var}")'
            return BuildResult(ok=True, command=cmd)

        # solve_system
        raw_eqs  = str(args.get("equations", ""))
        vars_str = str(args.get("vars", ""))
        if not raw_eqs or not vars_str:
            return BuildResult(ok=False, command="",
                               error="solve_system: missing equations or vars")

        eq_parts = [e.strip() for e in raw_eqs.split(",")]
        var_parts = [v.strip() for v in vars_str.split(",")]

        tilde_eqs: list[str] = []
        for ep in eq_parts:
            parts = self._eq_str_for_solve(ep, expr_lookup)
            if parts is None:
                return BuildResult(ok=False, command="",
                                   error=f"solve_system: cannot extract equation from {ep!r}")
            lhs, rhs = parts
            tilde_eqs.append(f"{lhs} ~ {rhs}")

        eqs_julia  = "[" + ", ".join(tilde_eqs) + "]"
        vars_julia = "[" + ", ".join(var_parts) + "]"
        var_strs   = "[" + ", ".join(f'"{v}"' for v in var_parts) + "]"

        cmd = (
            f'let _sols = Symbolics.solve_for({eqs_julia}, {vars_julia})\n'
            f'    join({var_strs} .* " = " .* string.(_sols), ", ")\n'
            f'end'
        )
        return BuildResult(ok=True, command=cmd)

    supports_operator_adjoint = True

    def adjoint_canonical(self, canonical: str) -> "str | None":
        """Return the canonical of the Hermitian adjoint of a linear operator
        expression, e.g. ``u*c(1) - v*c'(1)`` → ``u*c'(1) - v*c(1)``.

        Used by the L-1 canonical-constraint derivation to build new_i† from the
        forward definition new_i.  Operator adjoint is backend-specific, so the
        base class returns None and only the Julia handler implements it.
        """
        from notebook.runtime.shared_parser import SharedParser
        from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
        from notebook.runtime.julia_code_generator import JuliaCodeGenerator
        from notebook.models.ir import Equation as _EqIR
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.julia_handler_common import (
            _adjoint_ir, quantum_operator_names,
        )

        ctx = self._context or SymbolContext()
        surface = SharedParser().parse(canonical)
        if surface is None:
            return None
        ir = JuliaSemanticResolver().resolve(surface, ctx)
        if isinstance(ir, _EqIR):
            ir = ir.rhs
        op_names = quantum_operator_names(ctx, ir)
        jcg = JuliaCodeGenerator(context=ctx)
        jcg.static_index_mode = True
        return jcg.generate(_adjoint_ir(ir, op_names))

    @staticmethod
    def _operator_surrogates(*texts: str) -> "dict[str, str]":
        """Map operator-form atoms (c(1), c'(2)) to plain solve identifiers, because
        Symbolics @variables / solve_for cannot use a function-call form as a
        variable.  c(1)→nbA_c_1, c'(2)→nbA_cD_2.

        A 2+ argument operator like c(k,u) can appear differently spaced across
        the different source texts scanned here (e.g. the raw equation text
        "c(k,u)" vs. an already-normalized vars list "c(k, u)") — the argument
        interior is whitespace-normalized before building the surrogate name so
        both spellings resolve to the SAME surrogate (otherwise solve_for's
        equation list and variable list silently reference different symbols)."""
        import re as _re
        out: dict[str, str] = {}
        for text in texts:
            for m in _re.finditer(r"[A-Za-z_]\w*'?\([^)]*\)", text or ""):
                tok = m.group(0)
                if tok in out:
                    continue
                mm = _re.match(r"([A-Za-z_]\w*)('?)\(([^)]*)\)", tok)
                if not mm:
                    continue
                _norm_args = _re.sub(r'\s*,\s*', ',', mm.group(3).strip())
                out[tok] = (f"nbA_{mm.group(1)}{'D' if mm.group(2) else ''}_"
                            f"{_re.sub(r'[^A-Za-z0-9]', '_', _norm_args)}")
        return out

    def _build_inverse_transform(self, args: dict,
                                 expr_lookup: dict) -> "BuildResult":
        """L-1: invert a linear basis change new = M·old by solving the forward
        system for the OLD basis (old = M^{-1} new).

        Unlike Maxima (whose H-1 P2 binds the new-basis symbols to their RHS, so
        they must be unbound in a block()), Julia's H-1 is structure-preserving and
        the new-basis symbols stay free.  We still declare both the new basis and the
        old basis as Symbolics @variables so solve_for returns the old basis in terms
        of the new one rather than failing on undefined identifiers.
        """
        from notebook.runtime.command_builder import BuildResult

        raw_eqs   = str(args.get("equations", ""))
        vars_str  = str(args.get("vars", ""))               # old basis (solve-for)
        new_basis = str(args.get("new_basis", "")).strip()  # forward LHS symbols
        if not raw_eqs or not vars_str:
            return BuildResult(ok=False, command="",
                               error="build_inverse_transform: missing equations or vars")

        # Top-level-comma split: a 2-argument operator like c(k,u) or g(-k,d)
        # in this comma-joined list has its OWN internal comma, so a naive
        # .split(",") shreds it into "c(k" / "u)" fragments (only surfaced once
        # a multi-index operator was used — every prior L-1 test used single-
        # arg operators like c(1)/c(2), which have no internal comma to break).
        eq_parts  = _split_top_level_commas(raw_eqs)
        var_parts = _split_top_level_commas(vars_str)

        tilde_eqs: list[str] = []
        for ep in eq_parts:
            parts = self._eq_str_for_solve(ep, expr_lookup)
            if parts is None:
                return BuildResult(ok=False, command="",
                                   error=f"build_inverse_transform: cannot extract equation from {ep!r}")
            lhs, rhs = parts
            tilde_eqs.append(f"{lhs} ~ {rhs}")

        # Operator-form atoms (c(1), c'(2)) cannot be Symbolics solve variables;
        # swap them for plain surrogates, solve, and map the surrogates back in the
        # result string so the registered inverse keeps operator notation.
        import re as _re
        _new_basis_parts = _split_top_level_commas(new_basis) if new_basis else []
        _surr = self._operator_surrogates(
            " ".join(tilde_eqs), " ".join(var_parts), " ".join(_new_basis_parts))
        _orig_labels = list(var_parts)
        if _surr:
            def _swap(s: str) -> str:
                for _tok, _sf in _surr.items():
                    s = _re.sub(_re.escape(_tok), _sf, s)
                return s
            tilde_eqs = [_swap(e) for e in tilde_eqs]
            var_parts = [_swap(v) for v in var_parts]
            _new_basis_parts = [_swap(s) for s in _new_basis_parts]

        eqs_julia  = "[" + ", ".join(tilde_eqs) + "]"
        vars_julia = "[" + ", ".join(var_parts) + "]"
        var_strs   = "[" + ", ".join(f'"{v}"' for v in _orig_labels) + "]"

        # new_basis symbols also need the operator-call surrogate: they are
        # operator-form atoms too (e.g. g(k,u)), and @variables cannot declare
        # a function-call-syntax name any more than solve_for can solve for one.
        decl_syms: list[str] = []
        for s in _new_basis_parts + var_parts:
            if s and s not in decl_syms:
                decl_syms.append(s)
        decl = f"@variables {' '.join(decl_syms)}\n    " if decl_syms else ""

        _out = f'join({var_strs} .* " = " .* string.(_sols), ", ")'
        if _surr:
            _pairs = ", ".join(f'"{_sf}" => "{_tok}"' for _tok, _sf in _surr.items())
            _out = f"replace({_out}, {_pairs})"
        cmd = (
            f'let\n    {decl}_sols = Symbolics.simplify.(Symbolics.solve_for({eqs_julia}, {vars_julia}))\n'
            f'    {_out}\n'
            f'end'
        )
        return BuildResult(ok=True, command=cmd)

    def _build_metric_constraints(self, args: dict,
                                  expr_lookup: dict) -> "BuildResult":
        """Non-operator metric-preservation condition M G Mᵀ = G via Symbolics.
        Coeff-extracts M from the forward transform, builds G, and emits
        simplify.(M G Mᵀ − G) formatted as a Maxima-style matrix(...) string so the
        pipeline's matrix parser is shared with the Maxima path.  (Note: Symbolics
        does not fold the Pythagorean identity, so a rotation may leave a cos²+sin²−1
        entry — algebraic metrics like an overlap matrix S reduce cleanly.)"""
        from notebook.runtime.command_builder import BuildResult
        import re as _re

        forward = str(args.get("forward", ""))
        basis   = str(args.get("basis", ""))
        gram    = str(args.get("gram", ""))
        if not forward or not basis or not gram:
            return BuildResult(ok=False, command="",
                               error="derive_metric_constraints: missing forward/basis/gram")
        bsyms = [b.strip() for b in basis.split(",") if b.strip()]
        rhss: list[str] = []
        for f in [x.strip() for x in forward.split(",") if x.strip()]:
            rhs = expr_lookup.get(f, f)
            # Feeds `Symbolics.coeff(rhs, b)` AND the `@variables` declaration
            # scanned off it, so a mis-split warps the code and its declaration
            # separately.  Measured: `Symbolics.coeff(1}^{3} p_i = u*x + v*y, x)`.
            _rel = split_relation(rhs)
            if _rel is not None and "=" in _rel[1]:
                rhs = _rel[2]
            rhss.append(rhs)

        _kw = {"Symbolics", "coeff", "simplify", "size", "string", "join",
               "let", "end", "permutedims", "for", "in", "im", "pi"}
        syms: set[str] = set(bsyms)
        for blob in rhss + [gram]:
            syms.update(m.group(0) for m in _re.finditer(r"[A-Za-z_]\w*", blob)
                        if m.group(0) not in _kw)
        decl = f"@variables {' '.join(sorted(syms))}" if syms else ""

        m_rows = "; ".join(
            " ".join(f"Symbolics.coeff({rhs}, {b})" for b in bsyms) for rhs in rhss)
        g_rows = "; ".join(" ".join(c.strip() for c in r.split(","))
                           for r in gram.split(";"))
        cmd = (
            f"let\n    {decl}\n"
            f"    _M = [{m_rows}]\n"
            f"    _G = [{g_rows}]\n"
            f"    _C = Symbolics.simplify.(_M*_G*permutedims(_M) - _G)\n"
            f"    _n = size(_C, 1)\n"
            f'    "matrix(" * join(["[" * join([string(_C[i,j]) for j in 1:_n], ", ") '
            f'* "]" for i in 1:_n], ", ") * ")"\n'
            f"end"
        )
        return BuildResult(ok=True, command=cmd)

    def _build_apply_constraints(self, args: dict,
                                 expr_lookup: dict) -> "BuildResult":
        """Reduce an expression under polynomial side relations (e.g. u²+v²=1) via
        Symbolics.substitute, so the constraint folds the result (the Bogoliubov
        inverse denominator u²+v² collapses to 1)."""
        from notebook.runtime.command_builder import BuildResult

        target = str(args.get("expr", ""))
        rels   = str(args.get("relations", ""))
        if not target or not rels:
            return BuildResult(ok=False, command="",
                               error="apply_canonical_constraints: missing expr or relations")

        tgt = expr_lookup.get(target, target)
        # If the target is an equation "lhs = rhs", reduce the RHS only.
        # `tgt_lhs` is written back into the RESULT text
        # (`"{tgt_lhs} = " * string(_r)`), so this split is both CAS code and
        # what the researcher reads — the same two roles as the 代入 label.
        tgt_lhs = None
        _rel_t = split_relation(tgt)
        if _rel_t is not None and "=" in _rel_t[1]:
            tgt_lhs, tgt = _rel_t[0], _rel_t[2]

        import re as _re
        _KW = {"Dict", "Symbolics", "simplify", "substitute", "string", "let",
               "end", "im", "pi"}

        def _ids(s: str) -> set[str]:
            return {m.group(0) for m in _re.finditer(r"[A-Za-z_]\w*", s)
                    if m.group(0) not in _KW}

        subs: list[str] = []
        syms: set[str] = _ids(tgt)
        for r in [x.strip() for x in rels.split(",") if x.strip()]:
            rel = expr_lookup.get(r, r)
            _rel_r = split_relation(rel)
            if _rel_r is None or "=" not in _rel_r[1]:
                continue
            lhs, rhs = _rel_r[0], _rel_r[2]
            subs.append(f"{lhs} => {rhs}")
            syms.update(_ids(lhs))
            syms.update(_ids(rhs))
        if not subs:
            return BuildResult(ok=False, command="",
                               error="apply_canonical_constraints: no usable relations")

        decl = f"@variables {' '.join(sorted(syms))}\n    " if syms else ""
        reduced = (f"Symbolics.simplify(Symbolics.substitute({tgt}, "
                   f"Dict({', '.join(subs)})))")
        if tgt_lhs:
            cmd = (f'let\n    {decl}_r = {reduced}\n'
                   f'    "{tgt_lhs} = " * string(_r)\n'
                   f'end')
        else:
            cmd = f'let\n    {decl}string({reduced})\n    end'
        return BuildResult(ok=True, command=cmd)

    @staticmethod
    def parse_solve_result(result_str: str) -> "list[tuple[str, str]]":
        """Parse Julia solve result 'x = val1, y = val2' into [(x, val1), (y, val2)]."""
        import re as _re
        s = result_str.strip().strip('"')
        if not s:
            return []
        pairs: list[tuple[str, str]] = []
        # Split on ', ' only at top level (not inside parens/brackets)
        depth = 0
        cur: list[str] = []
        for ch in s:
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            if ch == "," and depth == 0:
                pairs.append("".join(cur).strip())  # type: ignore
                cur = []
            else:
                cur.append(ch)
        if cur:
            pairs.append("".join(cur).strip())  # type: ignore

        result: list[tuple[str, str]] = []
        for part in pairs:
            part = str(part).strip()
            # CAS OUTPUT: Symbolics writes `var = val`, `=` leftmost, so the
            # naive split was never wrong here (measured).  Routed through the
            # one definition anyway — safe only by the current output format is
            # the shape this ratchet exists to stop.
            _rel = split_relation(part)
            if _rel is not None and "=" in _rel[1]:
                result.append((_rel[0], _rel[2]))
        return result

    # ------------------------------------------------------------------
    # C-Ext Phase 1-A: re_evaluate_expr
    # ------------------------------------------------------------------

    def re_evaluate_expr(
        self,
        canonical: str,
        substitutions: "dict[str, str]",
        expr_lookup: "dict[str, str]",
        adapter: "object | None" = None,
        sympy_canonical: "str | None" = None,
    ) -> "str | None":
        """Re-evaluate *canonical* by substituting named Symbolics variables.

        Strategy: For each (lhs, rhs) pair in *substitutions*, apply
        Symbolics.substitute() on the numeric coefficients of the QuExpr.
        Multiple substitutions are applied sequentially (each result feeds
        the next), which is correct for independent symbol updates and
        sufficient for mean-field self-consistency iterations.

        For QuantumAlgebra expressions the scalar coefficients are
        Symbolics.Num values — the same Strategy B used in the
        ``substitute`` operation.  Operator-valued symbols (e.g. operator
        pairs c†c that have already been replaced by a scalar via A-2)
        appear as Symbolics parameters and are handled the same way.

        For SymPy expressions (sympy_canonical is not None), substitutions
        are applied via _nb_sympy_substitute sequentially, preserving
        operator order exactly as in the substitute operation.

        Returns:
            New canonical string, or None if adapter is unavailable.
        """
        if adapter is None:
            return None
        if not getattr(adapter, "is_running", False):
            return None

        if not substitutions:
            return canonical

        # SymPy path: expression is a SymPy chain result
        if sympy_canonical:
            return self._re_evaluate_sympy(
                sympy_canonical, substitutions, adapter
            )

        # Build a Julia let-block that applies substitutions sequentially.
        # Each pair: @variables lhs_sym; expr = Symbolics.substitute(expr, Dict(lhs => rhs))
        def _is_numeric(s: str) -> bool:
            try:
                float(s)
                return True
            except ValueError:
                return False

        # Resolve the source expression.
        # _nb_{expr_id} only exists for expressions produced by run_operation
        # (stored as Julia session variables).  For expressions registered via
        # register_expression only, we must re-evaluate from canonical.
        #
        # Strategy: probe whether _nb_{expr_id} exists in the Julia session.
        # If yes, use the session variable (avoids re-parsing issues with
        # operator expressions like c'(k)).
        # If no, fall back to evaluating the canonical string directly
        # (works only for pure scalar/Symbolics expressions).

        # Extract the expr_id from the canonical if it looks like a session var reference.
        # e.g. canonical == "Delta * c'(k) * c'(-k)" — we need to find the _nb_ var.
        # We scan expr_lookup to find the matching session variable.
        nb_src_var: str | None = None
        for eid, can in expr_lookup.items():
            if can == canonical:
                probe = adapter.execute(f"isdefined(Main, :_nb_{eid})")
                if probe.ok and probe.value.strip() == "true":
                    nb_src_var = f"_nb_{eid}"
                    break

        if nb_src_var is None:
            # No session variable — only safe for pure Symbolics expressions.
            # Declare lhs symbols and use Symbolics.substitute directly.
            syms_to_declare = [
                lhs.strip() for lhs in substitutions
                if not _is_numeric(lhs.strip()) and re.match(r'^[A-Za-z_]\w*$', lhs.strip())
            ]
            rhs_syms = [
                rhs.strip() for rhs in substitutions.values()
                if not _is_numeric(rhs.strip()) and re.match(r'^[A-Za-z_]\w*$', rhs.strip())
            ]
            all_syms = list(dict.fromkeys(syms_to_declare + rhs_syms))
            pairs = ", ".join(
                f"{lhs.strip()} => {rhs.strip()}" for lhs, rhs in substitutions.items()
            )
            stmts = []
            if all_syms:
                stmts.append(f"@variables {' '.join(all_syms)}")
            stmts.append(f"_nb_re_expr = {canonical}")
            stmts.append(f"_nb_re_sub = Dict({pairs})")
            stmts.append(
                "_nb_re_result = _nb_re_expr isa _QA.QuExpr ? "
                "normal_form(_QA.QuExpr(Dict(op => (c isa Symbolics.Num ? Symbolics.substitute(c, _nb_re_sub) : c) "
                "for (op, c) in _nb_re_expr.terms))) : "
                "Symbolics.substitute(_nb_re_expr, _nb_re_sub)"
            )
            stmts.append("_nb_re_result")
            cmd = "; ".join(stmts)
        else:
            # Use existing Julia session variable — safe for both QuExpr and Num.
            pairs = ", ".join(
                f"{lhs.strip()} => {rhs.strip()}" for lhs, rhs in substitutions.items()
            )
            stmts = [
                f"_nb_re_sub = Dict({pairs})",
                f"_nb_re_result = {nb_src_var} isa _QA.QuExpr ? "
                f"normal_form(_QA.QuExpr(Dict(op => (c isa Symbolics.Num ? Symbolics.substitute(c, _nb_re_sub) : c) "
                f"for (op, c) in {nb_src_var}.terms))) : "
                f"Symbolics.substitute({nb_src_var}, _nb_re_sub)",
                "_nb_re_result",
            ]
            cmd = "; ".join(stmts)

        result = adapter.execute(cmd)
        if not result.ok:
            return None
        return result.value.strip() or None

    def _re_evaluate_sympy(
        self,
        sympy_canonical: str,
        substitutions: "dict[str, str]",
        adapter: "object",
    ) -> "str | None":
        """Apply multiple substitutions sequentially to a SymPy canonical string.

        Each (lhs, rhs) pair is applied via _nb_sympy_substitute in order.
        The updated sympy_canonical threads through each step.

        Used by re_evaluate_expr when the source expression has sympy_canonical
        (i.e. it was produced by the SymPy chain: normal_order, wick_contract,
        substitute, etc.).

        Returns the new canonical string (display form), or None on error.
        The caller in engine.py is responsible for propagating sympy_canonical.
        """
        import re as _re
        ctx = self._context or SymbolContext()
        stats_json = self._collect_stats_json(ctx)

        current = sympy_canonical
        for lhs, rhs in substitutions.items():
            lhs = lhs.strip()
            rhs = rhs.strip()
            if not lhs or not rhs:
                continue

            # Build vars_str from all tokens in the current expression + lhs/rhs
            combined = current + " " + lhs + " " + rhs
            vars_found = sorted(set(_re.findall(r"\b([a-z])\b", combined)))
            vars_found = [v for v in vars_found if v not in ("c", "a", "e", "i", "j")]
            vars_str = ",".join(vars_found) if vars_found else "k"

            stats_esc = stats_json.replace("\\", "\\\\").replace('"', '\\"')
            sym_esc   = current.replace("\\", "\\\\").replace('"', '\\"')
            lhs_esc   = lhs.replace("\\", "\\\\").replace('"', '\\"')
            rhs_esc   = rhs.replace("\\", "\\\\").replace('"', '\\"')

            cmd = (f'_nb_sympy_substitute("{sym_esc}", "{lhs_esc}",'
                   f' "{rhs_esc}", "{stats_esc}", "{vars_str}")')
            result = adapter.execute(cmd)
            if not result.ok:
                return None
            # _nb_sympy_substitute returns (value_str, latex_str) as a Tuple
            # adapter.execute captures the display form of the Tuple
            val = result.value.strip()
            if not val:
                return None
            # Extract the first element (value_str) from Julia Tuple output
            # Format: ("value_str", "latex_str")  or just the value if already extracted
            m = _re.match(r'^\("([^"]*)"', val)
            if m:
                current = m.group(1)
            else:
                current = val

        return current





