r"""Spin labels in the EXPRESSION path must collapse to QA's 1-char indices.

The NL route (operator_substitute patterns) has canonicalized spin labels for
a while (`JuliaBackendHandler._canon_spin`: sigma→s, up→u, …), but the
expression path did not, so `a_{i,\sigma,k}` reached QuantumAlgebra as
`QuIndex("sigma")` and the researcher saw the raw CAS error

    ArgumentError: index names must be single character, got sigma

(Core Principle #1 violation; the "同種の漏れ" of open_items
JULIA-INDEXED-SCALAR, docs/handoffs/julia_backend_handoff.md §3).

Fix: one shared table (`notebook/runtime/spin_label_canon.py`) used by BOTH
routes, applied by `JuliaCodeGenerator._qa_quindex` at every QuIndex emission
site.  A multi-character index that is NOT a known spin label is refused with
a reason (i18n key below) instead of letting QA answer raw.
"""
import pytest

from notebook.models.quantity import Quantity
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.i18n import t
from notebook.runtime.julia_code_generator import JuliaCodeGenerator
from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
from notebook.runtime.shared_parser import SharedParser, UnknownNotationError
from notebook.runtime.spin_label_canon import canon_spin_label

_KEY = "parser.unsupported_notation.julia_multichar_index"


def _generate(src: str, ctx: SymbolContext | None = None) -> str:
    ctx = ctx or SymbolContext()
    ir = JuliaSemanticResolver().resolve(SharedParser().parse(src), ctx)
    return JuliaCodeGenerator(context=ctx).generate(ir)


class TestSpinLabelCollapses:
    def test_sigma_index_on_operator(self):
        # c is a quantum-index base by default; \sigma normalizes to "sigma".
        code = _generate(r"c_{k,\sigma}")
        assert '_QA.QuIndex("s")' in code
        assert "sigma" not in code

    def test_three_indices_with_sigma(self):
        # The exact spelling from the handoff memo: a_{i,\sigma,k}.
        code = _generate(r"a_{i,\sigma,k}")
        assert '_QA.QuIndex("s")' in code
        assert "sigma" not in code

    def test_negated_sigma_is_uppercase_S(self):
        # Spin flip -\sigma follows the negation convention (-k → K).
        code = _generate(r"c_{-\sigma}")
        assert '_QA.QuIndex("S")' in code
        assert "sigma" not in code

    def test_up_down_labels(self):
        code = _generate(r"c_{k,up}")
        # The canonical spin index is the ARROW (2026-08-22): an arrow cannot be
        # bound by a sum, so two arrows are two VALUES and `δ(↑,↓)` evaluates to
        # 0 — which is what made the Hubbard commutators come out clean.  A
        # letter carries no such warrant.  See spin_label_canon for the full
        # reasoning.  QA's rule is ONE unicode character, which an arrow is.
        assert '_QA.QuIndex("↑")' in code

    def test_single_char_indices_untouched(self):
        code = _generate(r"c_{k,s}")
        assert '_QA.QuIndex("k")' in code
        assert '_QA.QuIndex("s")' in code

    def test_bound_sigma_is_collapsed_too(self):
        r"""A BOUND `sigma` collapses as well — the exemption's premise was false.

        This test used to assert the opposite, on the stated ground that
        "canonicalization may only fire on names that actually reach QA as
        QuIndex literals, and a bound one does not".  That is true of the BODY
        and **false of the BINDER**: measured 2026-08-22 against a live session,

            QA.∑(:sigma, c'(1,:sigma)*c(2,:sigma))
            → ArgumentError("index names must be single character, got sigma.")

        and that raw CAS text was what the researcher saw for `\sum_\sigma` —
        the most ordinary spin sum there is (Core Principle #1).  The reason a
        fixed check gives is part of what it fixes; this one described a
        mechanism that does not hold, so the check outlived its premise
        (docs/developer_manuals/35 §2 ④).

        The binder and the body are renamed TOGETHER, because
        `_nb_eval_index_sum` pairs them by NAME.
        """
        from notebook.models.ir import IndexSum, IndexSymbol, IndexedSymbol, Symbol
        node = IndexSum(
            body=IndexedSymbol(base=Symbol(name="c"),
                               indices=[IndexSymbol(name="sigma")]),
            dummy_indices=[IndexSymbol(name="sigma")],
        )
        gen = JuliaCodeGenerator(context=SymbolContext())
        code = gen.generate(node)
        assert "_QA.∑(:s," in code              # bound by the sum, QA-legal
        assert "sigma" not in code              # binder AND body renamed
        assert 'QuIndex("s")' not in code       # still a bound symbol, not a literal

    def test_bound_sigma_static_mode_collapses(self):
        # In static_index_mode bound indices DO emit QuIndex — the spin
        # canonicalization must apply there.
        from notebook.models.ir import IndexSum, IndexSymbol, IndexedSymbol, Symbol
        node = IndexSum(
            body=IndexedSymbol(base=Symbol(name="c"),
                               indices=[IndexSymbol(name="sigma")]),
            dummy_indices=[IndexSymbol(name="sigma")],
        )
        gen = JuliaCodeGenerator(context=SymbolContext())
        gen.static_index_mode = True
        code = gen.generate(node)
        assert '_QA.QuIndex("s")' in code
        assert "sigma" not in code


class TestFunctionCallSpelling:
    r"""The SAME leak reaches QA through a SECOND function.

    `c_{k,\sigma}` (subscript spelling) is lowered by the IndexedSymbol branch,
    but `c(k,sigma)` — the spelling the BCS scenario actually uses
    (`c'(k,u) c(k,u)`, docs/scenarios/scenario_bcs_hamiltonian_jp.md) — goes
    through `_plain_sym_as_index` instead.  Both emit QA index literals, so
    both must collapse spin labels; testing only the subscript spelling is how
    a rule ends up half-applied (the very failure this fix exists to undo).
    """

    def _ctx(self) -> SymbolContext:
        ctx = SymbolContext()
        ctx.register("c", Quantity(name="c", kind="operator"))
        return ctx

    def test_annihilation_with_sigma(self):
        assert _generate("c(k,sigma)", self._ctx()) == "c(:k, :s)"

    def test_creation_with_sigma(self):
        assert _generate("c'(k,sigma)", self._ctx()) == "c'(:k, :s)"

    def test_single_char_labels_unchanged(self):
        # The BCS scenario's own spelling must be byte-for-byte untouched.
        ctx = self._ctx()
        assert _generate("c(k,u)", ctx) == "c(:k, :u)"
        assert _generate("c(k,s)", ctx) == "c(:k, :s)"


class TestNonSpinMultiCharRefused:
    def test_multichar_index_refused_with_reason(self):
        # "mu" is not a spin label: refuse before the CAS answers with
        # its raw ArgumentError.
        with pytest.raises(UnknownNotationError) as ei:
            _generate(r"c_{k,mu}")
        assert ei.value.i18n_key == _KEY
        assert ei.value.fmt_kwargs["index"] == "mu"

    def test_refusal_message_is_actionable_both_langs(self):
        with pytest.raises(UnknownNotationError) as ei:
            _generate(r"c_{k,mu}")
        for lang in ("ja", "en"):
            msg = t(_KEY, lang, **ei.value.fmt_kwargs)
            assert msg != _KEY          # key resolved
            assert "mu" in msg          # names the offending index
            assert "{index}" not in msg  # placeholders substituted


class TestSharedTable:
    def test_table_matches_nl_route(self):
        # The NL route's mapping is the same object — a drifted copy is how
        # the sigma leak happened in the first place.
        assert canon_spin_label("sigma") == "s"      # a VARIABLE: stays a letter
        assert canon_spin_label("up") == "↑"          # a VALUE: becomes the arrow
        assert canon_spin_label("down") == "↓"
        assert canon_spin_label("dn") == "↓"
        assert canon_spin_label("mu") is None

    def test_handler_canon_spin_delegates(self):
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        h = JuliaBackendHandler.__new__(JuliaBackendHandler)  # no adapter needed
        assert h._canon_spin("sigma") == "s"
        assert h._canon_spin("↑x") == "↑"  # fallback: first char, unchanged


class TestQaIndexRule:
    r"""QuantumAlgebra's actual index rule: ONE character, optionally followed
    by an integer subindex.

    From its README — "symbolic indices must be a single unicode character,
    with possibly an integer subindex", example ``f'(1,2,:i_9)`` → ``f†(12i₉)``
    — and confirmed against the running session 2026-08-01:

        QuIndex("k")   → k        QuIndex("k1")       → ArgumentError
        QuIndex("k_1") → k₁       QuIndex("omega")    → ArgumentError
        QuIndex("i_9") → i₉       param(:G_0,"k","ω") → G_0(kω)

    This check was first written as ``len(name) > 1``, which refuses `k_1` and
    `i_9` — spellings QA renders happily.  A guard that refuses what the CAS
    accepts is the same defect as one that passes what it rejects, just
    quieter: the researcher is told a working notation is unavailable.
    """
    from notebook.runtime.spin_label_canon import qa_index_name as _f

    @pytest.mark.parametrize("name", ["k", "i", "ω", "k_1", "i_9", "K_1"])
    def test_accepted(self, name):
        from notebook.runtime.spin_label_canon import qa_index_name
        assert qa_index_name(name) == name

    @pytest.mark.parametrize("name", ["k1", "omega", "kk", "k_a", "k_1_2"])
    def test_refused(self, name):
        from notebook.runtime.spin_label_canon import qa_index_name
        with pytest.raises(UnknownNotationError) as ei:
            qa_index_name(name)
        assert ei.value.i18n_key == _KEY

    def test_spin_labels_still_collapse_first(self):
        from notebook.runtime.spin_label_canon import qa_index_name
        assert qa_index_name("sigma") == "s"
        assert qa_index_name("up") == "↑"

    def test_subindexed_index_survives_index_emission(self):
        """Both emission doors accept it — QuIndex and the `:name` form."""
        from notebook.runtime.spin_label_canon import (qa_index_literal,
                                                       qa_index_symbol)
        assert qa_index_literal("k_1") == '_QA.QuIndex("k_1")'
        assert qa_index_symbol("k_1") == ":k_1"

    def test_negated_subindexed_index_keeps_the_convention(self):
        from notebook.runtime.spin_label_canon import qa_index_literal
        assert qa_index_literal("k_1", neg=True) == '_QA.QuIndex("K_1")'

    # NOTE: writing `c_{k_1}` as INPUT is a separate gap — the nested subscript
    # does not resolve (`generator.unsupported_ir_node`, measured 2026-08-01).
    # That is notation/IR work, recorded in open_items JULIA-INDEXED-SCALAR;
    # what is fixed here is that the index RULE no longer refuses the spelling.
