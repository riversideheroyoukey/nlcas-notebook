r"""H-4: "substitute at THIS index only".

`operator_substitute` reads a SYMBOLIC formal index as a template covering every
index.  That is right for a Bogoliubov transform — `c(k) → u a(k) + v b(-k)`
means "for all k" — and wrong for a mean-field decoupling on one site.  Measured
2026-08-17 in a live Julia session:

    H = J Sz_i Sz_j , substitute Sz(i) → Mz   →  H = Mz^2 J      (BOTH replaced)
    H = J Sz_1 Sz_2 , substitute Sz(1) → Mz   →  H = J Mz Sz_2   (concrete: fine)

so the defect is specific to a symbolic index, and it is silent: `Mz^2` is the
fully decoupled result, a different physics returned as a success.

**The filter already existed.**  The ticket proposed a future `formal_index_only`
option; `exact_index` was already implemented end to end — schema field, builder,
substituter — and with it the ticket's expected answer comes out byte for byte.
The only thing missing was a way for a RESEARCHER to ask for it, since a
symbolic index cannot be auto-detected: the text alone does not say whether `i`
names a site or stands for any site.  So the researcher says which, and these
are the spellings.

The default is deliberately unchanged.  Both readings are legitimate and the
notation does not choose between them.
"""
import pytest

from notebook.models.symbol_context import SymbolContext
from notebook.runtime.julia_backend_handler import JuliaBackendHandler


def _handler():
    ctx = SymbolContext()
    ctx.register_quantity("Sz", kind="operator")
    h = JuliaBackendHandler(None)
    h._context = ctx
    return h


_EXACT = [
    ("ja", "E1 の Sz(i) だけに Mz を代入して"),
    ("ja", "E1 の Sz(i) のみに Mz を代入して"),
    ("en", "In E1, substitute Mz for Sz(i) only"),
    ("en", "In E1, substitute Mz for only Sz(i)"),
]
_PLAIN = [
    ("ja", "E1 の Sz(i) に Mz を代入して"),
    ("en", "In E1, substitute Mz for Sz(i)"),
]


class TestTheModifierIsReadInBothLanguages:
    @pytest.mark.parametrize("lang,text", _EXACT)
    def test_only_sets_exact_index(self, lang, text):
        args = _handler().extra_pre_phase1(text, lang).args
        assert args["exact_index"] is True
        assert args["formal_index"] == "i"
        assert args["op_name"] == "Sz"

    @pytest.mark.parametrize("lang,text", _PLAIN)
    def test_without_it_the_default_is_unchanged(self, lang, text):
        """The template reading is what a Bogoliubov transform needs; changing
        the default would break the operation's main use."""
        args = _handler().extra_pre_phase1(text, lang).args
        assert "exact_index" not in args
        assert args["formal_index"] == "i"

    @pytest.mark.parametrize("lang,text", _EXACT + _PLAIN)
    def test_the_modifier_never_leaks_into_the_rhs(self, lang, text):
        """It is stripped before the substitution patterns run, so no part of it
        can end up inside the expression being substituted in."""
        args = _handler().extra_pre_phase1(text, lang).args
        assert args["rhs"] == "Mz"


class TestWhatTheModifierIsAnchoredTo:
    r"""`Sz(i) だけ` names ONE INSTANCE; a bare `Sz だけ` would be about WHICH
    OPERATOR and says nothing about the index.  Anchoring the modifier to the
    operator CALL — index and all — is what keeps those two apart, so a
    sentence that says "only" about something else must not set the filter."""

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 の Sz(i) に Mz だけを代入して"),
        ("en", "In E1, substitute only Mz for Sz(i)"),
    ])
    def test_only_about_the_rhs_does_not_set_the_filter(self, lang, text):
        res = _handler().extra_pre_phase1(text, lang)
        if res is not None and getattr(res, "args", None):
            assert "exact_index" not in res.args


class TestLiveJuliaEndToEnd:
    """The ticket's own example, driven through the NL route."""

    @staticmethod
    def _run(setup, instruction, lang):
        from notebook.engine import NotebookEngine
        from tests.test_debug_repl import _sup, _handle_interpret, _run as _r
        with NotebookEngine(title="h4", primary_backend="julia_full") as e:
            e.start()
            e.session.language = [lang, "en"]
            for t in setup:
                _r(_handle_interpret(t, e, _sup(), no_llm=True))
            _r(_handle_interpret(instruction, e, _sup(), no_llm=True))
            return list(e.expressions.values())[-1].latex

    _JA = ["S をスピン演算子とする", "J をスカラーとする", r"H = J Sz_i Sz_j"]
    _EN = ["define S as a spin operator", "let J be a scalar", r"H = J Sz_i Sz_j"]

    @pytest.mark.parametrize("lang,setup,instr", [
        ("ja", _JA, "E1 の Sz(i) だけに Mz を代入して"),
        ("en", _EN, "In E1, substitute Mz for Sz(i) only"),
    ])
    def test_only_the_named_index_is_replaced(self, lang, setup, instr):
        assert self._run(setup, instr, lang) == r"H = J Mz Sz_j"

    @pytest.mark.parametrize("lang,setup,instr", [
        ("ja", _JA, "E1 の Sz(i) に Mz を代入して"),
        ("en", _EN, "In E1, substitute Mz for Sz(i)"),
    ])
    def test_the_template_reading_still_replaces_both(self, lang, setup, instr):
        assert self._run(setup, instr, lang) == r"H = Mz^{2} J"
