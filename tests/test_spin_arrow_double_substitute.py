r"""J-3: `\uparrow`/`\downarrow` in the SIMULTANEOUS double substitution.

The ticket (2026-07-02) described a two-stage gap: the spin-label character
class in `_OP_SUBST_SPIN_RE` was `[A-Za-z0-9]+` and did not admit an arrow, and
`_SPIN_LABEL_CANON` had no `↑`/`↓` key either.  Measured 2026-08-17, the
Japanese route ACCEPTS both arrow spellings — `_normalize_latex_subscripts`
replaces them with `u`/`d` before the pattern ever runs, so neither stage is
reached.  The ticket was stale; this file is the pin that keeps it fixed, since
nothing tested the arrow spelling through this operation.

What the same measurement DID turn up is in `TestTheEnglishRouteIsNotMangled`:
the English wording was silently corrupting its right-hand sides.
"""
import pytest

from notebook.models.symbol_context import SymbolContext
from notebook.runtime.julia_backend_handler import JuliaBackendHandler


def _handler():
    ctx = SymbolContext()
    ctx.register_quantity("c", kind="operator")
    h = JuliaBackendHandler(None)
    h._context = ctx
    return h


_JA = ("E1 の c(k, {up}) に u_k a(k) を、c(-k, {dn}) に v_k b(k) を演算子代入して")
_EN = ("In E1, substitute u_k a(k) for c(k,{up}) "
       "and v_k b(k) for c(-k,{dn})")

#: (name, written up, written down, canonical up, canonical down).
#:
#: Every spelling reaches the same OPERATION; what each canonicalises to
#: depends on whether it NAMES the projection.  The arrows and the words `up` /
#: `dn` do, so they become the arrow — a VALUE, which is what lets `δ(↑,↓)`
#: evaluate to 0 (2026-08-22, see spin_label_canon).  A bare `u` does not: it is
#: an ordinary letter that might be any index, and reading a projection into it
#: would be the guess this system declines to make everywhere else.
_SPELLINGS = [
    ("ascii", "u", "d", "u", "d"),
    ("word", "up", "dn", "↑", "↓"),
    ("latex arrow", r"\uparrow", r"\downarrow", "↑", "↓"),
    ("unicode arrow", "↑", "↓", "↑", "↓"),
]


class TestEverySpellingReachesTheSameOperation:
    @pytest.mark.parametrize("name,up,dn,cup,cdn", _SPELLINGS)
    @pytest.mark.parametrize("template", [_JA, _EN], ids=["ja", "en"])
    def test_arrows_are_accepted(self, name, up, dn, cup, cdn, template):
        res = _handler().extra_pre_phase1(template.format(up=up, dn=dn), "ja")
        assert res is not None and res.operation == "operator_substitute"
        assert res.args["op_rules"] == [
            {"op_name": "c", "formal_index": "k", "spin_label": cup,
             "rhs": "u(k) * a(k)"},
            {"op_name": "c", "formal_index": "-k", "spin_label": cdn,
             "rhs": "v(k) * b(k)"},
        ]


class TestTheEnglishRouteIsNotMangled:
    r"""Found while measuring J-3, and worse than J-3 was.

    `_normalize_latex_subscripts` inserts an implicit `*` after a closing paren,
    and it runs over the WHOLE instruction.  In English the words after a paren
    are the instruction's own — "…a(k) for c(k,u) and…" became
    "…a(k) * for c(k,u) * and…" — so the captured right-hand side ended in a
    bare `*`.  Nothing failed: `u(k) * a(k) *` parses as a SYMBOL whose name is
    that entire string, so the substitution's RHS quietly stopped being a
    product and became one opaque atom.
    """

    @pytest.mark.parametrize("name,up,dn,cup,cdn", _SPELLINGS)
    def test_no_rhs_ends_in_an_operator(self, name, up, dn, cup, cdn):
        res = _handler().extra_pre_phase1(_EN.format(up=up, dn=dn), "en")
        for rule in res.args["op_rules"]:
            assert not rule["rhs"].rstrip().endswith("*")

    def test_the_rhs_still_parses_as_a_product_not_an_atom(self):
        from notebook.models.ir import Symbol
        from notebook.runtime.shared_parser import SharedParser
        res = _handler().extra_pre_phase1(_EN.format(up="u", dn="d"), "en")
        for rule in res.args["op_rules"]:
            parsed = SharedParser().parse(rule["rhs"])
            assert not isinstance(parsed, Symbol), (
                f"{rule['rhs']!r} collapsed into a single symbol")

    def test_genuine_juxtaposition_is_still_multiplication(self):
        """The carve-out names instruction words only — it must not switch the
        implicit-multiplication rule off for mathematics."""
        out = _handler()._normalize_latex_subscripts("u_k alpha(k) v(k) beta(k)")
        assert out == "u(k) * alpha(k) * v(k) * beta(k)"


class TestTwoRulesInBothLanguages:
    r"""EN-MULTI-OPSUBST — 日本語が解決済みなら、それは i18n の穴である。

    Japanese has three patterns for a two-rule substitution (same operator with
    two spin labels / same operator at ±k / two different operators); English
    had only the FIRST, because `_OP_SUBST_SPIN_RE_EN` requires a
    comma-separated spin label.  Measured 2026-08-18:

        JA  E7 の a(1) に E4 を、a'(1) に E5 を代入して        → 2 rules
        EN  In E7, substitute E4 for a(1) and E5 for a'(1)   → 1 rule

    and the English one does not FAIL — the single-rule pattern matches and
    `formal_index` swallows the rest of the sentence (`1) and E5 for a'(1`),
    recorded that way in `scenario_squeezed_state_en.json`.

    **Japanese being solved is what makes this an i18n gap rather than a feature
    request** (the ticket had gated it on "when English multi-rule substitution
    is needed").  The same instruction must work in both languages.
    """

    @staticmethod
    def _h():
        ctx = SymbolContext()
        for n in ("a", "c"):
            ctx.register_quantity(n, kind="operator")
        h = JuliaBackendHandler(None)
        h._context = ctx
        return h

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E7 の a(1) に E4 を、a'(1) に E5 を代入して"),
        ("en", "In E7, substitute E4 for a(1) and E5 for a'(1)"),
    ])
    def test_same_base_with_a_prime(self, lang, text):
        args = self._h().extra_pre_phase1(text, lang).args
        assert [(r["op_name"], r["formal_index"], r["rhs"])
                for r in args["op_rules"]] == [("a", "1", "E4"), ("a'", "1", "E5")]

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 の a(1) に X を、c(1) に Y を演算子代入して"),
        ("en", "In E1, substitute X for a(1) and Y for c(1)"),
    ])
    def test_two_different_operators(self, lang, text):
        args = self._h().extra_pre_phase1(text, lang).args
        assert [r["op_name"] for r in args["op_rules"]] == ["a", "c"]

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 の c(k) に X を、c(-k) に Y を演算子代入して"),
        ("en", "In E1, substitute X for c(k) and Y for c(-k)"),
    ])
    def test_the_nambu_pair_keeps_its_own_operation(self, lang, text):
        """`op(k)`/`op(-k)` on ONE operator is a different operation, not two
        rules — the shape decides, and it must decide the same way in both."""
        res = self._h().extra_pre_phase1(text, lang)
        assert res.operation == "operator_substitute_pair"
        assert (res.args["rhs"], res.args["rhs_neg"]) == ("X", "Y")

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E1 の c(k) に X を代入して"),
        ("en", "In E1, substitute X for c(k)"),
    ])
    def test_a_single_rule_is_still_a_single_rule(self, lang, text):
        args = self._h().extra_pre_phase1(text, lang).args
        assert "op_rules" not in args and args["op_name"] == "c"

    def test_no_formal_index_swallows_the_sentence(self):
        """The shape of the original defect: a captured index that is really the
        rest of the instruction."""
        for lang, text in (("en", "In E7, substitute E4 for a(1) and E5 for a'(1)"),
                           ("en", "In E1, substitute X for a(1) and Y for c(1)")):
            args = self._h().extra_pre_phase1(text, lang).args
            idxs = ([r["formal_index"] for r in args["op_rules"]]
                    if "op_rules" in args else [args.get("formal_index", "")])
            for i in idxs:
                assert " " not in i and ")" not in i, i
