"""2a-sym must not rewrite a symbol that IS the operation's variable.

Once a researcher has solved for a symbol (``E3: k = …``), that symbol is in
``lhs_alias_map``, and the supervisor pre-pass replaces its every occurrence
with the E-ID.  Two spellings must survive that: the differentiation variable
(``d/dk``, not ``d/dE3``) and H-9's solve-for variable (solve for the named
quantity ``k``, not for the whole expression ``E3``).

**Why this file exists.**  The guard used to be written TWICE — a negative
lookahead and a whole-command check, carrying byte-identical Japanese 35 lines
apart — and 2026-08-27 they were collapsed onto one shared pair of fragments
(``ja_op_var_tail`` / ``en_op_var_head`` in ``i18n/nl_patterns.json``).  Sharing
is the point, and it is also the risk: one edit now moves both guards at once,
and what the existing tests pinned was a single Japanese spelling family
(``E1 を k で 2 回微分して``, in ``test_diff_order_parsing``).  The order forms,
``について``, ``全微分`` and **every English mirror** were unpinned — measured by
instrumenting a full scenario replay, seven English lines reach this guard.

The English half is checked at this level on purpose: breaking
``en_op_var_head`` fails no end-to-end test, because the fixed-width lookbehind
beside it catches the trailing occurrence.  That redundancy is deliberate (see
the module's comment), so the only place the English words themselves can be
held to account is here — and it is not a formality: with that fragment broken,
exactly the four cases the lookbehind cannot see fail here — ``w.r.t.``, ``wrt``
(spellings it does not know) and the two whole-command ones, where the variable
that must survive is inside the FORMULA rather than after the keyword.
"""
import pytest

from ui.supervisor_alias_rewrite import rewrite_symbols_to_eids


class _Eng:
    """The one attribute the pass reads."""

    def __init__(self, alias):
        self.lhs_alias_map = alias


#: (japanese, english) — the SAME instruction, so a fragment edit that drops one
#: language shows up as a mirror that no longer matches.
MIRRORED = [
    ("E1 を k で微分して",            "Differentiate E1 with respect to k"),
    ("E1 を k で偏微分して",           "Take the partial derivative of E1 with respect to k"),
    ("E1 を k で全微分して",           "Take the total derivative of E1 with respect to k"),
    ("E1 を k で 2 回微分して",        "Differentiate E1 twice with respect to k"),
    ("E1 を k で二階微分して",         "Differentiate E1 w.r.t. k, order 2"),
    ("E1 を k で 3 次微分して",        "Differentiate E1 with respect to k to order 3"),
    ("E1 を k について微分して",        "Differentiate E1 wrt k"),
    ("E1 を k について 2 回偏微分して",  "Take the 2nd partial derivative of E1 with respect to k"),
    ("E1 を k について解いて",         "Solve E1 for k"),
]


class TestTheOperationVariableSurvivesInBothLanguages:
    @pytest.mark.parametrize("jp,en", MIRRORED)
    def test_japanese(self, jp, en):
        assert rewrite_symbols_to_eids(jp, _Eng({"k": "E3"})) == jp

    @pytest.mark.parametrize("jp,en", MIRRORED)
    def test_english(self, jp, en):
        assert rewrite_symbols_to_eids(en, _Eng({"k": "E3"})) == en


class TestTheGuardIsNotSimplyAlwaysOn:
    """A positive control.  Without it every assertion above would still pass
    if the rewrite stopped working altogether."""

    @pytest.mark.parametrize("text,want", [
        ("E1 と k の交換子",      "E1 と E3 の交換子"),
        ("simplify k + 1",       "simplify E3 + 1"),
        ("k を展開して",          "E3 を展開して"),
    ])
    def test_an_ordinary_occurrence_is_rewritten(self, text, want):
        assert rewrite_symbols_to_eids(text, _Eng({"k": "E3"})) == want


class TestTheWholeCommandIsSearchedNotJustTheAdjacentOccurrence:
    """SILENT-DIFF-VAR-ALIAS: the formula's OWN free variable must survive too,
    not only the one written after the keyword.  Measured against a live Maxima
    when it was found — the corrupted command returned 0 instead of a-b*y."""

    @pytest.mark.parametrize("text", [
        "differentiate a*x-b*x*y with respect to x",
        "Take the partial derivative of a*x - b*x*y with respect to x",
        "a*x-b*x*y を x で偏微分して",
    ])
    def test_every_occurrence_is_skipped(self, text):
        assert rewrite_symbols_to_eids(text, _Eng({"x": "E5"})) == text
