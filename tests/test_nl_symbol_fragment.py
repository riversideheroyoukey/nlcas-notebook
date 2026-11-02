r"""One question, one answer: how may a researcher spell a symbol name?

INTEGRAL-VAR-BRACED was filed as an integral bug.  Measured 2026-08-06, it was
not: the question "what can the variable of a differentiation / integration /
Taylor expansion look like?" was answered in THREE places, each partially, and
no two the same way.

    pattern            \xi_k   \xi_{k}   x_1   x_{1}
    diff_rule            OK      OK       OK    no
    taylor_expand_nl     OK      no       OK    no
    def_integral         OK      no       OK    no

The braced spellings that missed fell through to the LLM and came back
"Could not interpret", instead of the operation's own advice — which for an
undeclared subscripted name is the actionable one ("declare it as a single
name", see tests/test_integral_workflow.py).

The answer now lives once, in ``nl_patterns.json`` under ``_fragments.symbol``,
spliced in with ``{{symbol}}``.  This file pins the property that matters:
**every pattern that captures a variable accepts the same spellings, in both
languages.**  A new such pattern belongs in ``_VAR_PATTERNS`` below.

**And for a year that sentence was broader than this file.**  Measured
2026-08-27 (NL-GRAMMAR ①): ten patterns capture a variable and only three were
listed here, so the property was pinned over three-tenths of what it claimed.
The four that spelled the variable themselves were narrower than the fragment —
``ode_general`` took a bare identifier, ``specialize_condition`` a SINGLE
LETTER — and the consequence was not a refusal but a **different operation
answering**: ``E1 を \theta について微分方程式として解いて`` was claimed by
``differentiate`` and reported completed, and ``solve E1 for \theta in
general`` read its variable as the English word ``general``.  θ is the
pendulum's own variable.  All ten are listed below now; adopting the fragment
gained 7-11 spellings each and lost none.

A second, adjacent question fell out of the same measurement and is fixed with
it: ``ode_general``'s verb alternatives carried the PARTICLE (``を一般に解いて``),
so once the variable clause had consumed that particle none of them could
match.  ``一般に解いて`` and ``一般解を求めて`` had particle-less siblings; the
微分方程式 forms did not, which is why only they fell through.
"""
import pytest

from notebook.runtime.nl_patterns import _expand_fragments, nlp

# key → {lang: template with {v}}.  The captured group is var / var1.
_VAR_PATTERNS = {
    "def_integral": {
        "ja": "{v}^2 を {v} で 0 から 1 まで定積分して",
        "en": "integrate {v}^2 with respect to {v} from 0 to 1",
    },
    "taylor_expand_nl": {
        "ja": "E1 を {v} について 0 のまわりで 2 次までテイラー展開して",
        "en": "taylor expand E1 in {v} around 0 to order 2",
    },
    "diff_rule": {
        "ja": "E1 を {v} で微分して",
        "en": "differentiate E1 with respect to {v}",
    },
    # --- Joined 2026-08-27 (NL-GRAMMAR ①, second question).  The table above
    # was three keys; TEN patterns capture a variable, and the four below spelled
    # it themselves — narrowly.  `ode_general` took a bare `[A-Za-z]\w*` and
    # `specialize_condition` a SINGLE LETTER, so 7 and 11 of the twelve spellings
    # below never reached them.
    "solve_single": {
        "ja": "E1 を {v} について解いて",
        "en": "solve E1 for {v}",
    },
    # 2026-09-06（二十五問目）: 同じ要求の語順違い。実在する言い方なのに
    # どの規則も読めず失敗していた。
    "solve_from_expr": {
        "ja": "E1 から {v} を求めて",
        "en": "find {v} from E1",
    },
    "solve_solution_of_var": {
        "ja": "E1 の解を {v} について求めて",
        "en": "find the solution of E1 for {v}",
    },
    "solve_system_vars": {
        "ja": "E1 と E2 の連立方程式を {v} について解いて",
        "en": "solve E1 and E2 for {v}",
    },
    "ode_general": {
        "ja": "E1 を {v} について一般解を求めて",
        "en": "solve E1 for {v} in general",
    },
    "inv_transform_vars": {
        "ja": "E1 を {v} について逆変換",
        "en": "invert E1 for {v}",
    },
    "numeric_fork_sweep_nl": {
        "ja": "{v} を 0 から 1 まで 10 点で掃引して",
        "en": "sweep {v} from 0 to 1 with 10 points",
    },
    "z_transform_by_definition": {
        "ja": "f を定義に従って {v} から z へZ変換して",
        "en": "z-transform f from {v} to z by definition",
    },
    "specialize_condition": {
        "ja": "E1 で {v} = 0 のとき 1 になる条件として決めて",
        "en": "in E1, when {v} = 0, determine c such that this equals 1",
    },
    # Joined 2026-09-05: the ANSWER to "which variable should be solved for?".
    # It was `^[A-Za-z]$`, and a longer answer did not merely fail — the pending
    # question was dropped and the answer was registered as a new expression.
    "pending_var_answer": {"ja": "{v}", "en": "{v}"},
}

# Every spelling the notebook itself can DISPLAY back to the researcher, plus
# the declaration forms CLAUDE.md's subscript table calls out.
_SPELLINGS = [
    "x",                # bare
    "x_1", "x_{1}",     # numeric subscript, unbraced / braced
    "omega_D",          # ascii Greek name with a letter subscript
    r"\theta",          # LaTeX command
    r"\xi_k", r"\xi_{k}",   # command + subscript, unbraced / braced
    r"\dot{q}",         # command with a braced argument (analytical mechanics)
    r"\dot{q}_1",       # …and a subscript on top of it
    r"g_{\mu\nu}",      # multi-index (refused LATER, by the operation, not here)
    "H_BCS",            # multi-char subscript = part of the name
    "a_k_s",            # multiple subscripts
    # --- joined 2026-08-27 with the fragment's widening (① third question)
    "G^0",              # a superscript LABEL is a name (CLAUDE.md)
    r"\Sigma^{(HF)}",   # …braced, on a command
    "G^{(HF)}_{n}",     # …and a subscript after it
    "f(x)",             # applied — what an ODE is solved FOR
    "psi(x,t)",         # …with two arguments
]


#: The group names the variable is captured under.  Several, because Python
#: forbids repeating a group NAME across alternatives, so a key with two word
#: orders numbers them (``var1``/``var2``…) — and a key that takes a LIST calls
#: it ``vars``.  Reading only ``var``/``var1`` (as this helper did until
#: 2026-08-27) silently reports "no capture" for every other key, which is what
#: kept seven of the ten patterns out of this file.
_VAR_GROUPS = ("var", "var1", "vars", "var2", "var3", "var4", "nvar")


def _captured(key: str, lang: str, text: str):
    m = nlp(key, lang).search(text)
    if m is None:
        return None
    g = m.groupdict()
    for name in _VAR_GROUPS:
        if g.get(name):
            return g[name]
    return None


@pytest.mark.parametrize("key", sorted(_VAR_PATTERNS))
@pytest.mark.parametrize("lang", ["ja", "en"])
@pytest.mark.parametrize("var", _SPELLINGS)
def test_every_variable_pattern_accepts_every_spelling(key, lang, var):
    text = _VAR_PATTERNS[key][lang].format(v=var)
    assert _captured(key, lang, text) == var, (
        f"{key}/{lang} did not capture {var!r} from {text!r}")


@pytest.mark.parametrize("var", _SPELLINGS)
def test_the_two_languages_agree(var):
    """A spelling accepted in Japanese must be accepted in English.

    `diff_rule` EN ends at the variable, so there is no trailing context to
    force backtracking — measured, a greedy `[A-Za-z_]\\w*` gave `x_{1}` in JA
    and `x_` in EN.  The fragment forbids a trailing `_` in the base for exactly
    this reason; without a cross-language check the two drift apart silently.
    """
    for key, forms in _VAR_PATTERNS.items():
        ja = _captured(key, "ja", forms["ja"].format(v=var))
        en = _captured(key, "en", forms["en"].format(v=var))
        assert ja == en == var, f"{key}: ja={ja!r} en={en!r} expected {var!r}"


#: Keys that capture a variable WITHOUT ``{{symbol}}``, and why.  Measured
#: 2026-08-27; each entry is a decision, not a backlog.
#:
#: * ``solve_system_vars`` / ``inv_transform_vars`` — these capture a LIST
#:   (``x と y``), split downstream.  ``{{symbol}}`` describes one name and
#:   cannot express a list; the question there is a different one.
#:
#: ``solve_single`` was here until 2026-08-27, on the ground that its own regex
#: was a strict superset — Unicode Greek, superscript labels, applied functions.
#: Measured, **that reason was two-thirds wrong**: the labels and the calls were
#: real and the fragment absorbed them, and the Greek never arrives, because
#: ``normalize_request_greek`` converts ``ω`` → ``\omega`` once per request
#: BEFORE any interceptor.  What normalization leaves raw is ``Α Β Ε Ζ Η …`` —
#: the Greek letters with no TeX spelling of their own, indistinguishable from
#: Latin — and accepting one of those silently makes a second object for a
#: quantity already declared as ``A``, which is the defect that normalization's
#: own docstring exists to name.  Refusing it is the better answer, so
#: ``solve_single`` now shares the fragment.
_SPELLS_ITS_OWN = [
    "solve_system_vars[ja]", "solve_system_vars[en]",
    "inv_transform_vars[ja]", "inv_transform_vars[en]",
]


class TestEveryVariableCapturingPatternIsListed:
    """The table above must BE the set of variable-capturing patterns.

    This is the guard that would have caught the gap: for a year the file
    claimed to pin "every pattern that captures a variable" while listing three
    of ten, and nothing said so.  A new key with a ``var``-ish group and no row
    above fails here.
    """

    def test_no_variable_capturing_key_is_missing_from_the_table(self):
        import json
        import re
        from notebook.runtime.nl_patterns import _PATH
        data = json.loads(_PATH.read_text(encoding="utf-8"))
        found = set()
        for key, entry in data.items():
            if key.startswith("_") or not isinstance(entry, dict):
                continue
            for lang, src in entry.items():
                if lang.startswith("_"):
                    continue
                if any(f"(?P<{g}>" in src for g in _VAR_GROUPS):
                    found.add(key)
        missing = sorted(found - set(_VAR_PATTERNS))
        assert not missing, (
            "these patterns capture a variable but are not exercised here — add "
            f"a row to _VAR_PATTERNS: {missing}")

    def test_every_listed_key_uses_the_fragment_or_says_why_not(self):
        """Accepting the same spellings today is not the same as sharing the
        answer: a key that spells the variable itself drifts the moment the
        fragment grows — which is exactly how ``ode_general`` ended up seven
        spellings behind (measured 2026-08-27).

        Three keys legitimately do not use it, and each reason is a different
        one.  A key that is NOT in this table and spells the variable itself
        fails here, which is the point.
        """
        import json
        from notebook.runtime.nl_patterns import _PATH
        data = json.loads(_PATH.read_text(encoding="utf-8"))
        own = [f"{k}[{l}]" for k in _VAR_PATTERNS for l in ("ja", "en")
               if l in data[k] and "{{symbol}}" not in data[k][l]]
        assert sorted(own) == sorted(_SPELLS_ITS_OWN), (
            "a variable spelling drifted from the shared fragment — adopt "
            "{{symbol}}, or add it to _SPELLS_ITS_OWN with a reason: "
            f"{sorted(set(own) ^ set(_SPELLS_ITS_OWN))}")


class TestTheOdeVerbDoesNotCarryTheParticle:
    """``を`` belongs to the variable clause, not to the verb.

    The verb alternatives used to be written ``を一般に解いて`` / ``の微分方程式を
    解いて``; once ``を X について`` had consumed that particle, only the two that
    happened to have a particle-less sibling could still match.  So
    ``E1 を \theta について微分方程式として解いて`` fell through and
    ``differentiate`` answered it — **completed**, with the wrong operation.
    """

    @pytest.mark.parametrize("verb", [
        "一般に解いて", "一般解を求めて", "微分方程式として解いて",
        "微分方程式を解いて", "微分方程式の解を求めて",
    ])
    @pytest.mark.parametrize("clause", ["", "を x について", "を \\theta について"])
    def test_every_verb_reads_with_and_without_the_variable_clause(self, verb, clause):
        text = f"E1 {clause}{verb}".replace("  ", " ")
        assert nlp("ode_general", "ja").search(text), text

    @pytest.mark.parametrize("particle", ["を", "の"])
    def test_the_particle_form_still_reads(self, particle):
        """The spellings that already worked must keep working — the fix makes
        the particle optional, it does not move it."""
        assert nlp("ode_general", "ja").search(f"E1 {particle}一般解を求めて")


class TestTheFragmentMechanism:
    def test_an_unknown_fragment_is_an_error_not_a_literal(self):
        """Leaving `{{nope}}` in the regex would compile fine and never match —
        the silent failure this whole fix is about."""
        with pytest.raises(KeyError, match="unknown fragment"):
            _expand_fragments("^{{nope}}$", "some_key", "ja")

    def test_expansion_keeps_backslashes(self):
        """`re.sub` processes escapes in a STRING replacement; these fragments
        are mostly backslashes, so the replacement must be a function."""
        out = _expand_fragments("^(?P<var>{{symbol}})$", "k", "ja")
        assert "{{" not in out
        assert r"\\" in out or "\\[A-Za-z]" in out

    def test_no_pattern_references_a_fragment_that_does_not_exist(self):
        """Compiling every pattern in both languages is the whole check: an
        unknown reference raises inside nlp().

        A sub-key starting with ``_`` is METADATA, not a language — the same
        convention the file already uses at top level for ``_fragments`` and
        ``_comment``.  Patterns migrated out of source carry their reason with
        them as a per-key ``_comment`` (that is the point of moving them: the
        measurement that justifies a pattern must travel with it), and without
        this skip the test tried to compile Japanese prose as a regex.
        """
        import json
        from notebook.runtime.nl_patterns import _PATH
        data = json.loads(_PATH.read_text(encoding="utf-8"))
        checked = 0
        for key, entry in data.items():
            if key.startswith("_") or not isinstance(entry, dict):
                continue
            for lang in entry:
                if lang.startswith("_"):
                    continue
                nlp(key, lang)
                checked += 1
        assert checked >= 300, f"only {checked} patterns compiled — did the skip eat them?"

    def test_a_fragment_may_reference_a_fragment(self):
        """``fragment()`` expands, exactly as ``nlp()`` does.

        Measured 2026-08-27, the day ``ja_op_var_tail`` began referencing
        ``ja_digit``: ``fragment()`` handed back the RAW text, so the 2a-sym
        guard's regex carried a literal ``{{ja_digit}}``, compiled fine, and
        never matched — a differentiation variable was rewritten to an E-ID
        (``var: E5`` for ``var: x``).  The instruction corpus caught it, and so
        did that guard's own tests.

        Mutating ``fragment()`` back to the raw return fails eleven tests;
        mutating the fixed-point loop back to a single ``sub`` fails none, since
        only one level of nesting exists today.  The loop is for the next level.
        """
        from notebook.runtime.nl_patterns import fragment
        out = fragment("ja_op_var_tail")
        assert "{{" not in out, out
        # 断言は**機構**に当てる: 参照した2つの fragment の中身が展開されて
        # いること。語彙そのものを直書きすると、語を1つ足すたびにここが落ちる
        # ——実測 2026-09-06、`ja_deriv_word` に `常微分`/`導関数` を足したら
        # `(?:偏|全)?微分` という直書きで落ちた（機構は壊れていない）。
        from notebook.runtime.nl_patterns import fragment as _f
        assert _f("ordinal_token") in out, out     # 数の fragment が展開された
        assert _f("ja_deriv_word") in out, out     # 微分の語の fragment が展開された
        assert "微分" in out

    def test_no_fragment_leaves_a_reference_unexpanded(self):
        """The property, over every fragment — not just the nested one that
        happened to break."""
        import json
        from notebook.runtime.nl_patterns import _PATH, fragment
        frags = json.loads(_PATH.read_text(encoding="utf-8"))["_fragments"]
        for name in frags:
            assert "{{" not in fragment(name), name

    def test_a_fragment_cycle_is_an_error_not_a_hang(self, monkeypatch):
        """Running to a fixed point must terminate on a cycle, and say so."""
        import notebook.runtime.nl_patterns as NP
        monkeypatch.setattr(NP, "_load",
                            lambda: {"_fragments": {"a": "x{{b}}", "b": "y{{a}}"}})
        with pytest.raises(ValueError, match="unexpanded"):
            NP._expand_fragments("{{a}}", "k", "ja")
