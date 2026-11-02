r"""OPERAND-SPELLING (2026-09-11) — one answer to "how is an operator operand spelled".

The Heisenberg family asked that question with THREE hand-written character
classes, and each was wrong somewhere else.  Measured over 72 combinations
(7 word orders x 9 operand shapes, both languages), 18 were broken::

    ja  「OP を E1 のもとで…」   [^\s、,]+?  → `c(k,u)` came back as `u)`  — SILENT
    ja  「E1 のもとで OP を…」   julia_op_call → `c_{1\uparrow}` no match
    en  「evolve OP under E1」   \S+?        → `c(1, u)` no match

Every broken spelling is real: `c(k,u)` appears 118 times in the instruction
corpus, `c_{1\uparrow}` is a corpus spelling too, and `c(1, u)` — with the space
— is what Maxima's own `tex1` prints (`c\left(1 , u\right)`), so a researcher
reading a result off the screen and typing it back writes that one.

The one answer is the fragment `operator_operand`; the old name (`julia_op_call`)
already lied, since `Sz_1` is not a call, and a name that lies is how a fourth
copy gets written.
"""
import json
import pathlib
import re

import pytest

from notebook.runtime.nl_patterns import nlp

#: Every shape a researcher writes for an operator operand.  Spellings are taken
#: from the corpus and from what the system itself prints — not invented.
OPERANDS = [
    "c(1)",                     # single index
    "c(k,u)",                   # two indices, no space   — corpus, 118 rows
    "c(1, u)",                  # two indices, WITH space — what `tex1` prints
    "c'(k,u)",                  # creation operator
    "c'(1, u)",
    "Sz_1",                     # spin component, plain subscript
    r"c_{1\uparrow}",           # braced subscript with a LaTeX command — corpus
    r"c^\dagger_{1\uparrow}",   # the dagger spelling of the same
    "a(1)",
]

#: (label, key, language, sentence).  **Each key is measured with ITS OWN
#: sentence shape** — `of OP under HAM` and `of HAM for OP` are different keys,
#: and measuring one with the other's sentence reports a defect that is really a
#: mis-aimed probe (which is what happened first, 2026-09-11).
FORMS = [
    ("operator ja1", "heisenberg_operator_nl", "ja", "{op} を E1 のもとで時間発展させて"),
    ("operator ja2", "heisenberg_operator_nl", "ja", "E1 のもとで {op} を時間発展させて"),
    ("operator ja3", "heisenberg_operator_nl", "ja", "E1 のもとでの {op} のハイゼンベルク演算子"),
    ("operator en1", "heisenberg_operator_nl", "en", "evolve {op} under E1"),
    ("operator en2", "heisenberg_operator_nl", "en", "the Heisenberg operator of {op} under E1"),
    ("eom ja", "heisenberg_eom_op_first", "ja", "{op} の Heisenberg 方程式 E1"),
    ("eom en under", "heisenberg_eom_under_first", "en", "Heisenberg equation of {op} under E1"),
    ("eom en verb", "heisenberg_eom_verb_first", "en", "Heisenberg equation of E1 for {op}"),
]


def _operand_of(match, ) -> "str | None":
    if match is None:
        return None
    named = [v for k, v in match.groupdict().items() if v and k.startswith("op")]
    if named:
        return named[0]
    for group in match.groups():                 # positional-group keys
        if group and not re.fullmatch(r"E\d+", group.strip()):
            return group.strip()
    return None


class TestEveryWordOrderReadsEveryOperand:
    @pytest.mark.parametrize("label,key,lang,form", FORMS,
                             ids=[f[0].replace(" ", "_") for f in FORMS])
    @pytest.mark.parametrize("operand", OPERANDS)
    def test_operand_is_read_whole(self, label, key, lang, form, operand):
        got = _operand_of(nlp(key, lang).search(form.format(op=operand)))
        assert got == operand, f"{label}: {operand!r} read as {got!r}"


class TestTheOperandClassIsWrittenOnce:
    def test_no_key_in_the_family_spells_its_own_class(self):
        r"""`\S+` and `[^\s、,]` are the three answers this fold replaced."""
        patterns = json.loads(
            pathlib.Path("i18n/nl_patterns.json").read_text(encoding="utf-8"))
        offenders = []
        for key, entry in patterns.items():
            if not key.startswith("heisenberg") or not isinstance(entry, dict):
                continue
            for lang in ("ja", "en"):
                spelling = entry.get(lang)
                if isinstance(spelling, str) and (r"\S+" in spelling
                                                  or r"[^\s、,]" in spelling):
                    offenders.append(f"{key}[{lang}]")
        assert not offenders, offenders

    def test_the_old_name_is_gone(self):
        r"""`julia_op_call` said "call" and matched `Sz_1`; a name that lies is
        how the next copy gets written (the `en_op_var_head_wide` rename)."""
        raw = pathlib.Path("i18n/nl_patterns.json").read_text(encoding="utf-8")
        assert "julia_op_call" not in raw


class TestTheBaseMayNotEndInUnderscore:
    r"""`\w` includes `_`, so a greedy base eats the `_` of `c_{1\uparrow}` and
    stops at the brace.  A key with trailing context backtracks out of that; a
    key that ENDS at the operand has nothing to backtrack against — measured
    2026-09-11 on `Heisenberg equation of E1 for c_{1\uparrow}`, which read
    `c_`.  The `symbol` fragment records the identical trap.
    """

    def test_a_key_that_ends_at_the_operand(self):
        got = _operand_of(nlp("heisenberg_eom_verb_first", "en").search(
            r"Heisenberg equation of E1 for c_{1\uparrow}"))
        assert got == r"c_{1\uparrow}"
