r"""JSON↔TeX cross-check over every published pair (TEX-EXPORT-SCAN 残 b).

`test_tex_export_sanity.py` reads each `.tex` on its own and therefore only sees
defects somebody has already written a rule for.  This one reads the `.tex`
BESIDE its session JSON, which is the whole notebook — so a card that never
reached the document, a repeated or dangling equation number, and latex mangled
on the way out are all visible without a rule per defect.

Needs no TeX engine and no CAS: it reads files already in the tree.
"""
import json
import pathlib

import pytest

from notebook.runtime.tex_json_crosscheck import crosscheck

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAIRS = sorted(
    (j, j.with_suffix(".tex"))
    for j in ROOT.glob("domains/*/examples/scenario_*.json")
    if j.with_suffix(".tex").exists()
)


def test_pairs_exist():
    assert len(PAIRS) > 50, f"only {len(PAIRS)} json/tex pairs found"


@pytest.mark.parametrize("jp,tp", PAIRS, ids=lambda p: p.name)
def test_published_pair_agrees(jp, tp):
    findings = crosscheck(json.loads(jp.read_text(encoding="utf-8")),
                          tp.read_text(encoding="utf-8"))
    assert not findings, "\n".join(f"  {c}: {d}" for c, d in findings)


_TEX = r"""\begin{equation}
  H = p^2/(2 m)
  \label{eq:E1}
\end{equation}
\begin{equation}
  L = p_{\theta}
  \label{eq:E2}
\end{equation}
"""
_JSON = {"expressions": {
    "E1": {"latex": "H = p^2/(2 m)"},
    "E2": {"latex": "L = p_{\\theta}"},
}}


class TestTheGuardCatchesItsOwnDefects:
    """A guard that has never been shown to fail is not evidence of anything.

    The two shapes here are the ones a researcher found BY EYE in one file
    (open_items TEX-EXPORT-SCAN): a scaffold expression published, and an
    E7→E6 gap in the numbering.
    """

    def test_clean_pair_reports_nothing(self):
        assert crosscheck(_JSON, _TEX) == []

    def test_card_missing(self):
        """An expression that is neither internal nor hidden, absent from the
        document — the researcher sees it in the notebook and not in the paper."""
        j = json.loads(json.dumps(_JSON))
        j["expressions"]["E3"] = {"latex": "T = 0"}
        codes = [c for c, _ in crosscheck(j, _TEX)]
        assert codes == ["card-missing"]

    def test_scaffolding_is_not_a_missing_card(self):
        """`internal`/`hidden` is how the notebook already says "not for the
        document"; flagging those would be a false positive on every session."""
        for flag in ("internal", "hidden"):
            j = json.loads(json.dumps(_JSON))
            j["expressions"]["E3"] = {"latex": "T = 0", flag: True}
            assert crosscheck(j, _TEX) == []

    def test_label_duplicated(self):
        """Two equations carrying one number — the E7→E6 shape."""
        tex = _TEX.replace("eq:E2", "eq:E1")
        codes = [c for c, _ in crosscheck(_JSON, tex)]
        assert "label-duplicated" in codes

    def test_label_orphan(self):
        tex = _TEX.replace("eq:E2", "eq:E9")
        codes = [c for c, _ in crosscheck(_JSON, tex)]
        assert "label-orphan" in codes

    def test_latex_mangled_on_the_way_out(self):
        """The notebook still shows it correctly; only the export is wrong."""
        tex = _TEX.replace("H = p^2/(2 m)", "H = p^2/(2 n)")
        codes = [c for c, _ in crosscheck(_JSON, tex)]
        assert codes == ["latex-mangled"]

    def test_a_dropped_term_is_caught(self):
        tex = _TEX.replace("H = p^2/(2 m)", "H =")
        assert [c for c, _ in crosscheck(_JSON, tex)] == ["latex-mangled"]


class TestTheOneNormalisation:
    r"""`*`↔space is a DELIBERATE display bridge, not a defect.

    Measured over the 90 committed pairs, it accounts for all 58 differences
    and nothing else differs.  Widening this list is how the check would stop
    working — `tex_sanity`'s first `bare-star` rule was 23/23 false positives,
    and on a check that runs over every published example a false positive
    costs more than a miss.
    """

    def test_star_becomes_a_space(self):
        j = {"expressions": {"E1": {"latex": "a * b * c"}}}
        tex = "\\begin{equation}\n  a b c\n  \\label{eq:E1}\n\\end{equation}\n"
        assert crosscheck(j, tex) == []

    def test_indentation_is_ignored(self):
        j = {"expressions": {"E1": {"latex": "a+b"}}}
        tex = "\\begin{equation}\n        a + b\n  \\label{eq:E1}\n\\end{equation}\n"
        assert crosscheck(j, tex) == []

    def test_but_a_changed_symbol_is_still_caught(self):
        j = {"expressions": {"E1": {"latex": "a * b"}}}
        tex = "\\begin{equation}\n  a c\n  \\label{eq:E1}\n\\end{equation}\n"
        assert [c for c, _ in crosscheck(j, tex)] == ["latex-mangled"]
