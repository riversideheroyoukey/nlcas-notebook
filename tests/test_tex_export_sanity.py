"""Scan every published `.tex` example (EXAMPLES-DRIFT's companion guard).

The examples ARE the practice loop's output, and until 2026-08-14 nothing read
them.  Five defects were found by a researcher reading one file by eye; four had
been sitting in committed examples for weeks, and each has a mechanical shape.
This runs `notebook/runtime/tex_sanity.scan` over all of them.

It needs no TeX engine, so unlike the drift check it is NOT opt-in: it reads
files that are already in the tree and takes milliseconds.  A COMPILE pass would
answer a further question (does it typeset?) and is filed as open_items
TEX-EXPORT-SCAN.

The `test_catches_*` cases below are the point of the file: a guard that has
never been shown to fail is not evidence of anything.  Each feeds the scanner
the EXACT text that was published, taken from the commit that carried it.
"""
import pathlib

import pytest

from notebook.runtime.tex_sanity import scan

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES = sorted(ROOT.glob("domains/*/examples/*.tex"))


def test_examples_exist():
    assert len(EXAMPLES) > 50, f"only {len(EXAMPLES)} .tex examples found"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_published_example_scans_clean(path):
    findings = scan(path.read_text(encoding="utf-8"))
    assert not findings, "\n".join(
        f"  L{line} {code}: {detail}" for line, code, detail in findings)


class TestTheGuardCatchesItsOwnDefects:
    """Faithful reproductions of what was actually published."""

    def test_catches_raw_canonical_in_a_math_slot(self):
        # scenario_damped_oscillator_numeric_jp E6: the H-5 right-hand-side
        # scaffolding was registered with `latex=<canonical>`.  Maxima's `%e`
        # opens a LaTeX COMMENT, so the rest of that line vanished silently.
        src = ("\\begin{equation}\n"
               "  %e^-(t \\gamma) (\\cos((t \\sqrt{4 \\omega_{0}^2})/2)\n"
               "\\end{equation}")
        codes = {c for _, c, _ in scan(src)}
        assert "comment-in-math" in codes and "cas-syntax-leak" in codes

    def test_catches_double_subscript(self):
        # An IC free parameter surviving a solve: `x_ic1` unwrapped to a double
        # subscript, which is a HARD LaTeX error — the file would not typeset.
        src = r"\begin{equation}  \left(m s^2\right) x_{ic}_{1}  \end{equation}"
        assert "double-subscript" in {c for _, c, _ in scan(src)}

    def test_catches_angle_bracket_in_text_mode(self):
        # The preamble omits fontenc on purpose, so OT1 sets `>` as `¿`.
        assert "angle-bracket-in-text" in {c for _, c, _ in scan("D > 0 と仮定する")}

    def test_catches_over_escaped_replacement(self):
        # scenario_fourier_convolution: a sequential escape table re-escaped the
        # braces that `\^{}` had just inserted.
        src = r"z transform (1/2)\^\{\}n by definition"
        assert "over-escaped" in {c for _, c, _ in scan(src)}

    def test_catches_bare_greek_word(self):
        # An assume card (`omega > 0`) and the Fourier definition cards
        # (`(omega)`, `e^{-i omega t}`) set the italic WORD, not ω.
        src = r"$\displaystyle omega > 0$"
        assert "bare-greek-word" in {c for _, c, _ in scan(src)}

    def test_catches_cas_product_operator(self):
        assert "bare-star-in-math" in {
            c for _, c, _ in scan(r"\begin{equation}  a*b + c  \end{equation}")}


class TestTheGuardDoesNotCryWolf:
    """A guard that runs over every committed example cannot afford noise.
    The first `bare-star` rule reported 23 legitimate optimal-value stars and
    nothing else."""

    @pytest.mark.parametrize("src", [
        r"\begin{equation}  m^*=\hbar^2/(d^2\xi_k/dk^2)  \end{equation}",
        r"\begin{equation}  z^*=c_1x^*+c_2y^*  \end{equation}",
        r"\begin{equation}  \int_0^{\infty} x_{i}^{2}\, dt  \end{equation}",
        r"\begin{equation}  \varphi^{\ast}_{m,s}(x)  \end{equation}",
        r"\begin{equation}  \omega_{0}^2-4 \gamma^2  \end{equation}",
        "50\\% の確率で",                      # escaped percent in prose
        r"a \textless{} b",                    # already-escaped angle
    ])
    def test_legitimate_latex_is_clean(self, src):
        assert not scan(src), scan(src)
