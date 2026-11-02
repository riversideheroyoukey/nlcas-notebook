"""Tests for I-1: imaginary-unit ordering in a coefficient.

QuantumAlgebra/Symbolics emit coefficient factors in internal order, leaving the
imaginary unit at the end (e.g. ``-2 B \\gamma i S_{y_1}``).  _reorder_imaginary_unit
moves it right after the numeric coefficient, and must never disturb structural
LaTeX (fractions, groups) or unrelated numbers (subscripts).
"""
from notebook.runtime.julia_latex_mixin import _reorder_imaginary_unit as reorder


class TestReorderImaginaryUnit:
    def test_larmor_case(self):
        assert reorder(r"-2 B \gamma i S_{y_{1}}") == r"-2 i B \gamma S_{y_{1}}"

    def test_already_first_unchanged(self):
        s = r"-2 i B \gamma S_{y_{1}}"
        assert reorder(s) == s

    def test_no_imaginary_unchanged(self):
        s = r"-2 B \gamma S_{y_{1}}"
        assert reorder(s) == s

    def test_im_form(self):
        assert reorder(r"3 B im T") == r"3 im B T"

    def test_greek_only(self):
        assert reorder(r"5 \alpha \beta i") == r"5 i \alpha \beta"

    def test_double_imaginary_left_alone(self):
        # i * i = -1; reordering would be meaningless, so leave it untouched
        s = r"-2 i i B"
        assert reorder(s) == s

    # --- safety: must not corrupt structural LaTeX or unrelated numbers ---

    def test_fraction_not_broken(self):
        # \frac is not a factor, so the run stops before it; the fraction is intact
        out = reorder(r"-2 B i \frac{x}{y}")
        assert r"\frac{x}{y}" in out
        assert out == r"-2 i B \frac{x}{y}"

    def test_reorder_inside_fraction_numerator(self):
        assert reorder(r"\frac{2 B i}{\gamma}") == r"\frac{2 i B}{\gamma}"

    def test_subscript_number_not_triggered(self):
        # the 2 in x_2 must not be treated as a coefficient
        assert reorder(r"x_2 + 3 B i C") == r"x_2 + 3 i B C"

    def test_plain_real_polynomial_unchanged(self):
        s = r"2 B \gamma + 3 \alpha"
        assert reorder(s) == s
