"""Tests for Adviser-memo → LaTeX conversion in the TeX export (E-4).

Covers the root-cause fix for memos failing to compile: Unicode Greek / math
symbols in plain text are converted to inline-math LaTeX, math ($…$, $$…$$) is
preserved, bullet lines become itemize, and 【…】 becomes \textbf.
"""
from ui.notebook_bridge import (
    _tex_escape,
    _text_unicode_to_math,
    _note_inline,
    _note_to_latex,
)


class TestTexEscape:
    def test_specials_escaped(self):
        assert _tex_escape("a & b % c # d") == r"a \& b \% c \# d"

    def test_underscore_and_braces(self):
        assert _tex_escape("x_1 {y}") == r"x\_1 \{y\}"


class TestUnicodeToMath:
    def test_single_greek_wrapped_in_math(self):
        assert _text_unicode_to_math("ω") == r"$\omega$"

    def test_consecutive_symbols_share_one_math_group(self):
        # critical: must NOT produce adjacent $$ (display-math delimiter)
        out = _text_unicode_to_math("αβγ")
        assert out == r"$\alpha \beta \gamma$"
        assert "$$" not in out

    def test_math_symbols(self):
        assert _text_unicode_to_math("≈") == r"$\approx$"
        assert _text_unicode_to_math("√") == r"$\surd$"
        assert _text_unicode_to_math("∂") == r"$\partial$"

    def test_ascii_untouched(self):
        assert _text_unicode_to_math("energy E = m") == "energy E = m"


class TestNoteInline:
    def test_bracket_to_textbf(self):
        assert _note_inline("【結論】ok") == r"\textbf{結論}ok"

    def test_escape_then_unicode(self):
        # & escaped, ω → math, both in one segment
        assert _note_inline("ω & x") == r"$\omega$ \& x"


class TestNoteToLatex:
    def test_inline_math_preserved(self):
        out = _note_to_latex("エネルギーは $E = mc^2$ で表される")
        assert "$E = mc^2$" in out

    def test_unicode_in_plain_text_becomes_math(self):
        out = _note_to_latex("周波数 ω は √2 に比例 ∝")
        assert r"$\omega$" in out
        assert r"$\surd$" in out
        assert r"$\propto$" in out

    def test_bullets_become_itemize(self):
        out = _note_to_latex("* 第一項\n* 第二項")
        assert r"\begin{itemize}" in out
        assert r"\end{itemize}" in out
        assert out.count(r"\item") == 2

    def test_bullet_with_inline_math(self):
        out = _note_to_latex("* 第一項: $H_0$\n* 摂動 $V$")
        assert r"\item 第一項: $H_0$" in out
        assert r"\item 摂動 $V$" in out

    def test_dagger_unicode_converted(self):
        out = _note_to_latex("a† は生成演算子")
        assert r"$\dagger$" in out

    def test_display_math_preserved_and_no_trailing_break(self):
        out = _note_to_latex("式:\n$$E = mc^2$$\n以上")
        assert "$$E = mc^2$$" in out
        # a standalone display-math line must not get a trailing LaTeX \\
        assert "$$ \\\\" not in out and "$$\\\\" not in out

    def test_no_adjacent_dollar_dollar_from_consecutive_symbols(self):
        # α immediately followed by β must not create $$ (which would start
        # an unterminated display-math block and break compilation)
        out = _note_to_latex("係数 αβ について")
        assert "$$" not in out

    def test_bracket_header(self):
        out = _note_to_latex("【まとめ】系は安定")
        assert r"\textbf{まとめ}" in out

    def test_checkmark_converted_to_latex_symbol(self):
        # Several confirmation-message i18n strings (e.g.
        # interpreter.declared_time_dep) are prefixed with "✓ " for UI
        # display and reach the TeX narrative via record_quantity_update()'s
        # manual_note step. A raw "✓" (U+2713) is outside the default
        # upLaTeX font's glyph coverage and breaks compilation; it must be
        # converted to \checkmark (amssymb, already in both preambles).
        out = _note_to_latex("✓ Declared q as time-dependent (time variable: t)")
        assert r"$\checkmark$" in out
        assert "✓" not in out

    def test_checkmark_leading_line_still_compiles_shape(self):
        # J-8 (scenario-narrative-notes review, item 13): a leading ✓ at the
        # very start of a note line previously produced a raw Unicode glyph
        # as the first character of the TeX line.
        out = _note_to_latex("✓ 仮定しました: k > 0")
        assert out.startswith(r"$\checkmark$")


class TestBeautifyBareCanonicalLatex:
    """J-9 (scenario-narrative-notes review, item 11): a canonical-syntax RHS
    like "z * sigma * sqrt(L)" must render with real LaTeX backslashes
    (\\sigma, \\sqrt{...}), not the literal bare tokens — while never touching
    term order (Maxima's own tex1() renderer reorders terms even without
    simplifying, which corrupts input echoes / unevaluated H-1 RHS text).
    """
    def test_bare_greek_and_sqrt_converted(self):
        from notebook.runtime.backend_handler import beautify_bare_canonical_latex
        assert beautify_bare_canonical_latex("z * sigma * sqrt(L)") == r"z \sigma \sqrt{L}"

    def test_already_backslashed_greek_not_double_escaped(self):
        from notebook.runtime.backend_handler import beautify_bare_canonical_latex
        out = beautify_bare_canonical_latex(r"-J * Sz_1 * Sz_2 - \Gamma * (Sx_1 + Sx_2)")
        assert r"\\Gamma" not in out
        assert r"\Gamma (Sx_1 + Sx_2)" in out

    def test_nested_sqrt(self):
        from notebook.runtime.backend_handler import beautify_bare_canonical_latex
        assert beautify_bare_canonical_latex("sqrt(a + sqrt(b))") == r"\sqrt{a + \sqrt{b}}"

    def test_greek_with_alpha_subscript_converted(self):
        # 2026-08-09: an ALPHA subscript converts too, like the digit one below.
        # It used to be left bare on the theory that "index or part of the name"
        # was undecided — but that question does not reach the display: `\sigma_x`
        # renders the same either way, and left bare it published the italic WORD
        # "sigma" (measured: `\varphi_{m,s}`, `p_theta` in the example corpus).
        # What must NOT be touched is a Greek name inside a longer identifier,
        # pinned by the two tests either side of this one.
        from notebook.runtime.backend_handler import beautify_bare_canonical_latex
        assert beautify_bare_canonical_latex("sigma_x") == r"\sigma_{x}"
        assert beautify_bare_canonical_latex("sigmax") == "sigmax"
        assert beautify_bare_canonical_latex("Sx_1 + Sx_2") == "Sx_1 + Sx_2"

    def test_digit_subscripted_greek_converted(self):
        """Regression (2026-07-09, numeric-layer figure-caption work): a
        digit subscript is UNAMBIGUOUSLY part of the name (CLAUDE.md
        Subscript Classification — "always part of name"), unlike an alpha
        subscript. "omega_0" must render \\omega_{0}, not stay bare — the
        H-1 equation-form CAS card ("f = gamma * omega_0 * exp(-t)") showed
        gamma converted but omega_0 left literal because \\bomega\\b never
        matched inside "omega_0" (no word-boundary before a digit-only "_0",
        since "_" is a \\w character)."""
        from notebook.runtime.backend_handler import beautify_bare_canonical_latex
        assert (beautify_bare_canonical_latex("gamma * omega_0 * exp(-t)")
                == r"\gamma \omega_{0} exp(-t)")
        assert beautify_bare_canonical_latex("omega_0") == r"\omega_{0}"
        assert beautify_bare_canonical_latex("sqrt(gamma_1)") == r"\sqrt{\gamma_{1}}"


class TestYouCardLatex:
    """YOU-card rendering: bare formulas → inline math; natural language → text."""

    def test_nabla_formula_rendered_as_math(self):
        from ui.notebook_bridge import _you_card_latex
        assert _you_card_latex(r"\nabla \cdot E") == r"$\nabla \cdot E$"

    def test_bare_sigma_sqrt_beautified_in_you_card(self):
        from ui.notebook_bridge import _you_card_latex
        assert _you_card_latex("SS = z * sigma * sqrt(L)") == r"$SS = z \sigma \sqrt{L}$"
        assert _you_card_latex("E_x = 0") == "$E_x = 0$"
        # '*' multiplication becomes a space in math mode
        assert _you_card_latex("-k*sin(k*x - w*t) = -w*sin(k*x - w*t)/c") == \
            "$-k sin(k x - w t) = -w sin(k x - w t)/c$"

    def test_japanese_command_stays_text(self):
        from ui.notebook_bridge import _you_card_latex
        # CJK present and no math fragment → whole card stays escaped text, NOT math
        assert _you_card_latex("E をベクトル場として定義する") == \
            "E をベクトル場として定義する"
        # A script-bearing identifier (B_z) is rendered as an inline-math fragment
        # while the surrounding prose stays text — not whole-math, not escaped "B\_z".
        out = _you_card_latex("B_z を t で偏微分して")
        assert "$B_z$" in out and "を t で偏微分して" in out and r"\_" not in out
