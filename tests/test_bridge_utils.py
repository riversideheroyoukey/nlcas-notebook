"""Unit tests for bridge utility functions that don't need a running engine."""
from ui.interpret_pipeline import _fullwidth_to_halfwidth, _normalize_tokens, InputToken


class TestFullwidthToHalfwidth:
    def test_digits(self):
        assert _fullwidth_to_halfwidth("１２３") == "123"

    def test_uppercase(self):
        assert _fullwidth_to_halfwidth("ＡＢＣ") == "ABC"

    def test_lowercase(self):
        assert _fullwidth_to_halfwidth("ａｂｃ") == "abc"

    def test_operators(self):
        assert _fullwidth_to_halfwidth("＋－＊／＝") == "+-*/="

    def test_parens(self):
        assert _fullwidth_to_halfwidth("（ｘ＋１）") == "(x+1)"

    def test_ideographic_space(self):
        assert _fullwidth_to_halfwidth("ｘ　＋　１") == "x + 1"

    def test_japanese_unchanged(self):
        assert _fullwidth_to_halfwidth("代入する") == "代入する"
        assert _fullwidth_to_halfwidth("に対して") == "に対して"

    def test_mixed(self):
        result = _fullwidth_to_halfwidth("Ｅ１ に x＝１ を代入する")
        assert result == "E1 に x=1 を代入する"

    def test_halfwidth_unchanged(self):
        assert _fullwidth_to_halfwidth("E1 + x") == "E1 + x"

    def test_unicode_minus_sign(self):
        # U+2212 MINUS SIGN is always mathematical → convert unconditionally
        assert _fullwidth_to_halfwidth("x−1") == "x-1"

    def test_unicode_minus_sign_in_formula(self):
        # Full-width parens + full-width exponent + unicode minus
        assert _fullwidth_to_halfwidth("（ｘ＾２−１）") == "(x^2-1)"

    def test_katakana_prolonged_mark_as_minus(self):
        # ー between digits/math chars → hyphen-minus
        assert _fullwidth_to_halfwidth("xー1") == "x-1"

    def test_en_dash_as_minus(self):
        # U+2013 EN DASH between math chars → hyphen-minus
        assert _fullwidth_to_halfwidth("x–1") == "x-1"

    def test_em_dash_as_minus(self):
        # U+2014 EM DASH between math chars → hyphen-minus
        assert _fullwidth_to_halfwidth("x—1") == "x-1"

    def test_katakana_prolonged_mark_in_japanese_unchanged(self):
        # ー inside Japanese word context (no leading math char) → unchanged
        assert _fullwidth_to_halfwidth("テーブル") == "テーブル"

    def test_full_formula_with_mixed_unicode(self):
        # The bug case: （x＾2−1）／(xー１）→ (x^2-1)/(x-1)
        result = _fullwidth_to_halfwidth("（ｘ＾２−１）／（ｘー１）")
        assert result == "(x^2-1)/(x-1)"


class TestNormalizeTokens:
    def test_text_token_normalized(self):
        tokens = [InputToken(type="text", value="ｘに１を代入する")]
        result = _normalize_tokens(tokens)
        assert result[0].value == "xに1を代入する"

    def test_math_token_latex_normalized(self):
        tokens = [InputToken(type="math", latex="ｘ＾２", canonical="x^2")]
        result = _normalize_tokens(tokens)
        assert result[0].latex == "x^2"

    def test_math_token_canonical_normalized(self):
        tokens = [InputToken(type="math", latex="x^2", canonical="ｘ＾２")]
        result = _normalize_tokens(tokens)
        assert result[0].canonical == "x^2"

    def test_japanese_in_text_token_unchanged(self):
        tokens = [InputToken(type="text", value="展開して")]
        result = _normalize_tokens(tokens)
        assert result[0].value == "展開して"

    def test_multiple_tokens(self):
        tokens = [
            InputToken(type="math", latex="ｘ＋１", canonical="x+1"),
            InputToken(type="text", value="を展開して"),
        ]
        result = _normalize_tokens(tokens)
        assert result[0].latex == "x+1"
        assert result[1].value == "を展開して"
