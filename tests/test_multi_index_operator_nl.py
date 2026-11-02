"""多添字演算子（サイト＋スピン）が自然言語・括弧記法の両方で計算に届くこと。

実測 2026-08-22。QuantumAlgebra 自身は `{c(1,u), c'(1,u)}` を 1 と正しく返す——
壊れていたのは**指示を読む側**だけだった:

    c(1,u) と c'(1,u) の反交換子   → anticommutator(a="E1", b="c'(1")  → 生の ParseError
    [c(1,u), c'(1,u)]              → commutator(a="c(1", b="u), c'(1,u)")

前者は括弧を見ないトークン分割、後者は第1オペランドから `,` を除いた文字クラス。
どちらも「どのカンマが区切りかは括弧の深さで決まる」という一つの規則を、近似で
二箇所に書いていたもの。サイトとスピンを持つ演算子は凝縮系でいちばん基本的な
綴りで、それが計算に届かないと演算子代数そのものが使えない。
"""

import pytest

from notebook.runtime.llm_supervisor import _bracket_operands, _tokenize


class TestTokenizeKeepsBracketedCommas:
    """`_tokenize` は括弧の中では切らない。"""

    @pytest.mark.parametrize(
        "text, expected",
        [
            # 日本語
            ("c(1,u) と c'(1,u) の反交換子", ["c(1,u)", "と", "c'(1,u)", "の反交換子"]),
            ("c(1,u) の交換子", ["c(1,u)", "の交換子"]),
            # 英語
            ("anticommutator of c(1,u) and c'(1,u)",
             ["anticommutator", "of", "c(1,u)", "and", "c'(1,u)"]),
            # 添字の中のカンマ（LaTeX 側の綴り）
            (r"V_{p,q,s,n} を展開して", [r"V_{p,q,s,n}", "を展開して"]),
        ],
    )
    def test_bracketed_comma_is_not_a_separator(self, text, expected):
        got = [t for t in _tokenize(text) if t]
        for want in expected:
            if want.startswith(("c", "V", "a")):
                assert want in got, f"{want!r} が {got} に無い"

    def test_top_level_comma_still_separates(self):
        assert [t for t in _tokenize("a, b, c") if t] == ["a", "b", "c"]

    def test_unbalanced_close_does_not_swallow_the_rest(self):
        """閉じ過ぎでも負にはしない——書き損じで全体が1トークンに潰れるより良い。"""
        assert [t for t in _tokenize("a) b, c") if t] == ["a)", "b", "c"]


class TestBracketOperandsAreSplitByDepth:
    """`[X, Y]` / `{X, Y}` の区切りは深さ0のカンマ。"""

    @pytest.mark.parametrize(
        "text, opener, a, b",
        [
            ("[c(1,u), c'(1,u)]", "[", "c(1,u)", "c'(1,u)"),
            ("{c(1,u), c'(2,d)}", "{", "c(1,u)", "c'(2,d)"),
            ("[a(1), a'(1)]", "[", "a(1)", "a'(1)"),        # 従来通り
            ("[q, p]", "[", "q", "p"),                       # 従来通り
            ("[A, [B, C]]", "[", "A", "[B, C]"),             # 入れ子（従来は不一致）
        ],
    )
    def test_split(self, text, opener, a, b):
        got = _bracket_operands(text, opener)
        assert got is not None, f"{text!r} が一致しない"
        assert (got[0].strip(), got[1].strip()) == (a, b)

    def test_matrix_literal_still_matches_for_the_semicolon_guard(self):
        """行列リテラルは一致したうえで `;` で除外される（既存の防御を壊さない）。"""
        got = _bracket_operands("[2, 1; 1, 2]", "[")
        assert got is not None and ";" in got[2]

    @pytest.mark.parametrize(
        "text, opener",
        [("[A, B", "["), ("x + [A,B]", "["), ("{x}", "{"), ("[]", "["), ("", "[")],
    )
    def test_refuses_what_is_not_a_two_operand_bracket(self, text, opener):
        assert _bracket_operands(text, opener) is None


class TestSupervisorRoutesMultiIndexOperands:
    """スーパーバイザまで通したときに、演算子が切られずに届くこと。"""

    @staticmethod
    def _supervisor():
        from notebook.runtime.llm_supervisor import RuleBasedIntentParser
        from notebook.runtime.operation_validator import OperationValidator
        return RuleBasedIntentParser(OperationValidator(), "julia_full")

    @pytest.mark.parametrize(
        "text, op, a, b",
        [
            ("[c(1,u), c'(1,u)]", "commutator", "c(1,u)", "c'(1,u)"),
            ("{c(1,u), c'(1,u)}", "anticommutator", "c(1,u)", "c'(1,u)"),
        ],
    )
    def test_bracket_notation(self, text, op, a, b):
        r = self._supervisor()._try_bracket_parse(text)
        assert r is not None and r.operation == op
        assert r.args == {"a": a, "b": b}
