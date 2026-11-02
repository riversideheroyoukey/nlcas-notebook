r"""OPERAND-BOUNDARY ③（値集合の綴り, 2026-09-12）。

**同じ指示が、粒子によって違う形で返っていた。** `\sum_\sigma` は spin 1/2 なら
矢印に展開されるが、それ以外は展開されず QA の形式和のまま渡っていた——そして
QA は項をまたぐダミーを正準化しないので、実測（第2量子化セッション）::

    電子   [Σ_σ c†_{1σ}c_{2σ}, Σ_σ c†_{1σ}c_{1σ}]
             →  -c†_{1↑}c_{2↑} - c†_{1↓}c_{2↓}        2項に畳まれる
    光子   [Σ_σ a†_{1σ}a_{2σ}, Σ_σ a†_{1σ}a_{1σ}]
             →  形式和のまま + `\sum_{s,s_1}` の4演算子項     `completed`

項は打ち消し合うので**誤答ではなく未簡約**だが、研究者から見れば同じ指示が
理由なく別の形で返る。光子はシナリオに8箇所出る。

**綴りは物理の慣用を種別ごとに置く**（研究者の選択、2026-09-12）——整数スピンは
`⁺ ⁰ ⁻`（$\pi^+\pi^0\pi^-$ と同じ）。QA 側は**1文字なら何でも置けて区別も付く**
が（`⇑ ⇓ ⁺ ⁰ ⁻ ▲ ▼ α β` を実測）、**数字は駄目**: `c(1, QuIndex("0"))` は
`c(10)` と印字され、サイト 10 と同じ文字列になる。

**慣用の無い集合は展開しない**——3/2・5/2・グラビトン（$m=\pm2$）はそのまま。
$2S=3$ の粒子は粒子表にも核種表にも無いので、`⇑↑↓⇓` は足さない（通らない綴りは
文法に入れない）。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.spin_value_set import (PROJECTION_LABEL, PROJECTION_LATEX,
                                             index_labels)
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)


class TestWhichSetsHaveASpelling:

    @pytest.mark.parametrize("numerators,want", [
        ([-1, 1], ["↓", "↑"]),                 # spin 1/2
        ([-2, 2], ["⁻", "⁺"]),                 # ヘリシティ ±1（光子・グルーオン）
        ([-2, 0, 2], ["⁻", "⁰", "⁺"]),         # spin 1（マグノン・W/Z・²H・¹⁴N）
    ])
    def test_a_convention_exists(self, numerators, want):
        assert index_labels(numerators) == want

    @pytest.mark.parametrize("numerators", [
        [-3, -1, 1, 3],        # spin 3/2 —— 粒子表に $2S=3$ が無いので足していない
        [-5, -3, -1, 1, 3, 5],  # spin 5/2（²⁷Al・¹⁷O）
        [-4, 4],                # グラビトン（$m=\pm2$）
    ])
    def test_no_convention_means_no_expansion(self, numerators):
        assert index_labels(numerators) is None

    def test_a_single_value_is_not_expanded(self):
        r"""スピン 0 の値集合は `[0]`。1項に畳むと `a(1)` と `a(1⁰)` という
        **別の QA 対象**が同じセッションに並ぶし、畳む利得もそもそも無い."""
        assert index_labels([0]) is None

    def test_no_label_collides_with_a_site_number(self):
        r"""**数字は使えない**——`c(1, QuIndex("0"))` は `c(10)` と印字され、
        サイト 10 と同じ文字列になる（実測）。

        見るのは `isdecimal()` であって `isdigit()` ではない: **`⁰` は
        `isdigit()` では数字**（そしてそれがまさに生成器を壊した罠）だが、
        QA の印字では `c(1⁰)` と区別が付くので、綴りとしては安全である。
        """
        for label in PROJECTION_LABEL.values():
            assert len(label) == 1, label
            assert not label.isdecimal(), label

    def test_every_label_has_a_display_spelling(self):
        r"""生の unicode 上付きは `$…$` の中で組版できない——表示側の綴りは
        必ず在る（`PROJECTION_LATEX`）."""
        for label in PROJECTION_LABEL.values():
            assert label in PROJECTION_LATEX, label
            assert PROJECTION_LATEX[label], label


class TestTheGeneratorDoesNotReadASuperscriptAsAnInteger:
    r"""`str.isdigit()` は**上付き数字にも True** を返す。$m=0$ を `⁰` と綴った
    瞬間、生成器が「整数サイト」と読んで `m'(1, ⁰)` という未定義識別子を CAS へ
    渡し、生の `ParseError:` が返っていた（実測 2026-09-12）——`⁺`/`⁻` は通るので
    **3値の集合だけが落ちる**という形だった."""

    @pytest.mark.parametrize("text,want", [
        ("0", True), ("12", True), ("⁰", False), ("⁺", False), ("↑", False),
    ])
    def test_only_ascii_digits_are_integer_sites(self, text, want):
        from notebook.runtime.julia_code_generator import _IS_INTEGER_SITE
        assert _IS_INTEGER_SITE(text) is want

    def test_python_isdigit_would_have_said_yes(self):
        """罠そのものを固定する——これが False になったら註が古い."""
        assert "⁰".isdigit() is True


# ---------------------------------------------------------------------------
# 生きた経路（Julia/QuantumAlgebra）
# ---------------------------------------------------------------------------

def _commutator_of_two_sigma_sums(decl, op):
    async def go():
        nb.start_engine(title="proj", mode="second_quantization", language="ja",
                        domains=["condensed_matter"])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=["condensed_matter"],
                                primary_backend=engine.session.primary_backend)
            lines = [decl,
                     rf"H = \sum_\sigma {op}^\dagger_{{1\sigma}} {op}_{{2\sigma}}",
                     rf"N = \sum_\sigma {op}^\dagger_{{1\sigma}} {op}_{{1\sigma}}",
                     "E1 と E2 の交換子を求めて"]
            for line in lines:
                r = await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    engine, sup)
            out = engine.expressions.get(r.get("expr_id"))
            return r, (out.latex if out else "")
        finally:
            nb.stop_engine()
    return asyncio.run(go())


class TestTheSumIsExpandedForEveryValueSetThatHasASpelling:

    @pytest.mark.parametrize("decl,op,terms,mark", [
        ("c を電子の消滅演算子とする", "c", 2, r"\uparrow"),      # 1/2（変わらない）
        ("a を光子の消滅演算子とする", "a", 2, "{,+}"),           # ヘリシティ ±1
        ("m をマグノンの消滅演算子とする", "m", 3, "{,0}"),       # spin 1（3値）
    ])
    def test_it_reduces_to_one_term_per_projection(self, decl, op, terms, mark):
        r"""**印は 2026-09-12 に区切りつきへ変わった**（`{+}` → `{,+}`）。
        規則は同じで、変わったのは**印刷の綴り**である——区切りが無いと m=0 が
        `b_{10}` と印刷されて**サイト10と見分けが付かず**、打ち返すと生の
        `ParseError:` が返っていた（`tests/test_projection_round_trip.py`）。
        矢印はコマンドが境界なので区切りを持たない。"""
        r = _c = None
        r, latex = _commutator_of_two_sigma_sums(decl, op)
        assert r.get("status") == "completed", r
        # 形式和が残っていたら、それは畳まれていないということ
        assert r"\sum" not in latex, latex
        assert mark in latex, latex
        assert latex.count("^{\\dagger}") == terms, (terms, latex)

    def test_a_set_without_a_spelling_stays_a_formal_sum(self):
        r"""グラビトン（$m=\pm2$）に1文字の綴りは無い——展開せず形式和のまま。
        **断りではない**: 書けて、表示でき、QA が扱える形で残る."""
        r, latex = _commutator_of_two_sigma_sums(
            "g をグラビトンの消滅演算子とする", "g")
        assert r.get("status") == "completed", r
        assert r"\sum" in latex, latex


class TestWhatIsPublishedIsValidLaTeX:
    r"""**Latexify は上付きグリフを `\^<char>` と書く**（実測、`⁺` → `{\^+}`）。
    `\^+` はサーカムフレックスのアクセントで、`$…$` の中では組版できない——
    公開 .tex が通らなくなる（CLAUDE.md「公開 .tex は COMPILE する」）."""

    def test_no_circumflex_escape_survives(self):
        _r, latex = _commutator_of_two_sigma_sums(
            "a を光子の消滅演算子とする", "a")
        assert "\\^" not in latex, latex
        assert "⁺" not in latex and "⁻" not in latex, latex   # 生グリフも出さない
