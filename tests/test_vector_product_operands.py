r"""VEC-BINARY-OPERAND-WHITESPACE (2026-09-17) — 内積・外積は、30綴り中4つしか正しくなかった。

起票は「オペランドが空白で切られ `\nabla` が量になる」1件だった。**族を並べて測ると
（内積・外積 × 日英 × オペランドの形5つ）30セル中 OK は4**で、根は3つあった:

| 数 | 症状 | 根 |
|---|---|---|
| 6 | 英語の内積が**「A の微分表示をドット記法に変更」を `confirmed`**——設定を黙って書き換える | `notation_dot_kw[en]` が `notation` を任意にし語境界も無い（日本語は `記法` 必須）。同じ割れで `take d/dt of E1` も Leibniz への切替になっていた |
| 16 | `\vec{A}`・列ベクトル・`(1, 2, 3)` が生の `Maxima evaluation error` | 成分の組み立てが、名前でない文字列にも `_x` を付けて**成分名を発明**していた。ノルムは同じ問いに「降ろしてから読む・1列の行列はベクトル・丸括弧の組は理由付きで断る」と既に答えていた |
| 4 | `\nabla \times \vec{A} と \vec{B} の内積` が `nabla_x*times_x+…` を `completed` | 内積・外積は交換子と同じ文の形・同じ `a`/`b` 欄なのに、助詞で切る読み手の表に無く、**空白1つで1オペランド**に切る経路へ落ちていた（交換子が 2026-08-22 に踏んだ欠陥そのもの） |

**入れ子は断るのではなく計算する。** 単項の ∇ を読む読み手（`_try_single_vec_op`）も、
`curl(X)` を計算する成分の組み立ても既に在り、欠けていたのはオペランドを渡す継ぎ目だけ
だった。そして継ぎ目を作ると **COORD-NAME-BOUND が新しい出口から戻る**——`nb_dot` の
中に `nb_curl_*` が入るので、`y = x^3` のあとで生の CAS エラーになる。内積自体は座標を
束縛しないので、包むのは **∇ から来たオペランドだけ**（命令全体を包むと、もう一方の具体
成分 `[x, y, z]` の値が変わる。実測）。
"""
import asyncio
import json
import re
import string

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.i18n import t
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import (InputToken, InterpretRequest,
                                   run_interpret_pipeline)


def _ask(lang, setup, text, domain="electromagnetism"):
    async def go():
        nb.start_engine(title="vector-products", mode="standard", language=lang,
                        domains=[domain])
        try:
            engine = nb.get_engine()
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[domain],
                                primary_backend=engine.session.primary_backend)
            for line in setup:
                await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    engine, sup)
            return await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                engine, sup)
        finally:
            nb.stop_engine()
    return asyncio.run(go())


_FIELDS = {"ja": ["A をベクトル場とする", "B をベクトル場とする"],
           "en": ["define A as a vector field", "define B as a vector field"]}
_PHRASE = {("ja", "dot"): "{x} と {y} の内積を求めて",
           ("ja", "cross"): "{x} と {y} の外積を求めて",
           ("en", "dot"): "the dot product of {x} and {y}",
           ("en", "cross"): "the cross product of {x} and {y}"}
_DOT_AB = "A_z*B_z+A_y*B_y+A_x*B_x"
_CROSS_AB = "[A_y*B_z-A_z*B_y,A_z*B_x-A_x*B_z,A_x*B_y-A_y*B_x]"


def _canon(r):
    return (r.get("canonical") or "").replace(" ", "").replace("\n", "")


class TestTheEnglishDotProductIsNotANotationSwitch:
    """設定を黙って書き換えないこと——この族でいちばん害が大きかった。"""

    @pytest.mark.parametrize("text", ["the dot product of A and B",
                                      "compute the dot product of A and B"])
    def test_it_computes_the_dot_product(self, text):
        r = _ask("en", _FIELDS["en"], text)
        assert r.get("operation") != "set_notation", r
        assert r.get("status") == "completed", r
        assert _canon(r) == _DOT_AB, r

    def test_d_dt_without_the_word_notation_is_not_a_notation_switch(self):
        r = _ask("en", [r"L = (1/2) m \dot{q}^2"], "take d/dt of E1",
                 domain="classical_mechanics")
        assert r.get("operation") != "set_notation", r

    @pytest.mark.parametrize("lang,setup,text,label_key", [
        ("en", ["q is time-dependent"], "switch q to dot notation", "interpreter.notation_dot"),
        ("en", ["q is time-dependent"], "use Leibniz notation for q", "interpreter.notation_leibniz"),
        ("en", ["q is time-dependent"], "use d/dt notation for q", "interpreter.notation_leibniz"),
        ("ja", ["q を時間依存とする"], "q をドット記法にして", "interpreter.notation_dot"),
        ("ja", ["q を時間依存とする"], "q をライプニッツ記法にして", "interpreter.notation_leibniz"),
    ])
    def test_a_sentence_that_names_the_notation_still_switches_it(
            self, lang, setup, text, label_key):
        r = _ask(lang, setup, text, domain="classical_mechanics")
        assert r.get("operation") == "set_notation", r
        assert t(label_key, lang) in (r.get("message") or ""), r

    @pytest.mark.parametrize("text", [
        "the dot product of A and B", r"with respect to \dot{x}", "dotted"])
    def test_the_english_dot_key_needs_the_word_notation(self, text):
        """日本語が `記法` を必須にしているのと同じ。`\\dot{x}` は語境界の無い旧パターンに当たっていた。"""
        assert not nlp("notation_dot_kw", "en").search(text), text


class TestComponentsAreReadNotInvented:
    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("kind,want", [("dot", _DOT_AB), ("cross", _CROSS_AB)])
    def test_the_textbook_vector_mark(self, lang, kind, want):
        r = _ask(lang, _FIELDS[lang], _PHRASE[(lang, kind)].format(x=r"\vec{A}", y=r"\vec{B}"))
        assert r.get("status") == "completed", r
        assert _canon(r) == want, r

    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("setup", [
        ["u = [1, 2, 3]", "v = [4, 5, 6]"],
        [r"u = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}",
         r"v = \begin{pmatrix} 4 \\ 5 \\ 6 \end{pmatrix}"],
    ], ids=["list", "column-vector"])
    @pytest.mark.parametrize("kind,want", [("dot", "32"), ("cross", "[-3,6,-3]")])
    def test_a_concrete_vector_by_name(self, lang, setup, kind, want):
        """列ベクトルは TeX で来ると `matrix([1],[2],[3])` になる——ノルムが既に読んでいた形。"""
        r = _ask(lang, setup, _PHRASE[(lang, kind)].format(x="u", y="v"))
        assert r.get("status") == "completed", r
        assert _canon(r) == want, r

    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("kind", ["dot", "cross"])
    def test_a_parenthesised_tuple_is_refused_the_way_the_norm_refuses_it(self, lang, kind):
        r = _ask(lang, ["u = (1, 2, 3)", "v = (4, 5, 6)"],
                 _PHRASE[(lang, kind)].format(x="u", y="v"))
        assert r.get("status") == "failed", r
        err = r.get("error") or ""
        assert "Maxima" not in err, err
        assert err == t("vector.operand_not_a_vector", lang, operand="(1, 2, 3)"), err


class TestANablaOperandIsComputed:
    _SETUP = {"ja": ["A をベクトル場とする", "A_x = 0", "A_y = 0", "A_z = x*y", "b = [1, 1, 1]"],
              "en": ["define A as a vector field", "A_x = 0", "A_y = 0", "A_z = x*y",
                     "b = [1, 1, 1]"]}

    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("binding", [[], ["y = x^3"]], ids=["unbound", "y-bound"])
    def test_curl_dotted_with_a_vector(self, lang, binding):
        """(∇×A)·(1,1,1) = x − y。束縛があっても同じ——∇ 側だけを守っているから。"""
        r = _ask(lang, self._SETUP[lang] + binding,
                 _PHRASE[(lang, "dot")].format(x=r"\nabla \times \vec{A}", y="b"))
        assert r.get("status") == "completed", r
        assert _canon(r) == "x-y", r

    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("kind", ["dot", "cross"])
    def test_a_scalar_nabla_operand_is_refused_in_the_session_language(self, lang, kind):
        """∇·A はスカラー。以前は日英を同時に出す手書きの文面で、しかも「curl/divergence の
        引数」という別の文脈の話をしていた。"""
        r = _ask(lang, _FIELDS[lang],
                 _PHRASE[(lang, kind)].format(x=r"\nabla \cdot \vec{A}", y=r"\vec{B}"))
        assert r.get("status") == "failed", r
        assert r.get("error") == t("vector.operand_is_scalar", lang,
                                   operand=r"\nabla \cdot A"), r


class TestTheVectorMarkReadsTheDeclaration:
    r"""`\vec{A}` は `A` と同じ量——宣言の成分を読むこと。

    `vec(...)` の包みを外したあと、中身を**読み直さずに**デカルトの接尾辞を付けて
    いたので、カスタム座標系では存在しない成分 `A_x, A_y, A_z` を発明し、素の `A`
    （宣言どおり `A_xi, A_eta, A_zeta`）と答えが割れていた。宣言済みのデカルト場では
    両者が一致するので、族の測定（48セル）ではこの差は見えず、変異試験で見つかった。
    """

    def test_a_custom_coordinate_field_keeps_its_declared_components(self):
        from notebook.models.quantity import Quantity
        from notebook.models.symbol_context import SymbolContext
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        from notebook.runtime.maxima_command_builder import MaximaCommandBuilder

        class _Adapter:
            def execute(self, command):
                return None

        class _Session:
            coordinate_system = "custom"
            custom_coord_vars = ("xi", "eta", "zeta")
            custom_coord_lame = ("1", "1", "1")

        h = MaximaBackendHandler(MaximaCommandBuilder())
        h.set_symbol_context(SymbolContext())
        h.init_coordinate_system(_Adapter(), _Session())
        h._context.register("A", quantity=Quantity(
            name="A", kind="vector_field", coordinate_system="custom",
            components=["A_xi", "A_eta", "A_zeta"]))
        want = ("A_xi", "A_eta", "A_zeta")
        assert h._build_vec_components("A", "cartesian", {}) == want
        assert h._build_vec_components("vec(A)", "cartesian", {}) == want


class TestTheShadowIsPerOperand:
    """内積自体は座標を束縛しない——命令全体を包むと値が変わる（実測）。"""

    def test_only_the_differentiated_operand_is_wrapped(self):
        from notebook.runtime.maxima_subst_guard import shadow_differentiated_operand
        xyz = ("x", "y", "z")
        curl = "[nb_curl_x(A_x,A_y,A_z),nb_curl_y(A_x,A_y,A_z),nb_curl_z(A_x,A_y,A_z)]"
        assert shadow_differentiated_operand(xyz, curl) == f"block([x, y, z], {curl})"
        assert shadow_differentiated_operand(xyz, "[x,y,z]") == "[x,y,z]"
        assert shadow_differentiated_operand(xyz, "[nb_dot([1],[2]),a,b]") == "[nb_dot([1],[2]),a,b]"


_FIELD = re.compile(r"\{([^{}]*)\}")
_COMMAND_BEFORE = re.compile(r"\\[A-Za-z]+$")


def _latex_group_beside_a_placeholder(template: str):
    r"""A `\command{…}` argument written with SINGLE braces, in a template that also
    has a real placeholder — `.format` then reads the argument as a field.

    Only a group glued to a backslash command counts.  Measured on the table: a
    wider test ("preceded by `^`, `_` or `}`") flagged six entries and all six were
    INTENDED placeholders — `\sum_{sub}`, `{argmin}{boundary}`, `\times {name}` —
    so it could not tell the defect from ordinary use.
    """
    s = template.replace("{{", "\x00\x00").replace("}}", "\x01\x01")
    fields = [(m.group(1), s[:m.start()]) for m in _FIELD.finditer(s)]
    real = {n for n, before in fields
            if not _COMMAND_BEFORE.search(before)
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n)}
    latex = [n for n, before in fields if _COMMAND_BEFORE.search(before)]
    return f"LaTeX {latex} beside {sorted(real)}" if (real and latex) else None


class TestAMessageWithAPlaceholderActuallyFormats:
    r"""`t()` は `KeyError` のとき**黙って未置換の文面を返す**。置換欄と LaTeX の素の
    波括弧（`\begin{pmatrix}`）が同居すると、欄が埋まらないまま研究者に届く——この
    変更で書いた断り文が最初そうなっていた。

    **測定器は2回盲目だった。** 最初の版は「識別子の欄を全部渡して `.format` が通るか」
    で、`pmatrix` も識別子なので未修正の文面を通していた（較正テストが捕まえた）。
    """

    def test_every_message_with_a_placeholder_formats(self):
        d = json.load(open("i18n/messages.json", encoding="utf-8"))
        bad = [(k, lang, err)
               for k, e in d.items() if isinstance(e, dict)
               for lang, v in e.items()
               if not lang.startswith("_") and isinstance(v, str)
               for err in [_latex_group_beside_a_placeholder(v)] if err]
        assert bad == [], bad

    @pytest.mark.parametrize("template", [
        r"`{operand}` か `\begin{pmatrix} 1 \end{pmatrix}`",
        r"`{x}` は \frac{a}{b}",
        r"{name} と \mathrm{d}x",
    ])
    def test_it_reports_a_command_argument_beside_a_placeholder(self, template):
        assert _latex_group_beside_a_placeholder(template) is not None

    @pytest.mark.parametrize("template", [
        r"`\nabla \times {name}`",          # a placeholder after a command and a space
        r"\sum_{sub} と {sub}",              # a placeholder deliberately in an index slot
        "{argmin}{boundary}",               # two placeholders side by side
        "$^{{{mass}}}${element}",           # an escaped group around a placeholder
        r"`\begin{{pmatrix}}` {operand}",   # the escaped form
        r"\Phi_0 と \langle{X}\rangle",     # braces but no placeholder: never formatted
    ])
    def test_it_stays_silent_on_ordinary_use(self, template):
        assert _latex_group_beside_a_placeholder(template) is None
