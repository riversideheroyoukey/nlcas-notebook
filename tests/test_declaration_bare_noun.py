r"""英語だけ「裸の型名詞」が宣言の別名で、無関係な文が宣言として**実行**されていた。

実測 2026-09-07（四十三問目で `matrix` を直した際に兄弟も同型と分かり、起票
DECL-BARE-NOUN-EN として本問で根治）。

| 打った文 | 直す前（Julia セッション） |
|---|---|
| `the scalar curvature of E1` | `define_scalar_parameter` を **confirmed**（実行） |
| `the vector norm of E1` | `define_vector` |
| `the tensor rank of E1` | `define_tensor` |
| `the spin correlation of E1` | `define_spin_operator` |
| `the electron density of E1` | `define_fermion_operator` |
| `the boson occupation of E1` | `define_boson_operator` |
| `the phonon dispersion of E1` | 〃 |
| `the photon number of E1` | 〃 |
| `the fermion parity of E1` | `define_fermion_operator` |
| `the time dependent part of E1` | `declare_time_dependent` |

**`needs_confirmation` ですらない**——研究者が頼んでいない宣言が、成功として
実行されていた。名前は次のトークンから取られる。

**Maxima では無害に見えた。** 粒子演算子の宣言は `supported_backends` が
`julia_full` だけなので、Maxima セッションでは別名走査が候補ごと捨てる。
**綴りが安全だったのではなく、バックエンドの filter に救われていた**——
だから Maxima だけで測ると 11/19、Julia で測ると 9/9 が化ける。

**判別子は棚卸しから出た。** コーパスに実在する英語の宣言 49 綴りを数えると、
**すべて `as` を含む**:

    Define R, L, w, C as scalars      Declare F_0 as a scalar
    Define gamma, B as real scalars   declare G as a matrix
    A as a vector field               b as a boson operator
    S as the spin operator            define c as an electron annihilation operator

一方、宣言に化けた文には `as` が無い。これは日本語の `〜とする` / `〜として定義`
に対応する。日本語側はほぼすべて宣言句（`ベクトルとする` など）で、**裸の型名詞は
`パウリ` 1つだけ**——実測では発火しないが、コーパスにも0行なので「割れていない
だけで割れる形」として同じく外した。非対称は英語側にあった。

**`declare_time_dependent` は、この検査そのものが見つけた。** 最初に畳んだのは
scalar/vector/tensor/spin/粒子の6操作で、`time dependent` は数えていなかった
——「裸の型名詞を別名に持つ、必須欄が名前だけの操作」を**機械的に列挙する**検査を
書いた時点で落ち、測ると `the time dependent part of E1` が `declare_time_dependent`
を confirmed していた。四十問目と同じ形（同型を直したときこそ一覧に寄せる）。

**削る前に「その綴りが何で通っているか」を機械的に特定した。** 起票が警告した
とおり、思い込みで削ると壊れる——実測すると `S as the spin operator` は
**複合語ではなく裸の `spin`** で当たっており、`spin operator` / `boson operator` /
`fermion operator` は**別名として存在しなかった**（`define spin operator` という
3語形しか無い）。足さずに削れば、実在のシナリオ行が落ちていた。

`scan_aliases` は **longest-first・非重複**なので、`as a scalar`（3語）は
`scalar field`（2語）に勝つ。そのため場の側にも `as a scalar field`（4語）を
足してある——足さないと `A as a scalar field` が `define_scalar_parameter` に
化ける。

**わざと範囲に入れていないもの:** `define_function` の `set`（`set E1 to zero` が
`define_function` になる）。あれは型名詞ではなく**動詞**で、`set x to 3` のような
実在の用法があるので、削るかどうかは別に測る（open_items DECL-BARE-NOUN-EN 残）。
"""
import asyncio
import json
import pathlib

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

_SCHEMA = json.loads(
    (pathlib.Path(__file__).resolve().parent.parent / "schemas"
     / "common_operations.schema.json").read_text(encoding="utf-8"))

#: 名前だけを必須欄に持つ宣言操作——**この形が、次のトークンを名前にしてしまう**。
NAME_ONLY_DECLARATIONS = tuple(
    o["operation"] for o in _SCHEMA["operations"]
    if (o.get("operation") or "").startswith(("define", "declare"))
    and o.get("required_fields") == ["name"])

#: 裸で置いてはいけない型名詞。**日本語側には対応物が無い**（すべて宣言句）。
BARE_TYPE_NOUNS = ("scalar", "vector", "tensor", "matrix", "matrices",
                   "spin", "pauli", "boson", "bose", "bosonic",
                   "phonon", "photon", "fermion", "fermionic", "electron",
                   "time dependent", "time-dependent")


def _aliases(op):
    for o in _SCHEMA["operations"]:
        if o.get("operation") == op:
            return o.get("aliases") or {}
    raise AssertionError(op)


def _ask(lang, text, prep, mode="standard", backend="maxima"):
    nb.start_engine(title="dbn", mode=mode, language=lang, domains=[])
    calls: list = []
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend=backend)
        orig = eng.run_operation

        def spy(op, args):
            calls.append(op)
            return orig(op, args)
        eng.run_operation = spy

        def run(x):
            return asyncio.get_event_loop().run_until_complete(
                run_interpret_pipeline(InterpretRequest(
                    tokens=[InputToken(type="text", value=x)]), eng, sup))
        for p in prep:
            run(p)
        calls.clear()
        r = run(text)
        return calls, r.get("status"), r.get("operation")
    finally:
        nb.stop_engine()


PREP = ["[2, 1; 1, 2]", "f = x^2 + 1"]

#: 宣言ではない、ごく普通の研究者の文。**その語を含むだけ**で宣言になっていた。
NOT_DECLARATIONS = [
    "the scalar curvature of E1", "take the scalar part of E1",
    "the vector norm of E1", "normalize the vector E1",
    "the vector potential of E1",
    "the tensor rank of E1", "contract the tensor E1",
    "the spin correlation of E1", "flip the spin in E1",
    "the electron density of E1", "the boson occupation of E1",
    "the phonon dispersion of E1", "the photon number of E1",
    "the fermion parity of E1",
    "the time dependent part of E1", "plot the time dependent behaviour of E1",
]


class TestAnOrdinarySentenceIsNotADeclaration:
    @pytest.mark.parametrize("text", NOT_DECLARATIONS)
    def test_no_declaration_is_run(self, text):
        calls, st, op = _ask("en", text, PREP)
        ran = [c for c in calls if c in NAME_ONLY_DECLARATIONS]
        assert not ran, (text, st, calls)
        assert op not in NAME_ONLY_DECLARATIONS, (text, st, op)

    @pytest.mark.parametrize("text", NOT_DECLARATIONS)
    def test_no_declaration_is_even_offered(self, text):
        """直す前は `confirmed`——訊きもせず実行していた。"""
        _, st, _ = _ask("en", text, PREP)
        assert st != "confirmed", text


#: コーパスに実在する英語の宣言と、その日本語対。**壊してはいけない綴り。**
REAL_DECLARATIONS = [
    ("Declare F_0 as a scalar", "F_0 をスカラーとする", "define_scalar_parameter"),
    ("declare x_1 as a scalar", "x_1 をスカラーとする", "define_scalar_parameter"),
    ("Define A as a scalar", "A をスカラーパラメータとして定義して",
     "define_scalar_parameter"),
    ("define V as a scalar parameter", "V をスカラーパラメータとして定義して",
     "define_scalar_parameter"),
    ("declare G as a matrix", "G を行列とする", "define_matrix"),
    ("A as a vector field", "A をベクトル場として定義する", "define_vector_field"),
    ("define sigma as a Pauli matrix", "sigma をパウリ行列として定義する",
     "define_spin_operator"),
    ("phi as a time-dependent scalar field", "phi を時間依存のスカラー場として定義する",
     "define_scalar_field"),
]


class TestTheRealDeclarationsStillWork:
    """**削る前に「その綴りが何で通っているか」を機械的に特定した**——実測すると
    `S as the spin operator` は複合語ではなく裸の `spin` で当たっており、
    `spin operator` は別名として存在しなかった。"""

    @pytest.mark.parametrize("en,ja,want", REAL_DECLARATIONS)
    def test_english(self, en, ja, want):
        calls, st, _ = _ask("en", en, [])
        assert want in calls, (en, st, calls)

    @pytest.mark.parametrize("en,ja,want", REAL_DECLARATIONS)
    def test_japanese(self, en, ja, want):
        calls, st, _ = _ask("ja", ja, [])
        assert want in calls, (ja, st, calls)

    @pytest.mark.parametrize("text,want", [
        ("Define R, L, w, C as scalars", "define_scalar_parameter"),
        ("Define gamma, B as real scalars", "define_scalar_parameter"),
        ("Define u, v as scalars", "define_scalar_parameter"),
    ])
    def test_plural_and_adjectival_forms(self, text, want):
        calls, st, _ = _ask("en", text, [])
        assert want in calls, (text, st, calls)

    @pytest.mark.parametrize("text,want", [
        ("A as a scalar field", "define_scalar_field"),
        ("A as a vector field", "define_vector_field"),
    ])
    def test_the_longer_field_spelling_wins(self, text, want):
        """`scan_aliases` は longest-first なので、`as a scalar`（3語）を足すと
        `scalar field`（2語）に勝ってしまう。場の側に4語形を足して打ち消す。"""
        calls, st, _ = _ask("en", text, [])
        assert want in calls, (text, st, calls)

    @pytest.mark.parametrize("text,want", [
        ("A as a scalar field", "define_scalar_field"),
        ("A as a vector field", "define_vector_field"),
    ])
    def test_the_alias_layer_itself_answers_field(self, text, want):
        """**直接の消費者に当てる。** 上の実行経路は `decl_stmt` でも解決するので、
        4語形を消しても緑のまま通る（変異で実測）。効いているのは別名の層で、
        そこを消すと `A as a vector field` は **`define_vector`** に解決する
        ——`needs_expression_injection` ほか、別名表を読む消費者はそれを見る。"""
        from notebook.runtime.llm_supervisor import _tokenize
        from notebook.runtime.token_field_mapping import scan_aliases
        v = OperationValidator()
        toks = _tokenize(text)
        hits = [c for _, _, c in scan_aliases(toks, v.resolve_alias, v.max_alias_words)]
        assert hits == [want], (text, hits)


class TestTheAliasDataHasNoBareTypeNoun:
    """**数える検査そのものにする**（四十問目以来）。日本語側にはもともと
    裸の型名詞が無いので、これは日英を揃える検査でもある。"""

    @pytest.mark.parametrize("op", NAME_ONLY_DECLARATIONS)
    def test_english(self, op):
        bare = [a for a in (_aliases(op).get("en") or [])
                if a.lower() in BARE_TYPE_NOUNS]
        assert not bare, (
            f"{op}: 裸の型名詞 {bare} が英語の別名にある"
            "——`as a X` / `as Xs` の宣言句に置き換える")

    @pytest.mark.parametrize("op", NAME_ONLY_DECLARATIONS)
    def test_japanese_has_none_either(self, op):
        """日本語側は元から**ほぼ**宣言句だった。唯一の例外 `パウリ` は実測では
        発火しなかったが、コーパスにも0行なので同じく外してある。"""
        JA_BARE = ("スカラー", "ベクトル", "テンソル", "行列", "マトリックス",
                   "スピン", "パウリ", "ボソン", "フェルミオン", "電子", "時間依存")
        bare = [a for a in (_aliases(op).get("ja") or []) if a in JA_BARE]
        assert not bare, (op, bare)

    @pytest.mark.parametrize("op", NAME_ONLY_DECLARATIONS)
    def test_every_declaration_has_an_as_phrase(self, op):
        """英語の宣言は `as` を含む——それが日本語の `〜とする` に対応する。
        **この検査が `declare_time_dependent` を見つけた**（最初の6操作に
        入れていなかった）。"""
        en = _aliases(op).get("en") or []
        assert any(a.lower().startswith("as ") for a in en), (op, en)


class TestADeclarationDoesNotNeedAnExpressionInjected:
    """**変異が生き残って初めて見えた軸。** 「どの操作に届くか」だけを測ると、
    足した `as` 句を削っても全部緑になる——実在の宣言は `decl_stmt` などの
    nl_patterns 経路でも解決するから。効いているのは別の軸で、
    `needs_expression_injection`——**別名が当たらないと E-id が注入され、
    その注入テキストが公開実例の `user_text` に漏れる**（四十三問目で実測、
    英語のシナリオに `E19 に対して declare G as a matrix` と記録されていた）。

    実測（コーパスの宣言49綴り、注入が要ると判定される数）:

    | 状態 | maxima | julia_full |
    |---|---|---|
    | 直す前 | 11/49 | 26/49 |
    | 裸の名詞を削るだけ | **26/49**（悪化） | **44/49**（悪化） |
    | 削って `as` 句を足す | **3/49** | **16/49** |

    残る分は「そのバックエンドがその操作を持っていない」場合だけ（粒子演算子は
    julia_full 専用、ベクトル場と時間依存は maxima 専用）なので、
    **支持されているバックエンドで**測る。
    """

    @staticmethod
    def _supported(op):
        for o in _SCHEMA["operations"]:
            if o.get("operation") == op:
                return o.get("supported_backends") or []
        raise AssertionError(op)

    @pytest.mark.parametrize("en,ja,want", REAL_DECLARATIONS)
    def test_no_injection_on_a_backend_that_supports_it(self, en, ja, want):
        from notebook.runtime.llm_supervisor import LLMSupervisor as _S
        for be in self._supported(want):
            sup = _S(validator=OperationValidator(), llm_client=None,
                     domains=[], primary_backend=be)
            assert not sup.needs_expression_injection(en), (be, en)

    @pytest.mark.parametrize("text,want", [
        ("Define R, L, w, C as scalars", "define_scalar_parameter"),
        ("Define gamma, B as real scalars", "define_scalar_parameter"),
        ("S as the spin operator", "define_spin_operator"),
        ("b as a boson operator", "define_boson_operator"),
        ("g as a fermion operator", "define_fermion_operator"),
    ])
    def test_plural_and_operator_forms_need_no_injection(self, text, want):
        from notebook.runtime.llm_supervisor import LLMSupervisor as _S
        for be in self._supported(want):
            sup = _S(validator=OperationValidator(), llm_client=None,
                     domains=[], primary_backend=be)
            assert not sup.needs_expression_injection(text), (be, text)
