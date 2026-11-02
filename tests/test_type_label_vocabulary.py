r"""型の説明は何を宣言するか —— 2つの表が別々に答えていた。

実測 2026-09-05。宣言系13操作の ja 別名116件を、そのまま型の説明として
`_infer_type_from_label` に渡すと **64件が別の操作**を答えた。実パイプラインで
測ると **11綴りが `confirmed` を返しながら限定語を落として**いた:

| 打った文 | 直す前 | 落ちていたもの |
|---|---|---|
| `A を対称行列とする` | `define_matrix` | 対称性 |
| `A を反対称行列とする` | `define_matrix` | 反対称性 |
| `A をエルミート行列とする` | `define_matrix` | エルミート性 |
| `A を反対称とする` | `define_scalar_parameter` | 対称性そのもの |
| `A をスカラー関数とする` | `define_scalar_parameter` | 場であること |
| `A をベクトル関数とする` | `define_vector` | 場であること |
| `q を時間の関数として宣言` | `define_scalar_parameter` | 時間依存 |
| `A を座標系として定義する` | `define_scalar_parameter` | 座標系であること |
| `A を関数として定義` | `define_scalar_parameter` | 関数であること |

原因は2つ。**(a) 部分文字列照合なので限定語が無視される**（`対称`行列 → 行列、
ベクトル`関数` → ベクトル）。**(b) 語彙が足りない**（`スカラー関数`・`時間の関数`
は操作の別名には在るのに型語彙表に無い）。

畳み方は `notebook/runtime/type_label.py` の docstring にある。要点だけ:
**照合はラベル全体の完全一致**（部分文字列にすると `define` が `define matrix`
の中に出て何もかも `define_function` になる——試作で実測）、**優先順位表は残す**
（`時間依存のスカラー場` はどの別名とも一致しないので、そこが答える）。

`対称行列` は**型と対称性の両方**を言っているので、`declare_symmetry` に任意欄
`algebra` を足して両方を宣言する。Maxima の `declare` は (原子, 特性) の対を
取るので `declare(A, nonscalar, symmetric)` は**拒否され**、
`declare(A, nonscalar, A, symmetric)` が通る（実測）。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
import notebook.runtime.backend_handler as BH
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.type_label import (
    declaration_operation_for_label, label_stem, symmetry_value_for_label)
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


# --- 直った綴り（日英そろえてある）------------------------------------------
FIXED = [
    ("ja", "A を対称行列とする",            "declare_symmetry",       "symmetric",     "matrix"),
    ("ja", "A を反対称行列とする",           "declare_symmetry",       "antisymmetric", "matrix"),
    ("ja", "A をエルミート行列とする",        "declare_symmetry",       "hermitian",     "matrix"),
    ("ja", "A を反対称とする",              "declare_symmetry",       "antisymmetric", "scalar"),
    ("ja", "A を時間の関数として宣言",        "declare_time_dependent", None,            None),
    ("ja", "A をスカラー関数とする",          "define_scalar_field",    None,            None),
    ("ja", "A をスカラー関数として定義する",    "define_scalar_field",    None,            None),
    ("ja", "A をベクトル関数とする",          "define_vector_field",    None,            None),
    ("ja", "A をベクトル関数として定義する",    "define_vector_field",    None,            None),
    ("en", "define A as a symmetric matrix",      "declare_symmetry",    "symmetric",     "matrix"),
    ("en", "define A as a hermitian matrix",      "declare_symmetry",    "hermitian",     "matrix"),
    ("en", "define A as an antisymmetric function", "declare_symmetry",  "antisymmetric", "scalar"),
    ("en", "define A as a scalar field",          "define_scalar_field", None,            None),
    ("en", "define A as a vector field",          "define_vector_field", None,            None),
]

# --- 変わってはいけない綴り ---------------------------------------------------
UNCHANGED = [
    ("ja", "A を行列とする",              "define_matrix"),
    ("ja", "A をベクトルとする",           "define_vector"),
    ("ja", "A をテンソルとする",           "define_tensor"),
    ("ja", "A をスカラーとする",           "define_scalar_parameter"),
    ("ja", "A を実数とする",              "define_scalar_parameter"),
    ("ja", "A をスカラー場とする",          "define_scalar_field"),
    ("ja", "A を時間依存のスカラー場とする",   "define_scalar_field"),
    ("ja", "A を１電子エネルギーとする",      "define_scalar_parameter"),
    ("en", "define A as a matrix",       "define_matrix"),
    ("en", "define A as a vector",       "define_vector"),
    ("en", "define A as a scalar",       "define_scalar_parameter"),
    ("en", "define A, B as scalars",     "define_scalar_parameter"),
]


def _run(eng, sup, text):
    return asyncio.get_event_loop().run_until_complete(
        run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))


def _ask(lang, text):
    nb.start_engine(title="type_label", mode="standard", language=lang, domains=[])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend="maxima")
        resp = _run(eng, sup, text)
        qty = eng.symbol_context.get_quantity("A")
        return resp, qty
    finally:
        nb.stop_engine()


class TestTheSpellingReachesItsOwnOperation:
    @pytest.mark.parametrize("lang,text,want,symmetry,algebra", FIXED)
    def test_fixed(self, lang, text, want, symmetry, algebra):
        resp, qty = _ask(lang, text)
        assert resp.get("operation") == want, (text, resp.get("operation"))
        if symmetry is not None:
            assert getattr(qty, "symmetry", None) == symmetry, text
        if algebra is not None:
            assert getattr(getattr(qty, "math_type", None), "algebra", None) == algebra, text

    @pytest.mark.parametrize("lang,text,want", UNCHANGED)
    def test_unchanged(self, lang, text, want):
        resp, _ = _ask(lang, text)
        assert resp.get("operation") == want, (text, resp.get("operation"))


class TestTheTwoTablesAgree:
    """宣言系の別名は、型の説明として渡しても**自分の操作**に解決する。

    これがこの一問の不変条件そのもの。手前に専用の読みがある族（粒子演算子など）
    は `_infer_type_from_label` に届く前に答えるので、ここでは表どうしを直接見る。
    """

    def _decl_ops(self):
        v = OperationValidator()
        return [o for o in sorted(v.known_operations)
                if o.startswith(("define_", "declare_"))]

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_every_alias_resolves_to_its_owner(self, lang):
        v = OperationValidator()
        wrong = []
        for op in self._decl_ops():
            spec = v.get_spec(op) or {}
            backends = spec.get("supported_backends") or []
            for alias in (spec.get("aliases") or {}).get(lang, []):
                stem = label_stem(alias, lang)
                if not stem:
                    continue
                for backend in backends:
                    got = declaration_operation_for_label(stem, backend)
                    # 別の宣言操作が同じ語幹を先に登録していることはある
                    # （`実数` は scalar_parameter のもの）。**別の操作へ**
                    # 解決したときだけ咎める。
                    if got is not None and got != op:
                        owner = v.get_spec(got) or {}
                        if stem in [label_stem(a, lang)
                                    for a in (owner.get("aliases") or {}).get(lang, [])]:
                            continue          # 語幹を共有している（表の重複）
                        wrong.append((op, alias, stem, got))
        assert wrong == [], wrong


class TestWhatIsDeliberatelyNotRead:
    """入れなかったものにも理由がある（十一問目・十五問目と同じ判断）。"""

    @pytest.mark.parametrize("label", ["対称性", "反対称性", "declare symmetry",
                                       "set symmetry", "symmetry"])  # noqa: E501
    def test_a_generic_symmetry_word_names_no_value(self, label):
        """`対称性を宣言` は**どの対称性かを言っていない**。埋めると推測になる。"""
        assert symmetry_value_for_label(label) is None, label

    def test_the_refusal_names_the_missing_field(self):
        resp, _ = _ask("ja", "A を対称性を宣言する")
        assert resp.get("status") == "failed"
        assert "symmetry" in str(resp.get("error")), resp.get("error")

    @pytest.mark.parametrize("bare", ["define", "declare", "set", "as",
                                      "定義", "宣言", "設定"])
    def test_a_bare_declaration_verb_is_not_a_type(self, bare):
        """`define` は `define_function` の別名だが、型の説明としては何も
        言っていない。索引に入れると「A を定義する」が関数宣言になり、
        スカラー既定（`１電子エネルギー` などの物理ラベルのため）が壊れる。"""
        assert declaration_operation_for_label(bare, "maxima") is None, bare


class TestTheBackendFiltersTheAnswer:
    def test_a_julia_only_operation_is_not_offered_in_maxima(self):
        assert declaration_operation_for_label("フェルミオン演算子", "maxima") is None
        assert (declaration_operation_for_label("フェルミオン演算子", "julia_full")
                == "define_fermion_operator")

    def test_the_handler_names_its_own_backend(self):
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        assert MaximaBackendHandler._SCHEMA_BACKEND == "maxima"
        assert JuliaBackendHandler._SCHEMA_BACKEND == "julia_full"


class TestTypeAndSymmetryAreDeclaredTogether:
    """`対称行列` は両方を言っている。片方だけ宣言すれば他方が落ちる。"""

    def test_maxima_declares_both_in_one_command(self):
        from notebook.runtime.maxima_backend_handler import MaximaBackendHandler
        h = MaximaBackendHandler(None)
        both = h._build_declare_symmetry(
            {"name": "A", "symmetry": "symmetric", "algebra": "matrix"}).command
        # Maxima の `declare` は (原子, 特性) の**対**を取る。実測 2026-09-05:
        # `declare(A, nonscalar, symmetric)` は ok=False でどちらの特性も立たず、
        # `declare(A, nonscalar, A, symmetric)` が通る。
        assert both == "declare(A, nonscalar, A, symmetric)"
        only = h._build_declare_symmetry(
            {"name": "A", "symmetry": "antisymmetric"}).command
        assert only == "declare(A, antisymmetric)"


class TestTheLabelIsNormalisedTheSameWayOnBothSides:
    @pytest.mark.parametrize("alias,want", [
        ("対称行列とする", "対称行列"),
        ("スカラー場として定義する", "スカラー場"),
        ("時間の関数として宣言", "時間の関数"),
        ("ボゾン演算子を定義する", "ボゾン演算子"),
        ("反対称に設定", "反対称"),
    ])
    def test_japanese(self, alias, want):
        assert label_stem(alias, "ja") == want

    @pytest.mark.parametrize("alias,want", [
        ("define scalar field", "scalar field"),
        ("define_vector_field", "vector field"),
        ("an antisymmetric function", "antisymmetric function"),
        ("the matrix", "matrix"),
        ("real scalar", "real scalar"),
    ])
    def test_english_strips_the_head_and_the_article(self, alias, want):
        assert label_stem(alias, "en") == want


class TestThePriorityTableStillAnswersCombinedLabels:
    """組み合わせラベルはどの別名とも一致しないので、優先順位表が答える。"""

    @pytest.mark.parametrize("label,want", [
        ("時間依存のスカラー場", "define_scalar_field"),
        ("time-dependent scalar field", "define_scalar_field"),
        ("１電子エネルギー", "define_scalar_parameter"),
        ("BCS 秩序パラメータ", "define_scalar_parameter"),
    ])
    def test_falls_through_to_keywords(self, label, want):
        assert declaration_operation_for_label(label, "maxima") is None, label
        assert BH._type_from_keywords(label) == want


class TestTheOrderOfTheSymmetryWordsDoesNotMatter:
    """最初に「`反対称` は `対称` を含むので順序が効く」と註に書いたが、**偽だった**。

    並びを入れ替える変異（M7）はテストを1件も落とさなかった。理由を測ると、
    `反`/`anti` が識別接頭辞なので `re` は走査位置ごとに選択肢を試し、短い側が
    その位置で当たることがない。**落ちない変異はまず当て方を疑う**——ここは
    当て方ではなく、私の書いた理由のほうが誤っていた（十五問目と同じ形）。

    規則ではないと分かった以上、規則として固定はしない。**事実だけ**を残す。
    """

    _JA_SWAPPED = (r"(?P<symmetric>対称(?!性))|(?P<hermitian>エルミート)"
                   r"|(?P<antisymmetric>反対称(?!性))")
    _EN_SWAPPED = (r"(?i)(?P<symmetric>symmetric)|(?P<hermitian>hermitian)"
                   r"|(?P<antisymmetric>anti[-\s]?symmetric)")

    @staticmethod
    def _read(pattern, text):
        import re
        m = re.search(pattern, text)
        return next((k for k, v in (m.groupdict() if m else {}).items() if v), None)

    @pytest.mark.parametrize("text,want", [
        ("対称行列", "symmetric"), ("反対称行列", "antisymmetric"),
        ("エルミート行列", "hermitian"), ("反対称", "antisymmetric"),
        ("対称性", None), ("反対称性", None),
    ])
    def test_japanese_reads_the_same_either_way(self, text, want):
        assert symmetry_value_for_label(text) == want
        assert self._read(self._JA_SWAPPED, text) == want

    @pytest.mark.parametrize("text,want", [
        ("symmetric matrix", "symmetric"), ("antisymmetric", "antisymmetric"),
        ("anti-symmetric", "antisymmetric"), ("hermitian matrix", "hermitian"),
    ])
    def test_english_reads_the_same_either_way(self, text, want):
        assert symmetry_value_for_label(text) == want
        assert self._read(self._EN_SWAPPED, text) == want
