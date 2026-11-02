r"""昇降演算子の呼び名——`下降`/`lowering` は `消滅`/`annihilation` の別名。

実測 2026-09-08（worklist の `消滅` / `annihilation`、7行・3鍵）。語彙は
`消滅|生成` / `creation|annihilation` しか無く、**同じ対象の別の呼び名**が
落ちていた:

| 打った文 | 直す前 |
|---|---|
| `c を消滅演算子とする` | failed（統計が書かれていない＝正しい） |
| **`c を下降演算子とする`** / `上昇演算子` | **`define_scalar_parameter` を confirmed** |
| **`define c as a lowering / raising / ladder operator`** | **同上** |

CLAUDE.md が二十四問目として記録している「**演算子と書いた文がスカラーを
登録する**」——その当の欠陥が、別の綴りで残っていた。

**語彙を広げるだけでは足りなかった。** 向きは下流で

    op_type_key = "annihilation" if (kind == "消滅" or "annihil" in kind) else "creation"

と**値で引かれて**おり、`else` が `creation` を既定にしている。`下降` をパターンに
足すだけだと、**下降演算子が生成演算子として宣言される**——`multiply_expressions`
のテンプレートが `*` を直書きしていたのと同じ形（MATMUL-TEMPLATE-STAR）。
呼び名と向きの対応は `notebook/runtime/particle_ladder.py` に1つ置いた。

**`昇降` / `ladder` は族の名前で、向きを言っていない。** 「昇降演算子の宣言か」を
訊く鍵（`bare_ladder_op_decl`——返すのは「統計が書かれていない」という断りだけ）は
受けてよいが、向きを要求する読み（`particle_operator_decl`）は受けない。受ければ
既定で片方に倒れる。

**英語には枝が3つあった**（`typeC` / `typeD` / `typeE`）。直したとき `typeE` を
取りこぼし、`c is the electron lowering operator for …` の形だけが素の綴りのまま
残った——**枝の数を数えてから置換する**。

**利用者の一言で見つかった、生成側に固有の欠陥。** 「creation operator については
問題なし？」と訊かれて測り直したところ、**† の綴りが読めていなかった**——消滅演算子に
† は付かないので、消滅側を測っても出ない:

| 打った文 | 直す前 |
|---|---|
| `c' を生成演算子とする` | ✔ |
| `c† を生成演算子とする` | **当たらない** |
| **`c^\dagger を電子の生成演算子とする`** | **`dagger` という名前の演算子を宣言**（成功） |

正規表現が `c` で当たれず、後ろの `dagger` に latch していた。CLAUDE.md は
「`a^\dagger` と `a'` は両方とも生成演算子の入力として受け付ける」と書いているが、
それは**式**の側で、**宣言の読みは知らなかった**。しかも `bare_ladder_op_decl` は
名前を自前で `'*` と綴っており、`op_expr` を読む `particle_operator_decl` は `c†` を
読めるのにこちらは読めない——**同じ問い（演算子の名前はどう書けるか）に2つの答え**。

**測定器の誤りを2つ、途中で直している。** ①英語の文を**日本語セッション**で測って
`define_scalar_parameter` と報告した（言語が違えばパターンが違う）。②`op_type` という
欄名で向きを読もうとしたが、その名前の欄は無かった。
"""
import json
import pathlib

import pytest

from notebook.runtime.nl_patterns import nlp, pick_group
from notebook.runtime.particle_ladder import (
    LADDER_DIRECTION, LADDER_FAMILY, alternation, ladder_kind)

_JSON = json.loads(
    (pathlib.Path(__file__).resolve().parent.parent / "i18n" / "nl_patterns.json")
    .read_text(encoding="utf-8"))


class TestTheDirectionTableIsTheAuthority:
    @pytest.mark.parametrize("word,want", sorted(LADDER_DIRECTION.items()))
    def test_each_name_maps_to_its_direction(self, word, want):
        assert ladder_kind(word) == want, word

    @pytest.mark.parametrize("word", LADDER_FAMILY)
    def test_a_family_name_states_no_direction(self, word):
        """`昇降` / `ladder` は向きを言っていない——**`None` は「生成」ではない**。"""
        assert ladder_kind(word) is None, word

    @pytest.mark.parametrize("word", ["", "  ", "スカラー", "operator"])
    def test_an_unrelated_word_is_not_a_direction(self, word):
        assert ladder_kind(word) is None, word

    @pytest.mark.parametrize("kind", ["annihilation", "creation"])
    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_fragments_agree_with_the_table(self, kind, lang):
        """**パターンは JSON なので表を呼べない**——書き写しは許すが、
        書き写しが表と同じであることはここが数える。"""
        frag = _JSON["_fragments"][f"ladder_{'ann' if kind == 'annihilation' else 'cre'}_{lang}"]
        assert frag == f"(?:{alternation(kind, lang)})", (kind, lang, frag)


class TestNoKeySpellsTheNamesItself:
    """四十問目以来の形——**数える検査そのものにする**。英語の枝を1つ
    取りこぼしたのは、手で数えたからだった。"""

    KEYS = ("bare_ladder_op_decl", "particle_def_op_text", "particle_operator_decl")

    @pytest.mark.parametrize("key", KEYS)
    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_no_literal_left(self, key, lang):
        import re
        body = re.sub(r"\{\{\w+\}\}", "", _JSON[key][lang])
        left = [w for w in LADDER_DIRECTION if w in body]
        assert not left, f"{key}[{lang}] が自前で綴っている: {left}"

    @pytest.mark.parametrize("key", KEYS)
    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_fragment_is_spliced(self, key, lang):
        assert "{{ladder_" in _JSON[key][lang], (key, lang)


BARE_JA = ["c を消滅演算子とする", "c を生成演算子とする", "c を下降演算子とする",
           "c を上昇演算子とする", "a を昇降演算子とする",
           "c を消滅演算子として定義する", "c' を生成演算子とする",
           "c を消滅演算子と定義して"]
BARE_EN = ["define c as an annihilation operator", "define c as a creation operator",
           "define c as a lowering operator", "define c as a raising operator",
           "define c as a ladder operator", "c is the annihilation operator",
           "let c be an annihilation operator"]


class TestABareLadderOperatorIsRead:
    """粒子も統計も書かれていない形。**どちらとも決めず**、名前だけ読む。"""

    @pytest.mark.parametrize("text", BARE_JA)
    def test_japanese(self, text):
        m = nlp("bare_ladder_op_decl", "ja").search(text)
        assert m and m.group("name") in ("c", "c'", "a"), (text, m)

    @pytest.mark.parametrize("text", BARE_EN)
    def test_english(self, text):
        m = nlp("bare_ladder_op_decl", "en").search(text)
        assert m and m.group("name") == "c", (text, m)


PART_JA = [
    ("c を電子の消滅演算子とする", "annihilation"),
    ("d を電子の下降演算子とする", "annihilation"),
    ("e を電子の生成演算子とする", "creation"),
    ("f を電子の上昇演算子とする", "creation"),
    ("電子の消滅演算子を c とする", "annihilation"),
]
PART_EN = [
    ("define c as an electron annihilation operator", "annihilation"),
    ("define c as an electron lowering operator", "annihilation"),
    ("define c as an electron creation operator", "creation"),
    ("define c as an electron raising operator", "creation"),
    ("define c as the lowering operator for electrons", "annihilation"),
    ("c is the electron lowering operator", "annihilation"),
]


class TestTheParticleFormReadsTheDirection:
    """**向きは既定で埋めない。** 直す前の `else: creation` のままだと、
    下降演算子が生成演算子として宣言される。"""

    @pytest.mark.parametrize("text,want", PART_JA)
    def test_japanese(self, text, want):
        m = nlp("particle_operator_decl", "ja").search(text)
        assert m, text
        assert ladder_kind(pick_group(m, "type")) == want, (text, pick_group(m, "type"))
        assert pick_group(m, "particle").strip() == "電子", text

    @pytest.mark.parametrize("text,want", PART_EN)
    def test_english(self, text, want):
        m = nlp("particle_operator_decl", "en").search(text)
        assert m, text
        assert ladder_kind(pick_group(m, "type")) == want, (text, pick_group(m, "type"))
        assert "electron" in pick_group(m, "particle"), text

    def test_a_family_name_is_not_accepted_where_the_direction_is_required(self):
        """`電子の昇降演算子` は向きを言っていない——ここで受けると既定で倒れる。"""
        assert nlp("particle_operator_decl", "ja").search("c を電子の昇降演算子とする") is None
        assert nlp("particle_operator_decl", "en").search(
            "define c as an electron ladder operator") is None


class TestTheHandlerCarriesTheDirection:
    """**直接の消費者に当てる。** パターンと `ladder_kind` だけを測っていたので、
    下流の `else: creation` 既定に戻す変異（M7）が**生き残った**——読めていても
    渡っていなければ意味がない。`_op_type_key` がハンドラの出す欄。"""

    HANDLER = None

    @classmethod
    def _h(cls):
        if cls.HANDLER is None:
            from notebook.runtime.julia_backend_handler import JuliaBackendHandler
            cls.HANDLER = JuliaBackendHandler(None)
        return cls.HANDLER

    @pytest.mark.parametrize("lang,text,want", [
        ("ja", "c を電子の消滅演算子とする", "annihilation"),
        ("ja", "d を電子の下降演算子とする", "annihilation"),
        ("ja", "e を電子の生成演算子とする", "creation"),
        ("ja", "f を電子の上昇演算子とする", "creation"),
        ("en", "define g as an electron lowering operator", "annihilation"),
        ("en", "define h as an electron raising operator", "creation"),
        ("en", "define i as an electron annihilation operator", "annihilation"),
        ("en", "define j as an electron creation operator", "creation"),
    ])
    def test_direction_reaches_the_operation(self, lang, text, want):
        r = self._h().extra_pre_phase1(text, lang)
        assert getattr(r, "operation", None) == "define_fermion_operator", (text, r)
        assert r.args["_op_type_key"] == want, (text, r.args)
        assert r.args["_particle_key"] == "electron", (text, r.args)


class TestTheDaggerIsPartOfTheName:
    """**生成側に固有**——消滅演算子に † は付かないので、消滅側だけ測っても出ない。
    利用者の「creation operator については問題なし？」で見つかった。"""

    NAMES = ["c", "c'", "c†", "c^†", r"c^\dagger", r"c^{\dagger}"]

    @pytest.mark.parametrize("name", NAMES)
    def test_bare_japanese(self, name):
        m = nlp("bare_ladder_op_decl", "ja").search(f"{name} を生成演算子とする")
        assert m and m.group("name") == name, (name, m.group("name") if m else None)

    @pytest.mark.parametrize("name", NAMES)
    def test_bare_english(self, name):
        m = nlp("bare_ladder_op_decl", "en").search(
            f"define {name} as a creation operator")
        assert m and m.group("name") == name, (name, m.group("name") if m else None)

    @pytest.mark.parametrize("name", NAMES)
    def test_with_a_particle(self, name):
        m = nlp("particle_operator_decl", "ja").search(f"{name} を電子の生成演算子とする")
        assert m and pick_group(m, "op") == name, (name, m)

    @pytest.mark.parametrize("name", [r"c^\dagger", "c†", r"c^{\dagger}"])
    def test_the_base_name_is_registered_not_the_dagger(self, name):
        """直す前は `name='dagger'`——研究者の `c` が消えていた。"""
        from notebook.runtime.julia_backend_handler import JuliaBackendHandler
        r = JuliaBackendHandler(None).extra_pre_phase1(
            f"{name} を電子の生成演算子とする", "ja")
        assert r.args["name"] == "c", (name, r.args)
        assert r.args["_op_expr_display"] == name, (name, r.args)
        assert r.args["_op_type_key"] == "creation", (name, r.args)

    def test_the_two_keys_spell_the_name_the_same_way(self):
        """`bare_ladder_op_decl` は名前を自前で綴っていた（`'*` だけ）。"""
        assert "{{op_expr}}" in _JSON["bare_ladder_op_decl"]["ja"]
        assert "{{op_expr}}" in _JSON["bare_ladder_op_decl"]["en"]
        assert "{{op_expr}}" in _JSON["particle_operator_decl"]["ja"]
