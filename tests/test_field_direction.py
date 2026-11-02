r"""「磁場がある」と「どの向きか」は別の問い——訊いていたのは向きだけだった。

実測 2026-09-07（worklist の `field`、7行・3鍵）。`f1_field` は `横磁場|縦磁場` しか
知らないので、**向きの書かれていない磁場は磁場なしと区別できなかった**:

| 打った文 | 直す前 |
|---|---|
| `横磁場イジング模型 2サイトで S^z_1 の交換子を求めて` | `[H, S^z_1] = i h S^y_1` ✔ |
| **`磁場中のイジング模型 …`** | **`[H, S^z_1] = 0`** |
| **`Ising model in a magnetic field, …`** | **`0`** |
| `イジング模型 …`（磁場なし） | `0` ✔ |

磁場を書かなかった場合と**同じ答え**で、しかも `[H,S^z]=0`（S^z は保存する）は
横磁場なら偽。両方 `completed`。**日英どちらも同じ**なので、これは日英の割れでは
なく両方の欠陥だった。

**向きは既定しない。** どちらを既定しても物理の選択で（CLAUDE.md「問題依存の物理の
選択を共有コードに焼き込まない」）、`[H,S^z]` は横磁場なら `i h S^y`、縦磁場なら `0`
と**答えが変わる**。微分の種別（∂ か d か）とまったく同じ形なので同じ扱いにした
——読めなければ理由つきで断る。

**同じ4行が3箇所に写してあった**（`_parse_spin_commutator` /
`_parse_spectral_function` / `_parse_spin_spectral`）。第三の状態を足すと3箇所とも
直すことになるので、読みを `read_field` に、断りを `field_unstated_refusal` に出した。

**測定器の誤りを2つ、途中で直している。**
① 最初「`f1_field` に `en` 鍵が無い」と報告したが**誤り**——`en` は
`(?i)transverse|longitudinal` で、`field` という語を含まないので、
「`field` か `場` を含むパターン」で絞った grep が見落としていた。英語は元から 4/4。
② 綴り空間の期待値に「向き未記載＝横」と書いていた——**それは私が既定しないと
決めたはずの物理の選択**で、コードではなく期待値のほうが誤りだった。

綴り空間 ja **4/7 → 7/7**、en 4/4 → **6/6**、`spin_field_magnitude` ja **4/5 → 5/5**
（`実効磁場` は `有効磁場` と同じ語。英語は元から5綴りすべてを受けていた）。
"""
import asyncio
import pathlib
import re

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.i18n import t
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_f1 import FIELD_DIRECTION_UNSTATED, read_field
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _ask(lang, text):
    """**値で測る。** どの操作に届いたかでは足りない——磁場の項が落ちても
    `completed` で、答えだけが違っていた。"""
    nb.start_engine(title="fd", mode="standard", language=lang,
                    domains=["condensed_matter"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=[], primary_backend="maxima")
        r = asyncio.get_event_loop().run_until_complete(
            run_interpret_pipeline(InterpretRequest(
                tokens=[InputToken(type="text", value=text)]), eng, sup))
        vals = [(v.latex or v.canonical or "") for v in eng.expressions.values()]
        return r.get("status"), (vals[-1] if vals else ""), str(r.get("error") or "")
    finally:
        nb.stop_engine()


class TestTheDirectionIsRead:
    @pytest.mark.parametrize("lang,text,want", [
        ("ja", "横磁場イジング模型 2サイトで S^z_1 の交換子を求めて", ("h", "x")),
        ("ja", "横磁場中のイジング模型 2サイトで S^z_1 の交換子を求めて", ("h", "x")),
        ("ja", "横方向磁場イジング模型 2サイトで S^z_1 の交換子を求めて", ("h", "x")),
        ("ja", "縦磁場イジング模型 2サイトで S^z_1 の交換子を求めて", ("h", "z")),
        ("ja", "縦方向磁場イジング模型 2サイトで S^z_1 の交換子を求めて", ("h", "z")),
        ("en", "transverse field Ising model, 2 sites", ("h", "x")),
        ("en", "transverse-field Ising model with 2 sites", ("h", "x")),
        ("en", "longitudinal field Ising model, 2 sites", ("h", "z")),
    ])
    def test_stated(self, lang, text, want):
        assert read_field(text, lang) == want, text

    @pytest.mark.parametrize("lang,text", [
        ("ja", "イジング模型 2サイトで S^z_1 の交換子を求めて"),
        ("ja", "ハイゼンベルク模型 2サイトで S^z_1 の交換子を求めて"),
        ("en", "Ising model, 2 sites, commutator of S^z_1"),
        ("en", "Heisenberg model, 2 sites"),
    ])
    def test_no_field(self, lang, text):
        assert read_field(text, lang) is None, text

    @pytest.mark.parametrize("lang,text", [
        ("ja", "磁場中のイジング模型 2サイトで S^z_1 の交換子を求めて"),
        ("ja", "外部磁場のあるイジング模型 2サイトで S^z_1 の交換子を求めて"),
        ("ja", "外場のあるイジング模型 2サイトで S^z_1 の交換子を求めて"),
        ("en", "Ising model in a magnetic field, 2 sites"),
        ("en", "Ising model in an external field, 2 sites"),
    ])
    def test_mentioned_without_a_direction(self, lang, text):
        """**磁場なしと区別できることが要点。** 直す前はここが `None` で、
        磁場を書かなかった場合と同じ答えになっていた。"""
        assert read_field(text, lang) == FIELD_DIRECTION_UNSTATED, text
        assert read_field(text, lang) is not None, text


class TestAnUnstatedDirectionIsRefusedWithAReason:
    @pytest.mark.parametrize("lang,text", [
        ("ja", "磁場中のイジング模型 2サイトで S^z_1 の交換子を求めて"),
        ("en", "Ising model in a magnetic field, 2 sites, commutator of S^z_1"),
    ])
    def test_refused(self, lang, text):
        st, val, err = _ask(lang, text)
        assert st == "failed", (text, st, val)
        assert t("interpreter.field_direction_unstated", lang) in err, err

    @pytest.mark.parametrize("lang,text,want", [
        ("ja", "横磁場イジング模型 2サイトで S^z_1 の交換子を求めて", "i h S^{y}_{1}"),
        ("en", "transverse field Ising model, 2 sites, commutator of S^z_1",
         "i h S^{y}_{1}"),
    ])
    def test_a_stated_transverse_field_still_computes(self, lang, text, want):
        st, val, _ = _ask(lang, text)
        assert st == "completed" and want in val, (text, st, val)

    @pytest.mark.parametrize("lang,text", [
        ("ja", "イジング模型 2サイトで S^z_1 の交換子を求めて"),
        ("en", "Ising model, 2 sites, commutator of S^z_1"),
    ])
    def test_no_field_still_computes(self, lang, text):
        """磁場を書いていないのは**書き忘れではない**——訊かない。"""
        st, val, _ = _ask(lang, text)
        assert st == "completed" and val.endswith("= 0"), (text, st, val)

    def test_the_direction_changes_the_answer(self):
        """既定を置かない理由そのもの。横なら `i h S^y`、縦なら `0`。"""
        _, tx, _ = _ask("ja", "横磁場イジング模型 2サイトで S^z_1 の交換子を求めて")
        _, lg, _ = _ask("ja", "縦磁場イジング模型 2サイトで S^z_1 の交換子を求めて")
        assert tx != lg, (tx, lg)
        assert lg.endswith("= 0"), lg


class TestTheEffectiveFieldVocabularyIsSymmetric:
    """英語は5綴りすべてを受けるのに、日本語だけ同義語を1つ落としていた。"""

    @pytest.mark.parametrize("text", [
        "E2 の有効磁場の大きさを求めて", "E2 の有効磁場を求めて",
        "E2 のスピンに作用する有効磁場の大きさを求めて",
        "E2 の実効磁場の大きさを求めて", "E2 の有効磁場の強さを求めて",
    ])
    def test_japanese(self, text):
        m = nlp("spin_field_magnitude", "ja").search(text)
        assert m and m.group("eid") == "E2", text

    @pytest.mark.parametrize("text", [
        "Find the magnitude of the effective field of E2",
        "Find the effective field of E2",
        "the magnitude of the effective field acting on the spin of E2",
        "Find the effective field in E2",
        "Find the strength of the effective field of E2",
    ])
    def test_english(self, text):
        m = nlp("spin_field_magnitude", "en").search(text)
        assert m and m.group("eid") == "E2", text


class TestTheReadingLivesInOnePlace:
    """四十問目以来の形——**数える検査そのものにする**。直す前は同じ4行が3箇所に
    写してあり、第三の状態を足すと3箇所とも直すことになっていた。"""

    _SRC = pathlib.Path(__file__).resolve().parent.parent / "ui"

    def test_no_site_spells_the_field_keys_itself(self):
        offenders = []
        for f in self._SRC.rglob("*.py"):
            src = f.read_text(encoding="utf-8")
            for key in ("f1_field_transverse", "f1_field_longitudinal",
                        "f1_field_mention"):
                if f.name != "interpret_f1.py" and f'nlp("{key}"' in src:
                    offenders.append(f"{f.name}: {key}")
        assert not offenders, (
            f"磁場の鍵を `read_field` の外で読んでいる: {offenders}")

    def test_every_hamiltonian_call_asks_first(self):
        """`build_hamiltonian(field=…)` を呼ぶ手前で必ず断りを訊く。

        **数えるのは AST。** grep は「その規則を説明した docstring」まで数える
        ——最初にそう書いて、`build_hamiltonian` 4・guard 4 で落ちた（本文の
        呼び出しは3つずつ、残りは docstring と関数定義）。
        """
        import ast
        tree = ast.parse((self._SRC / "interpret_early.py").read_text(encoding="utf-8"))
        names = [n.func.attr if isinstance(n.func, ast.Attribute) else
                 getattr(n.func, "id", None)
                 for n in ast.walk(tree) if isinstance(n, ast.Call)]
        calls = names.count("build_hamiltonian")
        guards = names.count("field_unstated_refusal")
        assert calls >= 3, calls
        assert guards == calls, (
            f"build_hamiltonian の呼び出し {calls} に対し guard {guards}")

    def test_the_old_merged_key_is_gone(self):
        """`f1_field`（向きの鍵と在ることの鍵を兼ねていた）は分割済み。"""
        import json
        d = json.loads((pathlib.Path(__file__).resolve().parent.parent
                        / "i18n" / "nl_patterns.json").read_text(encoding="utf-8"))
        assert "f1_field" not in d
        for k in ("f1_field_transverse", "f1_field_longitudinal", "f1_field_mention"):
            assert set(d[k]) >= {"ja", "en"}, k
