r"""「演算子」と書いた文がスカラーを登録していた（Maxima セッション）。

実測 2026-09-06（`magnetic_resonance`＝Maxima のセッション）。**silent-wrong 8件**:

| 打った文 | 直す前 | 登録 |
|---|---|---|
| `c を電子の消滅演算子とする` | confirmed / `define_scalar_parameter` | **スカラー** |
| `c をフェルミオン演算子とする` | 同上 | スカラー |
| `c を消滅演算子とする` | 同上 | スカラー |
| `define c as an electron annihilation operator` | 同上 | スカラー |
| `define c as a fermion operator` | 同上 | スカラー |
| `define c as an annihilation operator` | 同上 | スカラー |
| `define S as a Pauli matrix` | confirmed / `define_matrix` | **行列** |
| `let c be a fermion operator` | failed | — |

CLAUDE.md が「核スピン演算子」で記録した defect とまったく同じ形で、
**スピンについては直され、粒子演算子については残っていた。**

**受けるのではなく断る（利用者の判断）。** スピンは `spin_algebra` が CAS-bypass
で実装されているので Maxima でも受けて意味があるが、フェルミオン／ボソン代数は
Julia/QuantumAlgebra にしかない。受け入れると `c'(k) c(k)` がただの積になり、
その後の計算が全部おかしくなる——「第二量子化には Julia セッションが要る。
バックエンドはセッション作成時に固定される」と名指すほうがよい（原則 #6）。

**研究者が書いていない統計を断り文で名指さない**（利用者の指摘）。粒子も統計も
書かない `c を消滅演算子とする` では、系が代表として `define_fermion_operator` を
選んでいるだけなので、名指すのは**族**にする。

**訊くのは Julia だけ。** Maxima では答えても計算できないので訊かない
（利用者の指摘）——共有層が最初に「このバックエンドで動かせるか」を訊き、
動かせなければそこで断る。Julia は動かせるので次の問い（統計は書かれているか）へ進む。

**冠詞は粒子名ではない。** `define c as an annihilation operator` の `an` が
`particle_operator_decl[en]` の粒子として捕まっていた（実測）——粒子が書かれて
いない綴りである。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.backend_handler import names_particle_operator_declaration
from notebook.runtime.julia_backend_handler import JuliaBackendHandler
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import nlp
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.type_label import declaration_unsupported_here
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

#: Maxima セッションで**断られる**べき綴り（第二量子化の演算子）。
REFUSED_ON_MAXIMA = [
    ("ja", "c を電子の消滅演算子とする"),
    ("ja", "c を電子の生成演算子とする"),
    ("ja", "a を光子の生成演算子とする"),
    ("ja", "c をフェルミオン演算子とする"),
    ("ja", "a をボゾン演算子とする"),
    ("ja", "c を消滅演算子とする"),
    ("en", "define c as an electron annihilation operator"),
    ("en", "define c as a fermion operator"),
    ("en", "define a as a boson operator"),
    ("en", "define c as an annihilation operator"),
    ("en", "let c be a fermion operator"),
]
#: Maxima セッションでも**通る**べき綴り（CAS-bypass の実装がある）。
ACCEPTED_ON_MAXIMA = [
    ("ja", "S をスピン演算子とする"),
    ("ja", "S をパウリ行列とする"),
    ("ja", "S を電子のスピン演算子とする"),
    ("en", "define S as a spin operator"),
    ("en", "define S as a Pauli matrix"),
]


def _ask(lang, text):
    nb.start_engine(title="opdecl", mode="standard", language=lang,
                    domains=["magnetic_resonance"])
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["magnetic_resonance"], primary_backend="maxima")
        resp = asyncio.get_event_loop().run_until_complete(
            run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                eng, sup))
        qty = (eng.symbol_context.get_quantity("c")
               or eng.symbol_context.get_quantity("a")
               or eng.symbol_context.get_quantity("S"))
        return resp, qty
    finally:
        nb.stop_engine()


class TestASentenceThatSaysOperatorNeverDeclaresAScalar:
    @pytest.mark.parametrize("lang,text", REFUSED_ON_MAXIMA)
    def test_refused_with_a_reason(self, lang, text):
        resp, qty = _ask(lang, text)
        assert resp.get("status") == "failed", (text, resp.get("status"),
                                                resp.get("operation"))
        assert resp.get("operation") != "define_scalar_parameter", text
        assert qty is None, (text, "スカラーとして登録されている")
        err = str(resp.get("error") or "")
        # 理由（このバックエンドでは扱えない）と対処（Julia）を名指す。
        assert ("Julia" in err), (text, err)

    @pytest.mark.parametrize("lang,text", ACCEPTED_ON_MAXIMA)
    def test_spin_is_still_accepted(self, lang, text):
        """**変えてはいけない挙動。** スピンは `spin_algebra` が CAS-bypass で
        実装されているので Maxima でも受ける。"""
        resp, qty = _ask(lang, text)
        assert resp.get("operation") == "define_spin_operator", (text, resp)
        assert getattr(qty, "kind", None) == "operator", (text, qty)

    @pytest.mark.parametrize("lang,text", [
        ("ja", "E_kin をスカラーとする"),
        ("ja", "E_kin を１電子エネルギーとする"),
        ("en", "define k as a scalar"),
        ("en", "define A as a matrix"),
    ])
    def test_an_ordinary_declaration_is_untouched(self, lang, text):
        resp, _ = _ask(lang, text)
        assert resp.get("status") == "confirmed", (text, resp)


class TestTheRefusalNamesOnlyWhatWasWritten:
    """研究者が書いていない統計を名指さない（利用者の指摘、2026-09-06）。"""

    def test_a_bare_ladder_operator_names_the_family(self):
        resp, _ = _ask("ja", "c を消滅演算子とする")
        err = str(resp.get("error") or "")
        assert "第二量子化の演算子" in err, err
        assert "フェルミオン" not in err, ("書かれていない統計を名指している", err)

    def test_a_written_statistics_is_named(self):
        resp, _ = _ask("ja", "c をフェルミオン演算子とする")
        assert "フェルミオン" in str(resp.get("error") or "")


class TestOnlyJuliaAsksForTheStatistics:
    """Maxima では答えても計算できないので訊かない（利用者の指摘）。"""

    def test_julia_asks(self):
        h = JuliaBackendHandler(None)
        r = h.extra_pre_phase1("c を消滅演算子とする", "ja")
        assert r is not None and not r.ok
        assert "フェルミオン" in str(r.error) and "ボソン" in str(r.error), r.error

    def test_maxima_refuses_without_asking(self):
        resp, _ = _ask("ja", "c を消滅演算子とする")
        err = str(resp.get("error") or "")
        assert "Julia" in err
        assert "ボソン" not in err, ("Maxima で統計を訊いている", err)


class TestTheReadingIsShared:
    """読みは文法なので共有層に置く（`read_spin_operator_declaration` と同じ隣）。

    Julia ハンドラの中にあったので、**Maxima セッションでは誰も訊いていなかった**。
    """

    @pytest.mark.parametrize("text,want", [
        ("c を電子の消滅演算子とする", "define_fermion_operator"),
        ("c をフェルミオン演算子とする", "define_fermion_operator"),
        ("a をボゾン演算子とする", "define_boson_operator"),
        ("c を消滅演算子とする", "define_fermion_operator"),
        ("define c as an electron annihilation operator", "define_fermion_operator"),
        ("define a as a boson operator", "define_boson_operator"),
    ])
    def test_it_names_a_particle_operator_declaration(self, text, want):
        assert names_particle_operator_declaration(text) == want, text

    @pytest.mark.parametrize("text", [
        "E_kin をスカラーとする", "A を行列とする", "S をスピン演算子とする",
        "define k as a scalar", "define A as a matrix",
    ])
    def test_it_declines_everything_else(self, text):
        assert names_particle_operator_declaration(text) is None, text


class TestUnsupportedIsNotTheSameAsUnknown:
    """「そんな宣言は無い」と「このバックエンドでは動かせない」は別の答え。"""

    def test_unsupported_here(self):
        assert (declaration_unsupported_here("fermion operator", "maxima")
                == "define_fermion_operator")
        assert declaration_unsupported_here("fermion operator", "julia_full") is None

    def test_unknown_is_still_none(self):
        assert declaration_unsupported_here("１電子エネルギー", "maxima") is None
        assert declaration_unsupported_here("スカラー", "maxima") is None


class TestAnArticleIsNotAParticleName:
    def test_a_bare_annihilation_operator_names_no_particle(self):
        """`define c as an annihilation operator` の `an` が粒子として
        捕まっていた（実測 2026-09-06）。"""
        assert nlp("particle_operator_decl", "en").search(
            "define c as an annihilation operator") is None

    @pytest.mark.parametrize("text,particle", [
        ("define c as an electron annihilation operator", "electron"),
        ("define c as the creation operator for photon", "photon"),
    ])
    def test_a_real_particle_is_still_read(self, text, particle):
        m = nlp("particle_operator_decl", "en").search(text)
        assert m is not None, text
        assert particle in str(m.groupdict().values()), (text, m.groupdict())
