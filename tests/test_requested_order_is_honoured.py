r"""書かれた次数が結果を変えないなら、黙らない。

実測 2026-09-06。F-1 の3経路が、研究者の書いた次数を**読み取ったうえで捨てるか、
範囲へ黙ってクランプ**していた。どれも `completed`、`warnings` は `None`:

| 打った文 | 返っていたもの | 本当の上限 |
|---|---|---|
| `…S^x の8次までのモーメント` | `M_0, M_2`（1〜8 どれでも同じ） | 2次 |
| `…S^x の5段の連分数` | `b_1^2`（1〜5 どれでも同じ） | 1段 |
| `…高温展開を6次まで` | `⟨H²⟩⟨H³⟩⟨H⁴⟩`（`max(2,min(4,n))`） | 2〜4次 |

**読みは正しかった**——`_parse_spin_spectral` は `depth=8` をちゃんと取れている。
捨てていたのは計算側で、一般式の枝は `general_low_moments`（名前が low）／
`general_b1_squared` を呼ぶだけだった。

**返っていたのは「別の問いへの正しい答え」である。** 研究者は4次モーメントを見たくて
打ち、2次の式を受け取って先へ進む——このプロジェクトがいちばん警戒している形。
だから警告ではなく、`operand_guards` と同じ「使う地点で理由付きで断る」に倣った。

**断り文の案内する道は実測してある**（[[拒否の理由が入力に当たっているか測る]]）:
格子を具体的に書けば次数は効く——3サイト鎖で 2/4/6/8 がそのとおり返る。

**コーパスとシナリオは上限内**（`高温展開を4次まで`＝上限、一般モーメントは `2次`）
なので、この断りは既存の実例を1つも止めない——実装前に測って確認した。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.i18n import t
from notebook.runtime import spin_algebra as _spa
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.requested_order import (ENERGY_MOMENT_RANGE,
                                              GENERAL_CONTFRAC_MAX,
                                              GENERAL_MOMENT_MAX,
                                              energy_moment_refusal,
                                              general_spin_moment_refusal)
from ui.interpret_f1 import _parse_energy_moment, _parse_spin_spectral
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline

DOM = ["condensed_matter"]

#: 断られるべき綴り（書かれた次数がその枝で届かない）。
REFUSED = [
    ("ja", "ハイゼンベルク模型で一般の格子の S^x の4次までのモーメントを求めて"),
    ("ja", "ハイゼンベルク模型で一般の格子の S^x の8次までのモーメントを求めて"),
    ("ja", "ハイゼンベルク模型で一般の格子の S^x の3段の連分数を求めて"),
    ("ja", "ハイゼンベルク模型の高温展開を一般式で6次まで求めて"),
    ("ja", "ハイゼンベルク模型の高温展開を一般式で1次まで求めて"),
    ("en", "Find the moments up to 8th order of S^x for the Heisenberg model on a general lattice"),
    ("en", "Find the high-temperature expansion of the Heisenberg model in general form up to 6th order"),
]
#: 通り続けるべき綴り（上限内、または次数を書いていない）。
ALLOWED = [
    ("ja", "ハイゼンベルク模型で一般の格子の S^x のモーメントを求めて"),
    ("ja", "ハイゼンベルク模型で一般の格子の S^x の2次モーメントを求めて"),
    ("ja", "ハイゼンベルク模型で一般の格子の S^x の連分数を求めて"),
    ("ja", "ハイゼンベルク模型の高温展開を一般式で求めて"),
    ("ja", "ハイゼンベルク模型の高温展開を一般式で4次まで求めて"),
    ("ja", "ハイゼンベルク模型 3サイト鎖で S^x_2 の4次までのモーメントを求めて"),
    ("en", "Find the high-temperature expansion of the Heisenberg model in general form up to 4th order"),
]


def _ask(lang, text):
    nb.start_engine(title="ro", mode="standard", language=lang, domains=DOM)
    try:
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=DOM, primary_backend="maxima")
        return asyncio.get_event_loop().run_until_complete(
            run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                eng, sup))
    finally:
        nb.stop_engine()


class TestAnUnreachableOrderIsRefused:
    @pytest.mark.parametrize("lang,text", REFUSED)
    def test_it_fails_with_a_reason(self, lang, text):
        r = _ask(lang, text)
        assert r.get("status") == "failed", (text, r.get("latex"))
        msg = r.get("error") or ""
        # **理由が研究者の言葉であること**——生の CAS 文言でも内部名でもない。
        assert msg and "Traceback" not in msg, msg
        assert "_nb" not in msg and "depth" not in msg, msg

    @pytest.mark.parametrize("lang,text", ALLOWED)
    def test_what_was_reachable_still_computes(self, lang, text):
        r = _ask(lang, text)
        assert r.get("status") == "completed", (text, r.get("error"))


class TestTheRefusalNamesARoadThatWorks:
    r"""案内した回避策も測る——格子を具体的に書けば次数は効く。

    **この検査は 2026-09-10 に書き直した。** 元は `n次まで` が $M_{2n}$ を返すことを
    固定していた——実装から書いた検査で、断り文と表示のラベルが言っている
    「次数」とは別の量を正しいと宣言していた（起票 MOMENT-ORDER-MEANING）。
    断り文が名指す道は `4次まで` なので、それが $M_4$ を返してはじめて
    「名指した道は通る」と言える。
    """

    @pytest.mark.parametrize("n,highest,absent", [(2, "M_{2}", "M_{4}"),
                                                  (4, "M_{4}", "M_{6}")])
    def test_a_concrete_lattice_honours_the_order(self, n, highest, absent):
        r = _ask("ja", f"ハイゼンベルク模型 3サイト鎖で S^x_2 の{n}次までのモーメントを求めて")
        assert r.get("status") == "completed", r.get("error")
        body = (r.get("registered_expressions") or [{}])[-1].get("latex") or ""
        assert highest in body, body
        assert absent not in body, f"{n}次までなのに {absent} が入っている: {body}"

    @pytest.mark.parametrize("n,highest", [(1, "M_{2}"), (2, "M_{4}")])
    def test_a_written_depth_is_recursion_steps(self, n, highest):
        """`段` は段数——`次` とは別の量なので、同じ数でも結果が違う。"""
        r = _ask("ja", f"ハイゼンベルク模型 3サイト鎖で S^x_2 の{n}段までのモーメントを求めて")
        assert r.get("status") == "completed", r.get("error")
        body = (r.get("registered_expressions") or [{}])[-1].get("latex") or ""
        assert highest in body, body

    @pytest.mark.parametrize("n,highest", [(1, "M_{2}"), (2, "M_{4}")])
    def test_the_prefix_depth_spelling_is_read(self, n, highest):
        """`深さ2` —— 日本語は数の**前**に置く。直す前は数の後ろ（`2深さ`）しか
        受けておらず、**誰も打てない綴り**だった（既定の2段に落ちていた）。"""
        r = _ask("ja", f"ハイゼンベルク模型 3サイト鎖で S^x_2 の深さ{n}のモーメントを求めて")
        assert r.get("status") == "completed", r.get("error")
        body = (r.get("registered_expressions") or [{}])[-1].get("latex") or ""
        assert highest in body, body

    @pytest.mark.parametrize("lang,text,highest", [
        ("en", "Find the moments up to 2nd order of S^x_2 for the Heisenberg "
               "model on a 3-site chain", "M_{2}"),
        ("en", "Find the moments up to order 4 of S^x_2 for the Heisenberg "
               "model on a 3-site chain", "M_{4}"),
        ("en", "Find the moments up to depth 2 of S^x_2 for the Heisenberg "
               "model on a 3-site chain", "M_{4}"),
    ])
    def test_english_reads_the_same_two_units(self, lang, text, highest):
        """英語側は最初から `order` と `depth` を別の枝で綴っていた——
        **判別子はデータに在って、読む側が捨てていた。**"""
        r = _ask(lang, text)
        assert r.get("status") == "completed", r.get("error")
        body = (r.get("registered_expressions") or [{}])[-1].get("latex") or ""
        assert highest in body, body


class TestTheReadingWasNeverTheProblem:
    r"""次数は**読めていた**——捨てていたのは計算側である。"""

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_the_parser_reads_the_written_order(self, n):
        p = _parse_spin_spectral(
            f"ハイゼンベルク模型で一般の格子の S^x の{n}次までのモーメントを求めて", "ja")
        assert p and p["requested"] == n, p

    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_a_written_depth_is_reported_in_the_mode_s_units(self, n):
        """断りが比べるのは**その枝の単位**（モーメントなら次数）。`段` で書かれた
        ら次数に直してから比べないと、上限 2 と段数 2 を比べることになる。"""
        p = _parse_spin_spectral(
            f"ハイゼンベルク模型で一般の格子の S^x の{n}段までのモーメントを求めて", "ja")
        assert p and p["requested"] == 2 * n and p["depth"] == n, p

    def test_no_written_order_is_not_a_written_order(self):
        """次数を書いていない文は断らない（既定で動く）。"""
        p = _parse_spin_spectral(
            "ハイゼンベルク模型で一般の格子の S^x のモーメントを求めて", "ja")
        assert p and p["requested"] is None, p
        q = _parse_energy_moment("ハイゼンベルク模型の高温展開を一般式で求めて", "ja")
        assert q and q["upto_written"] is None, q

    @pytest.mark.parametrize("n", [1, 4, 6])
    def test_the_energy_parser_keeps_the_raw_value(self, n):
        """丸めた値は下流のために残し、**生の値**も持ち回る。"""
        q = _parse_energy_moment(
            f"ハイゼンベルク模型の高温展開を一般式で{n}次まで求めて", "ja")
        assert q["upto_written"] == n and q["upto"] == max(2, min(4, n)), q


class TestTheContinuedFractionCountsSteps:
    r"""**連分数だけ `次` と `段` が同じ数になるのは、対象が違うから。**

    連分数の「n次近似」は係数 $b_1^2 \dots b_n^2$ を持つ段のことで、モーメントの
    「n次」は $M_n$ そのものを指す。言語が不整合なのではなく、数えている物が違う。
    """

    @pytest.mark.parametrize("spelling", ["2次までの", "2段までの", "深さ2の"])
    def test_every_spelling_gives_two_levels(self, spelling):
        p = _parse_spin_spectral(
            f"ハイゼンベルク模型 4サイト鎖で S^x_2 の{spelling}連分数を求めて", "ja")
        assert p and p["mode"] == "contfrac", p
        assert p["depth"] == 2 and p["requested"] == 2, p

    def test_the_moment_family_does_not_agree_with_it(self):
        """同じ `2次` が、モーメントでは 1 段・連分数では 2 段になる。
        **これが起票の中身**——同じ語が別の量を指すこと自体は正しく、
        直す前は片方（モーメント）が対象を取り違えていた。"""
        m = _parse_spin_spectral(
            "ハイゼンベルク模型 4サイト鎖で S^x_2 の2次までのモーメントを求めて", "ja")
        c = _parse_spin_spectral(
            "ハイゼンベルク模型 4サイト鎖で S^x_2 の2次までの連分数を求めて", "ja")
        assert m["depth"] == 1 and c["depth"] == 2, (m, c)


class TestTheHighTemperatureBranchIsUnchanged:
    r"""⟨H^n⟩ には recursion が無いので、`次` と `段` は同じ n を指す。"""

    @pytest.mark.parametrize("word", ["次", "段"])
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_the_written_number_is_the_moment_order(self, word, n):
        q = _parse_energy_moment(
            f"ハイゼンベルク模型 3サイト鎖の高温展開を{n}{word}まで求めて", "ja")
        assert q and q["upto_written"] == n and q["upto"] == n, q


class TestTheCapsAreOneTable:
    @pytest.mark.parametrize("lang", ["ja", "en"])
    @pytest.mark.parametrize("mode,cap", [("moments", GENERAL_MOMENT_MAX),
                                          ("contfrac", GENERAL_CONTFRAC_MAX)])
    def test_at_the_cap_it_passes_and_above_it_refuses(self, lang, mode, cap):
        assert general_spin_moment_refusal(mode, cap, lang) is None
        why = general_spin_moment_refusal(mode, cap + 1, lang)
        assert why and str(cap) in why and str(cap + 1) in why, why

    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_the_energy_range_is_inclusive(self, lang):
        lo, hi = ENERGY_MOMENT_RANGE
        for n in range(lo, hi + 1):
            assert energy_moment_refusal(n, lang) is None, n
        for n in (lo - 1, hi + 1):
            why = energy_moment_refusal(n, lang)
            assert why and str(n) in why, (n, why)

    @pytest.mark.parametrize("key", ["order.general_moment_beyond",
                                     "order.general_contfrac_beyond",
                                     "order.energy_moment_out_of_range"])
    @pytest.mark.parametrize("lang", ["ja", "en"])
    def test_every_reason_is_translated(self, key, lang):
        got = t(key, lang, requested=9, cap=2, lower=2, upper=4)
        assert got != key and "{" not in got, (key, lang, got)


class TestTheCapsAreMeasuredNotAsserted:
    r"""**上限は計算の性質なので、計算を測って固定する。**

    最初に書いたテストは断りの表（`GENERAL_MOMENT_MAX` ほか）を読んでいたので、
    定数を1つ上げる変異が3種とも落ちなかった——境界がテストと一緒に動くからである。
    ここは `spin_algebra` に**直接**訊く。
    """

    def test_the_general_moment_formula_stops_at_M2(self):
        got = _spa.general_low_moments("heisenberg", "x")
        assert len(got) == GENERAL_MOMENT_MAX, got   # (M_0, M_2)

    def test_the_general_continued_fraction_stops_at_b1(self):
        """一般式で閉じているのは b_1^2 ひとつ——関数がそれ1本しか返さない。"""
        b1 = _spa.general_b1_squared("heisenberg", "x")
        assert b1 is not None and not isinstance(b1, (list, tuple)), b1
        assert GENERAL_CONTFRAC_MAX == 1

    @pytest.mark.parametrize("n,want", [(3, ["H^2", "H^3"]),
                                        (4, ["H^2", "H^3", "H^4"]),
                                        (5, ["H^2", "H^3", "H^4"]),
                                        (6, ["H^2", "H^3", "H^4"])])
    def test_the_energy_expansion_stops_at_H4(self, n, want):
        """`upto=5` を渡しても4次までしか出ない——だから上限は 4 である。"""
        import re
        got = re.findall(r"H\^\d+", _spa.energy_moment_general_latex("heisenberg", upto=n))
        assert got == want, (n, got)
        assert ENERGY_MOMENT_RANGE[1] == 4
