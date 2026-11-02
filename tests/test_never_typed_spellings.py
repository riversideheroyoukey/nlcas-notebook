r"""**文法が受けると言っているのに、誰も一度も打っていない綴り** —— 打ってみる。

実測 2026-09-10（CORPUS-WORKING-ONLY、`scripts/nl_spelling_coverage.py`）。
`i18n/nl_patterns.json` が受けると言っている綴りは、直す前で 770 件。そのうち

| 書かれている先 | 件数 |
|---|---|
| 指示コーパス（523行） | 275 (35%) |
| シナリオ `docs/scenarios/*.md` | 599 (77%) |
| テストのソースにリテラルとして | 687 (88%) |
| **どこにも無い** | **61 (7%)** |

指示コーパスはシナリオから収穫されており、シナリオは**通る指示**で書かれている。
だからコーパスの差分が 0 でも、その綴りが 0 行なら「安全」ではなく
「**測っていない**」——四十一・四十二問目でこれを2問連続で読み違えた。
記法コーパス（`tests/corpus/notation_corpus.json`）の
「通った式でなく打てる綴りを集める」と同じ形である。

このファイルは、その 61 件を**指示として実際に打つ**。書き方は記法コーパスと同じ
「**1族＝1つの意味 × その綴り全部**」で、各族に**既に打たれている綴り（対照）**を
1つ置き、一度も打たれていない綴りが**同じ操作・同じ引数・同じ結果**に届くことを
見る。届かないのではなく**違うものに届く**のが、この系でいちばん高くつく壊れ方
だからである。

**測る層は2つ、弱いほうを明示する。**

* `TestSpellingsThroughThePipeline` —— 研究者が打つ経路
  （`run_interpret_pipeline` ＋ LLM 無しの supervisor）。これが本体。
* `TestSpellingsAtThePatternLayer` —— パターンだけ。LLM が要る（`plan_*`）、
  ファイルを書く（格子図・handoff）、あるいは長い文脈が要る族はここに置く。
  **パターンが当たることは、その綴りが正しく読まれることを意味しない**
  （入口から測ると別の枝へ逃げる——[[feedback-verify-the-guard-catches-its-own-defect]] ④）。

**この 61 件のうち、壊れていたのは2件だった。** ひとつは `thrice`——
`ordinal_token` に載っていたのに `order_clause[en]` が `twice|once` を自前で
綴っていたので、`differentiate E1 with respect to x thrice` が **1階** を
`completed` で返していた（四十八問目、`tests/test_ordinal_is_one_reading.py`）。
もうひとつは `振っ`——表に在るのに `振って`・`振る`・`振り` のどれも当たらず、
**研究者が打てる綴りが1つも無かった**（`TestTheSweepVerbInflects`）。
残り 59 件は正しく動いていた——**それも測定の結果**であって、測るまでは
どちらか分からなかった。

四十八問目の3件目（`十次まで展開`）は**この一覧には出ない**。`十` は1文字で、
綴りの取り出しが日本語を2文字からしか数えないためである（道具の限界。
`scripts/nl_spelling_coverage.py` の docstring に挙げてある）。見つけたのは
同じ道具の**もう一方の測定**——日英で片側しか引いていない言語中立 fragment。
"""
import asyncio

import pytest

import ui.notebook_bridge as nb
from notebook.runtime.llm_supervisor import LLMSupervisor
from notebook.runtime.nl_patterns import fragment, nlp
from notebook.runtime.operation_validator import OperationValidator
from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline


def _run(domain, lang, setup, text):
    """1文を、研究者が通る経路で走らせて「何に届いたか」を返す。"""
    async def go():
        nb.start_engine(title="spellings", mode="standard", language=lang,
                        domains=[domain])
        try:
            engine = nb.get_engine()
            # `scripts/generate_scenario_examples._sup` と同じ作り方——
            # backend はエンジンのものに合わせる（`start_engine` は第２量子化
            # ドメインで julia_full を選ぶ）。
            sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                                domains=[domain],
                                primary_backend=engine.session.primary_backend)
            for line in setup:
                await run_interpret_pipeline(
                    InterpretRequest(tokens=[InputToken(type="text", value=line)]),
                    engine, sup)
            before = set(engine.steps)
            resp = await run_interpret_pipeline(
                InterpretRequest(tokens=[InputToken(type="text", value=text)]),
                engine, sup)
            new = [s for sid, s in engine.steps.items() if sid not in before]
            op = next((s.operation for s in new
                       if getattr(s, "kind", None) == "llm_supervision" and s.operation),
                      None)
            op = op or next((s.operation for s in new if s.operation), None)
            args = next((dict(s.operation_args or {}) for s in new
                         if s.operation), {})
            routes = [s.route for s in new if getattr(s, "route", None)]
            created = [(engine.expressions[e].canonical, engine.expressions[e].latex)
                       for s in new for e in (s.output_refs or [])
                       if e in engine.expressions]
            return {
                "status": resp.get("status"),
                "operation": resp.get("operation") or op,
                "route": resp.get("route") or (routes[0] if routes else None),
                "args": {k: v for k, v in args.items() if not k.startswith("_")},
                "created": created,
                "candidates": [c.get("operation") for c in (resp.get("candidates") or [])],
                "error": resp.get("error"),
            }
        finally:
            nb.stop_engine()

    return asyncio.run(go())


def _same(a, b):
    """2つの読みが「同じものに届いた」か —— 届き先・引数・作った式。"""
    return (a["status"], a["operation"], a["route"], a["args"],
            a["created"], a["candidates"], a["error"]) == \
           (b["status"], b["operation"], b["route"], b["args"],
            b["created"], b["candidates"], b["error"])


# --------------------------------------------------------------------------
#: 1族 = (ドメイン, 言語, 前置きの入力, 対照の綴り, 一度も打たれていない綴りたち)。
#: 対照は**既にどこかで打たれている**綴りで、族の意味を固定する。
FAMILIES = [
    # --- F-1 格子 -----------------------------------------------------------
    ("f1_kagome", "quantum_mechanics", "ja",
     [r"H = -t \sum_{\langle i j \rangle} c'_i c_j"],
     "H のカゴメ格子で分散関係を求めて",
     ["H のかごめ格子で分散関係を求めて", "H の籠目格子で分散関係を求めて"]),
    ("f1_lieb", "quantum_mechanics", "ja",
     [r"H = -t \sum_{\langle i j \rangle} c'_i c_j"],
     "H の Lieb 格子で分散関係を求めて",
     ["H のリーブ格子で分散関係を求めて"]),
    ("f1_nnn", "quantum_mechanics", "ja",
     [r"H = -t \sum_{\langle i j \rangle} c'_i c_j"],
     "H の三角格子で次近接 -t' を含めて分散関係を求めて",
     ["H の三角格子で第二近接 -t' を含めて分散関係を求めて"]),
    ("f1_onsite", "quantum_mechanics", "ja",
     [r"H = -t \sum_{\langle i j \rangle} c'_i c_j"],
     r"H のハニカム格子で質量 \Delta を含めて分散関係を求めて",
     [r"H のハニカム格子でサブ格子ポテンシャル \Delta を含めて分散関係を求めて"]),
    ("f1_crystal_kw", "quantum_mechanics", "ja", [],
     "次のボンドで結晶の分散関係を求めて: (0,1,(0),v),(0,1,(-1),w)",
     ["次のボンドでクリスタルの分散関係を求めて: (0,1,(0),v),(0,1,(-1),w)"]),
    ("f1_boundary_periodic", "statistical_mechanics", "ja", [],
     "Sz_i Sz_j の最近接和を 4 サイト 周期境界で展開して",
     ["Sz_i Sz_j の最近接和を 4 サイト トーラスで展開して"]),
    ("f1_boundary_cylinder", "statistical_mechanics", "ja", [],
     "Sz_i Sz_j の最近接和を 6 サイト 円筒で展開して",
     ["Sz_i Sz_j の最近接和を 6 サイト シリンダーで展開して"]),
    ("f1_ns_allpairs", "statistical_mechanics", "ja", [],
     "Sz_i Sz_j の全ペア和を 4 サイトで展開して",
     ["Sz_i Sz_j の全対和を 4 サイトで展開して",
      "Sz_i Sz_j の全ペアの和を 4 サイトで展開して"]),
    ("f1_degree", "statistical_mechanics", "ja", [],
     "グラフ (1,2),(2,3),(1,3) の次数列を求めて",
     ["グラフ (1,2),(2,3),(1,3) の配位数列を求めて"]),
    ("f1_terminator", "statistical_mechanics", "ja", [],
     "平方根終端の連続スペクトル密度を求めて",
     ["半円則のスペクトル密度を求めて", "半円スペクトルの密度を求めて"]),
    ("f1_general_spin", "statistical_mechanics", "ja", [],
     "ハイゼンベルク模型で一般スピンの一般の格子の S^x の2次モーメントを求めて",
     ["ハイゼンベルク模型で任意のスピンの一般の格子の S^x の2次モーメントを求めて"]),

    # --- F-9 角運動量（研究者が打つ量子数は6〜9個。誤読は「もっともらしい
    #     間違った値」で、どこにもエラーが出ない——CLAUDE.md F-9）------------
    ("angular_clebsch_gordan", "atomic_physics", "ja", [],
     "j1=1/2, m1=1/2, j2=1/2, m2=-1/2, J=1, M=0 のCG係数を計算して",
     ["j1=1/2, m1=1/2, j2=1/2, m2=-1/2, J=1, M=0 のクレブシュ・ゴルダン係数を計算して",
      "j1=1/2, m1=1/2, j2=1/2, m2=-1/2, J=1, M=0 のクレブシュ-ゴルダン係数を計算して"]),
    ("angular_wigner_eckart", "atomic_physics", "ja", [],
     "j=1/2, m=1/2, k=1, q=0, j'=3/2, m'=1/2 のWigner-Eckartの定理を書き下して",
     ["j=1/2, m=1/2, k=1, q=0, j'=3/2, m'=1/2 のウィグナー・エッカルトの定理を書き下して"]),
    ("angular_coupled_state", "atomic_physics", "ja", [],
     "j1=1/2, j2=1/2, J=1, M=0 を非結合基底で展開して",
     ["j1=1/2, j2=1/2, J=1, M=0 を直積基底で展開して",
      "j1=1/2, j2=1/2, J=1, M=0 を積基底で展開して",
      "j1=1/2, j2=1/2, J=1, M=0 を非結合状態で展開して"]),
    ("angular_lande_g", "atomic_physics", "ja", [],
     "L=1, S=1/2, J=3/2 の g 因子を求めて",
     ["L=1, S=1/2, J=3/2 の g ファクターを求めて"]),
    ("angular_couple", "atomic_physics", "ja", [],
     "I=3/2 と S=1/2 を合成して",
     ["I=3/2 と S=1/2 の角運動量を合成して"]),

    # --- スピン期待値 -------------------------------------------------------
    ("spin_expectation_ja", "statistical_mechanics", "ja",
     ["H = J S^z_1 S^z_2"], "E1 の期待値を求めて", ["E1 の熱平均を求めて"]),
    ("spin_expectation_en", "statistical_mechanics", "en",
     ["H = J S^z_1 S^z_2"], "the expectation value of E1",
     ["the thermal average of E1"]),
    ("spin_expectation_order", "statistical_mechanics", "en",
     ["H = J S^z_1 S^z_2"], "the 8th moment of E1", ["the eighth moment of E1"]),

    # --- 一般操作 -----------------------------------------------------------
    ("both_sides_subtract", "classical_mechanics", "ja", ["y = a x + b"],
     "E1 の両辺から b を引いて",
     ["E1 の両辺から b を減じて", "E1 の両辺から b を減算して"]),
    ("name_assign", "classical_mechanics", "ja", ["y = a x + b"],
     "E1 を K と置いて", ["E1 を K と命名して", "E1 を K に格納して"]),
    ("combine", "classical_mechanics", "ja", ["y = a x + b", "z = c x"],
     "E1 と E2 を引く", ["E1 と E2 をひく", "E1 と E2 をマイナス"]),
    ("subst_replace_alt", "classical_mechanics", "ja", ["y = a x + b"],
     "E1 の a を 2 で置き換えて", ["E1 の a を 2 と置換して"]),
    ("redeclare", "quantum_mechanics", "ja", ["x をスカラーとする"],
     "x をベクトルとして宣言し直して", ["x をベクトルとして宣言しなおして"]),

    # --- Bogoliubov / 計量 ---------------------------------------------------
    ("verify_canon", "quantum_mechanics", "ja", ["b = u a + v a'"],
     "E1 は正準変換ですか", ["E1 はユニタリですか"]),
    ("list_sep_ja", "quantum_mechanics", "ja", ["b = u a + v a'", "c = u a + v a'"],
     "E1 および E2 は正準変換ですか", ["E1 並びに E2 は正準変換ですか"]),
    ("metric_constraint", "quantum_mechanics", "ja",
     ["b = u a + v a'", "c = u a + v a'"],
     "E1, E2 の計量保存条件を求めて", ["E1, E2 の計量を保つ条件を求めて"]),
    ("metric_constraint_en", "quantum_mechanics", "en",
     ["b = u a + v a'", "c = u a + v a'"],
     "derive the metric conditions for E1, E2",
     ["derive the orthonormalization conditions for E1, E2"]),
    ("canon_constraint", "quantum_mechanics", "ja",
     ["b = u a + v a'", "c = u a + v a'"],
     "E1, E2 の正準性の条件を求めて", ["E1, E2 の交換関係を保つ条件を求めて"]),

    # --- 極値・簡約 ---------------------------------------------------------
    ("evaluate_extremum", "operations_research", "ja", [r"J = \max_i x_i"],
     "E1 を計算して", ["この最大値を計算して"]),
    ("evaluate_extremum_min", "operations_research", "ja", [r"J = \min_i x_i"],
     "E1 を計算して", ["この最小値を計算して"]),

    # --- 解析力学 -----------------------------------------------------------
    ("qdot_diff_kw", "analytical_mechanics", "ja",
     [r"L = (1/2) m \dot{q}^2 - V(q)"],
     r"E1 を \dot{q} で微分して", ["E1 を一般速度で微分して"]),
]


@pytest.mark.parametrize(
    "family,domain,lang,setup,control,spellings",
    FAMILIES, ids=[f[0] for f in FAMILIES])
class TestSpellingsThroughThePipeline:
    """一度も打たれていない綴りが、対照と**同じものに届く**こと。"""

    def test_the_control_spelling_is_read(self, family, domain, lang, setup,
                                          control, spellings):
        """対照が読まれること —— これが落ちたら族の意味が変わっている。

        対照を置く理由: 綴りだけを並べると「両方とも同じように壊れている」を
        「一致している」と読んでしまう。族の意味は対照が固定する。
        """
        got = _run(domain, lang, setup, control)
        # 「解釈できません」だけは対照として失格——族の意味を固定できない。
        # 理由つきの拒否は合格（拒否も読みの結果である）。
        assert not str(got["error"] or "").startswith("Could not interpret"), \
            (family, control, got)
        assert (got["route"] or got["operation"] or got["candidates"]
                or got["created"]), (family, control, got)

    def test_never_typed_spellings_reach_the_same_thing(
            self, family, domain, lang, setup, control, spellings):
        base = _run(domain, lang, setup, control)
        for spelling in spellings:
            got = _run(domain, lang, setup, spelling)
            assert _same(base, got), (family, spelling, base, got)


# --------------------------------------------------------------------------
#: パターン層でしか測れない族。**弱い**測定であることを承知で置く:
#: LLM が要る（`plan_*`）／ファイルを書く（格子図・handoff）／語順の文脈が要る。
#: それでも「その綴りが表に載っているだけで誰も打っていない」状態よりは強い。
PATTERN_ONLY = [
    ("f1_save_fig_noun", "ja", "格子図", ["格子の図", "ラティス図"]),
    ("f1_lattice_mention_safe", "ja", "副格子", ["サブ格子", "格子の図"]),
    ("f1_graph_allpairs", "ja", "全ペア", ["完全グラフ"]),
    ("matrix_unary_diagonalize", "ja", "対角化", ["モード行列"]),
    ("relabel_index", "ja", "E1 の k を -k に置き換え",
     ["E1 の k を -k に指標変換", "E1 の k を -k にリラベル"]),
    # 辞書形も表に在る。ファイルを書く操作なので、ここはパターン層で測る。
    ("scalar_handoff_nl", "ja", "E1 を Maxima セッションに渡して",
     ["E1 を Maxima セッションに引き継いで", "E1 を Maxima セッションに持ち出して",
      "E1 を Maxima セッションに引き継ぐ", "E1 を Maxima セッションに持ち出す"]),
    ("numeric_fork_sweep_nl", "ja", "x を 0 から 1 まで掃引して",
     ["x を 0 から 1 まで振って"]),
    ("plan_eigensystem", "ja", "固有状態", ["固有値と固有ベクトル"]),
    ("plan_auto_select", "ja", "解法を自動で選んで", ["解き方を自動で選んで"]),
    ("plan_candidates", "ja", "解き方の候補を挙げて", ["解き方の候補を挙げて"]),
    ("plan_freeform", "ja", "手順を計画して", ["手順を計画して"]),
]


@pytest.mark.parametrize("key,lang,control,spellings", PATTERN_ONLY,
                         ids=[p[0] for p in PATTERN_ONLY])
class TestSpellingsAtThePatternLayer:
    def test_the_control_matches(self, key, lang, control, spellings):
        assert nlp(key, lang).search(control), (key, control)

    def test_the_never_typed_spellings_match_too(self, key, lang, control, spellings):
        for spelling in spellings:
            assert nlp(key, lang).search(spelling), (key, spelling)


class TestTheAmbiguousWordSaysItIsAmbiguous:
    """`きれい` は「同じものに届く」族ではない —— **曖昧だと言う**族。

    対照（`簡約`）は簡約に届き、`きれい` は候補を出して研究者に訊く。
    ここを「同じに届くこと」で測ると、族の意味のほうを取り違える。
    """

    def test_the_control_reaches_simplify(self):
        got = _run("classical_mechanics", "ja", ["y = a x + b"], "E1 を簡約して")
        assert got["operation"] == "simplify", got

    def test_the_never_typed_spelling_asks(self):
        got = _run("classical_mechanics", "ja", ["y = a x + b"], "E1 をきれいにして")
        assert got["status"] == "needs_confirmation", got
        assert "simplify" in got["candidates"], got


class TestTheSweepVerbInflects:
    r"""`振る` は五段、`掃引` はサ変 —— 同じ語尾を共有できない。

    直す前の綴りは ``(?:掃引|振っ)\s*(?:評価|計算)?\s*(?:して|する)?\s*$`` で、
    `掃引して` / `掃引する` は通るのに **`振って` はどの形も当たらなかった**
    （`振っ` のあとに来られるのは `して` か `評価`／`計算` だけ）。表には
    `振っ` と書いてあるのに、研究者が打てる綴りは1つも無い、という状態。
    活用の表は動詞の型ごとに持つ（`ja_oku_infl` が置く型、`ja_ru_infl` が振る型）。
    """

    @pytest.mark.parametrize("text", [
        "x を 0 から 1 まで掃引して", "x を 0 から 1 まで掃引する",
        "x を 0 から 1 まで振って", "x を 0 から 1 まで振る",
        "x を 0 から 1 まで振り", "x を 0 から 1 まで振って計算して",
        "x を 0 から 1 まで 5 点で振って",
    ])
    def test_every_inflection_is_read(self, text):
        m = nlp("numeric_fork_sweep_nl", "ja").search(text)
        assert m and m.group("var") == "x" and m.group("max") == "1", text

    def test_the_bare_stem_is_not_a_sentence(self):
        """`振っ` 単体は日本語ではない —— 広げすぎていないことの確認。"""
        assert nlp("numeric_fork_sweep_nl", "ja").search("x を 0 から 1 まで振っ") is None


class TestFragmentSpellings:
    """鍵ではなく fragment に住んでいる綴り —— `nlp()` では引けない。"""

    @pytest.mark.parametrize("name,spelling", [
        ("compute_verb_en", "calculates"),
        ("times_adverb_en", "thrice"),
        ("ordinal_token", "eighth"),
        ("list_sep_ja", "並びに"),
        ("ja_ru_infl", "って"),
    ])
    def test_the_fragment_matches_the_spelling(self, name, spelling):
        import re
        assert re.search(fragment(name), spelling, re.I), (name, spelling)
