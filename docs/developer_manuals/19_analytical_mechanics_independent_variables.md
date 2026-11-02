# 19. 解析力学：座標と速度を独立変数とするモデル

**状態:** 実装済み（2026-06-18）。`analytical_mechanics` ドメインの時間微分を独立アトム・
モデルに移行し、`classical_mechanics` ドメインを新設して意味論を分離した。
単振り子・二重振り子（速度結合）・連成振動で検証済み。テスト
`tests/test_engine.py::TestLagrangianIndependentAtoms`。

## 1. 問題（根本原因）

同じ記法 $\dot q$ が、文脈で**意味が二つ**ある：

- **解析力学**：$\dot q$ は $q$ と**独立な変数**。$L(q,\dot q,t)$ で $\partial L/\partial\dot q$ は
  $\dot q$ を独立に扱う（Euler-Lagrange の定義）。
- **一般微積／純粋古典力学**：$\dot q = dq/dt$（$q$ に**従属**）。

現状システムはこの二律背反を**一つの大域モデル＋ハック**で無理に橋渡ししている：
`q_dot`（独立アトム）／`'diff(q(t),t,1)`（従属・関数形）／`'diff(q,t,1)`（depends 形）が混在し、
`differentiate` の速度偏微分では block で一時変数へ退避して独立性を演出している。

この綱引きが、S-b/S-d シナリオ実行中に見つかった一連のバグの**共通の根**だった：

- ② EOM 符号（再パースで括弧落ち）／③ `∂L/∂q` への `_nb_qdot_tmp` 漏れ／⑤ wrt_qdot が対象を誤る／
  ① `\dot{q}` 入力が `q_dot` のまま速度微分が 0。

「時間微分が出るたびに互いの修正が衝突する」状態。これを分野ごとの**意味論の分離**で根治する。

## 2. 提案モデル：独立アトムの徹底（analytical_mechanics）

一般化座標 $q_1,\dots,q_n$ と、その速度・加速度・… を**独立な記号アトム**として扱う：

$$q_i,\quad \dot q_i\,(=q_i\_dot),\quad \ddot q_i\,(=q_i\_ddot),\quad \dddot q_i,\dots$$

| 操作 | 実装 | 例（単振り子 $L=\tfrac12 ml^2\dot q^2 - mgl(1-\cos q)$） |
|---|---|---|
| $\partial L/\partial\dot q$ | `diff(L, q_dot)` | $ml^2\dot q$（$q$ と独立なので素直） |
| $\partial L/\partial q$ | `diff(L, q)` | $-mgl\sin q$（$\dot q^2$ 項は $q$ に依らず 0） |
| **$d/dt$（全時間微分）** | $\displaystyle\sum_i\sum_{k\ge0}\frac{\partial E}{\partial q_i^{(k)}}\,q_i^{(k+1)}$ | $d/dt(ml^2\dot q)=ml^2\ddot q$ |
| EOM | $\dfrac{d}{dt}\dfrac{\partial L}{\partial\dot q}-\dfrac{\partial L}{\partial q}$ | $ml^2\ddot q+mgl\sin q$ |

ここで $q_i^{(0)}=q_i,\ q_i^{(1)}=q_i\_dot,\ q_i^{(2)}=q_i\_ddot,\dots$。

**全時間微分の定義（唯一ここだけが $q\leftrightarrow\dot q$ の従属を使う）:**

$$\frac{dE}{dt}=\sum_i\left(\frac{\partial E}{\partial q_i}\dot q_i
  +\frac{\partial E}{\partial\dot q_i}\ddot q_i
  +\frac{\partial E}{\partial\ddot q_i}\dddot q_i+\cdots\right)$$

二重振り子 $\partial L/\partial\dot a = mpq\,\dot b\cos(a{-}b)+(m{+}M)p^2\dot a$ の全時間微分は

$$-mpq\,\dot a\dot b\sin(a{-}b)+mpq\,\dot b^2\sin(a{-}b)+(m{+}M)p^2\ddot a+mpq\,\ddot b\cos(a{-}b)$$

と連鎖律の総和で**一発で正しく**出る（block 退避不要）。

## 3. ドメイン分離（classical_mechanics 新設）

意味論が違うので分ける：

| ドメイン | $\dot q$ の扱い | $d/dt$ | 対象 |
|---|---|---|---|
| **analytical_mechanics**（再定義） | **独立アトム** $q,\dot q,\ddot q$ | **全時間微分**（連鎖律総和） | Lagrangian/Hamiltonian、Euler-Lagrange、基準振動 |
| **classical_mechanics**（新設） | $\dot q=dq/dt$（従属・$q(t)$ 関数形） | 通常の関数微分 | Newton 力学（運動方程式の直接積分・衝突・軌道） |

それ以外（一般微積・物理）は従来の $d/dt$（関数微分）を保つ。

**先日の「古典力学ドメインは不要」からの転回:** 計算能力はドメインに依らないが、$\dot q$ の**意味論**が
ドメインで異なる（独立 vs 従属）。これが分離の本質的理由。

## 4. 実装方針（スケッチ）

- **ゲート:** セッションの domains に `analytical_mechanics` が含まれる（または「Lagrangian モード」フラグ）
  ときのみ独立アトム・モデルを有効化。`BackendHandler` に `lagrangian_independent_velocities`
  のような能力フラグを置き、Maxima ハンドラで分岐。
- **速度偏微分（∂/∂q̇）:** independent モードでは `diff({expr}, {q}_dot)`（block 退避を撤去）。
- **位置偏微分（∂/∂q）:** `diff({expr}, {q})`（独立なので速度項は自動で 0）。
- **時間全微分（d/dt）:** 新しいビルド経路。一般化座標一覧（time_dependent 量）と各々の最高階を見て、
  $\sum_i\sum_k \mathrm{diff}(E,q_i^{(k)})\cdot q_i^{(k+1)}$ を Maxima 式で組む。`q_i^{(k)}` は `q_i`,
  `q_i_dot`, `q_i_ddot`, `q_i_dddot` の素アトム。
- **canonical:** `\dot{q}` → `q_dot`（latex_converter の既存中間形）を**そのまま素アトムとして使う**
  （関数形 `'diff(q(t),t,1)` には変換しない）。表示は `q_dot`→`\dot{q}` の既存正規化で綺麗。
- **撤去するハック:** `differentiate` の velocity-partial block（`_nb_qdot_tmp` 退避・関数形検出）、
  `apply_time_dep_map` の analytical 経路での使用、pre_process の関数形変換（試作・撤回済み）。
  classical_mechanics 側は従来の関数形・depends 経路を流用。

**実装箇所（2026-06-18）:**
- ゲート `MaximaBackendHandler.is_lagrangian_mode()`（`"analytical_mechanics" in ctx.domains`）。
- `differentiate` の Lagrangian 分岐（`build_command`）：∂/∂q̇・∂/∂q は素アトムの `diff`、
  d/dt は全時間微分の連鎖律総和。
- `wrt_qdot_canonical()`：Lagrangian モードで `q_dot` を返す（関数形にしない）。
- `build_command` 冒頭の `q_dot→'diff` 変換と `simplify_raw_command` の `apply_time_dep_map` を
  Lagrangian モードで抑止（アトム保持）。
- 表示：`postprocess_result_latex` で `q_{dot}`→`\dot{q}`、`q_{ddot}`→`\ddot{q}`。
- domains 同期：`engine.start()` と `notebook_bridge.start_engine()`・`debug_repl` で
  `symbol_context.domains` を `session.domains` から再同期（構築後設定に対応）。

## 5. 移行

- **シナリオ:** S-b/S-d を `\dot{q}` 入力（数式記法）に書き換え（現状は CAS 文法 `diff(q(t),t)` を
  使っており Core Principle #1 違反。本再設計で解消）。実例 `.json/.tex` も再生成。
- **テスト:** `test_engine.py` の `TestDiffWrtQdot`、EL/EOM、二重振り子・連成振動の E2E を独立アトム・
  モデルで作り直す。classical_mechanics 側の関数形 d/dt も別テストで担保。
- **AM-1**（open_items）：②③⑤は本再設計で「ハックごと撤去」に置き換え、①④は本モデルで自然解消の見込み。

## 6. 未決・リスク

- **classical_mechanics と analytical_mechanics の両方が active なセッション**の扱い（どちらの $d/dt$ か）。
  → セッションは単一の力学パラダイムを選ぶ前提にする（混在は警告）。
- **既存の Maxima depends 経路**（H-9 solve の `\ddot{q}` 解、re_evaluate 等）への影響。
  classical_mechanics 側に温存し、analytical 側は独立アトムに寄せる切り分けが要る。
- **メニュー再割当:** S-h（中心力の保存量）等を analytical/classical のどちらにするかは別途。
- Julia セッションでの解析力学（現状ほぼ Maxima 前提）。

## 7. 関連

- 発見の経緯と暫定バグ: [open_items.md](../../KNOWN_ISSUES.md) §AM-1。
- シナリオ: [scenario_double_pendulum_lagrangian.md](../scenarios/scenario_double_pendulum_lagrangian_jp.md)（S-d）、
  [scenario_coupled_oscillations.md](../scenarios/scenario_coupled_oscillations_jp.md)（S-b）。
- バックエンド分離方針: [03_backend_paradigm_unification.md](03_backend_paradigm_unification.md)、CLAUDE.md。
