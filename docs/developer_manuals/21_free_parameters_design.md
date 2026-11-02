# 自由パラメータと解の特殊化（一般解 ↔ 特定解）設計仕様 — FREE-PARAM

操作が生む**自由度（任意定数・初期値プレースホルダ）を名前付き記号として保持**し、**条件/代入で特殊化**する
一般機構。数学流（一般解を残す）と物理流（条件を入れて特定解にする）を、一つのオブジェクトと一つの操作で
扱う。S-e（RLC 伝達関数）はこの機構の一事例（Laplace の初期値）として後で実装する。

## 1. 動機

同じ「一般解 → 条件で特殊化」という骨格が、ノートブックの各所に断片的に現れる：

- **不定積分** $\int f\,dx = F(x)+C$：$C$ は積分定数。今は Maxima の `integrate` が $C$ を落とす＝特殊化できない。
- **ODE 一般解（ode2）** $y=k_1 y_1 + k_2 y_2$：$k_1,k_2$ は任意定数。条件は `ic1`/`ic2`/`bc2`。
- **ODE を Laplace で解く**：$q(0),\dot q(0)$ が任意定数（＝一般解の自由度）。条件は値の代入 or `atvalue`。
- **劣決定の solve**：自由変数が残る。

Maxima は機構を**解法ごとにバラしている**（atvalue / ic1・ic2 / `+C` を落とす…）が、**数学的には一つの概念**：
*自由パラメータを持つ一般解を、条件で特殊化する*。

> **二つの立場（why の核心）。** 物理は「初期値・境界値は最初から分かっている → 特定解が欲しい」。数学は
> 「一般解（任意定数つき）が欲しい → 条件は要らない（必要なら後で当てる）」。Maxima の二系統
> （atvalue/desolve＝物理流、ode2+ic1/ic2＝数学流）はこの立場の違いを反映している。本機構は両立場を
> **「残す＝一般／入れる＝特定」という一軸**に統一する。

## 2. 抽象（データモデル）

- `Expression.free_parameters: list[str]` — この結果が持つ自由パラメータの名前（例 `["C"]`, `["k_1","k_2"]`,
  `["q_0","qd_0"]`）。デフォルト空。`integral_descriptor`/`lattice_descriptor` と同じく、結果に付帯する
  「一般形」情報。
- **表示**：自由パラメータは数式記法（$C,\ k_1,\ q(0)$）で「未確定」として出す。**CAS 構文は見せない**
  （内部の `%k1`・`at('diff(...),t=0)` 等は表示・入力に出さない＝Core Principle #1）。

## 3. 自由パラメータを「生む」操作（対象範囲）

実装は次の順で進める：

1. ✅ **不定積分** → $+C$。（最初の実装対象。最も普遍的・現状の欠落）— **実装済み**（値代入・条件特殊化も）
2. **ODE 一般解（ode2）** → $k_1,k_2$。
   - ✅ **生成（2a）実装済み**：`solve_ode_general`（`En を一般に解いて`／`En の一般解を求めて`）。
     `ode2` を駆動し、Maxima の `%k1,%k2,%c` を CAS 内で衝突回避済みの自由パラメータ名へ改名（`%`-痕跡を表示に出さない）。
     従属/独立変数はセッションの時間依存量から推論。記号係数が `ode2` の対話質問
     （"zero or nonzero?"）を誘発し永続アダプタを固める問題は、係数を一時的に正と仮定→直後に
     `forget` して回避（グローバルなリークなし）。
   - ✅ **特殊化（2b）実装済み**：`specialize_by_conditions`（`En を q(0)=q0, q'(0)=v0 で特殊化して`）。
     方程式形 $q=…$ の RHS に対し、各条件 $d^n q/dt^n|_{t=t_0}=$ 値を連立して全自由パラメータを solve し、
     代入して特定解を返す（`ic1/ic2/bc2` と等価だが、CAS 機構を隠した条件系として表現）。
     条件数 < パラメータ数の**部分特殊化**にも対応：未確定の定数は backend プレースホルダ（Maxima `%rN`）
     として戻るので、元の定数名に戻して `free_parameters` に残す（`BackendHandler.is_free_solve_const`）。
3. **ODE を Laplace で解く** → $q(0),\dot q(0)$。（ここで S-e に接続）
   - **方針（利用者決定）**：Maxima 組み込みの `laplace`（不透明な `%at` IC 項を生む）を**使わず**、
     **定義 $\mathcal L\{f\}(s)=\int_0^\infty f(t)e^{-st}dt$ に忠実に積分**する。s 領域の像は $Q(s),V(s)$
     （表現1）で表すが、$Q$ は魔法の記号ではなく**定義カードで明示**（表現2の定義を CAS カードに残す）。
     未知関数の導関数項は**部分積分を明示して導く**（$\int_0^\infty\dot q\,e^{-st}dt=[q e^{-st}]_0^\infty
     +s\int_0^\infty q e^{-st}dt = sQ-q(0)$）。境界項 $q(0),\dot q(0)$ は Maxima の `subst`/`%at` から
     取り出すのではなく、**こちらが名前付き自由パラメータとして導入**する → `%at`/`depends` の壁が消える。
   - ✅ **3a 実装済み**：明示関数の定義ラプラス変換。`laplace_by_definition`
     （`<f> を定義に従って t から s へラプラス変換して`）。`∫₀^∞ f e^{-st}dt` を `assume(s>0,…)` で
     収束を保証して評価（対話質問でアダプタが固まるのを回避・直後 `forget`）。定義カード
     $\mathcal L\{f\}(s)=\int_0^\infty f e^{-st}dt = $ 値 を表示。`tests/test_laplace_definition.py`。
   - ✅ **3b 実装済み**：未知時間依存関数の導関数を変換。`laplace_by_definition`
     が `engine._laplace_td_term`（`q`/`q_dot`/`q_ddot` を検出）→ `_laplace_derivative_transform` で
     像 $s^nQ-\sum_j s^{n-1-j}q^{(j)}(0)$・IC 自由パラメータ `q_ic{j}` を生成。
     $\mathcal L\{\dot q\}=sQ(s)-q(0)$、$\mathcal L\{\ddot q\}=s^2Q(s)-sq(0)-\dot q(0)$、$\mathcal L\{q\}=Q(s)$。
     境界項は Maxima `subst`/`%at` からではなく**こちらが導入**（壁が消える）。単項変換は部分積分チェーン
     も表示（`TestLaplaceDerivativeByParts`）。`tests/test_laplace_definition.py`。
   - ✅ **3c 実装済み**：ODE 全体を**線形性**で s 領域へ（`_laplace_linear_transform`：各項を
     s 領域像で subst し Maxima が係数を分配）→ s 領域方程式が IC 自由パラメータ付きで得られる。
     ラプラス変換は**線形操作**なので、線形部のみ像に置換し、**非線形項（$x^2$ 等）は誤った $X^2$ を出さず
     未評価 $\mathcal L\{\cdot\}(s)$ のまま保持**（準同型の誤謬を避ける、`nb_lap(...)`）。表示は最終 s 領域
     方程式のみ（部分積分の inline 再導出 `⇒` は冗長なので出さない）。表示ブリッジで `q_ic0→q(0)`・`Q→Q(s)`。
   - ✅ **伝達関数まで全て一般操作で**（S-e 専用操作なし）：① `q(0)=0, q'(0)=0 とする`
     （`substitute_free_parameters`：IC 自由パラメータの**値代入を明示**）→ ② `H = Q/V`（H-1 定義）→
     ③ `Q について解いて`（一般 solve、s 領域の実在記号 $Q$ が対象）→ ④ `… を … に代入して`（一般代入）
     → $H=\dfrac{1}{Ls^2+Rs+1/C}$。**S-e を FREE-PARAM＋一般操作で再建**（`%at`/`atvalue`/builtin laplace
     /専用 `zero_initial_conditions`・`transfer_function` なし）。`tests/test_laplace_definition.py::TestSeTransferFunction`。
     旧 S-e の `%at`・`laplace(...)`・係数抽出依存の棚上げ実装は全削除。
4. ✅ **劣決定/パラメトリック solve** → 自由変数（**実装済み**）。式の本数 < 未知数の連立を解くと
   Maxima は自由変数を `%rN` で返す。`engine._rename_free_solve_vars` が各 `%rN` を**元の自由変数名**に改名
   （`x = (1-z)/2` の形・`z=z` の自明行は落とす。純粋に対応しない `%rN` は `_fresh_free_param("t")`）。解の各行に
   `free_parameters=[z]` を付与。値代入（`En で z=0 とする` → `substitute_free_parameters` の plain 形）で特殊化。
   `tests/test_free_parameters.py::TestUnderdeterminedSolve`。（注：1 本の式を多変数について解くのは不自然なので
   非対応＝劣決定は**連立**で扱う。1 本なら `x について解いて` で 1 変数を他で表す。）

**将来の拡張候補**（自由度を生む他の操作）：漸化式の解（`solve_rec`）の任意定数、固有ベクトルの規格化自由度、
斉次線型系の零空間基底係数、級数解の任意係数、ポテンシャルのゲージ自由度（領域依存）。発見しだい本表に追加。

## 4. 特殊化操作（新・一般）

`条件 … を与えて` / `… を代入して`：自由パラメータを**値の代入**または**条件を解いて**確定する。

- **値代入**：`C=0`、`q(0)=0`（伝達関数の零初期条件はこれ）。
- **条件**：`F(0)=0`（→ $C$ を解く）、`y(0)=1, y'(0)=0`（→ $k_1,k_2$）。
- **内部ルーティング**：解法に応じて「単純代入 / `ic1`・`ic2`・`bc2` / `atvalue`」へ振り分ける。
  CAS 都合（どの機構か）は隠す。利用者は**問題レベルの条件**を数学として宣言するだけ。

## 5. 数学流と物理流が一軸に

| 立場 | 操作 | 結果 |
|---|---|---|
| 数学（一般解を残す） | 自由パラメータを保持したまま | $F(x)+C$, $k_1 y_1+k_2 y_2$, $q(0),\dot q(0)$ を含む形 |
| 物理（特定解にする） | 条件/代入で特殊化 | 自由パラメータが確定した特定解 |

既存の deferred-eval（`integral_descriptor`・`lattice_descriptor`＝未評価の一般形を持ち、明示操作で具体化）と
**同じ哲学**。自由パラメータつき一般解も「一般形オブジェクト」、特殊化が「評価」。

## 6. 実装計画

1. **モデル**：`Expression.free_parameters` を追加。
2. **不定積分 $+C$**：`integrate_indefinite` が $C$ を自由パラメータとして付与・表示。
3. **特殊化操作**：`条件を与えて`/`代入して` で $C$ を確定（値代入 or 条件 solve）。ここで全サイクルを最小実証。✅
4. 同じ枠で ②ODE 一般解（$k_1,k_2$ ＋ ic1/ic2）→ ③Laplace（$q(0),\dot q(0)$）→ S-e へ展開。
   - ②生成（2a）✅・②特殊化（2b）✅実装済み（`solve_ode_general`／`specialize_by_conditions`、
     `tests/test_free_parameters.py` の `test_ode_*`）。多条件特殊化は instance 3・4 でも再利用する共通基盤。
   - ③Laplace ✅実装済み（3a 明示関数の定義積分／3b 部分積分による導関数変換／3c ODE 線形性→ゼロIC→伝達関数）。
     **S-e を FREE-PARAM 上で再建完了**（`tests/test_laplace_definition.py`）。

## 7. S-e（棚上げ中）との接続

S-e の零初期条件は、本機構では「Laplace が生んだ自由パラメータ $q(0),\dot q(0)$ に 0 を代入して特殊化」。
これにより、現在 S-e で詰まっている「IC 項に代入が効かない（`%at`・depends）」を、**IC 項を最初から
substitutable な名前付き記号（$q_0,\dot q_0$）で保持する**ことで根本回避する。S-e は本機構の③の一事例。

**関連:** [20_definite_integral_change_of_variables.md](20_definite_integral_change_of_variables.md)（deferred-eval
の先例）、`notebook/models/expression.py`（`integral_descriptor`/`lattice_descriptor`）。
