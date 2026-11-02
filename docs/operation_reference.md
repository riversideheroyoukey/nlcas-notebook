# 操作リファレンス

本システムが対応する共通 operation の一覧（schema 定義は 109 操作。Maxima 93／Julia 50 がそれぞれ対応）。

自然言語で操作を指示するときは「エイリアス例」の言葉がそのまま使えます。
必須引数が不足している場合は LLM supervisor が確認ダイアログを出します。

---

## 表の見方

| 列 | 説明 |
|---|---|
| operation | 内部 operation 名（ログ・テストで使用） |
| 日本語名 | 通常 UI での表示 |
| 必須引数 | 省略できない引数。`expr` は式番号（E1 など）、その他は値 |
| 省略可能引数 | 省略すると自動補完または既定値が使われる |
| エイリアス例（日本語） | 自然言語指示で使える代表的な言葉 |

---

## 定義

| operation | 日本語名 | 必須引数 | エイリアス例（日本語） |
|---|---|---|---|
| `define_function` | 定義する | expr | 定義、定義する、関数定義、関数を定義する |
| `define_scalar_parameter` | スカラーパラメータを定義する | name | スカラーとする、スカラーパラメータとする |
| `define_vector` | ベクトルを定義する | name | ベクトルとする、ベクトル変数とする |
| `define_tensor` | テンソルを定義する | name | テンソルとする、テンソル変数とする |
| `define_matrix` | 行列を定義する | name | 行列とする、行列変数とする |

（ベクトル場・スカラー場・座標系の定義は「ベクトル解析」節、量子演算子・粒子の定義は「量子代数」節を参照。）

---

## 方程式変形

| operation | 日本語名 | 必須引数 | エイリアス例（日本語） |
|---|---|---|---|
| `multiply_both_sides` | 両辺に乗じる | expr, factor | 両辺にかける、両辺に掛ける、両辺を乗じる |
| `add_to_both_sides` | 両辺に加える | expr, term | 両辺に加える、両辺に足す、両辺に足して |
| `subtract_from_both_sides` | 両辺から引く | expr, term | 両辺から引く、両辺を引く、両辺から引いて |
| `divide_both_sides` | 両辺を割る | expr, factor | 両辺を割る、両辺で割る、両辺を割って |

---

## 代数変形

| operation | 日本語名 | 必須引数 | 省略可能 | エイリアス例（日本語） |
|---|---|---|---|---|
| `substitute` | 代入する | expr, lhs, rhs | — | 代入、代入する、置換 |
| `use_relation` | 関係式を使う | expr, relation_ref | — | 関係式を使う、関係式を適用 |
| `simplify` | 簡約する | expr | — | 簡約、簡約する、単純化 |
| `rational_simplify` | 有理式として約分・通分する | expr | — | 有理簡約、有理式として簡約 |
| `expand` | 展開する | expr | — | 展開、展開する |
| `factor` | 因数分解する | expr | — | 因数分解、因数分解する |
| `collect` | 変数についてまとめる | expr, var | — | まとめる、整理、整理する |
| `radical_simplify` | 根号を簡約する | expr | — | 根号を簡約、ルートを整理 |
| `partial_fraction` | 部分分数分解する | expr, var | — | 部分分数、部分分数分解 |
| `log_contract` | 対数をまとめる | expr | — | 対数をまとめる、ログをまとめる |
| `trig_simplify` | 三角関数を簡約する | expr | — | 三角簡約、三角関数を簡約 |
| `trig_expand` | 三角関数を展開する | expr | — | 三角展開、三角関数を展開 |
| `trig_reduce` | 三角関数を縮約する | expr | — | 三角縮約、三角関数を縮約 |
| `trig_to_exponential` | 三角関数を指数形式にする | expr | — | オイラー形式、指数形式に変換 |
| `multiply_expressions` | 式を掛け合わせる | a, b | assign_to | 式を掛け合わせる、乗じる、掛け合わせる |
| `combine_expressions` | 式を組み合わせる | expr_a, expr_b, op | assign_to | 式を足す、式を引く、式を掛ける、式を割る |

---

## 方程式への操作（H-5〜H-9）

LLM 不要・ルールベース処理。登録済み方程式（`LHS = RHS` 形式）に対して後続操作を行う。

| 機能 | トリガー例 | 処理内容 |
|---|---|---|
| **H-5** 右辺を評価 | `E3 の右辺を計算して` | RHS に `simplify` 等を適用して新規式登録 |
| **H-5z** 右辺に値を設定 | `E3 の右辺を 0 とする` / `とおく` / `と置く` | 新規方程式 `RHS = X` を登録（CAS 評価なし）。E-ID 参照・全角数字対応 |
| **H-7** 右辺への操作 | `E3 の右辺を展開して` | RHS に `expand` / `simplify` / `factor` 等を適用 |
| **H-8** 両辺への操作 | `E3 の両辺を m で割る` | 四則演算を両辺に適用し LHS も変換 |
| **H-9** 方程式を解く | `E3 を x について解いて` | 単一変数 / 連立にも対応 |

---

## 方程式を解く

| operation | 日本語名 | 必須引数 | エイリアス例（日本語） |
|---|---|---|---|
| `solve_equation` | 方程式を解く | equation, var | 方程式を解く、解く、解を求める |
| `solve_system` | 連立方程式を解く | equations, vars | 連立方程式を解く、連立を解く |
| `numeric_solve` | 数値的に解く | function, initial_guess | 数値解、数値的に解く |

---

## 微積分・解析

| operation | 日本語名 | 必須引数 | 省略可能 | エイリアス例（日本語） |
|---|---|---|---|---|
| `differentiate` | 微分する | expr, var | order（既定 1） | 微分、微分する、導関数を求める |

**高階微分（order）の指定:** 「`N 階微分`」「`N 次微分`」「`N 回微分`」のいずれの数え方でも
`order=N` として解釈される（ルールベース、LLM 不使用）。例：「E1 を k で 2 回微分して」→ `order=2`。
- Julia は `_nb_differentiate(expr, var, order)` が `Differential(var)` を order 回適用（テンプレートに `defaults:{order:1}`）。Maxima は `diff(expr,var,order)` がネイティブ対応。
- 導関数をさらに**同じ変数**で微分すると LHS は入れ子（`d/dk(d/dk(X))`）でなく `d²/dk²(X)` に畳み込まれる（位置→速度→加速度）。異なる変数・混合偏微分は畳み込まない。
- 注意：solve した変数（`lhs_alias_map` に登録済み）で微分する場合も、微分変数はエイリアス展開されず元のシンボルのまま使われる。
| `integrate_indefinite` | 不定積分を求める | expr, var | — | 不定積分、積分、積分する |
| `integrate_definite` | 定積分を求める | expr, var, lower, upper | — | 定積分、定積分を求める |
| `limit` | 極限を求める | expr, var, point | — | 極限、極限を求める、極限値 |
| `summation` | 和を求める | expr, var, lower, upper | — | 和、総和、シグマ |
| `product_expr` | 積を求める | expr, var, lower, upper | — | 積、総積 |
| `laplace_transform` | ラプラス変換する | expr, t, s | — | ラプラス変換 |
| `inverse_laplace` | 逆ラプラス変換する | expr, s, t | — | 逆ラプラス変換 |
| `change_variable` | 変数変換する | expr, old_var, new_expr | — | 変数変換、置換積分、積分変数を変換 |
| `change_integration_variables` | 積分の多変数変数変換（ヤコビアン込み） | expr, old_vars, new_vars, transform | — | ヤコビアン、座標変換して積分 |

---

## ハミルトン力学（ポアソン括弧）

| operation | 日本語名 | 必須引数 | エイリアス例（日本語） |
|---|---|---|---|
| `poisson_bracket` | ポアソン括弧を計算する | a, b | ポアソン括弧、ポワソン括弧、ポアソンの括弧 |

ポアソン括弧 $\{f,g\}=\sum_i\left(\dfrac{\partial f}{\partial q_i}\dfrac{\partial g}{\partial p_i}-\dfrac{\partial f}{\partial p_i}\dfrac{\partial g}{\partial q_i}\right)$
は量子の `commutator` の古典版で、**Maxima** で偏微分の和として構成される（ルールベース、LLM 不使用）。

- **正準座標対を先に宣言する**：「`p_r を r の共役運動量とする`」または「`r と p_r を正準対とする`」
  で $(q_i,p_i)$ を登録（`SymbolContext.canonical_pairs` に保存、save/load 永続）。以後
  「`f と g のポアソン括弧を求めて`」が宣言済みの全対について和を自動構成する。
- オペランド `a` / `b` は E-ID・H-1 名（`H`, `L_z`）・素のシンボル（`r`, `p_r`）のいずれも可。
- 正準対が未宣言なら、宣言を促すメッセージを返す。
- 例：`{r, p_r}=1`、`{H, L_z}=0`（中心力の角運動量保存）、`{H, r}=-p_r/m`（Hamilton 運動方程式）。
- **ブラケット記法**：`[H, L_z]` のように書いても（`f と g のポアソン括弧を求めて` と同様に）
  `poisson_bracket` に解決される（Maxima セッション限定；`commutator` の `[X, Y]` 記法と
  バックエンド排他的に共存する — 2026-07-08）。

---

## ベクトル解析と LaTeX ナブラ（$\nabla$）記法

物理・工学ドメインで多用されるベクトル解析演算子（`rot` / `curl`, `div`, `grad`, `laplacian`）は、LaTeX 形式のナブラ記法（$\nabla$）やラプラシアン記法（$\Delta$）からセマンティクスを維持したまま、プリプロセッサ層で自動解決されます。

### 1. 自動解決ルール一覧

| 入力 LaTeX パターン | 変換後の関数表記 | 数学的意味 | 備考 |
|:---|:---|:---|:---|
| `\nabla \cdot F` | `div(F)` | **発散 (Divergence)** | $F$ はベクトル（$\vec{F}$, $\mathbf{F}$）、カッコ式も可 |
| `\nabla \times F` | `rot(F)` | **回転 (Curl)** | $F$ はベクトル、Julia では `rot` / `curl` 双方可 |
| `\nabla^2 \phi` または `\Delta \phi` | `laplacian(phi)` | **ラプラシアン (Laplacian)** | $\phi$ はスカラーまたはベクトル |
| `\nabla \phi` | `grad(phi)` | **勾配 (Gradient)** | $\phi$ はスカラー |

* オペランドには、ギリシャ文字（例: `\phi`）、ベクトル表記（例: `\vec{F}`, `\mathbf{F}`）、丸括弧でグループ化された複合式（例: `(A + B)`）などが高精度に自動キャプチャされます。
* **入力前提条件**: 手書き数式画像から LaTeX への OCR 変換や手動入力等により、**LaTeX 形式の文字列がシステムに入力された場合**、その数学的意味を完全に維持して CAS 計算へのトレースが実行されることを保証します（手書き画像認識自体の精度はスコープ外）。

### 2. 各バックエンドでの対応

* **Maxima (`vect` パッケージ)**:
  - システムが `load(vect)` を自動ロードするため、`express(grad(f))`、`express(div(F))`、`express(rot(F))`、`express(laplacian(f))` 等を通じて自動的に微分の和へ美しく展開されます。
* **Julia (`Symbolics.jl` / `Latexify.jl`)**:
  - `grad`, `div`, `rot`, `curl`, `laplacian` が事前定義された `@syms` 記号オブジェクトとして登録されており、変数宣言なしでそのまま数式中で記号的に処理・評価できます。

### 3. ベクトル解析・座標系の操作一覧

| operation | 日本語名 | 必須引数 | エイリアス例（日本語） |
|---|---|---|---|
| `define_vector_field` | ベクトル場を定義する | name | ベクトル場とする、ベクトル関数とする |
| `define_scalar_field` | スカラー場を定義する | name | スカラー場とする、スカラー関数とする |
| `define_coordinate_system` | 座標系を定義する | vars, lame | 座標系を定義する、カスタム座標系を定義する |
| `divergence` | 発散を計算する | expr | 発散、発散を計算する、∇· |
| `gradient` | 勾配を計算する | expr | 勾配、勾配を計算する、∇ |
| `scalar_laplacian` | スカラーラプラシアンを計算する | expr | ラプラシアン、ラプラシアンを計算する、∇² |
| `cross_product` | 外積を計算する | a, b | 外積、クロス積 |
| `dot_product` | 内積を計算する | a, b | 内積、スカラー積 |
| `toggle_einstein_summation` | アインシュタイン縮約を切り替える | enabled | アインシュタインの縮約記法を有効にして、アインシュタイン縮約を切り替え |

（注：`curl`/`rot`（回転）は上記の自動解決ルールで `\nabla\times F` として扱う。非デカルト座標のベクトルラプラシアンは未対応＝`docs/open_items.md` §F-3残。）

---

## 級数展開

| operation | 日本語名 | 必須引数 | エイリアス例（日本語） |
|---|---|---|---|
| `taylor_expand` | テイラー展開する | expr, var, point, order | テイラー展開、テイラー展開する |
| `series_expand` | 級数展開する | expr, var, point, order | 級数展開、べき級数展開 |

---

## 複素解析

| operation | 日本語名 | 必須引数 | エイリアス例（日本語） |
|---|---|---|---|
| `real_part` | 実部を取り出す | expr | 実部、実部を求める |
| `imaginary_part` | 虚部を取り出す | expr | 虚部、虚部を求める |
| `complex_conjugate` | 複素共役を求める | expr | 複素共役、共役 |
| `to_rectangular` | 直交形式に変換する | expr | 直交形式、デカルト形式 |
| `to_polar` | 極形式に変換する | expr | 極形式、極形式に変換 |
| `residue` | 留数を求める | expr, var, point | 留数、留数を求める |

> **活用形もルールベース:** 「実部を取り出し**て**」「虚部を求め**て**」のような **て形**の指示も LLM なしで
> 解決する（トークナイザが操作の動詞語尾を剥がして操作名詞を露出。
> `i18n/nl_patterns.json` の `verb_tail_strip`——2026-08-27 まで `llm_supervisor._JP_VERB_TAIL`）。
>
> **虚数単位（電気工学）:** `electrical_engineering` ドメインでは `j`＝虚数単位／`i`＝電流。入力 `j` は
> 計算用に `%i` へ変換し、結果は値レベルで `%i→j` に戻して表示する（添字・上付き・指数の `i` は非破壊）。

---

## 線形代数

| operation | 日本語名 | 必須引数 | エイリアス例（日本語） |
|---|---|---|---|
| `determinant` | 行列式を求める | expr | 行列式、det |
| `inverse_matrix` | 逆行列を求める | expr | 逆行列 |
| `transpose` | 転置行列を求める | expr | 転置、転置行列 |
| `trace` | 対角和を求める | expr | トレース、対角和 |
| `matrix_multiply` | 行列積を求める | a, b | 行列積、行列の積 |
| `eigenvalues` | 固有値を求める | expr | 固有値 |
| `eigenvectors` | 固有ベクトルを求める | expr | 固有ベクトル |
| `diagonalize` | 対角化する（正規化した固有ベクトルを列に並べた直交変換行列 P） | expr | 対角化、モード行列、変換行列 |
| `diagonal_matrix` | 固有値を並べた対角行列 D を求める | expr | 対角行列 |
| `characteristic_polynomial` | 特性多項式を求める | expr, var | 特性多項式、固有値方程式、永年方程式 |

---

## 線形基底変換（L-1）

一般の線形基底変換 new = M·old から、逆変換・正準/計量拘束を導出する（設計＝`docs/developer_manuals/18_linear_transform_design.md`）。Maxima・Julia 両対応。

| operation | 日本語名 | 必須引数 | エイリアス例（日本語） |
|---|---|---|---|
| `build_inverse_transform` | 逆変換を作る | equations, vars | 逆変換を作る、逆基底変換 |
| `derive_canonical_constraints` | 正準拘束を求める | forward | 正準拘束、Bogoliubov 拘束、正準条件 |
| `derive_metric_constraints` | 計量保存条件を求める | forward, basis, gram | 計量保存条件、直交条件、計量を保つ条件 |
| `apply_canonical_constraints` | 拘束を適用して簡約 | expr, relations | 拘束を適用、正準拘束のもとで簡約 |

「E1, E2 は正準か」と尋ねると para-unitary 充足を判定する（`derive_canonical_constraints` 経由）。

---

## 情報量測度（F-16 — information_theory ドメイン）

`log`／`summation` の組み合わせのみで構成される純粋テンプレート操作（新規 CAS ロジックなし）。

| operation | 日本語名 | 必須引数 | エイリアス例（日本語） |
|---|---|---|---|
| `shannon_entropy` | エントロピーを計算する | p, var, lower, upper | エントロピー、エントロピーを計算する、シャノンエントロピー |
| `kl_divergence` | KLダイバージェンスを計算する | p, q, var, lower, upper | KLダイバージェンス、カルバック・ライブラー・ダイバージェンス、相対エントロピー |
| `mutual_information` | 相互情報量を計算する | pxy, px, py, xvar, xlower, xupper, yvar, ylower, yupper | 相互情報量、相互情報量を計算する |

> **注意（フィールド数が多い操作の自然言語起動について）:** 上記3操作はいずれも必須引数が4個以上あり、
> `compute the entropy of p=1/6, var=x, lower=1, upper=6` のような `field=value` 列挙形式の自然言語指示は、
> ローカル Supervisor LLM がフィールドを安定して取り違えることがある（`docs/open_items.md` [BP-6] 参照。
> 既存の `summation` 操作でも同様の問題が確認済みで、これら新規操作固有の不具合ではない）。
> 確実に呼び出すには `run_operation` の直接呼び出し、または確認ダイアログでのフィールド補完を推奨する。
> チャネル容量（$\max_{p(x)}I(X;Y)$）は本実装の対象外（`mutual_information` と F-14 の
> `stationary_conditions`/`solve_system` を組み合わせる応用例として別途扱う）。

実例: [scenario_information_theory.md](scenarios/scenario_information_theory_jp.md)。

---

## 制約付き最適化（F-14 — mathematical_economics ドメイン）

ラグランジュ未定乗数法（等式制約のみ、KKT／不等式制約は対象外）。

| operation | 日本語名 | 必須引数 | 省略可能 | エイリアス例（日本語） |
|---|---|---|---|---|
| `form_lagrangian` | ラグランジアンを作る | expr, constraints | multiplier_prefix | ラグランジアン、ラグランジアンを作る、ラグランジアンを構成する |
| `implicit_derivative` | 陰関数微分（比較静学）を計算する | expr, dependent_var, parameter | — | 比較静学、陰関数微分、比較静学を計算する |

`form_lagrangian` の `constraints` は `solve_system` の `equations` と同じ**カンマ区切りの E-id 文字列**
（`"E4, E5"`）。乗数は `lambda_1`, `lambda_2`, … と自動採番される（既存シンボルとの衝突を回避）。

`stationary_conditions`（ラグランジアンの停留条件を変数の数だけ一括登録）は複数式を1指示で登録する
エンジンメソッドのため、上記の schema 駆動操作とは呼び出し経路が異なる（下表）。

| トリガー例 | 処理内容 |
|---|---|
| `E3 の停留条件を x1, x2, lambda_1 について求めて` | `∂E3/∂x1=0`, `∂E3/∂x2=0`, `∂E3/∂λ1=0` を一括登録（`solve_system` へそのまま渡せる） |

実例: [scenario_utility_maximization.md](scenarios/scenario_utility_maximization_jp.md)。

---

## 変換系（F-8 — signal_processing・control_theory ドメイン）

ラプラス変換と同じ「定義に忠実に積分する」設計（Maxima 組み込み `fourier`/`ztrans` は使わない）。
`fourier_transform_by_definition`／`inverse_fourier_by_definition`／`z_transform_by_definition` は
複数式を扱わないが schema 非登録のエンジンメソッド（NL トリガー専用）で、`convolve` のみ通常の
schema 駆動操作。

| トリガー例 | 処理内容 |
|---|---|
| `exp(-t^2) を定義に従ってフーリエ変換して` | $X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-i\omega t}dt$ を評価 |
| `E1 を定義に従って逆フーリエ変換して` | $x(t)=\frac{1}{2\pi}\int_{-\infty}^{\infty}X(\omega)e^{i\omega t}d\omega$ を評価 |
| `(1/2)^n を定義に従ってZ変換して` | $X(z)=\sum_{n=0}^{\infty}x[n]z^{-n}$ を評価（境界項が残ることがある、下記参照） |

| operation | 日本語名 | 必須引数 | 省略可能 | エイリアス例（日本語） |
|---|---|---|---|---|
| `convolve` | 畳み込みを計算する | f, g, var, dummy | kind（`"continuous"`\|`"discrete"`、既定 continuous） | 畳み込み、畳み込みを計算する |

**既知の制限:** 未知関数の導関数のフーリエ変換（境界項付き）・線形性による方程式全体の変換・逆Z変換は
対象外。Z変換は `nusum` の境界項が自動消去されないことがある（`limit` へ包むと symbolic な底で Maxima
が対話質問を発しハングするため、意図的に安全な素の `nusum` のままにしている）。

実例: [scenario_fourier_convolution.md](scenarios/scenario_fourier_convolution_jp.md)。

---

## 確率解析（F-15 — financial_mathematics ドメイン）

`apply_ito_lemma` は既存の `differentiate` を3回呼び出して伊藤の補題の公式に代入するだけの
orchestration（新規の確率解析ロジックはゼロ）。schema 非登録のエンジンメソッド。

| トリガー例 | 処理内容 |
|---|---|
| `apply Ito's lemma to log(S)` / `log(S) に伊藤の補題を適用して` | GBM $dS=\mu S\,dt+\sigma S\,dW$ に従う $f(S,t)$ について $df=\left(\partial_tf+\mu S\partial_Sf+\tfrac12\sigma^2S^2\partial_S^2f\right)dt+\sigma S\partial_Sf\,dW$ を計算 |

SDE 記法（$dS=\mu S\,dt+\sigma S\,dW$）・Black–Scholes PDE の記述・操作は新規操作不要——既存の
一般式登録（H-1）で表現できる。Black–Scholes PDE を実際に解いてオプション価格の閉形式公式を導出する
ことは対象外（特殊関数を含む境界値問題は Maxima の対応範囲を大きく超える）。

実例: [scenario_ito_black_scholes.md](scenarios/scenario_ito_black_scholes_jp.md)。

---

## ゲーム理論（F-17 — game_theory ドメイン）

利得行列は既存の `[a,b;c,d]` 行列リテラルで表現（新規コードなし）。

| operation | 日本語名 | 必須引数 | エイリアス例（日本語） |
|---|---|---|---|
| `pure_nash_equilibria` | 純戦略ナッシュ均衡を求める | payoff_a, payoff_b | 純戦略ナッシュ均衡、純戦略ナッシュ均衡を求める |

`mixed_strategy_indifference_conditions`（無差別条件式を2本（各プレイヤー分）一括登録）は schema
非登録のエンジンメソッド。

| トリガー例 | 処理内容 |
|---|---|
| `find the mixed strategy equilibrium of E1 and E2` / `E1 と E2 の混合戦略均衡を求めて` | 2×2 の標準的なケースで、プレイヤー1・2それぞれの無差別条件式を1本ずつ登録 |

**既知の制限:** 支配戦略の逐次消去・ゼロ和ミニマックス・一般 m×n 混合戦略は対象外（前者は反復回数が
動的で単一コマンドに収まらない別アルゴリズム、後2つは既存機能——不等式制約経路や F-14 の
`stationary_conditions`——で組める）。混合戦略の無差別条件は 2×2 の場合のみ対応。

実例: [scenario_battle_of_the_sexes.md](scenarios/scenario_battle_of_the_sexes_jp.md)。

---

## 仮定

| operation | 日本語名 | 必須引数 | エイリアス例（日本語） |
|---|---|---|---|
| `add_assumption` | 仮定を追加する | condition | 仮定する、assume |
| `remove_assumption` | 仮定を削除する | condition | 仮定を削除、forget |

仮定は UI の仮定タイル（青）でも管理できます。タイルの ✕ で削除、仮定ボックスから直接追加できます。

---

## 常微分方程式

| operation | 日本語名 | 必須引数 | エイリアス例（日本語） |
|---|---|---|---|
| `solve_ode` | 常微分方程式を解く | equation, y, x | 微分方程式を解く、ODE |

---

## 数値計算

| operation | 日本語名 | 必須引数 | エイリアス例（日本語） |
|---|---|---|---|
| `numerical_eval` | 数値評価する | expr | 数値評価、数値計算 |
| `numerical_eval_hp` | 高精度数値評価する | expr | 高精度数値評価、多倍長精度評価 |

---

## 汎用作図（数値計算層）

曲線・等高線・不等式領域・注目点の作図は、式を選択して「💬 対話で仕様を確定」から
指定します。上表の `numerical_eval` とは別の数値タスクで、CASの共通operationの追加ではありません。

| レイヤー | 指定する登録式・値 | 意味 |
|---|---|---|
| `line` | スカラー式＋横軸1本 | 曲線。方程式なら右辺などを先にスカラー式として登録 |
| `contour` | スカラー式とレベル、または等式とレベル0 | 等高線・等式の境界 |
| `region` | 不等式の式番号のリスト | すべてを満たす領域。別条件の比較は別レイヤー |
| `point` | 実数の横・縦座標、または計算済みスカラー式の参照 | 注目点。最適化計算は行わない |

`contour`・`region`・`point`には横・縦2本の軸と有限の範囲・標本点数が必要です。
PNG/SVG/PDF、標本CSV、診断JSONを保存できます。領域は格子近似です。
[利用手順](user_guide.md#複数の式領域を組み合わせたグラフ)・
[再現例](scenarios/scenario_cartesian_plots.md)・
[構造化仕様とAPI](developer_manuals/36_cartesian_plot_layers.md)を参照。

---

## 出力

| operation | 日本語名 | 必須引数 | エイリアス例（日本語） |
|---|---|---|---|
| `to_latex` | LaTeX形式で出力する | expr | LaTeX出力、ラテックス出力 |
| `to_plain_text` | テキスト形式で出力する | expr | テキスト出力、文字列出力 |

---

## 量の記法・メタデータ

| operation | 日本語名 | 必須引数 | 省略可能 | エイリアス例（日本語） |
|---|---|---|---|---|
| `declare_time_dependent` | 時間依存を宣言する | name | time_parameter | 時間依存を宣言、時間の関数として宣言 |
| `set_notation` | 記法を切り替える | name, format | — | ドット記法にする、Leibniz記法にする、d/dt記法にする |
| `set_latex_name` | LaTeX名を設定 | name, latex | — | LaTeX名を設定、LaTeX名を変更、と表示する |

`format` には `"dot"` または `"leibniz"` を指定します。この操作は、指定したシンボルの時間微分表示記法（`notation` 属性）を個別かつ独立に変更・設定します。設定された記法はシンボルテーブルに保存・永続化されます。

---

## 量子代数（Julia backend — `julia_full` セッションで有効）

Julia セッション（第二量子化カテゴリから開始）で使用できる操作です。

| operation | 日本語名 | 必須引数 | エイリアス例（日本語） |
|---|---|---|---|
| `commutator` | 交換子を計算する | a, b | 交換子、交換子を計算する |
| `anticommutator` | 反交換子を計算する | a, b | 反交換子 |
| `normal_order` | 正規順序にする | expr | 正規順序、正規順序化 |
| `vacuum_expectation` | 真空期待値を計算する | expr | 真空期待値 |
| `collect_by_operators` | 演算子を共通因子としてまとめる | expr | 演算子でまとめる、係数をまとめる、因数分解する（演算子）|
| `relabel_index` | インデックスを付け替える | expr, old_index, new_index | インデックスを付け替える、k を p に変える |
| `collect_spin_pm` | Sx/Sy 表示を Sp/Sm 記法で整理する（表示専用） | expr, sp, sm | Sp/Sm でまとめ直す、梯子演算子でまとめる、Sp/Sm の記法でまとめる |
| `define_boson_operator` | ボソン演算子を定義する | name | ボソン演算子を定義 |
| `define_fermion_operator` | フェルミオン演算子を定義する | name | フェルミオン演算子を定義 |
| `define_spin_operator` | スピン演算子を定義する | name | スピン演算子を定義 |
| `heisenberg_equation` | ハイゼンベルク方程式を作る | operator, hamiltonian | ハイゼンベルク方程式 |

### ブラケット記法（Phase 0 — Julia セッション専用）

`[X, Y]` / `{X, Y}` をそのまま入力すると LLM を経ずに即時解決します。

| 入力例 | 解釈 |
|---|---|
| `[a(1), a'(1)]` | `commutator(a=a(1), b=a'(1))` |
| `[a(1), a'(1)] を計算して` | 同上（NL サフィックスを無視）|
| `{f(1), f'(1)}` | `anticommutator(a=f(1), b=f'(1))` |

### 量子演算子代入 `operator_substitute`（Julia セッション専用）

量子演算子 `c(k)` を別の演算子の線形結合で置き換え、`normal_form()` で整理します。Bogoliubov 変換・ユニタリ変換・偏光分解などに使います。

#### 単一ルール（1演算子を置き換える）

```
E1 の c(k) に u_k alpha(k) + v_k beta'(-k) を代入する
E1 の a(k) = u g(k) + v g'(-k) を代入して
```

#### 多ルール同時代入（複数演算子を同時に置き換える）

```
E1 の a(k) に X を、b(k) に Y を代入する
```

`a` と `b` が別演算子の場合に適用されます。順次ではなく**元の式**に対して同時に適用されます。

#### スピン添字付き 2添字演算子の代入

```
E1 の c(k,↑) に u g(k,↑) + v g'(-k,↓) を、c(-k,↓) に u g(-k,↓) - v g'(k,↑) を代入する
E1 の a(k,R) に (1/sqrt(2))*(b(k,x) - i*b(k,y)) を、a(k,L) に (1/sqrt(2))*(b(k,x) + i*b(k,y)) を代入する
```

| スピン記法 | 内部正規化 |
|---|---|
| `↑` / `\uparrow` | `u` |
| `↓` / `\downarrow` | `d` |
| `σ` / `\sigma` | `s` |
| `T`, `L`, `R`, `x`, `y` | そのまま（1文字）|

スピン添字を指定した場合、`c(k, ↑)` のみが置き換えられ `c(k, ↓)` は変更されません。

#### Nambu ペア方式（正負の運動量で異なる置換）

```
E1 の c(k) に rhs1 を、c(-k) に rhs2 を代入する
```

`c(k)` と `c(-k)` を独立に置換します（`operator_substitute_pair` 操作）。

#### 注意

- **SymPy 式への代入**：`E2` が代数インデックス演算（`normal_order`・`wick_contract` 等）の結果（SymPy チェーン式）のとき、`substitute` はそのまま SymPy 側で処理されます。演算子順序は保持されます（`k→-k`・`k→k+q`・スカラー係数への代入も対応）。
- **QA 式（`_nb_E2` 形式）への連鎖代入は不可**：`E2` が `normal_form()` の QA 結果のとき、`E2` を直接対象にできません。元の式（例 `E1`）に多ルール代入を使ってください。
- **虚数単位 `i`**：RHS 中の `i` は自動的に Julia の `im` に変換されます（`i` が未宣言の場合）。

### 粒子名およびスピン演算子の定義（Phase 0.5 — Julia セッション専用）

素粒子・準粒子名と「消滅/生成演算子」の組み合わせ、あるいは「スピン演算子」「パウリ行列」などの表現を直接指定すると、統計性（ボソン/フェルミオン/非可換スピン代数）を自動判定して定義操作にルーティングし、LLM を経ずに即時実行します。

スピン演算子 `S` を定義すると、派生するすべての演算子 `Sx`, `Sy`, `Sz`, `Sp`, `Sm` も自動的に `"noncommutative"` (非可換) として登録されます。

**スピン成分のサイト付き表記（入力書法）:**

| 書法 | 例 | 備考 |
|---|---|---|
| サフィックス | `Sz_1`, `Sx_k` | 成分を識別子のサフィックスとして表記 |
| 上付き | `S^z_1`, `S^x_k` | 成分を LaTeX 上付きで表記 |

`S_z_1`（S の直後にアンダースコア）は**使用不可**。`S_z` は `Sz` とは別の識別子になる。  
**最初に使用した書法がセッション全体の表示スタイルとして記憶される**（`SymbolContext.spin_display_styles`）。

| 入力例 | 実行される操作 |
|---|---|
| `a'(k) を光子の生成演算子とする` | `define_boson_operator(name=a)` |
| `電子の消滅演算子を c と定義して` | `define_fermion_operator(name=c)` |
| `フォノンの消滅演算子を b と定義する` | `define_boson_operator(name=b)` |
| `h を正孔の消滅演算子とする` | `define_fermion_operator(name=h)` |
| `S をスピン演算子とする` | `define_spin_operator(name=S)` |
| `σ をパウリ行列と定義する` | `define_spin_operator(name=σ)` |

対応粒子は `models/particles.json` に定義（フェルミオン: 電子・陽電子・陽子・中性子・ミューオン・タウ粒子・ニュートリノ・各クォーク・正孔 他、ボソン: 光子・フォノン・マグノン・クーパー対・ヒッグス粒子 他、計 32粒子）。

### 演算子ペア置換・Wick 収縮（Julia セッション専用）

| 操作 | 記法 | 用途 |
|---|---|---|
| `pair_substitute` | `pair_rules` に `{pattern, rhs}` のリスト。`pattern` は式入力と同じユーザー記法（`c'(k)*c(k)`、多添字可）。順序非依存・raw bares も後方互換。**2演算子ペア専用**（3個以上は wick_contract へ誘導） | 一致する2演算子積を c-number に置換 |
| `wick_contract` | `contraction_rules` に `{op1, op2, value}` のリスト | n 次演算子積から指定ペアを収縮、残余 (n-2) 次積が残る |
| `declare_symmetry` | `{name, symmetry}` | 2引数関数の対称性を宣言（`antisymmetric`/`symmetric`/`hermitian`） |
| `expand_neighbor_sum` | `{body, index1, index2, n_sites?, boundary, pair_constraint, lattice_shape, edges}` | ペア和を明示和に展開。`pair_constraint`: `nearest_neighbor`(1D既定)/`all_pairs`(分子Σ_{i<j})/`square_2d`/`triangular_2d`/`honeycomb_2d`/`custom`。2Dは`lattice_shape="LxxLy"`・境界 open/periodic/cylinder。`custom` は `edges="(1,2),(2,3)"`。F-1。Julia/Maxima両対応 |

**`pair_substitute`（2演算子積 → c-number）:**

```python
# 数密度演算子 c'(k)*c(k) を n_k に置換（ユーザー記法）
engine.run_operation("pair_substitute", {
    "expr": "E3",
    "pair_rules": [
        {"pattern": "c'(k)*c(k)", "rhs": "n_k"},
        {"pattern": "c'(-k)*c(-k)", "rhs": "n_k"},
    ]
})
# 旧来の raw bares 表示（c†(k) c(k)）も後方互換で受け付ける
```

**`wick_contract`（n 次積からペアを抽出）:**

収縮値テンプレート記法：
- `"rho"` — plain scalar（Phase 1）
- `"rho(INDEX1,INDEX2)"` — 運動量のみ（Phase 2）
- `"rho(INDEX1_0,INDEX1_1,INDEX2_0,INDEX2_1)"` — 運動量＋スピン 4添字（推奨）

標準記法：
- **ρ** = $\langle a^\dagger(k,\sigma)\,a(p,\tau)\rangle$ — 一粒子密度行列（通常平均）
- **κ** = $\langle a(k,\sigma)\,a(p,\tau)\rangle$ — 異常密度行列
- **κStar** = $\langle a^\dagger(k,\sigma)\,a^\dagger(p,\tau)\rangle$ — 異常密度行列共役

```python
terms = engine.wick_contract("E3", [
    {"op1": "c'", "op2": "c",
     "value": "rho(INDEX1_0,INDEX1_1,INDEX2_0,INDEX2_1)"},
    {"op1": "c",  "op2": "c",
     "value": "kappa(INDEX1_0,INDEX1_1,INDEX2_0,INDEX2_1)"},
    {"op1": "c'", "op2": "c'",
     "value": "kappaStar(INDEX1_0,INDEX1_1,INDEX2_0,INDEX2_1)"},
], include_constant=False)
# → 全収縮項が個別の Expression として返る
```

**`declare_symmetry`（κ の反対称性を宣言）:**

```python
engine.run_operation("declare_symmetry", {"name": "kappa", "symmetry": "antisymmetric"})
# 以後の wick_contract で kappa(j,i) → -kappa(i,j) に自動正規化
```

**`collect_by_operators`（演算子を共通因子としてまとめる）:**

演算子積を共通因子として係数を合算します。Bogoliubov 変換後などで式をまとめる際に常に使います。
代数インデックス（`-k`, `k+q`）を含む SymPy 式に対して動作します。

```python
# -u²ξ·c†(k)c(-k) + v²ξ·c†(k)c(-k) → (v²-u²)ξ·c†(k)c(-k)
engine.run_operation("collect_by_operators", {"expr": "E3"})
```

**`collect_spin_pm`（Sx/Sy 表示を Sp/Sm 記法で整理・表示専用）:**

EOM や交換子計算後の Sx/Sy/Sz 表示を、ユーザー定義の Sp/Sm 記法に整理します。
**表示専用**（式番号は割り当てられません）。計算の継続は元の QuExpr から行います。

```
# まず H-1 で Sp/Sm を定義する
Sp = Sx(1) + im*Sy(1)         ← E3 として登録
Sm = Sx(1) - im*Sy(1)         ← E4 として登録

# EOM 計算後に整理
E5 = heisenberg_eom(Sm_op, H)
「E5 を Sp/Sm でまとめ直して」
→ collect_spin_pm(expr=E5, sp=E3, sm=E4)
→ (-ω/2) Sm(1) a(1)  ← 表示のみ（式番号なし）
```

多サイト系（Heisenberg モデル等）での Sz をコンテキスト演算子として自動保持:

```
H = J*(Sx(1)*Sx(2) + Sy(1)*Sy(2) + Sz(1)*Sz(2))
heisenberg_eom(Sm_op, H) → -2Ji·Sx(1)·Sz(2) + 2Ji·Sx(2)·Sz(1) + ...
「Sp/Sm でまとめ直して」  → (-2Ji) Sm(1) Sz(2) + (2Ji) Sm(2) Sz(1)
```

**`relabel_index`（インデックスの付け替え）:**

ダミー変数インデックスを別の記号に付け替えます。SymPy 式（`k→-k`等の代数インデックス後）にも対応。

```python
# E2 の k を p に付け替え
engine.run_operation("relabel_index", {"expr": "E2", "old_index": "k", "new_index": "p"})
```

**`re_evaluate_expr`（自己無撞着反復）:**

定義済みシンボルを式に代入して再評価します。QA 式・SymPy 式の両方に対応しています。
複数の置換は順次適用され、SymPy 式の場合は演算子順序が保持されます。

```python
# H_MF (E5) のパラメータ rho を更新して再評価（QA 式・SymPy 式どちらでも動作）
new_expr = engine.re_evaluate_expr("E5", {"rho": "0.42"})

# SymPy 式（代数インデックス操作後）のスカラー係数を更新
# e.g. xi*cd(k)*c(-k) → xi=0 で消去
new_expr = engine.re_evaluate_expr("E3", {"xi": "0"})

# 複数シンボルを同時更新
new_expr = engine.re_evaluate_expr("E5", {"xi": "xi_k", "Delta": "Delta_0"})
```

### 摂動展開（Maxima + Julia 両対応）

| 操作 | 入力 | 出力 |
|---|---|---|
| `extract_order` | `expr`（スカラー/演算子）, `var`, `order` | λⁿ の係数（演算子もしくはスカラー） |
| `truncate_series` | `expr`（スカラー/演算子）, `var`, `order` | order 次以下の項のみ残す |

```python
# A型: H = H₀ + ε V + ε²W から各次を取り出す（演算子にも対応）
engine.run_operation("extract_order",  {"expr": "E1", "var": "ep", "order": 1})  # → V
engine.run_operation("truncate_series",{"expr": "E1", "var": "ep", "order": 1})  # → H₀ + ε V
```

**演算子級数への対応：** `H = ξ c†c + ε (c†(1)c(2) + c†(2)c(1))` のような QuExpr（量子演算子の摂動展開）にも透過的に動作。演算子の順序・反交換関係は `normal_form` により保持される。

**スカラーの場合** は Maxima・Julia 両対応（`taylor_expand` の結果に対しても使える）。

---

### 演算子テイラー展開（Julia セッション専用）

| 操作 | 説明 |
|---|---|
| `operator_taylor_expand` | `f(H) ≈ Σ aₖ Hᵏ`。スカラー関数 f を演算子 H に適用 |

```python
engine.run_operation("operator_taylor_expand", {
    "func":  "exp(-beta*x)",   # スカラー関数（x が H に置き換わる）
    "op":    "E1",             # 演算子 H の式番号
    "var":   "x",              # f 中のダミー変数
    "order": 3                 # 展開次数
})
```

| 例 | 結果 |
|---|---|
| `exp(-βH)`, `H = ξ c†c`（射影） | `1 + (e^{-βξ}-1)·c†c` へ収束 |
| `exp(-βH)`, `H = t(c†c(12)+c†c(21))`（ホッピング） | 対角・非対角・4体項が各次に出現 |
| `1/(1-H)` | resolvent 展開 `1 + H + H² + H³ + ...` |
| `cos(H)` | `1 - H²/2 + H⁴/24 - ...`（偶数冪のみ） |

**仕組み：**
1. `func` を `var` でスカラー Taylor 展開 → 係数 `aₖ`（Symbolics）
2. QuantumAlgebra で `Hᵏ = H·H·…·H`（`normal_form` で反交換関係適用）
3. `Σ aₖ · Hᵏ` を組み立てて返す

射影演算子（`n² = n`）等の代数的性質は `normal_form` が自動適用する。

---

## 操作の指示例

### シンプルな例

```
E1 を展開して
E2 を x で微分する
E3 を x について因数分解
```

### 引数が必要な例

```
E1 を x=0 の周りで 3 次のテイラー展開
E2 を 0 から 1 まで x で積分
E3 の x についての極限（x → ∞）
E4 の両辺に 2 をかけて
E1 の x に E2 を代入
```

### 複数式を組み合わせる例

```
E1 と E2 を足して
E2 から E1 を引いて
E1 と E3 を掛けて
E2 を E1 で割って
```

### 仮定の例

```
x > 0 を仮定する
m は正の実数とする
```

### 関数定義の例

```
f(x) = sin(x) を定義する
H = p^2/(2m) + V(x) を定義する
```

---

## LLM supervisor の動作

指示が曖昧な場合（「整理して」「簡単にして」など）は候補が提示されます。

```
「整理する」は複数の意味があります：
  1. 一般的に簡約する
  2. 有理式として約分・通分する
  3. 展開して和の形にする
  4. 因数分解して積の形にする
  5. 指定した変数についてまとめる
```

番号または言葉で選択すると操作が実行されます。
