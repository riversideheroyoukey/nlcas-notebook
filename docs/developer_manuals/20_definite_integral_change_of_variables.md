# 定積分の変数変換（ヤコビアン）設計仕様 — INTEGRAL-COV

実践ループ **S-o** の本命機能。**範囲つき定積分そのもの**を別の座標へ変数変換し、被積分関数だけでなく
**積分範囲も新変数に書き換わった定積分**を表示してから評価する。本書は、なぜこの形にしたか（設計判断）と、
どこをどう拡張するか（拡張点）を一本にまとめる。実装の所在は §7、利用者向けの操作手順は
[user_guide.md](../user_guide.md) の「定積分の変数変換」、再現シナリオは
[scenario_coordinate_jacobian.md](../scenarios/scenario_coordinate_jacobian_jp.md)。

## 1. 動機と要件

変数変換は定積分計算でごく日常的（高校段階の置換積分から多重積分の極座標・球座標まで）。当初の実装は
`change_integration_variables` が **被積分関数×|J| を返すだけ**で、積分範囲を持たず、何がどう変換されたか
利用者に伝わらなかった。利用者が要求した本来の 4 段は：

1. **範囲が設定された定積分の式**を利用者が与える
2. **旧変数→新変数の変換**を指示する
3. **新しい積分範囲が示された定積分の式**が計算され、表示される
4. 変換された定積分が**評価**される

この 4 段を経ないと「何がどう変換されたか」が理解できない、という要求が設計の核心である。

## 2. 中心となる設計判断

### 2.1 表示専用・未評価の定積分オブジェクト（deferred-eval）

ステップ 1 の「範囲つき定積分」は **CAS に渡さず表示だけ**できる必要がある（利用者要求）。これを
`Expression.integral_descriptor` という記述子で表現する。これは既存の `lattice_descriptor`（未展開の
ボンド和 Σ_C）と**同じ遅延評価設計**で、

- 式は ∫…∫ の形で表示されるが、`canonical` は CAS 実行可能な値ではない（登録時に CAS を呼ばない）、
- 変数変換でこの記述子が書き換わり、
- 評価ステップで初めて反復積分を CAS に投げる。

> **なぜ noun-form（`'integrate(...)`）にしないか:** Maxima の quote 形でも未評価表示はできるが、(a) 表示
> 整形が Latexify に依存し制御しづらい、(b) 領域述語（円板・球）を「∫ の入れ子」で表せない、(c) バックエンド
> 非依存にしたい。記述子（Python 構造）＋表示 LaTeX を自前生成する方が、Core Principle #1（CAS 構文を
> 見せない）と #10（バックエンド置換可能）に合致する。

### 2.2 入力は 2 形式（領域を式から抽出）

利用者にわかりやすい入力として、**LaTeX の定積分記法**と**領域述語**の両方を受ける。

| 形式 | 例 | 抽出される領域 |
|---|---|---|
| LaTeX 多重積分 | `\int_{0}^{1} \int_{0}^{1} (x+y) \, dx \, dy` | 矩形（定数境界） |
| 領域述語 | `x^2+y^2<=1 の領域で f を定積分` | 円板（半径 1） |
| 領域述語（3D） | `x^2+y^2+z^2<=1 の領域で f を定積分` | 球（半径 1） |

LaTeX の入れ子 ∫ は**外側（左の `\int`）が最後に書かれた微分を束縛する**慣習でペアリングする
（`\int_a^b\int_c^d f\,dx\,dy` → 外側 (a,b)↔dy、内側 (c,d)↔dx）。`atan2(v,u)` 等の関数引数カンマは
深さ優先のトップレベル分割で保護する。

### 2.3 新しい積分範囲の解決順序

領域→領域の写像は一般に自動導出できない（矩形を線型変換すると平行四辺形になり軸平行でない）。そこで
**できるだけ算出し、ダメなら段階的にフォールバック**する。優先順位は：

1. **利用者の明示上書き** — 新変数指定に範囲を添える（`r:0..1, theta:0..2π`）。意図的指定なので最優先。
2. **自動テーブル** — 標準座標変換 × 標準領域。極座標×円板 → `θ:0..2π, r:0..R`、球座標×球 →
   `φ:0..2π, θ:0..π, r:0..R`。変換の RHS 形と領域 kind から決定論的に算出。
3. **LLM 提案** — `_llm_derive_bounds`（llm_client があるとき）。新範囲を JSON で問い合わせる。
4. **利用者へ要求** — `needs_bounds` ステータスを返し、範囲指定を促す。

> **順序の意図:** 1 は「LLM の提案に満足できないときに利用者が値を入れる」ための上書き経路で、明示された
> 以上それを最優先で使う。利用者が指定しない通常時は **自動算出 → LLM → 要求** の順に降りる。

### 2.4 測度の符号

`change_integration_variables` は **符号付きの $\det J$** を挿入する（絶対値ではない）。標準座標の正方向
順では $r,\ r^2\sin\theta>0$ なので、定積分の値を誤らせない。一般変換で向きを反転する新変数順を選ぶと
負の係数が出るため、**新変数は $\det J>0$ となる順序で並べる**（教科書の慣習）。`abs()` で包むと
`abs(r)` のように clean な簡約を妨げるため、blanket-abs は採らない。

## 3. ワークフロー（4 段）と期待値

| 段 | 入力（自然言語） | 結果 |
|---|---|---|
| 1 | `x^2+y^2<=1 の領域で x^2+y^2 を定積分` | $\iint_{x^2+y^2\le1}(x^2+y^2)\,dx\,dy$（未評価） |
| 2-3 | `E1 を x,y から r,theta へ x=r*cos(theta), y=r*sin(theta) で変数変換して` | $\int_{0}^{2\pi}\!\int_{0}^{1} r^3\,dr\,d\theta$（新範囲を自動算出） |
| 4 | `この定積分を評価して` | $=\dfrac{\pi}{2}$ |

球座標·単位球の体積 → $4\pi/3$、一般ヤコビアン（LaTeX 入力＋利用者指定範囲、$\det J=5$）→ $35/2$。

## 4. ビルディングブロック（被積分関数のみ・単変数）

定積分オブジェクトを介さず、被積分関数だけを変換する経路も決定論で残す（VARCHG-NL）：

- `x^2+y^2 を x,y から r,theta へ x=r*cos(theta), y=r*sin(theta) で変数変換して` → 生 Jacobian 積 →
  `簡約して` → $r^3$（`change_integration_variables` → `simplify`）。
- 単変数定積分 `E3 を r で 0 から 1 まで定積分して`（`integrate_definite`）。π 境界は
  `BackendHandler.normalize_bound`（π/`\pi` → `%pi`）で正規化。
- 反復積分の表示は `lhs_formatting`：`d\theta`（ギリシャ）・`2\pi`（`%pi`→display）・`_rhs_of_latex` で
  内側の等式を入れ子にしない。

## 5. バックエンド対応と CAS 能力差

現状は **Maxima** が対象（standard セッション）。簡約は SIMP-UNIFY 済みの `simplify`（trigsimp∘ratsimp）で
$\sin^2+\cos^2=1$ を畳む。Julia（Symbolics.jl）は 3 次元・球座標の入れ子ピタゴラス簡約を完了できない
（CAS 能力差・キーワードの問題ではない）。詳細は [scenario_coordinate_jacobian.md](../scenarios/scenario_coordinate_jacobian_jp.md)。

## 6. 拡張点

- **領域テーブルの拡充**：円柱座標、楕円領域、一般の半径記号などを
  `integral_form.derive_new_bounds` / `classify_transform` に追加。
- **領域述語の拡張**：`a<=x<=b, c<=y<=d` の矩形、`|x|+|y|<=1` 等。
- **Julia バックエンド対応**：`build_eval_command` は Maxima `integrate` を組むため、バックエンド別の
  eval コマンド生成に切り出す（Backend Encapsulation Rule）。
- **測度の絶対値オプション**：向き非依存にしたい場合の `|det J|` モード。

## 7. 実装の所在

| 要素 | 場所 |
|---|---|
| 記述子フィールド | `notebook/models/expression.py` `Expression.integral_descriptor` |
| 純ヘルパ（parse/display/eval/table） | `notebook/runtime/integral_form.py` |
| エンジン 3 メソッド | `notebook/engine.py` `register_integral_form` / `change_variables_integral_form` / `evaluate_integral_form`（＋ `_cas_value_and_latex`） |
| 入力・評価インターセプト | `ui/interpret_pipeline.py` `_try_integral_input` / `_try_evaluate_integral` |
| 変数変換（オブジェクト分岐＋単変数＋新範囲解決） | `ui/interpret_pipeline.py` `_try_change_vars` / `_try_definite_integral` / `_parse_new_var_spec` / `_llm_derive_bounds` |
| 境界正規化・反復積分表示 | `notebook/runtime/{backend_handler,maxima_backend_handler}.py` `normalize_bound`、`notebook/runtime/lhs_formatting.py` |
| テスト | `tests/test_integral_form.py`・`tests/test_integral_workflow.py`・`tests/test_change_vars_intercept.py` |

**関連:** [13_vector_field_design_spec.md](13_vector_field_design_spec.md)（ベクトル演算の式入力）、
[11_definition_substitution_rules.md](11_definition_substitution_rules.md)（H-1 と LHS エイリアス）。
