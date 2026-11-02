# 数値仕様 IR（拡張設計）— スライス2「数値計算 ChatBot」の中核

**2026-11-02 現行実装追記:** 本書のIRは実装済み。以下は初版の設計経緯を含む。
`presentation.kind="cartesian"` の型付きレイヤーを追加し、複数式・2軸の仕様を決定的に描画する。
`codegen_route_for_spec(spec)` はcartesian presentationを優先し、旧flat射影だけでは経路を決めない。
LLMコード生成にも完全な `spec` と凍結済み `plot_sources` を渡す。
一般のheatmap/surface等すべてが決定的になったわけではない。
[36_cartesian_plot_layers.md](36_cartesian_plot_layers.md) に有効なJSON例と検証制限を集約。

**初版:** 2026-07-09（草案・議論用）
**初版時status:** DRAFT。論点A（コード生成の前段＝仕様確定の中間表現）の設計提案のみ。
コードは未変更。対話UX（論点B）・LLM 配線・codegen 変更は本書の範囲外。
**関連:** [22_numeric_layer_design.md](22_numeric_layer_design.md)（実装済みの縦割り）、
[23_plan_workflow_design.md](23_plan_workflow_design.md)（`plan_adviser` の対話→構造化案→承認パターン）、
experiments/theory_to_numeric_design_principles.md (reference omitted from the public snapshot)（§2 対話→IR、§3 役割分離）、
experiments/theory_to_numeric_implementation_memo.md (reference omitted from the public snapshot)（§2.1 NumericTask、§7 problem_type、§12 会話ログで保持しない）

---

## 0. なぜ書くか（スライス2の見積もり違いの反省）

スライス2を「スライス1のモーダルに coder 選択とレビュー承認を足すだけ」と見積もったのは誤り
だった。原因は `NumericTask` が実質 parameter_sweep 専用の形（`problem_type` ＋ 単一
`sweep_parameters` ＋ `fixed_values` ＋ スカラー `target_quantity`）で、**「利用者が何を
計算したいか」を表す器がない**こと。利用者は 1 次元掃引を望むとは限らない — 収束まで反復、
2 次元グラフ、行列の数値対角化…。設計原理 §2 が言う「LLM が対話で仕様を詰める」の**吐き先＝
仕様 IR** をまず設計しないと、対話 UX を作っても手戻りする。本書はその IR を定義する。

## 1. 要件（設計原理・実装メモから確定済み。再議論しない）

1. **対話の産物は会話ログではなく構造化 IR**（メモ §12）。ChatBot のゴールは「埋まった仕様」。
2. IR は次の 4 つの便益を支えねばならない（設計原理 §2）: **① solver 差し替え ② コード再生成
   ③ 別言語移植 ④ 分岐間比較**。この 4 つが「構造化 vs 会話ログ」の判定基準。
3. **来歴は IR に含む**: 元式 ID・仮定・分岐（既存 `NumericTask` フィールドを踏襲）。
4. **役割分離**（設計原理 §3）: IR は研究者が承認する対象。LLM は IR を提案するが最終権限を持たない。
5. **既存の parameter_sweep/plotting 決定的経路を壊さない**（additive・後方互換）。

## 2. 設計方針 — 「型ごとに欄を足す」のをやめる

拡張性の敵は `problem_type: Literal[...]` ＋ 型ごとに散らばる平坦フィールドである。新しい
計算を足すたびにモデル改変（Literal 追加・新フィールド追加・codegen 分岐追加）が要る。

代わりに **計算の構造を直交ファセットに分解**する。計算とは抽象的に:

> **どの入力**を（どう動かして）→ **どの出力**を、**どの手法**で得て、**どう提示するか**

この 4 ファセット（＋来歴）で計算を記述すれば、`problem_type` は「意味を決める主キー」から
「テンプレート/solver 既定を選ぶ**ラベル**」に格下げされる。**計算の形は inputs/outputs/method
が担い、problem_type が担わない** — これが拡張性の核心。新しい型は「新しいデータ（入力の
役割・出力の形・手法の組み合わせ）」であって「新しいスキーマ」ではない。

**柔軟 vs 構造のトレードオフ（明示）**: 完全自由な JSON は最も柔軟だが便益①〜④を捨てる
（diff も再生成も比較もできない只の会話ログ）。本設計は**固定された枠（intent/inputs/
outputs/method/presentation）の中に、値としての開放（kind・shape・params は開いた文字列/
dict）**を置く「甘い中間点」を採る。枠は構造を、値は柔軟性を担う。

## 3. ファセット分解（提案する IR の形）

`NumericTask` に **`spec: NumericSpec | None`** をネストして追加する（additive。旧タスク・
旧セーブは `spec=None` のまま読める）。`NumericSpec` は intent と4つの構造ファセット:

```
NumericSpec
├── intent      : IntentSpec        何をしたいか（ラベル＋自然文の目標）
├── inputs       : list[InputVar]    独立変数と、その動かし方
├── outputs      : list[OutputSpec]  出てくる量と、その「形」
├── method       : MethodSpec        solver・停止条件・許容誤差・近似
└── presentation : PresentationSpec  結果の見せ方（図種・表・スカラー）
```

来歴（`source_expression_ids`・`source_assumption_ids`・`source_branch_id`・
`source_canonical`/`source_latex`）は **`NumericTask` 側に据え置く** — spec は「計算の記述」、
task は「その計算の来歴と実行状態」を持つ、という責務分離。

### 3.1 IntentSpec — 何をしたいか
```
IntentSpec
├── kind : str    既知種のレジストリ参照（下記）。未知でも可＝LLM 経路の担当
└── goal : str    自然文の目標（LLM が対話とノート文脈から言語化。人間可読の来歴）
```
- `kind` は **Literal ではなく開いた文字列＋レジストリ**（`domains/` config や plan workflow
  registry と同じ思想 — CLAUDE.md #12）。既知種（`parameter_sweep` / `plotting` /
  `root_finding` / `numeric_integration` / `eigenvalue_problem` / `function_evaluation` …）は
  レジストリに登録され、「既定 method・想定 output shape・どの codegen 経路が担うか」を宣言する。
  **未知の kind はエラーにせず**、LLM codegen 経路（レビュー必須）に回る。
- `goal` は codegen プロンプトと結果還流の人間可読ラベルに使う。

### 3.2 InputVar — 独立変数と動かし方（`fixed_values`＋`sweep_parameters` を統合）
```
InputVar
├── name : str
├── role : "sweep" | "fixed" | "free" | "boundary"   （下記）
└── spec : dict     role ごとに開いた形
```
- `role="sweep"` → `spec={min,max,points,scale?}`。**掃引変数の本数が次元** — 1 本＝1D、
  2 本＝2D。「2 次元グラフ」は「sweep が 2 本」で自然に表現され、スキーマ変更不要。
- `role="fixed"` → `spec={value}`（スライス1の `fixed_values` と等価）。
- `role="free"` → 反復の初期値等 `spec={initial}`（root_finding/収束計算の起点）。
- `role="boundary"` → 境界条件 `spec={at, value}`（メモ §2.1 `boundary_conditions` の構造化）。
- role は「変数がどう計算に参加するか」で本質的に有界なので**小さな閉集合**にする（値の
  検証と codegen 分岐が効く）。ここが唯一「開くか閉じるか」を後で見直しうる箇所。

### 3.3 OutputSpec — 出力の「形」（＝欠けていた expected_output_kind）
```
OutputSpec
├── name  : str
├── shape : str          "scalar" | "series" | "field" | "vector" | "matrix" | "trace" | …（開放）
├── over  : list[str]    どの入力に対する量か（series は [x]、field は [x,y]、trace は [iteration]）
└── role  : str          "primary" | "diagnostic" | …（任意。主結果か副産物か）
```
- **outputs が list** なのが要点。行列対角化＝`[{eigenvalues, vector}, {eigenvectors, matrix}]`
  のような**複数出力**が自然に載る（スカラー 1 個前提の `target_quantity` では表せなかった）。
- `shape` は開いた文字列（レジストリで既知種を管理）。`over` が独立変数への参照を張るので、
  次元・軸の意味が構造として残る（→ 提示や比較で使える）。

### 3.4 MethodSpec — 手法・停止条件（＝「収束まで」の居場所）
```
MethodSpec
├── name   : str    "direct_eval" | "fixed_point" | "newton" | "quadrature" | "numpy_eig" | …（開放）
└── params : dict   tol・max_iter・step・rtol… solver 依存。開いた dict
```
- 「収束するまで計算」＝`method={name:"fixed_point", params:{tol:1e-8, max_iter:1000}}`。
  **停止条件は params の中に構造として入る**（会話ログに埋もれない）。
- `solver_candidates`（メモ §2.1）は「MethodSpec の候補リスト」として自然に一般化できる
  （Adviser が複数手法を提案 → 研究者が選ぶ、は plan_adviser の候補モーダルと同型）。

### 3.5 PresentationSpec — 見せ方（output shape と分離）
```
PresentationSpec
├── kind    : str    "line" | "heatmap" | "contour" | "surface" | "table" | "scalar" | "convergence" | …
└── options : dict   軸ラベル・対数軸・カラーマップ… 開いた dict
```
- output shape と**あえて分離**する: 同じ 2D field でも heatmap/contour/断面の選択があり、
  同じ trace でも収束プロットの見せ方が複数ある。「何が出るか」と「どう見せるか」は別問題。

## 4. 3 つの実例を IR で表現（スキーマ変更ゼロで表せることの実証）

**① parameter_sweep（スライス1・y=sin(kx)e^{-ax} を x で掃引）**
```
intent  : {kind:"parameter_sweep", goal:"evaluate y over x"}
inputs  : [{x, sweep, {min:0,max:10,points:100}}, {k, fixed, {value:2}}, {a, fixed, {value:0.3}}]
outputs : [{y, series, over:[x], primary}]
method  : {name:"direct_eval", params:{}}
present : {kind:"line"}
```

**② 2 次元グラフ（z=f(x,y) を x,y で掃引）**
```
inputs  : [{x, sweep, {...}}, {y, sweep, {...}}, …fixed]
outputs : [{z, field, over:[x,y], primary}]
present : {kind:"heatmap"}
```
→ 差分は「sweep が 2 本・output が field」だけ。**スキーマは同一**。
これはIRの表現例で、heatmapの決定的実装を意味しない。現行の決定的2軸作図は
`presentation.kind="cartesian"` と `layers` で指定する（docs/36）。

**③ 収束まで反復（g(x)=x の不動点）**
```
intent  : {kind:"root_finding", goal:"iterate until |Δx|<tol"}
inputs  : [{x, free, {initial:x0}}, …fixed]
outputs : [{x_star, scalar, primary}, {trace, trace, over:[iteration], diagnostic}]
method  : {name:"fixed_point", params:{tol:1e-8, max_iter:1000}}
present : {kind:"convergence"}
```
→ 停止条件は method.params に構造化。反復履歴は trace 出力として残る。

**④ 行列の数値対角化**
```
intent  : {kind:"eigenvalue_problem", goal:"eigenvalues/vectors of M at these params"}
inputs  : […fixed params to substitute]
outputs : [{eigenvalues, vector, primary}, {eigenvectors, matrix, primary}]
method  : {name:"numpy_eig", params:{}}
present : {kind:"table"}
```
→ 複数出力が list で自然に載る。

## 5. 後方互換（スライス1 を 1 行も壊さない）

- `spec` は additive optional。旧タスク・旧セーブは `spec=None` で読める（既存の
  「additive キー・旧ファイル互換」方針そのまま）。
- 純関数の**双方向射影**を用意する:
  - `flat_to_spec(task)`: スライス1 が作った平坦フィールド（problem_type/fixed_values/
    sweep_parameters/target_quantity）から faceted `spec` を導出（parameter_sweep の場合）。
  - `spec_to_flat(spec)`: faceted spec が parameter_sweep 相当なら平坦フィールドに落とし、
    **既存の決定的 `templates.generate_code` をそのまま再利用**（当面 codegen を書き換えない）。
- 初版では決定的経路をflat射影で動かした。現行のCartesian作図は完全なfaceted specを
  直接読む。flat射影の `problem_type="plotting"` は互換用で、複数レイヤーの描画意味を表さない。
  LLM経路にもfull specを渡し、method/presentationを射影で失わない。

## 6. 拡張の仕方（新しい計算＝データ、スキーマ改変なし）

例: **モンテカルロ積分**を足す →
- intent kind レジストリに `"monte_carlo_integration"` を 1 エントリ登録（既定 method・
  想定 output shape・担当 codegen 経路を宣言）。
- IR は `inputs:[{n_samples, fixed, {value:1e6}}, …]`, `outputs:[{integral, scalar},
  {stderr, scalar, diagnostic}]`, `method:{name:"mc", params:{seed:0}}` で表せる。
- **`NumericSpec` のクラス定義は不変**。増えるのはレジストリの 1 行と（決定的にしたければ）
  テンプレート 1 本だけ。未知種のままなら LLM 経路が担う。

## 7. plan_adviser パターンの流用（論点B への布石・本書では設計のみ）

対話→ faceted spec 案→承認、は C-Ext-5 `plan_adviser` の「LLM が JSON 契約で提案 → 厳格
パース（壊れは棄却）→ `validate` 門番 → 候補/確認モーダル → 研究者承認」と**構造が同一**。
論点B ではこの機構を、契約＝ `NumericSpec` JSON、門番＝ `validate_numeric_spec`（下記）に
差し替えて被せる見込み。**IR が決まれば門番の検証項目が決まり、対話 UX はその上に載る。**

`validate_numeric_spec`（実装は論点B）が見るべき最低限:
- inputs の name が式の自由シンボルに含まれるか／sweep spec の min<max・points≥2／
  fixed の value が数値リテラル（スライス1 の敵対的入力規則を踏襲）／outputs.over が
  inputs の name を参照しているか／method.params の型。**壊れた spec は実行前に落とす。**

## 8. 小論点の決定（2026-07-09）

1〜2 は**提案どおり採用（決定・2026-07-09）**。シナリオと実需要で見直しうる（下記の見直し条件）。
3 は後方互換要件（§1-5）から実質確定。4〜5 は IR の形を縛らない**実装時の順序判断**。

1. **`intent.kind` = 開放＋レジストリ（決定）。** 未知種はエラーにせず LLM 経路が担う。決定的
   経路に載せたい種だけレジストリへ昇格。**見直し条件**: 未知種が実運用で頻出し、決定的化の
   要求が集中したら、その種をレジストリ既知種へ昇格（スキーマは変えない — データ追加のまま）。
2. **`InputVar.role` = 閉集合（sweep/fixed/free/boundary）（決定）。** 検証・codegen 分岐が効く。
   **見直し条件**: 上記 4 役割に収まらない参加形（例: 確率変数のサンプリング軸、時間発展の
   時間軸）が具体シナリオで必要になったら、その時点で 1 役割だけ足す（開放化はしない）。
3. **`spec` は `NumericTask` にネスト・additive（確定）。** §1-5（スライス1 不破壊）から自動的に
   決まる。作り替えは後方互換コストが高い。
4. **決定的codegenのfaceted対応は需要駆動。** 当初はflat射影でテンプレートを流用した。
   2026-11-02にCartesian表示の直接処理を追加。一般の表示形式をすべて実装したわけではない。
5. **`solver_candidates` → MethodSpec 候補リストへの一般化は Adviser 複数手法提示が要るときに
   （実装時判断）。** 単一 method で開始。plan_adviser の候補モーダルを流用予定。IR の形には影響しない。
