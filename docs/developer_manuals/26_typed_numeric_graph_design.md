# Typed Numeric Computation Graph 設計 — 数値計算の多段連鎖・型付き成果物・構造化ループ

**2026-11-02 現行実装追記:** 共通作図では `NumericTask.plot_sources` に全参照式
（式番号・canonical・LaTeX・分岐）のスナップショットを保存する。参照IDを `source_expression_ids`
とforkステップの `input_refs` に含め、stale判定にも参照式全体を含める。再生成は凍結した式を使う。
レイヤー別CSVは `NB_MANIFEST` の `series` 成果物、描画診断は `cartesian_sampling` として記録する。
PNGはfigure、CSVはtable、SVG/PDF/診断JSONは現行のartifact分類ではlogで、型の新設はない。
図・データの保存リンクもtimeline serializerで再読込時に復元する。
計算DAGの「数値グラフ」と描画機能の違い・完全な仕様は
[36_cartesian_plot_layers.md](36_cartesian_plot_layers.md) を参照。

**初版:** 2026-07-11（設計 G0）
**status:** **G1 実装済み（2026-07-11・同日）**: NB_MANIFEST 契約（テンプレ emit＋runner 両対応＋
LLM プロンプト）・typed artifact フィールド・`value_from` 連鎖束縛＋実行前型検査・
`source_hash` stale 検出（`/numeric/tasks` で報告）・evaluate 決定的テンプレ
（`function_evaluation` をレジストリで deterministic へ昇格）。
テスト29件（`tests/test_numeric_graph.py`）— 2ノード連鎖 E2E（a·b=12 → c²=144）・
複素値の型付けと実数スロットへの束縛拒否・manifest 経由の series 型付けを実 sandbox で検証。
**G2 実装済み（2026-07-11・同日）**: ①式差し戻し（`reflux_numeric_value` — H-1 等式機構で
`name = value` を登録、数値来歴をステップに刻印。`POST /numeric/reflux/{artifact_id}`）
②minimize_1d 決定的テンプレ（scipy 導入 — docs/22 §7 の予約どおり。境界解を還流メッセージで
明示）③**構造化ループ＋領域融合**（`self_consistent_loop`: 複数式本体＋更新写像＋停止条件を
1スクリプト・1 sandbox 実行に融合。非収束は honest failed step。Dottie 不動点・2式融合・
発散で実機検証）④matrix evaluate（Surface AST 段で行列を分解し要素ごとにスカラー
whitelist lowering を再利用 — 2要素行が Commutator 正規化に食われない。複素行列も正直に
型付け）。テスト計62件。
**G3 実装済み（2026-07-11・同日）**: ①eigen ノード（`eigenvalue_problem` を deterministic へ
昇格 — 0掃引=固有値〈昇順 ordering〉＋固有ベクトル、1掃引=固有値分枝の図+CSV〈S-y 避交差、
ギャップ=2Δ を実機検証〉。Hermite 判定で eigh/eig 切替・非 Hermite は診断と還流メッセージで
明示）②グラフビュー（`numeric_graph_view` — 参照から DAG を導出、source/value/reflux の
3種エッジ＋stale 枠。🕸 ヘッダボタン→SVG モーダル、ブラウザ実地検証済み）③分岐比較
ビュー（§10-c — 読み取り専用・緩いグルーピング・B-4 境界を明記）。テスト計79件。
guard（§9-④）と Julia 数値バックエンドは引き続き需要待ち。
**発端:** 指示書 `typed_numeric_computation_graph_design_instruction.md` (reference omitted from the public snapshot)（外部作成・
2026-07-11 検討）。同指示書は実装済みの docs/22/24/25 を知らずに書かれているため、
本書は既存アーキテクチャとの**和解版**である — 採らなかった提案と理由を §2 に明記する。
**関連:** [22_numeric_layer_design.md](22_numeric_layer_design.md)（実装済みの単段縦割り）、
[24_numeric_spec_ir_design.md](24_numeric_spec_ir_design.md)（faceted NumericSpec IR — 本書のノード記述の土台）、
[25_numeric_chatbot_design.md](25_numeric_chatbot_design.md)（対話→仕様確定）、
[23_plan_workflow_design.md](23_plan_workflow_design.md)（プラン層 — skip 意味論・停止安全の先例）、
evaluations.md (reference omitted from the public snapshot)（S-ac 測定台帳 — 契約遵守率の測定枠組み）

---

## 1. 目的 — 単発タスクから「型付き成果物が流れる多段計算」へ

実装済みの数値層は **1 タスク＝1 式＝1 実行**で完結し、出力は figure/table/NB_RESULT
（min/max スカラー）に限られる。研究の実務では結果が次の計算の入力になる:

```text
node_1: E(a,x) を x=1.0 固定で a∈[0,10] で最小化 → a_star, E_min
node_2: F(a,x,y) に a = a_star を代入して評価
node_3: H(k) を k=0.5 で行列として評価
node_4: node_3 の行列の固有値・固有ベクトル
```

現状の具体的な欠落（すべて実地で確認済みの痛点）:

- **成果物に数学的な型がない** — NB_RESULT はスカラー前提。行列・固有値リスト・
  掃引表・最適化結果を次段に渡す器がない（S-y: eigenvalues のリスト出力が直接
  フォーク不可、solve 経由の迂回が検証済み経路）。
- **連鎖の原始演算がない** — ある実行の結果値を別のタスクの固定値に束縛できない。
- **結果の理論層への式差し戻しが未実装**（docs/22 §13 フェーズ2 残項目）。
- **複数式にまたがる収束ループが表現できない**（§8 — 物性の自己無撞着計算の典型形）。

目標は「式評価器」ではなく、**理論層の Expression を参照し、数値条件の下で
評価・掃引・最適化・代入・後続計算を行う typed workflow** である（指示書 §1 の
この問題設定自体は全面的に採用する）。

## 2. 指示書からの設計変更（和解 5 点＋細目）

| # | 指示書の提案 | 本書の採用形 | 理由 |
|---|---|---|---|
| 1 | `NumericExperimentNode` を新設 | **ノード＝`NumericTask`＋`spec`（既存）**。足りない typed I/O だけを additive に追加 | 指示書のノード構造（operation / numeric_context / method / outputs）は docs/24 の faceted `NumericSpec` とほぼ 1 対 1。二重スキーマは docs/24 §0 で反省した手戻りの再演 |
| 2 | `expression_ref + version + hash + frozen_ir` | **スナップショット継続＋コンテンツハッシュ追加**（§7） | `Expression` は実質 append-only で版履歴を持たない。`source_canonical` スナップショットが既に frozen IR（canonical は IR の直列化形）。版機構は理論層への大手術で得るものが薄い |
| 3 | 数値評価は Julia（`backend:"julia"`） | **Python sandbox（`.venv-numeric`）継続** | sandbox-exec 方式 B は実機検証済み・確定（docs/22 §7）。Julia 数値バックエンドは `SandboxRunner` 差し替えの将来選択肢に留める |
| 4 | 全ノードを決定的実装として記述 | **型付けの主対象は出力契約（§4）**。codegen は既存のテンプレ/LLM 二経路のまま | この層の本来の目的はバイブコーディング経路（docs/22 冒頭）。契約を型付ければ、テンプレ・LLM・人手編集のどれが書いたコードでも graph 検証と連鎖が同じに機能する |
| 5 | `NumericComputationGraph` を独立保存オブジェクトに | **グラフ＝参照から導出されるビュー＋検証関数**（§6） | engine の 3 辞書＋参照フィールドで DAG は一意に導出できる。edges の二重帳簿は整合性維持コストだけ増える |

細目: (a) 指示書 §4.1 の `artifact_type` は既存 `NumericArtifact.artifact_type`
（code/result/figure/table/log — ファイル種別）と名前衝突するため、数学型は
**`value_kind`** と命名する。(b) `units` は本システムに単位系が存在しないため
スコープ外（metadata に置くのは自由）。(c) 成果物の置き場所は本リポジトリ規約
（フラットな `docs/NN_*.md`）に従い本書＝docs/26 とする。

## 3. Typed artifact — `NumericArtifact` の additive 拡張

新モデルは作らない。`NumericArtifact` に**値レベルの型情報**を additive に足す
（旧セーブは既定値でロード — 数値層の後方互換方針そのまま）:

```text
NumericArtifact（既存: artifact_id / task_id / artifact_type / content / path / metadata）
├── value_kind : str   "scalar" | "vector" | "matrix" | "ordered_list" | "table"
│                      | "sampled_function" | "optimization_result" | "trace" | …（開放）
├── value_type : str   "real_float64" | "complex_float64" | "integer" | …（開放）
├── shape      : list  [] | [n] | [m,n] | …（value_kind に応じて）
├── ordering   : dict  {kind:"sorted_by_value", ascending:true} 等。順序に意味がある場合のみ
└── value      : 小さな値の inline JSON（大きな値は従来どおり path のファイル）
```

- `value_kind` の語彙は **`OutputSpec.shape` の語彙と同一系**に揃える（spec が
  「出るはずの形」を宣言し、artifact が「実際に出た形」を持つ — §6 の検証で突き合わす）。
  docs/24 と同じく**開いた文字列＋レジストリ**で、既知種の互換規則（どの value_kind が
  どの入力 role に束縛できるか）をレジストリが宣言する。
- 既存の figure/table artifact は `value_kind` なし（既定 ""）のままでよい —
  型付けが要るのは**次段の入力になり得る値**だけ。

## 4. Typed manifest 契約 — NB_RESULT の拡張（本設計の要）

生成コードの出力契約を、単一スカラー前提の `NB_RESULT {json}` 1 行から
**成果物宣言リスト**へ拡張する:

```text
NB_MANIFEST {"artifacts": [
  {"name":"a_star","value_kind":"scalar","value_type":"real_float64","value":2.274},
  {"name":"E_min","value_kind":"scalar","value_type":"real_float64","value":-0.499},
  {"name":"trace","value_kind":"trace","file":"trace.csv",
   "ordering":{"kind":"indexed_by_parameter"}}
 ],
 "diagnostics": {"success":true,"method":"brent","iterations":37,
                 "boundary_solution":false,"warnings":[]}}
```

- **テンプレート経路と LLM 経路の両方が同じ契約に従う。** テンプレは決定的に emit、
  LLM はシステムプロンプトの出力契約に追加（遵守率は S-ac の測定指標に「manifest
  遵守率」を足して実測する — 台帳 evaluations.md (reference omitted from the public snapshot)）。
- **後方互換:** 旧 `NB_RESULT` 行も引き続き受理する（runner は両方読む）。旧コード
  artifact の再実行を壊さない。
- runner は manifest の各エントリを typed `NumericArtifact` として取り込む。宣言と
  spec の `outputs` の突き合わせ（欠落・余剰・形不一致）は diagnostics に記録する
  （LLM 経路では警告、テンプレ経路では不一致＝バグなのでテストで担保）。
- **diagnostics はノード必須**（指示書 §6 採用）: 最低限 `success / method / message /
  warnings`、反復法では `iterations / converged / final_residual`、最適化では
  `boundary_solution`。誠実さを強制できるのはテンプレ経路のみ — LLM 経路は
  S-ac 流の実測で品質を追う（強制はレビューゲートの仕事）。

## 5. 連鎖の原始演算 — artifact を次段タスクに束縛する

`NumericTask` に **`input_artifact_refs: list[str]`** を additive に追加し、
`InputVar.spec` に値の代わりに参照を書けるようにする:

```text
inputs: [{name:"a", role:"fixed", spec:{value_from:"na_1a2b3c4d"}}]   ← a_star の artifact
```

- fork/validate 時に**型検査**: 参照先 artifact の `value_kind` が消費側 role の
  要求と適合するか（fixed ← scalar のみ、等 — 互換規則は §3 のレジストリ）。
  `ordered_list` を scalar に、matrix を 1 変数 minimize に、complex を real 前提の
  操作に — は**実行前に** i18n 済みエラーで落とす（指示書 §3.3/§15.7 採用）。
- 実行時に artifact の値を数値リテラルとして解決して束縛する。**敵対的入力規則は
  維持** — 解決後の値も `float()` 検証を通す（テンプレ経路の既存規則を踏襲）。
- 来歴: `input_artifact_refs` がそのまま依存エッジになる（§6）。`numeric_result`
  step の `operation_args` に解決済みの値と参照元を両方記録する。
- **理論層への式差し戻し**（docs/22 §13 フェーズ2 残項目）は同じ機構の理論層版:
  artifact 値を理論層の Expression として登録（数値来歴付き）。実装は G2（§11）。

## 6. グラフ＝導出ビュー＋検証関数（独立モデルなし）

エッジはすべて既存/新設の参照フィールドから一意に導出できる:

```text
expression → task      : task.source_expression_ids
artifact   → task      : task.input_artifact_refs（新設）
task       → artifact  : artifact.task_id
artifact   → expression: 差し戻しで作られる式の created_by_step 来歴（G2）
```

`validate_numeric_graph(engine)`（純関数群）が検証する: 参照の存在／型適合（§5）／
**非巡回性**（タスクレベルの DAG — ループは §8 の構造化 1 ノードに閉じるので巡回
エッジは常にエラー）／stale（§7）。可視化・分岐比較のグラフ**ビュー**も同じ導出で
作る（UI は G3）。独立の Graph モデルの保存は、実需要（例: 分岐比較ビューの永続
レイアウト）が出た時点で再検討する。

## 7. Stale 検出 — スナップショット＋コンテンツハッシュ

フォーク時に **`source_hash`**（`source_canonical`＋関連 assumptions の正規化列の
sha256）を `NumericTask` に additive 保存する。stale 判定は「参照元 expr_id の
**現在の** canonical＋assumptions のハッシュ ≠ `source_hash`」— Expression への
版機構は導入しない（§2-2）。加えて連鎖では**推移的 stale**: 上流タスクが stale なら
その成果物を消費する下流も stale。初期実装はフラグの保存と `validate_numeric_graph`
での報告のみ（UI の「再計算しますか？」は後回し — 指示書 §11 の初期範囲どおり）。

## 8. 構造化ループと領域融合（複数式にまたがる収束計算）

**動機:** 物性の自己無撞着計算（BCS ギャップ方程式 Δ=f(Δ)、平均場反復、Hartree 系）
は複数の理論式にまたがる大きなループになる。ノードごとに sandbox プロセスを起動する
「インタープリタ型」実行は sandbox 起動 ~0.7 s/回 × 反復数で成立しない（プロセス内
numpy の反復あたりコストと 4〜5 桁差 — 最適化ではなく成立条件）。

**方針: グラフの記述単位と実行の粒度を切り離す。**

1. **ループはグラフ上「構造化された 1 ノード」** — タスクレベルの DAG に巡回エッジを
   持ち込まない（XLA/JAX の `while_loop` と同型の制限）。intent kind レジストリに
   `self_consistent_loop` を 1 エントリ追加（スキーマ改変なし — docs/24 §6 の拡張原則）。
   - 本体 = `source_expression_ids`（既に list）＋評価順序
   - 更新写像 = どの出力が次反復のどの `free` 入力に戻るか
     （`method.params.update: {"Delta": "Delta_new"}` — 当面 params 内のデータ）
   - 停止条件 = `method.params` の `tol / max_iter`（docs/24 実例③で表現済み）
2. **実行時はループ本体全体を 1 本のスクリプトに融合**し、1 回の sandbox 実行にする。
   反復はプロセス内の for ループ。
   - 本体の全式がホワイトリスト内 → 決定的ループテンプレート（`expr_to_numpy` の
     lowering を合成: 初期化→順に評価→更新→収束判定→trace 記録）。ground truth
     にもなり diffcheck の適用範囲が広がる。
   - 1 式でもホワイトリスト外 → **領域ごと** LLM 経路（レビュー必須）。分岐判定は
     既存の `codegen_route` の一箇所判定に載る。
3. **来歴は反復ごとに step を作らない**（タイムライン洪水の防止）。ループの来歴＝
   spec（本体式 ID 列・更新写像・停止条件）＋ `trace` artifact（over:[iteration]）＋
   diagnostics（iterations / converged / final_residual）の 3 点で完結し、還流は
   収束結果の `numeric_result` step 1 件。

**退けた代替案（記録）:** 常駐 sandbox ワーカー（Maxima 永続アダプタ方式の IPC
流し込み）。反復ごとの直列化コストが残り、長寿命プロセスが状態を抱えることで
「実行 1 回＝記録 1 件・密閉」という sandbox の性質が弱まる。将来「グラフを対話的に
触りながら小ノードを頻繁に再実行する」UI が欲しくなった場合の再検討事項としてのみ残す。

**スコープ境界:** 反復のたびに CAS の記号操作が要るループ（毎周で式を導出し直す類）は
数値層の責務外 — それは理論層プランの反復であり、N-1 の PoC スコープ（docs/22 §1）
と seed-to-coding-agent の境界がそのまま適用される。

## 9. 制御フローの方針（ノードでの分岐）

「分岐」を 4 種に切り分け、**タスクレベルの DAG に制御フローを持ち込まない**原則
（§8 と同じ）で配置する:

| 種類 | 例 | 置き場所 |
|---|---|---|
| ① fan-out / fan-in | Δ* を凝縮エネルギーと状態密度の両計算が消費 | DAG の標準機能（`input_artifact_refs` の複数参照）。論点なし |
| ② データレベル条件分岐 | 区分関数・レジーム判定（相図の点ごとの場合分け）・収束判定 | **ノード内部**（融合スクリプトの `if`/`np.where`）。点/反復ごとの高頻度分岐をエッジにすると §8 の速度問題を再演する |
| ③ 手法フォールバック | brent 不収束→黄金分割、初期値変更再試行 | **method ポリシー**（`solver_candidates` ＋ diagnostics の `fallback_used`/`initial_guess`。docs/24 §8-5 の MethodSpec 候補リスト一般化と同線） |
| ④ グラフレベル条件実行 | 「Δ*>0 ならノード B を走らせる」 | **P0〜P1 では入れない**。研究者が結果を見て次を決めるのが本来の運用（監督原則）。需要が出たら自由な制御エッジではなく**ノード上のガード** `guard:{artifact, predicate}` として構造化（偽なら `skipped_by_guard` を来歴に記録 — skip 意味論はプラン層の停止安全契約を流用） |

**用語の区別（重要）:** 「分岐＝branch」は理論層の概念（仮定違いのフォーク。
`source_branch_id` でスナップショット済み、分岐間比較はフェーズ2/B-4 と同時設計）。
グラフ側の条件実行は **guard** と呼び、混同しない。

## 10. 操作種別と経路（指示書 §7 の再配置)

| 操作 | 段階 | codegen 経路 | 備考 |
|---|---|---|---|
| evaluate（固定値で値/系列） | G1 | 決定的（WL内。intent kind `function_evaluation` を deterministic へ昇格） | 連鎖の主働き。manifest で typed scalar を返す最初の実装 |
| substitute（artifact→次段束縛） | G1 | codegen 不要（fork/validate/実行時解決） | §5 |
| parameter_sweep / plotting | 済 | 決定的（実装済み） | manifest 対応の改修のみ |
| self_consistent_loop | G2 | WL内本体=決定的テンプレ／WL外=LLM | §8。ギャップ方程式が実証シナリオ |
| minimize_1d | G2 | 決定的（scipy を requirements へ — docs/22 §7 の予約どおり）＋LLM | optimization_result（argmin/minimum/trace/diagnostics） |
| matrix evaluate | G2 | 要検討（行列 canonical の lowering 拡張が最重量） | S-y の痛点の根本対応 |
| eigen_decomposition | G3 | 決定的（numpy.linalg.eig）＋LLM | eigenvalue_list＋eigenvector_matrix、ordering 必須 |
| 差し戻し（artifact→理論式） | G2 | — | フェーズ2 残項目の実装 |
| minimize_nd / find_root_nd / 数値積分 | G3 以降 | 主に LLM | レジストリ済みの kind はそのまま |
| ODE/PDE・大規模配列・並列掃引 | non-goal | — | seed-to-coding-agent 境界（docs/22 §1） |

### 10-b. バックエンド適用範囲（Maxima / Julia セッション・2026-07-11 明文化）

数値層の**実行系は理論バックエンドと独立**（常に Python/numpy ＋ sandbox、§2-3）で、
G1/G2 の機構（manifest・typed artifact・連鎖束縛・stale・差し戻し）はすべて backend
非依存に動く。**差し戻しは julia_full セッションで実機検証済み**（H-1 の P2 意味論差は
数値等式では無害 — 記号は登録され未定義警告も出ない）。非対称なのは codegen だけ:

| | Maxima セッション | Julia セッション |
|---|---|---|
| 決定的テンプレート | ◎（`expr_to_numpy` は Maxima 系 canonical 前提） | opaque `_nb_E*` は明示エラー。プレーンなスカラー canonical は偶然通ることが多いが保証外 |
| LLM 経路 | ◎ | **◎ — 本命経路**。`source_latex` からコード生成（docs/22 §14、S-ac 段階③で実地検証） |
| diffcheck ground truth | WL内であり | なし（レビューゲートが唯一の防御） |

この非対称は仕様（テンプレート＝WL内の高速経路＋ground truth、LLM＝一般経路）であって
欠陥ではない。Julia セッション向けに決定的テンプレートを持つには
`JuliaSemanticResolver` 系 lowering の追加が必要 — 需要が出るまで着手しない。

### 10-c. 分岐比較ビュー（G3-3・B-4 との同時設計、2026-07-11）

**単位:** 「同じ計算を違う分岐でやった」の判定は **(intent kind, target_quantity) の一致
× `source_branch_id` の相違**。それ以上の同一性（式の系譜追跡）は判定せず、束ねた上で
研究者が見て判断する — 自動マッチングの誤爆（別物を同じ扱い）が最悪の故障モードなので、
グルーピングは緩く・表示は来歴つきで。

**表示:** グループ（kind·target）を行、分岐を列に取り、各セルに task_id・status・
最新実行のサマリ（`numeric_result` step の visible_action — i18n 済みの一行）・図の
サムネイル（既存 `/numeric/figure/{id}` 配信）。分岐が2つ以上あるときだけ 🕸 グラフ
モーダルの下部に出す（比較する相手がいなければ出さない）。

**B-4（分岐統合）との境界:** 本ビューは**読み取り専用**で、成果物とスナップショットしか
参照しない — CAS セッション状態には一切触れない。B-4 が見送られた理由（CAS state
contamination）はブランチの**統合・再生**の問題であり、比較の表示はその懸念と独立に
安全である。逆に言えば、本ビューは分岐の統合・数値タスクの別分岐への再実行までは
**しない**（それは B-4 本体の設計と同時にやる）。

## 11. 実装計画（G1〜G3）

**G1（P0 縦割り — 連鎖の最小一周）**
1. typed manifest 契約（§4）: runner の両対応（NB_RESULT/NB_MANIFEST）、テンプレの
   emit 改修、LLM システムプロンプト改修、S-ac 指標追加。
2. `NumericArtifact` の型フィールド（§3）＋ value_kind 互換レジストリ。
3. `input_artifact_refs`＋`value_from` 束縛＋実行前型検査（§5）。
4. `source_hash`＋stale フラグ（保存と検証報告のみ、§7）。
5. evaluate の決定的テンプレ（§10）。
6. テスト: evaluate scalar（manifest/型/診断）／連鎖 substitute（来歴エッジ・型検査
   負系 i18n 両言語）／stale フラグ／save/load（旧セッション互換・typed フィールド往復）。

**G2（P1 — 演算の横展開と物理実証）**
構造化ループ＋領域融合（§8、ギャップ方程式シナリオ・S-ac 拡張）／minimize_1d
（scipy 追加）／matrix evaluate／artifact→理論式の差し戻し。

**G3（P2 — グラフの見える化）**
eigen 系ノード／グラフビュー UI／分岐比較（B-4 と同時設計）／guard（需要が出たら）
／Julia 数値バックエンド（SandboxRunner 差し替え、必要になったら）。

**明示的 non-goals（指示書 §14 継承＋追加):** 多変数/制約付き最適化・ODE/PDE・
大規模配列・並列計算・完全な provenance UI（以上指示書どおり）、**ノード単位の
インタープリタ実行**（§8 により棄却）、**自由な制御エッジ**（§9）、units（§2 細目）、
外部ストレージバックエンド（`path` の現行機構で足りる間は保留）。

## 12. 既存コードへの影響範囲（G1 時点）

| ファイル | 変更 | 性質 |
|---|---|---|
| `notebook/models/numeric.py` | NumericArtifact に §3 の 5 フィールド、NumericTask に `input_artifact_refs`/`source_hash` | additive・後方互換 |
| `notebook/numeric/templates.py` | NB_MANIFEST emit、evaluate テンプレ追加 | 契約拡張（旧行併記） |
| `notebook/numeric/llm_codegen.py` | システムプロンプトの出力契約に manifest 追記 | プロンプトのみ |
| `notebook/engine_numeric_mixin.py` | fork の artifact 参照受理・run の manifest 取り込み・stale 判定 | 拡張 |
| `notebook/numeric/spec_validate.py` | ノード間型検査（`value_from` 適合） | 拡張 |
| `notebook/numeric/intent_registry.py` | `function_evaluation` を deterministic へ、value_kind 互換規則 | データ変更 |
| `notebook/numeric/sandbox.py` | **変更なし**（融合実行も 1 スクリプト実行のまま） | — |
| `ui/notebook_bridge.py` | fork リクエストの `value_from` 受理・tasks 一覧の型情報 | 拡張 |
| `i18n/messages.json` | 型検査エラー等の新キー | 追加 |

UI（連鎖の組み立て・グラフビュー）は G3 まで既存モーダルのまま — G1 の連鎖は
bridge API とテストで一周させる（スライス 1 以前の numeric 層と同じ進め方）。
