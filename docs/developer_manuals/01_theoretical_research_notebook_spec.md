# CAS支援付き理論研究者デジタルノート 詳細仕様書

**版:** 0.2（2026-05-04 更新）  
**目的:** システム全体のアーキテクチャ・機能仕様・データ仕様を定義する。  
**対象利用者:** 理論研究者・大学院生。特に古典物理、場の理論、物性理論、量子多体系を扱う研究者。  
**基本思想:** 利用者にCAS文法を覚えさせず、黒板・紙のノートで理論計算している感覚を保ったまま、CASとLLMの支援を受けられる計算ノートを実現する。

---

## 1. エグゼクティブサマリ

本システムは、手書き数式入力、自然言語操作、CAS実行、履歴管理、分岐管理、LLMによるスーパーバイズと物理的助言を統合した、理論研究者向けのデジタルノートである。

利用者は以下のように作業する。

- 数式は既存の手書き入力システムで入力する。
- 式は `E1`, `E2`, ... のような式番号で管理される。
- 利用者は「E3を展開」「E5にE2を代入」「この結果を考察」など、数学的操作を自然言語で指示する。
- システムは自然言語を直接CAS命令に変換せず、まず共通operationへ変換する。
- 共通operationを、セッション開始時に選んだprimary CASに対応するbuilderがCAS命令へ変換する。
- CAS文法は通常UIには表示しない。
- 履歴は線形ログではなく、途中からフォーク可能なDAGとして扱う。
- LLM supervisorは操作全体を監督し、LLM adviserは必要に応じて物理的考察と次手を提案する。

最小実装では、Maxima primaryの標準記号計算ノートから開始し、その後Julia full primaryの物性・量子多体系ノートへ拡張する。

---

## 2. 設計原則

### 2.1 CAS文法を利用者に見せない

利用者が見るのは、CASコマンドではなく数学的操作である。

表示例:

```text
E3 を展開する
E5 に E2 を代入する
E7 を有理式として簡約する
E8 を x について解く
E10 の交換子を計算する
```

通常UIに表示しないもの:

```maxima
ratsimp(E7)
subst(rhs,lhs,expr)
```

```julia
Symbolics.simplify(expr)
QuantumAlgebra.comm(a,b)
```

CAS文法はbackend specと監査ログに保存するが、利用者の通常作業画面には出さない。

### 2.2 自然言語から直接CAS命令を生成しない

自然言語入力は以下の段階を通る。

```text
自然言語指示
  -> LLM/ruleによる意図解釈
  -> 共通operation候補
  -> operation validator
  -> backend builder
  -> CAS command
  -> CAS実行
```

LLMがCASコマンド案を出してもよいが、最終実行コマンドは必ずbuilder-facing specから再生成する。

### 2.3 セッション開始時にprimary CASを固定する

保存・再開・履歴再投入を安定させるため、セッション開始時にprimary backendを決定する。

操作ごとに自動でMaximaとJuliaを切り替えることは避ける。fallbackや比較実行は利用者確認を伴う明示的イベントとして扱う。

### 2.4 人間主導、CAS検証、LLM助言

本システムは自律研究エージェントではない。基本は以下である。

```text
人間: 方針決定、結果採用、研究判断
CAS: 厳密計算、記号変形、数値計算
LLM supervisor: 操作解釈、曖昧性解消、履歴整理
LLM adviser: 物理的考察、次手提案
```

LLMの考察はadviceであり、ノート本線に自動採用しない。

---

## 3. 利用モード

### 3.1 Standard Symbolic Mode

主対象は可換代数、方程式、微積分、行列、級数展開である。

推奨対象:

- 古典力学
- 解析力学
- 古典場の理論
- 可換な有効理論計算
- 標準的な理論計算

primary backend:

```text
Maxima
```

### 3.2 Condensed Matter / Quantum Many-Body Mode

主対象は非可換演算子、第2量子化、生成消滅演算子、交換子、反交換子、正規順序、数値計算連携である。

推奨対象:

- 物性理論
- 量子多体系
- 第2量子化
- 格子模型
- Green関数
- Bogoliubov変換
- 有効Hamiltonian

primary backend:

```text
Julia full
```

想定パッケージ:

```julia
Symbolics
SymbolicUtils
Nemo
Groebner
AbstractAlgebra
SymbolicNumericIntegration
SymbolicLimits
TaylorSeries
SpecialFunctions
NonlinearSolve
Roots
CommonSolve
ModelingToolkit
SymPy
QuantumAlgebra
Latexify
JSON3
JLD2
HTTP
```

### 3.3 Cross-Backend Check Mode

主導出とは別に、MaximaとJuliaで結果を比較するためのモードである。

例:

```text
E7 を Maxima と Julia の両方で簡約して比較する
E12 の積分を Maxima と Julia/SymPy fallback で比較する
```

比較結果は本線の導出履歴とは区別して保存する。

---

## 4. 全体アーキテクチャ

```text
[User]
  -> [UI / Notebook]
  -> [Handwriting Math Input]
  -> [Local LLM Supervisor]
  -> [Intent / Operation Layer]
  -> [Operation Validator]
  -> [Backend Builder]
  -> [CAS Adapter]
       -> Maxima persistent REPL
       -> Julia persistent process
  -> [Result Normalizer]
  -> [Notebook State / History DAG]
  -> [LLM Adviser, optional]
```

### 4.1 コンポーネント一覧

| コンポーネント | 役割 | 実装方針 |
|---|---|---|
| Handwriting Math Input | 数式入力 | 既存システムを利用 |
| Notebook UI | 式、仮定、履歴、branchを表示 | CAS文法は通常非表示 |
| Local LLM Supervisor | 操作監督、曖昧性解消、履歴整理 | 長コンテキストのローカルLLM想定 |
| Operation Layer | backend非依存operation | JSON schemaで定義 |
| Validator | operationと引数を検証 | deterministic |
| Backend Builder | CAS command生成 | spec準拠、deterministic |
| CAS Adapter | Maxima/Julia実行 | 常駐プロセス |
| Result Normalizer | 結果整形 | text, LaTeX, canonicalを生成 |
| History DAG | 履歴・branch管理 | replay可能なイベント列 |
| LLM Adviser | 物理的考察・次手提案 | API LLMまたは高性能ローカルLLM |
| Numeric Layer（N-1） | 理論式からフォークした数値タスクの決定的コード生成・sandbox 実行・来歴付き還流 | 理論層と並ぶ別レイヤー。[22_numeric_layer_design.md](22_numeric_layer_design.md)（フェーズ1実装済み・UI 未実装） |

---

## 5. 数式入力仕様

### 5.1 前提

数式入力には既存の手書き入力システムを用いる。手書き入力システムは、最低限以下のような出力を返す。

```json
{
  "latex": "\\frac{x^2-1}{x-1}",
  "canonical_math": "(x^2 - 1)/(x - 1)",
  "structure": {
    "type": "fraction",
    "numerator": "x^2 - 1",
    "denominator": "x - 1"
  }
}
```

### 5.2 数式と操作の分離

数式入力と操作指示は分離する。

例:

```text
数式入力: E1 = (x^2 - 1)/(x - 1)
自然言語指示: これを有理式として簡約
```

内部表現:

```json
{
  "target_ref": "E1",
  "operation": "rational_simplify"
}
```

### 5.3 入力前提条件と保証境界

本システムにおける手書き入力（OCRなど前段認識処理）と数式解釈・評価（CASへの変換トレース）の境界および前提保証は以下の通りとする。

* **手書き入力のスコープ**: 手書き数式画像から LaTeX 形式文字列への認識変換精度やOCRデバイス・インターフェース側の処理自体は、システムの検証・動作保証の**境界外（アウトオブスコープ）**とする。
* **数学的セマンティクスの完全保証**: 手書きOCRやキーボード等による入力の結果として、**LaTeX 形式の数式文字列（例: `\nabla \cdot F`）がシステムに渡されれば、手書きかどうかを問わず、数学的セマンティクス（発散、回転、勾配、ラプラシアン等）を高精度にキャプチャ・自動解決し、CASバックエンド（Maximaの微分展開やJuliaの記号オブジェクト）での厳密な計算および追跡が可能であることを完全に保証する。**

---

## 6. ノートデータモデル

### 6.1 式番号

式は `E1`, `E2`, ... として表示する。内部ではbranchをまたいでも衝突しないようにglobally uniqueな `expr_uid` を持つ。

```json
{
  "expr_uid": "expr_8f4a2",
  "display_id": "E13",
  "branch_id": "main",
  "latex": "x+1",
  "canonical": "x + 1",
  "ir_ast": {
    "type": "Add",
    "args": [
      {"type": "Symbol", "name": "x"},
      {"type": "Number", "value": 1}
    ]
  },
  "created_by_step": "S12"
}
```

### 6.2 SymbolContext (Semantic Layer)

セッション内で定義された全シンボルを管理する。バックエンドに依存せず、式の形式的推論を行うためのメタデータ（統計性や次元など）を保持する。統計性の種類として、`"bose"` (ボソン)、`"fermi"` (フェルミオン)、およびスピン演算子などの `"noncommutative"` (非可換) をサポートする。

```json
{
  "q": {
    "quantity": {
      "name": "q",
      "kind": "scalar",
      "time_dependent": true,
      "time_parameter": "t"
    },
    "statistics": {
      "kind": "bose"
    }
  }
}
```

### 6.3 仮定

仮定は式とは別のfirst-class objectとして扱う。

```json
{
  "assumptions": [
    {
      "assumption_id": "A1",
      "display": "m > 0",
      "canonical": "m > 0",
      "scope": "branch",
      "branch_id": "main"
    },
    {
      "assumption_id": "A2",
      "display": "a_k はボソン消滅演算子",
      "canonical": "boson_annihilation(a_k)",
      "scope": "session"
    }
  ]
}
```

### 6.4 Step

```json
{
  "step_id": "S12",
  "kind": "cas_operation",
  "branch_id": "main",
  "parent_step_ids": ["S11"],
  "timestamp": "2026-04-30T10:00:00+09:00",
  "user_text": "E3 を展開して",
  "visible_action": "E3 を展開する",
  "operation": "expand",
  "operation_args": {
    "expr_ref": "E3"
  },
  "backend": "maxima",
  "rendered_command": "expand(E3)",
  "show_command_to_user": false,
  "output_refs": ["E4"],
  "state_affecting": true,
  "status": "success"
}
```

---

## 7. セッション仕様

### 7.1 Session Metadata

```json
{
  "session_id": "session_2026_04_30_001",
  "title": "Hamiltonian derivation",
  "mode": "standard_symbolic",
  "primary_backend": "maxima",
  "fallback_policy": "ask_before_cross_backend",
  "show_cas_syntax_to_user": false,
  "language": ["en", "ja"],
  "llm_supervision": {
    "mode": "full",
    "model_location": "local",
    "supervise_before_execution": true,
    "supervise_after_execution": true,
    "auto_execute_suggestions": false
  },
  "created_at": "2026-04-30T00:00:00+09:00"
}
```

### 7.2 再開手順

```text
1. session metadataを読む
2. primary backendを起動する
3. startup/preambleを実行する
4. active branchのlineageを求める
5. state_affectingなstepだけを順にreplayする
6. 式番号、仮定、採用済み結果を復元する
7. 作業を再開する
```

---

## 8. 履歴DAGとフォーク

### 8.1 履歴はDAGとして扱う

各stepは親stepを持つ。フォーク時には新branchを作る。
ただし、CAS の内部状態に対する「完全な時間旅行（任意の時点でのインメモリ状態への瞬時復帰）」は狙わない。UI・履歴・式参照レベルでのブランチ構造は保持しつつも、保存や再開は snapshot 方式を用い、状態が大きく分岐する危険な分岐（Dangerous Fork）については、安全性を優先してファイルレベルのフォーク（別セッションファイルへの分離）によって扱う。

```json
{
  "branch_id": "branch_002",
  "forked_from_step_id": "S12",
  "created_at": "2026-04-30T12:30:00+09:00",
  "label": "try expansion before solving",
  "primary_backend": "inherit"
}
```

### 8.2 Branch Metadata

```json
{
  "branches": {
    "main": {
      "branch_id": "main",
      "root_step_id": null,
      "head_step_id": "S20",
      "label": "main derivation"
    },
    "branch_002": {
      "branch_id": "branch_002",
      "forked_from_step_id": "S12",
      "head_step_id": "S17b",
      "label": "low-energy limit analysis"
    }
  }
}
```

### 8.3 フォーク手順

```text
1. フォーク元step S_kを選ぶ
2. 新branch_idを作る
3. forked_from_step_id = S_k として保存
4. S_kまでの状態を snapshot または replay で再構成する
5. 新しい操作は新branchへ追加する
6. 元branchは変更しない
7. （必要なら別ファイルとしてセッションごとフォークする）
```

### 8.4 Branch Merge

初期実装では自動mergeは行わない。利用者が明示的に「branch_002のE18をmainに採用」と指定した場合だけ採用stepを作る。

```json
{
  "step_id": "S25",
  "kind": "branch_adoption",
  "branch_id": "main",
  "operation": "adopt_expression_from_branch",
  "source_branch_id": "branch_002",
  "source_expr_id": "E18",
  "output_ref": "E21",
  "visible_action": "branch_002 の E18 を main に採用"
}
```

---

## 9. 自然言語Instruction Layer

### 9.1 対応言語

日本語と英語に対応する。翻訳辞書ではなく、自然言語phraseから共通operation候補へのmappingとして実装する。

**入力:** セッション言語に関わらず、日本語・英語どちらで入力しても受け付ける。`OperationValidator._build_alias_index()` が全言語のエイリアスを単一インデックスに統合する。

**表示:** バックエンドメッセージ (`i18n/messages.json`) とフロントエンド UI 文字列 (`i18n/ui_messages.json`) の2テーブルで管理。セッション開始時に選択した言語で表示する。

**デフォルト言語:** `config/default_settings.json` の `"language"` フィールドがシステム全体のデフォルトを決定する唯一の設定源。バックエンド (`start_engine`) とフロントエンド (`GET /api/notebook/ui-lang`) がともに `get_default_language()` を参照する。現在のデフォルトは `"en"`。

詳細は `docs/developer_manuals/15_i18n_design.md` を参照。

### 9.2 日本語マッピング例

```json
{
  "展開して": {"candidate_operations": ["expand"], "ambiguity_policy": "auto"},
  "因数分解して": {"candidate_operations": ["factor"], "ambiguity_policy": "auto"},
  "簡単にして": {
    "candidate_operations": ["simplify", "rational_simplify", "expand", "factor", "trig_simplify"],
    "default_operation": "simplify",
    "ambiguity_policy": "ask_if_target_form_unspecified"
  },
  "整理して": {
    "candidate_operations": ["simplify", "rational_simplify", "collect", "expand", "factor"],
    "ambiguity_policy": "ask_if_target_form_unspecified"
  },
  "代入して": {"candidate_operations": ["substitute"], "ambiguity_policy": "ask_if_missing_fields"},
  "解いて": {"candidate_operations": ["solve_equation", "solve_system", "numeric_solve", "solve_ode"]},
  "微分して": {"candidate_operations": ["differentiate"], "ambiguity_policy": "ask_if_variable_unspecified"},
  "積分して": {"candidate_operations": ["integrate_indefinite", "integrate_definite"]},
  "交換子を計算して": {"candidate_operations": ["commutator"], "ambiguity_policy": "auto"},
  "正規順序にして": {"candidate_operations": ["normal_order"], "ambiguity_policy": "auto"}
}
```

### 9.3 英語マッピング例

```json
{
  "expand": {"candidate_operations": ["expand"], "ambiguity_policy": "auto"},
  "factor": {"candidate_operations": ["factor"], "ambiguity_policy": "auto"},
  "simplify": {
    "candidate_operations": ["simplify", "rational_simplify", "expand", "factor", "trig_simplify"],
    "ambiguity_policy": "ask_if_target_form_unspecified"
  },
  "substitute": {"candidate_operations": ["substitute"]},
  "solve": {"candidate_operations": ["solve_equation", "solve_system", "numeric_solve", "solve_ode"]},
  "differentiate": {"candidate_operations": ["differentiate"]},
  "integrate": {"candidate_operations": ["integrate_indefinite", "integrate_definite"]},
  "compute commutator": {"candidate_operations": ["commutator"]},
  "normal order": {"candidate_operations": ["normal_order"]}
}
```

---

## 10. 曖昧性解消

曖昧な指示に対してはCAS文法ではなく数学的操作候補を提示する。

例:

```text
「整理する」は複数の意味があります。

1. 一般的に簡約する
2. 有理式として約分・通分する
3. 展開して和の形にする
4. 因数分解して積の形にする
5. 指定した変数についてまとめる
```

構造化レスポンス:

```json
{
  "status": "ambiguous",
  "target_ref": "E12",
  "user_text": "E12 を整理して",
  "choices": [
    {"choice_id": "c1", "display_ja": "一般的に簡約する", "operation": "simplify"},
    {"choice_id": "c2", "display_ja": "有理式として約分・通分する", "operation": "rational_simplify"},
    {"choice_id": "c3", "display_ja": "展開して和の形にする", "operation": "expand"},
    {"choice_id": "c4", "display_ja": "因数分解して積の形にする", "operation": "factor"},
    {"choice_id": "c5", "display_ja": "指定した変数についてまとめる", "operation": "collect", "requires": ["var"]}
  ]
}
```

---

## 11. 共通Operation Layer

### 11.1 初期operation一覧

```text
algebraic_rewrite:
  substitute
  use_relation
  simplify
  rational_simplify
  expand
  factor
  collect
  trig_simplify

equation_solving:
  solve_equation
  solve_system
  numeric_solve

calculus:
  differentiate
  integrate_indefinite
  integrate_definite
  limit

series:
  taylor_expand
  series_expand

linear_algebra:
  determinant
  inverse_matrix
  eigenvalues
  eigenvectors

output:
  to_latex
  to_plain_text

quantum_algebra:
  commutator
  anticommutator
  normal_order
  vacuum_expectation
  define_boson_operator
  define_fermion_operator
  heisenberg_equation
```

### 11.2 Operation定義例

```json
{
  "operation": "rational_simplify",
  "feature": "algebraic_rewrite",
  "display": {"ja": "有理式として約分・通分する", "en": "rationally simplify"},
  "required_fields": ["expr"],
  "optional_fields": ["assign_to"],
  "state_affecting_default": false,
  "ambiguity_group": "simplification"
}
```

---

## 12. Maxima Backend Spec

### 12.1 起動

```bash
maxima --very-quiet --disable-readline
```

Startup command:

```maxima
display2d:false$
```

### 12.2 Request Template

```maxima
res:{command}$
print("<<<RESULT_BEGIN>>>")$
print(res)$
print("<<<RESULT_END>>>")$
```

結果取得は marker 間を読む。

### 12.3 Operation Mapping

```json
{
  "backend": "maxima",
  "operations": {
    "substitute": {"template": "subst({rhs},{lhs},{expr})"},
    "simplify": {"template": "ratsimp({expr})"},
    "rational_simplify": {"template": "ratsimp({expr})"},
    "expand": {"template": "expand({expr})"},
    "factor": {"template": "factor({expr})"},
    "solve_equation": {"template": "solve({equation},{var})"},
    "differentiate": {"template": "diff({expr},{var})"},
    "integrate_indefinite": {"template": "integrate({expr},{var})"},
    "limit": {"template": "limit({expr},{var},{point})"},
    "taylor_expand": {"template": "taylor({expr},{var},{point},{order})"},
    "to_latex": {"template": "tex1({expr})"}
  }
}
```

---

## 13. Julia Backend Spec

### 13.1 起動preamble

```julia
using Symbolics
using SymbolicUtils
using JSON3
using Latexify
using JLD2

using Nemo
using Groebner
import AbstractAlgebra

using SymbolicNumericIntegration
using SymbolicLimits
using TaylorSeries
using SpecialFunctions

using NonlinearSolve
using Roots
using CommonSolve
using ModelingToolkit

using SymPy
using QuantumAlgebra
```

### 13.2 名前衝突対策

Juliaでは複数packageが同名関数をexportするため、builderは可能な限り完全修飾名を使う。

```julia
Symbolics.expand(expr)
Symbolics.simplify(expr)
Nemo.factor(poly)
QuantumAlgebra.comm(a,b)
```

### 13.3 Operation Mapping

```json
{
  "backend": "julia_full",
  "operations": {
    "substitute": {"template": "Symbolics.substitute({expr}, Dict({lhs} => {rhs}))"},
    "simplify": {"template": "Symbolics.simplify({expr})"},
    "expand": {"template": "Symbolics.expand({expr})"},
    "factor": {"template": "Nemo.factor({expr})"},
    "solve_equation": {"template": "Symbolics.symbolic_solve({equation}, {var})"},
    "numeric_solve": {"template": "Roots.find_zero({function}, {initial_guess})"},
    "differentiate": {"template": "Symbolics.expand_derivatives(Symbolics.Differential({var})({expr}))"},
    "to_latex": {"template": "Latexify.latexify({expr})"},
    "commutator": {"template": "QuantumAlgebra.comm({a}, {b})"},
    "normal_order": {"template": "QuantumAlgebra.normal_form({expr})"}
  }
}
```

---

## 14. Fallback Policy

原則としてprimary backendのみを使う。失敗時に自動で別backendへ切り替えない。

UI例:

```text
この操作は現在の計算モードでは完了できませんでした。

1. 未解決として残す
2. 別CASで試す
3. 操作を変更する
```

Fallback event:

```json
{
  "step_id": "S42",
  "visible_action": "E17 の不定積分を求める",
  "operation": "integrate_indefinite",
  "primary_backend": "maxima",
  "status": "failed",
  "fallback": {
    "used": true,
    "backend": "julia_full",
    "user_confirmed": true,
    "result_adopted": false
  }
}
```

---

## 15. LLM Supervisor Layer

### 15.1 推奨モデル配置

- supervisor: ローカル長コンテキストLLM
- adviser: API LLMまたは高性能ローカルLLM

supervisorは常時または高頻度で動作するため、低コスト・低レイテンシ・プライバシーを重視してローカルが望ましい。

### 15.2 役割

Supervisorは以下を行う。

- 自然言語指示をoperation候補へ写像する
- 対象式・branch・仮定を特定する
- 不足引数を検出する
- 曖昧性解消の質問を作る
- CAS実行前チェックを行う
- CAS実行後の軽い整合性確認を行う
- 履歴stepのvisible_actionを整える
- branch/fallback/adviser呼び出しを提案する

Supervisorは以下を行わない。

- CASの代わりに複雑な式変形を確定結果として出す
- builder-facing specに存在しないoperationを実行する
- 利用者確認なしにbackend変更、branch merge、仮定変更を行う

### 15.3 Supervisor Output

```json
{
  "supervisor_status": "ready",
  "interpreted_intent": {
    "language": "ja",
    "target_refs": ["E3"],
    "operation_candidates": [
      {
        "operation": "expand",
        "confidence": 0.82,
        "args": {"expr_ref": "E3"}
      }
    ]
  },
  "user_visible_message": {"ja": "E3 を展開する操作として解釈しました。"},
  "requires_user_confirmation": false,
  "safety_or_validity_notes": [],
  "recommended_next_actions": []
}
```

---

## 16. Operation Validator

LLM supervisorの出力は必ずvalidatorを通す。

Validatorは以下を確認する。

- operationが登録済みか
- 必須引数が揃っているか
- 対象式が存在するか
- 現在のprimary backendでsupportedか
- branch/session状態と矛盾しないか
- fallbackが必要なら利用者確認があるか

Output:

```json
{
  "valid": true,
  "operation": "expand",
  "args": {"expr_ref": "E3"},
  "backend": "maxima"
}
```

---

## 17. LLM Adviser Layer

### 17.1 役割

AdviserはCAS結果に対して物理的・数理的な考察を行い、次に試すべきoperationを提案する。

例:

```text
E(k) = -2t cos(k)
```

Adviser出力例:

```json
{
  "interpretation": {
    "summary": "これは一次元最近接tight-binding模型の典型的な分散関係である。",
    "physical_meaning": ["バンド幅は4tである。", "k=0近傍では有効質量近似が可能である。"],
    "caveats": ["格子定数を1としている可能性がある。", "tの符号規約に注意する。"]
  },
  "suggested_next_steps": [
    {"display_ja": "k=0近傍で展開する", "operation": "taylor_expand", "args": {"expr_ref": "E12", "var": "k", "point": "0", "order": 2}},
    {"display_ja": "群速度を計算する", "operation": "differentiate", "args": {"expr_ref": "E12", "var": "k"}}
  ],
  "confidence": "medium",
  "requires_verification": true
}
```

### 17.2 呼び出しタイミング

Adviserは毎回呼ばない。以下の場合に呼ぶ。

- 利用者が「考察して」「次に何をすべき？」と明示したとき
- 結果が重要式として採用されたとき
- 分散関係、Hamiltonian、作用、Green関数などが得られたとき
- 計算が行き詰まったとき
- supervisorが高次考察が必要と判断したとき

### 17.3 履歴上の扱い

```json
{
  "step_id": "A19",
  "kind": "llm_advisory",
  "model_location": "api",
  "parent_step_ids": ["S18"],
  "input_refs": ["E13"],
  "output_type": "advice",
  "adopted": false
}
```

---

## 18. UI仕様

### 18.1 通常表示

表示するもの:

- 式
- 仮定
- 数学的操作
- 結果
- 履歴
- branch
- LLMの考察と次手候補

表示しないもの:

- CAS文法
- raw stdout/stderr
- builder内部テンプレート

### 18.2 詳細表示

必要に応じて表示するもの:

- 共通operation名
- 使用backend
- 処理時間
- fallback有無
- LLM supervisor/adviserの採用状況

### 18.3 開発者表示

表示するもの:

- 実際のCASコマンド
- stdout/stderr
- parser output
- builder output
- raw CAS result
- LLM prompt/raw output

---

## 19. 保存仕様

### 19.1 保存対象

- session metadata
- backend profile
- active branch
- expression table
- assumptions
- operation history DAG
- branch metadata
- LLM supervision events
- LLM advisory events
- rendered backend commands
- CAS raw results
- adopted/rejected status

### 19.2 正本

`user_text`のみを正本にしない。必ず以下を保存する。

```text
user_text
semantic operation
operation args
rendered backend command
result
state_affecting
branch_id
backend
```

---

## 20. セキュリティ・信頼性

### 20.1 コード実行制限

Juliaは汎用言語なので、builderは危険な関数を出力しない。

禁止例:

```text
eval
include
run
open
read
write
ccall
; shell execution
```

MaximaでもOSコマンドやファイル操作につながる機能は制限する。

### 20.2 LLM出力の扱い

LLM出力は提案であり、直接実行しない。必ずvalidatorとbuilderを通す。

### 20.3 検証操作

重要な結果には検証operationを提案する。

- 元の方程式への代入
- 差を簡約して0になるか確認
- 極限で既知結果と照合
- Hermiticity確認
- 交換関係確認

---

## 21. MVP開発範囲

### 21.1 Phase 1: Maxima Primary MVP ✅ 完了

- 手書き入力から式登録 ✅
- E番号管理 ✅
- 日本語/英語の基本操作解釈（59 operations、alias scan + LLM fallback） ✅
- Maxima常駐adapter（marker方式、errcatch） ✅
- 操作: substitute, simplify, rational_simplify, expand, factor, solve, diff, integrate, to_latex, define_function, 方程式両辺変形4種 など ✅
- 履歴保存・セッション再開（JSON、replay可能） ✅
- branch作成とbranch選択 ✅
- LLM supervisor（3-phase: rule alias scan → LLM fallback → 確認 UI） ✅
- 手書き入力画面（OCR pipeline, KaTeX プレビュー） ✅
- メインノート画面（タイムライン、仮定タイル、メモタイル、保存ダイアログ） ✅
- 仮定管理（add / remove / remove_by_canonical、assumption step 記録） ✅
- メタステップ記録（user_input / llm_supervision / ambiguity_resolution / manual_note） ✅
- ブランチサマリ（手動・自動生成テキスト） ✅

### 21.2 Phase 2: Julia Full Backend（✅ 完了）

- Julia 常駐 adapter（driver script 方式、マーカー I/O）
- QuantumAlgebra.jl + Symbolics.jl + Latexify.jl 統合
- `@boson_ops` / `@fermion_ops` によるオペレーター定義
- `comm` / `normal_form` / `vacExpVal` / `heisenberg_eom` など 15 操作テンプレート
- `second_quantization` セッションモード（ドメイン選択から自動推定）
- 第２量子化カテゴリ 6 分野（凝縮系・量子光学・場の量子論・核物理・量子化学・スピン系）

### 21.3 Phase 3: Adviserと研究支援（✅ 部分完了）

- LLM Adviser 実装済み（CAS 結果 → 自然言語考察、複数 LLM 選択対応）
- Adviser UI 実装済み（考察ボタン、モーダル、KaTeX レンダリング、メモ保存）
- 比較・検証モード（未着手）

---

## 22. 開発ブランチ分割案

### branch: core-schema

成果物:

- common operation schema
- session schema
- expression schema
- history DAG schema
- branch schema

### branch: maxima-adapter

成果物:

- Maxima persistent REPL adapter
- marker-based output capture
- Maxima backend spec JSON
- 基本operation実行テスト

### branch: julia-adapter

成果物:

- Julia persistent process adapter
- julia_cas_preamble.jl
- Julia backend spec JSON
- 完全修飾名方針のテスト

### branch: handwriting-integration

成果物:

- 手書き入力出力の受け取りAPI
- LaTeX/canonical/structureの変換
- 式登録UIとの接続

### branch: llm-supervisor

成果物:

- local LLM supervisor interface
- intent resolution JSON
- ambiguity handling
- validator接続

### branch: llm-adviser

成果物:

- API adviser interface
- physical interpretation prompt
- structured advice output
- adviser history event

### branch: notebook-ui

成果物:

- E番号表示
- 操作候補表示
- branch表示
- 通常/詳細/開発者モード切替

### branch: replay-fork

成果物:

- session replay
- branch lineage replay
- fork from step
- manual branch adoption

---

## 23. 最初に作るべきファイル

```text
schemas/common_operations.schema.json
schemas/session.schema.json
schemas/history_step.schema.json
schemas/branch.schema.json
schemas/expression.schema.json
schemas/llm_supervisor_output.schema.json
schemas/llm_adviser_output.schema.json
backends/maxima.full.json
backends/julia.full.json
language/ja_intents.json
language/en_intents.json
runtime/maxima_adapter.py
runtime/julia_adapter.py
runtime/operation_validator.py
runtime/builder.py
runtime/replay.py
runtime/branch_manager.py
prompts/supervisor_system.md
prompts/adviser_physics_system.md
```

---

## 24. 受け入れ基準

MVPの受け入れ基準（2026-05-17 時点）:

| # | 基準 | 状態 |
|---|---|---|
| 1 | セッション開始時に計算モードとprimary CASを選べる | ✅ |
| 2 | 手書き入力結果から式をE番号付きで登録できる | ✅ |
| 3 | 「E1を展開」「E2を簡約」などの操作を実行できる | ✅ |
| 4 | CAS文法を通常UIに表示しない | ✅ |
| 5 | 操作は共通operationを経由して実行される | ✅ |
| 6 | 履歴を保存し、再起動後にreplayできる | ✅ |
| 7 | 任意のstepからbranchを作成できる | ✅ |
| 8 | branchを選択して再開できる | ✅ |
| 9 | LLM supervisorが曖昧性解消または操作候補提示を行える | ✅ |
| 10 | LLM adviserがCAS結果に対して物理的考察と次手候補を返せる | ✅ |
| 11 | LLM提案は自動実行されず、validatorとbuilderを通る | ✅ |
| 12 | Julia バックエンドで量子代数操作（交換子・正規順序・EOM）を実行できる | ✅ |

---

## 25. 典型シナリオ

### 25.1 標準記号計算

```text
1. 利用者が標準記号計算モードを選択する。
2. primary backend = Maxima。
3. 手書きで (x^2 - 1)/(x - 1) を入力する。
4. システムが E1 として登録する。
5. 利用者が「有理式として簡約」と指示する。
6. operation = rational_simplify。
7. Maximaが計算する。
8. E2 = x + 1 がノートに追加される。
9. 履歴に S1, S2 が保存される。
```

### 25.2 物性・量子多体系

```text
1. 利用者が物性・量子多体系モードを選択する。
2. primary backend = Julia full。
3. a, a† をボソン演算子として定義する。
4. Hamiltonian H を入力する。
5. 利用者が [H, a_k] を計算と指示する。
6. operation = commutator。
7. Julia/QuantumAlgebraが計算する。
8. 結果をE番号付きで表示する。
9. adviserが「Heisenberg方程式を作る」「低エネルギー極限を調べる」などを提案する。
```

### 25.3 フォーク

```text
1. セッションをロードする。
2. main branchのS12まで戻る。
3. 「ここから別の仮定で進める」を選ぶ。
4. branch_002を作成する。
5. branch_002で新しい仮定を追加する。
6. main branchは変更しない。
```

---

## 26. まとめ

本仕様の中心は、理論研究者がCAS文法ではなく計算の本質に集中できるデジタルノートを作ることである。

中核設計は次の通りである。

```text
手書き数式入力
  -> 式番号付きノート
  -> LLM supervisorによる操作解釈
  -> 共通operation
  -> deterministic builder
  -> primary CAS
  -> 結果・履歴・branch
  -> LLM adviserによる物理的考察
```

最初はMaxima primaryの標準記号計算ノートとして実装し、その後Julia full backendと物性・量子多体系モードを追加する。履歴はDAGとして扱い、保存・再開・途中フォークを実現する。LLMは操作監督と助言に使うが、計算そのものはCASに委ねる。

---

## 実装状況（2026-05-24 時点）

### 実装済み ✅

| カテゴリ | 実装内容 |
|---|---|
| コアエンジン | Session / Branch / Step / Expression / Assumption モデル |
| | Common operation schema（64 operations） |
| | Maxima persistent adapter（marker方式、errcatch、LaTeX出力） |
| | Operation validator（registry / 必須引数 / 式参照 / backend support） |
| | Command builder（テンプレート駆動、deterministic） |
| | NotebookEngine（全コンポーネント配線） |
| | define_function（side-effect op、Maxima := 変換） |
| | 方程式両辺変形 4 種（multiply / add / subtract / divide both sides） |
| | 関数定義式への解析演算（LHS 自動抽出） |
| | combine_expressions（2式の四則演算） |
| | diff_wrt_qdot（∂/∂q̇ 解析力学偏微分） |
| 3層アーキテクチャ | Semantic Layer: SymbolContext / SymbolProperty（記号の意味論管理） |
| | Formal Layer: ir.py（バックエンド非依存 AST）|
| | Execution Layer: MaximaIRCompiler、CasOperation IRノード（Phase 4 完了） |
| 量の記法 | Quantity / MathType モデル |
| | declare_time_dependent（時間依存宣言 + Maxima depends()） |
| | set_notation（個別シンボルごとの dot ↔ Leibniz 記法切り替え・保存対応） |
| | set_latex_name（LaTeX 表示シンボル変更） |
| | ドット記法（\dot{q}, \ddot{q}）自動適用 |
| | Notation Bridge Phase 1-2（Index/Dummy sum, DiracDelta, Display Bridge, 虚数衝突回避, NonCommutativeMul ロック）|
| 履歴・branch | History replay（lineage 計算 + state_affecting step 再実行） |
| | Branch fork + expression adoption |
| | StepKind 全種（cas_operation / expression_registration / assumption_add/remove / user_input / llm_supervision / ambiguity_resolution / branch_fork / manual_note / branch_adoption / llm_advisory） |
| | Step.input_refs（DAG 辺）、Step.user_confirmation |
| | Assumption 管理（add / remove / remove_by_canonical + 永続化） |
| | Meta-step 記録（record_user_input / llm_supervision / ambiguity_resolution / manual_note） |
| | delete_note_step（manual_note 削除 + lineage 再配線） |
| | Session 保存 / 読み込み（JSON、assumptions 含む） |
| | Maxima スナップショット保存・読み込み（.mac、replay フォールバック） |
| | Julia カスタムシステムイメージ構築自動化（PackageCompiler） |
| | ブランチサマリ（手動・自動生成、manual_note として記録） |
| | LaTeX エクスポート（タイムライン → 自己完結 .tex ドキュメント） |
| LLM | LLM Supervisor 3-phase（rule alias scan → LLM fallback → 確認 UI） |
| | Domain-aware プロンプト（supervisor_hint.txt / ocr_hint.txt 自動注入） |
| | Scope-based LLM routing（local / cloud per role） |
| | Julia full adapter（persistent REPL、QuantumAlgebra統合） |
| | LaTeX → Maxima/Julia canonical converter |
| | Ollama 起動管理（OLLAMA_MAX_LOADED_MODELS=1） |
| | LLM Adviser（CAS 結果への考察・質問・メモ保存、多 LLM 選択） |
| | Vision-aware refiner pipeline（元画像 + draft LaTeX → 修正 LaTeX） |
| 研究分野 | 25 分野ライブラリ（物理 8・数学 5・工学 3・経済 4・生物 1・教育 4） |
| | セッション開始時の分野選択 UI（カテゴリ別チェックボックス） |
| | domains セッション保存・復元 |
| i18n | バックエンド i18n（messages.json + t(key, lang) ヘルパー） |
| | フロントエンド i18n（ui_messages.json + _t() + applyI18n()） |
| | セッション言語選択（新規セッションモーダル） |
| UI | 手書き入力画面（OCR pipeline, KaTeX プレビュー, LaTeX エディタ） |
| | メインノート画面（タイムライン、仮定タイル青・メモタイル黄） |
| | 手書きモーダル（iframe、keep-alive） |
| | 保存ダイアログ（上書き / 名前付き保存 / 自動保存 5 スロット） |
| | Branch 操作 UI（fork モーダル・スナップショット保存、branch セレクタ） |
| | LLM Adviser UI（考察ボタン・モーダル・多 LLM ドロップダウン） |
| テスト | 725 件 all pass（pytest） |

### 未実装 ❌

| カテゴリ | 内容 | 仕様参照 |
|---|---|---|
| UI | Branch 可視化（DAG ツリー表示）— テキストサマリで代替中 | §18 |
| その他 | Fallback / cross-backend check step 記録 | §14 |
