# Developer manual — public index

Source architecture and implementation references. Start with [the user guide](user_guide.md) and [the setup guide](setup_guide.md). Internal development policies, measurements and handoff logs are not distributed.

## 全体像・仕様

| 文書 | 概要 |
|---|---|
| [01_theoretical_research_notebook_spec.md](developer_manuals/01_theoretical_research_notebook_spec.md) | 詳細仕様書（システム全体の目的・データモデル・パイプライン） |
| [02_history_spec.md](developer_manuals/02_history_spec.md) | 履歴モデル仕様（分岐／フォーク／リプレイ） |

## バックエンド・実行エンジン

| 文書 | 概要 |
|---|---|
| [03_backend_paradigm_unification.md](developer_manuals/03_backend_paradigm_unification.md) | Maxima（評価時に確定）vs Julia（構造保持）のパラダイム差を統一 API で吸収 |
| [04_second_quantization_engine_design.md](developer_manuals/04_second_quantization_engine_design.md) | 第２量子化エンジン（QuantumAlgebra＋SymPy ハイブリッド、QA→SymPy 一方通行） |
| [05_algebraic_index_architecture.md](developer_manuals/05_algebraic_index_architecture.md) | 代数インデックス（`-k`, `k+q` 等の k 空間物理）アーキテクチャ |

## 記法ブリッジ・添字処理

| 文書 | 概要 |
|---|---|
| [06_index_notation_design.md](developer_manuals/06_index_notation_design.md) | 文字添字（i, j, k…）の一般化と Domain IR 拡張の抽象設計 |
| [07_maxima_index_lowering_policy.md](developer_manuals/07_maxima_index_lowering_policy.md) | Maxima / Julia 両バックエンドの添字 lowering・静的縮約・アルファタイジング方針 |
| [08_notation_bridge_catalog.md](developer_manuals/08_notation_bridge_catalog.md) | 理論記法と各 CAS 記法の差異カタログ・実装状況の監査 |
| [09_notation_bridge_p2_design_spec.md](developer_manuals/09_notation_bridge_p2_design_spec.md) | Notation Bridge P2 設計仕様（曖昧性確認 UI・notation_usage 等） |
| [10_notation_bridge_parser_resolver_concerns.md](developer_manuals/10_notation_bridge_parser_resolver_concerns.md) | 3層コンパイラ（SharedParser / SemanticResolver / CodeGenerator）の設計懸念 |
| [32_user_cas_boundary_repair_plan.md](developer_manuals/32_user_cas_boundary_repair_plan.md) | 利用者書式 ⇄ CAS 書式 境界の修繕（AUDIT-SUBST-INDEXED 根治）**完了・実施記録**：`canonical` が利用者表現と CAS ハンドルを兼ね CAS コードだけが一級でなかった診断、不変条件 R0〜R7、Phase 0〜6。**この一件で見つかった silent-wrong 6件**（添字付き代入・Maxima の行列リテラルが0・`k/r^2`が sqrt・`^`の結合方向・添字の1文字分割・往復のたびに式が伸びる）と、**実測が設計を覆した箇所**（全操作への一括 lowering／応急処置の撤去／型を訊くダイアログ／別名 lowering の既定化）を §8〜§9 に記録 |

## 定義・記号・分野

| 文書 | 概要 |
|---|---|
| [11_definition_substitution_rules.md](developer_manuals/11_definition_substitution_rules.md) | 名前付き定義（H-1）と代入の規則（lhs_alias_map・単語境界マッチ等） |
| [12_symbol_domain_design.md](developer_manuals/12_symbol_domain_design.md) | シンボル定義域（Real / Integer / 物理パラメータ）の宣言と型伝播 |
| [13_vector_field_design_spec.md](developer_manuals/13_vector_field_design_spec.md) | ベクトル場・スカラー場（div / curl / grad / Laplacian）設計 |
| [14_wick_contraction_spec.md](developer_manuals/14_wick_contraction_spec.md) | Wick 収縮（`wick_contract`）操作の設計仕様 |
| [18_linear_transform_design.md](developer_manuals/18_linear_transform_design.md) | 線型基底変換と正準性（L-1）：逆変換・正準/計量拘束・適用・代入・検証、統一原理 $M^\dagger G M=G$ |
| [19_analytical_mechanics_independent_variables.md](developer_manuals/19_analytical_mechanics_independent_variables.md) | 解析力学：座標と速度を独立アトムとするモデル（実装済み）。d/dt＝全時間微分、classical_mechanics 分離 |
| [20_definite_integral_change_of_variables.md](developer_manuals/20_definite_integral_change_of_variables.md) | 定積分の変数変換（INTEGRAL-COV）：表示専用の未評価定積分オブジェクト・4 段ワークフロー・新範囲の解決順序（利用者>テーブル>LLM>要求）・測度の符号判断 |
| [21_free_parameters_design.md](developer_manuals/21_free_parameters_design.md) | 自由パラメータと解の特殊化（FREE-PARAM）：操作が生む任意定数（+C・k₁k₂・q(0)）を名前付き記号で保持し、条件/代入で特殊化。数学流（一般解）↔物理流（特定解）を一軸に。S-e はこの一事例 |

## 公開・配布

| 文書 | 内容 |
|---|---|
| [github_publication.md](github_publication.md) | APIEMS2026向け公開候補の構成、個別設定・履歴の除外、配布用コピーの作成、未確定のライセンスと公開手順 |

## 数値計算層・プラン層

| 文書 | 概要 |
|---|---|
| [22_numeric_layer_design.md](developer_manuals/22_numeric_layer_design.md) | 数値計算層（N-1）: 理論式→NumericTask フォーク→決定的雛形→sandbox 実行→来歴付き還流。sandbox-exec 方式 B の検証結果・データモデル・API・フェーズ計画 |
| [23_plan_workflow_design.md](developer_manuals/23_plan_workflow_design.md) | Plan/Workflow 層（C-Ext-5）: 1つの高次指示→複数 CAS 操作の計画・実行。固定 workflow 優先・実行前一括検証・停止安全 Executor・plan_execution step。PoC は eigensystem workflow |
| [24_numeric_spec_ir_design.md](developer_manuals/24_numeric_spec_ir_design.md) | 数値仕様 IR（スライス2「数値計算 ChatBot」の中核）: 計算構造を直交ファセット（intent/inputs/outputs/method/presentation）に分解し、新しい計算型を「データ追加」で表す拡張設計。flat↔faceted 射影・intent kind レジストリ・validate 門番 |
| [25_numeric_chatbot_design.md](developer_manuals/25_numeric_chatbot_design.md) | 数値計算 ChatBot（対話→仕様確定・スライス2 論点B）: B-γ「提案→洗練」対話モデル・plan_adviser 機構の被せ替え・LLM 経路のみ2段承認（spec＋コード）・編集コードの取り込み・経路1本化（フォーム/対話とも NumericSpec→fork） |
| [26_typed_numeric_graph_design.md](developer_manuals/26_typed_numeric_graph_design.md) | Typed Numeric Computation Graph（G1〜G3 実装済み・2026-07-11）: 数値計算の多段連鎖。typed manifest 契約（NB_MANIFEST）・artifact→次段束縛（value_from＋実行前型検査）・ハッシュ stale 検出・evaluate/minimize_1d/構造化ループ（領域融合）/matrix evaluate/eigen（固有値分枝掃引含む）の決定的テンプレ・式差し戻し（reflux）・🕸 グラフビュー・分岐比較（§10-c、B-4 境界）。バックエンド適用範囲は §10-b |
| [36_cartesian_plot_layers.md](developer_manuals/36_cartesian_plot_layers.md) | 汎用の曲線・等高線・不等式領域・注目点の重ね描き。Adviserの構造化仕様、式参照の保存、決定的描画、格子近似の限界、図・CSV・診断JSONの保存、完全な1軸／2軸JSON・API・検証範囲。[再現例](scenarios/scenario_cartesian_plots.md) |

## システム・運用

| 文書 | 概要 |
|---|---|
| [15_i18n_design.md](developer_manuals/15_i18n_design.md) | 多言語化設計（messages.json / ui_messages.json・ハードコード禁止規約） |
| [16_llm_model_setup.md](developer_manuals/16_llm_model_setup.md) | LLM モデル設定ガイド（`models/` のロール別エンドポイント・OCR エンジン） |

## 監査

| 文書 | 概要 |
|---|---|
| [17_code_audit_2026-06-13.md](developer_manuals/17_code_audit_2026-06-13.md) | コード監査報告（god-object の mixin 分割・AUD-1〜4 等） |
| [35_one_question_many_answers.md](developer_manuals/35_one_question_many_answers.md) | **同じ問いに複数の場所が別々に答えている**——閉じない型の記述と事例表。見分け方（逃げ道フラグ・片方が既に正しい・起票が実測と違う・検査の理由が偽・片側だけ移設済み）、直し方、現在ある機構の網の目 |
| [27_interpreter_registry_and_router_split_design.md](developer_manuals/27_interpreter_registry_and_router_split_design.md) | 第3次監査（2026-07-11・保守性主眼）の設計判断・根本診断「分割より追加の形」＋Stage R0〜R6が適用したパターン: インターセプタ・レジストリ化／bridge ルータ分割／engine 背骨のフェーズ抽出／リゾルバのテーブル化／フロント分割／衛生ラチェットテスト |

---

## 関連文書（本マニュアルの対象外・相互参照）

**利用者向け**
- [user_guide.md](user_guide.md) — 利用者向けガイド（操作の使い方・自然言語コマンド例）
- [operation_reference.md](operation_reference.md) — 共通 operation のリファレンス一覧
- [refusals.md](refusals.md) — **計算を断っている処理の一覧**（システムの限界の可視化）。
  記法ごとに、理由・backend・回避策・起票 ID、そして拒否が
  🟡 暫定（CAS の primitive 不足＝自前実装で解決しうる）／🟠 判断待ち／🔵 確定の別。
  判断手順は CLAUDE.md「A Missing CAS Primitive Is Not a Reason to Refuse a Notation」
- [setup_guide.md](setup_guide.md) — セットアップガイドの表紙（言語選択、
  [setup_guide_jp.md](setup_guide_jp.md)/[setup_guide_en.md](setup_guide_en.md) へ）。
  プラットフォーム別（macOS/Linux/Windows）のインストール手順

**動作検証（シナリオ）**
- [scenarios.md](scenarios.md) — **シナリオ集の表紙・索引**。実践ループの 🟢 実現済みカタログ
  （S-a/b/d/g/h/i/k/o）＋検証系・その他のシナリオ＋ドメイン/バックエンドの決め方（作成ガイド）。
  個別の `scenario_*.md` はすべてここから辿る。✅/⏳ の候補は [open_items.md](../KNOWN_ISSUES.md) のメニュー参照。

**課題管理・評価記録**
- [open_items.md](../KNOWN_ISSUES.md) — 未実装・課題・先送り事項の一元管理（優先度一覧＋詳細）
- evaluations.md (reference omitted from the public snapshot) — LLM 評価台帳（S-aa/S-ab/S-ac など再実行型シナリオの時系列測定記録）

**その他**
- `archive/` — 役目を終えた旧文書（実装ガイド・旧監査・命令書等）
- `archive/experiments/` — 挙動調査メモ・初期設計メモ（役目を終えたもの。docs/22〜26 に昇格済み）

---

## 文書を追加するときの運用

技術文書を増やす場合の番号付けポリシー：

1. **末尾に次の連番**を付ける（例：現在 19 までなら `20_new_topic.md`）。**既存番号は振り直さない・途中挿入しない**。
   番号は「安定 ID＋目次番号との一致＋ソート」用で、分野そのものは表さない。
2. 上の表の**該当分野のセクション**にリンク1行を追記する（テーマ上の位置づけは目次が真実）。
3. 他文書・コードから参照するなら、参照リンクも書く。
4. `pytest tests/test_developer_manual_index.py` を実行。索引の番人テストが
   「全リンクが実在」「`docs/` の全技術文書が索引に載っている」を検証する
   （`scenario_*` と利用者向け文書は対象外）。索引への記載漏れはテストで落ちる。

> シナリオ文書は番号を付けず `scenario_*` 接頭で命名し、表紙 [scenarios.md](scenarios.md) の
> カタログに 1 行追記する（🟢 実現済みなら 🟢 表へ、open_items 側の候補からは外す）。
> 利用者向け（`user_guide` / `operation_reference`）・表紙（`scenarios`）・課題管理（`open_items`）は番号なし。
