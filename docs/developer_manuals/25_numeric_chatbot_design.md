# 数値計算 ChatBot（対話→仕様確定）設計 — スライス2 論点B

**2026-11-02 現行実装追記:** Adviserには選択式以前の履歴に加え、現在の式カタログ
（後から計算された座標・レベル値を含む）と共通作図の仕様を渡す。JSONの候補は1件のみを受理する。
複数式の自由シンボル照合は、fork時に参照式全体から行う。提案APIは失敗時に `stage`
（`adviser` / `parse`）、`detail`、最大2000文字の `response_excerpt` を返す。
接続失敗と仕様パース失敗を混同しない。提案の応答予算は4096トークン。
共通作図のUI表示・承認・ダウンロードは [36_cartesian_plot_layers.md](36_cartesian_plot_layers.md) 参照。

**初版:** 2026-07-10（草案・議論用）
**status:** 実装済み（2026-07-10）。§8 の実装順序どおり `spec_adviser.py`（LLM 非関与の提案骨格）
→ bridge 配線（`POST /numeric/propose_spec`・spec 対応 `POST /numeric/fork`・編集対応
`POST /numeric/approve`）→ 対話パネル UI（決定的経路・LLM 経路ともブラウザ実地検証済み）
まで完了。バックエンド134件・ブラウザE2E（両経路＋洗練ループ＋エラー経路）で確認済み。
**関連:** [24_numeric_spec_ir_design.md](24_numeric_spec_ir_design.md)（論点A・仕様 IR）、
[23_plan_workflow_design.md](23_plan_workflow_design.md) §10（`plan_adviser` の提案→パース→門番→承認ループ）、
[22_numeric_layer_design.md](22_numeric_layer_design.md)（決定的経路・LLM コード生成経路）、
experiments/theory_to_numeric_design_principles.md (reference omitted from the public snapshot) §2（対話→IR）・§3（役割分離）

---

## 0. 位置づけ — 何を作るか

論点A で「対話の吐き先＝ faceted `NumericSpec`」を実装した。論点B は**その spec を対話で
確定させる過程**を作る。設計原理 §2 の「LLM は研究者との対話を通じて、何を数値化するか・
どの量を出力するか・何を固定/掃引するか・境界条件/近似をどうするかを詰める支援を行う。
ゼロから発明せず、ノートの定義・仮定・履歴を参照する」を、実装の形にする。

利用者は 1 次元掃引を望むとは限らない（収束するまで計算・2 次元グラフ・行列の数値対角化
…）。スライス1 のフォームはこれを表現できない。ゆえに ChatBot が要る。

## 1. 決定事項（2026-07-10 確定・再議論しない）

1. **対話モデル = B-γ「提案→洗練」。** LLM がノート文脈に根ざした `NumericSpec` 案を提示
   → 研究者が承認 or 自然文で修正指示（「k を掃引して」「収束するまでに変えて」）→ 再提案。
   多ターンで詰められる（対話）が、初回が当たれば 1 ターンで済む（劣化しない）。面接式
   （B-β）でも一発提案（B-α）でもない中間。
2. **承認ゲートは LLM 経路のみ 2 段。** spec 承認（「正しいものを計算しているか」）＋ コード
   承認（「コードは正しいか」）。決定的経路（parameter_sweep）はコード承認不要（スライス1
   で確立・差分検証の正解データ側）。
3. **編集したコードは実際に走らせる。** スライス1 のコードパネルは編集可能に見えて run が
   テキストエリアを無視する既存ギャップがある。LLM 経路は「編集可能・レビュー可能な雛形」
   が核心なので、**承認時に編集済みコードを取り込み、承認済み artifact を実行する**。

## 2. 全体フロー（提案）

```text
式を選択（スライス1 と同じ「🔢 数値計算へ」）
  ↓ ── 簡単ケース ──                    ── 一般ケース（ChatBot）──
  ↓ スライス1 のフォーム                  対話パネルを開く
  ↓ （sweep 変数＋固定値）                 ↓ ①researcher: 目標を自然文で（「収束するまで解いて」）
  ↓                                       ↓ ②LLM: ノート文脈＋会話履歴から NumericSpec 案を JSON で提案
  ↓                                       ↓    ├ 厳格パース（壊れは棄却・id はこちらが振る）
  ↓                                       ↓    └ validate_numeric_spec 門番（自由シンボル照合つき）
  ↓                                       ↓ ③researcher: 承認 or 自然文で修正 → ②へ戻る（B-γ ループ）
  ↓                                       ↓ 〈spec 承認〉
  └───────────────┬───────────────────────┘
                  ↓ fork_numeric_task(expr_id, spec=…)     ← 論点A で実装済み。両経路が1本化
                  ↓ codegen_route_for_spec(spec) で分岐（表示仕様＋intentレジストリ）
       ┌── deterministic ──┐        ┌────────── llm ──────────┐
       │ generate_numeric_ │        │ generate_numeric_code_llm │
       │ code（無レビュー）  │        │  ↓ コード表示（編集可）      │
       │                   │        │  ↓〈コード承認〉= approve   │
       │                   │        │    （編集済みを取り込む）     │
       └─────────┬─────────┘        │  ↓ 差分検証（正解があれば）   │
                 │                  └────────────┬──────────────┘
                 └────────────┬──────────────────┘
                              ↓ run_numeric_task → 図・表・要約をタイムライン還流（スライス1 と同じ）
```

**経路の1本化（重要）**: スライス1 のフォーム＝簡単ケース用の「決定的 spec ビルダー」、
ChatBot＝一般ケース用の「LLM spec ビルダー」。**両者とも `NumericSpec` を吐いて同じ
`fork_numeric_task(spec=…)` に流れる**。fork 後は完全なspecを受ける `codegen_route_for_spec`
（論点A レジストリ）が決定的/LLM を選ぶ。

## 3. 対話ターンの機構（`plan_adviser` の被せ替え）

`plan_adviser`（C-Ext-5 §10）の骨格をそのまま流用し、契約と門番だけ差し替える:

| plan_adviser | 数値 ChatBot（本設計） |
|---|---|
| `chat(system, user) -> str` callable（LLM 非依存・偽 chat でテスト可） | 同じ（`_resolve_coder_chat` で解決） |
| system prompt = 操作カタログの JSON 契約 | system prompt = **`NumericSpec` JSON 契約**（intent/inputs/outputs/method/presentation） |
| 厳格パース（壊れ棄却・step_id はこちら） | **厳格パース**（壊れ棄却・`NumericSpec` に検証） |
| `validate_plan` 門番 | **`validate_numeric_spec` 門番**（論点A・実装済み。式の自由シンボルを渡す） |
| 候補提示モーダル（R-8 流用）で研究者選択 | spec を提示（差分プレビュー）→ 承認 or 修正指示 |

**新規モジュール（実装フェーズ）**: `notebook/numeric/spec_adviser.py`
（`plan_adviser.py` と対をなす）。
- `build_spec_prompt(narrative, target, free_symbols, kinds, history)`:
  ノート文脈（`_build_session_narrative`）＋対象式（canonical/latex/自由シンボル）＋
  既知 intent kind カタログ（論点A レジストリ）＋**これまでの会話履歴**を組む。
- `parse_spec(raw) -> NumericSpec | None`: JSONDecoderでJSONオブジェクトを読み、
  単一のspec候補を型検証する。fenceや周辺テキストを含む返答も扱うが、複数候補・壊れはNone。
- `propose_spec(chat, …) -> (spec | None, issues)`: 提案 → パース → `validate_numeric_spec`。
  issues が残る spec も**破棄せず返す**（研究者に「ここが未確定」と示して修正させる ＝
  B-γ の洗練ループの燃料）。

**会話状態**: B-γ は多ターンなので、対話は `(会話履歴, 直近の spec 案)` を持つ。会話履歴は
**タスクに紐づけて保存**する（設計原理 §2「会話ログだけで保持しない」の逆命題ではない —
spec が正本、会話は補助来歴）。最小実装では NumericTask に `chat_history` を additive で
持たせるか、対話中はメモリ、確定時に spec だけ確定、を選ぶ（§7 論点2）。

## 4. spec 承認 → fork → codegen 分岐

- **spec 承認**: 研究者が提示 spec を承認（または最後の修正を反映して承認）した時点で
  `fork_numeric_task(expr_id, spec=…)` を呼ぶ（論点A・実装済み。validate は fork 内で再走
  ＝ defense in depth）。
- **codegen 分岐**: fork後、`codegen_route_for_spec(spec)` で:
  - `presentation.kind="cartesian"` → 決定的な共通作図。旧flatで表せない複数式・2軸も対象。
  - その他はintentレジストリで経路を選ぶ。対応する掃引・評価・最小化・構造化ループ・
    行列／固有値問題は `generate_numeric_code`（コードレビュー不要）→ run。
    いずれも対話で作った仕様自体の承認は必要。
  - `llm`（それ以外・未知種）→ `generate_numeric_code_llm`（未承認 artifact）→ **コード承認**
    → run。差分検証は「決定的テンプレが書ける種」でのみ効く（§5）。

## 5. コード承認ゲート（LLM 経路・第2段）と編集の取り込み

- `generate_numeric_code_llm` は**未レビュー**で artifact を保存し、`run_numeric_task` は
  未承認コードの実行を拒否する（論点フェーズ2・実装済み）。
- **編集の取り込み（新規・決定事項3）**: コードパネルは編集可能。研究者が編集して承認する
  流れを、`approve_numeric_code` が**編集後ソースを受け取って新しい承認済み artifact として
  保存**する形に拡張する（現状 `approve_numeric_code(artifact_id)` は metadata に
  `reviewed=True` を立てるだけ。編集ソースを渡せるようにする）。決定的経路は編集不要なので
  テキストエリアを読み取り専用にして「編集が効かない」誤解を消す。
- **差分検証**: `differential_check_llm_code` は決定的テンプレが存在する種でのみ実行し、
  LLM コードと正解の min/max 要約を突き合わせる（実装済み）。承認の判断材料。

## 6. UI（薄いパネル・実装フェーズ）

論点Aの「経路1本化」と B-γ により、UI 増分は小さい:

| 要素 | バックエンド | 残作業 |
|---|---|---|
| 対話パネル（入力欄＋提案 spec 表示＋承認/修正ボタン） | 提案API（実装フェーズで追加） | 中：チャット列＋ spec カード描画 |
| spec カード（intent/inputs/outputs/method を人間可読に） | spec は JSON で来る | 小：faceted → 表の描画 |
| コードパネル（編集可）＋承認ボタン | generate_llm/approve（approve を編集対応に拡張） | 小：既存パネルに承認ボタン＋編集送信 |
| 差分検証バッジ | diffcheck（実装済み） | 小：一致/不一致表示 |
| coder 選択 | `/numeric/coders`（実装済み。既定順は Advisor→Supervisor） | 小：ドロップダウン |

## 7. 小論点の決定（2026-07-10）

5論点すべて提案どおり採用（決定）。実運用の需要で見直しうる（論点Aの§8と同じ扱い）。

1. **提案は 1 案（決定）。** B-γ は「1 案を洗練」を既定とする。初回だけ 2〜3 の解法候補を
   出す拡張（plan_adviser の複数提示と同型）は、洗練ループが行き詰まる実例が出てから足す。
   **見直し条件**: 同じ目標で洗練ループが 3 ターン以上かかる事例が実運用で頻出したら、
   初回のみ複数候補提示を追加（`plan_adviser` の候補モーダルを流用）。
2. **会話履歴はメモリ保持・確定 spec のみ永続化（決定）。** 設計原理「spec が正本」に忠実で
   軽い。対話中の履歴は `spec_adviser` の呼び出しスコープ内に閉じ、`NumericTask` には
   確定した `spec` だけが乗る（論点A のとおり additive）。**見直し条件**: 「対話の経緯を
   後から追いたい」という具体的な需要が出たら、`chat_history` を additive フィールドとして
   足す（スキーマは壊さない）。
3. **spec 提示は全体表示＋変更点ハイライト（決定）。** 実装時の UI 判断（IR の形に影響しない）。
4. **未知 kind は明示（決定）。** 「この種は決定的検証ができません」と研究者に示す。承認判断
   に必要な情報を隠さない、という役割分離（設計原理§3）の精神に沿う。
5. **対応する仕様は自動で決定的経路へ。** `codegen_route_for_spec` がCartesian表示を優先し、
   その他はintentレジストリで選ぶ。flat表現可能性だけで判断しない。仕様承認後に
   別の経路選択操作は不要で、LLMコードを使う場合に限ってコード承認を追加する。

## 8. 実装フェーズ（完了・2026-07-10）

1. **✓ `spec_adviser.py`**（`notebook/numeric/spec_adviser.py`）: プロンプト構築
   （`build_spec_prompt`：初回／洗練ターンの両対応）・厳格パース（`parse_spec`）・
   `propose_spec`（提案→`validate_numeric_spec`門番。issue が残る spec も破棄せず返す —
   plan_adviser との意図的な差分）。偽 chat でのテスト18件（LLM 非依存）。
2. **✓ bridge**: `POST /numeric/propose_spec`（narrative＋目標＋会話履歴→spec案＋issues。
   Adviser LLM を `_resolve_adviser_chat` で解決・issues はサーバ側で翻訳済み文字列として
   返す）。`POST /numeric/fork` に `spec` body を追加（`codegen_route` をレスポンスに含め、
   決定的/LLM経路の分岐をサーバ側の一箇所に集約）。`approve_numeric_code`/
   `POST /numeric/approve` に編集ソース対応（承認は新規 artifact を作る——スライス1の
   「編集が実行に反映されない」ギャップを解消）。
3. **✓ UI**: 「💬 対話で仕様を確定」ボタン（スライス1の「🔢 数値計算へ」と常に並列表示）。
   対話パネル：spec カード（plain DOM描画・LLM由来コンテンツをinnerHTMLに使わない）・
   issue 表示・洗練入力・承認ボタン。承認後は `codegen_route` に従い決定的（読取専用コード・
   即実行可）/LLM（編集可能コード・コード承認するまで実行不可・差分検証バッジ）に自動分岐。
4. **✓ E2E**: ブラウザ実地検証（Claude Preview）——両経路（決定的・LLM）・洗練ループ
   （`prior_spec`が直前specに正しく更新される事を確認）・Adviser未接続時のエラー経路・
   スライス1既存フローの非回帰、すべて実クリックで確認。バックエンド134件緑。
