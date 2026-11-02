# Plan/Workflow 層（C-Ext-5）設計スケッチ — 優先1〜3' PoC 実装

**初版:** 2026-07-04（優先1〜2 の最小縦割り PoC 実装と同時に作成。同日中に A/B 層と
優先3の Adviser 候補ループまで実装 — §9 の状況更新と §10 を参照）
**2026-07-16 追記:** 優先3'（自由計画 LLM プランナー）と UI（プラン結果のステップ内訳
表示）を §11 の推奨最小増分どおり実装 — §12 を参照。
**status:** Plan IR・検証器・停止安全 Executor・固定 workflow 2本・NL トリガ（日英）・
逐次ステータス API・粒度ポリシー・retry・Adviser 候補提示ループ・**自由計画 LLM プラン
ナー（優先3'）・プラン結果 UI（ステップ内訳表示）**まで実装済み。
未実装: Supervisor 自動選択・自由計画の文脈供給強化・失敗時の再計画ループ・B-4（分岐統合）。
**関連:** [open_items.md C-Ext-5](../../KNOWN_ISSUES.md)（課題管理・優先順位表）、
evaluations.md (reference omitted from the public snapshot)（S-ae: 自由計画の実地評価）、
experiments/theoretical_research_chatbot_implementation_memo.md (reference omitted from the public snapshot) §3–15（元仕様）

---

## 1. 目的と設計上の核心

「1つの高次指示 → 複数ステップの CAS 操作を計画・実行し、中間結果を保存しながら最終結果を提示する」
1入力多出力への拡張。**核心は、プラン層が新しい CAS 権限を一切持たないこと**:

- プランの各ステップは**既存の common operation の名前**しか持てない。
- 実行は既存の `OperationValidator` → `run_operation` 経路へ**そのまま委譲**する。
- したがってコア原則 #4（LLM は CAS コマンド生成の最終権限を持たない）は**構造的に**保たれる —
  仮に将来 LLM がプランを提案しても、提案できるのは決定的層が既に知っている操作の並びだけである。

もう1つの核心は**固定 workflow が LLM プランナーより優先**されること（実装メモ §5.6）。
頻出の複合依頼（固有状態を求める等）は決定的な配線で応え、LLM 計画は「登録済み workflow に
該当しない依頼」だけの担当にする — 確認過多と計画誤りの両方を抑える構え。

## 2. 実装済みの縦割り（優先1〜2）

```text
「E1 の固有状態を求めて」/ "Find the eigenstates of E1"
  ↓ ui/interpret_plan.py（狭い複合トリガ。「固有値を求めて」単発依頼は既存の1操作経路のまま）
engine.create_workflow_plan("eigensystem", E1)
  ↓ notebook/runtime/plan_workflows.py
SupervisorPlan P1 = [characteristic_polynomial($target, x),
                     eigenvalues($target), eigenvectors($target)]
  ↓ validate_plan  … 実行前の一括検証（未知操作・バックエンド適合・必須引数・参照の前方性）
engine.execute_plan("P1")
  ↓ 各ステップを run_operation へ委譲（$target/$prev/$sN を式IDに解決）
中間結果はそれぞれ通常の Expression + cas_operation step としてノートに残る
  ↓
plan_execution step  … サマリ（goal・各ステップの成否・output_refs=生成式3件）
```

## 3. データモデル（`notebook/models/plan.py`）

実装メモ §5.1 の名前を踏襲し、リポジトリの pydantic 流儀に適合:

| モデル | 要点 |
|---|---|
| `PlanStep` | `step_id`（"s1"…）・`operation`（既存操作名のみ）・`args`（式値には placeholder 可）・`description`（構築時に i18n 済み）・`depends_on` |
| `SupervisorPlan` | `plan_id`（"P1"…連番）・`goal_text`・`source`（workflow/llm/user）・`workflow_name`・`target_expr_id`・`steps`・`status`（draft→validated→completed/stopped/failed）・`step_results` |
| `PlanStepResult` | ステップごとの success/failed/**skipped**・出力式ID・値・エラー。**停止後も部分結果が残る** |

**placeholder 規約:** `$target`＝プラン起動時の対象式／`$prev`＝直前ステップの出力／
`$s<k>`＝ステップ s\<k\> の出力。前方参照は検証で拒否。

`PlanCandidate`（複数解法候補＋Adviser 選択、優先3）は**意図的に未実装** — 候補提示 UI と
セットでないと意味を持たないため。

## 4. 検証と停止安全（`plan_workflows.validate_plan` / `engine_plan_mixin`）

- **実行前一括検証**: 全ステップについて (a) 操作が schema に存在 (b) セッションの backend が
  サポート (c) 必須引数が充足 (d) placeholder/depends_on が後方参照のみ、を確認。
  1つでも落ちれば**何も実行せず** ValueError（i18n 済み）。
- **実行時の停止安全契約**: ステップ k が失敗したら (1) エラーを `step_results[k]` に記録
  (2) 残りは `skipped` (3) `status="stopped"` (4) **成功済みの部分結果（式）は保持**
  (5) 例外は外へ漏らさない — 研究者はタイムラインの failed サマリで確認して判断する。
  N-1 の「実行失敗は failed step であって例外ではない」と同じ思想。
- 実行直前に**現在のセッション状態で再検証**（作成後に対象式が削除された等に備える）。

## 5. 履歴・永続化

- `StepKind.plan_execution` を追加。サマリ step の `operation_args` に plan_id・workflow・
  goal・全 step_results を持ち、`output_refs` に生成式を列挙 — **来歴が1エントリで完結**。
  各ステップの CAS 実行は従来どおり個別の cas_operation step としても残る（二重帳簿ではなく、
  「実行の詳細」と「プランとしての要約」の2ビュー）。
- `engine.plans` / `_plan_counter` を save/load に additive キーとして統合（旧セッション互換）。

## 6. テスト（`tests/test_plan_workflow.py`、15件）

検証6（未知操作・必須欠落・前方参照・先頭 `$prev`・backend 不適合・i18n 両言語）／
実行3（`$prev` 連鎖の実 Maxima 検証・**実行時失敗の停止安全**（部分結果・skipped・failed サマリ）・
eigensystem E2E＋save/load 往復）／互換1（plan キー無し旧ファイル）／
NL トリガ4（日英発火・**単発依頼の非発火**・対象式なしの非発火）＋登録確認1。

## 7. 未実装（フェーズ計画、open_items C-Ext-5 の優先順位表に対応）

| 優先 | 内容 | 備考 | 状況 |
|---|---|---|---|
| 3 | Adviser 相談ループ（`PlanCandidate` 複数提示→ユーザー選択 or 自動選択＋理由保存） | R-8 の candidates モーダルが流用できる見込み | ✅ 実装済み（2026-07-04） |
| 3' | LLM プランナー（workflow 不一致の依頼に対する計画提案） | 提案は operation 名の列に限定（§1 の構造的ガード） | ✅ 実装済み（`_try_plan_freeform`、2026-07-16、§12） |
| — | UI（ゴール・選択した方法・ステップ進捗の表示） | サマリ step は既にタイムラインに出る | ✅ 実装済み（`ui/notebook_plan.js`、ステップ内訳のみ・ポーリングは未実装、§12） |
| — | Supervisor 自動選択（`auto_choice_score`） | 研究者選択の実績が貯まってから検討（§9 C） | ❌ 未実装 |
| 5 | 発展 workflow（演算子の行列表現・交換子を含む Plan、Julia セッション対応） | registry へ登録するだけの構造は完成済み | ❌ 未実装 |

## 8. 既知の制限（PoC）

- workflow は `eigensystem` の1本のみ。対象は行列式（validate は式の「行列らしさ」までは
  見ない — 非行列に適用すると eigenvalues が実行時に失敗し、停止安全経路に入る）。
- プランは線形実行のみ（`depends_on` は記録するが並列化・分岐はしない）。
- `create_workflow_plan` の追加引数（特性多項式の変数名等）は NL トリガからは未公開。

## 9. 残課題の考察（2026-07-04、PoC 実装直後の検討）

難易度で3層に分ける。A・B は需要が出たときに1件ずつ足せる状態、**本丸は C**（LLM が
関与する層）で、これは N-1 フェーズ3（LLM 仕様確定・結果解釈）と同じ構図——
決定的な骨格は完成し、LLM を「提案者」として載せる部分だけが残っている。

> **状況更新（2026-07-04 同日）: A・B のうち B-4 以外を実装した。**
> - **A-1 ✓** `heisenberg_components` workflow 追加（Larmor 型: dS^α/dt = i[H,S^α] を
>   x/y/z の3ステップで。Julia 専用 — backend 検証が Maxima セッションでの誤用を事前拒否）。
>   eigensystem の Julia セッション E2E も検証済み（placeholder 連鎖は Julia でも動く）。
> - **A-2 ✓** 特性多項式の変数名を NL から指定可能（「変数 s で」/ "with variable s"。
>   英語の裸の "in s" は貪欲すぎるため意図的に不採用）。
> - **B-1 ✓（API レベル）** Executor が実行中に plan レコードを**ステップごとに逐次更新**
>   （status="running"＋部分 step_results）。`GET /plan/status/{id}`・`GET /plans` で
>   ポーリング可能。UI ストリーミング自体は据え置き（ポーラが読む形は確定）。
> - **B-2 ✓** `PlanStep.present=False` で中間結果を hidden 化。**実装で判明した意味論**:
>   実行中に隠すと `_active_expr_ids` から外れて次ステップの `$prev` 検証が壊れるため、
>   隠蔽は**プラン完了後に一括適用**。停止したプランは部分結果を調査できるよう
>   意図的に全て可視のまま残す。
> - **B-3 ✓** `retry_plan(plan_id, args_override)` — 停止プランの再試行は**新しいプラン**
>   （`retried_from` で来歴接続、旧レコードは不変）。成功済みステップは再実行せず
>   結果を持ち越し、override されたステップだけ修正後の引数で実行。
> - **B-4（分岐統合）は見送り**: branch を跨ぐプラン実行は CAS state contamination
>   （監査 §F で安全と確認した現行の分岐設計を崩しかねない）と N-1 フェーズ2の
>   分岐比較ビューに重なるため、両者を同時に設計するときに着手する。

### B-4 と N-1 フェーズ2「分岐比較ビュー」の関係（2026-07-04 の検討）

両者は「**同一の計算手続きを、仮定違いの branch それぞれに適用して結果を並べる**」
という同じ問いの理論層版（B-4: workflow の結果式）と数値層版（N-1: 図・表・要約）で、
共有すべき部品が2つある。

**共有部品1 — 比較レコード＋ビュー（本体）**: どちらも最終形は
「行＝branch（仮定・近似ラベル付き）／列＝結果」のマトリクス。来歴の形は既に揃っている
（`plan_execution` step と `numeric_result` step がともに source_branch_id・仮定・
パラメータ・結果を保持）ので、比較とは「手続きが同じで branch が違う来歴レコードの
JOIN」である。比較オブジェクト（手続き ID × [(branch, 結果)]）とビューを別々に
発明すると、二重実装に加えて「branch ごとの理論固有値と数値掃引を並べる」横断比較が
不可能になる — **1つであるべき**。

**共有部品2 — branch 文脈での実行（非対称）**: 数値層は NumericTask がフォーク時に
canonical をスナップショットするため実行は sandbox で完結し **CAS を触らない** —
branch 別フォーク＋実行は現行実装で既に可能で、残るのは純粋にビューの問題。
一方プラン層は各ステップが生きた CAS プロセスを通るため、1セッション内で
branch A→B と実行を続けることは監査 §F が「起きない」と確認した contamination を
自ら持ち込むことになる。安全化には「branch ごとの snapshot 再ロード」か
「branch ごとの別 CAS プロセス」の実行プリミティブが必要（B-4 固有の重い部分）。

**帰結（推奨順序）**: ①比較レコード＋ビューを N-1 フェーズ2 として先に作る
（CAS 状態問題なし・数値層だけで検証可能）→ ② B-4 は①のビューを再利用し、
branch 文脈の安全な実行プリミティブだけを追加する。B-4 を単独で先行させると、
ビュー不在のまま最も危険な実行プリミティブの設計を強いられ、後で N-1 と
統合し直すことになる。

### A. 機械的に積み上げられる残作業（型はできている）

- **workflow の追加（優先5）**: registry に登録関数を1つ書くだけの構造は完成。ただし
  各 workflow は「どの操作をどの順で呼ぶか」のキュレーションが必要（物理の知識作業）。
  次の候補（演算子の行列表現→交換子、Julia セッション）は、**現 PoC が Maxima でしか
  E2E 検証していない**ため、初回配線時に placeholder 連鎖（`$prev`/`$sN`）と Julia の
  `_nb_E*` opaque canonical との相性を検証する必要がある。
- **workflow パラメータの NL 公開**: 特性多項式の変数名等は `create_workflow_plan` の
  引数としては存在するが、NL トリガからは渡せない（現状 `x` 固定）。

### B. 設計判断が必要な残作業（中難度）

- **UI — ステップ進捗の表示**: 完了後のサマリ step はタイムラインに出るが、実行中の
  進捗表示がない。現 Executor は同期実行なので3ステップの固有系なら問題ないが、長い
  プランでは「いまどのステップか」を返す仕組み（ポーリング or ストリーミング）が要り、
  bridge の応答モデルを変える設計判断を伴う。
- **中間結果の粒度問題**: 各ステップが通常の Expression としてノートに登録される。
  3ステップなら「中間結果が残る」のは美点だが、10ステップのプランではタイムラインが
  氾濫する。「どのステップの結果を表に出し、どれを hidden にするか」のポリシー
  （実装メモの `expected_output_kind`/`final_presentation` に相当）は未設計。
- **プランの再実行・編集**: 現在のプランは一回限りの実行記録。「ステップ2の引数を
  直して再実行」「途中から再開」は未実装——停止安全で部分結果が残る設計はその土台だが、
  再開時の式ID再配線が必要。
- **分岐（branch）との統合**: 「仮定違いの branch ごとに同じプランを実行して比較」は
  理論ノート特有の強みとして構想にあるが、プランは現在 active branch でしか動かない。
  branch 横断のプラン実行・比較ビューは N-1 の分岐比較（フェーズ2）と共通の未設計領域。

### C. 本質的に難しい残課題（この機能の核心的リスクが残っている場所）

- **Adviser 候補ループ（優先3、`PlanCandidate`）**: 「解法候補を複数提示→ユーザー選択
  or 自動選択＋理由保存」。**提示 UI は R-8 の candidates モーダルが流用できる**ので
  配線は安い。難しいのは**候補の生成源**——Adviser LLM に「この問題への解法候補を JSON
  で返させる」品質と、その JSON 契約の設計（壊れた候補の検証は `validate_plan` が担える
  のが救い）。
- **LLM プランナー（固定 workflow 不一致の依頼への計画提案）**: これが C-Ext-5 が「大」
  とされた本体。構造的ガード（提案は既知操作名の列に限定・実行前一括検証）で**壊れた
  計画**は防げるが、残る難しさ:
  - **意味的に妥当な計画か**は構文検証では判定できない（操作の順序が数学的に意味をなす
    か）。ここは §2.4 の確認 UI（計画をユーザーに見せて承認を得る）で吸収するのが現実解
    で、その承認 UI が前提条件。
  - **失敗時の再計画ループ**: 実装メモは「失敗時は停止して Supervisor へ差し戻す」まで
    求めている。現 PoC は「停止して記録」まで——差し戻し（エラーを LLM に渡して修正案を
    得る→再承認→再実行）は LLM 品質依存の未踏領域。
  - **確認頻度の設計**: 毎回計画承認を求めると操作性を壊す。固定 workflow は無確認・
    LLM 計画は要承認、という二層が自然だが、その境界の運用は実使用で調整が要る。

**載せる順序の推奨**: 6（候補提示: 既存モーダル流用で LLM 出力の検証も効く）→
7（自由計画）。いずれも `validate_plan` が最後の門番になる構造は既に効いている。

## 10. Adviser 候補提示ループ（優先3・2026-07-04 実装）

§9 C-6 を推奨順序どおり実装した。**LLM の提案品質以外はすべて決定的に実装・検証済み。**

```text
「E1 の解法の候補を出して」/ "solution candidates for E1"（明示依頼のみ発火 —
                                                いつ相談すべきかは推測しない）
  ↓ ui/interpret_plan.py::_try_plan_candidates
Adviser LLM へ JSON 契約で相談（notebook/runtime/plan_adviser.py）
  - プロンプト: goal＋backend で使える操作カタログ（名前＋必須引数）＋対象式
  - 契約: [{title, summary, recommended_when, risk, steps:[{operation, args}]}]
  - 厳格パース: 壊れた JSON・不正項目は黙って棄却（[] / skip）。step_id は
    こちらが振り直す（LLM の ID は信用しない）
  ↓ 候補ごとに validate_plan（門番）— 未知操作・引数欠落は理由付きで棄却
候補モーダル（R-8 と同じ candidates 機構）に生存候補のみ提示
  ↓ 研究者が選択
execute_plan_candidate 疑似オペ（notebook_bridge）
  - create_candidate_plan が**再検証**（defense in depth: 改竄ペイロードも
    ここで落ちる — テストで実証）
  - source="llm" プランとして停止安全 Executor で実行
  - record_ambiguity_resolution で選択の来歴を保存
```

- Adviser 未接続時は i18n 済みエラー（models/ の advisor ロール設定へ誘導）。
- 全候補が棄却された場合も明示エラー（棄却数を提示）。
- **Supervisor 自動選択（auto_choice_score）は未実装** — 研究者選択が先、という
  監督原則（§12 実装メモ）に従い、選択の実績が貯まってから検討する。
- テスト14件（`tests/test_plan_candidates.py`）: JSON 契約（正常・fenced・破損・
  不正項目 skip）・門番（混在提案から正当な候補だけ生存＋棄却理由）・カタログの
  backend フィルタ・NL トリガ日英・全棄却/Adviser 不在のエラー・選択→実行→
  来歴記録のフルループ（実 Maxima）・改竄ペイロードの再検証拒否。すべて偽 chat
  callable で決定的に実行（実 LLM 不要）。

## 11. 残り2項目の見通し（2026-07-04、優先3実装直後の検討）

> **2026-07-16 追記:** 本節の見通しどおり、両方とも小さな増分として実装した。
> 実装内容・実機確認結果は §12 を参照。本節は判断根拠の記録として残す。

残るのは「自由計画」と「UI」。**自由計画は深さ（LLM 品質と設計判断）、UI は広さ
（機械的な JS 作業）の問題**で、性質がまったく違う。

### 11.1 自由計画（workflow に該当しない依頼への LLM 提案）

**§10 の実装で、必要な部品はほぼ揃った。** JSON 契約・厳格パース・`validate_plan`
門番・承認モーダル・`execute_plan_candidate`（再検証つき実行）はすべて流用可能で、
**最小実装は「明示トリガ（「計画を立てて」/"plan this"）＋プロンプトの枠替え」だけ**
（候補提示＝「解法の代替案を1〜3個」、自由計画＝「この目標への手順」— 同じ契約）。
明示トリガで始めれば「いつ計画モードに入るか」を推測する必要もない。

実装済みの型では解決しない難所は2つ:

1. **文脈の供給** — 現在プロンプトへ渡すのは goal＋操作カタログ＋対象式 canonical
   のみ。まともな引数（変数名・式参照）の提案には宣言済みシンボル・仮定・直近の
   式履歴が要る。bridge の `_build_session_narrative`（Adviser 用セッション要約）が
   土台になるが、トークン予算と「何を見せると計画品質が上がるか」は試行錯誤。
2. **失敗→再計画ループ**（実装メモ「Supervisor へ差し戻す」）— 停止プランのエラーを
   LLM に渡して修正案→再承認→`retry_plan`。retry 機構は済みなので配線は書けるが、
   「エラーから正しい修正を提案できるか」は完全に LLM 品質依存で CI では検証不能。
   **実践ループでの実使用が唯一の評価器**（open_items シナリオ・メニュー S-aa。測定結果は evaluations.md (reference omitted from the public snapshot) の台帳に蓄積）。

意味的妥当性（操作順序が数学的に意味をなすか）は構文検証では判定できないが、
「LLM 計画は常に要承認・固定 workflow は無確認」の二層で吸収する設計（§2.4）は
そのまま効く。

### 11.2 UI

バックエンド API は全部先行実装済みなので、**新しい設計判断がほぼ要らない**:

| 要素 | バックエンド | 残る作業 |
|---|---|---|
| プラン進捗パネル | ✓ `GET /plan/status/{id}`（逐次更新済み） | ポーリング＋ステップ一覧の ✓/▶/✗/skipped 描画（notebook_main.js に100〜150行） |
| 候補モーダル | ✓ 既存モーダルで**既に動く** | 任意: カード型装飾 |
| プランサマリ表示 | ✓ `plan_execution` step は visible_action で出る | 任意: 内訳展開・停止プランの「再試行」ボタン |
| N-1 側（フォーク・コード・実行・図） | ✓ API 4本 | 図表示＝`<session_stem>_numeric/` 静的配信＋`<img>`（小）、パネル類は中規模 |

UI が後回しになってきたのは難しさでなく**広さ**（監査見積り L）と、機能が動く前に
画面を作ると手戻りするから。その条件は逆転済み — レイヤーは全部動いていて画面だけがない。

**推奨**: ①自由計画の最小実装（明示トリガ版）は小さい増分としていつでも可能。
文脈供給と再計画ループは実使用の結果を見てから。②UI は「プラン進捗＋数値層の
図表示」の2点に絞った薄い一枚から — この2つで N-1 と C-Ext-5 の両レイヤーが
日常操作で見えるようになる。フルパネル群は実践ループの需要駆動で。

---

## 12. §11 推奨増分の実装（2026-07-16）

§11 の推奨どおり、自由計画の最小実装と UI の薄い一枚を実装した。文脈供給強化・
失敗→再計画ループ・Supervisor 自動選択・B-4 は §9/§11 の見立てどおり見送っている。

### 12.1 自由計画（`_try_plan_freeform`）

`notebook/runtime/plan_adviser.py` に `_FREEFORM_SYSTEM_PROMPT` を追加し、
`propose_plan_candidates` へ任意の `system_prompt` 引数を渡せるようにした
（デフォルトは既存の `_SYSTEM_PROMPT`）。JSON 契約・`parse_candidates`・
`validate_plan` 門番は完全に共有し、重複コードはゼロ。

`ui/interpret_plan.py` に新トリガ `_try_plan_freeform`（「計画を立てて」/"plan this" 等、
NL 文法は `nl_patterns.json["plan_freeform"]`。候補提示の `plan_candidates` とは別語句）を
追加。候補提示との共通部分
（候補リスト→confirmation レスポンス構築）は `_candidates_to_response` ヘルパへ
抽出し、両トリガから共有。

### 12.2 UI（`ui/notebook_plan.js`）

§11.2 の表が指摘していた「plan_execution が UI で何も描画されない」問題
（固定 workflow は最後の式だけ表示、停止プランは中身のない「❌ Error」表示）を、
新規 `initPlan({ appendEntry })`（`notebook_branch.js` と同じ DI パターン）による
ステップ内訳カード（✓/✗/⊘ + 対象 operation + 値/エラー）で解消した。

**§11.2 のポーリング提案はあえて実装しなかった**: 現在の Executor は同期実行で、
HTTP レスポンスが返る時点で `plan_step_results` は既に最終状態であり、ポーリング
すべき中間状態が存在しない。ポーリング API（`GET /plan/status/{id}`）自体は温存し、
プランが長くなって初めて要るようになる。

### 12.3 実機確認（実 LLM、S-ae）

実際に稼働中の Adviser LLM で自由計画を検証した。「plan this for E1」に対し LLM が
2つの候補（対称行列の解析・行列式/トレースの確認）を提案し、そのうち1つを選択して
実行したところ、LLM が提案した `define_matrix` 操作の引数 `expr` がスキーマ上許可
されておらず、実行時にステップ1で失敗した。これは実装前なら「❌ Error」としか表示
されなかったケースだが、新しい UI で「✗ define_matrix（エラー内容）」＋残り4ステップ
「⊘ skipped」という内訳が正しく表示され、停止安全設計（部分結果の保持）が UI 上でも
機能することを確認できた。固定 workflow（eigensystem）でも3ステップ全てが個別に
表示されるようになった（従来は最後の式のみ）。詳細ログは
evaluations.md S-ae (reference omitted from the public snapshot) に記録。

**まとめ**: §9 C で「本質的に難しい」とされた自由計画そのものは、§11 の見立て
どおり最小実装が小さく安全な増分で済んだ。難所として残っているのは、§11.1 が
指摘した文脈供給の強化と失敗→再計画ループであり、これらは引き続き実践ループでの
実使用データを待つ。
