# 数値計算層（N-1）設計文書 — フェーズ1 縦割り実装

**2026-11-02 現行実装追記:** 曲線・等高線・不等式領域・点を組み合わせる
宣言的作図を追加した。`presentation.kind="cartesian"` は複数式・2軸でも決定的経路を使う。
仕様・成果物保存・制限は [36_cartesian_plot_layers.md](36_cartesian_plot_layers.md) を参照。
以下のフェーズ記述は当初の実装経緯を含む。分岐比較・eigenの現行仕様は docs/26 が正本。

**初版:** 2026-07-04（フェーズ1 縦割り PoC 実装と同時に作成。同日中にフェーズ2の中心
＝**バイブコーディング経路**を実装 — §15）
**status:** フェーズ1のエンジン層・sandbox・還流＋**LLM コード生成経路（レビューゲート・
差分検証つき）**まで実装済み。**UI スライス1（決定的経路の GUI 一周・2026-07-09）**・
**UI スライス2（対話→仕様確定の ChatBot、[25_numeric_chatbot_design.md](25_numeric_chatbot_design.md)、
2026-07-10）実装済み**（§13）。**フェーズ2の主要項目は Typed Numeric Computation Graph
（[26_typed_numeric_graph_design.md](26_typed_numeric_graph_design.md)）の G1/G2 として
2026-07-11 実装済み** — typed manifest・連鎖束縛・stale 検出・evaluate/minimize_1d/
構造化ループ（領域融合）/matrix evaluate・**式差し戻し**。残: UI スライス3・分岐比較
ビュー（G3）。

> **本レイヤーの発端（意図の明記・2026-07-04 追記）:** CAS から戻ってきた式は解析的に
> それ以上進めないことがほとんどで、**数値で評価したい** — その「TeX 書式の式から
> コードを作る過程」を、LLM にコードを書かせ研究者がレビューして走らせる
> **バイブコーディング**で行うのが本来の狙いである（設計原理文書の第3段落）。
> 決定的テンプレート（§6）はこの狙いの代替ではなく、ホワイトリスト内の式に対する
> **レビュー不要の高速経路**＋LLM 経路の**数値的グラウンドトゥルース**（§15 差分検証）
> として併存する。
**関連:** [open_items.md N-1](../../KNOWN_ISSUES.md)（課題管理・経緯）、
experiments/theory_to_numeric_design_principles.md (reference omitted from the public snapshot)（設計原理）、
experiments/theory_to_numeric_implementation_memo.md (reference omitted from the public snapshot)（実装指針メモ）

---

## 1. 目的と位置づけ

理論計算の終点をそのまま数値コード生成の入口とみなさず、**理論層と並ぶ「数値計算層」を
同一ノート内の相互参照可能な別レイヤー**として設ける。理論式・仮定・分岐から数値計算タスクを
フォークし、決定的にコードを生成し、sandbox で安全に実行し、結果を**来歴付きで理論ノートの
タイムラインへ還流**する。

スコープの割り切り（重要）: N-1 は「解析的に解けない問題に数値で**当たりをつける** PoC の
生成・実行・来歴管理」までで完結する。大規模計算は責務外 — 必要になったら N-1 が生成した
コードを種として Coding Agent に拡張を依頼する。この割り切りが sandbox を軽く保つ
（モデル B、§6）。

## 2. データフロー（実装済みの縦割り）

```text
理論式 Expression (例: E1 = sin(k*x)*exp(-a*x))
  ↓ engine.fork_numeric_task(expr_id, sweep, fixed, ...)
NumericTask N001                       … canonical/latex/仮定/branch をスナップショット
  │                                    （タイムラインに numeric_fork step）
  ↓ engine.generate_numeric_code(task_id)
expr_to_numpy                        … SharedParser → MaximaSemanticResolver
  │                                    → NumpyExprGenerator（3層コンパイラ流儀）
templates.generate_code              … 決定的雛形（LLM 非関与）
  │                                    → NumericArtifact(type="code")
  ↓ engine.run_numeric_task(task_id)
SandboxRunner                        … <セッション保存先>_numeric/<task>/<exec>/ でサンドボックス実行（OS別実装、§7）
  │                                    → figure.png / table.csv / NB_RESULT JSON
  ↓
NumericExecution + NumericArtifacts  … 実行記録・成果物
  ↓
numeric_result step                  … 来歴（元式ID・仮定・近似・パラメータ・要約）付きで
                                       タイムラインへ還流。save/load で永続化
```

## 3. データモデル（`notebook/models/numeric.py`）

| モデル | 役割 | 要点 |
|---|---|---|
| `NumericTask` | 数値計算仕様（フォークの中間表現） | `task_id`（`N001`, `N002`… ゼロ埋め連番）。**`source_canonical`/`source_latex` はフォーク時のスナップショット** — 後で理論式を編集しても来歴が壊れない。`source_assumption_ids`・`source_branch_id`・`problem_type`・`fixed_values`・`sweep_parameters`・`approximations`・`status`（draft→code_generated→executed/failed） |
| `NumericArtifact` | 成果物 | `artifact_type ∈ {code, result, figure, table, log}`。小さいテキスト（code/result/log）は `content` にインライン、ファイル（figure/table）は `<セッション保存先>_numeric/` 上のﾊﾟｽを `path` に保持 |
| `NumericExecution` | sandbox 実行1回の記録 | `runtime`（インタープリタパス）・`package_info`（numpy 等のバージョン）・`parameter_snapshot`・stdout/stderr・`return_status`・`timed_out`・`duration_s`・`result_artifact_ids` |

仕様は**会話ログではなく必ず構造化保存**する（実装メモ §12）。engine は
`numeric_tasks` / `numeric_artifacts` / `numeric_executions` の3辞書と `_numeric_counter` を持ち、
save/load に **additive キー**として統合済み（旧セッションは既定値でロード、後方互換）。

## 4. 来歴（provenance）と履歴統合

- `StepKind` に **`numeric_fork`** と **`numeric_result`** を追加（`notebook/models/step.py`）。
- fork step: `input_refs=[元式ID]`、`operation_args={task_id, problem_type}`。
- 還流 step: `operation_args` に `source_expression_ids / source_assumption_ids /
  source_branch_id / approximations / parameter_snapshot / summary / artifact_ids` を全部持つ —
  「どの式から・どの仮定で・どの設定で・何が出たか」がタイムライン上の1エントリで完結する。
- `visible_action` は i18n 済みサマリ（例:「数値計算 N001: y(x) — 最小 −0.499（x=2.274）／…」）。
- 分岐との統合: task は `source_branch_id` を保持するので、フォーク違いの task を
  branch 単位で比較できる（比較**ビュー**はフェーズ2）。

## 5. 理論式 → numpy 式の lowering（`notebook/numeric/expr_to_numpy.py`）

3層コンパイラの流儀をそのまま踏む: `SharedParser`（Surface AST）→
`MaximaSemanticResolver`（Semantic IR）→ **`NumpyExprGenerator`**（`ExprGenerator` の派生）。

- **ホワイトリスト方式**: `sin/cos/tan/asin…/sinh…/exp/log/sqrt/abs/floor/round/sign` のみ
  `np.*` へ写像。`%pi→np.pi`・`%e→np.e`・`%i→1j`。`Pow` は必ず `(base)**(exp)` と括弧付き。
- **fail-fast**: 未知関数は `UnknownNotationError("numeric.unsupported_function")`、
  スカラー式以外の構造（IndexedSymbol・IndexSum・Derivative・行列…）は
  `numeric.unsupported_construct`。**黙って間違った数式に変換することが最悪の故障モード**なので、
  R-3 と同じ「unknown → 明示エラー」規律に従う（engine 境界の
  `_translate_unknown_notation` がセッション言語へ翻訳）。
- 戻り値は `(numpy式文字列, 自由記号集合)`。自由記号は掃引変数か `fixed_values` で
  全て束縛されなければならない（未束縛は `numeric.missing_values`）。

## 6. 決定的コード雛形（`notebook/numeric/templates.py`）

- フェーズ1の `problem_type`: **`parameter_sweep`**（figure+CSV）と **`plotting`**（figure）。
  掃引変数はちょうど1個。
- 生成コードは**レビュー可能な足場**（設計原理: 完成品ではなく確認可能な雛形）。
  冒頭 docstring に task_id・元式・branch・仮定・近似の来歴ヘッダを埋め込む。
- 機械可読な結果契約: 生成コードは最後に **`NB_RESULT {json}`** を1行出力する
  （target/sweep_var/min/max）。runner がこれを result artifact として取り込む。
- **敵対的入力規則**: `fixed_values`・掃引範囲は生成コードへ文字列として埋め込まれるため、
  `float()` で**数値リテラル検証**してから埋め込む（`numeric.bad_value`）。
  式や任意コードは値として通らない。
- この決定的経路に LLM は関与しない。**ただしこれは「数値層に LLM を入れない」という
  意味ではない** — コア原則 #3/#4 が禁じるのは CAS **コマンド**生成の権限委譲であり、
  数値コードの生成はバイブコーディング経路（§15）で LLM の責務である（設計原理文書
  どおり）。テンプレートは「ホワイトリスト内・レビュー不要・グラウンドトゥルース」
  という役割を持つ高速経路。

## 7. sandbox（`notebook/numeric/sandbox.py`）— 方式 B・OS 別に3実装・実機検証済み

> クロスプラットフォーム実装の設計判断・調査経緯（bwrap のコンテナ内非互換の
> 切り分け・proportionality の再検討・Landlock 実装詳細）の完全版は
> [29_deployment_and_sandbox_design.md](29_deployment_and_sandbox_design.md) 参照。
> 以下は数値層アーキテクチャ内での位置づけの要約。

**macOS**（本命）は `sandbox-exec` の **default-deny プロファイル**によるローカル隔離
（open_items の「モデル B」を採用・確定）。2026-07-04 に本機で正負両系を検証:

| 保証 | 検証結果 |
|---|---|
| numpy/matplotlib コードの実行 | ✓（Agg バックエンド、図生成まで約 0.7 s） |
| ネットワーク遮断 | ✓ `(deny network*)` — DNS 解決から失敗。子プロセス（curl）にも継承 |
| 書込制限 | ✓ 実行ディレクトリ subpath のみ許可。repo 本体への書込は PermissionError |
| 秘密領域の read 遮断 | ✓ deny 規則の実効を実在ディレクトリ（sessions/）で確認。~/.ssh・~/.aws・~/.gnupg・~/.config/gh・sessions/ を deny |
| 暴走停止 | ✓ ホスト側 `subprocess.run(timeout=…)` kill ＋ 子プロセス `RLIMIT_CPU` |

**Linux は2つの実装を実プローブで自動選択する**（`_resolve_linux_mode()`、
`sys.platform` 固定判定ではなく実際に試して決める。2026-07-20/21、open_items G-2 に
調査経緯の詳細）：

1. **ベアメタル Linux（Docker を使わない場合）: `bubblewrap`（bwrap）**。
   unprivileged user+network+mount namespace で macOS と**同水準の本物のサンドボックス**
   を再現（`--ro-bind / /` → 秘密ディレクトリ・`sessions/` を `--tmpfs` で隠す → 自タスク
   work dir だけ `--bind` で再公開、macOS SBPL の allow→deny→re-allow と同じ構造を bind
   順序のシャドーイングで再現）。Ubuntu 22.04 実機で①ネットワーク遮断・②書込制限・
   ③他セッション read 遮断・④タイムアウト強制終了を全て実測確認、追加権限不要。
2. **Docker コンテナ内: 専用低権限ユーザー + Landlock（ネットワークのみ）**。
   bwrap は内部のループバック設定処理（fork した子プロセスへの `waitpid()`）がコンテナ内
   で `ECHILD` で失敗し（バージョン・SIGCHLD・namespace 構成・seccomp 等7仮説を切り分け
   たが原因不明、upstream バグの可能性）、断念してこちらにフォールバックする。
   `ctypes` で `landlock_create_ruleset`/`landlock_restrict_self` を直接呼び出し
   （`prctl(PR_SET_NO_NEW_PRIVS)` が前提）、TCP bind/connect を拒否（ABI v4・
   kernel 6.7+、追加権限不要——Landlock は kernel 設計上 unprivileged）。書込制限は
   Dockerfile で作成した専用ユーザー（`useradd --system numeric`）への `chown`/`chmod`。
   **正直な評価**：これは macOS/ベアメタル bwrap と同水準の「本物のサンドボックス」では
   ない。①ネットワークは TCP のみ（UDP・raw socket は対象外）②書込制限は堅牢（実測
   `EACCES`/`EROFS`）③他セッション・他タスクからの read 隔離は**弱い**——`sessions/`
   自体は non-listable だが、実行中タスクの祖先しかロックしないため兄弟ディレクトリは
   無施錯のまま、かつ `task_id`（`N001`…連番）・`execution_id`（タイムスタンプ文字列）が
   乱数でないため exact path の推測が容易、同じ低権限ユーザーを共有するタスク同士も
   相互隔離できない。
   **なぜこれで受容できるか**：数値層に到達するコードは①決定的テンプレート
   （§6 の関数ホワイトリストがネットワーク I/O・パス操作を一切表現できず構造的に
   不可能）②LLM 生成コード（`approve_numeric_code` の人間レビュー必須ゲート）の2経路
   のみで、OS レベルの分離はその**二次的なバックストップ**という位置づけ。フル汎用
   アドバサリアルサンドボックス（Landlock+iptables 等の重装備）は本レイヤーの実際の
   脅威モデルに対して過剰と判断し、「無いよりマシ」程度の分離を意図的に選んだ
   （2026-07-21、ユーザーとの proportionality 議論）。

ネイティブ Windows（WSL2 を使わない場合）向けの実装は未着手（AppContainer 等の別系統
プリミティブが必要、`setup_guide` が推奨しない経路のため需要が出るまで着手しない）。
**Docker Desktop/WSL2 経由の Windows は実機未検証**（open_items G-3）——上記の Linux
フォールバックが WSL2 のカーネルでも同様に動くはずという未確認の推測に留まる。

- 実行環境は**最小化した env**（`PATH=/usr/bin:/bin`、`HOME`/`TMPDIR`/`MPLCONFIGDIR` は
  実行ディレクトリ配下、`MPLBACKEND=Agg`、`NB_NUMERIC_OUT`=実行ディレクトリ）。
- インタープリタは**専用 venv `.venv-numeric`**（numpy/matplotlib/scipy。アプリ venv を
  汚さず、`package_info` を実行ごとに記録）。`NB_NUMERIC_PYTHON` 環境変数で差替可。
  セットアップ: `python3 -m venv .venv-numeric && source .venv-numeric/bin/activate &&
  pip install -r notebook/numeric/requirements.txt`（推奨パッケージは
  `notebook/numeric/requirements.txt` に集約。テンプレートが実際に import するものだけを
  列挙し、テンプレが要求した時点でそこに追加する — scipy は G2 の minimize_1d で追加）。

**単一共有 venv の方針（2026-07-11 明文化）:** `.venv-numeric` は計算の内容・対象が
変わっても**常に同じものを使い続ける**。これは意図的な選択で、次の3点に支えられている:
① 依存を計算内容と独立に保つ設計 — 決定的テンプレはホワイトリスト、LLM 経路は
「os/json/numpy/matplotlib 以外 import 禁止」契約で、計算の**内容**が変わっても
**依存スタック**は変わらない。② 再現性は環境の同一性でなく**記録**で担保 —
`NumericExecution.package_info` が実行1回ごとに実バージョンを来歴に残すので、後で
venv を更新しても「どの結果がどのバージョンで出たか」は追える。③ 実行間の隔離は
venv でなく sandbox＋実行ディレクトリの責務。利用者負担も最小（セットアップ1回）。
「プロジェクト別に環境を分けたい」場合は `NB_NUMERIC_PYTHON` で**今日でも粗く可能**
（アプリプロセス単位）。プロジェクト固有の重いスタック（qutip・ase 等）を要求し始めたら、
それは多くの場合 N-1 のスコープ境界（PoC → 生成コードを種に Coding Agent へ）を越える
サインでもある。境界内に留まる中間ケースが実際に出たら、**セッション単位のインタープリタ
指定**（Session/NumericTask に環境パスを持たせ来歴に記録）を additive に足す —
`SandboxRunner` は python パスをパラメータ化済みなので小変更で済む（需要待ち、
open_items N-1 参照）。
- 生成コードは自前の決定的テンプレート由来だが、**プロファイルはそれに依存しない**
  （敵対的入力として扱う）。将来ユーザー編集コードや LLM 生成コードが走っても前提は同じ。
- 必要になれば方式 A（コンテナ/microVM）へ**格上げできる二段構え**。sandbox 実装は
  `SandboxRunner` に閉じているので差替は局所的。

## 8. 実行ディレクトリと成果物

**成果物はセッション JSON の保存先に同居する**（2026-07-05 決定 — 独立した
`sessions/<名前>_files/` バンドルは採用せず、保存済み JSON のフォルダ名・
ファイル名（stem）をそのまま流用する）。理論式を数値タスクへフォークする
前にセッションが未保存なら `run_numeric_task` は `numeric.save_session_first`
で拒否し、bridge 側（`/numeric/run`, `/numeric/diffcheck`）は代わりに自動保存
してから実行する。autosave のローテーション保存はこのアンカー
（`EngineNumericMixin._numeric_output_root()` が参照する
`engine._last_save_path`）を動かさない — `engine.save(path, is_autosave=True)`
を呼ぶ。

```text
sessions/experiment.json       ユーザーが保存したセッション
sessions/experiment_numeric/   ← <stem>_numeric/（アンカー）
└── N001/                     task_id
    └── nx_20260706T142301123456/  execution_id
        ├── .sandbox.sb       生成プロファイル（監査用に残す）
        ├── code.py           実行したコード（artifact の複製）
        ├── figure.png        → NumericArtifact(type="figure", path=…)
        └── table.csv         → NumericArtifact(type="table", path=…)
```

`sessions/` は丸ごと gitignore 済みなので、成果物も自動的に追跡対象外になる
（旧 `numeric_runs/` という単独ディレクトリの gitignore エントリは廃止）。

## 9. API

**エンジン（`notebook/engine_numeric_mixin.py`、mixin パターンは AUD-4 の流儀）**

| メソッド | 役割 |
|---|---|
| `fork_numeric_task(expr_id, problem_type, target_quantity, sweep_parameters, fixed_values, approximations)` | フォーク＋来歴スナップショット＋`numeric_fork` step |
| `generate_numeric_code(task_id)` | 決定的雛形生成 → code artifact（status→code_generated） |
| `run_numeric_task(task_id, timeout_s)` | sandbox 実行 → execution/artifacts 記録 → `numeric_result` step 還流（status→executed/failed）。code artifact が無ければ自動生成。**実行失敗は例外ではなく failed step として記録**（研究者がタイムラインでログを確認する） |

**bridge（`ui/notebook_bridge.py`、UI 未実装でもフローを一周できる最小 API）**

- `POST /numeric/fork` — NumericForkRequest（expr_id・problem_type・sweep_parameters・fixed_values…）
- `POST /numeric/generate/{task_id}` — 生成コードと meta を返す（レビュー用）
- `POST /numeric/run/{task_id}` — execution と artifacts を返す
- `GET  /numeric/tasks` — task/execution 一覧（将来のパネル用）

## 10. i18n

`numeric.*` の13キー（ja/en）を `i18n/messages.json` に追加済み
（forked / executed_summary / exec_failed / sandbox_unavailable / missing_values /
unsupported_function / unsupported_construct / unsupported_problem_type /
need_one_sweep / bad_value / task_not_found / task_source_missing / executed）。

## 11. テスト（`tests/test_numeric_layer.py`、18件）

- lowering 単体（関数・冪・定数・fail-fast の i18n 両言語）
- 雛形（来歴ヘッダ・NB_RESULT 契約・CSV 有無・未束縛/不正値/型ゲートの負系・
  生成コードの `compile()` 検証）
- sandbox 負系（**ネット遮断・実行ディレクトリ外書込遮断・timeout kill** を毎回回帰）
- E2E（fork→run→還流→save/load。**物理検証**: sin(kx)e^{−ax} の最大位置が
  解析値 arctan(k/a)/k と一致）
- 実行失敗の記録（例外でなく failed step になること）
- 永続化（旧セッション互換・task 往復）

sandbox / `.venv-numeric` が無い環境では sandbox 依存テストは **skip**（suite は緑を保つ）。

## 12. 責務分離（設計原理 §「役割の分離」との対応）

| 主体 | 実装状態（2026-07-04 現在） |
|---|---|
| 決定的テンプレート | ホワイトリスト内の式の**高速経路**（レビュー不要）＋LLM 経路の**数値的グラウンドトゥルース**（§15 差分検証） |
| **コーディング LLM** | **一般経路のコード生成者**（§15、設計原理どおり）。TeX/canonical＋仕様＋出力契約からスクリプトを書く。CAS には触れない。生成物は**レビュー承認までは実行不能**（エンジン側ゲート） |
| sandbox | 実行環境。解釈しない。**コードの出所を信用しない**（テンプレート/LLM/人手編集のいずれでも同一の隔離）。実行条件を記録 |
| 研究者 | 仕様の指定、**LLM コードのレビュー・承認**（`approve_numeric_code` — 監督が API 形をとる）、結果の妥当性判断 |
| （フェーズ3の LLM） | 仕様確定チャット・結果要約は未実装の追加レイヤー |

## 13. 未実装（フェーズ計画）

- **フェーズ1 UI スライス1（実装済み・2026-07-09）:** 決定的テンプレート経路を GUI で一周。
  式選択時の「🔢 数値計算へ」ボタン→タスク編集モーダル（掃引変数＋範囲＋固定値の自動列挙）
  →生成コード表示→実行→図のタイムライン還流。`GET /numeric/params/{expr_id}`（自由シンボル
  自動列挙・テンプレートと同一 lowering を再利用）・`GET /timeline` の numeric_fork/numeric_result
  シリアライズ・`GET /numeric/figure/{artifact_id}`（パストラバーサル防御つき図配信）を追加。
  ブラウザで実地検証済み（`tests/test_numeric_ui_bridge.py`）。damped_oscillator シナリオが
  GUI だけで回る。**このスライスで sandbox の潜在バグも解消**（session-anchored 出力が
  `sessions/` の deny-read と衝突し自分の code.py を読めなかった問題を修正）。
- **フェーズ1 UI スライス2（実装済み・2026-07-10）:** 当初「LLM バイブコーディング経路の
  UI をスライス1のモーダルにボタンを足すだけ」と見積もったが誤りと判明——
  `NumericTask`（`problem_type`＋単一`sweep_parameters`＋`fixed_values`）は
  `parameter_sweep`/`plotting` 専用の形で「何を計算したいか」を伝える欄がなく、
  利用者が欲しいのは1変数掃引とは限らない（収束反復・2次元掃引・行列対角化 等）。
  **再スコープして論点A（拡張可能な仕様IR、[24_numeric_spec_ir_design.md](24_numeric_spec_ir_design.md)）→
  論点B（対話→仕様確定のChatBot、[25_numeric_chatbot_design.md](25_numeric_chatbot_design.md)）
  の順で設計・実装。** ファセットIR（`NumericSpec`：intent/inputs/outputs/method/
  presentation）が計算の形を「型」でなく「データ」として表現し、B-γ「提案→洗練」対話
  （`spec_adviser.py`、`plan_adviser`の提案→パース→門番→承認パターンを踏襲）が対話で
  IRを確定、決定的/LLM経路への分岐は`codegen_route_for_spec`（表示仕様＋intentレジストリ）が一箇所で
  決定。「💬 対話で仕様を確定」ボタンがスライス1の「🔢 数値計算へ」と常に並列表示され、
  LLM経路のみコード編集→承認（スライス1で「編集が実行に反映されない」ギャップだった点を
  解消）→差分検証の2段ゲート。ブラウザで両経路・洗練ループ・エラー経路を実地検証済み。
  coder既定選択の設計判断（決定済み・未実装のまま——実際にはspec提案はAdviserロール・
  コード生成はcoderロールという役割分離になったため、coder選択UI自体は当初想定と違う
  形で必要になる）: Advisor設定先を優先、Supervisor設定先（localhost・ネットワーク不要）
  を最低優先度のフォールバックに。
- **フェーズ1 UI スライス3（当初の未実装項目）:** 専用タスク一覧パネルは別項目。
  分岐比較ビューは docs/26 G3 で実装済み。複数レイヤーの作図は2026-11-02に
  対話経路＋共通rendererとして追加（docs/36）。旧フォームに専用作図エディタを追加したわけではない。
- **フェーズ2（→ docs/26 G2 として 2026-07-11 実装済み）:** 結果値の**式としての
  差し戻し**（`reflux_numeric_value` — H-1 等式機構）・minimize_1d（scipy）・
  構造化ループ（領域融合）・matrix evaluate。`root_finding` / `numeric_integration` /
  `eigenvalue_problem` の決定的テンプレは**作らない方針に収束** — intent レジストリで
  LLM 経路（レビュー必須）が受け皿（eigen 系の決定的化は docs/26 G3 候補）。
  残: 分岐比較ビュー（B-4 と同時設計）。
- **フェーズ3:** LLM spec-elicitation は docs/25 で実装済み。残: solver 候補の高度化・
  多言語コード・理論層への再提案。

## 14. 既知の制限

- **決定的テンプレート経路**: 関数ホワイトリスト内のみ。対応する計算形は掃引/
  プロット・1点評価・1次元最小化・収束ループ・行列評価・固有値問題（docs/26 §10）と
  Cartesianレイヤー作図（docs/36）。2軸の等高線・不等式領域も後者で扱える。
  その他の多変数計算・特殊関数・未対応の表示形式は §15 のLLM経路（レビュー必須）。
- `source_canonical` は Maxima 系 canonical を前提。Julia セッションの `_nb_E*`
  opaque canonical はテンプレート lowering 不可（明示エラー）だが、**LLM 経路は
  `source_latex`（表示 TeX のスナップショット）を読める** — S-ac 段階③で実地検証済み
  （evaluations.md (reference omitted from the public snapshot)）。適用範囲の非対称は docs/26 §10-b に明文化。
- 生成コードの編集は承認時に反映される（docs/25 §5 — 承認が新 artifact を作る）が、
  再生成時の編集保全・差分管理は未設計。
- **リスト canonical**（eigenvalues の出力等）はフォーク不可（誘導つきエラー。
  solve 経由で個別式にするのが検証済み経路 — open_items S-y）。行列 **literal**
  （`matrix(...)`）の1点数値評価は docs/26 G2 で対応済み。

## 15. バイブコーディング経路（フェーズ2の中心・2026-07-04 実装）

**発端（冒頭の意図）を実装したもの**: 解析的に進めない式の TeX/canonical から、
コーディング LLM がスクリプトを書き、研究者がレビューし、sandbox が走らせる。

```text
NumericTask（仕様）＋式の TeX・canonical
  ↓ engine.generate_numeric_code_llm(task_id, chat)   ※bridge: POST /numeric/generate_llm/{id}
コーディング LLM（`coder` ロール — 既定 `ollama_qwen25_coder_32b`、
  `coder_target` 設定と API 指定で切替可・advisor フォールバック。doc 16「ロール一覧」）
  - システムプロンプト＝出力契約: numpy/matplotlib(Agg)のみ・NB_NUMERIC_OUT 配下へ
    書出し・NB_RESULT {json} を最後に1行・単一の fenced Python ブロックのみ・
    可読でレビュー可能なコード
  - ホワイトリスト内の式なら決定的テンプレートを「出発点の足場」として同梱
  ↓ extract_code: fenced 抽出 → compile() 構文門番（通らない応答は artifact にならない）
code artifact（source="llm", reviewed=False）
  ↓ 研究者レビュー → engine.approve_numeric_code   ※bridge: POST /numeric/approve/{artifact_id}
  ↓ run_numeric_task — **レビューゲート**: 未承認の LLM コードは実行拒否
    （エンジン側で強制。UI の作法ではない）
sandbox 実行 → 来歴付き還流（フェーズ1と同一機構）
```

**誤変換（最悪の故障モード）への三段防御:**
1. **研究者レビュー**（承認が API 形をとる — 監督原則の実装）
2. **sandbox**（出所を問わず: ネット遮断・書込制限・timeout）
3. **差分検証** `differential_check_llm_code`（bridge: POST /numeric/diffcheck/{id}）—
   ホワイトリスト内の式では決定的テンプレートが**数値的グラウンドトゥルース**になる。
   LLM 版とテンプレート版を sandbox で並走させ NB_RESULT の min/max を突き合わせる。
   「コードは正常に走るが式を write し間違えている」ケース（sin→cos 等）を数値で検出
   できることをテストで実証済み。

**テスト13件（`tests/test_numeric_vibe.py`、すべて偽 chat callable）:**
プロンプト内容（TeX・canonical・仕様・足場）／抽出門番（fenced・生コード・コード無し・
構文エラー、i18n 両言語）／レビューゲート（未承認実行拒否→承認→実行）／
**ホワイトリスト外の式（zeta(x)）でテンプレートは失敗するが LLM 経路は artifact を
生成できる**（この経路の存在理由の直接検証）／差分検証（正しいコード→match、
sin→cos の誤変換→両方正常終了するのに mismatch 検出）。
残る未検証は実 LLM の生成品質のみ — 実践ループ S-ac が評価器（測定結果はevaluations.md (reference omitted from the public snapshot) の台帳に時系列で蓄積する）。
