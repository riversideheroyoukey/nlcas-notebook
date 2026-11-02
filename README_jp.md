# CAS支援付き理論研究者デジタルノート

公開名は **NLCAS Notebook**（`nlcas-notebook`）。APIEMS2026で扱うResearch Notebookの実装です。
後継は [Out-of-Order CAS (OOCAS)](https://github.com/riversideheroyoukey/oocas) です
（アクセス可否はOOCAS側の公開設定・権限によります）。
[公開準備・検証](docs/github_publication.md)／[第三者コンポーネント](THIRD_PARTY_NOTICES.md)。
本体コードのライセンスは未選定です。

理論研究者がCAS文法を覚えずに、自然言語と手書き数式で計算ノートを操作できるシステム。

**コア機能:**
- **方程式形式入力（H-1）**: `H = J * x * y` のように LHS = RHS 形式で式登録、H がシンボルとして再利用可能
- **式番号参照（H-2）**: 数式内で `E1 * E4 + E3` のように式番号を使用、自動展開
- **P1 自動更新**: シンボル定義時に既存式の未定義シンボル一覧を自動更新
- **P2 自動代入**: Maxima では `H : RHS` を送信、Julia は明示的な substitute が必要
- **方程式操作シリーズ（H-5〜H-9）**: 登録済み方程式への後続操作（LLM 不要・ルールベース）
  - **H-5** 右辺を評価: `E3 の右辺を計算して`
  - **H-5z** 右辺に値を設定: `E3 の右辺を 0 とする` → 新規方程式 `RHS = 0` を登録
  - **H-7** 右辺への操作: `E3 の右辺を展開/整理/因数分解して`
  - **H-8** 両辺への操作: `E3 の両辺を m で割る`
  - **H-9** 方程式を解く: `E3 を x について解いて` / 連立方程式も対応
- **複数ステップの解法プラン（C-Ext-5）**: 「固有状態を求めて」のような固定ワーク
  フロー、「解法の候補」による Adviser 提案・選択、「計画を立てて」による自由計画、
  「自動選択して」による Supervisor 自動選定——いずれも実行前に確認を挟む。失敗時は
  安全に停止し、🔁 再試行／🔄 再計画で続行可能（詳細は
  [user_guide.md](docs/user_guide.md)）
- **汎用作図（数値計算層）**: 対話で曲線・等高線・不等式領域・注目点を重ね描き。
  式参照と描画仕様から共通処理で生成し、PNG/SVG/PDF・標本CSVを保存できます
  （[使い方](docs/user_guide.md#複数の式領域を組み合わせたグラフ)、[再現例](docs/scenarios/scenario_cartesian_plots.md)）。
- 手書き入力 → OCR → 式番号付きノート
- 自然言語指示 → LLM supervisor → 共通 operation → CAS（Maxima / Julia）→ 結果
- 履歴は replay 可能な DAG（分岐・再開・フォーク対応）
- CAS 文法は通常 UI に表示しない

**Julia セッション（第二量子化）固有の機能:**
- `operator_substitute`: 量子演算子を線形結合で置き換える（Bogoliubov 変換等）
- `pair_substitute`: 演算子ペア → c-number 置換（数密度 `c†(k)c(k)→n_k`、BCS ギャップ等）
- `wick_contract`: Wick 収縮（n 次演算子積から指定ペアを取り出し c-number に置換、(n-2) 次積が残る）
  - 4添字収縮値 `ρ(k,σ,p,τ)` / `κ(k,σ,p,τ)` による HF/BCS デカップリング
  - `declare_symmetry`: κ の反対称性 `κ(j,i)=-κ(i,j)` 等をシステムに宣言
- `re_evaluate_expr`: 収縮後の式のパラメータを更新して再評価（自己無撞着反復）
- `extract_order`: スカラー/演算子の多項式から λⁿ 項を取り出す（例: `H=H₀+λV` → n=1 → `V`）
- `truncate_series`: n 次以上の項を打ち切る（スカラー・演算子両対応）
- `operator_taylor_expand`: `f(H) ≈ Σ aₖ Hᵏ` — 演算子に解析関数を適用（高温展開・resolvent 等）
- 2添字演算子 `c(k, σ)`: スピン/偏光添字付き演算子の定義と代入
- スピン記法の自動正規化: `↑→u`, `↓→d`, `σ→s`（QuantumAlgebra.jl QuIndex 対応）
- スカラーパラメータのドメイン宣言: `@variables X::Real` / `@variables X::Integer`
- 虚数単位の自動変換: 未宣言の `i` → Julia `im`（宣言済み変数・束縛添字は変換しない）
- Heisenberg 方程式: `a_1 の Heisenberg 方程式 E1` 形式で入力

**入力補助:**
- Unicode ギリシャ文字（`ω`, `α`, `Δ` 等）を自動的に TeX 表記（`\omega` 等）に変換
- 添字付きシンボル宣言時の確認ダイアログ: `xi_k をスカラーとする` → 「成分インデックス」か「名前の一部」かを選択

**解析力学（独立アトム・モデル）:**
- `analytical_mechanics` ドメインでは座標 $q$ と速度 $\dot q$ を**独立変数**として扱う（Euler-Lagrange の定義どおり）。`\dot{q}` 入力のみで CAS 文法を見せず、$\partial L/\partial\dot q$・$d/dt$（全時間微分）・運動方程式が綺麗に出る
- Newton 力学（従属 $\dot q=dq/dt$）用に `classical_mechanics` ドメインを分離（設計＝[docs/19](docs/developer_manuals/19_analytical_mechanics_independent_variables.md)）

**F-1：多サイト格子ハミルトニアン（CAS 非経由の純 Python）:**
- **格子和**: `Σ_{<ij>}` を有限展開（具体化）／展開せぬシンボリック保持の両面で。プラガブル生成器（鎖・正方・三角・ハニカム・カゴメ・Lieb・任意グラフ、境界 open/periodic/cylinder）
- **格子図版**: ボンド構造を TikZ/SVG/PDF で描画し TeX エクスポートへ自動挿入（周期境界は破線）
- **解析的バンド構造**: tight-binding 分散 ε(k)（単/多バンド・t–t′・質量項・SSH・抽象結晶記述子）
- **合成スピン代数**: ネスト交換子（EOM/モーメント階層）・∞温度モーメント・連分数→スペクトル関数 S(ω)・比熱の高温展開・一般スピン S

**L-1：一般の線形基底変換:**
- 順変換 N 本から逆変換 `old = M⁻¹ new` を自動生成（Bogoliubov・スピン昇降・座標変換等）
- 正準/計量拘束 $M^\dagger G M = G$ の導出・適用・para-unitary 検証（Maxima/Julia 両対応、設計＝[docs/18](docs/developer_manuals/18_linear_transform_design.md)）

---

## 動作環境

- Python 3.14（`.venv/` 内）— `requirements.txt` / `requirements-numeric.txt` にピン
- Maxima 5.49（`maxima` コマンドが PATH にあること）
- Julia 1.12（`julia` コマンドが PATH にあること、第２量子化セッション用）—
  パッケージは `Project.toml` / `Manifest.toml` にピン
- Ollama（LLM supervisor 用、なくても起動可）

---

## インストールとクイックスタート

**macOS（推奨・ワンショット）:**

```bash
./setup.sh          # 前提ツール + 両 venv + ピン留め Julia 環境を導入
./scripts/start.sh  # → http://localhost:8000/notebook.html
```

**Linux（Docker、Ollama はホストで実行）:**

```bash
docker compose up --build   # → http://localhost:8000/notebook.html
```

**Windows（ネイティブ・推奨・実機検証済み、2026-07）:**

```powershell
.\setup_windows.ps1
.\scripts\start_windows.ps1   # → http://localhost:8000/notebook.html
```

**Windows（Docker Desktop、Docker 運用が必要な場合のみ・実機未検証・低優先度）:**

```powershell
docker compose -f docker-compose.windows.yml up --build   # → http://localhost:8000/notebook.html
```

Linux/Windows での Ollama 導入・接続、`docker-compose.windows.yml` が何を上書きして
いるかを含む詳細手順は **[docs/setup_guide_jp.md](docs/setup_guide_jp.md)** を参照。
数値計算層（N-1）のサンドボックスは macOS（sandbox-exec）・ベアメタル Linux（bwrap）・
ネイティブ Windows（専用低権限アカウント＋Task Scheduler、実機検証済み）は本物の
default-deny サンドボックス、Docker コンテナ内は Landlock＋専用ユーザーの簡易分離
（詳細: [docs/developer_manuals/29](docs/developer_manuals/29_deployment_and_sandbox_design.md)）。

ローカル推論サーバー（既定 Ollama）は `ui/app.py` 起動時に自動起動します。プロバイダ・コマンド・環境変数は `config/default_settings.json` の `local_llm_server` で設定でき、Ollama → vLLM / LocalAI の切り替えは設定編集のみ（詳細は `docs/developer_manuals/16_llm_model_setup.md`）。

---

## CLI（テスト・デバッグ用）

```bash
source .venv/bin/activate
python cli.py --title "セッション名"
```

| コマンド | 説明 |
|---|---|
| `reg <式>` | 式を登録（例: `reg (x^2-1)/(x-1)`） |
| `simplify E1` | 有理簡約 |
| `diff E1 x` | x で微分 |
| `taylor E1 x 0 3` | テイラー展開（x=0、3次） |
| `solve E1 x` | 方程式を解く |
| `fork S3 label` | S3 からブランチを分岐 |
| `switch <branch_id>` | ブランチを切り替え |
| `replay` | 現在ブランチを再生 |
| `history` | ステップ履歴を表示 |
| `exprs` | 式テーブルを表示 |
| `save <path>` | セッションを保存 |
| `help` | コマンド一覧 |

---

## テスト

```bash
source .venv/bin/activate

# 日常イテレーション：高速サブセット（julia/maxima を起動しない純粋な
# ユニット/パーサ/スキーマテストのみ、約1分40秒 / 2,773 件。tests/conftest.py が
# start_engine 等の呼び出しをソースから自動検出して @pytest.mark.slow を
# 付与するため手動タグ付け不要——マーカーは pytest.ini に登録済み）
pytest tests/ -m "not slow"

# マージ前のフルスイート（約13分40秒 / 4,192 件。Julia への CAS 実行が 57%、
# Julia プロセスの起動がなお 20% を占める。内訳と削減余地は
# docs/open_items.md の T-2 参照。件数は `pytest tests/ --collect-only -q` で
# 常時実測可能）
pytest tests/

# 特定モジュール：Engine テスト
pytest tests/test_engine.py -v

# 特定モジュール：新規操作テスト
pytest tests/test_new_operations.py -v

# 特定ファイルのみ
pytest tests/test_engine.py -v
```

## デバッグ REPL（`ui/debug_repl.py`）

ブラウザを使わずに解釈パイプライン・保存・読み込みを stdin/stdout でテストできる。
`--no-llm` 時は規則ベース解釈のみで動作し、確認ダイアログは第1候補を自動選択する。

```bash
# 対話モード（LLM あり）
./scripts/debug_repl.sh --lang ja

# 対話モード（LLM なし・規則ベースのみ）
./scripts/debug_repl.sh --no-llm --lang ja

# パイプモード（非対話・自動テスト向け）
printf "x^2 + 1\nE1 を x で微分して\nquit\n" | ./scripts/debug_repl.sh --no-llm --lang ja

# セッション読み込み
./scripts/debug_repl.sh --no-llm --session sessions/mysession.json

# ドメイン指定（LLM に domain-specific ヒントを注入）
./scripts/debug_repl.sh --lang ja --domains analytical_mechanics
```

詳細なサンプル入出力は PROGRESS.md (reference omitted from the public snapshot) のセクション 8 を参照。

---

## ディレクトリ構成

```
Research_Notebook/
│
├── notebook/               # コアエンジン
│   ├── engine.py           # NotebookEngine（全コンポーネント配線）
│   ├── engine_history_mixin.py  # EngineHistoryMixin — 履歴・ステップ管理（記録・削除・採番）
│   ├── engine_persistence_mixin.py  # EnginePersistenceMixin — セッション永続化（save/load・CASスナップショット）
│   ├── models/             # Pydantic データモデル
│   │   ├── session.py
│   │   ├── expression.py   # Expression, Assumption (IRツリー保持)
│   │   ├── quantity.py     # Quantity, MathType
│   │   ├── symbol_context.py # SymbolContext (Semantic Layer)
│   │   ├── ir.py           # 抽象構文木 AST (Formal Layer)
│   │   ├── step.py         # Step, StepKind, StepStatus
│   │   └── branch.py
│   └── runtime/            # 実行層（3層コンパイラアーキテクチャ）
│       ├── shared_parser.py        # SharedParser — Surface AST 生成（Juxtaposition / BracketExpr 等）
│       ├── semantic_resolver.py    # BaseSemanticResolver — 意味解決基底クラス
│       ├── julia_semantic_resolver.py  # JuliaSemanticResolver（Commutator / Mul 昇格）
│       ├── maxima_semantic_resolver.py # MaximaSemanticResolver
│       ├── code_generator.py       # BaseCodeGenerator — コード生成基底クラス
│       ├── julia_code_generator.py     # JuliaCodeGenerator
│       ├── maxima_code_generator.py    # MaximaCodeGenerator
│       ├── backend_handler.py      # BackendHandler ABC（抽象メソッド + no-op フック）
│       ├── maxima_backend_handler.py   # MaximaBackendHandler — Maxima 固有処理
│       ├── julia_backend_handler.py    # JuliaBackendHandler — Julia 固有処理・粒子演算子パーサー
│       ├── julia_handler_common.py     # Julia ハンドラのモジュールレベル IR/演算子ヘルパー・正規表現・定数
│       ├── julia_sympy_mixin.py        # JuliaSympyMixin — SymPy チェーン・ルーティング群
│       ├── julia_build_mixin.py        # JuliaBuildMixin — CAS コマンド構築群（build_command + _build_*）
│       ├── julia_undefined_mixin.py    # JuliaUndefinedSymbolMixin — 未定義シンボル検出
│       ├── julia_latex_mixin.py        # JuliaLatexMixin — LaTeX 表示・整形群
│       ├── lhs_formatting.py       # LHS／微分 LaTeX 整形・等式パース（engine.py から分離）
│       ├── backend_factory.py      # BackendFactory — make_backend(mode) でアダプター一式を生成
│       ├── command_builder.py      # CommandBuilder 共通基底クラス（テンプレート駆動）
│       ├── maxima_command_builder.py   # MaximaCommandBuilder
│       ├── julia_command_builder.py    # JuliaCommandBuilder
│       ├── builder.py              # re-export shim（BuildResult / MaximaCommandBuilder / JuliaCommandBuilder）
│       ├── latex_converter.py      # LatexConverter 共通基底クラス（構造ルール）
│       ├── maxima_latex_converter.py   # MaximaLatexConverter（%pi / %e 等）
│       ├── julia_latex_converter.py    # JuliaLatexConverter（pi / Inf / im 等）
│       ├── maxima_ir_compiler.py   # MaximaIRCompiler（再帰下降パーサー）
│       ├── ir_compiler.py          # re-export shim → MaximaIRCompiler
│       ├── expr_generator.py       # ExprGenerator — IR AST → 文字列（バックエンド非依存）
│       ├── maxima_adapter.py       # Maxima 常駐 REPL
│       ├── julia_adapter.py        # Julia 常駐プロセス（第２量子化セッション用）
│       ├── _julia_driver.jl        # Julia driver script（QuantumAlgebra.jl + Symbolics.jl ロード）
│       ├── operation_validator.py  # operation 検証
│       ├── replay.py               # 履歴再生
│       ├── branch_manager.py       # ブランチ管理
│       ├── llm_supervisor.py       # LLM 3-phase 解釈（extra_pre_phase1 フック対応）
│       ├── llm_client.py           # LLM API クライアント
│       ├── llm_adviser.py          # LLM 考察（adviser）
│       ├── index_contractor.py     # 添字の静的縮約・Kronecker δ 適用
│       ├── auto_einstein.py        # Einstein 縮約（上下添字の自動和）
│       ├── notation_profile_loader.py  # ドメイン別 notation profile 読み込み
│       ├── lattice_figure.py       # 格子図生成（TikZ/SVG/PDF・F-1・CAS 非経由）
│       ├── spin_algebra.py         # 合成スピン代数（ネスト交換子・∞T モーメント・連分数・S(ω)・F-1 3c・CAS 非経由）
│       └── i18n.py                 # t(key, lang) 翻訳ヘルパー
│
│   （notebook/ 直下には他に engine_numeric_mixin.py・engine_plan_mixin.py・
│    engine_laplace/fourier/finance/game/optimization/free_params/deferred_forms/wick_mixin.py
│    など機能族ごとの mixin、および numeric/（下記）がある）
│
├── notebook/numeric/       # 数値計算層（N-1、CAS では解析的に解けない式を実行）
│   ├── sandbox.py          # SandboxRunner — OS別サンドボックス実装（後述）
│   ├── templates.py        # 決定的テンプレ（parameter_sweep/plotting/eigen 等）
│   ├── plot_spec.py / plot_codegen.py / plot_runtime.py / plot_prompt.py  # 汎用の宣言的作図
│   ├── expr_to_numpy.py    # 3層コンパイラ流儀の関数ホワイトリスト lowering
│   ├── llm_codegen.py      # バイブコーディング経路（レビューゲート必須）
│   └── spec_adviser.py / spec_projection.py / spec_validate.py / graph.py / intent_registry.py
│
├── ui/                     # Web UI
│   ├── app.py              # FastAPI アプリ（サーバーエントリポイント）
│   ├── notebook_bridge.py  # /api/notebook/* エンドポイント（薄いルーティング層）
│   ├── interpret_pipeline.py # 自然言語 → operation 変換パイプライン本体（AUD-5 でパーサを下記へ分割）
│   ├── interpret_common.py # interpret_* 共有の低レベル定数（_EIDS_RE 等）
│   ├── interpret_equation_ops.py # H-5/H-7/H-8/H-9 パーサ（solve・両辺操作・RHS→操作）
│   ├── interpret_l1.py     # L-1 線形基底変換・正準/計量拘束のパーサ
│   ├── interpret_f1.py     # F-1 格子和・分散・結晶記述子・スピン代数のパーサ
│   ├── session_io.py       # セッション保存・読み込み・自動保存
│   ├── vlm_client.py       # OCR パイプライン（domain ヒント注入）
│   ├── apfel_client.py     # Apple Vision Framework ブリッジ（macOS NPU OCR）
│   ├── image_utils.py      # 手書き画像の前処理（余白トリミング等）
│   ├── katex_utils.py      # KaTeX ローカルアセット管理
│   ├── debug_repl.py       # デバッグ用 stdin/stdout REPL（--no-llm 対応）
│   ├── llm_server_launcher.py  # ローカル推論サーバーの設定駆動起動（Ollama/vLLM/LocalAI 切替可）
│   ├── routes/              # bridge から分割された APIRouter 群（numeric/history/persistence/
│   │                        #  export/operations/session/advisory、全49エンドポイント）
│   ├── interpret_plan.py / interpret_numeric.py  # C-Ext-5 プラン層・N-1 数値フォークのNLトリガ
│   ├── notebook.html       # メインノート画面
│   ├── notebook_main.js    # DOM/状態ハブ。ES modules へ機能族分割済み（notebook_numeric.js /
│   │                        #  notebook_plan.js / notebook_branch.js / notebook_timeline.js 等）
│   ├── notebook_main.css   # メインノート CSS
│   ├── index.html          # 手書き入力画面
│   └── main.js             # 手書き入力 JS
│
├── schemas/
│   └── common_operations.schema.json   # 109 operations 定義
│
├── backends/
│   ├── maxima.full.json    # Maxima コマンドテンプレート（93 操作）
│   └── julia.full.json     # Julia コマンドテンプレート（50 操作、QuantumAlgebra.jl + Symbolics.jl 両対応）
│
├── setup.sh / setup_windows.ps1   # macOS / ネイティブ Windows ワンショットインストーラ
├── scripts/                # 起動・ビルド・プロビジョニングスクリプト
│   ├── start.sh / start_windows.ps1  # UI サーバー起動（venv 自動有効化、mac・Windows）
│   ├── setup_windows_numeric_sandbox.ps1  # G-4 数値サンドボックスの管理者権限プロビジョニング
│   ├── debug_repl.sh       # デバッグ REPL 起動
│   ├── build_sysimage.sh / build_julia_sysimage.jl  # Julia カスタムシステムイメージ構築
│   └── sysimage_precompile.jl
│
├── models/                 # LLM エンドポイント設定
│   ├── default/config.json         # デフォルトモデルルーティング
│   ├── ollama_qwen25_coder_32b/    # supervisor（推奨・稼働中）
│   ├── ollama_gpt_oss_20b/         # supervisor（代替・稼働中）
│   ├── ollama_qwen3_vl/            # OCR + supervisor（稼働中）
│   ├── cloud_openai_gpt54/         # adviser 用（API キー要）
│   ├── lmstudio_remote_gpt-oss-120b/ # adviser 用（LM Studio リモート）
│   ├── lmstudio_gpt_oss_20b/       # LM Studio 経由の代替
│   ├── engines/                    # OCR 特化エンジン設定
│   ├── refiners/                   # OCR 後処理設定（vision_input 対応）
│   ├── particles.json              # 粒子テーブル（32粒子: フェルミオン18 + ボソン14）
│   └── domains/                    # 研究分野別プロンプトヒント（32 分野）
│       ├── categories.json         # カテゴリ定義（order・default_backend）
│       └── <domain_name>/          # meta.json / supervisor_hint.txt / ocr_hint.txt
│
├── i18n/                   # 翻訳テーブル
│   ├── messages.json       # バックエンドメッセージ（engine / supervisor / adviser）
│   ├── ui_messages.json    # フロントエンド UI 文字列
│   └── nl_patterns.json    # 自然言語入力の文法パターン（言語別 named-group 正規表現）
│
├── config/                 # アプリケーション設定
│   └── default_settings.json  # "language" — システムデフォルト言語
│
├── sessions/               # 保存セッション（JSON / .mac スナップショット）
│   └── autosave/           # 自動保存ローテーション（slot_1〜slot_5）
│
├── tests/                  # pytest テストスイート（件数は `pytest --collect-only -q` で実測）
│
├── docs/                   # 設計文書。索引は developer_manual.md（技術文書 01〜29）を参照。
│   ├── developer_manual.md                    # デベロッパーマニュアル索引（表紙）
│   ├── user_guide.md                          # ユーザーガイド（操作入力規則・全機能）
│   ├── operation_reference.md                 # 共通 operation リファレンス
│   ├── setup_guide.md                         # セットアップガイド表紙（jp/en、OS別詳細手順）
│   ├── open_items.md                          # 未実装・課題・先送り事項の一元管理
│   ├── scenarios.md / scenarios/               # 実践ループ・シナリオ表紙＋詳細（scenario_*_jp.md / _en.md）
│   └── developer_manuals/                      # 番号付き技術文書（01〜29、developer_manual.md から辿る）
│
├── cli.py                  # CLI エントリポイント
├── PROGRESS.md             # 実装進捗
└── CLAUDE.md               # 開発方針（AI 向け）
```

---

## 依存パッケージのインストール

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn pydantic httpx pytest anyio
```

Maxima のインストール（macOS）:

```bash
brew install maxima
```

数値計算層（N-1、任意）を使う場合は専用の実行環境を作る（アプリ本体の venv とは分離）:

```bash
python3 -m venv .venv-numeric
source .venv-numeric/bin/activate
pip install -r notebook/numeric/requirements.txt
```

---

## セッションの保存・読み込み

UI の「💾 保存」ボタン → 上書き保存 または ファイル名を指定して保存。

自動保存は 5 分ごとに `sessions/autosave/slot_N.json`（N=1〜5）へローテーション保存。

コードから直接操作する場合:

```python
from notebook.engine import NotebookEngine

# 保存
with NotebookEngine(title="実験") as e:
    e.save("sessions/my_session.json")

# 読み込み
with NotebookEngine.load("sessions/my_session.json") as e:
    result = e.replay_current_branch()
```

---

## 利用モード

| モード | primary backend | 用途 |
|---|---|---|
| standard_symbolic | Maxima | 古典力学・解析力学・場の理論・通常の記号計算 |
| second_quantization | Julia（QuantumAlgebra.jl） | 凝縮系・量子光学・場の量子論・核物理・スピン系など第２量子化を使う分野 |

新規セッションモーダルで第２量子化カテゴリのドメインを一つでも選択すると、自動で `second_quantization` モード（Julia バックエンド）になります。

---

## 関連文書

- [デベロッパーマニュアル（技術文書の索引）](docs/developer_manual.md) — 設計・アーキテクチャ・内部仕様の表紙・目次
- [ユーザーガイド](docs/user_guide.md)
- [操作リファレンス](docs/operation_reference.md)
- [未実装・課題一覧](KNOWN_ISSUES.md)
- 旧 E-3 ワークフロー検証を分割した独立シナリオ: [Jaynes-Cummings](docs/scenarios/scenario_jaynes_cummings_jp.md)・[ハイゼンベルク2サイト](docs/scenarios/scenario_heisenberg_two_site_jp.md)・[クーロンゲージ](docs/scenarios/scenario_coulomb_gauge_jp.md)・[理想気体](docs/scenarios/scenario_ideal_gas_jp.md)
- [単振り子解析力学シナリオ（Maxima）](docs/scenarios/scenario_pendulum_lagrangian_jp.md)
- [シンボルドメイン・添字分類設計](docs/developer_manuals/12_symbol_domain_design.md)
- [ベクトル場・座標系設計仕様](docs/developer_manuals/13_vector_field_design_spec.md)
- [線形基底変換（L-1）設計](docs/developer_manuals/18_linear_transform_design.md)
- [解析力学：座標と速度を独立変数とするモデル（doc 19）](docs/developer_manuals/19_analytical_mechanics_independent_variables.md)
- [Notation Bridge カタログ](docs/developer_manuals/08_notation_bridge_catalog.md)
- [添字記法・Einstein 縮約設計](docs/developer_manuals/06_index_notation_design.md)
- [LLM モデル設定ガイド](docs/developer_manuals/16_llm_model_setup.md)
- [i18n 設計](docs/developer_manuals/15_i18n_design.md)
- [全体仕様書](docs/developer_manuals/01_theoretical_research_notebook_spec.md)
- [履歴設計仕様](docs/developer_manuals/02_history_spec.md)
- [数値計算層（N-1）設計](docs/developer_manuals/22_numeric_layer_design.md) — 理論式→sandbox 実行→来歴付き還流（フェーズ1実装済み・UI 実装済み — 「🔢 数値計算へ」）
- [Cartesian plot layers](docs/developer_manuals/36_cartesian_plot_layers.md) — 汎用作図仕様・API・式参照の保存・格子近似の制限

## Julia の事前コンパイル（Time-to-first-X 問題の解消）

Julia バックエンドの初回起動時やモジュール読み込み時間を短縮するため、カスタム・システムイメージ (`sysimage.dylib`) の構築スクリプトを用意しています。パッケージの追加・更新があった際は、以下のコマンドを実行してシステムイメージを再構築してください。

```bash
# Juliaカスタムシステムイメージの構築（数分かかります）
./scripts/build_sysimage.sh
```

構築後は、システムが自動的にこの `sysimage.dylib` を読み込んで起動し、複雑な計算でもミリ秒単位で結果が返るようになります。
