# セットアップ・配布ガイド (Setup & Distribution Guide)

Research Notebook の導入手順。対象プラットフォームは以下の2系統。

| 環境 | 推奨手段 | LLM (Ollama) | 数値層 (N-1) |
|---|---|---|---|
| **macOS**（本命） | `./setup.sh`（ネイティブ） | ホストネイティブ (Metal) | 動作可 (sandbox-exec) |
| **Linux ネイティブ**（Docker を使わない場合） | 手動導入 | ホストネイティブ | 動作可 (bubblewrap・実機検証済み) |
| **Windows ネイティブ**（推奨・実機検証済み） | `setup_windows.ps1` | ホストネイティブ | 動作可（専用アカウント＋Task Scheduler・実機検証済み） |
| **Linux / Windows（Docker）**（Windows は需要駆動・実機未検証） | Docker (`docker compose`) | **ホストネイティブ**（別途） | 動作可（簡易分離・下記参照） |

> **設計上の前提**：Ollama は 20〜32B 級のモデルを使うため、全プラットフォームで
> **ホスト側で GPU を使って動かす**（macOS=Metal, Linux=CUDA）。コンテナには入れない。
> したがって「Docker で GPU パススルー」という難所は発生しない。

各プラットフォームの節はそれだけ読めば導入が完結するように独立させてある
（§2 macOS・§3 Linux・§4 Windows）。共通事項は §1・§6 以降にまとめた。

---

## 1. 前提ツール (Prerequisites)

| ツール | バージョン（検証済み） | 用途 |
|---|---|---|
| Python | 3.14.4 | アプリ本体・数値層 |
| Maxima | 5.49.0 | 記号計算バックエンド（標準セッション） |
| Julia | 1.12.6 | 第二量子化セッション（QuantumAlgebra/Symbolics/SymPy） |
| Ollama | 0.20.7 | LLM Supervisor / OCR / 数値コード生成 |

Python 依存は `requirements.txt`（アプリ `.venv`）と `requirements-numeric.txt`
（数値層 `.venv-numeric`）に**実 venv から完全ピン**。Julia 依存は `Project.toml` /
`Manifest.toml` にピン（QuantumAlgebra 1.6.0 / Symbolics 7.19.0 / Latexify 0.16.10 /
SymPy 2.3.3 / SymbolicNumericIntegration 1.10.0 / JSON3 1.14.3）。

---

## 2. macOS（推奨：ワンショット）

### 2.1 導入

```bash
./setup.sh
```

これが実行する内容：

1. Homebrew で `maxima` / `julia` / `ollama` を導入（未導入分のみ）
2. `.venv`（アプリ）と `.venv-numeric`（数値層）を作成し requirements を pip install
3. Playwright の Chromium を導入（UI テスト用。失敗してもアプリには無影響）
4. `Project.toml` から Julia 環境を `Pkg.instantiate()`
5. （任意）Julia sysimage をビルド（約10分、起動高速化）
6. （任意）Ollama モデルを pull

オプション：

```bash
./setup.sh --with-sysimage   # sysimage を確認なしでビルド
./setup.sh --no-models       # Ollama モデルの pull をスキップ
./setup.sh --ci              # 非対話（任意ステップは全て no）
```

### 2.2 手動で行う場合（setup.sh の中身）

```bash
brew install maxima julia ollama
python3.14 -m venv .venv        && ./.venv/bin/pip install -r requirements.txt
python3.14 -m venv .venv-numeric && ./.venv-numeric/bin/pip install -r requirements-numeric.txt
julia --project=. -e 'using Pkg; Pkg.instantiate()'
# 任意：./scripts/build_sysimage.sh
```

### 2.3 Ollama モデル

```bash
brew install ollama
ollama serve &                       # Metal で GPU 利用（Apple Silicon）
ollama pull qwen2.5-coder:32b gpt-oss:20b qwen3-vl:latest
```

使用モデルの役割別対応は §5 参照（`setup.sh` が上記を案内・実行する）。

### 2.4 起動

```bash
./scripts/start.sh
# → http://localhost:8000/notebook.html
```

Ollama を起動していなくてもアプリは起動する（LLM 支援機能のみ無効）。

---

## 3. Linux（Docker）

アプリスタック（Python + Maxima + Julia）だけをコンテナ化し、**Ollama はホスト**で動かす。

> **数値層 (N-1) は Docker コンテナ内でも動作するが、簡易的な分離にとどまる
> （macOS/ベアメタル Linux と同等の「本物のサンドボックス」ではない）。**
> `notebook/numeric/sandbox.py` はまず bwrap を試すが、コンテナ内では bwrap
> 内部のループバック設定処理が `Failed RTM_NEWADDR: No child processes` で
> 失敗する（原因不明、複数の仮説を実機で切り分け済み、詳細は
> `docs/open_items.md` §G-2）。そのため Docker では代わりに**専用低権限
> ユーザー＋Landlock（ネットワークのみ）**にフォールバックする——追加権限
> （`cap_add`）は一切不要。実際に確保できる範囲：①ネットワーク拒否は
> **TCP のみ**（UDP・raw socket は対象外）②自タスクの work dir 外への
> 書込みは拒否（堅牢）③他セッション・他タスクからの read 隔離は**弱い**
> （`sessions/` 自体は非公開だが、`task_id` が連番・`execution_id` が
> タイムスタンプ文字列なので他タスクのパスは推測しやすく、同じ低権限
> ユーザーを共有するため兄弟タスクとの相互隔離もできない）。「無いよりマシ
> 程度の分離」として受容した判断：数値層に到達するコードは①決定的テンプレ
> ート（関数ホワイトリストがネットワーク/パス操作を表現できず構造的に
> 不可能）②LLM 生成コード（`approve_numeric_code` の人間レビュー必須）の
> 2経路のみで、OS レベルの分離はその二次的なバックストップという位置づけ。
> 詳細・調査経緯は `docs/open_items.md` §G-2 参照。
>
> ほか：Apple Vision OCR は無効（`ui/apfel_client.py` は macOS 専用・既定未配線、
> Docker では Ollama 側の VLM `qwen3-vl` で手書き OCR）。sysimage はイメージ内で
> ビルドされる（アーキ依存のためホストのものは使わない）。

### 3.1 Ollama の導入・接続（Linux・GPU 利用）

```bash
# 1) Ollama 本体
curl -fsSL https://ollama.com/install.sh | sh

# 2) NVIDIA GPU を使う場合：ドライバ + CUDA が入っていること
nvidia-smi                           # GPU が見えることを確認

# 3) 常駐（systemd がインストール時に有効化される。手動なら:）
ollama serve &

# 4) モデル取得（役割別対応は §5）
ollama pull qwen2.5-coder:32b
ollama pull gpt-oss:20b
ollama pull qwen3-vl:latest

# 5) 動作確認（コンテナからも届く localhost）
curl http://localhost:11434/api/tags
```

Ollama は既定で `127.0.0.1:11434` を listen する。`docker-compose.yml` は
`network_mode: host` を使うため、コンテナ内の `localhost:11434` がそのままホストの
Ollama に届く（`models/*/endpoint.json` の変更は不要）。別ホストの Ollama を使う／
ブリッジネットワークにする場合は `OLLAMA_HOST=0.0.0.0:11434` で listen させ、
`base_url` を該当ホストに向ける。

### 3.2 起動

```bash
docker compose up --build
# → http://localhost:8000/notebook.html
```

---

## 4. Windows（ネイティブ推奨）

> **ネイティブ Windows（Docker を使わない）を推奨する。** アプリ本体・数値
> サンドボックスとも実機検証まで完了しており（§4.1）、Docker Desktop 経由
> （§4.2）は WSL2 が前提でセットアップの手間がむしろ多く、数値サンドボックス
> もネイティブ側より弱い（TCP のみ）。Docker 運用を積極的に希望する理由が
> なければ §4.1 を使う（`docs/open_items.md` §G-3）。

### 4.1 ネイティブ Windows（推奨・実機検証済み）

アプリ本体（CAS＋LLM のノートブック機能）はただの Python（FastAPI/uvicorn）なので、
Windows ネイティブで動かない理由は無い。数値層のサンドボックスも実装済み
（専用低権限アカウント＋Task Scheduler＋アカウント単位ファイアウォールルール、
`docs/open_items.md` §G-4）だが、**別途・管理者権限での一回限りのセットアップ**
（`scripts\setup_windows_numeric_sandbox.ps1`、下記）が必要——それを実行するまでは
既存のフェイルセーフで数値計算の実行だけがブロックされる（アプリ本体には影響しない）。

**方針は `setup.sh` と同じ**：Python・Julia・Maxima・Ollama は各公式インストーラに
任せ、このスクリプトは①導入確認、②アプリ側のセットアップ（venv・pip・Julia
instantiate）だけを行う。

```powershell
# 前提ツールの公式インストーラ（このスクリプトは自動導入しない、確認のみ）：
#   Python 3.14  https://www.python.org/downloads/
#   Julia        https://julialang.org/install/（juliaup）
#   Maxima       https://maxima.sourceforge.io/download.html
#   Ollama       https://ollama.com/download/windows（任意、無くてもアプリは起動する）

.\setup_windows.ps1
.\scripts\start_windows.ps1
# → http://localhost:8000/notebook.html
```

**既知の注意点：**

- **PowerShell の実行ポリシー。** 既定で `Restricted`/`RemoteSigned` になっている
  環境では `.ps1` スクリプトの実行自体がブロックされる場合がある。エラーが出たら
  管理者権限不要の範囲で `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` を
  実行するか、`powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1` で
  起動する。
- **Maxima は公式インストーラが PATH に自動追加しないことがある。** 導入後
  `maxima` コマンドが見つからない場合、インストール先の `bin\` フォルダ
  （例：`C:\maxima-5.49.0\bin`）を手動で PATH に追加する。
- **PyCall/SymPy.jl の Python 解決バグへの対策込み。** Docker で発覚した
  「PyCall がベースの Python を拾って sympy が見つからず `Pkg.precompile()` が
  失敗する」問題（`docs/developer_manuals/29_deployment_and_sandbox_design.md`
  §2.2）と同じ地雷が、公式インストーラで入れた Python が PATH にある場合ここでも
  起こり得るため、`setup_windows.ps1` は Julia のビルド前に `$env:PYTHON = ""`
  を設定し、PyCall に独自 Conda ブートストラップを強制する（macOS の実績経路と
  統一）。
- **`.venv-numeric` のパスは Windows レイアウト**（`Scripts\python.exe`）に
  対応済み（`notebook/numeric/sandbox.py`）。
- **数値計算の実行には追加のセットアップが必要。** `setup_windows.ps1`（このスクリプト）
  は非昇格のまま完結するよう意図的に数値サンドボックスの導入を含めていない。
  管理者権限の PowerShell で一度だけ以下を実行する：
  ```powershell
  .\scripts\setup_windows_numeric_sandbox.ps1
  ```
  専用の低権限ローカルアカウント（`numeric`）を作成し、パスワードを DPAPI 暗号化して
  保存、Task Scheduler の `/ru` に必要な「バッチジョブとしてログオン」権を付与、
  そのアカウント宛の全プロトコル送信ブロックのファイアウォールルールを追加する。
  これを実行するまでは数値計算のコード生成・レビューまでは動くが、**実行**は
  明確なエラーでブロックされる。

詳細・実機検証の経緯は `docs/developer_manuals/29_deployment_and_sandbox_design.md` §5・§7。

---

### 4.2 Windows（Docker Desktop、Docker運用が必要な場合のみ・実機未検証）

> **数値層 (N-1) は Docker コンテナ内でも動作するが、簡易的な分離にとどまる
> （§3 と同じ設計・同じ制約）。** Docker Desktop（WSL2 バックエンド＝Linux
> カーネル）上でも `notebook/numeric/sandbox.py` は bwrap → 専用低権限ユーザー
> ＋Landlock（ネットワークのみ）の順にフォールバックする——追加権限不要。
> bwrap がコンテナ内で失敗する既知問題（§3参照）は WSL2 でも起きる可能性が
> 高い（未検証だが同じ Docker/containerd 系統のため）ので、実際には後者の
> 経路が使われる想定。確保できる範囲・受容した限界は §3 の説明と同一
> （ネットワークは TCP のみ、他タスク・他セッションとの read 隔離は弱い）。
> **この経路自体（`docker-compose.windows.yml` の実ビルド動作含む）は実機で
> 一度も確認していない**（`docs/open_items.md` §G-3——優先度は需要駆動に
> 格下げ済み。特別な理由がなければ §4.1 のネイティブ Windows を使うこと）。
>
> ほか：Apple Vision OCR は無効（Docker では Ollama 側の VLM `qwen3-vl` で
> 手書き OCR）。sysimage はイメージ内でビルドされる。

#### 4.2.1 Ollama の導入・接続（Windows）

- **推奨**：WSL2（Ubuntu）を用意し、上記 §3.1 の Linux 手順で Ollama を導入。
  Docker Desktop の WSL2 バックエンド経由で、アプリコンテナから
  `host.docker.internal:11434` に自動的に届く。NVIDIA GPU は WSL2 + CUDA で利用可
- ネイティブ Windows の Ollama（`https://ollama.com/download/windows`）でも同様に
  `host.docker.internal:11434` 経由で接続できる（設定変更不要）

使用モデルの役割別対応は §5 参照。

#### 4.2.2 起動

Windows 専用の compose ファイルを用意済み：

```powershell
docker compose -f docker-compose.windows.yml up --build
# → http://localhost:8000/notebook.html
```

`docker-compose.yml`（Linux 用）との違いは2点：

1. **`network_mode: host` を使わない。** Docker Desktop のホストネットワーキングは
   既定オフのベータ機能なので、通常のブリッジネットワーク＋ポートマッピング
   （`ports: ["8000:8000"]`）にしている。
2. **Ollama への到達を `host.docker.internal` 経由にする。** Docker Desktop がこの
   ホスト名をホストへ自動解決するため `extra_hosts` の追加設定は不要。ただし
   `models/*/endpoint.json` はネイティブ macOS/Linux 用に `localhost` で書かれて
   いるため、既定ロール（Supervisor/Coder=`qwen2.5-coder:32b`、代替=`gpt-oss:20b`、
   OCR エンジン=`qwen3-vl`）に対応する4ファイルだけを `host.docker.internal` 書換え
   版に差し替えて `models.windows/` に用意し、`docker-compose.windows.yml` が
   バインドマウントで上書きする（ファイル単位のマウントなので `system_prompt.txt`
   等の兄弟ファイルはイメージのものがそのまま使われる）。他のロールを別の
   ローカル Ollama プロファイルに向け替える場合は、対応するファイルを
   `models.windows/` に追加し同様にマウント行を足す。

### 4.3 参考：Docker Desktop で macOS を使う場合（非推奨経路）

macOS では §2 の `./setup.sh`（ネイティブ）を推奨。どうしても Docker を使うなら、

`docker-compose.windows.yml` と同じ考え方（ブリッジ＋ポートマッピング＋
`host.docker.internal` 書換え）を流用できる。

---

## 5. 使用モデル（役割別・全プラットフォーム共通）

`config/default_settings.json` の実際の既定値ベース（詳細は
`docs/developer_manuals/16_llm_model_setup.md`）。アプリは OpenAI 互換エンドポイント
（既定 `http://localhost:11434`）に接続する。プロバイダの切替（Ollama → vLLM /
LM Studio 等）は `config/default_settings.json` の `local_llm_server` と
`models/*/endpoint.json` の `base_url` を編集するだけ。

| 役割 | モデル | 設定場所 |
|---|---|---|
| Supervisor/Coder（推奨・既定） | `qwen2.5-coder:32b` | `models/ollama_qwen25_coder_32b/` |
| Supervisor（代替） | `gpt-oss:20b` | `models/ollama_gpt_oss_20b/` |
| 手書き OCR / VLM（既定） | `qwen3-vl:latest` | `models/engines/ollama_qwen3_vl/` |

Adviser の既定ターゲットはユーザー所有のリモート LM Studio マシン
（`models/lmstudio_remote_gpt-oss-120b/`）— 単一マシン構成では到達しないため、
ローカルで使いたい場合は `config/default_settings.json` の `adviser_target` を
上記のいずれかのローカル Ollama プロファイルに向け替える。

### GPU が無い / 小型モデルにしたい場合

`models/*/endpoint.json` と各役割の設定を、より小さいモデル（例：`llama3.2` 等）に
差し替える。役割ごとに独立して変更できる。CPU のみでは 20〜32B 級は実用速度に
ならないため、モデルの縮小を推奨。

---

## 6. テスト（開発者向け）

```bash
source .venv/bin/activate
pytest                      # Maxima/Julia を起動、秒〜分オーダー
playwright install chromium # UI テストを走らせる場合は一度だけ
```

---

## 参照

- `README_jp.md` / `README_en.md` — 機能概要
- `docs/developer_manuals/16_llm_model_setup.md` — LLM/モデル設定の詳細
- `docs/developer_manuals/22_numeric_layer_design.md` — 数値層（N-1）の設計と実行モデル
- `docs/open_items.md` §G — セットアップ・配布の未解決事項（クロスプラットフォーム
  数値サンドボックス等）
