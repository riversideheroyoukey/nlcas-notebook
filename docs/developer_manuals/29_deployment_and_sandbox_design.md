# 配布・セットアップとクロスプラットフォーム数値サンドボックス設計 — G-1/G-2/G-3

**初版:** 2026-07-20〜21（公開準備。依存の完全ピン・macOS/Linux/Windows 配布・数値層
サンドボックスの OS 別実装を、Ubuntu 実機での実ビルド・実 API 検証まで含めて設計・実装）
**status:** macOS・Linux（ベアメタル／Docker、Ubuntu 実機）は完了・検証済み
（open_items G-1/G-2）。**Windows は実機未検証**（open_items G-3）——ネイティブ
Windows のアプリセットアップ（`setup_windows.ps1`、2026-07-21 実装）と Docker
Desktop/WSL2 経由の両方とも、設計・実装は完了しているが実機で一度も動かしていない。

この文書は open_items の G セクション（セットアップ・配布）の設計判断・調査経緯を
まとめたもの。ステータスの一次情報は `docs/open_items.md` §G（進行中の課題管理）、
本文書はその**設計文書版**（why と実装詳細の永続的な記録）。

---

## 1. 動機

公開に向けて「他の誰かが動かせる」状態にする必要があった。障害は3層：

1. **依存が固定されていない**（Python venv・Julia パッケージとも実 venv の状態に依存、
   別マシンで再現しない）
2. **Ollama を含む全スタックのセットアップが煩雑**（Maxima + Julia + Ollama + Python の
   個別導入）
3. **数値層のサンドボックス（`notebook/numeric/sandbox.py`）が macOS `sandbox-exec`
   専用**——Linux/Windows では数値層が一切動かない

3つとも実機（Ubuntu 22.04.5 LTS、kernel 6.8.0、Docker 29.6.2、IP 192.168.1.44）に
rsync でツリーを転送し、実際にビルド・起動・API 呼び出しをして検証した
（「動くはず」で終わらせない、が本文書全体を通した方針）。

## 2. 依存の完全ピン（G-1）

### 2.1 Python

`requirements.txt`（アプリ `.venv`）・`requirements-numeric.txt`（数値層専用
`.venv-numeric`）を、**実際に動いている venv から `pip freeze` した結果をそのまま**
使う。抽象的な「動くはずのバージョン範囲」ではなく、実証済みの1点を固定する。

### 2.2 Julia：3つの落とし穴

`Project.toml` + `Manifest.toml`（`=` 厳密 compat）で QuantumAlgebra/Symbolics/
Latexify/SymPy/SymbolicNumericIntegration/JSON3 をピン。Docker での実ビルドで
初めて発覚した実バグ3件：

1. **`julia_adapter.py` がグローバル環境（`~/.julia/environments/v1.12`）に依存**
   していた。`--project=<repo>` に変更してリポジトリローカルの `Project.toml` を
   使うよう修正。sysimage も同じ project に対してビルドし直す必要があった
   （`scripts/build_julia_sysimage.jl`）。
2. **`Project.toml` に `name`/`uuid` を書いていたため `Pkg.precompile()` が壊れる**。
   `name`/`uuid` があるとリポジトリを「実パッケージ」とみなし `src/<Name>.jl` を
   探しに行って `Missing source file` で失敗する。このリポジトリは Julia コードを
   `_julia_driver.jl` というスクリプトとして実行するだけで `using ResearchNotebook`
   される想定は無い——`[deps]`+`[compat]` のみの environment-only manifest に修正。
   ローカル（macOS）では `Pkg.precompile()` を単体で明示的に呼んだことが無かったため、
   この潜在バグに気づいていなかった。
3. **`PyCall.jl`/`SymPy.jl` の Python 解決が `.venv` と無関係**。`SymPy.jl` は
   `_pyimport_conda("sympy", "sympy")` で Python の sympy を要求するが、これは
   PyCall が選ぶ Python（ビルド時に PATH 上の `python3` を見つければそちらを、
   無ければ独自のプライベート Conda を自動構築）に依存し、アプリの `.venv`/
   `.venv-numeric` とは**完全に無関係**。macOS ではこれまで PyCall が独自の
   private Conda（`~/.julia/conda/...`）を選んでおり sympy が既に入っていたため
   問題が露呈していなかった。Docker のベースイメージ（`python:3.14-slim-trixie`）
   では PyCall が `/usr/local/bin/python3`（sympy 未導入）を拾ってしまい
   `ModuleNotFoundError` で `Pkg.precompile()` が失敗。`ENV PYTHON=""` を
   Julia ビルドの前に設定し、PyCall に独自 Conda ブートストラップを強制することで
   macOS の実績経路に統一した。

### 2.3 Ollama はホストネイティブ、Docker に入れない

20〜32B 級モデルは実アクセラレータ必須（macOS=Metal, Linux=CUDA）。Ollama を
コンテナに入れないことで「Docker で GPU パススルー」という難所自体を非目標化できる。
実際のロール既定（`config/default_settings.json` 実測、`docs/16_llm_model_setup.md`
と一致）：

| 役割 | モデル | 備考 |
|---|---|---|
| Supervisor/Coder | `qwen2.5-coder:32b` | 推奨・既定 |
| Supervisor | `gpt-oss:20b` | 代替 |
| OCR/VLM | `qwen3-vl:latest` | 既定 |

## 3. プラットフォーム別配布

- **macOS**（本命）：`setup.sh`（brew → 両 venv → pip → Julia instantiate → 任意
  sysimage → Ollama モデル案内）。
- **Linux**：`Dockerfile`（アプリスタックのみ、Maxima + Julia 公式バイナリ）＋
  `docker-compose.yml`（`network_mode: host` でホストの Ollama にそのまま到達）。
- **Windows（Docker Desktop）**：`docker-compose.windows.yml`（ブリッジ＋ポート
  マッピング、`host.docker.internal` 経由で Ollama）。`models/*/endpoint.json` は
  ネイティブ macOS/Linux 用に `localhost` で書かれているため、既定ロールに対応する
  4ファイルだけ `host.docker.internal` 書換え版を `models.windows/` に用意し
  ファイル単位でバインドマウント（ディレクトリ単位でなくファイル単位なので
  `system_prompt.txt` 等の兄弟ファイルはイメージのものがそのまま使われる）。
- **Windows（ネイティブ、Docker 不使用）**：`setup_windows.ps1`（`setup.sh` と対、
  §5.1）。Python・Julia・Maxima・Ollama は公式インストーラに任せ、導入確認とアプリ側
  セットアップのみを担当。

`docs/setup_guide_jp.md`/`docs/setup_guide_en.md` が利用者向けの実行手順（本文書は
設計側）。

## 4. 数値サンドボックスの OS 別実装（G-2）

`notebook/numeric/sandbox.py` の `SandboxRunner.sandbox_available()` は
`sys.platform` の静的判定ではなく、**各プラットフォームで実際にプローブして**
利用可能かどうかを決める（macOS の「本機で経験的に検証した」という既存の姿勢を
Linux 側にも徹底したもの）。

### 4.1 macOS：`sandbox-exec`（既存、変更なし）

`(deny default)` の SBPL プロファイル。詳細は
[22_numeric_layer_design.md](22_numeric_layer_design.md) §7。

### 4.2 ベアメタル Linux：`bubblewrap`（bwrap）— 本物のサンドボックス

unprivileged user+network+mount namespace で macOS と同水準の隔離を再現：

- `--unshare-net` でネットワーク遮断
- `--ro-bind / /` → 秘密ディレクトリ・`sessions/` を `--tmpfs` で隠す → 自タスクの
  work dir だけ `--bind` で再公開——macOS SBPL の allow→deny→re-allow
  （last-match-wins）と同じ構造を bind 順序のシャドーイングで再現
- `sandbox_available()` は実際に bwrap を1回起動して確認する実プローブ
  （`_linux_bwrap_probe()`）

**Ubuntu 22.04.5 実機で実測検証済み**（2026-07-20）：`SandboxRunner.run_file()` を
直接呼ぶスモークテストで、①ネットワーク遮断（`ENETUNREACH`）②work dir 内書込み成功
／外は絶対パスでも `EROFS`／他セッションディレクトリへの書込みも `ENOENT`（tmpfs
shadow で不可視化）③他セッション read 拒否（`ENOENT`）④タイムアウト強制終了
（3秒設定で3.049秒に `SIGKILL`）を確認。**追加権限（`cap_add`/setuid）は一切
不要**——`kernel.unprivileged_userns_clone=1`・`kernel.apparmor_restrict_
unprivileged_userns=0`（Ubuntu 22.04 は 24.04 の AppArmor 強化の対象外）という
条件が効いている。

### 4.3 Docker コンテナ内：bwrap 断念 → 専用低権限ユーザー + Landlock

#### bwrap がコンテナ内で失敗する（未解決の上流バグ）

Ubuntu 実機に Docker Engine（29.6.2）を導入し実ビルド・実起動（アプリ本体・CAS・LLM
は HTTP 200 で完全動作）した上で、コンテナ内で bwrap を実行すると：

1. `Operation not permitted`（unprivileged userns 制限、想定通り）→
   `cap_add: [SYS_ADMIN]` で解消
2. その先で**別の失敗** `bwrap: loopback: Failed RTM_NEWADDR: No child processes`
   （`ECHILD`、bwrap 内部のループバック設定処理——netlink ヘルパーを fork して
   `waitpid()` する処理——が失敗）

以下7つの仮説を実地で切り分けたが、いずれも原因ではなかった：

| 仮説 | 検証方法 | 結果 |
|---|---|---|
| bwrap のバージョン差 | ホストの 0.6.1 バイナリをコンテナにコピーして同じ argv で実行 | 同じ失敗 |
| SIGCHLD 継承 | `signal.signal(SIGCHLD, SIG_DFL)` で明示リセット、PID1 の `SigIgn` マスクも確認 | 再現、PID1 も無視していない |
| `docker exec` 起因 | Python `subprocess.run()` 経由で再テスト | 同一エラー |
| namespace フラグ構成 | `--unshare-pid`/`--unshare-user` の有無、外側 `unshare --net` へのネスト化 | 別の失敗モード（`Failed to make / slave: Permission denied`）に変わるだけ |
| Docker の seccomp プロファイル | `security_opt: seccomp=unconfined` | 解消せず |
| レースコンディション | 5回リトライ | 5/5 とも同じ失敗 |
| userns-remap 設定 | `docker info` で確認 | 無効、cgroup v2/systemd driver で特殊な設定なし |

`unshare --net -- true` 単体（bwrap 不使用）はカーネルレベルで成功することを確認済み
——問題は bwrap 自身の内部ロジックに限局しており、この環境固有の非互換
（`containers/bubblewrap` 側の未報告バグの可能性）と判断し、bwrap 経由は断念した。

#### 方針転換：そもそも何を守る必要があるのか（proportionality の再検討）

bwrap で試行を重ねる中で、「数値層専用の狭いサンドボックスであり、一般用途の
アドバサリアルサンドボックスを目指しているわけではない」という指摘（ユーザーとの
議論、2026-07-21）を受けて設計を見直した。数値層に実際に到達するコードは2経路
しかない：

1. **決定的テンプレート**（`notebook/numeric/expr_to_numpy.py` の `_NUMPY_FUNCS`）
   ——`sin/cos/tan/exp/log/sqrt/abs/floor/round/sign` 等、純粋な数学関数15個の
   固定辞書のみ。ホワイトリスト外の呼び出しは `UnknownNotationError` で
   fail-closed。**ネットワーク I/O・ファイルパス操作を表現する文法自体が存在しない**
   ——サンドボックスの出番以前に構造的に不可能。
2. **LLM 生成コード**（`llm_codegen.py`）——`approve_numeric_code` による
   **人間レビュー必須ゲート**を通らないと実行されない（CLAUDE.md Numeric Layer
   原則2）。

この2経路を踏まえると、bwrap や「Landlock+iptables」のような重装備の汎用
アドバサリアルサンドボックスは過剰と判断し、**OS レベルの分離は人間レビューの
見落としに対する二次的なバックストップ**という位置づけに設計を簡略化した。

#### 実装：専用低権限ユーザー + Landlock（ネットワークのみ）

- **①ネットワーク遮断：Landlock**（kernel 5.13+、ネットワークフックは ABI v4/
  kernel 6.7+）。`ctypes` で `landlock_create_ruleset`/`landlock_restrict_self`
  システムコールを直接呼び出し（`prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)` が前提
  条件、無いと `EPERM`）。`handled_access_net` に `LANDLOCK_ACCESS_NET_BIND_TCP`/
  `CONNECT_TCP` を指定し許可ルールを一切追加しない＝TCP bind/connect を全面拒否。
  **namespace を一切使わないため bwrap が壊れている箇所と無関係**——ベアメタルと
  コンテナ内の両方で追加権限ゼロで実測確認済み。
- **②書込制限：専用ユーザー + ファイルパーミッション**。Dockerfile で
  `useradd --system --no-create-home numeric` を作成。root から `numeric` への
  `setuid`/`setgid` は権限を「狭める」方向のみなので追加 capability 不要。
  タスクの work dir を `chown numeric:numeric` + `chmod 0700`。
- **③他セッション read 拒否：`sessions/` 自体を `chmod 0711`**（実行中タスクの
  祖先ディレクトリのみ、`_lock_down_task_dir()`）——non-listable にはなるが、
  下記の限界がある。

#### 正直な最終評価：「本物のサンドボックスではない」（ユーザー了承済み）

macOS/ベアメタル Linux と同水準の隔離ではないことを明記する：

| 性質 | 状態 | 根拠 |
|---|---|---|
| ①ネットワーク遮断 | **TCP のみ** | Landlock ABI v4 のネットワークフックは bind/connect TCP のみ。UDP・raw socket・AF_UNIX は対象外 |
| ②work dir 外への書込み拒否 | **堅牢** | 実測：`EACCES`/`EROFS` |
| ③他セッション・他タスクからの read 隔離 | **弱い** | `_lock_down_task_dir` は実行中タスクの祖先しかロックせず兄弟ディレクトリは無施錯のまま。`task_id` は単純連番（`N001`…、`engine_numeric_mixin.py` `_alloc_numeric_id`）、`execution_id` はタイムスタンプ文字列（`numeric.py` `_new_execution_id`、乱数 UUID ではない）なので exact path の推測が容易。数値タスク同士も同じ `numeric` UID を共有するため相互隔離もできない |

**「無いよりマシ程度の分離」として受容**（§4.3 の proportionality 論拠：構造的に
ネットワーク/パス操作ができないテンプレートと、人間レビュー必須の LLM 経路が
実質的な一次防御であるため）。

#### ディスパッチ実装

`_resolve_linux_mode()` が bwrap → uid_landlock の順に実プローブして選択・キャッシュ
（`"bwrap"` / `"uid_landlock"` / `"unavailable"`）。`run_file()`・
`engine_numeric_mixin.py` 等の呼び出し側は無改造——`_build_argv()`（コマンド前置き
方式）と `preexec_fn` 内の privilege drop + Landlock 呼び出し（uid_landlock モード）
に分岐を閉じ込めている。既存テストは元々 `SandboxRunner().available()` で
ゲートされていたため（`sys.platform` ハードコード無し）、両モードとも既存
テストファイル無改造で検証対象になる。

**Ubuntu 実機・実 API 検証（2026-07-21）**：`SandboxRunner.run_file()` を直接呼び、
`sandbox_available()`=True・選択モード=`uid_landlock` を確認。①ネットワーク拒否
（`Permission denied`）②work dir 内書込み成功／外は拒否③`sessions/` non-listable
④他セッション（ロックダウンなしの既存ディレクトリ）への read は推測パスで成功
（上表の既知の限界として記録・受容）を確認。

## 5. ネイティブ Windows（G-3）

**2026-07-21 に方針転換。** 当初「ネイティブ Windows（WSL2/Docker を使わない場合）は
スコープ外」としていたが、これは**数値層サンドボックスの Windows 実装**が対象外という
意味であり、**アプリ本体（CAS＋LLM のノートブック機能）がネイティブ Windows で動くか
どうかとは別の話**だった。アプリ本体は素の Python（FastAPI/uvicorn）なので Windows
で動かない理由が無いという指摘を受け、この2つを切り分けて設計し直した。

### 5.1 アプリのセットアップ：実装完了・実機フル検証済み（2026-07-21〜23）

**方針は `setup.sh` と同じ**：Python・Julia・Maxima・Ollama は各公式インストーラに
任せ（`winget` 等での自動導入はしない——「公式インストーラへの案内＋確認のみ」という
利用者の希望通り）、`setup_windows.ps1`（リポジトリルート、`setup.sh` と対の位置）が
①導入確認 ②アプリ側セットアップ（`.venv`/`.venv-numeric` 作成・pip install・Julia
`Pkg.instantiate()`）を行う。`scripts/start_windows.ps1` が `scripts/start.sh` に対応。

**Docker で見つけたのと同じ地雷を先回りして対策（設計段階）**：
- **PyCall/SymPy.jl の Python 解決バグ**（§2.2）対策として、Julia ビルド前に
  `$env:PYTHON = ""` を設定（Docker・macOS と同じ対策）。
- **`notebook/numeric/sandbox.py` の `DEFAULT_NUMERIC_PYTHON`** を
  `sys.platform == "win32"` 分岐で `Scripts\python.exe` レイアウトに対応。
- **PowerShell はネイティブ EXE の失敗で自動停止しない**（bash の `set -e` に相当する
  ものが無い）——全ての重要なネイティブコマンド呼び出し直後に `$LASTEXITCODE` を
  明示チェックする `Assert-LastSuccess` ヘルパーで対応。
- **Maxima の公式 Windows インストーラは PATH に自動追加しないことがある**——
  `setup_windows.ps1` の導入確認メッセージに手動追加の手順を明記。

**実機初回実行（2026-07-21/22、Z440 上の VFIO GPU パススルー Windows 11 VM）で
即クラッシュ→修正。** 設計段階の対策だけでは防げない、実機でしか踏めない罠が
最初の一歩（ツール検出）で早速露呈した：

1. **Windows PowerShell 5.1 は `$ErrorActionPreference = "Stop"` の下で、ネイティブ
   exe が stderr に何か書いただけでスクリプト全体を終了させる**——`2>$null` で
   redirect していても関係なく変換される（PS 5.1 固有の既知の癖）。
2. **`python.exe` は未インストールでも Windows の「App Execution Alias」
   （Microsoft Store 誘導スタブ、`%LOCALAPPDATA%\Microsoft\WindowsApps\` 配下）が
   PATH 上に存在する**ため、`Get-Command python` だけでは本物と区別できない。

修正：`$ErrorActionPreference` をスクリプト全体で `"Continue"` に変更（既存の
`Assert-LastSuccess`/`$LASTEXITCODE` 明示チェックのみで制御）、`Test-RealVersion`
ヘルパーを新設して `2>&1 | Out-String` でストリームを合流させた上で
`"Python \d+\.\d+"` 等の実際のバージョン文字列パターンに一致するかで真偽判定する
よう変更（Julia/Ollama のバージョン表示にも同じ合流パターンを適用）。同じく
`Confirm-Step`（オプション機能の Yes/No プロンプト）も、非対話環境（Claude Code CLI
がエージェントとして駆動する場合等）で `Read-Host` が `PSInvalidOperationException`
を投げる問題を `try/catch` で修正——元の失敗条件そのもの（Windows 側 Claude Code
CLI セッション自身の起動方式）で再現・再検証済み。

**ツール導入から `setup_windows.ps1` 完走まで（2026-07-22/23）。** `winget` 経由で
Python 3.14.6（`Python.Python.3.14`）・Julia 1.12.6（`Julialang.Juliaup`、公式推奨の
juliaup 経由）・Maxima 5.49.0（`MaximaTeam.Maxima`）・Ollama 0.32.1
（`Ollama.Ollama`）を導入。追加で発見した罠：`winget search`/`install` は非対話環境
では `--accept-source-agreements` を明示しないと `0x8a150042` で失敗する。Maxima
インストーラは PATH に自身を追加しない（事前警告どおり）ため `bin\` を手動で
ユーザー PATH に追加。`setup_windows.ps1` が `[ok] Setup complete.` まで完走
（venv 作成・pip install・Julia `Pkg.instantiate/precompile` 約40分＝292パッケージ
ビルドを含む、すべて成功、`~\.julia` にキャッシュ済みなので次回以降は高速）。

**アプリ起動で Windows 固有のブロッカーを2件発見、両方とも修正・実機で解消
確認済み：**

- **`notebook/runtime/maxima_adapter.py:96`**：公式 Maxima Windows インストーラは
  `maxima.bat` のみを提供し `.exe` が無いため、`subprocess.Popen(["maxima",...],
  shell=False)`（Win32 CreateProcess は拡張子なし名を `.exe`/`.com` としてしか
  PATH 解決せず `.bat`/`.cmd` を解決できない）が `FileNotFoundError` になる。
  `_maxima_argv()` ヘルパーを新設し、`os.name == "nt"` の明示分岐＋
  `shutil.which()` で解決先が `.bat`/`.cmd` なら `cmd.exe /c` 経由で起動するよう
  修正（macOS/Linux の `["maxima",...]` パスは無変更、`tests/test_maxima_adapter.py`
  に `TestMaximaArgvWindowsWorkaround` として5件のユニットテストを追加）。
- **`ui/app.py` の lifespan 起動（`katex_utils.ensure_katex_assets()` 経由）＋
  `cli.py`**：Windows コンソールの既定コードページ（日本語ロケールでは `cp932`）は
  UTF-8 ではないため、絵文字（`✅`/`📥`/`✗`）を含む `print()` が
  `UnicodeEncodeError` を投げ、FastAPI の lifespan 全体をクラッシュさせていた
  （下記 `ui/llm_server_launcher.py` のバグと同じ「未捕捉例外が lifespan 全体を
  落とす」パターン）。`os.name == "nt"` の時のみ `sys.stdout`/`sys.stderr` を
  `reconfigure(encoding="utf-8", errors="replace")` する2行をそれぞれのエント
  リポイント先頭に追加（macOS/Linux は元々 UTF-8 ロケールなので分岐は素通り）。
- **`ui/llm_server_launcher.py:60`**（クロスプラットフォームのバグ、Windows は
  発見場所が最初だっただけ）：Ollama 未検出時の `subprocess.Popen(["ollama",
  "serve"])` が `FileNotFoundError` を未捕捉のまま lifespan をクラッシュさせて
  いた——「Ollama 無しでもアプリは動く」という謳い文句と矛盾。`start()` を
  `try/except OSError` で囲み、失敗時はメッセージを出して `None` を返すだけに
  修正（LLM 機能のみ無効化、アプリ全体はクラッシュしない）。同ファイルの
  `stop()` の `os.killpg`/`os.getpgid`（POSIX 専用）も、Windows 版 `os` モジュール
  に属性として存在しないこと（`hasattr` で実機確認）、`start_new_session=True`
  は Windows では CPython 本体（`subprocess.py` の `_execute_child`、Windows 版は
  `unused_start_new_session` という仮引数名で明示的に無視）が静かに無視する
  仕様であることを実機で確認した上で、`os.name == "nt"` の明示分岐で
  `proc.terminate()` を使うよう修正。

**実機 end-to-end 検証（2026-07-23）：** 両修正を反映後、VM 上で
`http://127.0.0.1:8000` への HTTP リクエストが 200 を返すことを確認。さらに
`POST /api/notebook/session/new` → `POST /api/notebook/register`（`(x+1)^2`）→
`POST /api/notebook/operation`（`expand`）を実行し、`x^2+2*x+1` という正しい
展開結果が返ることを確認——`maxima_adapter.py` の修正がプロセス起動だけでなく
実際の双方向通信（stdin/stdout 経由の Maxima 計算）まで機能することの実証。

**VM への直接 SSH アクセスを確立（2026-07-23）：** VM（libvirt の NAT 網、Mac 側
からは直接ルーティングされない）に対し、ホストの Ubuntu 実機を踏み台
（`ssh -J`）にした公開鍵認証での SSH/scp アクセスを設定——以降の Windows 実機
検証（G-4 含む）はこの経路で直接ファイル転送・コマンド実行・動作確認ができる
ようになった。副次的な発見として、Windows OpenSSH Server はアカウントの
Job Object にセッション単位で子プロセスを巻き込む挙動があり、SSH コマンドが
終了すると `Start-Job`/`Start-Process` で起動した子プロセスも道連れで終了する
——長時間動かすプロセス（アプリ本体等）は Task Scheduler
（`Register-ScheduledTask`+`Start-ScheduledTask`、または `schtasks`）経由で
起動すると SSH セッションから切り離せる（G-4 の実装でも同じ手法を採用）。
また DPAPI（`ConvertTo-SecureString`/`ConvertFrom-SecureString`、G-4 で使用）は
この SSH 非対話呼び出しでは暗号化・復号とも失敗する（ユーザープロファイルが
完全にロードされないため）——Task Scheduler 経由や通常の対話ログオンでは
問題なく動作する。

### 5.2 数値層サンドボックス：実装完了（G-4、§7 参照）

当初「引き続きスコープ外」としていたが、2026-07-24 に実装・実機検証まで完遂した
（専用低権限アカウント＋ Task Scheduler ＋アカウント単位ファイアウォールルール）。
詳細は §7、特に完成形は §7.5。

### 5.3 Windows（Docker Desktop/WSL2 経由）も実機未検証——優先度は需要駆動に格下げ（2026-07-24）

`docker-compose.windows.yml`・`models.windows/` の実ビルド動作、および数値
サンドボックスの Linux フォールバック（bwrap → uid_landlock）が WSL2 のカーネル
上でも同様に動くかは、「同じ Docker/containerd 系統だから動くはず」という推測に
留まる。WSL2 のカーネルバージョンはディストリ・更新状況に依存し、検証済みの
Ubuntu 実機（6.8.0）と同一とは限らない——特に Landlock ABI v4（ネットワークフック）
は kernel 6.7+ が必要なので、古い WSL2 カーネルでは利用できない可能性がある。

**§5.1（ネイティブ Windows）・§7（G-4 数値サンドボックス）が実機検証まで完了
したため、この経路を積極的に追う理由は薄いと判断した。** Docker Desktop on
Windows はそもそも WSL2 が前提で、公式インストーラを叩くだけのネイティブ
Windows よりセットアップの手間がむしろ多い。数値サンドボックスの質も
ネイティブ側が上（G-4 はアカウント単位ファイアウォールルールで全プロトコル
遮断、この経路の Linux フォールバックは TCP のみで「無いよりマシ」程度と
§4.3 で明示的に受容した弱い分離）。Docker 運用を積極的に希望する需要が
出るまで着手しない。

詳細・未検証項目のチェックリストは `docs/open_items.md` §G-3。

## 6. 検証方法論（今後の同種作業への教訓）

- **「動くはず」で終わらせず実機で確認する**。macOS 上の推論だけでは Docker 固有の
  問題（PyCall の Python 解決・bwrap のコンテナ内非互換）は発見できなかった
  ——実際に SSH で Ubuntu 実機にツリーを転送し、都度ビルド・実行して確認した。
- **手作りスクリプトでなく実際の API を呼ぶ**。`SandboxRunner.run_file()` を
  直接呼ぶスモークテストで検証し、`bwrap`/`landlock` の生コマンドを個別に叩く
  診断とは別に、最終的には必ず実プロダクションコードパスを通して確認した。
- **一部プラットフォームの検証を全体完了として扱わない**。macOS・Linux の検証を
  もって G-1/G-2 を「解消」とし open_items の優先度表から削除した際、Windows が
  未検証のまま消えてしまう過剰な closure になった（ユーザー指摘で訂正、G-3として
  再度切り出した）。

## 7. ネイティブ Windows 数値サンドボックス（G-4・未実装・保留中）

「Windows で数値計算が一切できないままでは拙い」という指摘（2026-07-21）を受けて
設計した。**実機/VM が無いため実装は保留**だが、着手すればすぐ実装に入れるよう
ここに計画を残す。

### 7.1 見つかった2つの前提バグ（実装前に必ず直す必要がある）

設計を検討する過程で、`notebook/numeric/sandbox.py` の**現状のコードが Windows で
どう振る舞うか**を机上で追ったところ、2つの致命的な問題が見つかった：

1. **モジュールレベルの `import pwd` / `import resource` が無条件**（ファイル
   冒頭）。両方とも POSIX 専用モジュールで Windows には存在しない。
   `engine_numeric_mixin.py` は `SandboxRunner` を関数内で遅延 import している
   ため**アプリ起動時のクラッシュにはならない**が、ユーザーが数値タスクの可否
   確認や実行を初めて試みた瞬間に `ModuleNotFoundError` の素の例外になる——
   意図していた「`sandbox_available()`=False → `ValueError`
   （`numeric.sandbox_unavailable`）で明示的にブロック」という設計上のフェイル
   セーフにすら到達しない。**この2つの import を `sys.platform` で分岐させる
   だけの小さな修正だが、これをやらない限り Windows で数値層 UI に触れると
   確実に落ちる。**
2. **`run_file()` が `subprocess.run(..., preexec_fn=_limits)` を全 OS 共通で
   呼んでいるが、`preexec_fn` は POSIX 専用**——Windows に渡すと
   `ValueError: preexec_fn is not supported on Windows platforms`。Windows は
   fork ベースではなく `CreateProcess` ベースなので、POSIX 版
   （`os.setuid`+`landlock_restrict_self` を `preexec_fn` に積む）と同じ構造を
   再利用できない。Windows 分岐は `subprocess.run` を経由せず、生の
   `CreateProcessAsUser`（後述）を直接呼ぶ構造にする必要がある——「もう1つ
   `elif` を足すだけ」では済まず、`run_file()` のプロセス起動部分そのものを
   Windows 用に書き分ける設計変更が要る。

### 7.2 実装計画（当初案・実機/VM 無しの机上設計 — §7.4 の実機検証で見直し）

> **2026-07-23 追記：** 本節は実機/VM が無い段階での机上設計であり、Z440 の
> Windows 11 VM で実機検証した結果、②③の `LogonUser`+`CreateProcessAsUser`
> 案には見過ごしていた前提の欠陥が見つかった（§7.4 参照）。**本節は経緯の記録
> として残すが、実装はこの案では進めない。** ①のファイアウォール遮断案は
> そのまま有効。

（以下、当初案）

namespace 系のプリミティブ（bwrap/Landlock 相当）は Windows には無いため、
**OS のアカウント/ACL とファイアウォールという枯れた仕組み**で代替する——
「専用低権限ユーザー + 追加権限ゼロで足せるネットワーク遮断」という Docker 版の
構図をそのまま Windows のプリミティブに置き換える形。

**①ネットワーク遮断：Windows ファイアウォールの実行ファイル単位アウトバウンド
拒否ルール。**

```powershell
netsh advfirewall firewall add rule name="ResearchNotebook-NumericBlock" `
  dir=out action=block program="<repo>\.venv-numeric\Scripts\python.exe" `
  enable=yes profile=any
```

`.venv-numeric` は数値層専用インタープリタ（他の用途では使われない）なので、
この実行ファイルパスを狙い撃ちで遮断すれば足りる。**Docker の Landlock
（TCP のみ）より広い保証**——プロトコル問わず全遮断できる。`setup_windows.ps1`
が一度だけ設定（要管理者権限の一時昇格）、以降の実行時には権限昇格不要
（システム全体に効く静的ルールとして残る）。

**②③書込み・read 制限：専用ローカルアカウント（`numeric`）＋ ACL。**

`setup_windows.ps1` が `New-LocalUser` で作成（要管理者権限、ファイアウォール
設定と同じ一回きりのセットアップ）。ランダム生成パスワードは DPAPI
（`ConvertTo-SecureString`/`ConvertFrom-SecureString`——現在の Windows
アカウントでのみ復号可能、平文保存を避ける）で保存。

数値実行時（`sandbox.py` の Windows 分岐）：

1. 保存したパスワードを DPAPI で復号
2. `LogonUser()`（ctypes、`advapi32.dll`）でトークン取得
3. `icacls <workdir> /grant numeric:(OI)(CI)F /inheritance:r` で work dir の
   ACL を専用アカウントのみに絞る（Linux の `chown`/`chmod` 相当。生の
   `SetNamedSecurityInfo` WinAPI ではなく枯れた `icacls.exe` を shell out
   する方針——Landlock の時と同様、自前実装よりテスト済みツールを優先）
4. `CreateProcessAsUser()`（`kernel32.dll`）でそのアカウントとして起動。
   stdout/stderr は `CreatePipe` + `STARTUPINFO.hStdOutput`/`hStdError` で
   手動配線が必要（`subprocess.Popen` の `PIPE` とは別物、Python 側に読み出し
   コードを書く）
5. タイムアウトは `WaitForSingleObject` + 超過時 `TerminateProcess`
   （`subprocess.run(timeout=...)` は使えないので手動実装）。`Job Object` の
   `PerProcessUserTimeLimit` を併用すれば POSIX `RLIMIT_CPU` により近い CPU
   時間制限も追加できる（あれば尚良い、必須ではない）

**Docker 版と同じ限界を継承する**：数値タスクは全て同じ `numeric` アカウントを
共有するため、タスク間の相互隔離はできない（`task_id`/`execution_id` が連番・
タイムスタンプであるという同じ事実が Windows でも当てはまる）。「本物のサンド
ボックスではないが無いよりマシ」という Docker 版と同一の評価になる見込み。

### 7.3 リスク評価と進め方（当初案時点の記述）

Landlock 実装（3個のシステムコール、~40行）より遥かに大きい WinAPI 実装
（`LogonUser`・`CreateProcessAsUser`・パイプ/ハンドル管理・`Job Object`・
`icacls` 連携）で、構造体レイアウト・呼び出し規約等、実機無しでは検証できない
要素が多い。**実機/VM が用意でき次第、bwrap→Landlock 移行の時と同じ手順**
（1機能ずつ実装→実機で疑似スクリプトで検証→本実装に組込→
`SandboxRunner.run_file()` 経由で end-to-end 検証、§6 参照）で進める。
「実装したので動くはず」で終わらせず、必ず実機で確認してから完了とすること
——G-3 で得た教訓（一部プラットフォームの検証を全体完了として扱わない）を
ここでも徹底する。

上記の通り実機で疑似スクリプト検証を行った結果、②③の方式自体を見直すことに
なった。詳細は §7.4。

### 7.4 実機検証による設計見直し（2026-07-23、Z440 の Windows 11 VM で実施）

G-3 で構築した VM（実機、VFIO GPU パススルー）が使えるようになったため、
「実装したので動くはず」で終わらせず、②③（`LogonUser`+`CreateProcessAsUser`）
案そのものを本実装の前に疑似スクリプトで検証した。結果、**この案は前提から
崩れることが判明し、Task Scheduler を使う方式に転換した。**

**発見：`SeAssignPrimaryTokenPrivilege` は Administrators グループにも
デフォルトでは付与されていない。**

`CreateProcessAsUser` の呼び出しには `SE_ASSIGN_PRIMARYTOKEN_NAME` と
`SE_INCREASE_QUOTA_NAME` の2つの特権が必要（MSDN）。後者は Administrators
グループがデフォルトで保持しているが、**前者は既定では LOCAL SERVICE /
NETWORK SERVICE にしか付与されておらず、Administrators グループにすら
含まれていない**——これは UAC の昇格・非昇格とは無関係な事実で、実機で
完全に昇格した（High Mandatory Level）Administrator の PowerShell セッション
から `whoami /priv` を実行しても `SeAssignPrimaryTokenPrivilege` は一覧に
出てこないことを確認した：

```
> whoami /priv | Select-String 'SeAssignPrimaryToken|SeIncreaseQuota'
SeIncreaseQuotaPrivilege   ...   有効
```
（`SeAssignPrimaryToken` は該当行なし）

つまり②③案は、**アプリを昇格せずに動かしても・昇格して動かしても**、
事前に `secedit`（ローカルセキュリティポリシー）でこの特権をアプリの
実行アカウントに明示的に付与しない限り `ERROR_PRIVILEGE_NOT_HELD` で
確実に失敗する。当初案はこの前提の欠落に気づかないまま「一回きりの
セットアップ（アカウント作成＋ファイアウォール規則）で足りる」としていた
——本節の検証で見つかった、実装前で良かった欠陥。

**代替：Task Scheduler（`schtasks`/`Register-ScheduledTask`）は同じ制約下で
成功した。**

`SeAssignPrimaryTokenPrivilege` を持たない同じ Administrator セッションから、
`schtasks /create ... /ru <専用アカウント> /rp <パスワード>` →
`schtasks /run` で、実際に専用の低権限アカウント（テスト用に作成した
`rn_test_lowpriv`）としてコマンドが実行されることを確認した
（`whoami` の出力が `win_z440\rn_test_lowpriv` になることで確認）。
Task Scheduler サービス自体が SYSTEM 権限でアカウント切替を代行するため、
**呼び出し側（アプリ本体）の特権には依存しない。**

必要な一回きりのセットアップは以下の2つのみ（`setup_windows.ps1` が担当、
要管理者権限の一時昇格——当初案と同じ想定）：
1. `New-LocalUser` で専用アカウント作成（当初案と同じ）
2. そのアカウントに **「バッチジョブとしてログオン」権
   （`SeBatchLogonRight`）を `secedit` で付与**（新規に必要と判明した手順。
   実機で `secedit /configure /db ... /cfg ... /areas USER_RIGHTS` により
   付与に成功し、以降タスクの実行が通ることを確認済み）

どちらの手順も**専用アカウント側だけに触れ、アプリを実行するアカウント側の
権限には一切変更を加えない**——当初案が想定していた「一回きりのセットアップ
のみ、実行時は無権限で良い」という設計目標自体は達成できている。

**見直した実装計画：**

- ①ネットワーク遮断：当初案のまま（ファイアウォール実行ファイル単位
  アウトバウンド拒否ルール）変更なし。
- ②③書込み・read 制限：`LogonUser`+`CreateProcessAsUser`+手動パイプ配線を
  **`schtasks`/`Register-ScheduledTask` に置き換え**。stdout/stderr は
  ライブパイプではなく、リダイレクト先ファイルをポーリングして読む方式
  （実機検証で実際に採用した方式と同じ）。ctypes での構造体レイアウト・
  呼び出し規約のリスクが実質消える。
- プロセスツリーの強制終了：`TerminateProcess` 単体ではなく **`Job Object`
  （`CreateJobObject`+`AssignProcessToJobObject`+`TerminateJobObject`）を
  中核に格上げ**。当初案では `Job Object` は「CPU 時間制限があれば尚良い」
  という副次的な扱いだったが、Windows には POSIX のプロセスグループ
  シグナルに相当するものが無く、タイムアウト時に子プロセスごと確実に
  終了させるには `Job Object` が本筋——`llm_server_launcher.py` の
  Windows 対応（2026-07-23、`os.killpg`/`os.getpgid` が Windows の `os`
  モジュールに存在しないことを実機で確認済み）で得た知見とも整合する。
- Docker 版と同じ限界（タスク間の相互隔離不可）は変わらず継承する見込み。

当時は未着手・未実装だった——本節はあくまで「机上設計の欠陥を実装前に
実機で潰した」という設計見直しの記録。§7.5 で実装まで完遂した経緯を記す。

### 7.5 実装完了・実機フル検証（2026-07-24）

§7.4 の方針で実装を進めたところ、**Job Object を中核に据える案も実機で不十分と
判明**——Windows の Python venv の `Scripts\python.exe` は実体コピーではなく
**リダイレクタスタブ**で、実行するとベースインタープリタを別プロセスとして
`re-exec` する。結果、実プロセスツリーは `wrapper powershell.exe → venv スタブ
→ ベース python.exe` の3階層になり、`taskkill /PID <pid> /T /F`
（ラッパーの PID に対して）・`schtasks /end /tn` のどちらも孫プロセスを
取りこぼした（実機で複数回再現）。最終的に **`taskkill /F /T /FI
"USERNAME eq numeric"`**（アカウント名でのフィルタ一括終了）に落ち着いた——
数値タスクは全て `numeric` アカウント専用で動くため、ツリーの深さに関係なく
確実に掃討できる。単発の呼び出しはスケジュールタスク自身の起動遅延
（`LogonUser` 経由のプロセス起動は瞬時ではない）と競合して取りこぼすことが
あったため、猶予期間（20秒）中繰り返しスイープする方式にした。

同じ実機検証で、当初案①（ネットワーク遮断）にも修正が必要と判明した：
`.venv` と `.venv-numeric` は**同じベース Python インタープリタにリダイレクト**
するため（前述のスタブ re-exec）、実行ファイルパスを狙い撃つファイアウォール
ルールでは両者を区別できない——`.venv-numeric` を塞ぐと `.venv`（メインアプリ、
Adviser LLM の外部呼び出し等）まで巻き添えになる。**`-LocalUser`（SDDL 形式、
アカウント SID 指定）でのファイアウォールルール**に変更し、アカウント単位で
遮断することでこの問題を回避した。

**実装:**
- `notebook/numeric/sandbox.py`：`_run_file_windows()`（schtasks 起動、
  stdout/stderr はリダイレクトファイル経由でポーリング取得、`icacls` による
  書込み制限、タイムアウト時のアカウント名一括 kill スイープ）、
  `_windows_numeric_probe()`（`sandbox_available()` の Windows 分岐）、
  `_windows_decrypt_password()`（DPAPI 復号、PowerShell 経由）。§7.1 の
  前提バグ（`import pwd`/`resource` の無条件 import、`preexec_fn`）も
  `_run_file_posix`/`_run_file_windows` への分岐で解消。
- `scripts/setup_windows_numeric_sandbox.ps1`（新規）：管理者権限必須の
  一回限りセットアップ。`setup_windows.ps1` 本体（非昇格のままにする方針）
  とは意図的に分離。
- 書込み制限：`icacls` で専用アカウントに `F`、呼び出し元アカウント
  （`$env:USERNAME` で動的取得）に `(RX)`（読み取り専用——実行中のサンドボックス
  への改竄を防ぐ）、`SYSTEM` に `F`（Task Scheduler 自体の動作に必要）。

**実機 end-to-end 検証**（Z440 の Windows 11 VM、Task Scheduler 経由——後述の
理由で SSH 非対話呼び出しは使わない）：`SandboxRunner().available()` → `True`、
実コード実行で `whoami` が `numeric` を返す（アカウント分離）・外部接続の試行が
`PermissionError` で遮断（ネットワーク遮断）・work dir 外への書込みが拒否・
work dir 内への書込みは成功・`created_files` が内部管理ファイルを含まず正しく
検出、タイムアウトシナリオ（無限ループ、`timeout_s=5`）で `TIMED_OUT: True`・
約30秒後に完全終了・**孤立プロセスなし**を確認。

**副次的な発見：** DPAPI（`ConvertTo-SecureString`/`ConvertFrom-SecureString`）
は SSH の非対話呼び出し（`powershell -Command`/`-File`）ではユーザープロファイルが
完全にロードされないため暗号化・復号とも失敗する（実機で確認、暗号化側ですら
「アクセスが拒否されました」）。Task Scheduler 経由や通常の対話ログオンでは
問題なく動作する——アプリは常に `start_windows.ps1` 経由の対話起動か Task
Scheduler 経由で動かすため、実運用上の懸念ではない。

詳細は `docs/open_items.md` §G-4。

**関連:** [22_numeric_layer_design.md](22_numeric_layer_design.md) §7（数値層全体の
アーキテクチャ内での sandbox の位置づけ）、`docs/open_items.md` §G（進行中の
ステータス管理）、`docs/setup_guide_jp.md`/`docs/setup_guide_en.md`（利用者向け
セットアップ手順）。
