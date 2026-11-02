# LLM モデル設定ガイド

本システムの LLM は `models/` ディレクトリで管理します。
新しいモデルの追加、既存モデルの切り替え、ロールの割り当てをここで説明します。

---

## models/ ディレクトリ構成

```
models/
├── default/
│   └── config.json              # グローバルルーティング設定
├── ollama_qwen25_coder_32b/     # supervisor（現在の推奨）
│   └── endpoint.json
├── ollama_gpt_oss_20b/          # supervisor（代替）
│   └── endpoint.json
├── ollama_qwen3_vl/             # OCR + supervisor
│   └── endpoint.json
├── cloud_openai_gpt54/          # adviser 用（API キー要）
│   └── endpoint.json
├── lmstudio_gpt_oss_20b/        # LM Studio 経由の代替
│   └── endpoint.json
├── engines/                     # OCR 特化エンジン
│   ├── chandra-2/
│   ├── glm-ocr/
│   └── ollama_qwen3_vl/         # qwen3-vl:latest engine
├── refiners/                    # OCR 後処理
│   ├── llama-3.2-vision-11b/
│   ├── ollama_qwen3_vl/         # DEPRECATED (thinking mode 問題により engine へ移動)
│   └── ollama_internvl35/       # blaifa/internvl3.5:8b refiner
└── domains/                     # 研究分野別プロンプトヒント
    ├── categories.json          # カテゴリ定義
    └── <domain_name>/           # 各分野ディレクトリ
        ├── meta.json
        ├── supervisor_hint.txt
        └── ocr_hint.txt
```

ディレクトリ名が `endpoint.json` を含む場合、自動でエンドポイント候補として認識されます。

---

## グローバルルーティング設定（`models/default/config.json`）

```json
{
  "supervisor_scope": "local",
  "adviser_scope": "local",
  "preferred_supervisor": "ollama_qwen25_coder_32b",
  "preferred_adviser": "lmstudio_remote"
}
```

| フィールド | 説明 |
|---|---|
| `supervisor_scope` | supervisor を探すスコープ（`local` / `cloud`） |
| `adviser_scope` | adviser を探すスコープ（`local` / `cloud`） |
| `preferred_supervisor` | 優先して使う supervisor の `name`（endpoint.json の `name` フィールドと一致） |
| `preferred_adviser` | 考察モーダルのドロップダウンで先頭に表示する adviser の `name` |

adviser は `advisor` ロールを持つ全エンドポイントが自動列挙されます。`preferred_adviser` はドロップダウンの並び順とデフォルト選択のみに影響します。

---

## endpoint.json の書き方

### ローカル Ollama モデル

```json
{
  "name": "ollama_my_model",
  "scope": "local",
  "base_url": "http://localhost:11434/v1",
  "model": "my-model:latest",
  "api_key_env": null,
  "timeout": 120,
  "roles": {
    "supervisor": {
      "temperature": 0.1,
      "max_tokens": 2048,
      "num_ctx": 32768
    },
    "advisor": { "temperature": 0.2 }
  }
}
```

`num_ctx` は Ollama 固有のコンテキスト長設定です。`options.num_ctx` として Ollama API に渡されます。省略した場合は Ollama のデフォルト値が使用されます。

### LM Studio モデル

```json
{
  "name": "lmstudio_my_model",
  "scope": "local",
  "base_url": "http://localhost:1234/v1",
  "model": "openai/my-model",
  "api_key_env": null,
  "timeout": 120,
  "roles": {
    "supervisor": { "temperature": 0.0 }
  }
}
```

### クラウド API モデル（OpenAI 互換）

```json
{
  "name": "cloud_my_model",
  "scope": "cloud",
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4o",
  "api_key_env": "OPENAI_API_KEY",
  "timeout": 60,
  "roles": {
    "advisor": { "temperature": 0.3 }
  }
}
```

### フィールド一覧

| フィールド | 必須 | 説明 |
|---|---|---|
| `name` | ✅ | ユニークな識別子。`config.json` の `preferred_supervisor` と一致させる |
| `scope` | ✅ | `local`（プライベート）または `cloud`（外部 API） |
| `base_url` | ✅ | OpenAI 互換 API のエンドポイント URL |
| `model` | ✅ | モデル ID（プロバイダの表記に従う） |
| `api_key_env` | — | API キーを格納する環境変数名。不要なら `null` |
| `timeout` | — | タイムアウト秒数（既定 60） |
| `roles` | ✅ | このエンドポイントに割り当てるロールと設定 |

---

## ロール一覧

| ロール名 | 用途 | 推奨 temperature |
|---|---|---|
| `supervisor` | 自然言語 → operation 変換、曖昧性解消 | 0.0 |
| `advisor` | CAS 結果への物理的考察・次手提案 | 0.2–0.3 |
| `vision_math_engine` | 手書き画像の数式認識（OCR） | 0.0 |
| `command_parser` | 操作コマンドの抽出 | 0.0 |
| `expression_extractor` | 数式の抽出 | 0.0 |
| `coder` | 数値計算層のコード生成（バイブコーディング経路、docs/22 §15）。`max_tokens` は 4096 推奨（スクリプトはプラン候補 JSON より長い） | 0.2 |

1 つのエンドポイントが複数のロールを担当できます。

**coder ロールの解決順**（`ui/notebook_bridge.py::_resolve_coder_chat`）:
API の明示指定（`POST /numeric/generate_llm/{id}` の body `{"coder": name}`）＞
`config/default_settings.json` の `coder_target`（既定: `ollama_qwen25_coder_32b`）＞
coder ロールを持つエンドポイントの先頭 ＞ advisor チェーンへフォールバック。
選択可能な一覧は `GET /numeric/coders`（将来の UI は数値層に入った時にここから
ドロップダウン表示する想定）。現在 coder ロールは `ollama_qwen25_coder_32b`
（Qwen2.5-Coder 32B・既定）と `lmstudio_remote_gpt-oss-120b` の2つに登録済み。

---

## OCR refiner の config.json

refiner は `models/refiners/<name>/config.json` で設定します。

```json
{
  "id": "my-refiner",
  "type": "refiner",
  "vision_input": true,
  "api_config": {
    "model_name": "my-model:latest",
    "base_url": "http://127.0.0.1:11434",
    "endpoint_type": "ollama",
    "timeout": 120
  },
  "parameters": {
    "temperature": 0.1,
    "num_predict": 1024,
    "num_ctx": 8192
  },
  "paths": {
    "prompt_file": "system_prompt.txt"
  }
}
```

### `vision_input` フラグ

| 値 | 動作 |
|---|---|
| `true` | OCR engine が出力した draft LaTeX と**元の手書き画像の両方**を refiner に渡す。VLM (Vision Language Model) を使う場合はこちら |
| `false` または省略 | draft LaTeX のテキストのみを渡す（テキスト専用モデル向け） |

`vision_input: true` の refiner は画像を直接参照して誤認識を修正できるため、VLM refiner では必ず `true` に設定してください。

---

## 新しいモデルを追加する手順

1. `models/` 以下に新しいディレクトリを作成する

```bash
mkdir models/ollama_my_new_model
```

2. `endpoint.json` を作成する（上記テンプレートを参考に）

3. supervisor として使う場合は `models/default/config.json` の `preferred_supervisor` を更新する

```json
{
  "preferred_supervisor": "ollama_my_new_model"
}
```

4. サーバーを再起動する（設定はプロセス起動時に読み込まれます）

---

## ローカル推論サーバーの管理（プロバイダ切り替え可能）

アプリは起動時に**ローカルの OpenAI 互換推論サーバー**を自動起動します。既定は
Ollama ですが、**Ollama / vLLM / LocalAI などへの切り替えは設定ファイルの編集だけ**で済みます
（クライアント側は `models/*/endpoint.json` の `base_url` + `endpoint_type` で既に
プロバイダ非依存）。

### 起動（自動）

`python -m uvicorn ui.app:app` 起動時に、`ui/app.py` の lifespan が
`ui/llm_server_launcher.py` を通じてサーバーを起動します（出力は抑制）。設定は
`config/default_settings.json` の `local_llm_server`:

```json
"local_llm_server": {
  "auto_start": true,
  "command": ["ollama", "serve"],
  "port": 11434,
  "env": { "OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_MAX_LOADED_MODELS": "1" },
  "startup_wait_seconds": 3
}
```

- `OLLAMA_KEEP_ALIVE=-1`: モデルを常駐させ、呼び出しごとの再ロード（と大量の読み込みログ）を防ぐ。
- `OLLAMA_MAX_LOADED_MODELS=1`: 同時ロードを 1 つに制限しメモリ不足を回避（**Ollama では必須級**）。
- ポートが既に開いている場合は何も起動しない（外部で起動済みの Ollama には手を出さない）。
- `auto_start: false` にするとアプリは何も起動せず、サーバーはユーザー管理（systemd 等）になる。

### 別プロバイダへの切り替え例

**vLLM:**
```json
"local_llm_server": {
  "auto_start": true,
  "command": ["vllm", "serve", "Qwen/Qwen2.5-Coder-32B-Instruct", "--port", "8001"],
  "port": 8001, "env": {}, "startup_wait_seconds": 60
}
```
あわせて `models/ollama_qwen25_coder_32b/endpoint.json`（または使用中のモデル）の
`base_url` を `http://localhost:8001/v1` に変更する。

**LocalAI（システムサービスとして自前管理）:** `"auto_start": false` にし、`base_url` を
LocalAI のエンドポイントに向ける。

### 手動起動（Ollama を自前で動かす場合）

```bash
OLLAMA_KEEP_ALIVE=-1 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

この場合 `local_llm_server.auto_start` を `false` にしておくとよい。

### モデルの確認

```bash
ollama list
```

### モデルのダウンロード

```bash
ollama pull qwen2.5-coder:32b
ollama pull gpt-oss:20b
ollama pull qwen3-vl:latest
ollama pull blaifa/internvl3.5:8b   # refiner 候補（4b も選択可）
```

---

## 現在の稼働構成

| 用途 | エンドポイント | モデル | 状態 |
|---|---|---|---|
| supervisor（推奨） | `ollama_qwen25_coder_32b` | qwen2.5-coder:32b | ✅ 稼働中 |
| supervisor（代替） | `ollama_gpt_oss_20b` | gpt-oss:20b | ✅ 稼働中 |
| OCR + supervisor（代替） | `ollama_qwen3_vl` | qwen3-vl:latest | ✅ 稼働中 |
| OCR engine（代替） | `ollama_qwen3_vl`（engines） | qwen3-vl:latest | ✅ 稼働中 |
| OCR refiner | `ollama_internvl35`（refiners） | blaifa/internvl3.5:8b | ✅ 稼働中 |
| adviser（LM Studio リモート・推奨） | `lmstudio_remote` | openai/gpt-oss-120b | ✅ 稼働中 |
| adviser（クラウド） | `cloud_openai_gpt54` | gpt-4o | 🔑 API キー要 |
| adviser / supervisor（LM Studio 代替） | `lmstudio_gpt_oss_20b` | openai/gpt-oss-20b | ⏸ 予備 |

---

## トラブルシューティング

### Ollama が起動しない

```bash
# プロセス確認
pgrep ollama

# ポート確認
lsof -i :11434

# ログ確認
OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

### supervisor がルールベースのみで動作する（LLM fallback が効かない）

Ollama が起動していない場合、Phase 1（rule alias scan）のみで動作します。
「展開して」「微分して」など明確な指示は動作しますが、
「整理して」など曖昧な指示は解釈できずエラーになる場合があります。

### reasoning モデルの注意

`gpt-oss:20b` などの reasoning モデルは内部で思考トークンを消費します。
`max_tokens` を小さく設定すると内容が空になることがあります（既定設定は問題なし）。

### モデルが認識されない

`models/default/config.json` の `preferred_supervisor` フィールドが
対象 endpoint.json の `name` フィールドと完全一致しているか確認してください。

---

## 研究分野ヒント（Domain Hints）

セッション開始時に研究分野を指定すると、対応するプロンプトヒントが Supervisor と OCR エンジンに自動的に注入されます。

### 分野ディレクトリ構成

```
domains/
├── categories.json          # カテゴリ定義（表示順・日本語名）
└── <domain_name>/
    ├── meta.json            # {"label": "...", "label_en": "...", "category": "..."}
    ├── supervisor_hint.txt  # LLM Supervisor への追加コンテキスト
    └── ocr_hint.txt         # OCR エンジンへの追加コンテキスト
```

### 利用可能な分野（29 分野）

| カテゴリ | default backend | 分野 |
|---|---|---|
| 物理 | Maxima | 解析力学, 電磁気学, 量子力学, 量子化学, 統計力学, 相対論, 流体力学 |
| 数学 | Maxima | 線形代数, 微分幾何学, 複素解析, 確率統計, 微分方程式 |
| 工学 | Maxima | 信号処理, 制御理論, 情報理論 |
| 経済 | Maxima | 数理経済学, 計量経済学, 数理ファイナンス, ゲーム理論 |
| 生物 | Maxima | 数理生物学 |
| **第２量子化** | **Julia** | 凝縮系の物理, 量子光学, 場の量子論, 核物理, 量子化学（第２量子化）, スピン系 |
| 教育 | Maxima | 中学数学, 高校数学, 中学理科, 高校物理 |

第２量子化カテゴリのドメインを一つでも選択すると、セッション開始時に自動で Julia バックエンド（`second_quantization` モード）が選択されます。複数カテゴリにまたがった選択でも同様です。

### 新しい分野を追加する

1. `domains/` 以下に新しいディレクトリを作成する

```bash
mkdir domains/my_domain
```

2. `meta.json` を作成する

```json
{
  "label": "分野名（日本語）",
  "label_en": "Domain Name",
  "category": "physics"
}
```

`category` は `categories.json` に定義されたキーのいずれかを指定してください（`physics` / `mathematics` / `engineering` / `economics` / `biology` / `second_quantization` / `education`）。

`second_quantization` カテゴリに追加した分野は自動的に Julia バックエンドが選択されます。他のカテゴリは Maxima バックエンドになります。

3. プロンプトヒントファイルを作成する

```
domains/my_domain/supervisor_hint.txt
domains/my_domain/ocr_hint.txt
```

どちらのファイルも省略可能です。存在しない場合はその分野のヒントはスキップされます。

4. サーバーを再起動すると `GET /api/notebook/domains` で新しい分野が返されます。
