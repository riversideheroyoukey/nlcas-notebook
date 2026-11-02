# シンボル定義域（Domain）設計仕様

## 1. 背景と問題

### 1.1 発端：BCS 超伝導計算での ΔStar 再出現

BCS ハミルトニアンの Bogoliubov 変換計算において以下の問題が発生した：

1. E1 に `Δ* → Δ`（DeltaStar → Delta）を代入して E2 を得る
2. E2 に Nambu 変換（演算子代入）を適用すると E3 に `Δ*` が再出現する

**原因**：Julia の Symbolics.jl で `@variables Delta` を宣言すると、Delta はデフォルトで複素数（`Complex` 型）として扱われる。QuantumAlgebra.jl が随伴演算（adjoint）を計算する際、複素スカラー係数 `Delta` の随伴は `conj(Delta) ≠ Delta` として `DeltaStar`（`conj(Delta)` の別名）を生成する。

**根本解決**：`@variables Delta::Real` と宣言すれば、Symbolics.jl は `conj(Delta) = Delta` を自動適用し、`DeltaStar` が一切生成されない。

### 1.2 一般化

同種の問題は BCS 以外でも発生する：
- 任意の量子多体系で実数パラメータが存在する場合
- Maxima でも `assume(x, real)` がないと複素数として扱われることがある
- 物理で頻出の「整数量子数 n」「正の分散 ω_k」などの制約も CAS に伝えられると有用

---

## 2. 設計方針

### 2.1 基本原則

- シンボルの定義域（数の分類）は **定義時に一度だけ指定** し、セッション全体で有効
- CAS バックエンドが各定義域に対応する型宣言・仮定宣言を自動生成する
- どの定義域が物理的に意味を持つかの判断は **利用者または Advisor LLM に委ねる**
- 指定なし（デフォルト）は複素数（現状維持）

### 2.2 定義域の分類

| domain 値 | 数学的意味 | Julia (Symbolics.jl) | Maxima |
|-----------|-----------|---------------------|--------|
| `"complex"` | 複素数 ℂ（デフォルト） | `@variables x` | （なし） |
| `"real"` | 実数 ℝ | `@variables x::Real` | `assume(x, real)` |
| `"positive"` | 正の実数 ℝ₊ | `@variables x::Real` ※1 | `assume(x > 0)` |
| `"negative"` | 負の実数 ℝ₋ | `@variables x::Real` ※1 | `assume(x < 0)` |
| `"nonneg"` | 非負実数 ℝ≥0 | `@variables x::Real` ※1 | `assume(x >= 0)` |
| `"nonpos"` | 非正実数 ℝ≤0 | `@variables x::Real` ※1 | `assume(x <= 0)` |
| `"integer"` | 整数 ℤ | `@variables x::Integer` | `declare(x, integer)` |
| `"natural"` | 自然数 ℕ（正整数） | `@variables x::Integer` ※2 | `declare(x, integer)` + `assume(x > 0)` |

※1 Julia の型システムには Positive 型がないため、`::Real` で宣言する。正値制約は Advisor LLM が示唆。  
※2 Julia に Natural 型はなく `::Integer` で代替。正値制約は同上。

---

## 3. データモデル変更

### 3.1 `models/quantity.py`

`Quantity` に `domain` フィールドを追加：

```python
class Quantity(BaseModel):
    ...
    domain: Optional[str] = None
    # Allowed: "real" | "complex" | "positive" | "negative"
    #          | "nonneg" | "nonpos" | "integer" | "natural"
```

`MathType.field`（`"real" | "complex"`）との関係：
- `domain` が設定されている場合は `domain` が優先
- `domain in ("positive", "negative", "nonneg", "nonpos", "integer", "natural")` → 暗黙的に real
- 後方互換性のため `MathType.field` は保持

### 3.2 `schemas/common_operations.schema.json`

`define_scalar_parameter` の `optional_fields` に `"domain"` を追加。

---

## 4. バックエンド実装

### 4.1 Julia バックエンド（Phase 1 実装対象）

`julia_backend_handler.py` の `build_command` で `define_scalar_parameter` を明示的にインターセプトし、domain に応じた `@variables` 宣言を生成：

```python
_DOMAIN_TO_JULIA_TYPE = {
    "real":     "Real",
    "positive": "Real",   # Julia に Positive 型はない
    "negative": "Real",
    "nonneg":   "Real",
    "nonpos":   "Real",
    "integer":  "Integer",
    "natural":  "Integer",
    # "complex" → アノテーションなし（デフォルト）
}
```

生成コマンド例：
- `define_scalar_parameter("Delta", domain="real")` → `@variables Delta::Real`
- `define_scalar_parameter("n", domain="integer")` → `@variables n::Integer`
- `define_scalar_parameter("xi", domain="positive")` → `@variables xi::Real`

また `_julia_driver.jl` に domain レジストリを追加し、`_nb_autodeclare` が同名変数の自動宣言時に正しい型を使えるようにする。

### 4.2 Maxima バックエンド（Phase 2 対象）

`define_scalar_parameter` 時に `assume()` / `declare()` を emit する。

### 4.3 UI（Phase 2 対象）

シンボル定義モーダルに domain ドロップダウンを追加。

---

## 5. 自然言語認識（LLM Supervisor）

### Phase 1 で対応するパターン

`define_scalar_parameter` の解釈時に `domain` を抽出する：

| 入力パターン | domain |
|------------|--------|
| `実数`, `実数として`, `real`, `real scalar` | `"real"` |
| `正の実数`, `正実数`, `positive real`, `positive` | `"positive"` |
| `負の実数`, `negative real` | `"negative"` |
| `非負`, `nonneg`, `non-negative` | `"nonneg"` |
| `整数`, `integer` | `"integer"` |
| `自然数`, `natural number` | `"natural"` |

実装：`llm_supervisor.py` の `define_scalar_parameter` 解釈パスにパターンマッチを追加。

---

## 6. 実装フェーズ

### Phase 1（現在実装）
- [x] `Quantity.domain` フィールド追加
- [x] スキーマに `domain` optional field 追加
- [x] Julia バックエンド：`@variables Name::Type` の emit
- [x] Julia ドライバ：domain レジストリ + `_nb_autodeclare` 更新
- [x] Engine：`domain` を `register_quantity` に伝播
- [x] LLM Supervisor：自然言語からの domain 抽出
- [x] i18n：domain 表示文字列

### Phase 2（将来）
- [ ] Maxima バックエンド：`assume()` / `declare()` の emit
- [ ] UI：シンボル定義モーダルに domain ドロップダウン
- [ ] `MathType.field` との統合・整理

### Phase 3（将来）
- [ ] CAS 計算結果の domain 継承（e.g. `real + real = real`）
- [ ] Advisor LLM による domain 由来の物理的示唆（「正値制約があるため sqrt が簡単化されます」）

---

## 7. 考慮事項・制限

### 7.1 Julia の制約
- `Positive` 型は Julia 標準にないため正値制約は `::Real` + Advisor 示唆で対処
- `@variables n::Integer` を宣言しても、QuantumAlgebra の Param としては整数制約が CAS 計算に反映されない（Symbolics の挙動次第）

### 7.2 後方互換性
- `domain` は Optional（デフォルト None）なので既存セッションの読み込みに影響なし
- `domain=None` の場合は現状と同じ `@variables Name`（複素数デフォルト）

### 7.3 変更前のセッション
- 旧セッション（domain 指定なし）で定義された `Delta` は依然として複素数
- 再定義（`Deltaを実数スカラーとして定義`）することで domain を付与可能

---

## 8. 添字付きシンボル宣言時の添字分類ルール

`xi_k をスカラーとする` のように添字付きで宣言した場合、添字が「成分インデックス」（strip して base name を使う）なのか「名前の一部」（full name をそのまま使う）なのかを自動判定する。

### 分類ルール

| 添字パターン | 例 | 判定 | 挙動 |
|---|---|---|---|
| 数字のみ | `omega_0`, `Delta_1` | 名前の一部 | `omega_0` をそのまま登録 |
| 2文字以上のアルファベット | `H_BCS`, `a_ks`, `k_B`は2文字→keep | 名前の一部 | `H_BCS` をそのまま登録 |
| 複数添字（`_` または `,` を含む） | `a_k_s`, `a_{k,s}` | 成分インデックス | `a` に strip して登録 |
| 1文字アルファベット | `xi_k`, `v_i` | 曖昧 | 確認ダイアログを表示 |

### 確認ダイアログ（1文字アルファベット添字の場合）

- 選択肢A: 「k は成分インデックス（xi の k 成分）」→ base name `xi` を登録
- 選択肢B: 「k は名前の一部（xi_k という固有の量）」→ `xi_k` を登録

### 設計根拠

- `omega_0` = omega-naught（基準周波数）— `_0` は名前の一部
- `H_BCS` = BCS ハミルトニアン — 複数文字ラベルは固有名
- `a_k_s` = 運動量 k・スピン s の演算子 — 複数添字は常に成分
- `xi_k` = `xi` の k 成分にも `xi_k` という固有量にもなりえる — 確認が必要

### 実装箇所

`notebook/runtime/backend_handler.py` の `BackendHandler.extra_pre_phase1` 内
`_subscript_role(name)` ヘルパー関数。
