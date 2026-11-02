# 代数インデックスアーキテクチャ設計書

**作成日:** 2026-06-06  
**経緯:** A-1「代数インデックスのサポート」の根本解決に向けた設計検討

---

## 1. 問題の本質

### QuantumAlgebra (QA) の制約

```julia
QuIndex("k+q")
# → ArgumentError: index names must be single character, got k+q.
```

QA の `QuIndex` はアーキテクチャレベルで**単一文字インデックスのみ**を受け入れる（バージョン 1.6.0 確認済み）。

これにより `-k`、`k+q`、`a*k+b` を含む運動量空間の演算子は QA では直接扱えない。

### 現行の暫定対処「大文字規則」

```
-k  →  QuIndex("K")  （大文字 K で代理）
```

**問題点：**
- QA は `K` と `k` を別のシンボリック変数として扱う
- `δ(k, K)` が `δ(k, -k) = 0` に評価されない
- `k+q` には適用不可（2文字以上のインデックス名が使えない）

---

## 2. 現在の SymPy ハイブリッド実装（Phase 1〜3 完了）

`_has_algebraic_indices()` が代数インデックスを検出し、自動的に SymPy ルーティングを行う。

| Phase | 関数 | 内容 | 状態 |
|---|---|---|---|
| 1 | `_nb_sympy_fermion_comm` | フラット積 `c†(k+q)c(k)` の交換子 | ✅ |
| 2 | `_nb_sympy_sum_fermion_comm` | IndexSum ラップ済み H の交換子 | ✅ |
| 3 | 自動ルーティング | `generate(Commutator)` → `_has_algebraic_indices` → SymPy | ✅ |

**透過的な操作（全 Step 完了 2026-06-06）：**

| Phase | 操作 | 実装関数 | 状態 |
|---|---|---|---|
| 1-3 | `commutator`（交換子） | `_nb_sympy_fermion_comm`・`_nb_sympy_sum_fermion_comm` | ✅ |
| Step 1 | `normal_order`（正規順序） | `_nb_sympy_normal_order` | ✅ |
| Step 2 | `wick_contract`（Wick 収縮） | `_nb_sympy_wick` | ✅ |
| Step 3 | チェーンルーティング | `_get_sympy_canonical` + `sympy_canonical` フィールド | ✅ |
| Step 4 | `commutator`（混合 QA+SymPy） | `_nb_sympy_general_comm` 派生則 | ✅ |
| Step 5 | `vacuum_expectation` | `_nb_sympy_vacuum_expectation` | ✅ |
| Step 6 | `substitute`（代入） | `_nb_sympy_substitute` | ✅ |
| A-11 | `collect_by_operators`（演算子因数分解） | `_nb_sympy_collect_by_operators` | ✅ |
| A-11 | `combine_expressions`（SymPy+SymPy） | `_maybe_sympy_combine` | ✅ |
| A-11b | `relabel_index` | `_maybe_sympy_substitute` へ委譲 | ✅ |
| A-10 | `re_evaluate_expr`（シンボル一括代入） | `_re_evaluate_sympy` | ✅ |

---

## 3. 3バックエンドアーキテクチャ（提案）

### 役割分担

```
┌─────────────────────────────────────────────────────────────┐
│  Maxima                                                       │
│  └── スカラー解析（微積分・ODE・ベクトル解析・古典力学・電磁気学）│
│  └── 演算子代数なし                                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  QuantumAlgebra (QA)                                          │
│  └── 有限サイズ実空間系（整数サイトインデックス: 1,2,3,...）   │
│      - Hubbard 小クラスター（2〜8サイト）                     │
│      - Jaynes-Cummings、Cavity QED（単一・少数モード）        │
│      - Heisenberg 鎖（小サイズ）、Ising モデル                │
│      - 多軌道モデル（Kanamori等）小クラスター                 │
│  └── 単文字インデックスの形式的操作（k の形式計算）            │
│  └── 特長：高速、LaTeX 出力が綺麗、正規順序が正確             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  SymPy (sympy.physics.secondquant ベース)                     │
│  └── k空間・運動量空間の量子多体系                            │
│      - BCS 超伝導（-k Cooper ペア）                           │
│      - Bogoliubov 変換（k+q → 準粒子）                       │
│      - Fermi 液体（k+q 散乱）                                 │
│      - 熱力学極限の計算                                       │
│  └── 代数インデックスを含む全操作                             │
│  └── QA 結果の後続計算（SymPy チェーン）                      │
└─────────────────────────────────────────────────────────────┘
```

### 自動判定ロジック（完成後）

```
ユーザー入力式の IR に対して _has_algebraic_indices() を実行
  
  True（k+q, -k, a*k+b が含まれる）
    → 全操作を SymPy にルーティング
    → 結果は LaTeX で表示（OpaqueExpr）
    → 後続計算は SymPy チェーンで継続
  
  False（整数サイトまたは単文字インデックス）
    → QA にルーティング（現在と同じ）
    → 結果は QuExpr として登録
    → QA 操作を継続可能
```

**ユーザーは QA/SymPy を意識する必要がない。**

### 確定設計原則：QA → SymPy は一方通行（2026-06-06）

第２量子化セッションは QA で始まる。代数インデックス（`-k`, `k+q` 等）が
出現した時点で SymPy に移行し、**以後はすべての操作を SymPy で継続する。**
SymPy → QA への逆行は設計しない。

**理由：**
- k 空間（BCS・Bogoliubov 等）と実空間（Hubbard 有限格子）は
  物理的に別レジームであり、代入で相互変換はできない
- SymPy 結果（`_nb_E2 = Tuple{String,String}`）は QA の QuExpr ではない
- `c(-1)`（サイト -1）は有限格子 QA モデルで意味を持たない
- QA への逆変換（`k` → `:k`, `-k` → K 規則）は非自明かつ利益がない

**未対応操作の対処方針：**
QA に戻すのではなく、SymPy チェーンを拡張して SymPy 内で完結させる。
（例：`substitute k→p` → SymPy `subs(Symbol("k"), Symbol("p"))`）

---

## 4. チェーン問題と解決策

### 問題

SymPy の計算結果は現在 `OpaqueExpr`（文字列 canonical）として登録される：

```
[H_BCS, c(k)] → "-xi*c(k) - Delta*c†(-k)"
```

この結果に対してさらに `normal_order` や `wick_contract` を呼ぶと：
- QA には `-k` を含む式を渡せない
- SymPy には「QA 型でない生文字列」を渡すパスがない

### 解決策：`Expression.sympy_canonical` フィールド

**❌ 採用しない案：`sympy::` プレフィックスを canonical に埋め込む**

`canonical` フィールドは engine.py・ui/ 全体で 60 箇所以上参照されており：
- `SharedParser().parse(canonical)` が `"sympy::-xi*c(k)..."` でエラー
- `_auto_detect_substitutions` が `"sympy"` を symbol 名として誤検出
- session save/load 時の副作用

副作用のリスクが大きすぎる。

**✅ 採用する案：専用フィールドを Expression モデルに追加**

```python
# notebook/models/expression.py
class Expression(BaseModel):
    ...
    canonical: str           # 変更しない（常に人間が読める表示用文字列）
    
    # 追加フィールド
    sympy_canonical: str | None = None
    # None でなければ、この式は SymPy 計算結果
    # 値: SymPy 評価可能な文字列 e.g. "-xi*Symbol('k') * AnnihilateFermion(k)"
    # canonical は表示用のまま（変更しない）
```

**後続操作でのチェーン処理:**

```python
# julia_backend_handler.py の build_command
def _should_use_sympy_chain(self, expr_ref: str, expr_lookup: dict) -> bool:
    """SymPy 結果かどうかを判定"""
    expr = self.engine.expressions.get(expr_ref)
    return expr is not None and expr.sympy_canonical is not None

# SymPy チェーンパス
if self._should_use_sympy_chain(expr_ref, expr_lookup):
    sympy_str = expr.sympy_canonical
    return f'_nb_sympy_normal_order("{sympy_str}", "{stats}", "{vars_str}")'
```

**この設計の利点:**
- `canonical` を読む既存コードは一切壊れない
- `sympy_canonical is None` でない式は SymPy 継続可能と判定
- セッション保存・復元も JSON に自然に乗る（Pydantic が処理）
- 後から `sympy_canonical` を削除しても `canonical` は残る

---

## 5. 実装状況（全 Step 完了 2026-06-06）

### Step 1: `_nb_sympy_normal_order` ✅ 実装済み

**Julia 実装（`_julia_driver.jl`）:**

```julia
# _nb_sympy_normal_order(expr_str, stats, vars_str)
# stats: "fermion" or "boson"
# 返り値: (value_str, latex_str)
function _nb_sympy_normal_order(expr_str::String, stats::String, vars_str::String)
    sp = SymPy.PyCall.pyimport("sympy")
    sq = SymPy.PyCall.pyimport("sympy.physics.secondquant")
    
    vars = [sp.Symbol(v) for v in split(vars_str, ",") if !isempty(v)]
    ctx = _build_sympy_context(vars, stats)
    
    # expr_str を SymPy 式に変換
    expr_sym = SymPy.PyCall.pyeval(expr_str, ctx)
    
    # NO() で正規順序化
    result = sq.NO(expr_sym)
    result_expanded = sp.expand(result)
    
    value_str = string(result_expanded)
    latex_str = sp.latex(result_expanded)
    return (value_str, latex_str)
end
```

**Python 側の自動ルーティング（`julia_code_generator.py`）:**

```python
# build_command("normal_order", ...) の中で
if self._has_algebraic_indices(ir_expr):
    return f'_nb_sympy_normal_order("{expr_str}", "{stats}", "{vars_str}")'
else:
    return f'_nb_eval_site_deltas(normal_form({qa_expr}))'
```

---

### Step 2: SymPy チェーンパス ✅ 実装済み

**`notebook/models/expression.py` に `sympy_canonical` フィールド追加:**

```python
class Expression(BaseModel):
    ...
    sympy_canonical: str | None = None
    # SymPy 計算結果の場合にセット。canonical は表示用のまま変更しない。
```

**SymPy 結果の登録（`engine.py` 結果処理）:**

```python
# SymPy hybrid の結果を登録するとき
new_expr = self.register_expression(
    canonical=display_str,       # 表示用（変更なし）
    latex=latex_str,
    sympy_canonical=sympy_str,   # SymPy 継続用
)
```

**後続操作でのチェーン（`julia_backend_handler.py`）:**

```python
def _get_sympy_canonical(self, expr_ref: str) -> str | None:
    """式が SymPy 結果なら sympy_canonical を返す、なければ None"""
    if self._engine:
        expr = self._engine.expressions.get(expr_ref)
        if expr and expr.sympy_canonical:
            return expr.sympy_canonical
    return None
```

---

### Step 3: `normal_order` の透過化テスト ✅ 実装済み

```python
# テスト: k+q → SymPy, 整数サイト → QA
def test_normal_order_routing_sympy():
    # c'(-k)*c(k) → SymPy（-k が代数インデックス）
    # δ(k,-k) が正しく 0 に評価される

def test_normal_order_routing_qa():
    # c'(1)*c(2) → QA（整数サイト）
    # δ(1,2)=0 が _nb_eval_site_deltas で正しく評価される
```

---

### Step 4: `_nb_sympy_wick` ✅ 実装済み

```julia
# _nb_sympy_wick(expr_str, rules_json, stats, vars_str)
# rules_json: [{"op1": "cd", "op2": "c", "value": "rho(INDEX1,INDEX2)"}]
function _nb_sympy_wick(expr_str::String, rules_json::String, 
                        stats::String, vars_str::String)
    # wicks() を使って Wick 収縮を計算
    # INDEX1_N テンプレートを SymPy の symbolic delta に対応付け
    ...
end
```

---

### Step 5: `vacuum_expectation` の透過化 ✅ 実装済み

`_nb_sympy_vacuum_expectation` を実装。正規順序後に演算子ゼロ項を抽出。
`δ(-k,k)=0` が正しく評価される。

### A-11: `collect_by_operators` / `combine_expressions` ✅ 実装済み（2026-06-06）

- **`_nb_sympy_collect_by_operators`**: 演算子積を共通因子とした因数分解。
  各項を演算子列でグループ化 → SymPy `factor()` で係数合算。
  例: `-u²ξ·c†(k)c(-k) + v²ξ·c†(k)c(-k)` → `(v²-u²)ξ·c†(k)c(-k)`
- **`_nb_sympy_substitute`**: 代入時に演算子順序を保持（正規順序化なし）。
  スカラー係数と各演算子インデックスに `subs()` を適用。
- **`_maybe_sympy_combine`**: SymPy+SymPy 加算。`"A + -B"` パターン回避。
- **`relabel_index`**: `_maybe_sympy_substitute` に委譲（QA K-convention 除去）。

---

### 全 Step 完了条件（確認済み）

- k+q を含む式に `normal_order` を適用すると SymPy が処理 ✅
- `δ(k,-k)` が正しく 0 に評価される ✅
- 整数サイトの `normal_order` は QA のまま（速度劣化なし）✅
- `wick_contract` と `vacuum_expectation` も透過的 ✅
- SymPy チェーン：SymPy 結果をさらに SymPy で処理可能 ✅
- `substitute`・`collect_by_operators`・`relabel_index` も SymPy チェーン対応 ✅
- `re_evaluate_expr` も SymPy チェーン対応（スカラー・インデックス一括置換） ✅
- ユーザーは QA/SymPy を意識しない ✅

---

## 6. 影響ファイル一覧

| ファイル | 実装内容 |
|---|---|
| `notebook/runtime/_julia_driver.jl` | `_nb_sympy_normal_order`・`_nb_sympy_wick`・`_nb_sympy_vacuum_expectation`・`_nb_sympy_substitute`・`_nb_sympy_collect_by_operators`・`_nb_split_sum_str` |
| `notebook/runtime/julia_backend_handler.py` | `build_command` SymPy ルーティング（全 Step）・`_get_sympy_canonical`・`_maybe_sympy_*` ヘルパー群 |
| `notebook/models/expression.py` | `sympy_canonical: str \| None` フィールド追加 |
| `notebook/engine.py` | SymPy 結果登録時に `sympy_canonical` をセット |
| `tests/test_algebraic_index_routing.py` | 27 テスト（Step 1-6・A-11・Display Bridge ガード）|

---

## 7. 既知の制約（実装後も残るもの）

1. **SymPy は遅い**: PyCall オーバーヘッドあり。整数サイト系は QA のまま使う理由。
2. **SymPy 結果の可読性**: QA の LaTeX 出力より冗長になる場合がある（`KroneckerDelta(p, k+q)` 等）。`postprocess_result_latex` でクリーンアップが必要。
3. **k+q 散乱での総和**: `Σ_k δ(p, k+q) f(k)` の `k` について和を取ると `f(p-q)` になるが、この「δ 関数の縮約」は SymPy の `doit()` で処理できる（追加実装不要）。

---

## 8. 参照

- `docs/open_items.md` § A-1 — 代数インデックスの課題
- `docs/developer_manuals/04_second_quantization_engine_design.md` — QA エンジン設計
- `notebook/runtime/_julia_driver.jl` — `_nb_sympy_fermion_comm`（Phase 1-3 実装済み）
- `notebook/runtime/julia_code_generator.py` — `_has_algebraic_indices()`・自動ルーティング
