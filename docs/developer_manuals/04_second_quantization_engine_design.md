# Second Quantization Engine Design

**Status**: QuantumAlgebra (QA) + SymPy hybrid operational (2026-06-06).  
**Last updated**: 2026-06-06

> **設計原則: QA → SymPy は一方通行**  
> セッションは QA で始まる。代数インデックス（`-k`, `k+q` 等）が出現した時点で
> SymPy に移行し、**以後はすべての操作を SymPy で継続する。SymPy → QA への逆行はない。**  
> 詳細: `docs/developer_manuals/05_algebraic_index_architecture.md` §3

---

## 1. 背景と問題

Julia バックエンドは `QuantumAlgebra.jl` を用いて正規順序化・交換子計算を行う。
しかし `QuantumAlgebra` は **カテゴリカル（名前ラベル）インデックスのみ**に対応しており、
代数的なインデックス式（`-k`、`k+q`、`a*k+b` 等）をサポートしない。

```julia
# QuantumAlgebra の制約
QuIndex(Symbol("k"))      # ✅ 単一文字ラベル
QuIndex(Symbol("-k"))     # ❌ ArgumentError: single character required
c(-QuIndex(:k))           # ❌ MethodError: -(::QuIndex) は未定義
c(k)  # k は Symbolics Num  # ❌ MethodError: QuIndex(::Num) は未定義
```

この制約は設計上のもので、パッチによる回避は不可能。

---

## 2. 分野別インデックス分類

第２量子化を用いる分野は「インデックスに算術が現れるか否か」で二分される。

### Group A: カテゴリカルインデックス（QuantumAlgebra で対応可）

| 分野 | 典型的インデックス | 例 |
|---|---|---|
| 量子光学・空洞 QED | モード番号、偏光ラベル | `a†_k a_k`（k = 1,2,... モード） |
| 原子・分子物理 | 離散量子数 (n,l,m) | `c†_{nlm} c_{n'l'm'}` |
| 量子化学 (CI, CC) | 分子軌道ラベル (p,q,r,s) | `c†_p c†_q c_s c_r` |
| 核物理（殻模型） | (j,m) 等の量子数 | `c†_{jm} c_{j'm'}` |
| スピン系 | サイトラベル | `S†_i S_j` |

これらは QuantumAlgebra の `@fermion_ops`/`@boson_ops` + `normal_form` + `comm` で現状動作する。

### Group B: 代数インデックス（QuantumAlgebra 不可 → SymPy が必要）

| 分野 | 典型的インデックス | 例 |
|---|---|---|
| 凝縮系・BCS 超伝導 | `-k`（時間反転対称） | `c†_k c†_{-k}`（Cooper ペア） |
| BEC・Bogoliubov 変換 | `-k` | `b_k = u_k a_k + v_k a†_{-k}` |
| 格子物理・スピン波 | `k+q`（運動量保存） | `c†_{k+q} c_k a_q` |
| 超格子・Zone folding | `k+G`（逆格子ベクトル） | `c†_{k+G}` |
| 量子場理論（摂動論） | `p±k`（ループ積分） | `ψ†(k) ψ(k+q)` |
| 核物理（ペアリング相関） | `-m`（time-reversal） | `c†_{jm} c†_{j,-m}` |

分野名ではなく**「インデックスに算術が現れるか」が判断基準**。

---

## 3. 実験結果（2026-05-27）

### 3.1 試みたアプローチとその評価

| アプローチ | 結果 | 理由 |
|---|---|---|
| `QuIndex(Symbol("-k"))` | ❌ クラッシュ | 単一文字制約違反 |
| `-sumindex(1)` | ❌ クラッシュ | `-(::QuIndex)` 未定義 |
| `c(-k)` (Symbolics 変数) | ❌ クラッシュ | `QuIndex(::Num)` 未定義 |
| `sumindex(-1)` (負 num) | ⚠️ 動くが不正確 | `δ(#₋₁, :q)` は常に 0 |
| `QuIndex(Symbol("K"))` (大文字規則) | ⚠️ クラッシュなし | 物理的 δ 関数が評価不能 |
| `sympy.physics.secondquant` | ❌ 不適 | 内部インデックス `_i`,`_a` が混入 |
| **生 SymPy + KroneckerDelta** | ✅ **正しく動作** | 代数引数を保持、subs で評価可能 |

### 3.2 SymPy アプローチの検証結果

```julia
# Julia ドライバ内で実行可能（SymPy.PyCall 経由）
sp = SymPy.PyCall.pyimport("sympy")
k, q = sp.symbols("k q")
c = sp.Function("c"); cd = sp.Function("cd")
delta = sp.KroneckerDelta

# フェルミオン版ライプニッツ則: [AB, C] = A{B,C} - {A,C}B
# {c†(p), c(r)} = δ(p,r)

# BCS ペアリング項の交換子
# [c†_k c†_{-k}, c_q] = c†_k δ(-k,q) - δ(k,q) c†_{-k}
result = cd(k)*delta(-k, q) - delta(k, q)*cd(-k)
# → "-cd(-k)*KroneckerDelta(k, q) + cd(k)*KroneckerDelta(-k, q)"
# LaTeX: - c^\dagger_{-k}\,\delta_{kq} + c^\dagger_k\,\delta_{-k,q}  ✅ 正しい物理
```

```julia
# 代数インデックスも問題なし
c(sp.symbols("a")*k + sp.symbols("b"))
# → "AnnihilateFermion(a*k + b)"  ✅
```

### 3.3 SymPy アプローチの特性

- **δ(-k, q)** は `KroneckerDelta(-k, q)` としてシンボリックに保持される
- **自動簡約なし**: `δ(-k,k)` → 0 は自動では行われない（k=0 かどうか不明なため）
- **subs で評価可**: `delta(-k, q).subs(q, -k)` → `1` ✅
- **LaTeX 出力**: `sp.latex(result)` で直接生成可能

---

## 4. アーキテクチャ設計

### 4.1 原則

これは **Julia ↔ Maxima のバックエンド切り替えではない**。
SymPy は Julia ドライバ内部のヘルパーであり、`SymPy.PyCall` 経由で既に統合されている。
CLAUDE.md の「バックエンド切り替えはユーザー確認が必要」原則には抵触しない。

### 4.2 コンポーネント構成（目標状態）

```
Julia バックエンド
├── QuantumAlgebra ヘルパー群（既存）
│   ├── _nb_eval_index_sum(body, dummy)
│   ├── _nb_eval_continuous_integral(...)
│   └── normal_form, comm  ← 直接 QuantumAlgebra 呼び出し
│
└── SymPy ヘルパー群（✅ 全実装済み 2026-06-06）
    ├── _nb_sympy_fermion_comm(H_str, c_str, vars_str)
    │     フェルミオン交換子: [H, c(q)] を SymPy で計算
    ├── _nb_sympy_boson_comm(H_str, a_str, vars_str)
    │     ボゾン交換子: [H, a(q)] を SymPy で計算
    ├── _nb_sympy_normal_order(expr_str, stats, vars_str)
    │     正規順序化（フェルミオン/ボゾン共通）
    ├── _nb_sympy_wick(expr_str, rules_json, stats, vars_str)
    │     Wick 収縮（contraction_rules 適用）
    ├── _nb_sympy_vacuum_expectation(expr_str, stats, vars_str)
    │     真空期待値（正規順序後にゼロでない項を抽出）
    ├── _nb_sympy_substitute(expr_str, lhs_str, rhs_str, stats, vars_str)
    │     代入（演算子順序保持、k→-k・k→k+q 対応）
    ├── _nb_sympy_collect_by_operators(expr_str, stats, vars_str)
    │     演算子積を共通因子とした係数合算（因数分解）
    ├── _nb_sympy_general_comm(a_str, b_str, stats, vars_str)
    │     混合 QA+SymPy 式の交換子（派生則による展開）
    └── _re_evaluate_sympy (Python, handler メソッド)
          複数シンボルを _nb_sympy_substitute で順次適用（re_evaluate_expr の SymPy パス）
```

### 4.3 コードジェネレータの振る舞い

```
IndexSymbol の生成規則（現在の実装）:
  unbound, 非負         → QuantumAlgebra.QuIndex(Symbol("k"))
  bound (IndexSum 内)   → QuantumAlgebra.sumindex(1)
  unbound, 負 (-k)      → QuantumAlgebra.QuIndex(Symbol("K"))  ← 大文字規則（暫定）
  bound, 負 (-k)        → QuantumAlgebra.sumindex(-1)           ← 暫定

SymPy ヘルパー実装後の目標:
  代数インデックスを含む交換子 → _nb_sympy_fermion_comm / _nb_sympy_boson_comm
  代数インデックスのない交換子 → 引き続き QuantumAlgebra.comm
```

### 4.4 セッションモードと利用エンジン

```
SessionMode.second_quantization (量子光学・原子物理・量子化学)
  → QuantumAlgebra ヘルパー（Group A フィールド向け）

SessionMode.condensed_matter（凝縮系・格子物理）
  → 代数インデックス検出時に SymPy ヘルパーを使用
  → 通常インデックスは引き続き QuantumAlgebra

将来: SessionMode.quantum_field_theory 等での対応も同様
```

---

## 5. 実装状況（全 Phase 完了 2026-06-06）

### Phase 1: _nb_sympy_fermion_comm ✅ 実装済み

**対象操作**: `normal_order` in BCS / 凝縮系セッション  
**入力形式**:
```julia
_nb_sympy_fermion_comm(
    "cd(k)*cd(-k)",   # 演算子積（文字列）
    "c(q)",            # 対象演算子（文字列）
    "k,q"             # シンボリック変数（カンマ区切り）
)
# → "-cd(-k)*KroneckerDelta(k, q) + cd(k)*KroneckerDelta(-k, q)"
# → LaTeX: "- c^\dagger_{-k}\\,\\delta_{kq} + c^\dagger_k\\,\\delta_{-k,q}"
```

**アルゴリズム**:
1. 引数を SymPy 式として parse
2. 積の因子列に分解
3. フェルミオン版ライプニッツ則を再帰適用: `[A₁A₂…Aₙ, C] = Σᵢ (A₁…Aᵢ₋₁){Aᵢ,C}Aᵢ₊₁…Aₙ × (-1)^(Σⱼ<ᵢ sign(Aⱼ))`
4. 反交換子 `{c†(p), c(q)} = δ(p,q)`, `{c(p), c(q)} = 0` を適用
5. SymPy.expand() + LaTeX 出力

### Phase 2: 正規順序化 ✅ 実装済み

`_nb_sympy_normal_order(expr_str, "fermion"/"boson", vars_str)`  
代数インデックスを含む式の正規順序化。`δ(-k,k)=0` 正しく評価。

### Phase 3: CodeGenerator への統合 ✅ 実装済み

`JuliaBackendHandler.build_command` が `_get_sympy_canonical()` で判定し  
`_maybe_sympy_*` ヘルパー経由で SymPy チェーンにルーティング。

---

## 6. 大文字規則の扱い（暫定対処 → 正式実装済み）

`-k` インデックスに対する旧来の大文字規則（`julia_code_generator.py`）は
SymPy ハイブリッド実装（Phase 1-3）で置き換え済み。

現在のルーティング（`julia_backend_handler.py`）:
- `_has_algebraic_indices()` が代数インデックスを検出
- 検出時：`_nb_sympy_*` ヘルパーに自動ルーティング
- `sympy_canonical` が設定された式：後続操作も SymPy チェーンで継続
- 検出なし（整数サイト）：従来の QA (`normal_form`, `comm`) で処理

```python
# julia_code_generator.py の大文字規則はフォールバックとして残存するが
# 代数インデックスを含む演算子は build_command 前に SymPy 側へ分岐するため
# 実際には到達しない
```

---

## 7. QuantumAlgebra.jl の既知制約まとめ

### 7.1 代数インデックス不可（§1 参照）

`QuIndex` は単一文字ラベルのみ受け付ける。`-k`、`k+q` 等の算術インデックスはクラッシュする。
現行の暫定対処は大文字規則（`-k` → `QuIndex("K")`）。正式対応は SymPy ハイブリッド（§5）。

### 7.2 異種フェルミオン演算子の交換問題（2026-05-31 発見）

**問題:**  
`@fermion_ops a` と `@fermion_ops b` を**別々に宣言**すると、`a` と `b` の間の交換関係は **commute（交換）** として扱われる。

```julia
@fermion_ops a; @fermion_ops b
normal_form(a(1)*b(1) + b(1)*a(1))
# → 2 a(b1) b(1)    ← 交換！（本来は 0 であるべき）
```

これは QuantumAlgebra.jl の設計上の仕様であり、パッチ不可。

**影響を受けるケース:**  
スピン↑を `a`、スピン↓を `b` として別々に宣言した BCS 計算。対角化後の係数が誤った符号になる。

**正しいアプローチ: 2添字演算子**  
単一 `@fermion_ops c` を宣言し、第2引数でスピンを区別する。

```julia
@fermion_ops c
normal_form(c(k, QuIndex("u"))*c(k, QuIndex("d")) + c(k, QuIndex("d"))*c(k, QuIndex("u")))
# → 0    ← 正しく反交換
```

ノートブックでの使い方:
```
c を電子のフェルミオン演算子として定義する
E1 の c(k, ↑) に rhs1 を、c(-k, ↓) に rhs2 を代入する
```

`operator_substitute` の `spin_label` フィルタが `c(k,↑)` と `c(k,↓)` を独立に代入する。

---

## 8. 関連ファイル

| ファイル | 役割 |
|---|---|
| `notebook/runtime/_julia_driver.jl` | Julia ドライバ（ヘルパー関数群） |
| `notebook/runtime/julia_code_generator.py` | IR → Julia コード文字列変換 |
| `notebook/runtime/julia_backend_handler.py` | 結果の後処理・LaTeX 変換 |
| `domains/condensed_matter/supervisor_hint.txt` | 凝縮系向け LLM ヒント |
| `domains/quantum_optics/supervisor_hint.txt` | 量子光学向け LLM ヒント |
| `tests/test_compiler_arch.py` | コンパイラ層のユニットテスト |
| `tests/experiments/test_quantum_delta_integration.py` | Julia ライブ統合テスト |
