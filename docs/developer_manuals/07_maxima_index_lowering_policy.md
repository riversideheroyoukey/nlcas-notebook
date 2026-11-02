# 仕様変更提案: Maxima における文字添字 lowering 方針の修正

## 1. 背景

現在の Index Notation Phase 1 実装では、添字付きスカラー式 `x_i` が Maxima では `x[i]` に lowering されている。

しかし、理論計算における `x_i` は、必ずしも配列アクセスを意味しない。多くの場合、次のような意味で使われる。

```text
i でラベルされた記号族
モード係数
展開係数
格子点上の場
成分表示
離散添字付きの理論的な量
```

したがって、Maxima における scalar indexed family の default lowering を `x[i]` に固定するのは望ましくない。

今後は、Maxima では default を `x(i)` とし、`x[i]` は明示的に配列・ベクトル・行列・テンソル成分として扱う場合に限定する。

---

## 2. 基本原則

### 2.1 Notebook IR が正本である

Notebook 内部では、添字付き量は常に `IndexedSymbol` として保持する。

```text
x_i
```

は内部的には、

```text
IndexedSymbol(base=x, indices=[i])
```

であり、`x(i)` や `x[i]` は backend へ渡すための lowering 表現にすぎない。

### 2.2 ユーザー表示は変えない

backend lowering が `x(i)` であっても `x[i]` であっても、ユーザーに見せる表示は常に自然な添字表記にする。

```text
x_i
A_{ij}
c_k
phi_i
```

### 2.3 Maxima の `[]` は明示指定時のみ使う

Maxima の `x[i]` は、配列・リスト・行列・テンソル成分として扱う意味合いが強い。

したがって、`x[i]` は次のような場合に限定する。

```text
vector component
matrix component
array access
tensor component
Maxima の matrix / list / array と連携する計算
```

scalar indexed family の default として `x[i]` を使わない。

---

## 3. 追加する設定

`SymbolProperty`、`Quantity`、または `MathType` 周辺に、添字付き量の backend lowering 方針を表す設定を追加する。

推奨設定名:

```python
index_lowering_style
```

値の候補:

```python
"backend_default"
"function_call"
"array_access"
"atom"
"quantum_index"
```

---

## 4. 各 lowering style の意味

### 4.1 `backend_default`

backend と `MathType` に応じて標準方針を選ぶ。

Maxima の default:

```text
scalar / indexed_family      → function_call
vector / matrix / tensor     → array_access
operator / function_like     → function_call
```

Julia / Symbolics の default:

```text
scalar / indexed_family      → function_call
matrix / tensor              → 当面 function_call
```

Julia / QuantumAlgebra の default:

```text
operator                     → quantum_index
```

### 4.2 `function_call`

添字を関数引数として lowering する。

Maxima:

```text
x_i       → x(i)
c_k       → c(k)
phi_i     → phi(i)
A_{ij}    → A(i,j)
```

Julia / Symbolics:

```text
x_i       → x(i)
A_{ij}    → A(i,j)
```

用途:

```text
添字付き記号族
展開係数
モード係数
格子点上の場
離散ラベル付きの量
理論計算での一般的な添字付き量
```

### 4.3 `array_access`

添字を配列・行列アクセスとして lowering する。

Maxima:

```text
x_i       → x[i]
A_{ij}    → A[i,j]
T_{ijk}   → T[i,j,k]
```

Julia / Symbolics:

```text
将来的に x[i], A[i,j] または Symbolics array に対応
```

用途:

```text
明示的に配列、ベクトル、行列、テンソル成分として宣言された量
Maxima の matrix / list / array と連携する計算
```

### 4.4 `atom`

添字付き量を評価されにくい単一アトムとして lowering する。

Maxima:

```text
x_i       → x_i
A_ij      → A_ij
```

用途:

```text
表示・置換用の不透明な添字付き記号
CAS に添字構造を解釈させたくない場合
```

### 4.5 `quantum_index`

Julia / QuantumAlgebra 用の添字 lowering。

Julia / QuantumAlgebra:

```text
free index:
  a_i     → a(QuantumAlgebra.QuIndex(:i))
  a†_j    → a'(QuantumAlgebra.QuIndex(:j))

dummy index in IndexSum:
  a_j     → a(QuantumAlgebra.sumindex(n))
```

用途:

```text
ボソン演算子
フェルミオン演算子
QuantumAlgebra.jl の演算子添字
```

実際に `QuIndex(:i)` が適切か、または `a(:i)` 等が適切かは、実機確認済みの形式に従う。

---

## 5. Maxima lowering の新しい default

Maxima では以下を default にする。

```text
Scalar indexed family:
  x_i       → x(i)

Symbolic coefficient:
  c_k       → c(k)

Lattice field:
  phi_i     → phi(i)

Function-like indexed quantity:
  f_i       → f(i)

Operator-like indexed quantity:
  a_i       → a(i)

Explicit vector / matrix / tensor component:
  x_i       → x[i]
  A_{ij}    → A[i,j]
  T_{ijk}   → T[i,j,k]
```

`[]` を使うのは、`SymbolContext`、`MathType`、または `index_lowering_style` により、その量が array / vector / matrix / tensor component として明示されている場合に限定する。

推論できない場合、Maxima では `function_call` を安全な default とする。

---

## 6. 実装方針

1. `IndexedSymbol` の Maxima lowering ロジックを修正する。
2. scalar indexed family の default を `x[i]` から `x(i)` に変更する。
3. matrix / array / tensor と明示されているものは `array_access` により `A[i,j]` を使えるようにする。
4. `index_lowering_style` が未指定の場合は `backend_default` とみなす。
5. 既存の `MathType` や `SymbolContext` から可能な範囲で default style を推論する。
6. 推論できない場合、Maxima では `function_call` を使う。
7. Julia / QuantumAlgebra の `quantum_index` lowering は壊さない。
8. Julia / Symbolics は当面 `function_call` を default としてよい。
9. ユーザー UI や LaTeX 表示は変更しない。
10. notebook JSON には `IndexedSymbol` と lowering style 情報を保存できるようにする。ただし既存保存形式との後方互換を維持する。

---

## 7. テスト変更

### 7.1 既存テストの期待値変更

現在の期待値:

```text
Scalar index:
  x_i → Maxima: x[i]
```

を次に変更する。

```text
Scalar indexed family:
  x_i → Maxima: x(i)
```

### 7.2 追加テスト

#### Test 1: Maxima scalar indexed family

入力:

```text
x_i
```

期待:

```text
IR:
  IndexedSymbol(base=x, indices=[i])

Maxima lowering:
  x(i)
```

#### Test 2: Maxima symbolic coefficient

入力:

```text
c_k
```

期待:

```text
Maxima lowering:
  c(k)
```

#### Test 3: Maxima matrix component

`A` を matrix として `SymbolContext` に登録した場合:

```text
A_{ij}
```

期待:

```text
Maxima lowering:
  A[i,j]
```

#### Test 4: Maxima function_call override

`A` が matrix でも、`index_lowering_style="function_call"` が指定されている場合:

```text
A_{ij}
```

期待:

```text
Maxima lowering:
  A(i,j)
```

#### Test 5: Maxima array_access override

`x` が scalar でも、`index_lowering_style="array_access"` が指定されている場合:

```text
x_i
```

期待:

```text
Maxima lowering:
  x[i]
```

#### Test 6: Maxima atom override

`x` に `index_lowering_style="atom"` が指定されている場合:

```text
x_i
```

期待:

```text
Maxima lowering:
  x_i
```

#### Test 7: Julia / QuantumAlgebra regression

以下が従来通り動くことを確認する。

```text
a_i
a^\dagger_j
IndexSum(a_j, j)
```

期待:

```text
free index:
  QuIndex(:i) 相当

dummy index:
  sumindex(n) 相当
```

実際の表現は、既存の Julia / QuantumAlgebra 実装に合わせる。

#### Test 8: 表示の不変性

backend lowering が `x(i)` でも `x[i]` でも、ユーザー表示は常に次を維持する。

```text
x_i
A_{ij}
```

---

## 8. 注意点

### 8.1 `x(i)` は lowering 表現にすぎない

Maxima の `x(i)` は Maxima 上では function application になる。  
しかし、Notebook IR ではこれは `IndexedSymbol` であり、通常の `FunctionCall` とは区別する。

### 8.2 `x_i` と通常関数 `x(i)` の衝突

`x_i` と通常の関数 `x(i)` は、Maxima 文字列としては同じ `x(i)` になる可能性がある。

したがって、Notebook 内部では必ず、

```text
IndexedSymbol
FunctionCall
```

を区別する。

必要であれば将来的に、Maxima lowering 時に添字付き量だけ別名へ落とすオプションを検討する。

例:

```text
x_i → x__idx(i)
```

ただし、初期実装では不要。

### 8.3 添字は通常の微分変数ではない

`i` が `IndexSymbol` である場合、通常の微分変数として扱わない。

たとえば、

```text
E1 を i で微分して
```

という指示が来た場合、将来的には確認またはエラーにする。

```text
i は添字です。微分変数として扱いますか？
```

のような確認を出すことが望ましい。

### 8.4 `array_access` は明示指定時のみ

`array_access` は、配列・行列・テンソルとして明示された量に限定する。

### 8.5 今回の範囲外

以下は今回の変更範囲外とする。

```text
Einstein summation convention の自動検出
KroneckerDelta 縮約
IndexSum の自動簡約
DiracDelta / Integral
上下添字・共変反変の本格対応
```

---

## 9. 完了条件

- Maxima scalar indexed family の default lowering が `x(i)` になっている。
- `index_lowering_style` により `function_call` / `array_access` / `atom` を選択できる。
- Julia / QuantumAlgebra の symbolic index lowering が壊れていない。
- 既存テストがすべて通る。
- 追加テストが通る。
- `PROGRESS.md` または index notation design note に、以下が追記されている。
  - Maxima の indexed scalar default は `function_call` である。
  - `array_access` は明示指定時のみ使用する。
  - Notebook IR の正本は `IndexedSymbol` であり、`x(i)` / `x[i]` は backend lowering 表現にすぎない。

---

## 10. 一文での要約

Maxima では、理論計算における一般の添字付き量 `x_i` は default で `x(i)` に lowering し、`x[i]` は配列・行列・テンソル成分として明示された場合にのみ使う。
