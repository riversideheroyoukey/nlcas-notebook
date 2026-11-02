# Notation Bridge の parser / resolver 設計に関する懸念材料

## 目的

Notation Bridge の parser / resolver 設計について、懸念材料を整理したい。

このシステムでは、CAS 文法を利用者から隠蔽し、利用者がノート・黒板・教科書で普段使う notation のまま入力できることを目指している。

現在の懸念は、次の点である。

---

## 1. 同じ surface notation が、分野や文脈によって異なる数学的処理を要求する

たとえば、以下のような notation がある。

### `f'(x)`, `X'X`, `a'(i)`

- 解析・高校数学では導関数
- 計量経済学や線形代数では transpose の可能性
- 第2量子化では `a'(i)` が生成演算子を意味する可能性

### `\Delta`

- 電磁気・流体・ベクトル解析ではラプラシアン
- 時系列・差分方程式では差分演算子
- 凝縮系物理ではギャップ関数
- 数理ファイナンスでは Greek delta
- 幾何・教育文脈では三角形記号

### `[A,B]`

- 量子力学・第2量子化では交換子
- 一般数学では区間やリスト
- プログラム的文脈では indexing / array access 的に見える可能性

### `{A,B}`

- 第2量子化では反交換子
- 一般数学では集合やペア

### `|A|`

- 絶対値
- 行列式
- ノルム
- 集合の濃度

### `E[X]`

- 確率統計では期待値
- 物理ではエネルギー関数や電場 `E` の関数に見える可能性

---

## 2. 理論的意味が分野で違っても、数学的処理が同じなら分岐させる必要はない

たとえば `x_i` は、分野によって理論的意味が異なる。

- 解析力学では一般化座標成分
- 経済学では財 `i` の消費量
- 数理生物では種 `i` の個体数
- 線形代数ではベクトル成分
- 統計力学ではサイト `i` の変数

しかし、多くの場合は `IndexedSymbol` として処理すればよい。

つまり、Notation Bridge が分岐すべきなのは「理論的意味が違う場合」ではなく、以下が変わる場合である。

- Domain IR が変わる
- 代数構造が変わる
- backend lowering が変わる
- simplification rule が変わる

---

## 3. parser が早すぎる段階で数学的意味を固定すると危険である

たとえば parser が `f'(x)` を常に `Derivative` として確定すると、econometrics の `X'X` や quantum_operator_algebra の `a'(i)` が壊れる。

逆に、prime を常に dagger / creation operator として扱うと、高校数学・解析の `f'(x)` が壊れる。

同様に、`\Delta f` を常に `Laplacian` として確定すると、`\Delta_k` の gap function や financial Greek Delta が壊れる。

---

## 4. parser と resolver の責務分離が必要である

懸念としては、parser を domain ごとに完全に分岐させると、保守が難しくなり、ルールが重複する。

一方で、完全に domain-independent な parser だけでは、以下のような特殊記法を拾いにくい。

- `(ij|kl)` の二電子積分
- `\sum_{\langle ij\rangle}` の最近接ペア和
- `[A,B]` の交換子候補

したがって、以下のような設計が妥当か検討したい。

- `SurfaceParser` はなるべく domain-independent にする
- `SurfaceParser` は prime, bracket, delta, subscript, superscript, juxtaposition などの表面的構造を Surface AST にする
- 数学的意味の確定は `DomainAwareResolver` が行う
- 必要な場合だけ optional domain-specific parse plugin を使う

---

## 5. domain-aware resolver がどの情報を使って意味を決めるべきか

候補としては、以下の優先順位を考えている。

### 1. Explicit SymbolContext declaration

例:

- `A is matrix`
- `a is fermion operator`
- `q is time-dependent`
- `X is random variable`

### 2. Active domain / notation profile

例:

- `econometrics`
- `quantum_chemistry_sq`
- `electromagnetism`
- `financial_mathematics`

### 3. Surrounding syntax

例:

- `f'(x)` は derivative 候補
- `X'X` は transpose 候補
- `a'(i)` は registered quantum operator なら creation operator 候補
- `\Delta_k` は subscripted symbol なので gap function 候補
- `\Delta f` は operator applied to function なので Laplacian 候補

### 4. Previous usage in session

例:

- すでに `\Delta_k` を gap function として使っているなら、以後も優先する

### 5. LLM supervisor ranking

周辺文脈から候補を順位づける。

### 6. User confirmation

confidence が低い場合は確認 UI へ回す。

この優先順位でよいか検討したい。

---

## 6. 低信頼度の場合に silent に決めるのは危険である

特に次の notation は、誤解釈すると計算結果が根本的に変わる。

- prime
- `\Delta`
- `[A,B]`
- `{A,B}`
- `|A|`
- `E[X]`
- `H(X)`
- `P(A)`
- `i`
- `\sigma`
- `\mu`
- `\lambda`

これらについては、confidence が低い場合、ユーザー確認 UI に候補を出すべきではないか。

---

## 7. 共通 profile と domain-specific override の設計が必要である

domain ごとにすべての notation rule を書くと重複が多すぎる。

たとえば、古典物理系は以下を共有できる。

- calculus
- classical_physics
- vector_calculus
- variational_calculus

第2量子化系は以下を共有できる。

- quantum_operator_algebra
- index_notation
- expectation_values

その上で、個別 domain は priority や meaning candidate を上書きすればよい。

懸念は、どこまでを共通 profile に置き、どこからを domain override にするかである。

---

## 8. 数学的処理が同じ場合に過剰に domain-specific rule を作ると、実装が複雑化する

たとえば `x_i` のような `IndexedSymbol` は、多くの分野で理論的意味が違っても同じ処理で足りる。

これを分野ごとに別 rule にすると、catalog が膨らみ、バグが増える。

したがって、catalog には以下を持たせるべきではないか。

- possible theoretical meanings
- mathematical treatment
- shared treatment?
- backend consequence
- requires disambiguation?

---

## 9. CAS 文法が UI に漏れる危険がある

利用者に以下を書かせてはいけない。

- Maxima: `diff(f,x)`, `sum(...)`, `x[i]`
- Julia/QuantumAlgebra: `a(QuantumAlgebra.QuIndex(:i))`, `a(sumindex(1))`
- Symbolics: `Differential(x)(f)`

利用者には、以下のような通常 notation を使わせたい。

- `f'(x)`
- `\sum_j \delta_{ij} f_j`
- `a_i^\dagger a_j`
- `\langle \psi | A | \psi \rangle`
- `\nabla \cdot E`

そのためには、入力時だけでなく CAS 結果の display bridge も必要である。

---

## 10. save/load 後も意味解決結果が壊れない必要がある

たとえば、以下が保存・読み込み後にも同じ Semantic IR として保持される必要がある。

- `IndexedSymbol x_i` と `FunctionCall x(i)`
- derivative prime と transpose prime
- gap function `\Delta_k` と Laplacian `\Delta f`
- quantum creation operator `a'(i)` と ordinary derivative `f'(x)`

---

## 11. 検討したい設計対象

これらの懸念を踏まえて、以下の役割分担について設計を検討したい。

- parser
- Surface AST
- DomainAwareResolver
- notation profiles
- SymbolContext
- confirmation UI
- display bridge
- save/load semantic preservation
