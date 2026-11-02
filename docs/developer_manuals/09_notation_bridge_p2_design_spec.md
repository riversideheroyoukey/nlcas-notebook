# Notation Bridge P2 設計仕様

**作成:** 2026-05-25  
**ベース:** `docs/developer_manuals/10_notation_bridge_parser_resolver_concerns.md`  
**スコープ:** Notation Bridge P2 の設計方針と実装指針

---

## 目的

Notation Bridge の parser / resolver 設計について、懸念材料を整理し P2 の実装方針を定める。

このシステムでは、CAS 文法を利用者から隠蔽し、利用者がノート・黒板・教科書で普段使う notation のまま入力できることを目指している。

---

## 懸念の分類

以下の懸念を3種類に分けて扱う。実装の優先順位の判断に使うこと。

| 種別 | 内容 |
|---|---|
| **A: 設計確認** | すでに現行アーキテクチャで対処済み。確認として記録する |
| **B: P2 拡張** | 現行設計の延長線上にある実装可能な追加機能 |
| **C: 根本的ギャップ** | 現行設計で未対処の構造的問題。P2 で解消すべき |

各セクションに `[A]` `[B]` `[C]` を付記する。

---

## 1. 同じ surface notation が、分野や文脈によって異なる数学的処理を要求する `[C]`

以下のような notation は、ドメインや宣言状態によって意味が変わる。

### `f'(x)`, `X'X`, `a'(i)`

- 解析・高校数学では `f'(x)` が導関数
- 計量経済学や線形代数では `X'X` が転置行列積
- 第2量子化では `a'(i)` が生成演算子

**現状の実装:**  
SharedParser の tokenizer は `'` を単独の QUOTE トークンとして切り出す。パーサーはその後の文脈（識別子への後置）で `PrimeExpr` ノードを生成する。JuliaSemanticResolver が SymbolContext を参照し、登録済み演算子なら creation operator として昇格させる。

**問題点:**  
~~Maxima セッションで `f'(x)` を入力すると、`f'` という名前のシンボルへの関数適用として処理され、derivative として解釈されない。~~ **✅ 解決済み (2026-06-01 / B-3)**: `MaximaSemanticResolver` で `f'(x)` → `diff(f(x),x)` 変換を実装。複合引数は `SymbolContext` の自由変数判定で microresolution。詳細は `docs/open_items.md` §B-3 参照。

### `\Delta`

- 電磁気・流体・ベクトル解析ではラプラシアン
- 時系列・差分方程式では差分演算子
- 凝縮系物理ではギャップ関数（例: `\Delta_k`）
- 数理ファイナンスでは Greek delta
- 幾何・教育文脈では三角形記号

**現状の実装:**  
`latex_converter.py` で `\Delta f` → `laplacian(f)` に変換。`\Delta_k` は subscript 付きのため `IndexedSymbol` として処理され Laplacian にならない。事実上は動作しているが、設計として意図した分岐ではない。

### `[A,B]`

- 量子力学・第2量子化では交換子
- 一般数学では区間やリスト

**現状の実装:**  
Julia セッションでは `JuliaBackendHandler` の Phase 0/0.5 で処理。Maxima セッションでは通過し区間・リストとして扱われる。Backend Encapsulation Rule に従った実装。

### `{A,B}` / `|A|` / `E[X]`

| 記法 | 候補 |
|---|---|
| `{A,B}` | 反交換子 / 集合 / ペア |
| `\|A\|` | 絶対値 / 行列式 / ノルム / 集合の濃度 |
| `E[X]` | 確率統計の期待値 / エネルギー関数 `E` の引数 |
| `H(X)` | 情報エントロピー / ハミルトニアン関数 |
| `P(A)` | 確率 / 演算子 |
| `\bar{X}` / `\overline{X}` | **名前**（H-1 の装飾名）/ 複素共役 / 平均 |

**`\bar{X}` の決着（2026-08-09、確認 UI 不要）:** 中身が単独の名前なら H-1 の
装飾名（`\bar{H}` → `H_bar`）、複合式なら複素共役（`\overline{f(x)}` →
`conjugate(f(x))`）。**尋ねないのは、綴りの形が答えを決めているから**である——
`\bar{H}` は「H バー」という量の書き方であり、`\overline{a+b}` を「a+b バー」という
名前だと読む読者はいない。以前は後者も名前に落ち、共役が無言で消えていた
（しかも `a+b_bar` と、`b` だけが装飾された）。
なお「平均」は、この決着では共役と同じ `conjugate` に落ちる。区別が要るなら
`mean(X)` のような明示的な関数で書く。

---

## 2. 理論的意味が分野で違っても、数学的処理が同じなら分岐させる必要はない `[A]`

たとえば `x_i` は、分野によって理論的意味が異なる。

- 解析力学では一般化座標成分
- 経済学では財 `i` の消費量
- 数理生物では種 `i` の個体数
- 統計力学ではサイト `i` の変数

しかし多くの場合は `IndexedSymbol` として処理すれば足りる。

**Notation Bridge が分岐すべき基準:**

- Domain IR ノードが変わる（例: `Laplacian` vs `IndexedSymbol`）
- 代数構造が変わる（可換 vs 非可換）
- backend lowering が変わる（`x(i)` vs `x[i]` vs `QuIndex`）
- simplification rule が変わる（交換関係・縮約規則など）

理論的意味の違いだけでは分岐させない。これが catalog 設計の基本方針。

> **現行実装との対応:** `IndexedSymbol` の共通処理はすでにこの原則に従っている。設計確認 `[A]`。

---

## 3. Parser が早すぎる段階で数学的意味を固定すると危険である `[C]`

### 現在の prime トークナイズの実態

SharedParser の tokenizer は `'` を `QUOTE` トークンとして切り出す。パーサーは `ID QUOTE` を `PrimeExpr` ノードに変換する（表面構造として中立）。意味の確定は SemanticResolver が行う。

- Julia 文脈: `JuliaSemanticResolver` が SymbolContext を参照し creation operator に昇格
- Maxima 文脈: `MaximaSemanticResolver` は現在 `PrimeExpr` を `Symbol(name="f'")` に変換するだけ → derivative として解釈されない

**危険なケース:**  
- `\Delta f` を常に Laplacian に変換: `\Delta_k` が壊れる（現状は偶然動いている）
- `[A,B]` を Julia 専用処理: Maxima 文脈で交換子を書けない（仕様と割り切っているが、将来の拡張で引っかかる可能性）

---

## 4. Parser と Resolver の責務分離 `[A/B]`

### 現行設計（設計確認 `[A]`）

```
SharedParser (Surface AST)
  └── Juxtaposition, BracketExpr, PrimeExpr, Subscript, Superscript 等
        ↓
SemanticResolver (Semantic AST)
  ├── JuliaSemanticResolver
  └── MaximaSemanticResolver
```

SharedParser はすでに domain-independent。意味の確定は SemanticResolver で行われている。この責務分離は正しく、維持すること。

### 追加提案: domain-specific parse plugin `[B]`

以下のような特殊記法は、Surface AST では構造を把握できるが、**どの SemanticResolver でも解決できない**ものがある。

| 記法 | 懸念 |
|---|---|
| `(ij\|kl)` | 二電子積分（化学）。`BracketExpr` としてパースされるが通常の関数適用に見えない |
| `\sum_{\langle ij\rangle}` | 最近接ペア和（凝縮系）。`\langle ij \rangle` 部分は angle bracket の `BracketExpr` としてパース可能 |

後者は実は「domain-specific parse plugin なし」で `DomainAwareResolver` 側の規則として対応できる可能性が高い。前者（二電子積分）のような真にドメイン固有の構文のみ plugin として実装する。

---

## 5. DomainAwareResolver が意味を決める優先順位 `[B/C]`

以下の優先順位を採用する。

### Priority 1: Explicit SymbolContext declaration `[A]`

```
a is fermion operator  →  SymbolContext.statistics["a"] = "fermi"
q is time-dependent    →  SymbolContext.time_dependent["q"] = True
X is matrix            →  SymbolContext.kind["X"] = "matrix"
```

すでに SymbolContext に宣言されている場合、それ以下の優先度は適用しない。確認 UI も不要。

### Priority 2: Active domain / notation profile `[B]`

セクション 7 で定義する notation profile（ドメインクラスターに対応する記法規則セット）を参照する。

### Priority 3: Surrounding syntax `[A]`

- `f'(x)` → derivative 候補（括弧付き関数適用）
- `X'X` → transpose 候補（直後に juxtaposition）
- `\Delta_k` → subscript 付きシンボル候補（Laplacian ではない）
- `\Delta f` → Laplacian 候補（無修飾シンボルへの適用）

これは既存の SemanticResolver が行っている処理の形式化。

### Priority 4: Previous usage in session `[C]`

**現状:** `ir_ast` は保存されないため、この層は未実装。

**提案:** `SymbolContext` に `notation_usage: dict[str, str]` フィールドを追加し、初回解決時に記録する。

```python
# 例: \Delta_k を gap function として使った場合
symbol_context.notation_usage["\\Delta_k"] = "gap_function"
```

保存・読み込み対応（Session.symbol_context 経由で永続化）。以後の `\Delta_k` はこのエントリを優先し、確認 UI を省略する。

### Priority 5: LLM supervisor ranking `[A]`

周辺文脈から候補を順位づける。既存のフォールバック経路。

### Priority 6: User confirmation `[B]`

confidence が低い場合に確認 UI へ。セクション 6 参照。

---

## 6. 低信頼度の場合に silent に決めるのは危険である `[B]`

特に次の notation は、誤解釈すると計算結果が根本的に変わる。

| 記法 | 主な候補 |
|---|---|
| `'` (prime) | derivative / transpose / creation operator |
| `\Delta` | Laplacian / difference operator / gap function / Greek delta |
| `[A,B]` | commutator / interval / list |
| `{A,B}` | anticommutator / set / pair |
| `\|A\|` | absolute value / determinant / norm |
| `E[X]` | expectation value / energy function |
| `H(X)` | entropy / Hamiltonian |
| `i` | imaginary unit / index variable / electric current |
| `\sigma` | standard deviation / Pauli matrix / surface charge density / conductivity |
| `\mu` | mean / chemical potential / magnetic permeability / Lorentz index |
| `\lambda` | eigenvalue / wavelength / Lagrange multiplier |

**確認 UI を発動する条件:**

```
confidence < threshold
AND SymbolContext に宣言なし
AND Previous usage なし
AND Active domain が一意に候補を絞れない
```

**確認 UI を発動しない条件（いずれか1つを満たせば）:**

- Priority 1 で SymbolContext に宣言済み
- Priority 4 で Previous usage が記録済み
- Active domain が候補を1つに絞れる（例: `econometrics` で `X'` は常に transpose）

`i` が現れるたびに確認 UI を出すような実装は避けること。

---

## 7. 共通 profile と domain-specific override の設計 `[B]`

ドメインごとにすべての notation rule を書くと重複が多い。以下のクラスター継承を設ける。

### クラスター定義（現行 `domains/` のドメイン名に対応）

**calculus_base**（共通基盤）
- 対応ドメイン: `high_school_math`, `junior_high_math`, `high_school_physics`, `differential_equations`
- 規則: prime = derivative, `\Delta` = difference operator（教育文脈）, `i` = index or imaginary unit（宣言優先）

**classical_physics**（古典物理共通）
- 継承: calculus_base
- 追加対応ドメイン: `analytical_mechanics`, `statistical_mechanics`, `fluid_dynamics`, `relativity`
- 規則: prime = time derivative（時間依存宣言あり）または derivative, `\nabla` = gradient/divergence/curl, `\Delta` = Laplacian（`\nabla^2` と同義）

**electromagnetism_profile**
- 継承: classical_physics
- 対応ドメイン: `electromagnetism`
- override: `E` は electric field（`E[X]` 期待値記法との衝突に注意）, `\sigma` = surface charge density または conductivity

**quantum_standard**（量子力学共通）
- 継承: classical_physics
- 対応ドメイン: `quantum_mechanics`, `quantum_chemistry`
- 規則: `[A,B]` = commutator, `\langle A \rangle` = expectation value, `|ψ\rangle` = ket

**second_quantization_profile**（第2量子化共通）
- 継承: quantum_standard
- 対応ドメイン: `condensed_matter`, `quantum_optics`, `quantum_field_theory`, `quantum_chemistry_sq`, `nuclear_physics`, `spin_systems`
- backend: `julia_full`
- 規則: prime = creation operator（登録済み演算子に限る）, `{A,B}` = anticommutator, `\Delta_k` = gap function 候補

**statistics_profile**
- 継承: calculus_base
- 対応ドメイン: `probability_statistics`, `mathematical_economics`, `econometrics`, `game_theory`, `mathematical_biology`, `information_theory`
- 規則: `E[X]` = expectation value, `H(X)` = entropy, `X'` = transpose, `\mu` = mean, `\sigma` = standard deviation, `\lambda` = parameter

**data:** クラスター定義は `domains/notation_profiles.json` に格納する（新設）。

---

## 8. 数学的処理が同じ場合に過剰に domain-specific rule を作ると、実装が複雑化する `[B]`

### Notation Catalog のスキーマ

各 notation エントリに以下の項目を持たせる。

```json
{
  "notation": "x_i",
  "surface_ast_node": "Subscript",
  "possible_theoretical_meanings": [
    "generalized coordinate component (analytical_mechanics)",
    "consumption of good i (econometrics)",
    "population of species i (mathematical_biology)",
    "vector component (linear_algebra)"
  ],
  "mathematical_treatment": "IndexedSymbol",
  "commutativity": "commutative",
  "shared_treatment": true,
  "backend_consequence": {
    "maxima": "x(i) または x[i]（index_lowering_style に依存）",
    "julia_full": "x[i] または QuIndex 下降"
  },
  "requires_disambiguation": false
}
```

```json
{
  "notation": "A'",
  "surface_ast_node": "PrimeExpr",
  "possible_theoretical_meanings": [
    "derivative f'(x) (calculus)",
    "transpose X' (linear_algebra, econometrics)",
    "creation operator a'(i) (second_quantization)"
  ],
  "mathematical_treatment": "context-dependent",
  "commutativity": "noncommutative（creation operator の場合）/ commutative（scalar の場合）",
  "shared_treatment": false,
  "backend_consequence": {
    "maxima": "Symbol(name='f'') として処理（derivative 解釈は未対応）",
    "julia_full": "SymbolContext 参照で creation operator に昇格"
  },
  "requires_disambiguation": true,
  "disambiguation_condition": "SymbolContext 宣言なし AND domain が候補を絞れない"
}
```

**`commutativity` フィールドの重要性:**  
`x_i` のような IndexedSymbol も、行列成分 `A_{ij}` の非可換代数では非可換になりうる。catalog に `commutativity` を明示することで、SemanticResolver が代数構造を正しく決定できる。

---

## 9. CAS 文法が UI に漏れる危険がある `[C]`

### Input Bridge（入力側）

利用者に書かせてはいけない記法:

- Maxima: `diff(f,x)`, `integrate(f,x)`, `sum(f_i, i, 1, n)`, `x[i]`
- Julia/QuantumAlgebra: `a(QuantumAlgebra.QuIndex(:i))`, `a(sumindex(1))`
- Symbolics: `Differential(x)(f)`

利用者が書くべき記法:

- `f'(x)` または `df/dx`
- `\int f(x) dx`
- `\sum_i f_i`
- `a_i^\dagger a_j`

### Display Bridge（出力側）

**Input Bridge と同程度に困難であり、同等の設計が必要。**

CAS が返す結果の canonical 表現を利用者向け notation に変換する逆方向のブリッジ。

| CAS 出力 | 表示すべき記法 |
|---|---|
| `a(QuantumAlgebra.QuIndex(:i))` | `a_i` |
| `normal_form(comm(a(1), a_adj(1)))` | `[a_1, a_1^\dagger]` |
| `_nb_E3` | `julia_repr`（`a†(1)a(1)` 等）から変換 |
| `\text{comm}(A, B)` | `[A, B]` |
| `Laplacian(phi)` | `\nabla^2 \phi` または `\Delta \phi` |

**現状の実装:**  
`BackendHandler.clean_latex()` / `postprocess_result_latex()` で部分的に処理しているが、以下が未整備:

- QuantumAlgebra の `QuIndex` → 添字記法の変換
- Symbolics の内部表現 → 数学記法の変換
- `Laplacian(phi)` → `\nabla^2 \phi` の逆変換

**P2 の実装方針:**  
`BackendHandler.canonicalize_display_latex(latex: str) -> str` を追加し、各バックエンドの display bridge をここに集約する。

---

## 10. save/load 後も意味解決結果が壊れない必要がある `[C]`

### 現状の問題

`ir_ast` は `Field(exclude=True)`（保存対象外）。ロード後は `replay_current_branch()` によって再生成される。再生時に SymbolContext が変化していると、意味解決結果が変わる可能性がある。

**具体的なリスクシナリオ:**

1. `a'(i)` を生成演算子として使ったセッションを保存
2. SymbolContext から `a` の登録が消えた状態でロードして replay
3. `a'` が creation operator ではなく Symbol として解釈される

### P2 で対処すべき項目

| 項目 | 対策 |
|---|---|
| `IndexedSymbol x_i` と `FunctionCall x(i)` の区別 | canonical 文字列が `x(i)` になるため、reload 後に `FunctionCall` として正しく復元される。問題なし |
| prime の意味（derivative vs creation operator） | SymbolContext の `statistics` が保存・復元されるため、登録演算子なら replay 後も creation operator に昇格される。問題なし（ただし SymbolContext が失われたときは危険）|
| `\Delta_k` gap function の記憶 | セクション 5 Priority 4 の `notation_usage` フィールドを Session に保存することで対処 |
| domain を変更してから reload した場合 | SymbolContext の `statistics` と `notation_usage` を canonical な正本とし、domain の変更後も上書きしない |

---

## 11. 設計対象の役割分担まとめ

| コンポーネント | 役割 | P2 での変更 | 状態 |
|---|---|---|---|
| `SharedParser` | Surface AST 生成。domain-independent。`PrimeExpr`, `BracketExpr`, `Subscript` 等 | 原則変更なし。特殊構文のみ plugin | ✅ 変更なし |
| `Surface AST` | 意味を持たない構造ノード群 | `AbsExpr`（`|A|`）ノードを追加検討 | 未実装 |
| `DomainAwareResolver` | notation profile + SymbolContext + surrounding syntax から意味を確定 | `notation_profiles.json` を参照する層を追加 | 部分実装（B-4 で LLM hint 注入済み；`SemanticResolver` への接続は未実装） |
| `notation profiles` | ドメインクラスター別の記法規則セット（継承あり） | `domains/notation_profiles.json` として新設 | ✅ **実装済み (2026-06-01)** |
| `SymbolContext` | 型・統計・`notation_usage` を保持 | `notation_usage: dict[str, str]` フィールドを追加 | ✅ **実装済み (2026-06-01)** |
| 確認 UI | 4条件をすべて満たした場合のみ発動 | 発動条件を明示的なルールとして実装 | ✅ **CLAUDE.md に明文化 (2026-06-01)** |
| Display Bridge | CAS 出力 → 人間向け記法変換 | `BackendHandler.canonicalize_display_latex()` に集約 | 未実装 (B-2) |
| save/load | SymbolContext + `notation_usage` が正本 | `notation_usage` を Session JSON に永続化 | ✅ **実装済み (2026-06-01)** |

---

## 実装順序の提案

| 優先度 | 項目 | 種別 | 状態 |
|---|---|---|---|
| ~~1~~ | ~~`SymbolContext.notation_usage` フィールド追加と永続化~~ | C（小規模） | ✅ 2026-06-01 |
| ~~2~~ | ~~確認 UI 発動条件の明示化（4条件チェック）~~ | B | ✅ 2026-06-01 |
| ~~3~~ | ~~`notation_profiles.json` 設計と全クラスター実装~~ | B | ✅ 2026-06-01 |
| 4 | Display Bridge: `BackendHandler.canonicalize_display_latex()` 骨格実装 | C | 未実装 |
| 5 | `DomainAwareResolver` への notation profile 参照接続 | B | 未実装 |
| ~~6~~ | ~~Maxima 文脈での prime-as-derivative 対応~~ | C | ✅ 2026-06-01 |
| 7 | `\|A\|` の Surface AST ノード (`AbsExpr`) 追加と disambiguation | B | 未実装 |
| 8 | `E[X]` / `H(X)` / `P(A)` の domain-aware 解決 | B | 未実装 |
