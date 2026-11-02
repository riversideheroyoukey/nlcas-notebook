# 設計メモ: 理論計算における文字添字（Index Notation）の一般化と Domain IR 拡張設計

本ドキュメントは、理論計算一般（非可換代数、テンソル解析、統計力学、電磁気学など）において、文字添字（$i, j, k, l$ など）を用いた数式表現を自然かつ厳密に扱うための **Index Notation 抽象設計** を整理したものです。

ユーザーからのフィードバックと設計レビューに基づき、より現実的で堅牢なフェーズ分けされたロードマップと設計指針を組み込んでいます。

---

## 🎯 設計目標と背景

現状のシステムでは、テンソルや行列、量子演算子の添字について、Julia/QuantumAlgebra バックエンドの制約や Maxima の仕様上、暗黙的または限定的な方法で添字を処理していました。しかし、本物の「電子研究ノート」として物理・数学の理論計算を支えるためには、ユーザーに数字添字を強制せず、一般的な学術論文と同様の **文字添字の自然な記述と厳密な代数演算** を両立させる必要があります。

### コア原則
1. **ユーザー優先の文字添字**: ユーザーは `x_i`, `A_{ij}`, `a_i`, `a^\dagger_j`, `\sum_i x_i` のような数式表記をそのまま使用できる。
2. **意味論の共通化と Backend 表現の分離**: 添字の物理的・数学的な「意味（Semantics）」は中立な Domain IR で一元管理し、各 CAS（Maxima, Symbolics, QuantumAlgebra）へのコード生成（Lowering）は、それぞれの backend に適した形式へ完全に分離・カプセル化する。
3. **既存アーキテクチャの継承**: すでに分離・独立化が完了している `BackendHandler`, `CommandBuilder`, `LatexConverter` の3層コンパイラ構造を活かし、安全かつ非破壊的にこの一般化を導入する。

---

## 📐 共通データモデル (Domain IR 拡張)

添字の代数表現をシステム内で構造化するため、以下の4つのコアモデルを Domain IR に追加・再定義します。

### 1. `IndexSymbol` (添字シンボル)
添字そのものを表す独立したノード。単なるスカラー変数と区別します。
> [!IMPORTANT]
> 添字シンボル自体に `role`（dummy/free）を静的フィールドとして固定するのではなく、**`IndexSum` などの束縛構造の中で動的に bound/free を判定・解決する方針**とします。同じ名前の添字（例: `i`）であっても、文脈によって自由添字（free）にもダミー添字（dummy）にもなり得るためです。
```python
class IndexSymbol(Expr):
    name: str       # 添字の名前 (例: "i", "j")
    domain: str     # 属する集合/対称性 (例: "discrete", "continuous")
    range: str      # 範囲 (例: "1..N", "-inf..inf")
```

### 2. `IndexedSymbol` (添字付きオブジェクト)
添字が取り付けられたスカラー、ベクトル、テンソル、または量子演算子。
```python
class IndexedSymbol(Expr):
    base: Expr                 # ベースとなる量 (例: Symbol("x"), Symbol("A"), Operator("a"))
    indices: list[IndexSymbol] # 適用されている添字のリスト (例: [IndexSymbol("i"), IndexSymbol("j")])
```

### 3. `IndexSum` (離散総和)
ダミー添字に関する総和を表すノード。明示的な `\sum` 構造からこのノードが構築され、そのスコープ内で指定された `IndexSymbol` が **bound (dummy) index** として束縛されます。
```python
class IndexSum(Expr):
    dummy_indices: list[IndexSymbol] # 和の対象となる束縛ダミー添字 (例: [IndexSymbol("i")])
    body: Expr                       # 総和の内部式 (例: IndexedSymbol("x", [IndexSymbol("i")]))
```

### 4. `KroneckerDelta` (クロネッカーデルタ)
添字同士の同一性を表現する中核ノード。
```python
class KroneckerDelta(Expr):
    left: IndexSymbol
    right: IndexSymbol
```

---

## 🗺️ アーキテクチャ構成図

入力から各バックエンドへの lowering までのデータの流れは以下のようになります。

```mermaid
graph TD
    User[ユーザーの LaTeX / 自然言語入力]
    -->|"x_i" や "a_i" のパース| Parser[SharedParser]
    -->|Surface AST: FunctionCall, Subscript| Resolver[SemanticResolver]
    
    subgraph 中立な中間表現 (Domain IR)
        Resolver -->|意味解決| Indexed["IndexedSymbol(base=Symbol('x'), indices=[IndexSymbol('i')])"]
        Resolver -->|意味解決| Sum["IndexSum(dummy_indices=[IndexSymbol('i')], body=...)"]
    end
    
    Indexed --> CodeGen[CodeGenerator]
    Sum --> CodeGen
    
    subgraph バックエンド Lowering (CodeGenerator)
        CodeGen -->|"A[i,j]"| Maxima["Maxima Code Generator (配列 / テンソル表現)"]
        CodeGen -->|"x[i]"| Symbolics["Julia/Symbolics (インデックス付き Num)"]
        CodeGen -->|"a(i)"| QA["Julia/QuantumAlgebra (QuIndex / sumindex)"]
    end
```

---

## 💬 10の検討項目に対する回答と設計指針

### 1. 現在の IR に `IndexSymbol` / `IndexedSymbol` を追加すべきか？
> [!IMPORTANT]
> **追加します。**  
> 添字を単なる「関数の引数」や「文字列の一部」として扱うと、代数的意味（縮約、共変・反変テンソルなど）の厳密なハンドリングや、LaTeX レンダリング時の `x_{i}` と `x(i)` の区別が不可能になります。
> `ir.py` に `IndexSymbol` と `IndexedSymbol` を正式なノードとして追加し、数式解釈の基本構成要素に昇格させます。

---

### 2. $x_i$, $A_{ij}$, $a_i$, $a^\dagger_j$ をどう表現するか？
すべての表現は、ベースオブジェクトの種類（`base`）と `IndexSymbol` の組み合わせとして Domain IR で一元表現します。

* **$x_i$ (添字付きスカラー)**:
  `IndexedSymbol(base=Symbol("x", kind="scalar"), indices=[IndexSymbol("i")])`
* **$A_{ij}$ (テンソル / 行列成分)**:
  `IndexedSymbol(base=Symbol("A", kind="matrix"), indices=[IndexSymbol("i"), IndexSymbol("j")])`
* **$a_i$ (量子演算子)**:
  `IndexedSymbol(base=Operator("a", kind="boson", adjoint=False), indices=[IndexSymbol("i")])`
* **$a^\dagger_j$ (量子生成演算子)**:
  `IndexedSymbol(base=Operator("a", kind="boson", adjoint=True), indices=[IndexSymbol("j")])`

添字そのもの（`IndexSymbol`）には `dummy`/`free` といった役割をあらかじめ固定せず、項の出現数や `IndexSum` といった外側の束縛構造（Binding Context）に基づいて、ランタイム時に動的に `bound` / `free` を判定する設計とします。

---

### 3. Maxima backend ではどう lowering するか？
Maxima での表現は一律の単一ルールとせず、**`MathType` や `SymbolContext`（量の定義情報）に応じて動的に lowering 表現を選択**します。

1. **配列表記 (標準スカラー / テンソル成分)**:
   `x_i` $\rightarrow$ `x[i]`、`A_{ij}` $\rightarrow$ `A[i,j]`
   （対象オブジェクトが `MathType` 上で scalar / matrix として定義されている場合）
2. **関数表記 (未定義の添字関数)**:
   `a_i` $\rightarrow$ `a(i)`
   （偏微分や陰関数依存性を持たせる必要がある文脈、または未知のシンボルで配列アクセスした際のエラーを防ぎたい場合）
3. **変数名結合 / アトム (フォールバック)**:
   `x_i` $\rightarrow$ `x_i`
   （数式処理系としての評価を完全に抑制し、単一の記号定数として扱いたい場合）

---

### 4. Julia/Symbolics backend ではどう lowering するか？
Julia/Symbolics.jl にはテンソルのネイティブな記号サポートがまだ薄いため、以下の2つのアプローチを選択的に採用します。

* **インデックス付き配列変数 (`Symbolics.variables`)**:
  `x[i]` の形式をそのまま記号配列として登録します：
  ```julia
  @variables x[1:N]
  ```
  ただし添字が「シンボル `i`」である場合は、`@variables x(t)[1:N]` やメタデータ付き `Num` として構築します。
* **関数的 Num 表現 (Lowering先)**:
  `x_i` $\rightarrow$ `x(i)`
  `x` を「添字を受け取る記号関数」として Symbolics 上で `@register_symbolic x(i)` 等で宣言し、添字付きの値を関数呼び出しの形式で下げます。Symbolics にとって最も安全で、チェインルールがそのまま動作する形式です。

---

### 5. Julia/QuantumAlgebra backend ではどう lowering するか？
QuantumAlgebra.jl は量子添字を `QuIndex` 型で厳密に管理しています。
どの Lowering 表現（`a(:i)` / `a(QuIndex(:i))` / `a(sumindex(1))`）が最も適切であるかについては、**必ず実機（Julia REPL環境）での小実験で動作確認をした上で固定**します。

> [!TIP]
> **実機挙動（想定）**:
> - 自由添字（free index）には `a(QuantumAlgebra.QuIndex(:i))` （または短縮形で定義されたヘルパー `a(i)`）が適切。
> - ダミー総和インデックス（`sumindex`）には `a(QuantumAlgebra.sumindex(1))` が適切。

---

### 6. 文字添字を通常のスカラー変数と誤認しないためのルール
数式解析中において、文字としての `i` が「添字インデックス」なのか「通常のスカラー変数 $i$」なのかを判定する厳密なルールが必要です。

1. **構文的位置 (Syntactic Position)**:
   LaTeX の `_` の中（例: `x_{i}`）にあるシンボルは、自動的に `IndexSymbol` と判定します。
2. **コンテキスト登録**:
   `SymbolContext` 上で `role="index"` として定義・登録されているシンボルは、常に `IndexSymbol` として扱います。
3. **自然言語におけるアソシエーション**:
   「$i$ を添字とする」「添字 $i, j$ に関する和」などのテキストから、LLM Supervisor または Phase 1 規則ベースで `IndexSymbol` として強制昇格させます。

---

### 7. free index と dummy index を将来どう区別するか？
アインシュタインの縮約記法のサポートや、自動縮約の安全性において、この区別は必須です。

* **動的な束縛構造解析 (Binding Structure Analysis)**:
  - `IndexSum` ノードの `dummy_indices` に登録されている `IndexSymbol` は、その総和の範囲（`body`）において **bound (dummy) index** として扱われます。
  - `IndexSum` のスコープ外にある添字や、出現回数が1回の添字は **free index** と判定されます。
  - 同じ添字名（例: `i`）であっても、`\sum_i A_{ii}` の内側の `i` は `bound`、外側の `x_i` の `i` は `free` として文脈ごとに動的に分離・解決します。

---

### 8. `IndexSum` と `KroneckerDelta` 縮約はどの段階で実装すべきか？
> [!IMPORTANT]
> **Phase 1 では自動縮約を実装しません。**  
> Phase 1 では、まずは文字添字の **「構造保持」「表示」「各バックエンドへの最小限の lowering」** の基礎を完成させます。
> 自動縮約（`IndexSum` と `KroneckerDelta` の相殺）は、**Phase 2** において安全で小さな規則（例：静的代入）から順次導入し、CAS固有の動作（QuantumAlgebra.jl 等の交換関係による動的縮約）と滑らかに連携させます。

---

### 9. 既存の `OpaqueExpr` / `secondary_repr` とどう共存させるか？
現在の Julia 実装では、高度な量子式は IR パースが難しいため `OpaqueExpr` でラップして `secondary_repr`（`julia_repr`）として格納しています。

* **共存ルール**:
  1. すべてのスカラー添字式（`x_i` 等）や標準的な微分・積分は、`ir.py` の構造化ノード（`IndexedSymbol` 等）として優先的にパース・解決されます。
  2. 解析不可能な高度な量子演算の積などに限り、フォールバックとして `OpaqueExpr` を維持します。
  3. 量子式であっても、`a(i)` のような単純な添字アクセスは `IndexedSymbol(base=Operator("a"), indices=[IndexSymbol("i")])` として構造化し、`OpaqueExpr` の領域を段階的に縮小していきます。

---

## 🗺️ 推奨ロードマップ (Phased Roadmap)

提案されたフェーズ分けに従い、以下の段階的なロードマップを策定しました。

### 🏁 Phase 1: 基礎構造と表示・最小 lowering 構築 (現在承認された目標)
* **対応ノード**: `IndexSymbol`, `IndexedSymbol`, 明示的 `IndexSum`, `KroneckerDelta` の構造保持
* **制約**: アインシュタイン自動縮約なし、クロネッカーデルタの自動縮約なし
* **LaTeX表示**: `\#_{n}` や `sumindex` を含まず、文字添字として美しくレンダリングする Pretty-print
* **Lowering**: Maxima（配列/関数）および Julia（Symbolics/QuantumAlgebra）への実機検証に基づいた最小限の lowering

### 🥈 Phase 2: 自動縮約とダミー添字制御の導入
* **縮約**: `IndexSum` + `KroneckerDelta` の基本的な静的縮約の実装
* **ダミー添字制御**: 衝突を避けるためのダミー添字の動的リネーム（アルファタイジング）
* **制約管理**: 添字の定義域（domain/range）のチェック機構

### 🥉 Phase 3: アインシュタイン記法と高度なテンソル空間
* **自動縮約**: アインシュタインの縮約記法のサポート（Domain設定による opt-in 制御）
* **物理表現**: 共変・反変（上下添字）の表現、テンソル積コンテキストの完全解決
* **連続領域拡張**: `DiracDelta` および `Integral` を用いた連続空間の sifting property 自動縮約

---

## 🧪 Phase 1 で追加すべき最小限のテスト

1. **IR構造化テスト**:
   `SharedParser` から `"x_i"` や `"A_{ij}"` が `IndexedSymbol` として正しく解釈され、`ir_ast` を構築できることの検証。
2. **明示的 `IndexSum` パーステスト**:
   `"IndexSum(x_i, i)"` または `"\sum_i x_i"` が `IndexSum` ノードに正しくマップされることの検証。
3. **LaTeX レンダリングテスト**:
   `IndexedSymbol` や `IndexSum` を LaTeX 表示した際、論文品質の美しい `x_{i}` や `\sum_{i}` が出力されることの検証。
4. **CAS Lowering 最小テスト**:
   - Maxima: `x_i` $\rightarrow$ `x[i]` (または定義に応じた `x(i)`) への変換。
   - Julia/Symbolics: `x_i` $\rightarrow$ `x(i)` への変換。
   - Julia/QuantumAlgebra: 実機での小実験で確認された最適な `a` へのインデックス適用形式の検証。

## 🛡️ 追加の保証事項および検証結果

ユーザーからのフィードバックに基づき、文字添字（Index Notation）処理の信頼性と堅牢性を固めるため、以下の5つの観点について実装の保証とテスト検証を行いました。

### 1. contract_indices / alphatize_indices の適用範囲の明文化
* **保証動作**: インデックスの自動縮約（`contract_indices`）およびダミー添字の衝突回避リネーム（`alphatize_indices`）は、**明示的な `IndexSum` の内部スコープにのみ適用**されます。
* **Einstein 縮約の非自動化**: アインシュタインの縮約記法（明示的な総和 `IndexSum` を伴わない、単なる添字の重複による暗黙的縮約）は、現時点では**意図的に自動適用しない**よう制御されています。これにより、物理式における意図しない自動変形や破壊的簡約化を防止し、研究ノートとしての厳密性を維持します。

### 2. IndexedSymbol と FunctionCall の session save/load 後も保持される区別の保証
* **課題**: Maxima backend lowering のデフォルトポリシーでは、添字付き記号 `x_i`（`IndexedSymbol`）と関数呼び出し `x(i)`（`FunctionCall`）の両方が `x(i)` になり得るため、保存・復元（save/load）した後の Notebook IR で両者を完全に区別できなくなる懸念がありました。
* **解決策と実装**:
  * セッション復元（`NotebookEngine.load`）の際、`SymbolContext` を `MaximaBackendHandler` に設定した後に `expr.ir_ast = parse_ir(canonical)` が呼び出されるのを利用し、`parse_ir()` 内部で `self._context` (SymbolContext) が利用可能な場合に `MaximaSemanticResolver().resolve()` を適用するように修正しました。
  * これにより、復元時に `x_i` は `IndexedSymbol`、`x(i)` は `FunctionCall` に正しく再解決され、区別が完全に保持されます。
* **自動テスト**: `tests/test_index_contraction.py::test_save_load_indexed_vs_function_distinction` を追加し、save/load を経ても両者の AST ノードの型（`IndexedSymbol` vs `FunctionCall`）が完全に同一であることを自動検証してパスすることを確認しました。

### 3. 自動縮約前の式と縮約後の式の扱い（履歴記録設計と将来方針）
研究ノートの履歴追跡可能性（Traceability）の観点から、元の入力構造、自動縮約などの変換履歴、および最終簡約結果のすべてを厳密に管理・追跡するための将来設計を以下のように定義します。

* **現状 (暫定設計)**:
  * 現在の実装では、`Expression.canonical` に縮約前の元の入力表現（例: `IndexSum(delta(i, j) * x_j, j)`）をテキストとして保持し、`Expression.ir_ast` に意味解決・縮約後の AST（例: `IndexedSymbol(x, [i])`）を格納しています。
  * ただし、`canonical` は元来 backend 実行および保存のためのCAS表現であり、「元の入力構造 of 正確な保存先」と見なすのは危険です。そのため、canonical だけから AST を再構築する現在の方式は暫定のものとします。

* **将来の標準設計方針 (追跡可能性の確立)**:
  1. **`raw_ir_ast` の導入**: 
     * これは「意味解決後・自動縮約前の AST」（KroneckerDelta や衝突ダミー添字を含んだ元の構造を持つ AST）として定義します。
  2. **`normalized_ir_ast` の導入 (既存の `ir_ast` からの移行)**:
     * これは「自動縮約・alphatize 適用後の normalized AST」として定義します。
     * 既存の `ir_ast` フィールドは、当面の間この `normalized_ir_ast` と同義として扱います。
  3. **`simplification_log` (簡約履歴ログ) の導入**:
     * 将来的には、`IndexSum` + `KroneckerDelta` contraction（縮約）や alphatizing（ダミー添字リネーム）の適用プロセスを記録する `simplification_log` を導入します。
     * これにより、「どのダミー添字がどのインデックスに置き換わったか」「どのクロネッカーデルタが消費されたか」「どの名前衝突が回避されたか」などの変換履歴をメタデータとして正確に保存します。
  4. **UI における可視化**:
     * フロントエンド UI では、必要に応じて「簡約前の式（元の物理的構造）」「簡約後の簡約式」「適用された自動変換のステップ（履歴）」の3つをユーザーがトグル表示したり、ツールチップで段階的に確認できる仕組みを提供します。

### 4. 非可換積（Mul）を含む IndexSum の縮約・アルファタイジングの順序保存
* **保証動作**: 量子代数演算など、非可換演算子の積 $A_j B_k$ に対するダミー添字 `j` の縮約（例: `j -> i`）やアルファタイジングが、演算子の左右順序（積の順序）を変更しないことを保証しています。
* **実装の安全性**: `index_contractor.py` における縮約・代入処理は、`Mul.args` のリスト構造のインデックスを維持したまま `substitute_symbol_or_index` をインプレースに適用（リスト内包表記）するため、元の非可換積の左右の順序関係を一切破壊しません。
* **自動テスト**: `tests/test_index_contraction.py::test_non_commutative_order_preservation` を追加し、$\sum_j \delta_{ij} A_j B_k \to A_i B_k$ および $\sum_j \delta_{ij} B_k A_j \to B_k A_i$ の両パターンで積順が厳密に維持されることを検証しパスしました。

### 5. 定義域・値域の不整合（ValueError）の安全な表示仕様
* **保証動作**: インデックスの定義域（domain/range）が不整合の際に投げられる `ValueError` は、システムのクラッシュを引き起こさず、UI や `debug_repl.py` で安全に捕捉され綺麗にレンダリングされます。
* **表示挙動**:
  * デバッグ REPL では、例外キャッチ機構により `[err] Domain mismatch...` のようなタグ付きのエラーメッセージが出力され、REPL は安全に継続します。
  * ブラウザ UI でも、パイプラインが `failed` ステータスの結果オブジェクト（`{"status": "failed", "error": "Domain mismatch..."}`）をフロントエンドに返すため、エラーダイアログやトースト表示として安全かつ適切に表示されます。
