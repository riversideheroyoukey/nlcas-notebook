# Notation Bridge Catalog Audit Report

理論屋が自然に用いる notation と各 CAS が要求する notation の差異を整理し、現在の実装状況と今後の優先順位を監査した結果です。

## 1. Notation Bridge Catalog 表

| Notation | Intended meaning | Current status | Existing implementation | Missing parts | Domain IR | Maxima lowering | Julia lowering | Display bridge | Risk if wrong | Tests present | Tests needed | Priority |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A. 添字系** | | | | | | | | | | | | |
| `x_i` | 添字付きスカラー/成分 | Implemented | `IndexedSymbol` ASTへのパースと `index_lowering_style` に基づく動的 lowering。 | なし | `IndexedSymbol` | `x(i)` | `x[i]` | `x(i) \to x_i` 復元 | low | `test_index_contraction`, `test_maxima_code_generator` | なし | P3 |
| `A_{ij}` | 行列/テンソル成分 | Implemented | 同上。MathType に応じて `A[i,j]` になる | なし | `IndexedSymbol` | `A(i,j)` または `A[i,j]` | `A[i,j]` | `A(i,j) \to A_{i,j}` 復元 | high (インデックス崩壊) | `test_index_contraction` | なし | P3 |
| `\sum_j x_j` | Dummy index sum | Implemented | `IndexSum`と縮約完全動作 | パース実装完了 | `IndexSum` | `sum(...)` | `sum(...)` | `sum \to \sum` 復元 | high | `test_notation_bridge_parsing` | 完了 | 完了 |
| `\sum_j \delta_{ij} f_j`| デルタ縮約 | Implemented | 静的縮約とパース完全動作 | 実装完了 | `IndexSum` + `KroneckerDelta` | `sum(...)` | `sum(...)` | Yes | high | `test_notation_bridge_parsing` | 完了 | 完了 |
| `\delta_{ij}` | Kronecker Delta | Implemented | `kron_delta`から`KroneckerDelta`へ | 実装完了 | `KroneckerDelta` | `kron_delta(...)` | `kron_delta(...)` | `kron_delta \to \delta` 復元 | high | `test_notation_bridge_parsing` | 完了 | 完了 |
| `\delta(x-y)` | Dirac Delta | Designed but not implemented | ASTノード (`DiracDelta`) は存在 | パース、lowering、簡約ロジック | `DiracDelta` | - | - | - | high (Kron_deltaと混同) | なし | 全体 | P2 |
| **B. 積・非可換積** | | | | | | | | | | | | |
| `2x` | スカラー乗算 | Implemented | `Juxtaposition` を経由して `Mul` に解決 | なし | `Mul` | `2*x` | `2*x` | なし (自動) | low | `test_ir_compiler` | なし | P3 |
| `xy` | スカラー乗算 | Implemented | `Juxtaposition` を経由して `Mul` に解決 | なし | `Mul` | `x*y` | `x*y` | なし (自動) | low | `test_ir_compiler` | なし | P3 |
| `A_i x_i` | 積 (成分同士) | Implemented | `Juxtaposition` -> `Mul` または `NonCommutativeMul` | アインシュタイン縮約 (Einstein sum) | `Mul` / `NonCommutative` | `*` または `.` | `*` | なし | high (積の順序破壊) | `test_index_contraction` | 縮約判定テスト | P2 |
| `a_i^\dagger a_j` | 演算子乗算 | Implemented | SymbolContext (statistics) を見て非可換積に解決 | なし | `NonCommutativeMul` | `.` | `*` (QuantumAlgebra) | なし | high (非可換性破壊) | `test_non_commutative_mul_lock` | なし | P3 |
| **C. 微分・プライム** | | | | | | | | | | | | |
| `df/dx` | 全微分 | Partially implemented | Leibniz記法の表示・展開は可能。`diff`で入力が必要 | LaTeX の分数微分記法パース | `Derivative` | `diff(f,x)` | `Differential(x)(f)` | Yes | low | `test_ir_compiler` | 分数微分パース | P2 |
| `\partial f / \partial x`| 偏微分 | Partially implemented | 同上 | LaTeX の偏微分分数記法パース | `Derivative` | `diff(f,x)` | `Differential(x)(f)` | Yes | low | `test_ir_compiler` | 分数偏微分パース | P2 |
| `\dot{q}` | 時間微分 (ドット) | Implemented | Leibniz ⇔ Dot 表示のシンボル毎の動的切替 | なし | `Derivative` | `diff(q,t)` | `Differential(t)(q)` | Yes | high (変数名として誤認) | `test_engine` | なし | P3 |
| `\ddot{q}` | 2階時間微分 | Implemented | 同上 | なし | `Derivative` | `diff(q,t,2)` | `Differential(t)^2(q)` | Yes | high | `test_engine` | なし | P3 |
| `f'(x)` | 空間微分 | Implemented | `PrimeExpr` からドメイン判定で Derivative に解決 | なし | `Derivative` | `diff(f,x)` | `Differential(x)(f)` | Yes | high (adjointと混同) | `test_prime_dynamic_resolution` | なし | P3 |
| `a'(i)` | 演算子 Adjoint | Implemented | ドメイン/MathType 判定で Adjoint に解決 | なし | `Adjoint` | `ctranspose(a(i))`| `a'(i)` | Yes | high (微分と混同) | `test_prime_dynamic_resolution` | なし | P3 |
| **D. Adjoint 等** | | | | | | | | | | | | |
| `a^\dagger` | 随伴 (演算子) | Implemented | `Adjoint` AST に解決 | なし | `Adjoint` | `ctranspose(a)` | `a'` | Yes (`\dagger` 復元) | high (累乗と混同) | `test_matrix_operations_disambiguation` | なし | P3 |
| `A^\dagger` | 随伴 (行列) | Implemented | 同上 | なし | `Adjoint` | `ctranspose(A)` | `A'` | Yes | high | `test_matrix_operations_disambiguation` | なし | P3 |
| `z^*` | 複素共役 | Implemented | `Conjugate` AST に解決。`^*` `^{*}` `^{\ast}` `^\ast` `^{\star}` は変換器入口（`_conjugate_mark`）で1つの綴りに揃える | なし | `Conjugate` | `conjugate(z)` | `conj(z)` | Yes (`*` 復元) | high | `test_complex_conjugate_input` | なし | P3 |
| `\psi^*(x)` | 複素共役（適用形） | Implemented (2026-08-09) | 上付き／下付きが引数リストを飲み込んだ形を意味解決層で読み直す。星を外した節点をそのまま既存判定に返すので添字の読みは変わらない | なし | `Conjugate(FunctionCall)` | `conjugate(psi(x))` | - | Yes | high | `test_complex_conjugate_input` | 単一/多重添字とも | P3 |
| `\varphi^*_{m,s}(x)` | 行列要素の共役 | Implemented (2026-08-09) | 同上（下付き側が飲み込む形）。以前は「多重添字の適用形」として拒否されていた——星を添字と数えていたため、単一添字でも拒否 | なし | `Conjugate` | `conjugate(phim, s (reference omitted from the public snapshot))` | 拒否（QuExpr は呼べない） | Yes | high | `test_complex_conjugate_input` | なし | P3 |
| `A^T` | 転置 | Implemented | `MathType` を判定し `Transpose` に解決 | なし | `Transpose` | `transpose(A)` | `transpose(A)`| Yes (`T` 復元) | high | `test_matrix_operations_disambiguation` | なし | P3 |
| `\bar{z}` | **名前**（共役ではない） | Implemented | H-1 の装飾名規約どおり `z_bar` という識別子。共役ではないのは意図的 | なし | Symbol | `z_bar` | `z_bar` | Yes | - | `test_complex_conjugate_input` | 変更しない | — |
| `\overline{f(x)}` | 複素共役（複合式） | Implemented (2026-08-09) | 中身が単独の名前でなければ `(inner)^*` に書き換え、共役機構に合流。括弧は釣り合いで走査 | なし | `Conjugate` | `conjugate(f(x))` | - | Yes | high | `test_complex_conjugate_input` | なし | P3 |
| **E. 交換子** | | | | | | | | | | | | |
| `[A,B]` | 交換子 | Implemented | Phase 0 ブラケットパーサによる前処理 | なし | `Commutator` | `A.B - B.A` | `comm(A,B)` | none | high | `test_julia_adapter` | なし | P3 |
| `[A,[B,C]]` | 多重交換子 | Implemented | 再帰的パースで対応 | なし | `Commutator` | 展開 | `comm` | none | high | `test_julia_adapter` | なし | P3 |
| `{A,B}` | 反交換子 | Implemented | `BracketExpr(kind="curly")` 経由 | なし | `Add` + `Mul` | 展開 | 展開 | none | high | `test_julia_adapter` | なし | P3 |
| `[a_i, a_j^\dagger]`| 添字付交換子 | Implemented | AST 再帰構築と解決 | なし | `Commutator` | 展開 | `comm` | none | high | `test_julia_adapter` | なし | P3 |
| **F. ベクトル解析** | | | | | | | | | | | | |
| `\nabla f` | 勾配 (grad) | Partially implemented | `LatexConverter` での正規表現プリパス置換 | Formal IR ノードの欠如 (`Gradient` 等) | `FunctionCall` | `grad(f)` | `grad(f)` | ? | low | `test_latex_converter` | IR E2Eテスト | P2 |
| `\nabla \cdot A` | 発散 (div) | Partially implemented | 同上 | 同上 | `FunctionCall` | `div(A)` | `div(A)` | ? | low | `test_latex_converter` | IR E2Eテスト | P2 |
| `\nabla \times A` | 回転 (curl) | Partially implemented | 同上 | 同上 | `FunctionCall` | `rot(A)` | `rot(A)` | ? | low | `test_latex_converter` | IR E2Eテスト | P2 |
| `\Delta f` | Laplacian | Partially implemented | 同上 | 同上 | `FunctionCall` | `laplacian` | `laplacian` | ? | low | `test_latex_converter` | IR E2Eテスト | P2 |
| **G. 積分・総和** | | | | | | | | | | | | |
| `\int f(x) dx` | 不定積分 | Partially implemented | `Integral` ノードは存在するがパース未実装 | LaTeX `\int` パース | `Integral` | `integrate` | `integrate` | none | low | なし | パーステスト | P1 |
| `\int_0^1 f(x) dx` | 定積分 | Designed but not implemented | なし | 積分区間属性・パース | `Integral` | - | - | - | low | なし | 全体 | P2 |
| `\int \delta(x-y)` | デルタ積分 | Designed but not implemented | なし | 縮約セマンティクス解決 | `Integral` | - | - | - | high | なし | 全体 | P2 |
| `\sum_i x_i` | 総和 | Implemented | 縮約・パース完全動作 | LaTeX `\sum` パース | `IndexSum` | `sum(...)` | `sum(...)` | Yes | high | `test_notation_bridge_parsing` | なし | 完了 |
| `\sum_{i=1}^N x_i` | 有界総和 | Designed but not implemented | なし | 境界属性・パース | `IndexSum` | - | - | - | low | なし | 全体 | P2 |
| **H. 方程式・代入** | | | | | | | | | | | | |
| `f(x)=x^2` | 関数定義 | Implemented | `Equation` ノードと状態管理 | なし | `Equation` | `=` | `~` | none | high | `test_engine` | なし | P3 |
| `x=1` | 変数代入 | Implemented | 同上 | なし | `Equation` | `=` | `~` | none | high | `test_engine` | なし | P3 |
| `E1`に`x=1`を代入| 操作(subst) | Implemented | `substitute` CAS 操作にバインド | なし | `CasOperation` | `subst` | `substitute` | none | high | `test_new_operations` | なし | P3 |
| **I. 特殊定数** | | | | | | | | | | | | |
| `\pi`, `e`, `\infty`| 基礎定数 | Implemented | 予約語マッピング | なし | `Symbol` | `%pi`等 | `pi`等 | none | low | `test_latex_converter` | なし | P3 |
| `i`, `\mathrm{i}` | 虚数・添字 | Implemented | 文脈依存解決 | なし | `Symbol` | `%i` または `i` | `im` | Yes | high | `test_imaginary_vs_index_i` | なし | P3 |
| `\hbar` | プランク定数 | Implemented | 予約語マッピング | なし | `Symbol` | `hbar` | `hbar` | none | low | `test_latex_converter` | なし | P3 |
| **J. 省略依存性** | | | | | | | | | | | | |
| `q` $\to$ `q(t)` | 時間依存変数 | Implemented | `SymbolContext` 経由の依存性補完 | なし | `FunctionCall` | `q(t)` | `q(t)` | Yes | high (定数と誤認) | `test_engine` | なし | P3 |
| `\partial_\mu \phi` | 共変微分 | Deferred | なし | テンソル解析基盤 | ? | ? | ? | ? | high | なし | 全体 | P2 |

---

## 2. 実装済み項目一覧 (Implemented)
- **添字系**: `x_i`, `A_{ij}`, `\delta_{ij}` (LaTeXパース含む完全動作)
- **積・非可換積**: `2x`, `xy`, `A_i x_i` (非縮約), `a_i^\dagger a_j` (非可換解決)
- **微分系**: `\dot{q}`, `\ddot{q}`, `f'(x)` (空間微分), `a'(i)` (Adjoint解決)
- **共役・転置等 (B4完了)**: `a^\dagger`, `A^\dagger`, `z^*`, `A^T` (MathTypeによる安全な解決機構)
- **交換子・反交換子 (B5一部完了)**: `[A,B]`, `[A,[B,C]]`, `{A,B}`, `[a_i, a_j^\dagger]` (Phase 0 パーサ)
- **方程式・代入**: `f(x)=x^2`, `x=1`, `L=T-V`, 代入操作 (`substitute`)
- **定数・省略依存性**: `\pi`, `e`, `i` / `\mathrm{i}`, `\hbar`, `\infty`, `q \to q(t)`
- **総和・積分**: `\sum_i x_i` (暗黙的縮約含む), `\int f(x) dx` (不定積分パースとバックエンド連動)

## 3. 部分実装項目一覧 (Partially implemented)

- **微分系 (`df/dx`, `\partial f / \partial x`)**: ASTとしての `Derivative` や Maxima `diff` コマンドからのパースは対応済みだが、LaTeX の分数形式 (`\frac{df}{dx}`) を直接 `Derivative` にパースする機能が未実装。
- **ベクトル解析 (`\nabla f`, `\nabla \cdot A` 等)**: `LatexConverter` の正規表現によるプリパス置換（`grad`, `div` 等の関数に変換）で動いているが、Formal Layer に専用の IR ノード（`Gradient` 等）が存在していない。

## 4. 未実装だが重要な項目一覧 (Designed but not implemented)
- **Dirac Delta と積分縮約 (`\int dy \delta(x-y) f(y)`)**: `DiracDelta` と `Integral` の AST ノードは定義されているが、解析・縮約・コード生成のセマンティクスが未結合。
- **定積分 (`\int_0^1 f(x) dx`)**: `Integral` AST に区間情報を持たせる設計が未実装。

## 5. 延期してよい項目一覧 (Deferred)
- **Einstein 縮約記法 (暗黙的総和)**: 誤判定時の式変形リスクが高いため、Phase 3 以降での明示的オプトインとして設計済み。
- **共変・反変テンソル添字 (`A^\mu_ \nu`, `\partial_\mu \phi`)**: べき乗との衝突回避やテンソル微分規則の整備が必要なため Phase 3 以降。

## 6. 実装しなくても当面実害が小さい項目一覧 (Not planned / low priority)
- ~~**`\bar{z}` (複素共役)**~~ — 2026-08-09 に決着。**単独記号の `\bar{z}` は共役ではなく名前**
  （H-1 の装飾名規約 `z_bar`）で、そこは変えない。共役として読むのは**複合式に
  かかった場合**（`\overline{f(x)}`、`\overline{a+b}`）で、これは実装した。
  「`z^*` で代替可能だから不要」という当初の判断が成り立たなかったのは、代替できるか
  どうかではなく**黙って落ちていた**から: `\overline{f(x)}` は `f(x)_bar` という無意味な
  名前になり、共役はエラーも警告もなく消えていた（`\overline{a+b}` に至っては `b` だけが
  装飾された）。書けない記法より、**書けてしまうが意味が変わる記法**のほうが危険である。

## 7. 追加すべき round-trip tests
（P1項目のラウンドトリップテストは実装済みです。今後はベクトル解析やEinstein縮約に対するテストが推奨されます。）

## 8. すぐ修正すべき P0/P1 項目
（現在 P0/P1 のタスクはすべて完了しています。`\sum`, `\delta`, `\int` のパースおよび統合は完了済みです。）

## 9. Phase 3 以降に回す項目
- 連続積分と Dirac Delta の縮約エンジン実装。
- Einstein 縮約記法（暗黙総和）の対応。
- テンソル解析（共変・反変添字）。
- ベクトル解析の Formal IR ノード化（現状のプリパス置換でも機能はしているため、後回し可能）。

## 10. 現在の実装で CAS 記法が UI に漏れる可能性がある箇所
1. **未登録変数の IndexedSymbol 化**: ユーザーが未登録の変数に対して `A_{i,j}` と入力し、それが Maxima に `A(i,j)` として渡された場合。`postprocess_result_latex` は `SymbolContext.registry` を参照して `A(i,j)` を `A_{i,j}` に戻すため、未登録変数の場合は変換されず、UI 上に `A\left(i, j\right)` のような関数呼び出しの形で漏れ出るリスクがあります。
2. **ベクトル解析関数の未定義漏れ**: `\nabla \cdot F` が `div(F)` に変換され計算された後、UI に `\mathrm{div}(F)` としてそのまま表示される可能性があります（現在 `div` を `\nabla \cdot` に戻す Display Bridge が不足しています）。
3. **境界付き総和（Bounded Sum）の出力**: 現状の Display Bridge は `sum(expr, idx)` の不定和を `\sum_{idx}` に復元しますが、Maxima が `sum(expr, idx, 1, N)` を返した場合の表示復元ロジックが不足しており、CAS の `sum(...)` 記法が漏れる可能性があります。

---
## 重点確認事項に対する結果サマリ

1. **IndexedSymbol と FunctionCall の区別**: `SymbolContext` への保存（`MathType` の永続化）により、save/load 後も安全に区別・復元されます。
2. **CAS 表現の UI 漏洩**: 概ね `postprocess_result_latex` でブロックされていますが、上記（項目10）に挙げた未登録変数やベクトル解析演算子などで漏れるリスクがあります。
3. **暗黙積の混同**: `Juxtaposition` ノードは `SymbolContext.statistics` を見て `Mul` または `NonCommutativeMul` に適切に分離されています（テスト完備）。
4. **`f'(x)` と `a'(i)` の区別**: ドメイン設定に基づいて `Derivative` と `Adjoint` に分離されています（テスト完備）。
5. **添字 `i` と虚数単位 `i` の区別**: `_{i}` は添字、`^{i x}` は虚数として文脈分離されています（テスト完備）。
6. **縮約履歴**: `ir.py` の `Expr.simplification_log` フィールドで追跡可能になっています。
7. **DiracDelta と KroneckerDelta の混同**: 別のASTノードとして厳格に分離されています。
8. **既存の Index Notation Phase 1 & 2**: 今回の B4 対応などを含め、`test_index_contraction.py` はオールグリーンであり、既存の実装は壊れていません。
