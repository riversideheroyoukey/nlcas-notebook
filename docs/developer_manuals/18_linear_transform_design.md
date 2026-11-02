# 18. 線型基底変換と正準性（L-1）設計仕様

CAS 支援ノートにおける**一般の線型基底変換**サブシステムの設計。座標変換・基準振動・回転・
Bogoliubov 変換・スピン昇降演算子などを、単一の枠組みで「逆変換生成 → 拘束導出 → 拘束適用 →
代入 → 正準性検証」までつなぐ。利用者向けの操作例は [user_guide.md](../user_guide.md) §線型変換、
課題・残項目は [open_items.md](../../KNOWN_ISSUES.md) §L-1 を参照。

## 1. 統一原理

基底変換 `new = M·old` が**計量 $G$ を保つ**条件は

$$M^\dagger G\,M = G$$

「ノルム保存／正準（反）交換関係保存」はその特別な場合で、$G$ が分野で変わるだけ：

| 分野 | 計量 $G$ | 拘束 | 例 |
|---|---|---|---|
| フェルミ演算子 | $\mathbb 1$ | $M^\dagger M=\mathbb 1$（ユニタリ） | $u^2+v^2=1$ |
| ボース演算子 | $\mathrm{diag}(\mathbb 1,-\mathbb 1)$ | para-unitary | $u^2-v^2=1$ |
| 正規直交ベクトル | $\mathbb 1$ | $M^\top M=\mathbb 1$（直交） | 回転：拘束なし |
| 非正規直交ベクトル | Gram 行列 $G_{ij}=\langle e_i,e_j\rangle$ | $M^\top G M=G$ | $a^2-1=0$ 等 |

演算子系では $G=\eta$ を**統計（bose/fermi）から自動決定**（$[,]$ と $\{,\}$ の符号差がそのまま $\eta$ の
符号）。$\eta$ を明示的に作らず、(反)交換子の計算からそのまま $M^\dagger\eta M=\eta$ を得る。

## 2. 操作セット

すべて `ui/interpret_pipeline.py` でルールベースにインターセプトし、専用操作へディスパッチする。

| 操作 | NL トリガ | 内容 |
|---|---|---|
| `build_inverse_transform` | 「E1, E2 の逆変換を作って」「… を c_k, c_mk について逆変換して」 | 順変換系を旧基底について解く（old = M⁻¹ new） |
| （行列入力） | 「M = [[…]]、旧基底 …、新基底 … の逆変換を求めて」 | M と基底を直接与え、順変換式を内部生成して上記へ |
| `derive_canonical_constraints` | 「E1 の正準拘束を求めて」「Bogoliubov 拘束」 | $\{new_i,new_j^\dagger\}_\pm=\delta_{ij}$＋異常項を導出 |
| `verify_canonical` | 「E1 は正準か？」「para-unitary か」 | 拘束を導出し判定（無条件正準／条件付き） |
| `derive_metric_constraints` | 「E1, E2 の計量保存条件を求めて」 | 内積宣言から Gram $G$ を組み $M G M^\top-G$ を計算 |
| `apply_canonical_constraints` | 「正準拘束のもとで E3 を簡約」「E3 を E2 の拘束で簡約」 | 多項式側関係で式を畳む（ratsubst / substitute） |
| （変換の代入） | 「E2, E3 を E1 に代入して」 | 種別で `substitute`（可換）/ `operator_substitute`（非可換）へ振り分け |

### 基底の自動同定

`_infer_old_basis`：新基底＝各式の LHS、旧基底＝RHS の原子（初出順）からスカラー宣言済み係数・既知関数・
新基底名を除外。残りが式の本数と一致すれば採用、曖昧なら基底を尋ねる。演算子原子は `c(1)`,`c'(2)` を
**丸ごと**保持（`_basis_atoms_in_order`）して base 名 `c` に潰さない。

## 3. バックエンド処理

| 論点 | Maxima | Julia |
|---|---|---|
| 逆変換 solve | `solve([eqs],[vars])` | `Symbolics.solve_for` ＋ `simplify` |
| 新基底の束縛 | H-1 P2 で RHS に束縛 → `block([新基底], …)` で局所化（0=0 縮退回避） | 構造保持で自由なまま → `@variables` 宣言 |
| 演算子変数 | （演算子は Julia 側） | `c(1)` は solve 変数にできない → サロゲート `c(1)→nbA_c_1`、結果で逆写像 |
| 計量行列 | `matrix([…])` | `[… …; …]`（中立 `"1, s; s, 1"` を各々整形） |
| 随伴生成 | — | `adjoint_canonical`（`_adjoint_ir`＋codegen） |
| 拘束適用 | `ratsubst(rhs, lhs, …)`（subst は多項式側関係不可） | `Symbolics.substitute` |
| 結果表示 | bare `{\it name}` を除去し `name_{sub}` に正規化（`postprocess_result_latex` §0） | Latexify 出力 |

行列入力 `[[…],[…]]` は**バックエンド非依存の中立記法**（Maxima の `matrix(...)` でも Julia の
`[a b; c d]` でもない）。各 build_command が自前のリテラルへ整形する。

## 4. 主要な設計判断

- **計量入力は「内積を式で宣言」**（`⟨e_1,e_2⟩ = s`、H-1 登録より前にインターセプトして
  `engine.gram_inner_products` に格納。未宣言は対角1・非対角0）。行列 $G$ の直接入力は future。
- **ダガーの自動生成はしない**（演算子）。完全な逆変換は共役式も含め N 本を明示入力させ、未知演算子数と
  式数を一致させる（フル Nambu）。
- **多モードの非対角／異常項**は標準アンザッツで自動的に消え、誤符号アンザッツでは追加拘束（例 $-2uv=0$）
  として露呈する。自明（0=0）・重複は除去。
- **拘束の自動 assume は不可**：Maxima の `assume` は等式拘束を拒否する。多項式側関係は ratsubst/substitute で
  適用するのが正道（`apply_canonical_constraints`）。導出済み拘束は `canonical_constraint_ids` で追跡し、
  「正準拘束のもとで簡約」で能動再利用する。

## 5. 関連実装・テスト

- パイプライン：`ui/interpret_pipeline.py`（各 `_parse_*` ＋ディスパッチ）。
- バックエンド：`notebook/runtime/maxima_backend_handler.py`、`notebook/runtime/julia_backend_handler.py`、
  `notebook/runtime/julia_build_mixin.py`、`notebook/runtime/julia_undefined_mixin.py`（L-1x：アンダースコア
  結合フルネームの subscript を未定義扱いしない）。
- スキーマ：`schemas/common_operations.schema.json`（`build_inverse_transform` /
  `derive_metric_constraints` / `apply_canonical_constraints`）。
- テスト：`tests/test_l1_inverse_transform.py`（ユニット・パイプライン・両 build_command）、
  `tests/test_debug_repl.py`（Maxima・Julia の実バックエンド E2E）、
  `tests/test_julia_sympy_routing.py`（L-1x の未定義検出回帰）。

## 6. スコープ外（follow-up）

- 出力 LaTeX の体裁統一（Maxima 並置 vs Julia `\cdot`、`^2` vs `^{2}`、項順）。記法の漏れではなく整形差。
- 行列 $M$ 明示入力の演算子版／para-unitary 制約の数値的充足検証。

詳細な経緯と残課題は [open_items.md](../../KNOWN_ISSUES.md) §L-1 / §L-1x を参照。

## 7. 行列リテラルと対角化ツールセット（S-i, 2026-06-21）

L-1 の基底変換とは独立に、**具体的な数表行列**を扱う一連の操作を整備した。第２量子化された
量子化学の基底変換でも行列操作が現れるため、Maxima・Julia 両バックエンドで動作する。

**2次元行列リテラル：** `[a, b; c, d]`（`;`=行区切り、`,`=列区切り）。

- `;` **あり** → 行列。`;` **なし** → 交換子・リスト（`[A, B]` は交換子）。
- 中立 IR `matrix(...)`（`FunctionCall`）に解決し、Maxima は `matrix([a,b],[c,d])`、
  Julia は `[a b; c d]` へ生成。表示は `\begin{pmatrix}` で数表化。
- 実装：`SharedParser`（`;` トークン＋行収集）、各 `SemanticResolver` の `matrix` 分岐
  （行要素を交換子化せず直接解決）、`JuliaCodeGenerator` の `matrix` 分岐。

**対角化ツールセット**（すべて `_try_matrix_ops` でルールベース、LLM 非経由）：

| 操作 | NL トリガ | 内容 |
|---|---|---|
| `eigenvalues` | 「固有値を求めて」 | 固有値 |
| `eigenvectors` | 「固有ベクトルを求めて」 | 固有ベクトル |
| `diagonalize` | 「対角化して」「モード行列／変換行列」 | 正規化固有ベクトルを列に並べた**直交行列 $P$**（$P^{-1}=P^\top$） |
| `diagonal_matrix` | 「対角行列を求めて」 | 固有値を並べた**対角行列 $D$** |
| `inverse_matrix` / `transpose` / `determinant` / `trace` / `matrix_multiply` | 「逆行列／転置／行列式／トレース／行列積」 | 行列基本操作 |

これらで相似変換 $P^{-1}AP=D=\mathrm{diag}(\lambda_i)$ が一通り回る。結果は命名・代入
（`P = En` / `En を P とする`）で CAS に束縛して後続計算に再利用できる
（`_register_named_alias` / `_try_name_assignment`）。

**設計判断：**
- `diagonalize` は固有ベクトルを正規化して直交 $P$ を返す（既定）。$D$ は別操作
  `diagonal_matrix` で得る（「対角化」と「対角行列」を明確に分離）。
- 行列値操作（`diagonalize`/`eigenvalues` 等）は新しい値を返すため、E-ID 引数の結果に
  ソース式の `input_canonical` を伝播させない（`_expand_expr_ids` は `canonical` 優先、
  不透明な `_nb_` 参照のみ `input_canonical` にフォールバック）。

**関連：** シナリオ [scenario_matrix_diagonalization.md](../scenarios/scenario_matrix_diagonalization_jp.md)、
操作一覧 [operation_reference.md](../operation_reference.md)、テスト `tests/test_matrix_literal.py`、
実例 `domains/linear_algebra/examples/`。commit d382355。
