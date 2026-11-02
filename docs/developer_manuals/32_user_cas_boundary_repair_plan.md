# 32. 利用者書式 ⇄ CAS 書式 境界の修繕（AUDIT-SUBST-INDEXED 根治）— 設計と実施記録

**状態:** **完了（2026-07-29〜30）。Phase 0 / 1a / 1c / 2 / 2b / 3 / 4 / 5 / 6 実施済み。**
実装状況と、実測が設計を変えた箇所は §8。1b と Phase 0 の計測ハーネスは意図的に未実装（理由は §8）。

**この文書は計画として書き始め、実施記録として残す。** 各 Phase の節には当初の計画がそのまま
残してあり、実測で覆った判断はその場に「【実測により撤回】」として併記してある——
**何を予想して何が違ったか**が、結論だけより次の担当者の役に立つため。
**前提文書:** audit_subst_indexed_handoff.md (reference omitted from the public snapshot)（症状・機構の引き継ぎ）、
[06_index_notation_design.md](06_index_notation_design.md)、[07_maxima_index_lowering_policy.md](07_maxima_index_lowering_policy.md)、
[10_notation_bridge_parser_resolver_concerns.md](10_notation_bridge_parser_resolver_concerns.md)

引き継ぎメモは `substitute` 単体の不具合として書かれているが、実測の結果、
**同じ根から出た黙った誤答が最低3種類あり、そのうち2つは本リポジトリの保存済みシナリオで
実際に発生している**ことが分かった。したがって本文書は局所修正（引き継ぎメモの案A/案B）を採らず、
**利用者書式と CAS 書式の境界そのものを不変条件として定義し直す**計画とする。

---

## 1. 診断 — `canonical` が3つの役割を兼ねている

システムには本来3つの別物がある。現状はこれが `Expression.canonical` 1フィールドに
同居しているため、「どれが入っているか」が式の由来によって変わる。

| 表現 | 意味 | 誰のものか | 現状の置き場所 |
|---|---|---|---|
| **利用者表現** (`user_form`) | 研究者が書いた／読む式。`V_i`, `omega_0`, `k_1` | 利用者 | 直接登録式の `canonical`、`latex`、Julia の `input_canonical` |
| **CAS コード** (`cas_code`) | バックエンドに送る文字列。`V(i)`, `omega_0`, `k(1)` | バックエンド | **どこにも保持されず、各所で場当たり的に再生成される** |
| **CAS ハンドル** (`cas_handle`) | CAS 側の名前・返値。`_nb_E3`、Maxima の出力文字列 | バックエンド | CAS 結果式の `canonical` |

`canonical` が利用者表現と CAS ハンドルを兼ねている一方、
**`cas_code` だけが一級の存在になっていない**。これが根本原因である。

その結果、同じ式が「どの経路を通ったか」で意味を変える：

| 操作 | Maxima が受け取る式本体 | 引数 | 整合 |
|---|---|---|---|
| `expand` / `simplify` / `factor` | `a * V(i) + b`（IR 由来） | — | ○ |
| `differentiate` | `a * V(i) + b`（IR 由来） | `V(i)`（IR 由来） | ○ |
| `solve_equation` | `a*V_i^2 + b`（生） | `V_i`（生） | ○（両方生） |
| **`substitute`** | **`a*V_i + b`（生）** | **`V(i)`（IR 由来）** | **✗** |

`substitute` は不整合が同一コマンド内で衝突する唯一の場所なので、そこだけが黙って壊れている。
だが**壊れているのは `substitute` ではなく境界**である。

---

## 2. 実測（2026-07-29）

以下はすべて本リポジトリ上で実行して確認した結果。計画の各判断はこの実測に依拠する。

### 2.1 症状は `V_i` に限らない — `omega_0`, `k_1`, `x_A` も同じ

未宣言の添字付き記号すべてが対象。**FREE-PARAM が自動命名する `k_1`, `k_2` は
エンジンが作る名前で、利用者が宣言する機会がない**ため、常にこの経路に落ちる。

保存済みシナリオ `sessions/S-x_damped_oscillation_v2_20260718_030254.json`
（減衰振動、docs/scenarios.md S-x）の ODE 一般解 E2 に対し、初期条件を課す
標準的な操作を実行した実測ログ：

```
declared: ['x', 'omega_0', 'gamma']          ← k_1, k_2 は未宣言
target  : E2  'x = %e^-(t*gamma)*(k_1*sin(...)+k_2*cos(...))'
free_parameters: ['k_1', 'k_2']
subst k_1=0 -> success                       ← 成功を返す
value   : '%e^-(t*gamma)*(k_1*sin(...)+k_2*cos(...))'   ← 何も変わっていない
cmd     : block([k], ratsimp(subst(k(1)=0, ...k_1*sin(...))))
```

**FREE-PARAM（docs/21）の中核操作である「一般解に初期条件を入れる」が、
自動命名パラメータに対して黙って効かない。**

### 2.2 逆方向も壊れている — 表示に CAS 書式が漏れる

未宣言の `omega_0` を含む式に `expand` をかけた実測：

```
in : omega_0^2*q + omega_0*p     表示 ω₀² q + ω₀ p
out: omega(0)^2*q+omega(0)*p     表示 \omega\left(0\right)^2 q + \omega\left(0\right) p
```

研究者のノートに **ω(0)** と表示される。これは Core Principle #1
（通常 UI に CAS 書式を出さない）の違反であり、`substitute` とは独立の不具合。
**往復（利用者書式 → CAS → 利用者書式）が閉じていない。**

なお `omega_0` を宣言すると両方向とも正しく動く：

```
declared: out canonical 'omega_0^2*q+omega_0*p'  latex '\omega_{0}^2 q+\omega_{0} p'
declared: subst omega_0=3 -> success  '9*q+3*p'
```

→ **計算の正しさが「宣言したかどうか」という、利用者に必要性を知らせていない
UI 操作に依存している。しかもどちらでもエラーは出ない。**

### 2.3 lowering は `SymbolContext` 依存 — 引き継ぎメモの案C はこれで却下できる

```
'k_B*T'      未宣言→ 'k(B) * T'      宣言済→ 'k_B * T'
'H_0 + H_1'  未宣言→ 'H(0) + H(1)'   H_0のみ宣言→ 'H_0 + H(1)'
```

正しい CAS コードは**その時点の宣言状態の関数**である。よって登録時に確定して
保存する（案C）と、後から宣言した瞬間に保存値が陳腐化する。案C は「由来で挙動が
変わる」問題を消すのではなく、**時刻依存に置き換えるだけ**。

→ **`cas_code` は保存しない。ビルド時に毎回導出する。**（§3 R0）

### 2.4 lowering はほぼ冪等 — 案A の「主リスク」は実測では小さい

引き継ぎメモが最大の懸念とした導関数ノーン形・`block()` 等は、
parse → resolve → generate を通しても**文字列が一切変わらない**：

```
'diff(q(t),t,2)        → "'diff(q(t),t,2)"        （変化なし）
'diff(q,t,2)           → "'diff(q,t,2)"           （変化なし）
q(t)                   → 'q(t)'                   （変化なし）
block([q], ratsimp(q)) → 'block([q], ratsimp(q))' （変化なし）
matrix([a,b],[c,d])    → 'matrix([a, b], [c, d])' （空白のみ）
abs(x)                 → 'abs(x)'                 （変化なし）
```

かつ検査した全ケースで `lower(lower(x)) == lower(x)`（唯一の例外が §2.5）。
**CAS 出力に対して lowering を再適用しても壊れない**ので、
`expr_lookup` に CAS コードを載せる設計（§4 Phase 3）は成立する。

### 2.5 Maxima セッションで `[...]` が交換子になる — 第3の黙った誤答（新発見）

`SemanticResolver._resolve_bracketexpr`（semantic_resolver.py:336）は
**バックエンド・分野を問わず**、要素2個の角括弧を無条件に `Commutator` にする。
Maxima セッションで `[...]` はリスト／行列リテラルなので、これは誤りである。

保存済みシナリオ `S-y_avoided_crossing`（2準位系のハミルトニアン）の
canonical をそのまま登録して実行した実測：

```
canonical: [[epsilon, Delta],[Delta, -epsilon]]
expand   -> success  '0'
simplify -> success  '0'
cmd: expand(((((epsilon)*(Delta) - (Delta)*(epsilon)))*(...)) - ...)
```

**2×2 ハミルトニアンが黙って 0 になる。** CLAUDE.md には
「Maxima uses () conversion」とあるが、実装はそうなっていない。
これは §1 の「CAS が2つあることで増えた複雑さ」が具体化した例で、
Julia/量子系の `[A,B]`＝交換子という規約が Maxima 側に漏れている。

### 2.6 保存セッション 53 式の分類（由来別）

`sessions/*.json` 16ファイルを **実エンジン（`NotebookEngine.load`）で読み込み、
各セッション自身の SymbolContext のもとで** lowering した結果。
**由来（直接登録／CAS 結果）で分けないと意味が読めない**ので分けて示す。

| 由来 | 総数 | lowering が恒等 | 変化あり |
|---|---|---|---|
| 直接登録（`cas_repr` なし） | 43 | 32 | 11 |
| CAS 結果（`cas_repr` あり） | 10 | 2 | 8 |

「変化あり」19件の内訳：

| 分類 | 件数 | 内容 | 扱い |
|---|---|---|---|
| 表記のみ・値は同じ | 12 | `160/3`→`160*3^-1`、`%e^-(t*g)`→`%e^(-1*t*g)`、`-(X)`→`-1*(X)`、空白 | Maxima 差分検査で無害と確定できる |
| 本来の lowering（直す対象） | 4 | `x_A`→`x(A)`、`k_1`→`k(1)` | 変化して**正しい**（§2.7 も参照） |
| **破壊** | 3 | `S-y` の行列リテラル → 交換子 | §2.5。Phase 2 で消える |

**破壊は §2.5 の1原因に集約されている。** これで引き継ぎメモが
「要調査・主リスク」としていた範囲は数値で確定した。

### 2.7 CAS 結果の canonical にも「宣言されていない名前」が混ざる

上表で CAS 結果 10件中 8件が変化しているのは、単なる表記差だけではない。
ODE 一般解の canonical

```
x = %e^-(t*gamma)*(k_1*sin(...)+k_2*cos(...))
```

は Maxima が返した文字列で、`k_1` は **Maxima 側に「名前」として存在する記号**
（FREE-PARAM が `%k1` から改名して注入したもの）。ところが `k_1` は
`SymbolContext` に登録されていないため、lowering は `k(1)` にしてしまう。

つまり **`Expression.free_parameters` の名前が宣言されていないこと自体が
不整合の一因**であり、これは Phase 4 とは独立に小さく直せる：
**自由パラメータを生成した時点で `SymbolContext` に登録する**。
これだけで §2.1 の `k_1 = 0` は正しく動くようになる（Phase 3 を待たずに）。

---

## 3. 目標不変条件

修繕後に成立し、テストで固定する性質。

| ID | 不変条件 |
|---|---|
| **R0** | `cas_code` は**保存しない**。`(user_form, SymbolContext)` からビルド時に毎回導出する。導出口は**1箇所**（`BackendHandler.to_backend_code`）だけ。 |
| **R1** | 冪等性：`lower(lower(x)) == lower(x)`。 |
| **R2** | 意味保存（既に正しかったものを壊さない）：**CAS が返した canonical** と、**宣言済み記号だけからなる利用者表現**については、CAS 上で `raw` と `lower(raw)` が等値（Maxima 差分検査）。<br>※「コーパス全式で等値」ではない — 未宣言記号の lowering は**意図的に意味を変える翻訳**（`V_i` は Maxima の `V_i` ではなく `V(i)` が正しい）。等値が破れた式は、raw 側が既に誤りか lowering 規約違反かのどちらかであり、Phase 0 の分類表で1件ずつ裁定する（§2.7 の `k_1` は前者）。 |
| **R3** | 往復閉包：`display(lower(x))` は**利用者が書いた記法**で表示され、それを再入力すると `lower` して同じ CAS コードになる。`V(i)` / `ω(0)` が利用者に見えることはない。 |
| **R4** | 沈黙禁止：lowering できない式は `refused` を返し、**操作は失敗する**。成功を返して何もしないことはない。 |
| **R5** | `substitute` は、左辺がどの変種でも CAS 式本体に出現しない場合、**失敗**する（理由付き）。 |
| **R6** | 角括弧の意味はバックエンド／宣言に従って決まる（Maxima=リスト/行列、Julia・量子系=交換子）。共有リゾルバが一方に決め打ちしない。 |
| **R7** | **CAS 交換可能性（Core Principle #10）**：本計画で追加するコードのうち、共有層（`expr_lowering.py` / `backend_handler.py` / `semantic_resolver.py` / `engine.py` / `notebook_bridge.py`）に **CAS 固有の import・トークン・分岐を1つも増やさない**。lowering 方針・角括弧の読み・不透明ハンドルの判定は、すべて `BackendHandler` / リゾルバの**サブクラス側**に置く。第3の CAS を足すとき、共有層は無改造で済むこと。 |

---

## 4. 修繕計画（当初のまま。実測で覆した箇所は各節に併記）

### Phase 0 — 導出口の一本化と計測基盤（土台）

**目的:** `cas_code` を一級の存在にし、blast radius を数値で固定する。

**成果物**
- 新規 `notebook/runtime/expr_lowering.py`
  ```python
  @dataclass(frozen=True)
  class Lowered:
      code: str
      status: Literal["lowered", "passthrough", "refused"]
      reason: str = ""      # refused/passthrough の理由（開発者向け・非 i18n）
  ```
- `BackendHandler.to_backend_code(text, ctx) -> Lowered` を追加
  （既定実装＝各ハンドラの parser/resolver/codegen 三点セットを通す）。
  - `MaximaBackendHandler`: 既定どおり lowering。
  - `JuliaBackendHandler`: `_nb_*` で始まる文字列は `passthrough`（CAS ハンドルなので触らない）。
    これは Backend Encapsulation Rule どおり、各サブクラスに閉じる。
  - **R7 の具体条件:** `expr_lowering.py` は `Lowered` 型と共通の流れだけを持ち、
    **CAS 固有の文字列を1つも含まない**（`_nb_` の判定も `%i` も Julia/Maxima 側）。
    第3の CAS は `BackendHandler` のサブクラスを1つ足すだけで載ること。
- 計測ハーネス（開発用スクリプト、`scripts/` 配下）：コーパス
  （`sessions/*.json` ＋ 全テスト実行中に生成される canonical ＋ シナリオ）
  を走査し、`identical / semantically-equal / differs / refused` に分類。
  意味同値の判定は**実 Maxima**への差分検査で行う。

**受入条件:** 分類表が出力でき、`differs`（意味不同値）が §2.5 由来のもの
だけであることを確認できる。**まだ本番経路は何も切り替えない。**

**規模:** 小〜中。既存挙動を一切変えないので安全。

---

### Phase 1 — 黙った誤答を止める安全弁（先に効かせる）

**目的:** 根治の完了を待たず、silent-wrong を silent でなくする。
Phase 3 以降が長引いても、この時点で「疑義が残る計算」は無くなる。

**1a. `substitute` の事後条件（R5）**
`_build_substitute` が組み立てた `_all_lhs` のどの変種も CAS 式本体に
出現しない場合、コマンドを投げずに `BuildResult(ok=False)` を返す。
判定基準は「結果が変わらなかったか」ではなく
**「左辺が式中に構文的に存在しないか」**（`subst(x=x, …)` のような
正当な no-op を誤検出しないため）。
メッセージは i18n キー（例 `engine.substitute_lhs_not_found`、ja/en 両方を
`i18n/messages.json` に追加）。

**1b. `refused` は必ず失敗になる（R4）**
Phase 0 の `to_backend_code` が `refused` を返したら、
既存の `UnknownNotationError` 経路に合流させて操作を失敗させる。
（「決して黙って誤訳しない」＝数値層テンプレートの whitelist と同じ方針。）

**1c. 自由パラメータを宣言する（§2.7）**
`free_parameters` に名前を割り当てた時点で、その名前を `SymbolContext` に
登録する（`k_1`, `k_2`, `C`, `q_0` …）。エンジンが作った名前が
「宣言されていない添字付き記号」に見えている状態を解消する。
Phase 3 を待たずに §2.1 が**正しく動く**ようになる、独立した小さな修正。

**受入条件:**
- 1a/1b: 未宣言の `V_i` への代入が **失敗**として報告される
  （この段階ではまだ成功しない。それでよい — 誤答より失敗が上位）。
- 1c: §2.1 の `k_1 = 0` が **成功し、実際に代入される**。
  FREE-PARAM のシナリオ（S-e, S-x）が緑。

**規模:** 小。**1a/1b と 1c は別コミットにする。**

---

### Phase 2 — 角括弧の読みをフック化する（R6/R7・§2.5 の修理）

**目的:** Phase 3 の前提。これを直さずに一括 lowering を入れると行列が壊れる。
同時に、**宣言済みの分離アーキテクチャへの回帰を戻す**作業でもある。

`semantic_resolver.py` のモジュール docstring は「バックエンド差はここに挙げた
フックだけ」と明記し、実際に10個のフック（`imaginary_unit_name`、
`_resolve_symbol_backend_special` 等）を列挙している。ところが
`_resolve_bracketexpr`（`semantic_resolver.py:336`）だけは**フックを持たず**、
Julia／量子系の規約（`[A,B]`＝交換子）を共有基底に直書きしている。
バグであると同時に R7 違反である。

- 角括弧の読みを**フックにする**（例 `_resolve_square_bracket` / `square_bracket_role`）。
  共有基底は「フックに訊く」だけにし、規約は一切持たない。
  - `MaximaSemanticResolver`: 既定はリスト／行列リテラル。
    両オペランドが非可換と宣言されている場合のみ交換子。
  - `JuliaSemanticResolver`: 現行どおり交換子（第2量子化・量子系）。
- `[[a,b],[c,d]]` は行列 IR ノードに解決する
  （`shared_parser.py:332` に既にある `matrix(...)` 経路と接続）。
- docstring のフック一覧に新フックを追記する（一覧が実装の唯一の索引なので、
  ここに載っていないフックは次の監査で見つからない）。

**受入条件:** §2.5 の再現ケースで `expand` が `0` ではなく行列の展開を返す。
`tests/test_bracket_normalization*` 系と第2量子化テストが緑。
共有基底に `Commutator` 化のロジックが残っていないこと。

**規模:** 中。共有リゾルバに触るため既存テストへの影響を要確認。
**関連:** open_items **AUDIT-BRACKET-BACKEND**。

---

### Phase 2b — CAS 分離をラチェットで固定する（R7）

**目的:** Phase 3・4 で共有層に手を入れる前に、**「CAS 固有コードは共有層に増えない」を
テストで固定**しておく。今やらないと、境界の作り直し中に新しい漏れが混ざっても気づけない。

**現状（2026-07-29 実測）:** 分離状態はおおむね良好で、残る漏れは3件だけ。

| | 件数 |
|---|---|
| `primary_backend ==` 分岐（共有層） | **0** |
| `isinstance(Julia/Maxima…Handler)` 判定 | **0** |
| CAS 固有 import（共有層6ファイル） | **2**（`backend_handler.py:284`、`engine.py:1949`） |
| フックなしの CAS 規約直書き | **1**（角括弧 → Phase 2 で解消） |

**作業**
1. `backend_handler.py:282-291` — 共有基底が `MaximaLatexConverter._GREEK_MAP` を
   import している。`\gamma`→`gamma` の表は CAS 中立（`%pi`/`%i` を除外している事実が
   それを示す）なので、**中立モジュールに表を置き、各コンバータがそこから
   自分の `_GREEK_MAP` を作る**向きに反転する。
2. `engine.py:1949` — `_execute_cas_command` が `MaximaResult` を直接構築している。
   中立な結果型（またはアダプタ側のファクトリ）に置き換える。
3. `BaseSemanticResolver.imaginary_unit_name` の既定 `"%i"` を廃し、
   サブクラスに宣言を強制する（新バックエンドが黙って Maxima の虚数単位を吐かないように）。
4. `tests/test_code_hygiene.py` に**分離ラチェット**を追加：上表の各カウントに上限を置く。
   1〜3 を潰したうえで **上限を 0** にする。以後の新規混入はテストで落ちる。

**受入条件:** 上表が全て 0。ラチェットが緑。第3の CAS を足す場合に
共有層が無改造で済むことが、テストで担保されている。

**規模:** 小〜中。**Phase 3 に着手する前に済ませる。**
**関連:** open_items **AUDIT-SHARED-CAS-LEAK**。

---

### Phase 3 — 順方向の一本化（`expr_lookup` が CAS コードを運ぶ）

**目的:** 「操作によって記号の意味が違う」を消す（§1 の表を全部 ○ にする）。

- `expr_lookup` の値を、Phase 0 の `to_backend_code` を通した **CAS コード**にする。
  性能のため**引数から参照されている E-id だけ**を遅延で lowering する
  （全式を毎回通さない）。

  **【2026-07-30 実測により縮小】全操作への一括適用は誤り。** `build_command` の
  冒頭で全操作に適用したところ、フルスイートで15件が落ち、うち**2件は実害**だった:
  - `poisson_bracket` が `-(p_r/m)` ではなく **`0`** を返す。ソースの `p_r` が
    `p(r)` に lowering される一方、括弧が使う**宣言済み正準対の名前は `p_r` のまま**
    なので、微分が噛み合わず 0 になる。
  - `convolve` が **Maxima をハングさせる**（30秒タイムアウト）。この builder は
    canonical テキストから生の `subst()` を組み立てるため。
  - 残り13件のうち `truncate_series` は `H_0` が `H(0)` になる（表示漏れ＝Phase 4 の
    領域）、他は `rendered_command` の表記差（`y = a*x` → `y=a * x`、意味同値）。

  **原因の一般形:** 汎用引数ループより**前で分岐する builder** は、ソースのテキストを
  「lowering されない別のテキスト」（正準対の名前・変数名・基底名・自前の `subst`）と
  組み合わせる。片側だけ lowering すると噛み合わなくなる。これらは**生のテキスト同士で
  自己整合している**（`solve_equation` が昔から動いていたのと同じ理由）。

  **よって lowering は `_build_substitute` の内側に置く**——不整合が**同一コマンド内で
  衝突する唯一の場所**だから。§1 の表で ✗ だったのはその行だけであり、そこが ○ になる。
  他の builder を揃えるには**もう片側も lowering する**必要があり、builder ごとの
  作業になる（本計画の範囲外・需要駆動）。
- 利用者表現が必要な少数の利用者（H-1 の `LHS = RHS` 復元、NL 側）向けに
  `__user__<eid>` キーを並置する（Julia の `__input__` と同じ作法）。
- `_build_substitute` は CAS コードの式本体を使う。
  **引き継ぎメモの案B（生の表層形を `_all_lhs` に足す）は入れない** —
  不整合を残したまま当たりを増やす対症療法であり、R5 の事後条件と衝突する。
- `_all_lhs` の**既存**変種（`q(t)` 形・depends 形）は残す。これは別目的
  （Maxima 側の同一物の複数綴り）であって、今回の不整合とは無関係。
- H-10 の「複合左辺は `ratsubst`」と、その直下のコメント群は**そのまま**。
- 撤去候補（コーパス検査で subsume を確認できたものだけ、1つずつ別コミット）：
  - `build_command` 冒頭の `ensure_explicit_mult` 文字列パッチ
    — 空白＝乗算を正規表現で繕っているもので、**本件と同じ根の応急処置**。
  - `_replace_dot_symbols`（`q_dot` → `'diff(q,t,1)` の文字列置換）。

  **【2026-07-30 実測により撤回】どちらも撤去できない。** 上の見込みは外れた：
  - `_replace_dot_symbols` は **IR がやらない意味写像**。lowering は
    `theta_dot` をアトムのまま残す（`m r^2 theta_dot` → `m * r^2 * theta_dot`）。
  - `ensure_explicit_mult` は **`_replace_dot_symbols` より前に走る必要がある**。
    ノーン形の直前に空白があると誤解析されるからで、
    `k 'diff(r,t,2)` は `k' * diff(r, t, 2)` に lowering される——
    クォートが `k` のプライムとして取られ、**未評価導関数が黙って消える**。
    明示 `*` を入れた `k * 'diff(r,t,2)` なら lowering は恒等。

  よって Phase 3 の受入条件から「このファイルの行数が減る」は取り下げる。
  理由は `_prepare_expr_lookup` の docstring に書いた（次に撤去を試みる人が
  最初に読む場所）。

**受入条件:** §1 の表の `substitute` 行が ○ になる（`a*V_i + b` に `V_i=5`
→ `5*a+b`）。`substitute` の由来非依存（直接登録／`expand` 経由で同結果）。
宣言の有無で結果が変わらない（`omega_0` が未宣言でも代入が効く）。
~~`maxima_backend_handler.py` の行数が減る~~（上記のとおり撤回）。

**規模:** 大。**本計画の主リスクはここに集約される。**

---

### Phase 4 — 逆方向の保証（R3・§2.2 の修理）

**目的:** 利用者に CAS 書式を見せない。

- `SymbolContext` に添字族レジストリを追加（例 `indexed_bases: dict[str, int]`、
  base → 添字個数）。`site_spins` と同じ**加算的な永続キー**なので、
  古いセッションファイルはそのまま読める。
- 記録場所は `NotebookEngine.register_expression`：登録式の `ir_ast` を走査し、
  `IndexedSymbol` の base を記録する（リゾルバに副作用を持たせない）。
  CAS 結果の canonical は `FunctionCall` に見えるため、記録は**必ず元の式から**行う。
- `BackendHandler._apply_display_bridge` は宣言済み記号に加えて
  `indexed_bases` も見る → `V(i)` → `V_i`、`omega(0)` → `omega_0` が復元される。
- **衝突時は復元しない**：同じ base が関数としても宣言されている場合
  （利用者が `V_i` と 関数 `V(i)` を同一セッションで使った場合）は復元を
  諦めて一度だけ警告する。既存の記法曖昧性 UI（docs/09）と同じ方針。
- `backend_handler.py` は衛生ラチェット 1600 行に対し現在 1599 行。
  **新規ロジックは `expr_lowering.py` 側に置く**か、理由を書いて天井を上げる。

**受入条件:** §2.2 の `expand` 後の表示が `ω₀² q + ω₀ p`。

**規模:** 中。

---

### Phase 5 — 不変条件をテストで固定

`tests/test_user_cas_roundtrip.py`（新規）でコーパス駆動テストを組む。

1. **R1 冪等**：`lower(lower(x)) == lower(x)`。
2. **R2 意味保存**：実 Maxima で `raw` と `lower(raw)` の差分が 0
   （CAS 非依存にしたい場合はマーカーで分離）。
3. **R3 往復閉包**：`lower(latex_to_canonical(display(lower(x)))) == lower(x)`。
   既存 `tests/test_roundtrip_display.py`（R-6）は
   `canonical → display → canonical` を見ているが、**利用者が書いた記法に
   戻るか**は見ていない。そこを埋める（§2.2 は R-6 が緑のまま起きた）。
4. **R4/R5 沈黙禁止**：`k_1=0`、`V_i=5` の回帰。
   **`status == success` だけを見るテストにしない** — 値を必ず検証する。
5. **エスケープハッチのラチェット**：`refused` / `passthrough` になる
   コーパス式の件数を上限固定し、下げる方向にしか動かせないようにする
   （`tests/test_code_hygiene.py` と同じ作法）。

**必ず通すこと（引き継ぎメモ §5 より）:** `tests/test_substitute_into.py`、
`test_derivative_substitution.py`、`test_silent_diff_var_alias.py`、
`test_derivative_lhs_collapse.py`、`test_derivative_lhs_restoration.py`、
`test_index_contraction.py`、`test_roundtrip_display.py`、`test_roundtrip_persistence.py`。
**最終判断はフルスイート**（3204 tests、約15分）。部分実行で判断しない。

---

### Phase 6 — 文書・課題整理

- 本文書を「計画」から「設計＋実施記録」に更新。
- `CLAUDE.md` に**「canonical の定義」**節を追加（§1 の3表現と R0）。
  今は canonical が何であるかがどこにも書かれていない。
- `docs/open_items.md`：AUDIT-SUBST-INDEXED を本文から削除して末尾アーカイブへ1行。
- `docs/developer_manual.md` に本書と引き継ぎメモを索引登録
  （`tests/test_developer_manual_index.py` は現在
  `audit_subst_indexed_handoff.md` 未登録で**赤**。ついでに解消する）。
- §2.5 の角括弧バグは、本計画が着手されない場合に備え、
  **今すぐ `open_items.md` に独立項目として登録する**（Phase 2 で消える）。

---

## 5. 却下・保留した選択肢

| 案 | 判断 | 理由 |
|---|---|---|
| 案B（`_all_lhs` に生の表層形を追加） | **却下** | 不整合を残したまま当たりを増やす対症療法。§1 の表は ✗ のまま。R5（事後条件）と衝突し、「たまたま当たる」を増やすと沈黙禁止が効かなくなる。 |
| 案C（登録時に canonical を正規化） | **却下** | §2.3 の実測により、正しい lowering は宣言状態の関数。登録時凍結は「由来依存」を「時刻依存」に置き換えるだけ。保存ファイル互換の負債も残る。 |
| 案A（`_build_substitute` の式本体も IR 経由） | **採用・一般化** | 方向は正しいが `substitute` 1箇所では足りない（§2.2、§2.5 は直らない）。Phase 3 として**境界全体**に適用する。実測（§2.4）により、メモが恐れた導関数形の破壊は起きないことを確認済み。 |
| 添字族を別名 lowering（`x_i → x__idx(i)`、docs/07 §8.2） | **保留** | `IndexedSymbol` と `FunctionCall` の曖昧性を根本から消せるが、CAS 出力・`%` 履歴・利用者が直接書いた `V(i)` に波及する。Phase 4 の `indexed_bases` で実害（表示崩れ）は消えるので、それでも衝突が残ることが実測で分かってから再検討する。 |

---

## 6. リスクと撤退条件

| リスク | 対処 |
|---|---|
| Phase 3 で既存シナリオが崩れる | Phase 0 の分類表で事前に把握。`differs`（意味不同値）が残る式は `refused` にして**失敗させる**（黙って通さない）。Phase 1 が先に入っているので、最悪でも誤答ではなく失敗になる。 |
| Julia 側の不透明 canonical (`_nb_E3`) を lowering してしまう | `to_backend_code` の `passthrough` で `JuliaBackendHandler` に閉じる。Julia は既に `__input__` で二重表現を扱っており、そこに合流させる。 |
| 性能（毎操作で lowering） | 参照された E-id のみ遅延 lowering。必要なら `(text, ctx リビジョン)` でメモ化。 |
| 衛生ラチェット | `backend_handler.py` は 1599/1600 で余裕なし → 新規ロジックは新モジュールへ。`maxima_backend_handler.py` は 2301/2350 だが Phase 3 は**撤去**が主なので減るはず。 |
| Phase 2 が量子系を壊す | 第2量子化テスト群を Phase 2 の受入条件に含める。角括弧の読みを決めるのは**リゾルバのサブクラス**なので、Julia 側は無変更で通るはず。 |

**撤退条件:** Phase 0 の計測で `differs`（意味不同値）が §2.5 以外に
系統的に出た場合、Phase 3 は中止し、Phase 1（安全弁）＋ Phase 2 ＋ Phase 4
のみで止める。その場合でも「黙った誤答」は無くなり、表示の漏れも直る。

---

## 7. 作業順序サマリ

| 順 | Phase | 内容 | 規模 | 独立コミット可 |
|---|---|---|---|---|
| 1 | 0 | `to_backend_code` 一本化＋計測ハーネス（挙動変更なし） | 小〜中 | ✓ |
| 2 | 1a/1b | 安全弁（R4/R5）。silent-wrong → 明示的失敗 | 小 | ✓ |
| 2' | 1c | 自由パラメータを宣言（§2.7）。`k_1=0` がその場で通る | 小 | ✓ |
| 3 | 2 | 角括弧の読みをフック化（§2.5 の行列破壊を修理） | 中 | ✓ |
| 3' | 2b | CAS 分離ラチェット（残り3件を潰して上限0） | 小〜中 | ✓ |
| 4 | 3 | `expr_lookup` が CAS コードを運ぶ（本丸） | 大 | ✓（撤去は1つずつ） |
| 5 | 4 | 逆方向の復元（`indexed_bases`・表示） | 中 | ✓ |
| 6 | 5 | 往復不変条件テスト＋ラチェット | 中 | ✓ |
| 7 | 6 | 文書・open_items 整理 | 小 | ✓ |

Phase 1 と Phase 2 は Phase 3 に依存しないので、Phase 3 が長引いても
先に利用者へ届く。**「疑義の残る計算」が無くなるのは Phase 1 完了時点**、
**「利用者が CAS 書式を見なくて済む」のは Phase 4 完了時点**。

---

## 8. 実装状況（2026-07-29〜30）

| Phase | 状態 | 実体 |
|---|---|---|
| 0 導出口の一本化 | **実装済み** | `notebook/runtime/expr_lowering.py`（`Lowered` / `to_backend_code`）、`BackendHandler.to_backend_code` ＋2フック、`tests/test_expr_lowering.py` |
| 0 計測ハーネス | 未着手 | §2.6 の一回きりの実測は済み。反復可能なスクリプト化は Phase 3 の直前で足りる |
| 1a substitute の安全弁 | **実装済み** | `MaximaBackendHandler._subst_lhs_may_match`、`tests/test_substitute_silence_guard.py` |
| 1b `refused` を失敗にする | **未着手（意図的）** | `to_backend_code` はまだどの操作経路からも呼ばれていない（Phase 0 は挙動不変が受入条件）。実際に `refused` が起きうるのは Phase 3 で経路に載せてから |
| 1c 自由パラメータの宣言 | **実装済み** | `EngineFreeParamsMixin._declare_free_parameters` ＋4呼び出し点（結果登録・solve 自由変数・Laplace IC ×2）、ロード時の再宣言、`tests/test_free_param_declaration.py` |
| 2 角括弧のフック化 | **実装済み** | `_square_bracket_role`（抽象フック）＋`_resolve_bracket_as_list`、両リゾルバの実装、`tests/test_bracket_backend_role.py` |
| 2b CAS 分離ラチェット | **実装済み** | 漏れ4件（＋⑤⑥）を是正、`tests/test_code_hygiene.py::test_shared_layers_have_no_cas_specific_code`（上限0）、`tests/test_cas_separation.py` |
| 3 順方向の一本化 | **実装済み（範囲を縮小）** | `expr_lowering.lower_expr_lookup` を `_build_substitute` の内側で適用、`_prepare_expr_lookup` 抽出、`tests/test_subst_indexed_root_fix.py`。**全操作への一括適用は実測で誤りと判明し撤回**（Phase 3 の節を参照） |
| 4 逆方向の復元 | **実装済み（衝突は復元せず・別名は opt-in）** | `notebook/runtime/indexed_display.py`（記録＋復元）、`SymbolContext.indexed_bases` / `called_bases`（永続・追加キー）、`index_lowering_style="index_alias"`、`tests/test_indexed_display_restore.py` |
| 5 不変条件の固定 | **実装済み** | `tests/test_user_cas_invariants.py`（保存セッション53式で R1/R2/R4、R3 は厳選＋既知ギャップの明示固定）。書く過程で R1 違反（往復のたびに `1 *` が増える）を発見・修正 |
| 6 文書整理 | **実装済み** | CLAUDE.md に「What `Expression.canonical` Is」節、open_items の本項目をアーカイブへ、本書を実施記録に |

**Phase 3 で残った非対称（意図的）。** `substitute` 以外の builder は生のテキスト同士で
自己整合したままなので、§1 の 2.2 表の「操作によって `V_i` の意味が違う」は**解消していない**。
解消していないが**黙って誤る場所は無くなった**——不整合が同一コマンド内で衝突するのは
`substitute` だけであり、そこは直った。他 builder を揃えるには片側ではなく**両側**を
lowering する必要がある（正準対の名前・変数名・基底名も）。builder ごとの作業で、
需要駆動の別項目とする。

**Phase 3 が掘り出した既存バグ2件（AUDIT-POW-DENOM）。** どちらも Phase 3 以前から
あった silent-wrong で、境界を広げたことで初めて見えた。計画が「計測すれば裁定できる」
と想定していた §2.6 の差分検査は**単独の canonical の往復**しか見ておらず、
これらは検出できなかった——CAS に評価させて初めて分かる。

- **分母のべきに括弧が付かない。** 除算は `base^-1` として生成されるが、底が `Pow` の
  とき括弧が無く、`k/r^2` が `k * r^2^-1` になっていた。`^` は右結合なので CAS は
  `k * r^(2^-1)` = `k*sqrt(r)` と読む。**`expand(k/r^2)` が `k*sqrt(r)` を返していた。**
  `1/r^2` ポテンシャルは物理で頻出する形であり、影響は小さくない。
  発見経路は中心力シナリオのエネルギー保存が 0 でなくなったこと。
- **`^` の結合方向がパーサと CAS で逆。** パーサは `a^b^c` を左結合で読み、IR は
  `(a^b)^c`、CAS は `a^(b^c)`。①を直すまで露見しなかったのは、生成側が `a^b^c` を
  無括弧で再出力し CAS が右結合で読み直していたから——**誤った AST から正しい答えが
  出ていた**。①の括弧がその隠蔽を外したので、同時に根治した。

教訓: 往復の冪等性・文字列一致だけでは足りない。**CAS に評価させる差分検査**
（`ratsimp(input - lowered) = 0`）が本命であり、Phase 5 の R2 はこの形で書く。

### Phase 4 の実測記録（2026-07-30）

計画は「`indexed_bases` で `V(i)` → `V_i` を復元、base が関数としても宣言されて
いたら復元を諦めて警告」だった。実装しながら測った結果、**設計判断が3回変わった**。

**① 復元の判断材料は CAS 出力ではない。** `V(i)` そのものからは添字か関数かを
区別できない——情報は lowering の時点で失われている。区別しているのは**由来**で、
利用者が登録した式の IR（そこでは `IndexedSymbol` と `FunctionCall` が別物）から
学習する。`SymbolContext.indexed_bases`。

**② 「宣言されていたら」では足りない。** 未宣言のまま `V_i` と `V(x)` を両方書けて
しまい、その場合 `V(x)` が `V_x` と表示された——**別物を同じ記法で見せる嘘**。
`called_bases` を足して両方の順序で検出し、衝突した base は**復元しない**
（CAS 書式が見えるのは不格好だが、嘘ではない）。代償として、衝突した base は
家族ごと復元対象から外れ、タイムラインが衝突前後で不揃いになる。

**③ 「型を訊く」ダイアログは出せない——選択肢の一方が黙って壊れる。**
利用者の提案（未宣言シンボルの型を訊く）を実装しかけて測ったところ:
- `V` を **function と宣言しても `V_i` は `V(i)` のまま**で、衝突は解消しない。
- `V_i` を **名前と宣言すると衝突は消えるが、和が壊れる**——
  `sum(V_i, i, 1, 3)` の展開が `V(1)+V(2)+V(3)` ではなく **`3*V_i`**。エラーなし。
  CLAUDE.md が「未宣言添字をアトムにするな、添字機構が壊れる」と警告していた
  とおりで、それを宣言経路で再現するところだった。

  → **安全な答えが存在しない質問は訊かない。** ダイアログ案は取り下げ。

**④ 唯一「衝突も消え、和も壊れない」のは別名 lowering（docs/07 §8.2）。**
`V_i` → `V__idx(i)`。実 Maxima で和・共存・代入・微分を確認。
ただし**既定にはできない**、理由は2つとも実測:
- 9ファイルだけで47件失敗（全体では数百件規模）。
- **保存セッションが壊れる**——旧ファイルの `V(i)` に新形 `subst(V__idx(i)=5)` は
  効かない。AUDIT-SUBST-INDEXED を保存互換の形で再生産することになる。

  → `index_lowering_style="index_alias"` として**記号ごとの opt-in** で提供。
  Julia/QuantumAlgebra には適用しない（添字が `QuIndex` 型なので衝突しない）。

**Julia への影響はゼロ。** `JuliaBackendHandler.parse_ir` は `None` を返すので
記録が走らず、`indexed_bases` は空のまま、復元も発火しない。実 Julia セッションで
確認済み。裏を返すと **Julia 側の表示漏れは Phase 4 では直っていない**（残課題）。

> **追記（2026-07-30、実測して切り分けた）。** この「残課題」は**到達不能**だった。
> Julia で `V_i` / `xi_k` は表示に届かない——`V(i)` を送った時点で CAS が
> `Sym V is not callable` で落ちる。復元すべき表示がそもそも生じない。現実的な
> 経路（H-1 の等式登録・フルネーム宣言・QA 演算子）はいずれも利用者自身の記法で
> 表示される。**実際に到達するのは表示ではなく機能の欠落**で、しかもその失敗
> メッセージが生の CAS 構文だった（原則#1違反）。拒否のしかたは根治し、
> 「動くようにするか」は open_items **JULIA-INDEXED-SCALAR** に移した。
> 皮肉なことに、そこで測った `Symbolics.@syms xi(..)` 経路は**動くが表示が
> `xi(k)` になる** ——Phase 4 が予言した表示漏れは、その道を選んだときに初めて
> 発生する。つまり Phase 4 の指摘は間違いではなく、**先回りだった**。

**Phase 3 は Phase 4 を必須にした。** 一括適用の実験で `truncate_series` の結果が
`H_0` → `H(0)` になったことから分かるとおり、lowering を広げるほど**表示に CAS 書式が
漏れる頻度が上がる**。`substitute` に絞った現状でも、代入結果の表示は lowering 後の
形（`V(i)`）になる。Phase 4（`indexed_bases` による逆方向復元）は「あとで綺麗にする」
項目ではなく、Phase 3 の対になる必須項目である。

**Phase 2 で採った判別規則。** Maxima の角括弧は「**要素が角括弧なら list**、
それ以外の2要素はこれまで通り commutator」。前者は構造だけで決まる（list の
交換子はどの CAS でも無意味）ので推測がない。後者を list に倒さなかったのは、
宣言済み行列に対する `[A, B]` が実在するワークフローだから（コード生成が `*` では
なく `.` を選ぶ — R-4/SR-4, `tests/test_matrix_commutator.py`）。
`list` 判定は**下方向に伝播させる**必要がある（`_resolve_bracket_as_list` が再帰）:
外側だけ list にしても各行が commutator に解決され、行列は `[0, 0]` に潰れる。
Julia 側は入れ子も commutator のまま（Jacobi 恒等式は通常の演算子代数）——
これがフックでなければならない理由そのもの。

**Phase 2b の追補（2026-07-30）— 「黙って片方の CAS を選ぶ既定」を全部潰した。**
最初の4件を直した後、同種の漏れが2件残っていた。どちらも**分岐ではなく既定値**なので、
「振る舞いを分岐していないから漏れではない」と一度は見送ったもの——だが危険なのは
分岐ではなく**宣言忘れが黙って通ること**の側だった。

- ⑤ `engine.py` の `_DEFAULT_BACKEND_FOR_MODE` はバックエンド名の**2つ目の台帳**で、
  `.get(mode, "maxima")` により**未登録の `SessionMode` を黙って Maxima にする**。
  原則#6 でバックエンドはセッション途中で切り替えられないので、気付いた時点で
  最初からやり直しになる。表を `backend_factory`（名前空間の宣言上の所有者）へ移し、
  **未登録モードは `ValueError`**。逆方向 `session_mode_for_backend` も同モジュールに置き、
  `_infer_session_mode` が `domains/categories.json` の `default_backend` を自分で
  `"julia_full"` と比較するのをやめた（実ドメインで検証: condensed_matter /
  quantum_field_theory / nuclear_physics → second_quantization、classical → standard）。
- ⑥ `LLMSupervisor` / `RuleBasedIntentParser` / `_make_supervisor`（bridge・debug_repl）の
  `primary_backend` 既定 `"maxima"` を**必須化**。忘れた呼び出しは Julia セッションで
  Maxima の操作集合に黙って絞られる。**呼び出し 72 箇所すべてが既に明示していた**ので
  既定は死んでおり、挙動は変わらない。

ラチェットに **backend 名リテラル**のカテゴリを追加した（既定引数・台帳・比較を検出。
`^(maxima|julia(_\w+)?)$` とアンカーするので docstring の散文は当たらない）。
対象も `llm_supervisor` / `operation_validator` / `debug_repl` まで広げ、上限はすべて0。
**「共有層に CAS 名の文字列が1つも無い」状態が固定された。**

**Phase 2b で見つかった漏れは3件ではなく4件。** §2 の grep 監査は
`primary_backend ==` を左辺固定の正規表現で見ていたため、`"maxima" == …` の
**逆順**を取りこぼしていた。ラチェットは **AST で走査**する実装にしたので、
逆順・`in (…)` 形も検出し、逆にコメント／docstring 中の言及は構造的に除外される
（この種の説明コメントは残すべきものなので、行ベースの grep では上限0にできない）。
4件目は engine の `UndefVarError` 判定＋`_nb_E<n>` 命名で、
`BackendHandler.missing_variable_expr_id` フックに移した
（復旧そのものは汎用のリプレイ機構なので `_restore_stored_expr_var` として engine 側に残す）。

**Phase 0 で確定した API。** 計画では `to_backend_code(text, ctx)` としていたが、
実装は **`handler.to_backend_code(text)`**（ctx 引数なし）。lowering は
`parse_ir` と同じくハンドラに束縛された `SymbolContext` を使う——ここで別途 ctx を
受け取れるようにすると、parse 側と generate 側で異なる文脈を渡せてしまい、
本件と同じ「2つの真実」を作り直すことになる。テストからは
`handler.set_symbol_context(ctx)` で差し替える。

**Phase 1a の適用範囲。** syntactic な不在チェックは**原子的な左辺のみ**に適用する。
`ratsubst` 経路（H-10 の複合左辺）は照合が代数的なので、符号反転・項順の違いで
verbatim には現れなくても正しく当たる。CAS 側で束縛されている記号
（H-1 P2 の `H : J*x*y`）を含む式も、Maxima が展開してから subst するため
テキストは証拠にならない——どちらも除外し、確信できる場合だけ失敗させる。

**Phase 1c の副作用（意図的）。** ロード時に、ファイル自身が記録している
`Expression.free_parameters` から宣言を復元する。これによりセッションファイルの
`quantity_registry` に `k_1`/`k_2` が増えるが、これは
「save → load → 同じ操作が同じ結果を返す」（`tests/test_roundtrip_persistence.py`
の R-6）を**満たすための復元**であって、pre-v2 の damage 修復ではない。
したがってバージョンゲートの**前**に置いてある。

---

## 9. この修繕で見つかった silent-wrong の一覧

当初の対象は1件だった。実測で6件になり、うち5件は AUDIT-SUBST-INDEXED とは
独立に存在していた既存バグである（境界を触ったことで初めて見えた）。

| # | 症状 | 発見経路 |
|---|---|---|
| 1 | 添字付き記号への `substitute` が黙って効かない | 引き継ぎ（F-13 実装中） |
| 2 | Maxima で行列リテラルが `expand` で `0` になる | 計画策定中の実測（AUDIT-BRACKET-BACKEND） |
| 3 | `expand(k/r^2)` が `k*sqrt(r)` を返す | Phase 3 実装中（AUDIT-POW-DENOM①） |
| 4 | `^` の結合方向がパーサと CAS で逆 | ③を直したら露見（AUDIT-POW-DENOM②） |
| 5 | `E_{rot}` が `E(r, o, t)` になる | 「`a_i(k)` は？」という質問から（AUDIT-SUBSCRIPT-SPLIT） |
| 6 | 境界を通すたびに式が伸びる（`1 *` が増える） | Phase 5 の冪等性テスト |

**共通するのは「エラーが出ないまま値が変わる」こと**で、いずれも
`status == success` を見るテストでは通り抜ける。だから Phase 5 の判定は
**実 CAS の差分**（`ratsimp(input - lowered) = 0`）で書いてある——
③④は文字列往復検査を何ヶ月も通り抜けていた。

未解決として登録したもの: AUDIT-INDEXED-CALL（多重添字＋引数）、
AUDIT-DERIV-DISPLAY-RT（導関数表示の再入力）、Julia 側の表示漏れ。

**追記（2026-07-30、3件に着手した結果）。** 3件とも実測から始めたところ、
**2件は起票時の記述と実物がずれていた**——境界を触ると見えるものが変わる、という
この修繕全体の教訓がそのまま繰り返された。

| 起票時 | 実測 | 結果 |
|---|---|---|
| AUDIT-DERIV-DISPLAY-RT は2種類 | **3種類**。`{q}^{(4)}` → `q^((4))` = q の4乗（4階微分が**べき乗**として読み直される）が漏れていた——起票時に見ていたのは「構造が壊れる」2件で、**値が変わる**方は数えられていなかった | **解決**（open_items アーカイブ） |
| Julia 側の表示漏れ | **到達不能**（上の Phase 4 追記）。実際の欠落は機能側で、しかも失敗メッセージが生の CAS 構文だった | 拒否を根治し、残りを **JULIA-INDEXED-SCALAR** に再起票 |
| AUDIT-INDEXED-CALL は多重添字全般 | 無区切り `A_{ij}(x)` / `T_{ijk}(x)` は**判定条件が古いままだったバグ**。AUDIT-SUBSCRIPT-SPLIT（同日・後）が「無区切りは1添字」に変えたのに、適用形の分岐が分割規則を**再掲**していた | 範囲縮小。真の多重添字のみ残り、設計判断は未着手 |

**「規則は1箇所」は Phase 2b だけの教訓ではなかった。** AUDIT-INDEXED-CALL の
縮小分は、CAS 名の重複台帳（⑤）と構造的に同じ失敗——分割規則が2箇所に書かれ、
片方だけが更新された——で、直し方も同じ（再掲をやめて**呼ぶ**）。

---

## 10. 再現手順（当時の症状／回帰確認用）

```python
# 症状1: 未宣言の添字付き記号への代入が黙って効かない
#   → Phase 1a 実装後は ValueError（理由付きの拒否）になる。
#     「正しく代入される」ようになるのは Phase 3。
from notebook.engine import NotebookEngine
with NotebookEngine(title="t") as e:
    ex = e.register_expression("a*V_i + b")
    st, val = e.run_operation("substitute",
                              {"expr": ex.display_id, "lhs": "V_i", "rhs": "5"})
    print(st.status.value, val)      # 修正前: success  'b+V_i*a'

# 症状2: 表示に CAS 書式が漏れる
with NotebookEngine(title="t") as e:
    ex = e.register_expression("omega_0^2*q + omega_0*p")
    st, _ = e.run_operation("expand", {"expr": ex.display_id})
    print(e.expressions[st.output_refs[0]].latex)
    # '\omega\left(0\right)^2 q+\omega\left(0\right) p'

# 症状3: Maxima セッションで行列リテラルが交換子になる
with NotebookEngine(title="t") as e:
    ex = e.register_expression("[[epsilon, Delta],[Delta, -epsilon]]")
    print(e.run_operation("expand", {"expr": ex.display_id})[1])   # '0'

# 症状1 の実データ版（保存済みシナリオ）→ Phase 1c で解消済み
e = NotebookEngine.load("sessions/S-x_damped_oscillation_v2_20260718_030254.json")
e.start()
# E2（ODE 一般解）に k_1 = 0 を代入
#   修正前: success、値は不変
#   修正後: success、'%e^-(t*gamma)*k_2*cos((t*sqrt(4*omega_0^2-4*gamma^2))/2)'
```
