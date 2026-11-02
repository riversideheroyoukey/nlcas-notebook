# 実践ループ サンプルシナリオ：BCS ハミルトニアンと Bogoliubov 変換（Julia バックエンド検証）

> （2026-07-02: BCS ハミルトニアンの Julia バックエンド実用検証（探索的ログ）を、
> 一般操作（演算子宣言・H-1・Heisenberg 方程式・L-1 逆変換・演算子代入）のみで再構成した。
> Bogoliubov 変換は「順変換の定義 → 逆変換の生成 → 逆変換結果のハミルトニアンへの代入」という
> 3段階（L-1 線形基底変換の枠組み）で組み立てる。）

**バックエンド:** Julia（QuantumAlgebra.jl）／**ドメイン:** 凝縮系物理（condensed_matter）。

### 物理的背景

平均場 BCS ハミルトニアン（運動量 $k$、スピン $\uparrow(u)/\downarrow(d)$）：

$$H_{\rm BCS} = \xi_k\left(c^\dagger_{k\uparrow} c_{k\uparrow} + c^\dagger_{-k\downarrow} c_{-k\downarrow}\right)
+ \Delta\left(c^\dagger_{k\uparrow} c^\dagger_{-k\downarrow} + c_{-k\downarrow} c_{k\uparrow}\right)$$

Bogoliubov 変換（順変換）は Nambu スピノル形式の完全系（生成・消滅演算子すべてを含む4本）で
定義する：

$$g_{k\uparrow} = u\,c_{k\uparrow} - v\,c^\dagger_{-k\downarrow}, \qquad
g_{-k\downarrow} = u\,c_{-k\downarrow} + v\,c^\dagger_{k\uparrow}$$

$$g^\dagger_{k\uparrow} = u\,c^\dagger_{k\uparrow} - v\,c_{-k\downarrow}, \qquad
g^\dagger_{-k\downarrow} = u\,c^\dagger_{-k\downarrow} + v\,c_{k\uparrow}$$

この4本を $c_{k\uparrow},c_{-k\downarrow},c^\dagger_{k\uparrow},c^\dagger_{-k\downarrow}$ について
逆変換すると（一般操作 `build_inverse_transform`）：

$$c_{k\uparrow} = \frac{v\,g^\dagger_{-k\downarrow} + u\,g_{k\uparrow}}{u^2+v^2}, \qquad
c_{-k\downarrow} = \frac{u\,g_{-k\downarrow} - v\,g^\dagger_{k\uparrow}}{u^2+v^2}$$

これを $H_{\rm BCS}$ に代入すると、対生成項 $g^\dagger_{-k\downarrow} g^\dagger_{k\uparrow}$ の
係数が

$$-\frac{\Delta(u^2-v^2) + 2uv\xi_k}{(u^2+v^2)^2}$$

となる。これがゼロになる条件（Bogoliubov 条件、$u^2+v^2=1$ の下で）
$\Delta(u^2-v^2) = -2uv\xi_k$ を満たすように $u,v$ を選べば、$H_{\rm BCS}$ は準粒子数演算子
$g^\dagger g$ のみの対角形になる。

---

### 一般操作での組み立て方（★このシナリオの肝）

1. $c$（電子）は「電子の消滅演算子として定義して」で宣言する（`models/particles.json` の
   粒子名辞書が「電子」→フェルミオンに解決）。
2. $g$（Bogoliubov 準粒子）は特定の粒子種ではないため、「フェルミオン演算子として定義して」
   （粒子名なしの bare 形）で宣言する。「準粒子の消滅演算子として」のように粒子名辞書に
   ない語を使うと、通常のスカラー宣言に誤って解決されてしまう（粒子名辞書は素粒子・
   準粒子の固有名詞リストであり、「準粒子」という総称語は登録されていないため）。
3. $H_{\rm BCS}$ は H-1 の等式形式で、2引数演算子 $c(k,\sigma)$（$\sigma=u/d$）を使って
   直接登録する。
4. Heisenberg 方程式は「$c(k,u)$ の Heisenberg 方程式 $E_n$」で取得する。
5. Bogoliubov 順変換は、Nambu スピノルの**全4成分**（$g_{k\uparrow}$・$g_{-k\downarrow}$・
   $g^\dagger_{k\uparrow}$・$g^\dagger_{-k\downarrow}$）を H-1 の等式形式で明示的に登録する
   （ダガーの自動生成はしない設計 — [doc 18](../developer_manuals/18_linear_transform_design.md) §4）。
6. 「$E_a,E_b,E_c,E_d$ を $c(k,u),c(-k,d),c'(k,u),c'(-k,d)$ について逆変換して」
   （`build_inverse_transform`、L-1）で旧基底（$c$ 側）を新基底（$g$ 側）で表す式を得る。
7. 得られた $c(k,u)=\dots$／$c(-k,d)=\dots$ を、E-id 参照のまま
   「$E_1$ の $c(k,u)$ に $E_a$ を、$c(-k,d)$ に $E_b$ を代入する」で $H_{\rm BCS}$ に
   同時代入する（`operator_substitute` の `op_rules`。$c^\dagger$ 側は自動的に随伴として
   処理されるため、個別に代入する必要はない）。

---

### ステップ

#### ST-1: セッション開始

| 項目 | 内容 |
|---|---|
| 操作 | 新規セッション、分野：「凝縮系物理」→ Julia バックエンドが自動選択される |
| 結果 | |

---

#### ST-2: 演算子とパラメータの宣言

| 項目 | 内容 |
|---|---|
| 入力 | `c を電子の消滅演算子として定義して` |
| 入力 | `g をフェルミオン演算子として定義して` |
| 入力 | `u, v をスカラーパラメータとして定義して` |
| 入力 | `xi, Delta をスカラーパラメータとして定義して` |
| 確認 | `c`・`g` がどちらも `@fermion_ops` で宣言されること（粒子名辞書に無い語は bare フェルミオン形を使う） |
| 結果 | |

---

#### ST-3: 平均場 BCS ハミルトニアンの登録（H-1）

| 項目 | 内容 |
|---|---|
| 入力 | `H_BCS = xi * (c'(k,u) c(k,u) + c'(-k,d) c(-k,d)) + Delta * (c'(k,u) c'(-k,d) + c(-k,d) c(k,u))` |
| 確認 | `c'(k,u)` が生成演算子、`-k` 引数がサイト（運動量）ラベルとして正しく処理されること |
| 結果 | |

---

#### ST-4: Heisenberg 方程式（運動方程式、任意）

| 項目 | 内容 |
|---|---|
| 入力 | `c(k,u) の Heisenberg 方程式 E1` |
| 期待結果 | $\dfrac{dc_{k\uparrow}}{dt} = -i\xi_k\, c_{k\uparrow} - i\Delta\, c^\dagger_{-k\downarrow}$ |
| 結果 | |

---

#### ST-5: Bogoliubov 順変換の登録（Nambu スピノル全4成分、L-1 ステップ①）

| 項目 | 内容 |
|---|---|
| 入力 | `g(k,u) = u * c(k,u) - v * c'(-k,d)` |
| 入力 | `g(-k,d) = u * c(-k,d) + v * c'(k,u)` |
| 入力 | `g'(k,u) = u * c'(k,u) - v * c(-k,d)` |
| 入力 | `g'(-k,d) = u * c'(-k,d) + v * c(k,u)` |
| 確認 | 4本とも H-1 として登録され、E-id が付与されること |
| 結果 | |

---

#### ST-6: 逆変換の生成（L-1 ステップ②）

| 項目 | 内容 |
|---|---|
| 入力 | `E3, E4, E5, E6 を c(k,u), c(-k,d), c'(k,u), c'(-k,d) について逆変換して`（`build_inverse_transform`。E-id はセッションのステップ経過に依存 — ST-4 の Heisenberg 方程式を省略した場合は1つずつ前にずれる） |
| 期待結果 | $c_{k\uparrow} = \dfrac{v g^\dagger_{-k\downarrow} + u g_{k\uparrow}}{u^2+v^2}$、$c_{-k\downarrow} = \dfrac{u g_{-k\downarrow} - v g^\dagger_{k\uparrow}}{u^2+v^2}$ 等4本 |
| 確認 | 2引数演算子（`c(k,u)` 等）を含む逆変換が正しく解けること |
| 結果 | |

---

#### ST-7: ハミルトニアンへの代入（L-1 ステップ③）

| 項目 | 内容 |
|---|---|
| 入力 | `E1 の c(k,u) に E7 を、c(-k,d) に E8 を代入する`（`E7`・`E8` は ST-6 で得た $c(k,u)=\dots$／$c(-k,d)=\dots$ の E-id。実際の番号はセッションのステップ経過に依存） |
| 確認 | 演算子の同時二重置換（`op_rules`）が E-id 参照のまま正しく適用され、結果が正規順序で整理されること |
| 期待結果 | $g^\dagger g$ の対角項に加え、対生成項 $g^\dagger_{-k\downarrow} g^\dagger_{k\uparrow}$ の係数が $-\dfrac{\Delta(u^2-v^2)+2uv\xi_k}{(u^2+v^2)^2}$ になる |
| 結果 | |

---

### 検証チェックリスト

- [x] `c`（電子）・`g`（準粒子、bare フェルミオン形）がどちらも正しくフェルミオン宣言される
- [x] 2引数演算子 $c(k,\sigma)$ の H-1 登録・表示が正しい
- [x] Heisenberg 方程式が主要 BCS 項を正しく返す
- [x] Nambu スピノル全4成分の順変換登録 → `build_inverse_transform` による逆変換が2引数演算子でも正しく解ける
- [x] 逆変換結果を E-id 参照のまま `operator_substitute`（`op_rules`）でハミルトニアンに同時代入でき、対生成項係数 $-(\Delta(u^2-v^2)+2uv\xi_k)/(u^2+v^2)^2$ が得られる（Bogoliubov 条件の起点）

---

## 確認された挙動・注意点

このシナリオを「順変換の定義 → 逆変換の生成 → 代入」という3段階（L-1 の枠組み）で組み立て
直す過程で、**4つの根本バグ**を発見・修正した。いずれも「2引数演算子（`c(k,u)` のようなスピン
添字つき演算子）を使う L-1 逆変換」という、既存テストが踏んでいなかった経路で初めて顕在化した。

| # | 症状 | 真因 | 修正 |
|---|---|---|---|
| 1 | `build_inverse_transform` が `@variables g(k u) g(-k d) ...` という壊れたコードを生成し `ParseError` | `_build_inverse_transform`（`julia_backend_handler.py`）が `vars`/`new_basis` 引数を素朴な `.split(",")` で分割しており、`c(k,u)` 自身が持つ**内部のカンマ**まで分割していた（単一引数演算子 `c(1)` にはこの内部カンマが無いため、既存テストでは踏まなかった） | 括弧の深さを追跡してトップレベルのカンマだけで分割する `_split_top_level_commas`（`julia_handler_common.py` に新設）に置き換え |
| 2 | 上記修正後も `@syms g'` エラー、および `nbA_c_k_u` と `nbA_c_k__u` のようにサロゲート変数名が食い違う | (a) 新基底（`decl_syms`）がサロゲート変換されないまま生の演算子呼び出し構文で `@variables` に渡っていた　(b) `_operator_surrogates` が同じ演算子 `c(k,u)` を「コンマの後にスペース有り／無し」で別々の文字列とみなし、別々のサロゲート名を生成していた | (a) 新基底もサロゲート置換を適用　(b) `_operator_surrogates` が引数内部の空白を正規化してからサロゲート名を生成するよう修正 |
| 3 | `E1 の c(k,u) に E6 を代入して`（単一置換で正しくルーティングされるべき文）が `ui/interpret_pipeline.py` の汎用単一代入ルール（`substitute_into`、`_DAINYU_RE`）に横取りされ、`c(k,u)` の第2引数 `u` がスカラー変数 `u`（Bogoliubov 係数）と衝突して `QuIndex(::Num)` エラー | `_DAINYU_RE` の「Em の＜任意の非空白文字列＞に」という寛容な形が、`c(k,u)` のような**演算子呼び出し＋添字**を単なる修飾語として無視し、添字を無視する全演算子一括置換（`operator_identity`）に回していた | `の＜...＞に` の部分が演算子呼び出しの形（`name(...)`）のときは `_DAINYU_RE` をスキップし、添字を尊重する `operator_substitute` に処理を譲るガードを追加 |
| 4 | `operator_substitute` の `op_rules`（同時二重置換）で `rhs` に E-id（例:`"E6"`）を書いても展開されず、文字通り未知の記号として代入されてしまう。展開自体を直しても、今度は展開結果に H-1 の「LHS = RHS」形が混入し `MethodError: no method matching *(::Num, ::Equation)` | (a) `_make_build_ctx` の引数コンパイルは、リスト引数の**文字列要素**は E-id 展開するが、`op_rules` は**辞書のリスト**なので展開ロジックが素通りしていた　(b) E-id 展開後の canonical が `engine.run_operation` 側で（`_resolve_equation_source` 用に）「LHS = RHS」形へ再構成されており、それをそのまま代入式としてパースすると `Equation` ノードになってしまう | (a) リスト内の辞書要素についても、E-id にマッチする文字列フィールドを展開する処理を追加　(b) 展開後に `Equation` 型であれば `.rhs` だけを取り出す処理を追加（`main_ir` に対する既存の同種処理と対称） |

**「準粒子の消滅演算子として定義して」は粒子名辞書に無くスカラー宣言に誤って解決される
（バグではなく、粒子名辞書の設計上の性質）。** `models/particles.json` は電子・フォノン・
マグノン等の**固有の粒子・準粒子名**のリストであり、「準粒子」という総称語はどの粒子とも
マッチしないため `notebook/runtime/julia_backend_handler.py` の `extra_pre_phase1` が
`matched=None` で通常の型宣言フォールバック（`super().extra_pre_phase1`）に流れ、結果として
`@variables g`（スカラー）が発行されていた。これは Bogoliubov 変換のように**特定の粒子種を
持たない汎用フェルミオン演算子**を導入する場合の正しい書き方が「$X$ をフェルミオン演算子
として定義して」（粒子名なしの bare 形、`_BARE_QUANTUM_OP_RE` で処理）であることを示す実例。

**確認日:** 2026-07-02（実機 Julia、generator 経由で全7ステップ完走。上記4件のバグ修正はこの
過程で発見・修正済み）。

---

## generator コマンド（この `scenario` ブロックが生成の単一ソース）

```scenario domain=condensed_matter
@note: BCSハミルトニアンを組み立てるための演算子・パラメータを準備する。$c$ は運動量 $k$・スピン $\sigma$（$u=\uparrow$, $d=\downarrow$）を持つ電子の消滅演算子。$g$ はこのあと Bogoliubov 変換で導入する準粒子の消滅演算子（特定の粒子種を持たないため、電子とは別に汎用フェルミオン演算子として宣言する）。$u, v$ は Bogoliubov 変換の係数（後で $u^2+v^2=1$ などの条件を課す）。$\xi_k$（xi）は電子の運動エネルギー（化学ポテンシャルからの相対値）、$\Delta$（Delta）はクーパー対を結びつけるペアポテンシャル（超伝導ギャップ）を表すスカラーパラメータ。
c を電子の消滅演算子として定義して
g をフェルミオン演算子として定義して
u, v をスカラーパラメータとして定義して
xi, Delta をスカラーパラメータとして定義して
@note: 平均場BCSハミルトニアンを登録する。第1項 $\xi_k(c^\dagger_{k\uparrow}c_{k\uparrow}+c^\dagger_{-k\downarrow}c_{-k\downarrow})$ は電子の運動エネルギー、第2項 $\Delta(c^\dagger_{k\uparrow}c^\dagger_{-k\downarrow}+c_{-k\downarrow}c_{k\uparrow})$ はクーパー対の生成・消滅を表すペアポテンシャル項。
H_BCS = xi * (c'(k,u) c(k,u) + c'(-k,d) c(-k,d)) + Delta * (c'(k,u) c'(-k,d) + c(-k,d) c(k,u))
@note: 運動方程式（ハイゼンベルク方程式 $dc/dt=i[H,c]$）を $c(k,u)$ について求め、この演算子が H_BCS とどのように結びついているかを確認する。
c(k,u) の Heisenberg 方程式 E1
@note: Bogoliubov 変換（準粒子への基底変換）を、電子演算子とその随伴すべてを含む Nambu スピノル形式の4本まとめて定義する（ダガーは自動生成しない設計のため、生成演算子側の式も明示的に書く）。
g(k,u) = u * c(k,u) - v * c'(-k,d)
g(-k,d) = u * c(-k,d) + v * c'(k,u)
g'(k,u) = u * c'(k,u) - v * c(-k,d)
g'(-k,d) = u * c'(-k,d) + v * c(k,u)
@note: 先に定義した4本の変換式を、電子演算子 c(k,u), c(-k,d), c'(k,u), c'(-k,d) について解き直し（逆変換）、準粒子演算子gで電子演算子を表す式を得る。
E3, E4, E5, E6 を c(k,u), c(-k,d), c'(k,u), c'(-k,d) について逆変換して
@note: 得られた逆変換の結果を BCS ハミルトニアンへ代入し、準粒子演算子 $g$ で表されたハミルトニアンを得る。対生成項 $g^\dagger_{-k\downarrow}g^\dagger_{k\uparrow}$ の係数がゼロになるよう $u, v$ を選ぶ（Bogoliubov 条件）と、ハミルトニアンは準粒子数演算子 $g^\dagger g$ のみの対角形になる。
E1 の c(k,u) に E7 を、c(-k,d) に E8 を代入する
```
