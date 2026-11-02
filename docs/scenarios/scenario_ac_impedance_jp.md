# 実践ループ サンプルシナリオ：交流回路のインピーダンスと共振条件

実践ループ・シナリオメニュー **S-n（電気工学）** の詳細化。直列 RLC 回路の複素インピーダンス
$Z=R+j\omega L+\dfrac{1}{j\omega C}$ を入力し、**実部・虚部の分離**と、**虚部 $=0$（共振条件）を解いて
共振角周波数 $\omega_0=\dfrac{1}{\sqrt{LC}}$** を導く。

**バックエンド:** standard（Maxima）／**ドメイン:** 電気工学（electrical_engineering）。

## 物理的背景

直列 RLC のインピーダンス：

$$Z=R+j\omega L+\frac{1}{j\omega C}=R+j\!\left(\omega L-\frac{1}{\omega C}\right)$$

- 実部 $\operatorname{Re}Z=R$
- 虚部（リアクタンス）$\operatorname{Im}Z=\omega L-\dfrac{1}{\omega C}$
- **共振条件** $\operatorname{Im}Z=0\ \Rightarrow\ \omega L=\dfrac{1}{\omega C}\ \Rightarrow\ \omega_0=\dfrac{1}{\sqrt{LC}}$

## このシナリオの肝：電気工学の虚数単位 `j`

**電気工学（electrical_engineering）ドメインのセッションでは `j` が虚数単位**として扱われる（C-4）。
入力に `j` をそのまま使え、Maxima では `%i` に変換される（他ドメインでは `j` は総和インデックス等の
物理的意味を保つ）。ユーザーは CAS の `%i` を意識しなくてよい。

## 手順と期待値

| ステップ | 入力 | 期待結果 |
|---|---|---|
| 1 登録 | `Z = R + j*w*L + 1/(j*w*C)`（E1） | $Z=R+j\omega L+\frac{1}{j\omega C}$（EE ドメインで `j` は虚数単位）|
| 2 実部 | `E1 の実部を求めて`（`real_part`）| $\operatorname{Re}Z=R$ |
| 3 虚部 | `E1 の虚部を求めて`（`imaginary_part`）→ E3 | $\operatorname{Im}Z=\omega L-\dfrac{1}{\omega C}$（リアクタンス）|
| 4 共振条件 | `E3 = 0` | $\omega L-\dfrac{1}{\omega C}=0$（**E3 の値を展開した明示的な等式**として表示）|
| 5 解く | 前式を `w について解いて`（`solve_equation`）| $\omega=\pm\dfrac{1}{\sqrt{LC}}$ → 共振 $\omega_0=\dfrac{1}{\sqrt{LC}}$ |

> **`E_n = X` の意味:** 左辺が既存の式 ID のとき、その**値を展開して `<値> = X` という明示的な等式**を作る。
> `E3 = 0` は `ωL−1/(ωC) = 0`（共振条件）として表示される。$X$ は 0 でも任意の値・他の式 ID でもよい。
> ノートでは「式に値をあらわに与えて表示する」こと自体に意味がある（解くだけなら式 E3 を直接
> `w について解く`＝`=0` とみなす、でも可）。

> **ポイント（2026-06-22 修正）:** `real_part`/`imaginary_part` は等式 `Z = RHS` の**右辺（値）に作用**する
> （`Re Z = R`、`Im Z = ωL−1/(ωC)`）。以前は等式オブジェクト全体に `realpart`/`imagpart` を適用していたため、
> `imagpart` は `imagpart(Z)=0` で偶然それらしく見え、**`realpart` は `L j w + 1/(C j w) + R = R` と壊れて**いた
> （根治：`_resolve_equation_source` の複素値オペ分岐、`_COMPLEX_VALUE_OPS`）。

## 確認された挙動・注意点

- **既存機能のみで完走**（追加機能不要）。`real_part` / `imaginary_part` / `solve_equation` と
  EE ドメインの `j→%i` 変換だけ。
- **操作名は `real_part` / `imaginary_part`**（`realpart`/`imagpart` は自然言語エイリアス）。
  等式 `Z = RHS` に対しては**右辺（値）に作用**し、`Re Z = R`・`Im Z = ωL−1/(ωC)` を返す
  （等式オブジェクト全体には適用しない＝2026-06-22 のバグ修正）。
- 共振は `Im Z = 0`。リアクタンス $\omega L-1/(\omega C)$ を 0 と置いた式を `solve_equation`
  （`equation` E-ID＋`var`）で解くと $\omega=\pm 1/\sqrt{LC}$。物理的な共振周波数は正根 $\omega_0=1/\sqrt{LC}$。
- セッションに `electrical_engineering` ドメインを設定しておくこと（`j` の虚数単位扱いの前提）。

### 虚数単位とルールベース解釈（2026-06-26 追加）

- **NL「実部を取り出して／求めて」「虚部を…」はルールベース**で `real_part`/`imaginary_part` に解決する
  （LLM 不要）。トークナイザが操作の動詞語尾（取り出して／求めて／計算して等）を剥がして操作名詞を
  露出させる（`i18n/nl_patterns.json` の `verb_tail_strip`。2026-08-27 まで
  `llm_supervisor._JP_VERB_TAIL`）。「微分を求めて」等も同様に改善。
- **`j`=虚数単位／`i`=電流**。入力 `j` は計算用に `%i` へ変換（`MaximaSemanticResolver`、ee 限定）、
  結果は**値レベルで `%i→j` に戻して表示**（`localize_imaginary_unit`）。`%i` は曖昧性ゼロのトークンなので、
  添字・上付き・指数の `i`（`V_i`, `x^i`, $e^{j\omega t}$ の `i`）は一切触れず、虚数単位だけ `j` 表示。
  ee では `i` を指数でも虚数単位に変換しない（`x^i` は i 乗のまま）。
- **添字つきの名前（`V_i` 等）を“名前”として往復させたいときは宣言する**。未宣言の添字は設計上
  インデックス扱い（総和・縮約のため）で操作時に `V(i)` になる（CLAUDE.md「添字分類」参照）。

**確認日:** 2026-06-22（実機 Maxima：Re Z=R、Im Z 等式 `0=Lw-1/(Cw)`、solve→ $w=\pm\sqrt{1/(CL)}$）。
ルールベース化・虚数単位表示は 2026-06-26 に追加（`tests/test_ac_impedance.py`）。

## generator コマンド（この `scenario` ブロックが生成の単一ソース）

```scenario domain=electrical_engineering
@note: 直列RLC回路のインピーダンスを扱うため、抵抗 R・インダクタンス L・角周波数 w（ω）・キャパシタンス C を実数のスカラーパラメータとして準備する。
R、L、w、C をスカラーとする
@note: 抵抗R・インダクタL・キャパシタCを直列接続した回路のインピーダンスは、複素数 Z=R+jωL+1/(jωC) で与えられる（j は虚数単位。この electrical_engineering ドメインでは工学の慣習に従い j を虚数単位、i を電流として扱う）。
Z = R + j*w*L + 1/(j*w*C)
@note: インピーダンス Z の実部（抵抗成分）を取り出す。
E1 の実部を求めて
@note: インピーダンス Z の虚部（リアクタンス成分）を取り出す。エネルギーを蓄えるL・Cによって生じる、電流と電圧の位相のずれの大きさを表す。
E1 の虚部を求めて
@note: 共振とは、リアクタンス（虚部）がゼロになる状態を指す。ここでは虚部の値を0と置いた等式を作る。
E3 = 0
@note: この等式を角周波数 w について解くと、共振の起きる角周波数が得られる。符号違いの2つの解のうち、物理的に意味を持つのは正の解 ω0=1/√(LC) である。
E4 を w について解いて
```
