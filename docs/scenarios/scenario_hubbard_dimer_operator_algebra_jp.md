# 実践ループ シナリオ：2サイト Hubbard 模型を、非可換演算子で最後まで計算する

> 2026-08-22 実施。**このシナリオは、答えを知っている人が TeX で書き写す
> 場面を一つも含まない。** 登録するのはハミルトニアンと、比べたい演算子の
> 定義だけで、**結果はすべて演算子代数が計算したもの**である。

**バックエンド:** Julia full（QuantumAlgebra）／**ドメイン:** 物性（condensed_matter）。

### なぜこのシナリオを足したか

姉妹シナリオ群（グリーン関数）で、記法が**書けて表示できる**ことは確かめられた
——が、量子系の段の多くは入力の正確さの確認になっていて、CAS が計算した行は
少なかった。実測（2026-08-22）: `scenario_greens_function_from_hamiltonian` で
CAS が計算した式は 30 行中 10 行、`spinful_matrix_elements` /
`tight_binding_bands` / `spin_chain_spectrum` は **0 行**。他ドメインは 60–89% で
ある。**演算子代数は本来この系のいちばん得意な場所**なので、そこが計算されて
いないのは記法の側の詰まりだった。

測ったところ、詰まっていたのは QuantumAlgebra ではなく**指示を読む側**である。
QA 自身は `{c(1,↑), c†(1,↑)} = 1` を最初から正しく返していて、次の4つが届いて
いなかった:

| 綴り | 届いていたもの |
|---|---|
| `c(1,u) と c'(1,u) の反交換子` | `b="c'(1"` → 生の `ParseError:` |
| `[c(1,u), c'(1,u)]` | `a="c(1"`, `b="u), c'(1,u)"` — **黙って違う場所で切る** |
| `c_{1\uparrow}` | 添字 `1 * u` として拒否（矢印を先に文字へ潰し、区切りを消していた） |
| `n_{1\uparrow} と n_{1\downarrow} の交換子` | `[c^†_{1↑}, c_{1↑}]` — **別の演算子を計算して成功** |

いずれも「どこが区切りか」を、括弧や助詞ではなく空白とカンマで近似していた
もので、根は一つである。

### 書いた和は、書いたとおりに計算される

初版はハミルトニアンを**長く展開して**書いていた。不自然だという指摘を受けて
測ったところ、和の記法は「使わなかった」のではなく**使うと黙って誤答していた**:

| 綴り | 実測（初回） |
|---|---|
| `\sum_{i=1}^{2}` | 範囲 1..2 が**消える**。`[U\sum_{i=1}^{2}\hat n_{i\uparrow}\hat n_{i\downarrow}, \hat c_{1\uparrow}]` が `\sum_i[U c^†_{1↓}c_{1↑}c_{1↓}]` を成功として返す——$i$ を含まない項に $\sum_i$ が掛かっている |
| `\sum_{\sigma}` | `ArgumentError("index names must be single character, got sigma.")` の**生の CAS 文**が返る |
| `\{A, B\}` | 反交換子として読まれず、ただの式として登録される |

QuantumAlgebra の $\sum$ は**形式和**で、範囲を置く場所が無い。だからといって
範囲を落としてよいことにはならない——**具体的な整数の範囲なら和は有限個の項で、
展開は機械的かつ厳密**である（このシステムは既に同じことを3箇所でしている）。
範囲が**書かれていない** `\sum_\sigma` は展開しない: そこは形式和が書かれたものと
正確に一致する。

### 矢印は値、σ は変数

`\uparrow` は QuantumAlgebra の添字にそのまま置ける（QA の規則は
「1 unicode 文字」で、矢印はそれに当たる）。そして**矢印は和で束縛できない**
——`\sum_\uparrow` とは誰も書かず、`\sum_\sigma` と書く。だから2つの異なる矢印は
2つの異なる**値**であり、`δ(↑,↓)=0` は整数サイトの `δ(1,2)=0` と同じ根拠で
落ちる。`\sigma` は文字のままにしてあるので、`\sum_\sigma` の中の `δ(s,s')` は
残る——**物理が要る区別（値か変数か）が、書かれたものから落ちてくる。**

これが無いと、下のどの交換子にも `δ_{↑↓}` の項が付いて回り、研究者が 0 と
知っているものをシステムは 0 にできなかった。

### 何が確認されるか

- **保存量が計算で出る**。$[\hat H,\hat N]=0$、$[\hat H,\hat S^z]=0$、$[\hat H,\hat S^+]=0$
- **SU(2) 代数が導出される**。$[\hat S^+,\hat S^-]=2\hat S^z$ を仮定せず計算し、
  差が 0 になることまで CAS に確かめさせる
- **保存しないものが、なぜ保存しないかの形で出る**。二重占有 $\hat D$ との交換子は
  跳び移り $w$ の4項だけになり、$U$ は消える
- **Pauli の恒等式** $\hat n^2=\hat n$ が正規順序化で出る
- **原子極限の厳密な時間発展**（PAIR-REP）:
  $\hat c_{1\uparrow}(t)=\hat c_{1\uparrow}+(1-e^{-iUt})\hat c^\dagger_{1\downarrow}\hat c_{1\uparrow}\hat c_{1\downarrow}$

```scenario domain=condensed_matter
@note: 電子の消滅演算子と、跳び移り $w$・オンサイト斥力 $U$・時間 $t$ を宣言する。
c を電子の消滅演算子として定義して
w をスカラーパラメータとして定義して
U をスカラーパラメータとして定義して
t をスカラーパラメータとして定義して
@note: **2サイト Hubbard 模型を、教科書どおりの和の記法で書く。** スピンの和 $\sum_\sigma$ は範囲が書かれていないので QuantumAlgebra の形式和にそのまま乗り、サイトの和 $\sum_{i=1}^{2}$ は**具体的な範囲があるので項に展開して**バックエンドへ渡される（QA の $\sum$ には範囲を置く場所が無い——だからといって範囲を黙って落とすのは、書いた式と違うものを計算することになる）。
H = -w \sum_{\sigma} (c^\dagger_{1\sigma} c_{2\sigma} + c^\dagger_{2\sigma} c_{1\sigma}) + U \sum_{i=1}^{2} c^\dagger_{i\uparrow} c_{i\uparrow} c^\dagger_{i\downarrow} c_{i\downarrow}
@note: 土台の反交換関係を、**括弧の記法そのまま**で聞く。同じサイト・同じスピンなら 1、サイトが違えば 0、**スピンが違っても 0**——最後の一つが、矢印を値として読めていないと $\delta_{\uparrow\downarrow}$ のまま残る。
\{c_{1\uparrow}, c^\dagger_{1\uparrow}\}
\{c_{1\uparrow}, c^\dagger_{2\uparrow}\}
\{c_{1\uparrow}, c^\dagger_{1\downarrow}\}
@note: **粒子数の保存。** 全粒子数 $\hat N=\sum_{i\sigma}\hat c^\dagger_{i\sigma}\hat c_{i\sigma}$ を書いて、$[\hat H,\hat N]$ を計算させる。$\sum_\sigma$ は**宣言した粒子**（電子）の射影 $\pm1/2$ に展開される。跳び移り項も $U$ 項も $\hat N$ を変えないので 0 になるはずである。
N = \sum_{i=1}^{2} \sum_{\sigma} c^\dagger_{i\sigma} c_{i\sigma}
[E1, N]
@note: **$S^z$ の保存。** $\hat S^z=\sum_{i\sigma}\sigma\,\hat n_{i\sigma}$ ——**教科書そのままの書き方**である。係数の位置の $\sigma$ には**値**（$\pm1/2$、厳密な有理数）が、添字の位置には**綴り**（矢印）が入る。ここも 0。
S_z = \sum_{i=1}^{2} \sum_{\sigma} \sigma c^\dagger_{i\sigma} c_{i\sigma}
[E1, S_z]
@note: **SU(2) 代数を導出する。** 昇降演算子 $\hat S^\pm=\sum_i c^\dagger_{i\uparrow}c_{i\downarrow}$（および逆向き）を書き、$[\hat S^+,\hat S^-]$ を計算させる。$2\hat S^z$ になるはずだが、それを**仮定せずに計算する**のがここの主題である。
S_p = \sum_{i=1}^{2} c^\dagger_{i\uparrow} c_{i\downarrow}
S_m = \sum_{i=1}^{2} c^\dagger_{i\downarrow} c_{i\uparrow}
[S_p, S_m]
@note: $2\hat S^z$ を書いて引かせ、恒等式であることを CAS に確かめさせる（0 になる）。
T = 2 \sum_{i=1}^{2} \sum_{\sigma} \sigma c^\dagger_{i\sigma} c_{i\sigma}
E11 から E12 を引く
@note: **$\hat H$ の SU(2) 対称性。** $[\hat H,\hat S^+]=0$。$U$ 項は $\hat n_\uparrow\hat n_\downarrow$ でスピンを反転させても不変、跳び移りはスピンを保つ。
[E1, S_p]
@note: **保存しないものを見る。** 二重占有 $\hat D=\sum_i\hat n_{i\uparrow}\hat n_{i\downarrow}$ は保存量ではない。交換子には**跳び移り $w$ の項だけ**が残り、$U$ は消える——二重占有を作り／壊すのは跳び移りだけだからである。
D = \sum_{i=1}^{2} c^\dagger_{i\uparrow} c_{i\uparrow} c^\dagger_{i\downarrow} c_{i\downarrow}
[E1, D]
@note: **運動方程式の右辺。** $i\hbar\,\partial_t\hat c_{1\uparrow}=[\hat c_{1\uparrow},\hat H]$ の芯である。跳び移りが隣のサイトへ、$U$ が反対スピンの数演算子を連れてくる。
[E1, c_{1\uparrow}]
@note: **入れ子の交換子も、書いたとおりに読まれる。** $[\hat c_{1\uparrow},[\hat c^\dagger_{1\uparrow},\hat c_{1\uparrow}]]=2\hat c_{1\uparrow}$。カンマがどこで区切るかは括弧の深さで決まるので、内側の括弧はオペランドの一部である。
[c_{1\uparrow}, [c^\dagger_{1\uparrow}, c_{1\uparrow}]]
@note: **Pauli の恒等式。** $\hat n^2$ を正規順序化させると $\hat n$ になる（同じ状態に2つ入れられない）。これも書き写しではなく代数の結果である。
c^\dagger_{1\uparrow} c_{1\uparrow} c^\dagger_{1\uparrow} c_{1\uparrow} を正規順序にして
@note: **原子極限の厳密な時間発展**（PAIR-REP、docs/34 §2）。跳び移りを切って $U$ 項だけ残すと、$\mathrm{ad}_H$ の連鎖が有限個の演算子で閉じるので、$\hat c_{1\uparrow}(t)$ が**打ち切りなしで**出る。
H_U = U c^\dagger_{1\uparrow} c_{1\uparrow} c^\dagger_{1\downarrow} c_{1\downarrow}
@note: 係数 $1-e^{-iUt}$ はスカラーで、QuantumAlgebra には置けない（c 数を cos/sin に分解する）——実体は「(係数, 演算子) の対」で、表示だけが $\hat c_{1\uparrow}(t)$ の書き方に見える。Hubbard 原子の厳密解そのものである。
c_{1\uparrow} を H_U のもとで時間発展させて
```
