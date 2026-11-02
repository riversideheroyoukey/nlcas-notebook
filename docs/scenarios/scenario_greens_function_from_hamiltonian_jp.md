# 実践ループ シナリオ：場の演算子で書いたハミルトニアンから、グリーン関数まで

> 2026-08-20 実施、2026-08-21 に**計算する版へ書き直し**。多体系のグリーン関数を、
> 教科書どおりの綴りで書き——そして**導出の各段を実際に CAS に計算させる**。
> 対角化による固有値の導入、グリーン関数の基底への移行、Dyson 方程式の形式解、
> 高次グリーン関数の Hartree–Fock 分解、時間から周波数へのフーリエ変換、そして
> $G$ について解くところまで。

**2026-08-21 の書き直しの理由（利用者の指摘）:** 初版は**定義を置いただけ**だった。
式は教科書どおりに書けていたが、導出の各段は研究者が答えを知って書き写したもので、
「計算できたか」は測れていなかった。**CAS の利用可能性を問う**なら、各段を操作として
通し、通らない段はその理由を測って書く必要がある。書き直した版の内訳は下表。

**バックエンド:** Maxima／**ドメイン:** 量子力学（quantum_mechanics）。

### なぜこれが確認に値するのか

このシナリオは、**「研究者がいつもの記法で書けるか」を測ったら 0 だった**という実測から
生まれた。同日、T=0 の多体グリーン関数を教科書どおりの綴りで18行打ち込んだところ、
**意味が保たれたものは 1 件も無かった**。しかも大半はエラーではなく、**別の意味の式として
成功していた**:

| 綴り | 実測（修正前） |
|---|---|
| `G = G^0 + G^0\Sigma G` | **`G*Sigma+1`**（`G^0` が Maxima で 1） |
| `v(\mathbf r-\mathbf r')` | **`v(0)`**（プライムが黙って捨てられる） |
| `\hat\psi` | **`p_hatsi`**（`psi` の中に `hat` が割り込む） |
| `\psi^\dagger` 族 | 5つの綴りが**5通りの別物**（転置・`dagger(x)` 乗・積…） |
| `\langle\ldots\rangle` | `langle`・`rangle` という**未定義記号2つの積** |
| `c_{k+q}` | **`c(k) + q`**（波括弧が落ちる） |
| `\delta(\mathbf r-\mathbf r')` | `KroneckerDelta requires exactly 2 arguments`（内部語） |
| `T[\ldots]`（時間順序） | 生の pydantic ValidationError |

さらに深いところに、**Maxima アダプタが構文エラーを全部「成功」として報告する**という
機構があった（`errcatch` は構文エラーを捕まえない。ガード変数が前のコマンドの値を保持し、
未束縛の `_res_`——アダプタ自身の作業変数名——が答えとして印字されていた）。**これがある
限り、正しく降ろせない記法はすべて「もっともらしい答え」として返ってきていた。**

このシナリオは、その根治が**実物で**効いていることを確認する。問いは「計算できるか」
ではなく、**書いたとおりの意味で通るか**である。

**続きを書いたら、同じ形がもう6件出た**（2026-08-21 実測）。運動方程式は前半に
無かった綴り——時間微分・近似の関係・添字つきの解を求める指示——を要求し、そこは
測られていなかった:

| 綴り | 実測（修正前） |
|---|---|
| `\frac{\partial}{\partial t}\hat c_{n\sigma}(t)` | **`'diff(\hat,t,1)*c_{n,sigma}(t)`**——装飾コマンドを微分し、名前は別の因子になる |
| `\frac{\partial \hat\psi(x,t)}{\partial t}` | **`psi_hat(x,t)/t`**——導関数と読まれず、$\partial$ が約分される（BP-9 そのもの） |
| `\frac{d}{dx}\sin x` | **`'diff(sin,x,1)*x`** ＝ **0** |
| `\hat T(\mathbf r)` | **`time_order(…, hat)`**——1体の運動エネルギーが時間順序演算子になる。**前半の $\hat H_0$（E4）に載っていた** |
| `A \approx B + C` | **`A*B*approx+C`**——`simplify` が成功として返す（`\neq` で直したのと同じ形） |
| `E32 を G^{(HF)}_{n\sigma}(\omega) について解いて` | 解釈されない——波括弧つきの添字も上付きラベルも、解く変数としては読まれない。読めるようにすると今度は、その名前が**定義済みなら式番号に置き換えられ**、研究者が書いていない名前が CAS に渡っていた（`solve(…, E22_psupHFn,sigma (reference omitted from the public snapshot))` の形）|

前の5件と同じで、**エラーになったものは一つも無い**。最初の3件は一つの原因で、
微分される対象を読む走査が `\hat` や `\sin` を「それだけで完結した名前」と見て
いた——コマンドは単独では量ではないので、これは推測ではなく文法の取りこぼしである。

**そして「計算する版」に書き直したら、さらに8件出た**（2026-08-21 実測）。**書けるか**を
測っていた間は見えず、**計算させて初めて**当たる層である:

| 段 | 実測（修正前） |
|---|---|
| 展開を定義に代入 | **`Maxima evaluation error`**（範囲なしの和が2引数の `sum(body,n)` で届く。理由も出ない） |
| 同上（そのあと） | `c_hatn, sigma (reference omitted from the public snapshot)` ——**このシステム自身が書く綴り**——を読み直すと `IndexedSymbol` で落ちる。括弧が名前の**最後の下線片**に付いていた |
| 同上（さらに） | `time_order(...)` の中でだけ `[I, sigma]` が**交換子**に化ける（抑止フラグが入れ子の引数まで効いていた） |
| 対角化の代入＋簡約 | δ が縮約されない。CAS が返した `'sum(...)`/`kron_delta(...)` が**ただの関数呼び出し**として読み直されていた |
| 両辺に左から掛ける | **左右を区別せず**、ラベルは左・値は右に掛かる。左辺は簡約されない（Maxima 自身は `A^^-1 . A` を 1 にできる） |
| `\hat H_0 をスカラーとする` | **`H`** を宣言していた（`H_hat_0` の下線を2つ数えて「多重添字」と読み、装飾ごと剥がしていた） |
| 結果の表示 | `psi_hat_sigma` が `psi_{hat}_{sigma}` ——**二重下付き＝組版できない LaTeX** |
| `define c as an electron annihilation operator` | 一般の宣言として**成功**し、`c` は QA 演算子にならない。交換子が **0** を返す（日本語の綴りは正しく通る） |

**「書ける」と「計算できる」は別の層である**——というのが、この書き直しで測れた
いちばん大きなことである。

### どの段が計算で、どの段が定義か

| 段 | 何をするか | 計算か |
|---|---|---|
| $\hat H_0$ の対角化 | 一般の $\sum_{nm}T_{n\sigma,m\sigma}\hat\rho$ に「基底が対角化する」($T=\varepsilon\delta$) を**代入して簡約** | ✅ **CAS**（δ が $m$ の和を潰し、**固有値がここで初めて現れる**） |
| グリーン関数の移行 | 定義に場の演算子の展開を**2回代入** | ✅ **CAS**（和は期待値・時間順序の中に入ったまま運ばれる） |
| Dyson の形式解 | $(1-G^0\Sigma)G=G^0$ の**両辺に $(1-G^0\Sigma)^{-1}$ を左から掛ける** | ✅ **CAS**（左辺が 1 に簡約され $G=(1-G^0\Sigma)^{-1}G^0$ が出る） |
| 運動方程式の交換子 | $[\hat c_{m\sigma},\hat H]$ | ❌ Maxima には演算子代数が無い → **姉妹シナリオ（Julia/QuantumAlgebra）で計算**し、結果を書き写す |
| Hartree–Fock 分解 | 4演算子を対に分ける（$\approx$） | ⬜ 近似そのもの（1行）。$\approx$ は**書けて表示され、計算では断られる** |
| フーリエ変換 | 逆変換の**定義の積分を書いて両辺を微分** | ✅ **CAS**（$-i\omega$ が降り、$i\hbar\partial_t\to\hbar\omega$ が出る） |
| $G$ について解く | 周波数領域の方程式を解く | ✅ **CAS**（Hartree–Fock のグリーン関数） |
| 近似の検算 | $\Sigma^{(HF)}=0$ を**代入** | ✅ **CAS**（自由伝播関数に戻る） |

**評価するものではない量**（任意基底の行列要素・全空間積分・期待値）は定義のまま
残る——評価しようがないからである。計算しているのは**添字と係数の代数**であって、
そこがこの系で機械にやらせられる部分である。

### 物理の筋

1. スピン依存の場の演算子と、その反交換関係
2. ハミルトニアン——1体項と2体項を、場の演算子で書く
3. スピン依存の正規直交系（直交規格化と完全性）。直交規格化は**成分に開く**
4. 場の演算子を正規直交系で展開する
5. **ハミルトニアンを新しい基底へ移す**——1体項は行列要素 $T_{n\sigma,m\sigma}$ と
   演算子の対に、2体項は**4モード＋2スピンの6添字**を持つ行列要素になる（スピンが
   2種類なのは、場の演算子で書いた形が2つだから）。ここまで基底は**任意**である
6. **その0次項を対角化する**——「当初導入した基底関数系が $\hat H_0$ を対角化する」
   という仮定は、行列要素についての1本の式 $T_{n\sigma,m\sigma}=\varepsilon_{n\sigma}\delta_{nm}$
   にすぎない。代入して整理すると δ が和を潰し、$\hat H_0$ は数演算子の和になる——
   **固有値 $\varepsilon_{n\sigma}$ は、この計算の結果として現れる**
7. **その基底の生成消滅演算子で**グリーン関数を定義する——**基底状態を明示した**
   $\langle\Phi_0|T[\cdots]|\Phi_0\rangle$。場の演算子で書いてから基底へ移す道もあるが、
   移すための代入が2回要るだけで得るものがない（**利用者の指摘、2026-08-22**——
   以前はそちらを通っていた）
8. **時間差についてフーリエ変換**して周波数 $\omega$ を導入し、自由伝播関数 $G^0$ を
   書く——$i\eta$ の符号は**フェルミ面を境に変わる**
9. Dyson 方程式を**成分で書く**——右辺第2項は基底の添字についての和になる。共通因子を
   括り出し、$G$・$\Sigma$ を**行列と宣言**して、**逆行列で一般解を書く**
10. **そこで初めて近似を入れる**——1次（Hartree）の自己エネルギーを、一般形の
    $\Sigma$ に代入する

ここまでが Dyson の道である。最後の一手だけが「計算」で、それ以外は**定義を置く**
作業である。任意基底の行列要素と同じ理由で、ここに現れる積分や期待値は評価する
ものではない——評価しようがない。ノートに求められるのは、**定義として置けて、
書いたとおりに表示され、後で使えること**である。

### 続き：運動方程式から自己エネルギーを決める

**Dyson 方程式は $\Sigma$ が何であるかを答えない。** 一般形のまま形式解を出しただけで、
中身は外から与えるしかない。もう一つの道——グリーン関数の**運動方程式**——では、$\Sigma$ が
ハミルトニアンから出てくる。

11. Heisenberg 描像の生成消滅演算子。指数の肩は**全**ハミルトニアン（相互作用描像の
    E10 とはそこが違う）
12. 演算子の運動方程式 $i\hbar\partial_t\hat c=[\hat c,\hat H]$。反交換関係で開くと、
    $\hat H_0$ からは固有値が、$\hat V$ からは**演算子3つの積**が出る
13. グリーン関数の運動方程式。時間微分は**時間順序の階段関数にも当たり**、そこから
    等時刻の反交換関係——$\hbar\delta(t-t')$ の項——が出る。2体項は**演算子4つ**の
    期待値、すなわち**高次のグリーン関数** $G^{(2)}$ を生む
14. **方程式は閉じていない。** 1体の方程式に2体が現れ、2体の方程式には3体が現れる。
    **近似とは、この鎖をどこかで切ることである**
15. 最低次の切り方（**Hartree–Fock 分解**）——4つの演算子を2つずつの対に分け、同時刻の
    対を占有数の期待値で置き換える。**等号ではないので $\approx$ と書く**。近似は
    この1行だけで、以降は代数である
16. 代入すると和が畳まれ、**自己エネルギーが出てくる**。第1項は Dyson の側で1次の
    近似として外から入れる Hartree 項**そのもの**、第2項は交換（Fock）項——
    フェルミオンであることがここで初めて効く。フーリエ変換して**$G$ について解く**と
    $G^{(HF)}_{n\sigma}(\omega)=\hbar/(\hbar\omega-\varepsilon_{n\sigma}-\Sigma^{(HF)}_{n\sigma})$、
    自己エネルギーを 0 に戻せば自由伝播関数に戻る（**この4段はすべて CAS が計算する**）

**近似が1行に閉じ込められている**ことが、この道の見どころである。$\approx$ を書いた
その1行以外はすべて等号で、どこで一般性を失ったかが後から読める。

### 何が確認されるか

- `\hat\psi_\sigma(\mathbf r)`——装飾つきの名前に添字が付き、引数を取る形が通る
- `\psi^\dagger` が**名前の装飾**として読まれる（Maxima には演算子代数が無いので、
  随伴は宣言された行列・ベクトル・テンソル・演算子のときだけ）
- `\delta_{\sigma\sigma'}`（Kronecker）と `\delta(\mathbf r-\mathbf r')`（Dirac）が
  **添字か引数リストか**で読み分けられる
- `\langle\Phi_0|A|\Phi_0\rangle` が `\langle A\rangle_{\Phi_0}` と**同じ量**になる
- `T[\cdots]` が時間順序として読まれる（素の `T` は運動エネルギーとも読めるので読まない）
- `G^0`・`\Sigma^{(1)}` の上付きが**ラベル**として読まれ、指数にならない
- `\sum_{\mathbf k'}`——ベクトル値の束縛添字が、包みの中の名前を走る
- **Dyson 方程式の一般解が逆行列で書ける**: $G=(1-G^0\Sigma)^{-1}G^0$。宣言が無ければ
  $^{-1}$ は逆数、積は可換積で、どちらも別の量になる
- `\frac{\partial}{\partial t}\hat c_{n\sigma}(t)`——**装飾つきの名前が微分される**
  （`\hat` は単独では量ではないので、微分の対象はその先まで）
- `A \approx B`——近似の関係が**書けて、表示され、計算では断られる**。「書き下せて
  から計算する」の形で、断りは使用地点にある
- `E32 を G^{(HF)}_{n\sigma}(\omega) について解いて`——**研究者の綴りのまま**解く変数を
  指定できる（ラベル・添字・引数つき。既に定義された名前が E-id に化けることもない）

```scenario domain=quantum_mechanics
@note: スピン依存の場の演算子から始める。ハットは名前の装飾であり、$\hat\psi$ は $\psi$ とは別の量である——だからそう書く。
\hat\psi_\sigma(\mathbf{r})
\hat\psi^\dagger_\sigma(\mathbf{r})
@note: 等時刻の反交換関係。右辺の2つの $\delta$ は**別物**で、スピンの $\delta_{\sigma\sigma'}$ は Kronecker、位置の $\delta(\mathbf r-\mathbf r')$ は Dirac である。添字か引数リストかで読み分けられる——物理はそう書き分けているので、推測は要らない。
\{\hat\psi_\sigma(\mathbf{r}), \hat\psi^\dagger_{\sigma'}(\mathbf{r}')\} = \delta_{\sigma\sigma'}\delta(\mathbf{r}-\mathbf{r}')
@note: $\hat H_0$ の添字 0 は**名前の一部**だと宣言しておく。宣言しなければ未宣言の添字は成分インデックスと読まれ（このノートの規則どおり）、あとで計算結果に $\hat H(0)$ と出る。宣言はこの系のほとんどの読みの判別材料である。
\hat H_0 をスカラーとする
@note: ハミルトニアンの1体項。$\hat T$ は運動エネルギーと外場をまとめた1体演算子。$\int d^3r$ は体積測度であって、1変数の積分ではない。
\hat H_0 = \int d^3r\, \hat\psi^\dagger_\sigma(\mathbf{r}) \hat T(\mathbf{r}) \hat\psi_\sigma(\mathbf{r})
@note: 2体項。$d^3r\,d^3r'$ の6次元で、$\mathbf r$ と $\mathbf r'$ は**別の点**である——プライムが黙って捨てられていた頃、$v(\mathbf r-\mathbf r')$ は $v(0)$ になっていた。
\hat V = \frac{1}{2}\int d^3r \int d^3r' \hat\psi^\dagger_\sigma(\mathbf{r}) \hat\psi^\dagger_{\sigma'}(\mathbf{r}') v(\mathbf{r}-\mathbf{r}') \hat\psi_{\sigma'}(\mathbf{r}') \hat\psi_\sigma(\mathbf{r})
@note: スピン依存の正規直交系。直交規格化は全空間積分で、基底が任意なので**評価するものではない**——積分そのものが条件である。
\int d^3r\, \varphi^*_{n\sigma}(\mathbf{r}) \varphi_{m\sigma'}(\mathbf{r}) = \delta_{nm}\delta_{\sigma\sigma'}
@note: 成分に開く。デカルト3成分の反復積分になり、範囲は書いていなくても全空間。
成分に展開して
@note: 完全性。右辺は Dirac の $\delta$ で、左辺の和は**モードとスピンの両方**を走る。
\sum_{n\sigma} \varphi_{n\sigma}(\mathbf{r}) \varphi^*_{n\sigma}(\mathbf{r}') = \delta(\mathbf{r}-\mathbf{r}')
@note: 場の演算子を正規直交系で展開する。ここから先の代数は、係数（スカラー）と演算子 $\hat c_{n\sigma}$ に分かれる。
\hat\psi_\sigma(\mathbf{r}) = \sum_n \varphi_{n\sigma}(\mathbf{r}) \hat c_{n\sigma}
@note: **ここから計算する。** 展開（E9）を $\hat H$ に入れて、**新しい基底でのハミルトニアンを書く**。まず1体項。行列要素 $T_{n\sigma,m\sigma}$ は E6 と同じ形の全空間積分で、評価するものではない。演算子の対 $\hat c^\dagger_{n\sigma}\hat c_{m\sigma}$ は**1つの名前**として運ぶ——Maxima には演算子代数が無いので、順序を要求しない形にしておく（演算子代数が要る計算は姉妹シナリオの Julia セッションでやる）。
\hat\rho_{n\sigma,m\sigma} = \hat c^\dagger_{n\sigma} \hat c_{m\sigma}
\hat H_0 = \sum_{n m} T_{n\sigma,m\sigma} \hat\rho_{n\sigma,m\sigma}
@note: 2体項も同じ変換で移る。**スピンは2種類しかない**——場の演算子で書いた形が $\sigma,\sigma'$ の2つで、基底展開で増えるのは**モードだけ**だからである。したがって独立な添字は**4モード＋2スピンの6つ**。和の添字は**区切って書く**（`\sum_{nmpq}` は「$nmpq$ という名の1添字」）。
\hat V = \frac{1}{2}\sum_{n m p q}\sum_{\sigma \sigma'} V_{n\sigma,m\sigma',p\sigma',q\sigma} \hat c^\dagger_{n\sigma} \hat c^\dagger_{m\sigma'} \hat c_{p\sigma'} \hat c_{q\sigma}
@note: ここまで基底は**任意**である。ここで**選ぶ**: 「当初導入した基底関数系が $\hat H_0$ を対角化する」という仮定は、行列要素についての1本の式にすぎない。平面波に特殊化するのとは違い、一般性はほとんど失われない。
T_{n\sigma,m\sigma} = \varepsilon_{n\sigma} \delta_{nm}
@note: 0次項に代入する。
E11 に E13 を代入して
@note: 整理すると**クロネッカーδが $m$ の和を潰し**、対角形が残る——**固有値 $\varepsilon_{n\sigma}$ はこの計算の結果として現れる**。手で書き写したのではない。$\hat\rho_{n\sigma,n\sigma}$ は数演算子である。
E14 を簡約して
@note: **0次項が対角になった。ここでグリーン関数を、この基底の生成消滅演算子で定義する。** 場の演算子で書いてから基底へ移す道もあるが、移すための代入が2回要るだけで、得るものがない——最初からこの基底で書けばよい。$T[\cdots]$ は時間順序（素の $T$ は運動エネルギーとも読めるので括弧で書く）、両側が同じ状態なので $\langle A\rangle_{\Phi_0}$ と同じ量である。
\hat c_{n\sigma}(t) = e^{i\hat H t/\hbar} \hat c_{n\sigma} e^{-i\hat H t/\hbar}
G_{n\sigma,m\sigma'}(t,t') = -i\langle\Phi_0| T[\hat c_{n\sigma}(t) \hat c^\dagger_{m\sigma'}(t')]|\Phi_0\rangle
@note: 自由伝播関数。$G^0$ の上付き 0 は**摂動次数のラベル**であって指数ではない。$i\eta$ の符号は**フェルミ面を境に変わる**。
G^0_{n\sigma}(\omega) = \frac{1}{\omega - \varepsilon_{n\sigma}/\hbar + i\eta\,\mathrm{sgn}(\varepsilon_{n\sigma}-\varepsilon_F)}
@note: Dyson 方程式を**成分で書く**——右辺第2項は基底の添字についての和で、$G$ が右辺にも現れる。
G_{n\sigma,m\sigma'}(\omega) = G^0_{n\sigma}(\omega)\delta_{nm}\delta_{\sigma\sigma'} + \sum_{p \sigma''} G^0_{n\sigma}(\omega) \Sigma_{n\sigma,p\sigma''}(\omega) G_{p\sigma'',m\sigma'}(\omega)
@note: 添字を落として行列として書く。**$G$・$\Sigma$・$G^0$ を行列と宣言する**と、積は行列積になり $^{-1}$ は**逆行列**になる。宣言しなければ $^{-1}$ は逆数、積は可換積で、どちらも別の量である。
G を行列とする
\Sigma を行列とする
G^0 を行列とする
@note: $G$ を左に括り出した形で書く。**$G = G^0 + G^0\Sigma G$ とは書かない**——この形は H-1 の定義文になり、以後 $G$ という名前がその式に束縛されて、解きたい未知数でなくなるからである（実測: そう書くと次の計算の左辺が $G^0\Sigma G+G^0$ に展開されて戻ってくる）。
(1 - G^0 \Sigma) G = G^0
@note: **形式解を計算で出す。** 両辺に $(1-G^0\Sigma)^{-1}$ を**左から**掛ける——非可換なので左右は別の式であり、どちらかは研究者が決める。左辺の $(1-G^0\Sigma)^{-1}(1-G^0\Sigma)$ が 1 に簡約されるところは CAS の仕事である。
E20 の両辺に (1 - G^0 \Sigma)^{-1} を左から掛けて
@note: ここまでが Dyson の道である。**$\Sigma$ が何であるかを、Dyson 方程式は答えない。** もう一つの道——グリーン関数の**運動方程式**——では、$\Sigma$ がハミルトニアンから出てくる。
@note: Heisenberg 描像の演算子。相互作用描像（E10）と違い、指数の肩に乗るのは**全**ハミルトニアン $\hat H=\hat H_0+\hat V$ である。
@note: 演算子の運動方程式。**微分されるのは $\hat c$ であって $c$ ではない**——ハットは名前の装飾なので、装飾ごと微分の中に入る。
i\hbar \frac{\partial}{\partial t} \hat c_{n\sigma}(t) = [\hat c_{n\sigma}(t), \hat H]
@note: **この交換子は Maxima では計算できない**——演算子代数が無いからである（それがこのセッションで演算子の対を1つの名前で運んでいる理由でもある）。**この役割分担は 2026-07-30 に決めてある**: 「グリーン関数の代数は Maxima、演算子代数が要る局面だけ Julia」（docs/33 §7）。**同じ交換子を QuantumAlgebra で実際に計算した**のが姉妹シナリオ scenario_greens_function_eom_commutator で、$[\hat H,\hat c_{m\sigma}]$ に対して $-\varepsilon_{m\sigma}\hat c_{m\sigma} + \sum V(\ldots)$ を返す。以下はその結果を（符号の向きを $[\hat c,\hat H]$ に合わせて）書き写したものである。
[\hat c_{m\sigma}, \hat H] = -\varepsilon_{m\sigma}\hat c_{m\sigma} - \sum_{p q s}\sum_{\sigma''} V_{m\sigma,p\sigma'',q\sigma'',s\sigma} \hat c^\dagger_{p\sigma''} \hat c_{q\sigma''} \hat c_{s\sigma}
@note: グリーン関数を基底の添字で書く（E19 で計算した形の、係数を落とした芯）。
@note: グリーン関数の運動方程式。時間微分は**時間順序の階段関数にも当たり**、そこから等時刻の反交換関係——$\hbar\delta(t-t')$ の項——が出る。残りは演算子の運動方程式そのもので、2体項が**演算子4つ**の期待値を生む。
i\hbar \frac{\partial}{\partial t} G_{n\sigma,m\sigma'}(t,t') = \hbar\delta(t-t')\delta_{nm}\delta_{\sigma\sigma'} + \varepsilon_{n\sigma} G_{n\sigma,m\sigma'}(t,t') + \sum_{p q s}\sum_{\sigma''} V_{n\sigma,p\sigma'',q\sigma'',s\sigma} G^{(2)}_{p\sigma'',q\sigma'',s\sigma,m\sigma'}(t,t')
@note: その高次のグリーン関数の定義。上付きの $(2)$ は $G^0$ の $0$ と同じく**ラベル**であって指数ではない。演算子3つは時刻 $t$、最後の1つだけが $t'$ である。
G^{(2)}_{p\sigma'',q\sigma'',s\sigma,m\sigma'}(t,t') = -i\langle\Phi_0| T[\hat c^\dagger_{p\sigma''}(t) \hat c_{q\sigma''}(t) \hat c_{s\sigma}(t) \hat c^\dagger_{m\sigma'}(t')]|\Phi_0\rangle
@note: **方程式は閉じていない。** 1体のグリーン関数の方程式に2体のものが現れ、2体のものの方程式には3体のものが現れる——階層は終わらない。**近似とは、この鎖をどこかで切ることである。**
@note: 最低次の切り方（Hartree–Fock 分解）。4つの演算子を**2つずつの対に分け**、同時刻の対を期待値で置き換える。第2項が負なのはフェルミオンの入れ替えによる。**等号ではないので $\approx$ と書く**——近似はこの1行だけで、以降は代数である。$\approx$ は書けて表示されるが、計算しようとすると理由つきに断られる（CAS に「およそ等しい」は無い）。
G^{(2)}_{p\sigma'',q\sigma'',s\sigma,m\sigma'}(t,t') \approx \langle \hat c^\dagger_{p\sigma''} \hat c_{q\sigma''}\rangle G_{s\sigma,m\sigma'}(t,t') - \delta_{\sigma\sigma''} \langle \hat c^\dagger_{p\sigma''} \hat c_{s\sigma''}\rangle G_{q\sigma'',m\sigma'}(t,t')
@note: 基底は $\hat H_0$ を対角化している（E15）ので、同時刻の期待値も対角である。
\langle \hat c^\dagger_{p\sigma''} \hat c_{q\sigma''}\rangle = \delta_{pq}\langle \hat n_{p\sigma''}\rangle
@note: 代入すると和が2つ畳まれ、残るのはモード $p$ とスピン $\sigma''$ についての和だけになる。これが自己エネルギーである——**運動方程式が $\Sigma$ を決めた。** 第1項は Dyson の側で1次の近似として置かれる Hartree 項で、第2項が交換（Fock）項：フェルミオンであることがここで初めて効く。
\Sigma^{(HF)}_{n\sigma,s\sigma} = \sum_{p \sigma''}\left[V_{n\sigma,p\sigma'',p\sigma'',s\sigma} - \delta_{\sigma\sigma''}V_{n\sigma,p\sigma'',s\sigma'',p\sigma}\right]\langle \hat n_{p\sigma''} \rangle
@note: 高次のグリーン関数が消え、方程式が**閉じた**。ただし**これはもう厳密な $G$ ではない**——分解を入れた後の量なので $G^{(HF)}$ と書き分ける。
i\hbar \frac{\partial}{\partial t} G^{(HF)}_{n\sigma,m\sigma'}(t,t') = \hbar\delta(t-t')\delta_{nm}\delta_{\sigma\sigma'} + \varepsilon_{n\sigma} G^{(HF)}_{n\sigma,m\sigma'}(t,t') + \sum_{s} \Sigma^{(HF)}_{n\sigma,s\sigma} G^{(HF)}_{s\sigma,m\sigma'}(t,t')
@note: **時間差についてフーリエ変換する。専用の操作は使わない**——逆変換の定義そのものを書き、両辺を $t$ で微分させる。$\omega$ と $(t-t')$ の間の空白は積の印である（空白が無いと $\omega$ を $(t-t')$ に適用した関数と読まれ、その読みは断られる）。
G^{(HF)}_{n\sigma}(t-t') = \frac{1}{2\pi}\int_{-\infty}^{\infty} G^{(HF)}_{n\sigma}(\omega) e^{-i\omega (t-t')} d\omega
@note: 微分は積分記号の中に入り、$-i\omega$ が降りてくる——これが $i\hbar\partial_t \to \hbar\omega$ の中身である。$\delta(t-t')$ も同じ変換で 1 になる。
E30 の両辺を t で微分して
@note: 周波数領域の方程式。$\hat H_0$ も $\Sigma^{(HF)}$ もこの基底で対角なので、対角成分だけを書く。**$\hbar$ の位置に注意**——Dyson（E22）の $\Sigma$ は角振動数の次元、ここの $\Sigma^{(HF)}$ はエネルギーの次元で、両者は $\hbar$ だけ違う。
(\omega - \varepsilon_{n\sigma}/\hbar) G^{(HF)}_{n\sigma}(\omega) = 1 + \Sigma^{(HF)}_{n\sigma} G^{(HF)}_{n\sigma}(\omega)/\hbar
@note: $G^{(HF)}$ について解く。**解く変数は研究者の綴りのまま**——上付きラベルも添字も引数も付いたまま——指定する。
E32 を G^{(HF)}_{n\sigma}(\omega) について解いて
@note: Hartree–Fock 近似のグリーン関数。極が $\varepsilon_{n\sigma}$ から $\varepsilon_{n\sigma}+\Sigma^{(HF)}_{n\sigma}$ へずれた——**相互作用が1粒子エネルギーをずらす**、というのがこの近似の内容である。
@note: 最後に、自己エネルギーを 0 に戻してみる。近似の入れ方が正しければ、相互作用を切ったときに自由伝播関数（E21、$i\eta$ を除く）へ戻らなければならない。
\Sigma^{(HF)}_{n\sigma} = 0
E33 に E34 を代入して
```
