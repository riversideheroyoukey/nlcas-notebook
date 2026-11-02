# 実践ループ サンプルシナリオ：伊藤の補題と Black–Scholes 方程式

幾何ブラウン運動 $dS=\mu S\,dt+\sigma S\,dW$ に従う資産の関数 $f(S,t)$ に**伊藤の補題**を
適用し、Black–Scholes 方程式を書き下す。

**バックエンド:** standard（Maxima）／**ドメイン:** 金融数学（financial_mathematics）。

## SCENARIO-MACRO の第1号 —— 一般操作だけで書き直した

**このシナリオは 2026-08-14 に、専用操作 `apply_ito_lemma` を使う形から、
一般操作だけで書く形へ書き直した**（open_items SCENARIO-MACRO）。

書き直した理由は「マクロが動かないから」ではない。**動くからこそ測定器が壊れる**：
このシナリオのために足した操作でこのシナリオを走らせれば、「機能は足りているか」の
答えは自明に「足りている」になり、問いが自分の答えを含んでしまう。実践ループが
確かめられるのは、そのマクシロ自身が、それが書かれた唯一の入力に対して動くことだけになる。

**削除したもの:** `apply_ito_lemma`（`notebook/engine_finance_mixin.py` ＋
`ui/interpret_finance.py`、共通操作スキーマにすら載っていない専用の入口）。中身は
`differentiate` を3回呼んで**固定の線形結合**に合成しているだけだった。

**移した知識:** マクロの docstring が持っていた伊藤の公式そのものは、**研究者が打つ
1つの段**になった（下の段 5）。公式は金融であって、システムの持ち物ではない。

**書き直しの前後で値が一致することを実測済み:**

| $f$ | 旧（マクロ） | 新（一般操作） |
|---|---|---|
| $\ln S$ | $df=\left(-\frac{\sigma^2-2\mu}{2}\right)dt+\sigma\,dW$ | $df=\left(\mu-\frac{\sigma^2}{2}\right)dt+\sigma\,dW$ |
| $S^2$ | $df=\left(S^2\sigma^2+2S^2\mu\right)dt+2S^2\sigma\,dW$ | $dg=\left(S^2\sigma^2+2S^2\mu\right)dt+2S^2\sigma\,dW$ |

同じ値で、並べ方だけが違う（旧は `ratsimp` を通していた）。

**ついでに直ったこと:** 旧シナリオの SDE と Black–Scholes 方程式は
`diff(V,t) + (1/2)*sigma^2*S^2*diff(V,S,2) + r*S*diff(V,S) - r*V = 0` という
**生の Maxima 構文**で書かれていた（Core Principle #1 違反が公開実例に載っていた）。
書き直しでどちらも LaTeX になった。

## 数学的背景

$S$ が幾何ブラウン運動 $dS=\mu S\,dt+\sigma S\,dW$ に従うとき、$f(S,t)$ の微分は
通常の連鎖律ではなく**伊藤の補題**に従う：

$$df=\left(\frac{\partial f}{\partial t}+\mu S\frac{\partial f}{\partial S}
  +\frac{1}{2}\sigma^2S^2\frac{\partial^2f}{\partial S^2}\right)dt
  +\sigma S\frac{\partial f}{\partial S}\,dW.$$

$\frac12\sigma^2S^2\partial^2_Sf$ の項が古典的な連鎖律との差で、$(dW)^2=dt$ から来る。
**この式そのものは金融の内容**であり、システムが提供するのは偏微分（`differentiate`）と
式の組み立てだけである。

## 手順と期待値

各段が独立の YOU カード。$f=\ln S$ と $g=S^2$ の2つで確かめる。

| 段 | 入力（自然言語 / 数式） | 結果 |
|---|---|---|
| 1 | `$f = \ln S$`（数式で登録） | $f=\ln S$（H-1 で $f$ を登録） |
| 2 | `E1 を t で偏微分して` | $\dfrac{\partial f}{\partial t}=0$ |
| 3 | `E1 を S で偏微分して` | $\dfrac{\partial f}{\partial S}=\dfrac{1}{S}$ |
| 4 | `E1 を S で2階偏微分して` | $\dfrac{\partial^2f}{\partial S^2}=-\dfrac{1}{S^2}$ |
| 5 | `$df = (E2 + \mu S E3 + \frac{1}{2}\sigma^2 S^2 E4) dt + \sigma S E3 dW$` | $df=\left(\mu-\dfrac{\sigma^2}{2}\right)dt+\sigma\,dW$ —— **伊藤の公式を研究者が書き、式番号参照（H-2）で段 2〜4 の結果を差し込む** |
| 6 | `$g = S^2$`（数式で登録） | $g=S^2$ |
| 7 | `E6 を t で偏微分して` | $\dfrac{\partial g}{\partial t}=0$ |
| 8 | `E6 を S で偏微分して` | $\dfrac{\partial g}{\partial S}=2S$ |
| 9 | `E6 を S で2階偏微分して` | $\dfrac{\partial^2g}{\partial S^2}=2$ |
| 10 | `$dg = (E7 + \mu S E8 + \frac{1}{2}\sigma^2 S^2 E9) dt + \sigma S E8 dW$` | $dg=\left(S^2\sigma^2+2S^2\mu\right)dt+2S^2\sigma\,dW$ |
| 11 | `$dS = \mu S dt + \sigma S dW$`（数式で登録） | SDE。専用操作は要らず、通常の式登録で書ける |
| 12 | `$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$` | Black–Scholes 方程式。同じく通常の方程式 |

## 確認された挙動・注意点

- **偏微分か全微分かは明示が要る。** `E1 を t で微分して` は
  「偏微分か全微分か判別できません」で**失敗する**（`differentiate_is_ambiguous`）。
  $f(S,t)$ で $S$ が $t$ に依存する状況なので、この曖昧さは実在する——システムが
  勝手に決めないのは正しい。研究者は「偏微分して」と書く。
- **式番号参照（H-2）が公式の組み立てを担う。** 段 5 は段 2〜4 の結果を `E2`〜`E4` で
  参照し、登録時に canonical へ展開される。存在しない式番号を書いた場合は
  `undefined_symbols` に入って ⚠ が出る（実測）ので、黙って記号になることはない。
- **結果の canonical は展開前の形を保つ。** 段 5 の canonical は
  `((0) + mu*S*(1/S) + …)*dt + …` で、表示だけが Maxima の `tex1` で簡約される。
  これは「`canonical` は研究者自身の書いたもの」という規則どおり——研究者が書いたのは
  伊藤の公式であって、その簡約結果ではない。後続の操作は数学的に等しい式に当たる。
- **Black–Scholes 方程式を記号的に解いて閉じた形の価格式を出すことは範囲外。**
  ここで作るのは方程式そのものまでで、その先は数値層（N-1）か別の設計の仕事。

## 一般操作だけで書くときの型（後続の書き直しはこれをなぞる）

1. **公式は研究者が書く。** マクロが埋め込んでいた固定の式は、式番号参照で組み立てる
   1つの段になる。システムが出すのは `differentiate` / `simplify` のような一般操作だけ。
2. **入力は LaTeX で書く。** `diff(V,t)` のような CAS 構文をシナリオに書かない
   （原則 #1）。書けない綴りが見つかったらそれが欠陥であり、シナリオの成果物である。
3. **マクロのテストは、罠を押さえているものだけ移す。** 操作の interface を主張する
   テストは操作と一緒に消える。ここでは
   `test_feeding_result_into_another_call_does_not_hang` が Fourier 側
   （`_transform_source_body`）の罠を**名指ししている唯一のテスト**だったので、
   `tests/test_fourier_transform.py` へ移した。値を主張していたテスト
   （textbook result）は、一般操作の経路に対する主張として書き直した。
4. **段の長さはコストではない。** 12 段は `apply_ito_lemma` 1 行より長いが、
   その 12 段が通ることこそ実践ループが出すべき成果物である。
5. **回避策はシナリオに書かない。** 欠陥を避けた綴りを実例に置くと、直ったあとも
   誰も書き換えず**恒久化する**——公開実例は「こう書くものだ」と読まれるからである。
   書くなら「一時的」と明記し、**シナリオ側に理由と起票 ID を、起票側に「修正したら
   このシナリオへ戻る」を**、両方に書く。片方だけでは直した人がシナリオを知らない。
   実例（[情報理論](scenario_information_theory_jp.md) の bit 単位、
   [強制振動](scenario_forced_oscillator_laplace_jp.md) の英語逆変換）。

```scenario domain=financial_mathematics
@note: 幾何ブラウン運動 $dS=\mu S\,dt+\sigma S\,dW$ に従う資産の関数に伊藤の補題を適用する。まず対数 $f=\ln S$ を登録する。
$f = \ln S$
@note: 伊藤の公式に必要な3つの偏導関数を、一般操作でそれぞれ求める。まず $\partial f/\partial t$。$f$ に $t$ が陽に現れないので 0 になる。偏微分か全微分かは研究者が明示する（$S$ が $t$ に依存するため、この曖昧さは実在する）。
E1 を t で偏微分して
@note: 次に $\partial f/\partial S$。
E1 を S で偏微分して
@note: 伊藤の補題に固有の2階項 $\partial^2 f/\partial S^2$。これが古典的な連鎖律との差を生む。
E1 を S で2階偏微分して
@note: 伊藤の公式 $df=(\partial_t f+\mu S\partial_S f+\frac12\sigma^2S^2\partial_S^2 f)dt+\sigma S\partial_S f\,dW$ を、式番号参照で組み立てる。公式そのものは金融の内容であり、研究者が書く。
$df = (E2 + \mu S E3 + \frac{1}{2}\sigma^2 S^2 E4) dt + \sigma S E3 dW$
@note: 別の関数 $g=S^2$ でも同じ手順を踏み、同じ公式が別の入力でも働くことを確かめる。
$g = S^2$
@note: $\partial g/\partial t$。
E6 を t で偏微分して
@note: $\partial g/\partial S$。
E6 を S で偏微分して
@note: $\partial^2 g/\partial S^2$。定数 2 になる。
E6 を S で2階偏微分して
@note: 同じ伊藤の公式を $g$ について組み立てる。段 5 と同じ形で、参照する式番号だけが変わる。
$dg = (E7 + \mu S E8 + \frac{1}{2}\sigma^2 S^2 E9) dt + \sigma S E8 dW$
@note: 原資産の SDE 記法そのものは、専用操作を必要とせず通常の式として登録できる（$dt$・$dW$ は普通の記号）。
$dS = \mu S dt + \sigma S dW$
@note: Black–Scholes 方程式も同様に通常の方程式として書ける。これを記号的に解いて閉じた形の価格式を出すことは、このシナリオの範囲外。
$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$
```
