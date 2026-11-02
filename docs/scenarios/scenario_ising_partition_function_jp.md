# 実践ループ サンプルシナリオ：2 次元イジング格子の分配関数（小格子・高温展開）

実践ループ・シナリオメニュー **S-g（統計力学）** の詳細化。2 次元イジング模型の**分配関数（自由エネルギー）の
高温展開**を、小格子で厳密な期待値とともに確認する。展開係数が**格子のループ（辺数 $|E|$・三角形数
$N_\triangle$・四角形数 $N_\square$）**で決まること、そして**三角格子では奇数次の項が現れる**（幾何学的
フラストレーションの符号）ことを見る。spin-algebra（CAS 非経由の自前スピン代数）で評価する。

## 物理的背景

イジング模型 $H = -J\sum_{\langle ij\rangle} s_i s_j$（$s=\pm\tfrac12$）の分配関数 $Z=\mathrm{Tr}\,e^{-\beta H}$
は、高温（$\beta\to0$）で

$$\ln Z = N\ln 2 + \tfrac12\kappa_2\beta^2 - \tfrac1{3!}\kappa_3\beta^3 + \tfrac1{4!}\kappa_4\beta^4 + \cdots$$

と展開でき、比熱 $C(\beta)=\beta^2\,\partial^2(\ln Z/N)/\partial\beta^2|_{\text{intensive}}$ は**連結なエネルギー
累積量** $\kappa_n=\langle H^n\rangle_c$ で書ける。無限温度モーメント $\langle H^n\rangle=\mathrm{Tr}\,H^n/\mathrm{Tr}\,1$
はグラフ不変量で表せ、

$$\langle H^2\rangle = c_2\,|E|,\qquad \langle H^3\rangle = c_3\,|E| + d_3\,N_\triangle,\qquad
\langle H^4\rangle = (\cdots)\,N_\square + \cdots$$

イジングでは $\langle H\rangle=0$（スピン反転対称）かつ $c_3=0$（単一ボンドの 3 次は消える）なので、
$\langle H^3\rangle = d_3 N_\triangle$ が**三角形数にのみ比例**する。よって：

- **2 部格子（正方）**：$N_\triangle=0\Rightarrow$ 奇数次 $\beta^3$ 項は消える。
- **三角格子**：$N_\triangle>0\Rightarrow \beta^3$ 項が現れる（幾何学的フラストレーション）。

これが本シナリオで確認する核心。

## セッション設定

- バックエンド：**標準シンボリック（Maxima）**。spin-algebra のモーメント／高温展開は CAS 非経由で評価される
  （`register_dispersion` 経由で登録）。ドメイン指定は不要。
- spin-½ の無限温度トレースを仮定（単一サイトで $(s)^2=1/4$）。

## 入力ステップ

### ステップ 1：エネルギーモーメントの一般式（グラフ不変量）

```
イジング模型の高温展開を一般式で求めて
→ ⟨H²⟩ = (1/16) J² |E|,  ⟨H³⟩ = (3/32) J³ N△
```

3 次が**三角形数 $N_\triangle$ のみ**に比例（イジングの $c_3=0$）。これが「3 次モーメント ↔ 3c-graph の
$\mathrm{tr}\,A^3$」を繋ぐ橋。

### ステップ 2：正方格子 2×2（小格子）に具体化

```
イジング模型 正方格子 2x2 の高温展開を求めて
→ ⟨H²⟩ = (1/4) J²,  ⟨H³⟩ = 0    (|E| = 4,  N△ = 0)
```

2×2 周期境界は 4-cycle（$|E|=4$、三角形なし）。一般式に $|E|=4,\,N_\triangle=0$ を代入した値と一致。
**手計算照合**：$\langle H^2\rangle = J^2\sum_{b}\langle (s_is_j)^2\rangle = 4\cdot\tfrac1{16}J^2 = \tfrac14J^2$
（異なるボンドの交差項は奇数冪で消える）。

### ステップ 3：比熱（＝分配関数）の高温展開

```
イジング模型 正方格子 2x2 の比熱の高温展開を求めて
→ C(β) = (1/4) J² β² + (1/32) J⁴ β⁴ + O(β⁵)
```

**$\beta^3$ 項が無い**（$N_\triangle=0$）。自由エネルギー $\ln Z$ の 2 階微分に対応。

### ステップ 4：4 次まで（四角形数 $N_\square$ の登場）

```
イジング模型 正方格子 2x2 の高温展開を4次まで求めて
→ ⟨H²⟩ = (1/4) J²,  ⟨H³⟩ = 0,  ⟨H⁴⟩ = (1/4) J⁴   (|E| = 4,  N△ = 0,  N□ = 1)
```

2×2 は唯一の 4-cycle を持つ（$N_\square=1$）。$\beta^4$ の起源。

### ステップ 5：累積量による $\beta^4$ 係数の手計算照合

$\langle H\rangle=0$ なので 4 次累積量は $\kappa_4 = \langle H^4\rangle - 3\langle H^2\rangle^2$。

$$\kappa_4 = \tfrac14 J^4 - 3\left(\tfrac14 J^2\right)^2 = \tfrac14 J^4 - \tfrac3{16}J^4 = \tfrac1{16}J^4,
\qquad \tfrac12\kappa_4 = \tfrac1{32}J^4.$$

ステップ 3 の $\beta^4$ 係数 $\tfrac1{32}J^4$ と一致。同様に $\beta^2$ 係数は $\kappa_2=\langle H^2\rangle=\tfrac14J^2$。

### ステップ 6：三角格子 3×3（フラストレーション → 奇数次の出現）

```
イジング模型 三角格子 3x3 の高温展開を求めて
→ ⟨H²⟩ = J²,  ⟨H³⟩ = (3/4) J³    (|E| = 16,  N△ = 8)
```

一般式に $|E|=16,\,N_\triangle=8$ を代入：$\tfrac1{16}J^2\cdot16=J^2$、$\tfrac3{32}J^3\cdot8=\tfrac34J^3$。
**三角形があるので $\langle H^3\rangle\ne0$。**

### ステップ 7：三角格子の比熱（$\beta^3$ 項が現れる）

```
イジング模型 三角格子 3x3 の比熱の高温展開を求めて
→ C(β) = J² β² − (3/4) J³ β³ + (5/16) J⁴ β⁴ + O(β⁵)
```

正方格子（ステップ 3）と対照的に、**$\beta^3$ 項 $-\tfrac34J^3\beta^3$ が出現**。係数は
$-\kappa_3\beta^3 = -\langle H^3\rangle\beta^3 = -\tfrac34J^3\beta^3$（イジングでは $\kappa_3=\langle H^3\rangle$）で
ステップ 6 と整合。**奇数次の有無が三角形数（幾何学的フラストレーション）の指標**である。

## 期待値まとめ（厳密・照合済み）

| 量 | 正方 2×2 | 三角 3×3 |
|---|---|---|
| 不変量 | $|E|=4,\ N_\triangle=0,\ N_\square=1$ | $|E|=16,\ N_\triangle=8$ |
| $\langle H^2\rangle$ | $\tfrac14J^2$ | $J^2$ |
| $\langle H^3\rangle$ | $0$ | $\tfrac34J^3$ |
| $\langle H^4\rangle$ | $\tfrac14J^4$ | — |
| $C(\beta)$ | $\tfrac14J^2\beta^2+\tfrac1{32}J^4\beta^4$ | $J^2\beta^2-\tfrac34J^3\beta^3+\tfrac5{16}J^4\beta^4$ |

## カバレッジ（このシナリオで検証される機能）

- **F-1 格子和の具体化**：一般式（グラフ不変量 $|E|,N_\triangle,N_\square$）↔ 具体格子（正方／三角）の照合。
- **エネルギーモーメント**：$\langle H^2\rangle,\langle H^3\rangle,\langle H^4\rangle$（無限温度トレース）。
- **比熱の高温展開**：連結累積量 $\kappa_n$ による自由エネルギー（分配関数）展開。
- **物理の可視化**：奇数次 $\beta^3$ 項の有無＝三角形数＝幾何学的フラストレーションの符号。
- バックエンド：標準シンボリック（Maxima）、spin-algebra の CAS 非経由評価。

## 関連

- 統一的な高温展開・モーメントの詳細は [user_guide.md](../user_guide.md) §エネルギーモーメントと三角形数。
- スピン**動的**スペクトル（交換子→モーメント→連分数→$S(\omega)$）は
  [scenario_spin_chain_spectrum.md](scenario_spin_chain_spectrum_jp.md)（S-a 発展版）。
- 課題管理は [open_items.md](../../KNOWN_ISSUES.md) §実践ループ S-g。

## generator コマンド（この `scenario` ブロックが生成の単一ソース）

```scenario domain=statistical_mechanics
@note: イジング模型 $H=-J\sum_{\langle ij\rangle}s_is_j$（$s=\pm\frac12$）の無限温度エネルギーモーメント $\langle H^2\rangle,\langle H^3\rangle$ を、格子の形状を指定せず一般式（辺数 $|E|$・三角形数 $N_\triangle$ のようなグラフ不変量）として求める。イジングでは $\langle H^3\rangle$ が三角形数 $N_\triangle$ のみに比例する。
イジング模型の高温展開を一般式で求めて
@note: 一般式を具体的な格子（周期境界の正方格子 $2\times2$）に当てはめる。この格子は4-cycle（$|E|=4$、三角形なし＝$N_\triangle=0$）なので、$\langle H^3\rangle=0$ になるはずである。
イジング模型 正方格子 2x2 の高温展開を求めて
@note: エネルギーモーメントから、比熱 $C(\beta)=\beta^2\,\partial^2(\ln Z/N)/\partial\beta^2$ の高温展開（$\beta$ の冪級数）を求める。これはMaximaへの直接のコマンドではない——再帰法・モーメント法にはCASの汎用ルーチンが存在しないため、この計算は本プロジェクト独自の spin-algebra モジュール（`notebook/runtime/spin_algebra.py`）による厳密で決定論的な閉形式計算として実行される。前段で求めたグラフ不変量によるエネルギーモーメント $\langle H^2\rangle,\langle H^3\rangle,\langle H^4\rangle$ を連結キュムラント $\kappa_n$（$\kappa_2=\langle H^2\rangle$、$\langle H\rangle=0$ より $\kappa_4=\langle H^4\rangle-3\langle H^2\rangle^2$）に変換し、冪級数 $C(\beta)=\kappa_2\beta^2-\tfrac1{3!}\kappa_3\beta^3+\tfrac1{2\cdot2!}\kappa_4\beta^4+\cdots$ として組み立てる。この $2\times2$ 格子では $\kappa_2=\langle H^2\rangle=\tfrac14J^2$、$\kappa_4=\tfrac14J^4-3(\tfrac14J^2)^2=\tfrac1{16}J^4$ となり、以下の $\beta^2$・$\beta^4$ 係数は誰でも手計算で検算できる——ブラックボックスではない。三角形がない格子なので、奇数次の $\beta^3$ 項が現れないはずである。
イジング模型 正方格子 2x2 の比熱の高温展開を求めて
@note: 展開を4次まで求める。$2\times2$ 格子には唯一の4-cycle（四角形数 $N_\square=1$）が存在し、これが $\beta^4$ 項の起源になる。
イジング模型 正方格子 2x2 の高温展開を4次まで求めて
@note: 今度は三角格子 $3\times3$ で同じ計算をする。この格子には三角形が存在する（$N_\triangle=8$）ため、$\langle H^3\rangle$ がゼロでない値を持つはずである（幾何学的フラストレーション）。
イジング模型 三角格子 3x3 の高温展開を求めて
@note: 三角格子について、同じ「モーメント→キュムラント→冪級数」の計算を繰り返す（上記の正方格子の比熱の注釈を参照——同じ spin-algebra モジュールによる計算であり、CASコマンドは経由しない）。正方格子（三角形なし）とは対照的に、奇数次の $\beta^3$ 項が実際に現れ、その係数は $-\kappa_3\beta^3=-\langle H^3\rangle\beta^3=-\tfrac34J^3\beta^3$ となる（イジング模型では $\langle H\rangle=0$ より $\kappa_3=\langle H^3\rangle$ が厳密に成り立つ）。
イジング模型 三角格子 3x3 の比熱の高温展開を求めて
```
