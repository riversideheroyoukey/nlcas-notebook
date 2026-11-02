# 実践ループ サンプルシナリオ：回転座標系の有効ハミルトニアン（S-m）

実践ループ・シナリオメニュー **S-m（磁気共鳴）** の詳細化。NMR/ESR で標準的な**回転座標系変換**を、
**一般のスピン操作の連続**で完動させる。カスタムのスピンハミルトニアンを登録し、回転座標系へ変換し、
有効磁場を求め、歳差の運動方程式を交換子で得る。スピン代数は CAS を経由しない（自己完結の Python 実装）。

> **設計方針（重要）:** シナリオは*潜在的な不具合を炙り出す*ためのものなので、特定シナリオ専用に答えを
> 焼き込んだ機能ではなく、**他でも使える一般操作**（スピン式パース・**式の加算**・**係数ベクトルのノルム**・
> 一般 `commutator`）の連続で完動させる。「回転枠に乗る」＝フレーム項 $+\omega S^z$ を**足す**だけなので
> 専用の変換操作は設けず明示的な加算で書く。有効磁場の大きさも $H=-\mathbf B\cdot\mathbf S$ の成分（係数）
> から直接ノルムを取る（場ベクトルを別途作らない）。離調 $\Omega=\omega_0-\omega$ は表示上の略記で、本
> シナリオは加算が返す $(\omega-\omega_0)$ の形をそのまま示す（$\Omega$ への代入は行わない）。

**バックエンド:** standard（Maxima）／**ドメイン:** 磁気共鳴（magnetic_resonance）。

## 物理的背景

回転磁場 $\omega$ で回転する座標系（$U=e^{-i\omega t S^z}$）へ移ると、Zeeman 項に $+\omega S^z$ の枠項が
加わり、実験室系の

$$H_{\mathrm{lab}} = -\omega_0 S^z - \omega_1 S^x$$

は、離調 $\Omega=\omega_0-\omega$ を使って

$$H_{\mathrm{rot}} = -\Omega S^z - \omega_1 S^x$$

になる。スピンが感じる**有効磁場**の大きさは $\omega_{\mathrm{eff}}=\sqrt{\Omega^2+\omega_1^2}$、
回転系での歳差は運動方程式 $[H_{\mathrm{rot}},S^z]=i\omega_1 S^y$ で表される（共鳴 $\Omega=0$ では
$\omega_1$ まわりの Rabi 歳差）。

## 手順と期待値

導出を**4 つの一般操作**（各々が独立の入力＝独立の YOU カード）で順に解いていく。各コマンドが 1 つの式を返す。

| 段 | 入力（自然言語） | 操作 | 期待結果 |
|---|---|---|---|
| 1 | `H_{lab} = -\omega_0 S^z_1 - \omega_1 S^x_1` | スピン H 登録（式パース）| $H_{\mathrm{lab}} = -\omega_0 S^z_1 - \omega_1 S^x_1$（E1）|
| 2 | `H_{rot} = E1 + \omega S^z_1` | **式の加算**（フレーム項を足す）| $H_{\mathrm{rot}} = (\omega-\omega_0) S^z_1 - \omega_1 S^x_1$（E2）|
| 3 | `E2 の有効磁場の大きさを求めて` | **係数ベクトルのノルム** | $\omega_{\mathrm{eff}} = \sqrt{(\omega-\omega_0)^2+\omega_1^2}$（E3）|
| 4 | `E2 と S^z_1 の交換子を求めて` | 一般 `commutator` | $[H_{\mathrm{rot}}, S^z_1] = i\omega_1 S^y_1$（E4）|

カスタムのスピンハミルトニアン（RHS がスピン式としてパースできる H-1 形）を E1 に登録し、**フレーム項
$+\omega S^z$ を E1 に足して** E2（＝「回転枠に乗る」の素直な演算）、E2 は $-\mathbf B\cdot\mathbf S$ の成分表示
なので**その係数から直接ノルム**を取って E3、一般 `commutator` で歳差 EOM の E4 を得る。**いずれも他のスピン式
にも使える一般操作**で、特定シナリオ専用の焼き込みではない。

## 確認された挙動・注意点

- **スピン代数は CAS 非経由**（`notebook/runtime/spin_algebra.py`）。ハミルトニアンは構造化データ（term
  list）として保持され、`parse_spin_expression` で E2 を字句解析して係数（場の成分）を直接読める。
  交換子・ノルム・加算はいずれも term list 上の一般演算（S-a/S-p と同系）。
- 各コマンドが 1 式を登録する（`register_dispersion` 経由・派生量なので LHS エイリアスなし）。
- 離調 $\Omega=\omega_0-\omega$ を使いたい場合は、別途 $(\omega-\omega_0)$ を $\Omega$ と置く一般代入を 1 段
  挟めばよい（本シナリオは加算が返す素の形を示す）。共鳴 $\omega=\omega_0$ では $H_{\mathrm{rot}}=-\omega_1 S^x$
  となり、$\omega_1$ まわりの Rabi 歳差になる。
- 英語版も同じパイプラインで実行される（`Find the magnitude of the effective field of E2`・`Find the
  commutator of E2 and S^z_1` 等、NL は i18n 化済み。式 `H_{rot} = E1 + \omega S^z_1` は英日共通）。
  → [scenario_rotating_frame_en.md](scenario_rotating_frame_en.md)

**再現例:** [`domains/magnetic_resonance/examples/`](../../domains/magnetic_resonance/examples/)
（`scenario_rotating_frame_{jp,en}.{json,tex}`）。

## generator コマンド（この `scenario` ブロックが生成の単一ソース）

```scenario domain=magnetic_resonance
@note: 実験室系のスピンハミルトニアン $H_{\rm lab}=-\omega_0S^z_1-\omega_1S^x_1$（$\omega_0$: Zeeman周波数、$\omega_1$: 回転磁場の強さ）を登録する。
H_{lab} = -\omega_0 S^z_1 - \omega_1 S^x_1
@note: 角周波数 $\omega$ で回転する座標系に移るため、フレーム項 $+\omega S^z$ を実験室系のハミルトニアンに足し、回転系でのハミルトニアン $H_{\rm rot}$ を得る。
H_{rot} = E1 + \omega S^z_1
@note: $H_{\rm rot}=-\mathbf B\cdot\mathbf S$ の形の係数（場の成分）から、スピンが感じる有効磁場の大きさ $\omega_{\rm eff}=\sqrt{(\omega-\omega_0)^2+\omega_1^2}$ を直接求める。
E2 の有効磁場の大きさを求めて
@note: 回転系でのハミルトニアン $H_{\rm rot}$ と $S^z$ の交換子を求め、回転系での歳差運動の運動方程式を得る。
E2 と S^z_1 の交換子を求めて
```
