# 実践ループ サンプルシナリオ：連成振動の基準モードと基準座標変換の逆変換

実践ループ・シナリオメニュー **S-b（解析力学）** の詳細化。2 質点・バネ結合の連成振動について、
Lagrangian から運動方程式を導き、**運動方程式の和・差で基準モードに分離**し、さらに
**基準座標変換の逆変換**（L-1）を求める。

**バックエンド:** standard（Maxima）／**ドメイン:** 解析力学（analytical_mechanics）。

## 物理的背景

質量 $m$ の 2 質点を、両端のバネ $k$ と結合バネ $c$ でつないだ系（変位 $x,y$）：

$$L = \tfrac12 m\dot x^2 + \tfrac12 m\dot y^2 - \tfrac12 k x^2 - \tfrac12 k y^2 - \tfrac12 c(x-y)^2$$

運動方程式：

$$m\ddot x + (k{+}c)x - c y = 0,\qquad m\ddot y + (k{+}c)y - c x = 0$$

**基準座標** $Q_1=x+y$（対称）, $Q_2=x-y$（反対称）で対角化：

$$m\ddot Q_1 + k Q_1 = 0\ (\omega_1^2=k/m),\qquad m\ddot Q_2 + (k{+}2c)Q_2 = 0\ (\omega_2^2=(k{+}2c)/m)$$

## セッション設定・記号宣言

- モード **standard（Maxima）**、ドメイン **解析力学**。速度は **`\dot{x}`（数式記法）** で入力。
  解析力学ドメインでは $x,\dot x,\ddot x$ が独立アトムとして扱われる（[doc 19](../developer_manuals/19_analytical_mechanics_independent_variables.md)）。

```
x を時間依存関数として定義する
y を時間依存関数として定義する
m と k と c をスカラーとする
```

## 入力ステップ

### ステップ 1：Lagrangian

```
Lag = (1/2)*m*\dot{x}^2 + (1/2)*m*\dot{y}^2 - (1/2)*k*x^2 - (1/2)*k*y^2 - (1/2)*c*(x - y)^2
→ E1
```

### ステップ 2：$x$ の運動方程式

```
E1 を \dot{x} で偏微分して          → E2 = m\dot{x}
E2 を t で全微分して                  → E3 = m\ddot{x}
E1 を x で偏微分して                → E4 = -c(x-y) - k x
EOMx = E3 - E4                      → E5 = m\ddot{x} - c y + (k+c)x
```

### ステップ 3：$y$ の運動方程式

```
E1 を \dot{y} で偏微分して          → E6 = m\dot{y}
E6 を t で全微分して                  → E7 = m\ddot{y}
E1 を y で偏微分して                → E8 = c(x-y) - k y
EOMy = E7 - E8                      → E9 = m\ddot{y} + (k+c)y - c x
```

### ステップ 4：基準モード（運動方程式の和・差）

```
sym  = E5 + E9   → E10 = m\ddot{y} + m\ddot{x} + k y + k x     （= mQ̈₁ + kQ₁ = 0,  ω₁²=k/m）
anti = E5 - E9   → E11 = m\ddot{x} - m\ddot{y} + (k+2c)x - (k+2c)y  （= mQ̈₂ + (k+2c)Q₂ = 0,  ω₂²=(k+2c)/m）
```

**物理照合:** 和では結合 $c$ が相殺して $k$ のみ（同相モード、低周波）、差では $k+2c$（逆相モード、高周波）。

### ステップ 5：基準座標変換の逆変換（L-1）

順変換 $Q_1=x_1+x_2,\ Q_2=x_1-x_2$ を旧座標について解く：

```
Q_1 = x_1 + x_2
Q_2 = x_1 - x_2
E12, E13 を x_1, x_2 について逆変換して
→ x_1 = (Q_1 + Q_2)/2
→ x_2 = (Q_1 - Q_2)/2
```

これで基準座標 $Q$ から元の座標 $x$ への逆変換が得られる。

## カバレッジ

- **Euler-Lagrange ワークフロー**（$\partial_{\dot q}$ → $d/dt$ → $\partial_q$ → EOM）を 2 座標で。
- **運動方程式の線形結合**（和・差）による基準モード分離。EOM 引き算の符号正しさ。
- **L-1 線形基底変換の逆変換**（基準座標変換 $Q\to x$）。
- バックエンド：Maxima、ドメイン 解析力学（独立アトム・モデル＝[doc 19](../developer_manuals/19_analytical_mechanics_independent_variables.md)）。

## 関連

- 速度結合のある発展版：[scenario_double_pendulum_lagrangian.md](scenario_double_pendulum_lagrangian_jp.md)（S-d）。
- L-1 一般論：[user_guide.md](../user_guide.md) §線型変換、[18_linear_transform_design.md](../developer_manuals/18_linear_transform_design.md)。
- 設計と実装：[doc 19](../developer_manuals/19_analytical_mechanics_independent_variables.md)（独立アトム・モデル）、
  [open_items.md](../../KNOWN_ISSUES.md) §AM-1（経緯）。

## generator コマンド（この `scenario` ブロックが生成の単一ソース）

```scenario domain=analytical_mechanics
@note: 両端の外側バネ（定数 $k$）と1つの結合バネ（定数 $c$）でつながれた、質量 $m$ の2質点系を扱う。2質点の変位をそれぞれ $x,y$ とする。
x を時間依存関数として定義する
y を時間依存関数として定義する
m と k と c をスカラーとする
@note: ラグランジアン $L=$運動エネルギー$-$ポテンシャルエネルギーを定義する。ポテンシャル項は、各質点の外側バネによる $\frac12kx^2,\ \frac12ky^2$ と、変位差 $(x-y)$ に比例する結合バネのエネルギー $\frac12c(x-y)^2$ からなる。
Lag = (1/2)*m*\dot{x}^2 + (1/2)*m*\dot{y}^2 - (1/2)*k*x^2 - (1/2)*k*y^2 - (1/2)*c*(x - y)^2
@note: オイラー・ラグランジュ方程式 $\frac{d}{dt}\frac{\partial L}{\partial\dot x}-\frac{\partial L}{\partial x}=0$ に従い、$x$ の運動方程式を導く：$L$ を $\dot x$ で偏微分 → その結果を $t$ で全微分 → $L$ を $x$ で偏微分 → 両者の差を取る、という手順。
E1 を \dot{x} で偏微分して
E2 を t で全微分して
E1 を x で偏微分して
EOMx = E3 - E4
@note: 同じ手順で $y$ の運動方程式を導く。
E1 を \dot{y} で偏微分して
E6 を t で全微分して
E1 を y で偏微分して
EOMy = E7 - E8
@note: 2つの運動方程式の和と差を取ることで連立方程式を分離する（基準モード分解）。和では結合定数 $c$ が相殺して対称モード（低周波、$\omega_1^2=k/m$）、差では $k+2c$ が現れる反対称モード（高周波、$\omega_2^2=(k+2c)/m$）が得られる。
sym = E5 + E9
anti = E5 - E9
@note: 基準座標 $Q_1=x+y$（対称モード座標）、$Q_2=x-y$（反対称モード座標）への変換を定義する。
Q_1 = x_1 + x_2
Q_2 = x_1 - x_2
@note: この変換を逆に解き、基準座標 $Q_1,Q_2$ から元の変位 $x,y$（式中では $x_1,x_2$）を表す式を得る。
E12, E13 を x_1, x_2 について逆変換して
```
