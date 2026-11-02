# 実践ループ サンプルシナリオ：二重振り子の解析力学（Euler-Lagrange）

実践ループ・シナリオメニュー **S-d（解析力学）** の詳細化。二重振り子の Lagrangian から
**Euler-Lagrange 方程式**を導出する。単振り子（[scenario_pendulum_lagrangian.md](scenario_pendulum_lagrangian_jp.md)）
の 2 自由度・**速度結合あり**への発展で、$\partial L/\partial\dot q$ → $\frac{d}{dt}$ → $\partial L/\partial q$ →
運動方程式 の一連を Maxima で辿る。

**バックエンド:** standard（Maxima）／**ドメイン:** 解析力学（analytical_mechanics）。

## 物理的背景

質量 $m_1,m_2$・長さ $l_1,l_2$ の二重振り子（角度 $\theta_1,\theta_2$）の Lagrangian：

$$L = \tfrac12(m_1{+}m_2)l_1^2\dot\theta_1^2 + \tfrac12 m_2 l_2^2\dot\theta_2^2
      + m_2 l_1 l_2\,\dot\theta_1\dot\theta_2\cos(\theta_1{-}\theta_2)
      + (m_1{+}m_2)g l_1\cos\theta_1 + m_2 g l_2\cos\theta_2$$

$\theta_1$ の Euler-Lagrange 方程式（標準形）：

$$(m_1{+}m_2)l_1^2\ddot\theta_1 + m_2 l_1 l_2\ddot\theta_2\cos(\theta_1{-}\theta_2)
  + m_2 l_1 l_2\dot\theta_2^2\sin(\theta_1{-}\theta_2) + (m_1{+}m_2)g l_1\sin\theta_1 = 0$$

これを導出する。

## セッション設定・記号宣言

- モード **standard（Maxima）**、ドメイン **解析力学**。
- 角度は時間依存、係数はスカラーとして**式の前に宣言**する。
- 速度は **`\dot{a}`（数式記法）** で入力する。解析力学ドメインでは $a,\dot a,\ddot a$ が
  独立アトムとして扱われ、CAS 文法（`diff(a,t)`）を見せる必要はない（[doc 19](../developer_manuals/19_analytical_mechanics_independent_variables.md)）。

```
a を時間依存関数として定義する        # θ₁
b を時間依存関数として定義する        # θ₂
M と m と p と q と g をスカラーとして定義する   # m₁,m₂,l₁,l₂,g
```

## 入力ステップ

### ステップ 1：Lagrangian の登録

```
Lag = (1/2)*(M + m)*p^2*\dot{a}^2 + (1/2)*m*q^2*\dot{b}^2 + m*p*q*\dot{a}*\dot{b}*\cos(a - b) + (M + m)*g*p*\cos(a) + m*g*q*\cos(b)
→ E1
```

### ステップ 2：一般化運動量 $\partial L/\partial\dot\theta_1$

```
E1 を \dot{a} で偏微分して
→ E2 = m p q \dot{b}\cos(b-a) + (m+M)p^2\dot{a}
```

速度結合 $\dot\theta_1\dot\theta_2\cos$ から $\dot b\cos(b-a)$ 項が出る。

### ステップ 3：$\frac{d}{dt}\!\left(\partial L/\partial\dot\theta_1\right)$

```
E2 を t で全微分して
→ E3 = -m p q \dot{b}(\dot{b}-\dot{a})\sin(b-a) + m p q \ddot{b}\cos(b-a) + (m+M)p^2\ddot{a}
```

### ステップ 4：$\partial L/\partial\theta_1$

```
E1 を a で偏微分して
→ E4 = m p q \dot{a}\dot{b}\sin(b-a) - g(m+M)p\sin a
```

> 速度 $\dot a,\dot b$ を含む項が正しく残る。独立アトム・モデルでは $\partial L/\partial a$ は
> $a$ についての素直な偏微分（$\dot a,\dot b$ は独立なので保存される）。

### ステップ 5：運動方程式 $\frac{d}{dt}\partial_{\dot\theta_1}L - \partial_{\theta_1}L$

```
EOMa = E3 - E4
→ E5 = -m p q \dot{b}^2\sin(b-a) + m p q \ddot{b}\cos(b-a) + (m+M)p^2\ddot{a} + (m+M)g p\sin a
```

**物理照合:** $\dot a\dot b\sin(b-a)$ 項が E3 と E4 で相殺し、$\sin(b-a)=-\sin(a-b)$ より
$-mpq\dot b^2\sin(b-a)=+m_2l_1l_2\dot\theta_2^2\sin(\theta_1{-}\theta_2)$。背景の標準形と一致。

### ステップ 6（任意）：$\theta_2$ の運動方程式

同様に `\dot{b}` → t 微分 → `b` 偏微分 → `EOMb = … - …` で $\theta_2$ の EL 方程式が得られる
（$\dot\theta_1^2\sin$ 項の符号が反転した対称形）。

## カバレッジ

- **速度に関する偏微分** $\partial L/\partial\dot\theta$（独立アトム $\dot\theta$ への素直な微分）。
- **時間に関する全微分** $\frac{d}{dt}$（$\ddot\theta,\dot\theta\,$積の連鎖律総和）。
- **位置に関する偏微分** $\partial L/\partial\theta$（速度結合項が正しく保存される）。
- **運動方程式の組立** $E3-E4$（負項を含む引き算の符号正しさ）。
- バックエンド：Maxima、ドメイン 解析力学（独立アトム・モデル＝[doc 19](../developer_manuals/19_analytical_mechanics_independent_variables.md)）。

## 関連

- 1 自由度版：[scenario_pendulum_lagrangian.md](scenario_pendulum_lagrangian_jp.md)。
- 連成振動（基準モード・基準座標変換の逆変換）：[scenario_coupled_oscillations.md](scenario_coupled_oscillations_jp.md)（S-b）。
- 設計と実装：[doc 19](../developer_manuals/19_analytical_mechanics_independent_variables.md)（独立アトム・モデル）、
  [open_items.md](../../KNOWN_ISSUES.md) §AM-1（経緯）。

## generator コマンド（この `scenario` ブロックが生成の単一ソース）

```scenario domain=analytical_mechanics
@note: 二重振り子を扱う。添字つき記号を避けるため、角度 $\theta_1,\theta_2$ を $a,b$、質量 $m_1,m_2$ を $M,m$、振り子の長さ $l_1,l_2$ を $p,q$、重力加速度を $g$ という記号に置き換える。$a,b$ は時間に依存する角度。
a を時間依存関数として定義する
b を時間依存関数として定義する
M と m と p と q と g をスカラーとして定義する
@note: 二重振り子のラグランジアン $L=$運動エネルギー$+$ポテンシャルエネルギーを定義する。運動エネルギー項には振り子1・2それぞれの回転エネルギーに加え、両者の角速度の積 $\dot a\dot b\cos(a-b)$ に比例する速度結合項が現れる。
Lag = (1/2)*(M + m)*p^2*\dot{a}^2 + (1/2)*m*q^2*\dot{b}^2 + m*p*q*\dot{a}*\dot{b}*\cos(a - b) + (M + m)*g*p*\cos(a) + m*g*q*\cos(b)
@note: 一般化運動量 $\partial L/\partial\dot\theta_1$ を求めるため、$L$ を $\dot a$（$\dot\theta_1$）で偏微分する。速度結合項から $\dot b$（$\dot\theta_2$）を含む項が現れる。
E1 を \dot{a} で偏微分して
@note: 求めた一般化運動量を時間で全微分し、$\frac{d}{dt}(\partial L/\partial\dot\theta_1)$ を求める。
E2 を t で全微分して
@note: オイラー・ラグランジュ方程式のもう一方の項 $\partial L/\partial\theta_1$ を求めるため、$L$ を $a$（$\theta_1$）で偏微分する。
E1 を a で偏微分して
@note: $\frac{d}{dt}(\partial L/\partial\dot\theta_1)$ から $\partial L/\partial\theta_1$ を引き、$\theta_1$ の運動方程式を組み立てる。
EOMa = E3 - E4
```
