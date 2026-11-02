# 実践ループ サンプルシナリオ：中心力ポテンシャルの保存量（角運動量・エネルギー）

実践ループ・シナリオメニュー **S-h（古典力学）** の詳細化。極座標 $(r,\theta)$ における中心力運動について、
**角運動量** $L_z$ と **エネルギー** $E$ が運動の保存量であることを、保存量を時間で微分し
**運動方程式を代入すると 0 になる**ことで確認する。

**バックエンド:** standard（Maxima）／**ドメイン:** 古典力学（classical_mechanics）。

> 解析力学ドメイン（[doc 19](../developer_manuals/19_analytical_mechanics_independent_variables.md)）が
> $q,\dot q$ を独立アトムとして扱うのに対し、古典力学（Newton 力学）ドメインでは
> 速度は **従属量** $\dot q = dq/dt$ として扱われ、Maxima の depends 形
> `'diff(q,t,N)` で表現される。`\dot{r}` などの数式記法で入力する（CAS 構文は不要）。

## 物理的背景

質量 $m$ の質点が中心力ポテンシャル中を平面運動する。極座標 $(r,\theta)$ での

- 角運動量 $L_z = m r^2 \dot\theta$
- エネルギー $E = \tfrac12 m(\dot r^2 + r^2\dot\theta^2) + U(r)$（ここでは Kepler ポテンシャル $U=-k/r$）

運動方程式（Euler-Lagrange）：

$$\ddot\theta = -\frac{2\dot r\dot\theta}{r},\qquad
  \ddot r = r\dot\theta^2 - \frac{k}{m r^2}$$

中心力では $\partial L/\partial\theta = 0$ なので $L_z$ は保存し、力が時間に陽に依らないので $E$ も保存する。

## セッション設定・記号宣言

- モード **standard（Maxima）**、ドメイン **古典力学**。速度は **`\dot{r}`（数式記法）** で入力。

```
r を時間依存関数として定義する
theta を時間依存関数として定義する
m と k をスカラーとする
```

## 入力ステップ

### ステップ 1：角運動量の保存

```
L_z = m r^2 \dot{theta}                              → E1
E1 を t で全微分して                                    → E2 = m r²θ̈ + 2 m r ṙ θ̇
E2 の \ddot{theta} を -2 \dot{r} \dot{theta} / r に置換して  → E3 = 0
```

$\theta$ の運動方程式 $\ddot\theta = -2\dot r\dot\theta/r$ を代入すると $dL_z/dt = 0$。**角運動量は保存量**。

### ステップ 2：エネルギーの保存

```
E_tot = (1/2) m (\dot{r}^2 + r^2 \dot{theta}^2) - k / r   → E4
E4 を t で全微分して                                          → E5 = dE/dt
E5 の \ddot{theta} を -2 \dot{r} \dot{theta} / r に置換して   → E6
E6 の \ddot{r} を r \dot{theta}^2 - k / (m r^2) に置換して    → E7 = 0
```

両方の運動方程式（$\ddot\theta$, $\ddot r$）を代入すると $dE/dt = 0$。**エネルギーは保存量**。

> **「t で全微分して」と明示する理由:** $L_z$ や $E$ のように時間依存量を含む式を
> $t$ で微分するとき、偏微分 $\partial/\partial t$（他を固定）と全微分 $d/dt$（連鎖律）は
> 別物です。システムは推測せず、どちらかを利用者に確認します（インタラクティブ時は
> 選択ダイアログ）。スクリプト・テストなど非インタラクティブでは「全微分」または
> 「偏微分」の明示が必須です。ここでは連鎖律を含む $d/dt$ を使うため「全微分」と書きます。

## カバレッジ

- **保存量の検証ワークフロー**（保存量 → $d/dt$ → 運動方程式の代入 → 0）を 2 つの保存量で。
- 古典力学（Newton 力学）ドメイン：速度・加速度は depends 形 `'diff(q,t,N)`。
  `\dot{r}`/`\ddot{theta}` 入力 → atom → depends 形に統一され、相殺が成立する。
- **導関数項の代入**（`\ddot{theta}` を運動方程式で置換）。複数トークン RHS
  （`-2 \dot{r} \dot{theta} / r`）の代入。
- 括弧前のスペース積（`m (\dot{r}^2 + …)`、`k / (m r^2)`）。
- バックエンド：Maxima、ドメイン 古典力学。

## 関連

- 解析力学（独立アトム・モデル）版の Euler-Lagrange：[scenario_coupled_oscillations.md](scenario_coupled_oscillations_jp.md)（S-b）、
  [scenario_double_pendulum_lagrangian.md](scenario_double_pendulum_lagrangian_jp.md)（S-d）。
- 設計：[doc 19](../developer_manuals/19_analytical_mechanics_independent_variables.md)（古典力学 vs 解析力学のドメイン分離）。
- 入力規約：[user_guide.md](../user_guide.md) §数式入力の注意点（スペース＝乗算、`\dot{}` 記法）。

## generator コマンド（この `scenario` ブロックが生成の単一ソース）

```scenario domain=classical_mechanics
@note: 極座標 $(r,\theta)$ で中心力ポテンシャル中を運動する質量 $m$ の質点を扱う。$k$ はケプラーポテンシャル $U=-k/r$ の強さを表すパラメータ。
r を時間依存関数として定義する
theta を時間依存関数として定義する
m と k をスカラーとする
@note: 角運動量 $L_z=mr^2\dot\theta$（質量×動径の2乗×角速度）を定義する。
L_z = m r^2 \dot{theta}
@note: 角運動量が保存量かどうかを調べるため、$L_z$ を時間 $t$ で全微分する（全微分は連鎖律を含み、$r,\theta$ の時間変化も考慮する）。
E1 を t で全微分して
@note: 運動方程式 $\ddot\theta=-2\dot r\dot\theta/r$ を $dL_z/dt$ に代入する。結果がゼロになれば角運動量は保存量である。
E2 の \ddot{theta} を -2 \dot{r} \dot{theta} / r に置換して
@note: エネルギー $E=\frac12 m(\dot r^2+r^2\dot\theta^2)-k/r$（運動エネルギーとケプラーポテンシャルの和）を定義する。
E_tot = (1/2) m (\dot{r}^2 + r^2 \dot{theta}^2) - k / r
@note: エネルギーが保存量かどうかを調べるため、$E$ を時間 $t$ で全微分する。
E4 を t で全微分して
@note: 運動方程式 $\ddot\theta=-2\dot r\dot\theta/r$ を代入する。
E5 の \ddot{theta} を -2 \dot{r} \dot{theta} / r に置換して
@note: もう一方の運動方程式 $\ddot r=r\dot\theta^2-k/(mr^2)$ も代入する。両方代入した結果がゼロになれば、エネルギーは保存量である。
E6 の \ddot{r} を r \dot{theta}^2 - k / (m r^2) に置換して
```
