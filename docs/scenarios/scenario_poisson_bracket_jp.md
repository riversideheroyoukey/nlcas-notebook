# 実践ループ サンプルシナリオ：ポアソン括弧（正準対の宣言 → {f,g}）

## ねらい

ハミルトン力学のポアソン括弧
$$\{f, g\} = \sum_i\left(\frac{\partial f}{\partial q_i}\frac{\partial g}{\partial p_i} - \frac{\partial f}{\partial p_i}\frac{\partial g}{\partial q_i}\right)$$
を、**正準座標対 $(q_i, p_i)$ を一度宣言**してから自然言語で計算する。基本括弧
$\{r, p_r\}=1$、中心力ハミルトニアンの保存則 $\{H, L_z\}=0$、Hamilton 運動方程式
$\{H, r\}=-p_r/m$ を確認する。

> ポアソン括弧は量子の交換子 `commutator` の**古典版**で、非可換代数を必要とせず
> **偏微分の和**として構成される。したがって Maxima がそのまま扱える（`diff` の和）。

## 正準対の宣言

ポアソン括弧は「どの座標とどの運動量が共役か」を知る必要がある。毎回書かせず、
セッションに一度宣言しておく（`SymbolContext.canonical_pairs` に保存、save/load 永続）。

```
p_r を r の共役運動量とする          ← 正準対 (r, p_r)
p_theta を theta の共役運動量とする     ← 正準対 (theta, p_theta)
```

「`r と p_r を正準対とする`」という言い方（座標, 運動量 の順）も使える。

## 入力ステップ

### ステップ 1：基本ポアソン括弧

```
r と p_r のポアソン括弧を求めて        → 1     （正準交換関係 {q,p}=1）
r と p_theta のポアソン括弧を求めて     → 0     （異なる対は可換）
```

### ステップ 2：保存則 {H, L_z} = 0

```
H = p_r^2/(2*m) + p_theta^2/(2*m*r^2) + U(r)   → 中心力ハミルトニアン（H-1 登録）
L_z = p_theta                                 → 角運動量（= 共役運動量 p_theta）
H と L_z のポアソン括弧を求めて                  → 0
```

$\{H, L_z\}=0$ は **$L_z$ が保存量**（$H$ が $\theta$ に依らない＝回転対称）であることを示す。

### ステップ 3：Hamilton 運動方程式

```
H と r のポアソン括弧を求めて        → -p_r/m     （ \dot r = {r, H} = ∂H/∂p_r ）
```

$\dot q = \{q, H\}$ なので、$\{H, r\} = -\{r, H\} = -\partial H/\partial p_r = -p_r/m$。

## オペランドの指定

`a` / `b` には次のいずれも使える。

- **E-ID**（`E4` など）
- **H-1 で登録した名前**（`H`, `L_z`）
- **素のシンボル**（`r`, `p_r`）— 式として登録していなくてよい

正準対が未宣言のまま括弧を求めると、宣言を促すメッセージを返す。

## カバレッジ

- **正準対の宣言と永続**（`共役運動量とする` / `正準対とする`、save/load 維持）。
- **ポアソン括弧** `poisson_bracket`（ルールベース、LLM 非経由、Maxima 偏微分の和）。
- 量子 `commutator` の古典対応（doc 18 §統一原理の計量保存とは別系統の微分構成）。
- バックエンド：Maxima。

## 関連

- 中心力の保存量（$d/dt$＋運動方程式の別解）：[scenario_central_force_conservation.md](scenario_central_force_conservation_jp.md)。
- 操作一覧：[operation_reference.md](../operation_reference.md) §ハミルトン力学（ポアソン括弧）。
- 設計の経緯：[open_items.md](../../KNOWN_ISSUES.md) §P-1。

## 確認された挙動・注意点（2026-07-08 再監査）

- **正準対の宣言がエクスポートされたTeXに一切表示されていなかった。** 確認メッセージは
  API レスポンスには含まれていたが、タイムラインのステップとして記録されていなかった。
  `declare_time_dependent` と同じ `engine.record_quantity_update(...)` パターンを
  適用し、確認メッセージ（「正準座標対 (r, p_r) を登録しました…」）が入力のエコーと
  次のステップの間に表示されるよう修正した。
- **`[X, Y]` ブラケット記法が、Maxima セッションでもポアソン括弧に解決するようになった。**
  以前は量子の `commutator`（julia_full 限定）にしか解決されず、Maxima では生の
  Maxima リストとして登録されてしまっていた。根本原因：「裸の数式ならそのまま登録する」
  フォールバックを迂回させる `operands_provided()` フックが、`[`/`{` で始まる
  ブラケット記法を `JuliaBackendHandler` にしか実装していなかった（このチェック自体は
  バックエンド固有の挙動ではなく、両方のオペランドがブラケット記法内に明示されている
  ことを認識するだけのものである）。Backend Encapsulation Rule に従い共有基底クラス
  `BackendHandler`（`notebook/runtime/backend_handler.py`）へ移動し、
  `llm_supervisor.py` の Phase-0 ブラケットパーサーで同じ `[X, Y]` パターンに対する
  候補として `commutator` と並べて `poisson_bracket` も試すようにした（両操作はバック
  エンドが排他的なので曖昧性は生じない）。`r と p_r のポアソン括弧を求めて` と
  `[r, p_r]` は今や等価な入力として両方使える。

## generator コマンド（この `scenario` ブロックが生成の単一ソース）

```scenario domain=classical_mechanics
@note: ポアソン括弧を計算するには、どの座標とどの運動量が共役（正準対）かをあらかじめ宣言しておく必要がある。動径座標 $r$ の共役運動量 $p_r$ を宣言する。
p_r を r の共役運動量とする
@note: 角度座標 $\theta$ の共役運動量 $p_\theta$ を宣言する。
p_theta を theta の共役運動量とする
@note: 基本的なポアソン括弧 $\{r,p_r\}$ を求める。正準対どうしなので正準交換関係 $\{q,p\}=1$ により1になるはずである。
r と p_r のポアソン括弧を求めて
@note: 中心力ハミルトニアン $H=p_r^2/(2m)+p_\theta^2/(2mr^2)+U(r)$（動径・角度方向の運動エネルギーとポテンシャル $U(r)$ の和）を定義する。
H = p_r^2/(2*m) + p_theta^2/(2*m*r^2) + U(r)
@note: 角運動量 $L_z=p_\theta$（角度の共役運動量そのもの）を定義する。
L_z = p_theta
@note: ハミルトニアン $H$ と角運動量 $L_z$ のポアソン括弧を求める。$H$ が角度 $\theta$ に依らない（回転対称）ので、これがゼロになれば $L_z$ は保存量であることが確認できる。
H と L_z のポアソン括弧を求めて
@note: ハミルトンの運動方程式 $\dot q=\{q,H\}$ を用いて動径座標 $r$ の時間発展を求めるため、$H$ と $r$ のポアソン括弧を求める。
H と r のポアソン括弧を求めて
```
