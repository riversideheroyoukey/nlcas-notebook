# 実践ループ サンプルシナリオ：与えられた E,B で Maxwell 方程式の整合確認

実践ループ・シナリオメニュー **S-f（電磁気）** の詳細化。具体的な電場 $\boldsymbol{E}$・磁場 $\boldsymbol{B}$ を
与え、ベクトル解析（div / curl）で **Maxwell 方程式が相互に整合する**こと、さらに整合条件として
**分散関係 $\omega=ck$** が導かれることを確認する。

**バックエンド:** standard（Maxima）／**ドメイン:** 電磁気（electromagnetism）。デカルト座標。

## 物理的背景

真空・無源の Maxwell 方程式（SI、$\mu_0\varepsilon_0=1/c^2$）：

$$\nabla\!\cdot\!\boldsymbol{E}=0,\quad
  \nabla\!\cdot\!\boldsymbol{B}=0,\quad
  \nabla\!\times\!\boldsymbol{E}=-\frac{\partial \boldsymbol{B}}{\partial t},\quad
  \nabla\!\times\!\boldsymbol{B}=\frac{1}{c^2}\frac{\partial \boldsymbol{E}}{\partial t}$$

**+x 方向に進む直線偏波の平面波**（$E_0=1$）：

$$\boldsymbol{E}=(0,\ \cos(kx-\omega t),\ 0),\qquad
  \boldsymbol{B}=\Bigl(0,\ 0,\ \tfrac{1}{c}\cos(kx-\omega t)\Bigr)$$

2 つの発散は恒等的に 0、2 つの回転方程式はともに整合条件 $\omega=ck$ を与える。

## ベクトル場への成分の与え方（★このシナリオの肝）

`define_vector_field` は**成分を抽象関数として登録するだけ**で、引数に成分式は取らない。
具体成分は**宣言後に H-1 等式形式（`E_y = ...`）で各成分に代入して**与える（[doc 13 §2](../developer_manuals/13_vector_field_design_spec.md)・
[user_guide「ベクトル解析」§3](../user_guide.md)）。座標 (x,y,z) 以外の自由シンボル（$t,k,\omega,c$）も使える。

## 手順と期待値

### 1. 場の宣言と成分付与

| 入力（自然言語 / 等式） | 効果 |
|---|---|
| `E をベクトル場として定義する` | 成分 `E_x,E_y,E_z` を抽象登録 |
| `B をベクトル場として定義する` | 成分 `B_x,B_y,B_z` を抽象登録 |
| `E_x = 0` / `E_y = cos(k*x - w*t)` / `E_z = 0` | $\boldsymbol{E}$ の成分を束縛 |
| `B_x = 0` / `B_y = 0` / `B_z = cos(k*x - w*t)/c` | $\boldsymbol{B}$ の成分を束縛 |

### 2. 発散（Gauss・単極子なし）

ナブラ記法を**式として入力**するとその場で計算される（`div E` / `div(E)` でも可）。

| 入力 | 期待結果 |
|---|---|
| `\nabla \cdot E` | $\nabla\!\cdot\!\boldsymbol{E}=0$ |
| `\nabla \cdot B` | $\nabla\!\cdot\!\boldsymbol{B}=0$ |

### 3. 回転（Faraday・Ampère の左辺）

| 入力 | 期待結果 |
|---|---|
| `\nabla \times E`（`curl E` / `rot E` も可）| $\nabla\!\times\!\boldsymbol{E}=[0,\,0,\,-k\sin(kx-\omega t)]$ |
| `\nabla \times B` | $\nabla\!\times\!\boldsymbol{B}=[0,\,k\sin(kx-\omega t)/c,\,0]$ |

### 4. 時間微分 $\partial\boldsymbol{B}/\partial t$ を**成分ごと**に作って再組立て

> **設計上の注意（重要）:** $\partial/\partial t$ は**ベクトル演算子ではない**。場の名前 `B` をそのまま
> `t で微分` してもベクトルとしては機能しない（便宜的書法は採らない）。**正攻法は「各成分を時間微分し、
> 結果を新しいベクトルに組み直す」**。3 成分をまとめて扱えないのは仕様として許容し、成分ごとに処理する。

> **偏微分で指定する（重要）:** 成分 $B_z=\cos(kx-\omega t)/c$ は $x$ と $t$ の多変数。`t で微分` だけだと
> 全微分か偏微分かが曖昧で確認を求められる（∂/d 判定）。$\partial\boldsymbol{B}/\partial t$ は $x$ 固定の
> **偏微分**なので、必ず **`t で偏微分して`** と書く。

| 入力 | 期待結果 |
|---|---|
| `B_x を t で偏微分して` / `B_y を t で偏微分して` | $0$ / $0$ |
| `B_z を t で偏微分して` | $\dfrac{\partial B_z}{\partial t}=\dfrac{\omega}{c}\sin(kx-\omega t)$ |
| `dBdt をベクトル場として定義する` ＋ `dBdt_x = 0` / `dBdt_y = 0` / `dBdt_z = w*sin(k*x - w*t)/c` | $\partial\boldsymbol{B}/\partial t$ を**ベクトルとして再組立て** |
| （確認）`\nabla \cdot dBdt` | $0$（組み立てたベクトルが後続計算に使えることの確認。div は式入力でそのまま計算される）|

### 5. Faraday/Ampère の整合 → 分散関係 $\omega=ck$（**成分ごとに比較**）

ベクトル等式 $\nabla\!\times\!\boldsymbol{E}=-\partial\boldsymbol{B}/\partial t$ は**成分ごとの等式**にして解く
（まとめての比較は行わない＝筋として成分ごと）。非自明な $z$ 成分：

| 入力 | 期待結果 |
|---|---|
| `-k*sin(k*x - w*t) = -w*sin(k*x - w*t)/c` を登録（$(\nabla\!\times\!\boldsymbol{E})_z=(-\partial\boldsymbol{B}/\partial t)_z$）| Faraday の $z$ 成分等式 |
| 前式を `w について解いて` | $\omega=ck$（分散関係。もう一根 $kx/t$ は $\sin=0$ の非物理解）|

Ampère も同様に $y$ 成分等式 $(\nabla\!\times\!\boldsymbol{B})_y=(\tfrac{1}{c^2}\partial\boldsymbol{E}/\partial t)_y$ を
作って解けば、独立に同じ $\omega=ck$ を与える。両 curl 方程式が整合し、Maxwell の 4 式が相互に整合する。

## 確認された挙動・注意点

- **既存の div / curl / 微分 / solve で完結**（追加機能不要）。デカルト座標の `nb_div`/`nb_curl_*` が
  成分式を引数に取り、H-1 で束縛した値で偏微分の和を展開する。
- **成分は等式で与える**（`define_vector_field` の引数には渡せない）。再現時に最も混乱しやすい点。
- **$\partial/\partial t$ はベクトルに直接効かせない**。成分ごとに `t で微分` し、結果を新しいベクトル場に
  組み直して使う（上記 §4）。組み立てたベクトルは `div`/`curl` 等の後続計算に使える。
- **ベクトル等式の比較は成分ごと**に行う（z 成分等式を登録 → `w について解く` → $\omega=ck$）。
  3 成分まとめての比較・時間微分は提供しないが、成分ごとで完結するため許容範囲（設計判断）。
- 時間 $t$・波数 $k$・角振動数 $\omega$・$c$ は自由シンボル。

**確認日:** 2026-06-22／2026-06-23 再確認（実機 Maxima）：div E=div B=0、curl E=$[0,0,-k\sin]$、
curl B=$[0,k\sin/c,0]$、成分時間微分 $\partial B_z/\partial t=\omega\sin/c$ を `dBdt` に再組立て→`div(dBdt)=0`、
Faraday $z$ 成分等式を解いて $\omega=ck$ を確認。

## generator コマンド（この `scenario` ブロックが生成の単一ソース）

```scenario domain=electromagnetism
@note: 電磁場を扱う。電場 $\mathbf E$ と磁場 $\mathbf B$ をベクトル場として準備する（成分は抽象のままで登録し、後で具体的な式を割り当てる）。
E をベクトル場として定義する
B をベクトル場として定義する
@note: $+x$ 方向に進む直線偏波の平面波として、電場の各成分（振幅1、波数 $k$、角振動数 $\omega$）を割り当てる。
E_x = 0
E_y = cos(k*x - w*t)
E_z = 0
@note: 対応する磁場の各成分を割り当てる。
B_x = 0
B_y = 0
B_z = cos(k*x - w*t)/c
@note: 電場の発散 $\nabla\cdot\mathbf E$ を計算する。真空中（電荷なし）では恒等的にゼロになるはずである（Gaussの法則）。
\nabla \cdot E
@note: 磁場の発散 $\nabla\cdot\mathbf B$ を計算する。磁荷が存在しないため恒等的にゼロになるはずである。
\nabla \cdot B
@note: 電場の回転 $\nabla\times\mathbf E$ を計算する（Faradayの法則の左辺）。
\nabla \times E
@note: 磁場の回転 $\nabla\times\mathbf B$ を計算する（Ampèreの法則の左辺）。
\nabla \times B
@note: $\partial/\partial t$ はベクトル演算子として直接扱えないため、成分ごとに時間偏微分する。ここではゼロでない成分 $B_z$ を $t$ で偏微分する。
B_z を t で偏微分して
@note: 求めた偏微分の結果を、新しいベクトル場として組み立てる。ここで `dBdt` という名前を付けるが、これは $d\mathbf B/dt$ を表すために選んだ単なる識別子であり（CASの予約語や特別な構文ではない、他の名前でも同様に動く）、成分ごとに割り当て直す。
dBdt をベクトル場として定義する
dBdt_x = 0
dBdt_y = 0
dBdt_z = w*sin(k*x - w*t)/c
@note: 組み立てた $d\mathbf B/dt$ が後続の計算（発散など）に使えることを、その発散を計算して確認する。
\nabla \cdot dBdt
@note: Faradayの法則 $\nabla\times\mathbf E=-\partial\mathbf B/\partial t$ の $z$ 成分どうしを等式として登録する（両辺の $z$ 成分を比較する）。
-k*sin(k*x - w*t) = -w*sin(k*x - w*t)/c
@note: この等式を角振動数 $\omega$ について解き、整合条件（分散関係）$\omega=ck$ を導く。
E16 を w について解いて
```
