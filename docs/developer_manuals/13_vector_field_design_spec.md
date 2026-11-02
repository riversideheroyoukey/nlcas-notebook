# ベクトル場・スカラー場 設計仕様

**作成:** 2026-06-03  
**対象バックエンド:** 現状は Maxima セッションで実装（アーキテクチャ上の制約ではなく実装範囲の選択）。
ベクトル解析の本体（div/grad/curl/Laplacian）はラメ係数 $h_1,h_2,h_3$ と偏微分で書けるバックエンド非依存の数学であり、
Symbolics.jl の `Differential`/`jacobian` で Julia へ拡張可能。Maxima 固有なのは `depends` による関数従属宣言と
`nb_div` 等の事前定義の2点のみで、いずれも Julia には自然な代替がある（Symbolics は変数が記号的で `depends` 自体が不要）。
凝縮系・量子多体（Julia 担当ドメイン）にも空間勾配の需要がある（Ginzburg–Landau の $|\nabla\psi|^2$ 項、実空間 BdG、
スピンテクスチャ等）。Julia 拡張は `docs/open_items.md` F-3 に将来項目として記載。
なお §6 ケース2 `change_integration_variables` は既に Maxima・Julia 両対応。  
**関連 open_items:** F-3, F-4, F-5

---

## 1. 動機

電磁気学・流体力学・古典力学では、$\boldsymbol{A}(\boldsymbol{r}, t)$ のような**ベクトル場**を
一体として扱い、$\nabla \cdot \boldsymbol{A}$, $\nabla \times \boldsymbol{A}$, $\nabla^2 \boldsymbol{A}$
のような演算を行いたい。

現状では成分 `Ax`, `Ay`, `Az` を個別にスカラー関数として宣言する方法しかないが、
これは記述が煩雑でユーザーに CAS の内部実装を意識させてしまう。

---

## 2. ベクトル場宣言 `define_vector_field`

### ユーザー入力

```
A(x,y,z,t) をベクトル場として定義する
A(r,theta,phi) をベクトル場として定義する（球座標セッション）
```

### 内部動作

1. **成分の SymbolContext 登録**  
   座標系に応じたサフィックスで成分シンボルを登録：

   | 座標系 | 成分名 |
   |---|---|
   | デカルト | `A_x`, `A_y`, `A_z` |
   | 球座標 | `A_r`, `A_theta`, `A_phi` |
   | 円柱座標 | `A_r`, `A_phi`, `A_z` |

2. **Maxima への `depends` 発行**  
   `depends([A_x, A_y, A_z], [x,y,z,t])`

3. **Quantity モデルへの保存**
   ```python
   Quantity(
       name="A",
       kind="vector_field",
       components=["A_x", "A_y", "A_z"],
       coordinate_system="cartesian"
   )
   ```

### 表示

`\boldsymbol{A}(x,y,z,t)` として表示。成分名（`A_x` 等）はユーザーに露出しない。

### 具体成分の与え方（確定運用・2026-06-22 S-f 検証で明文化）

`define_vector_field` は**成分を抽象関数として登録するだけ**で、引数に成分式は取らない
（許可フィールドは `name` / `coordinate_system` / `natural_vars` のみ）。
「与えられた $\boldsymbol{E},\boldsymbol{B}$」のような**具体成分は、宣言後に H-1 等式形式で各成分に代入して与える**：

```
E をベクトル場として定義する          # 成分 E_x, E_y, E_z を抽象登録
E_x = 0                               # 各成分に具体値を等式で束縛（H-1 → Maxima へ P2 代入）
E_y = cos(k*x - w*t)
E_z = 0
E の発散を求めて                       # nb_div(E_x,E_y,E_z) が束縛値で偏微分の和を計算 → 0
E の回転を求めて                       # nb_curl_* → [0,0,-k sin(k x - w t)]
```

- **`nb_div(Ax,Ay,Az) := diff(Ax,x)+diff(Ay,y)+diff(Az,z)`** のように、div/curl は**成分式を引数に取る
  Maxima 関数**。成分が H-1 で束縛されていれば、その値で展開される（未束縛なら抽象 `diff` のまま）。
- 座標変数（x,y,z）**以外の自由シンボル**（時間 $t$・波数 $k$・角振動数 $\omega$ 等）も成分式に使える。
  Faraday/Ampère の $\partial/\partial t$ は素の `t で微分`。
- これが §1「現状では成分を個別にスカラー関数として宣言する方法しかない」の**具体的な実現方法**。
  煩雑さを嫌って `components=[...]` 引数を足す案もあり得るが、§下記の理由（名前衝突・depends 一貫性・
  復元時マッピング）で見送り中。ユーザー導線は [user_guide.md](../user_guide.md)「ベクトル解析」§3 に記載。

### 時間微分 $\partial/\partial t$ はベクトル演算子ではない（成分ごとに扱う・確定運用）

`div`/`curl`/`grad` はベクトル演算子で成分テンプレート（`nb_div(Ax,Ay,Az)` 等）を持つが、
**時間微分はベクトル演算子ではない**。`B を t で微分` のように場の名前へ直接 $\partial/\partial t$ を効かせても
ベクトルとしては機能しない（裸の場記号を微分するだけ）。**正攻法は「各成分を時間微分し、結果を新しい
ベクトル場に組み直す」**：

```
B_z を t で微分して           # 成分の時間微分 → ∂B_z/∂t
dBdt をベクトル場として定義する  # 受け皿のベクトルを宣言
dBdt_z = w*sin(k*x - w*t)/c    # H-1 で成分を束縛して再組立て
dBdt の発散を求めて            # 組み立てたベクトルは後続計算に使える
```

3 成分をまとめての時間微分・ベクトル等式比較は提供しない（便宜的書法を避ける設計判断）。
成分ごとに処理して再組立てできれば十分、という方針（S-f Maxwell 整合シナリオで確認・
[scenario_maxwell_consistency.md](../scenarios/scenario_maxwell_consistency_jp.md)）。

### 「自動的に Ax, Ay, Az を生成する」方式を採用しない理由

- 次元の曖昧さ（3D/2D/4元ベクトル）
- 名前衝突（`Ax` が別のスカラーと重複する危険）
- `depends` の一貫性管理が困難
- セッション復元時のマッピング保持が複雑

---

## 3. 幾何座標系の設計

### 2種類の「座標系変換」の区別

**幾何座標系と参照フレーム変換は性質が異なるため、別々に扱う。**

| 種類 | 内容 | 設計 |
|---|---|---|
| **幾何座標系** | $\nabla$・div・curl 等の展開式が変わる | セッション開始時に選択 |
| **参照フレーム変換** | NMR/ESR の回転座標系等。ユニタリ変換によりハミルトニアン・演算子が変わる | F-4 として別設計 |

### 座標系の優先度ルール（高い順）

1. 演算ごとの明示指定（最優先）
2. `define_vector_field` 時の個別指定
3. セッション開始時の選択（デフォルト）

### 初期実装でサポートする3座標系

物理研究の95%以上をカバーする最小セット。

| 座標系 | 座標変数 | スケール因子 |
|---|---|---|
| デカルト (cartesian) | x, y, z | $h_1=h_2=h_3=1$ |
| 球座標 (spherical) | r, θ, φ | $h_r=1,\ h_\theta=r,\ h_\phi=r\sin\theta$ |
| 円柱座標 (cylindrical) | r, φ, z | $h_r=1,\ h_\phi=r,\ h_z=1$ |

2D 極座標は「円柱座標で z 依存なし」として扱う（独立した座標系は不要）。

### 実装方法：Maxima 関数の事前定義

セッション開始時に選択した座標系の Maxima 関数を事前定義しておく。

**球座標の例：**
```maxima
nb_div(Ar, Atheta, Aphi) :=
    diff(r^2*Ar, r)/r^2 +
    diff(sin(theta)*Atheta, theta)/(r*sin(theta)) +
    diff(Aphi, phi)/(r*sin(theta))$

nb_curl_r(Ar, Atheta, Aphi) :=
    (diff(sin(theta)*Aphi, theta) - diff(Atheta, phi)) / (r*sin(theta))$

nb_grad(f) := [diff(f,r), diff(f,theta)/r, diff(f,phi)/(r*sin(theta))]$

nb_laplacian_scalar(f) :=
    diff(r^2*diff(f,r), r)/r^2 +
    diff(sin(theta)*diff(f,theta), theta)/(r^2*sin(theta)) +
    diff(f,phi,2)/(r^2*sin(theta)^2)$
```

`∇·A` 入力時 → `nb_div(A_r, A_theta, A_phi)` を呼び出す。

---

## 4. 特殊座標系の拡張：`define_coordinate_system`

放物座標・楕円体座標等の特殊座標系も、ユーザーに Maxima 文法を見せることなく対応できる。

### 設計根拠

すべての**直交座標系**はラメ係数（スケール因子）$h_1, h_2, h_3$ で統一的に記述できる：

$$\nabla \cdot \boldsymbol{A} = \frac{1}{h_1 h_2 h_3}\left[\frac{\partial(h_2 h_3 A_1)}{\partial q_1} + \frac{\partial(h_1 h_3 A_2)}{\partial q_2} + \frac{\partial(h_1 h_2 A_3)}{\partial q_3}\right]$$

curl・grad・laplacian も同様に $h_i$ だけで決まる。

### ユーザー入力例

```
放物座標 (ξ, η, φ) を座標系として定義する
  スケール因子: h_ξ = sqrt(ξ²+η²), h_η = sqrt(ξ²+η²), h_φ = ξη
```

### 内部動作

ユーザーが指定した $h_1, h_2, h_3$ を使い、上記の一般式から
`nb_div`, `nb_curl`, `nb_grad`, `nb_laplacian_scalar` を自動生成して Maxima に送信。

**Maxima 文法は一切露出しない。**

---

## 5. 制約・注意点

### 5-1. 座標変数名の衝突（★★★ 最重要）

球座標の `r`, `θ`, `φ` をユーザーが別の意味で使うと衝突する。

```
球座標セッションで「r をボーア半径のスカラーとして定義する」→ 衝突
```

**対策：** `SymbolContext` で座標変数を予約語としてガード。定義しようとしたら警告を出す。

### 5-2. ベクトルのラプラシアン誤計算（★★ 要注意）

非デカルト座標では $\nabla^2\boldsymbol{A} \neq [\nabla^2 A_1, \nabla^2 A_2, \nabla^2 A_3]$。
Christoffel 記号由来の追加項が出る。例（球座標 r 成分）：

$$(\nabla^2\boldsymbol{A})_r = \nabla^2 A_r - \frac{A_r}{r^2} - \frac{2}{r^2\sin\theta}\frac{\partial(\sin\theta\, A_\theta)}{\partial\theta} - \frac{2}{r^2\sin\theta}\frac{\partial A_\phi}{\partial\phi}$$

**対策：** スカラーラプラシアン（`scalar_laplacian`）とベクトルラプラシアン（`vector_laplacian`）を
別操作として分離。`∇²A`（ベクトル引数）は当面「未サポート」と明示。

### 5-3. 中間結果がベクトル場として非認識

`∇·(A×B)` の計算では、`A×B` の結果が SymbolContext にベクトル場として登録されない。

**対策：** 当面は `define_vector_field` で直接定義したフィールドのみ `∇` 演算をサポート。
中間結果への適用はスコープ外とし、ユーザーに `A×B を E5 として登録してから ∇·E5` と入力するよう案内。

### 5-4. 異なる座標系のベクトル場の混在

同一式内で異なる幾何座標系のベクトル場を演算することは禁止。
（座標変換行列の自動適用は F-3 のスコープ外）

### 5-5. セッション復元時の整合性

旧セッション（座標系情報なし）の読み込み時は、デカルト座標にフォールバックしてログに警告。

---

## 6. 積分の変数変換（F-5）

ベクトル場設計と密接に関連する。

### ケース1：座標系由来の体積要素（F-3 と同時実装）

座標系バンドルに体積要素 `dV_factor` を含める：

| 座標系 | 体積要素 |
|---|---|
| デカルト | $dV = dx\,dy\,dz$ |
| 球座標 | $dV = r^2\sin\theta\,dr\,d\theta\,d\phi$ |
| 円柱座標 | $dV = r\,dr\,d\phi\,dz$ |

`∫f dV` と入力すると体積要素を自動挿入。

### ケース2：一般的な変数変換（独立実装）✅ 実装済み（2026-06-14、Maxima・Julia 両対応）

任意の変数変換 $(x,y) \to (u,v)$ に対するヤコビアン $|∂(x,y)/∂(u,v)|$ を自動計算。

```
change_integration_variables(expr, old_vars=[x,y], new_vars=[u,v], transform=[...])
```

- Maxima: `subst(...) * determinant(jacobian(transform, new_vars))`
- Julia: `substitute(..., Dict(old => transform)) * det(Symbolics.jacobian(transform, new_vars))`

詳細は `docs/open_items.md` F-5 を参照。

---

## 7. 参照フレーム変換（F-4）—— スコープ外

NMR/ESR の回転座標系等は**幾何座標系とは異なる問題**として別設計。

- 時間依存ユニタリ変換：$\hat{H}' = U\hat{H}U^\dagger - i\hbar U\dot{U}^\dagger$
- 演算子・状態・ハミルトニアン自体が変わる
- `transform_to_rotating_frame` 操作として将来設計（NMR/ESR ドメイン追加時）

---

## 8. 実装タスク一覧

| タスク | 規模 | 場所 | F-3/F-5 |
|---|---|---|---|
| `define_vector_field` を schema に追加 | 小 | `common_operations.schema.json` | F-3 |
| `Quantity.kind = "vector_field"` + `components`, `coordinate_system` フィールド | 小 | `models/quantity.py` | F-3 |
| Maxima `depends` 発行 + `nb_div` 等の事前定義 | 小〜中 | `maxima_backend_handler.py` | F-3 |
| `SymbolContext` への成分登録と座標変数の予約語ガード | 小 | `engine.py` | F-3 |
| `\boldsymbol{A}` 表示（clean_latex / display bridge） | 小 | `maxima_backend_handler.py` | F-3 |
| `div`/`curl`/`grad` テンプレート — 成分展開の自動化 | 中 | `backends/maxima.full.json` + `SemanticResolver` | F-3 |
| `scalar_laplacian` と `vector_laplacian` を別操作に分離 | 小 | `common_operations.schema.json` | F-3 |
| セッション開始時の座標系選択 UI | 小 | `notebook_main.js`, `session_io.py` | F-3 |
| `define_coordinate_system`（ラメ係数 → `nb_div` 等自動生成）| 中 | `maxima_backend_handler.py` | F-3 |
| セッション保存時の `A → 成分` マッピング保持 | 小 | `session_io.py` | F-3 |
| 旧セッション復元時のデカルトフォールバック | 小 | `session_io.py` | F-3 |
| 体積要素の自動挿入（座標系バンドルに追加）| 小 | `maxima_backend_handler.py` | F-5 |
| `change_integration_variables` 操作 ✅ | 中 | `common_operations.schema.json` + `maxima_backend_handler.py` + `julia_build_mixin.py`（Maxima・Julia 両対応） | F-5 |
