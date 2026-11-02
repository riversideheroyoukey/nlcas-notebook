# 実践ループ サンプルシナリオ：クーロンゲージの確認（電磁気学）

> （旧「E-3 研究ワークフロー検証シナリオ」を **1シナリオ＝1セッション**に分割した独立シナリオ。
> 2026-07-01: CAS 構文（`diff(phi,x)` 等）を露出していた旧版を、一般操作（スカラー場・ベクトル場宣言・
> 偏微分・`\nabla` 演算子・方程式の右辺操作・求解）のみで再構成した。）

**バックエンド:** Maxima／**ドメイン:** 電磁気（electromagnetism）。

### 物理的背景

電磁気学のゲージ変換において、クーロンゲージは $\nabla \cdot \mathbf{A} = 0$ を課すゲージ条件。
このゲージではスカラーポテンシャルがPoisson方程式を満たし、静電場が即時（non-retarded）に伝わる。

$$\nabla \cdot \mathbf{A} = 0 \quad \text{（ゲージ条件）}$$

$$\nabla^2 \phi = -\frac{\rho}{\varepsilon_0} \quad \text{（Poisson 方程式）}$$

$$\mathbf{E} = -\nabla\phi - \frac{\partial \mathbf{A}}{\partial t}, \quad \mathbf{B} = \nabla \times \mathbf{A}$$

検証すべき物理：
1. $\nabla \cdot \mathbf{B} = 0$ （B の発散がゼロ：磁荷なし）
2. クーロンゲージ条件のもとで $\nabla \cdot \mathbf{E} = -\nabla^2\phi$（Poisson 方程式）を確認

---

### 一般操作での組み立て方（★このシナリオの肝）

CAS 構文 `diff(phi,x)` を直接入力するのではなく、以下の一般操作の連続で $\mathbf E,\mathbf B$ を
組み立てる：

1. **「時間依存のスカラー場として定義する」で $x,y,z,t$ 依存を一度に宣言する。**
   スカラー場宣言（$x,y,z$）と時間依存宣言（$t$）を別々に行うと、Maxima の `depends()` は
   同じ記号への再宣言を**上書き**する（累積されない）ため、後者が前者の空間依存を消してしまう。
   この結合表現は内部で両方の変数を単一の `depends()` 呼び出しにまとめて送信する
   （詳細は「確認された挙動・注意点」参照）。
2. **ベクトル場は `\nabla \cdot` / `\nabla \times` をそのまま式として入力すれば計算される。**
   $\mathbf A$ を `A をベクトル場として定義する` で宣言し、成分 `A_x`,`A_y`,`A_z` に時間依存を
   追加するだけで、ゲージ条件 $\nabla\cdot\mathbf A$ も磁場 $\mathbf B=\nabla\times\mathbf A$ も
   **1ステップ**で得られる（[scenario_maxwell_consistency.md](scenario_maxwell_consistency_jp.md)
   と同じ流儀）。
3. $\partial/\partial t$ は**ベクトル演算子ではない**ため、$\mathbf E$ の組み立てだけは成分ごとに
   偏微分し、E-id 参照で合成する（`E_x = -E5 - E8` のように）。
4. ゲージ条件は「仮定する」（`assume`）ではなく、**方程式として登録 → H-9 で対象の偏微分について
   解く**（一般の solve 操作）。Maxima の `assume()` は不等式・符号推論用で、等式の代入規則としては
   働かないため。

---

### ステップ

#### EM-1: セッション開始

| 項目 | 内容 |
|---|---|
| 操作 | 新規セッション、分野：「電磁気学」、バックエンド：Maxima |
| 結果 | |

---

#### EM-2: ポテンシャルの宣言と E-id 化

| 項目 | 内容 |
|---|---|
| 入力 | `phi を時間依存のスカラー場として定義する`（$x,y,z,t$ すべてに依存） |
| 入力 | `A をベクトル場として定義する` |
| 入力 | `A_x`, `A_y`, `A_z` を`時間依存関数として定義する`（成分に $t$ 依存を追加） |
| 入力 | `phi0 = phi` / `Ax0 = A_x` / `Ay0 = A_y` / `Az0 = A_z`（差分可能な E-id として登録） |
| 確認 | 各記号が $x,y,z,t$ すべてに依存すると Maxima 側で認識されること |
| 結果 | |

---

#### EM-3: 電場・磁場の組み立て

**$\mathbf E$ は成分ごとの偏微分＋E-id 合成、$\mathbf B$ は `\nabla \times A` 一発：**

| 項目 | 内容 |
|---|---|
| 入力 | `E1 を x で偏微分して` / `を y で` / `を z で`（$\partial\phi/\partial x,y,z$） |
| 入力 | `E2 を t で偏微分して` / `E3 を t で` / `E4 を t で`（$\partial A_i/\partial t$） |
| 入力 | `E_x = -E5 - E8` / `E_y = -E6 - E9` / `E_z = -E7 - E10` |
| 入力 | `\nabla \times A`（$\mathbf B$ の3成分を一度に取得） |
| 入力 | `B をベクトル場として定義する` |
| 入力 | `B_x = E14[1]` / `B_y = E14[2]` / `B_z = E14[3]`（curl の結果を成分ごとに束縛） |
| 確認 | KaTeX で $E_x=-\partial\phi/\partial x-\dot A_x$、$B_x=\partial A_z/\partial y-\partial A_y/\partial z$ 等が正しく表示されること |
| 結果 | |

---

#### EM-4: $\nabla \cdot \mathbf{B} = 0$ の確認

| 項目 | 内容 |
|---|---|
| 入力 | `\nabla \cdot B` |
| 期待結果 | 0（`nb_div` が3成分をまとめて微分・合成し、混合偏微分が自動的に相殺される） |
| 物理的意味 | $\mathbf{B} = \nabla \times \mathbf{A}$ から自動的に保証される恒等式 |
| 結果 | |

---

#### EM-5: クーロンゲージ条件と Poisson 方程式

**ゲージ条件は `\nabla \cdot A` 一発、方程式として登録し H-9 で解く：**

| 項目 | 内容 |
|---|---|
| 入力 | `\nabla \cdot A`（$\nabla\cdot\mathbf A$ を1ステップで取得） |
| 入力 | `gauge = E19`（名前をつけて E-id 化） |
| 入力 | `E20 の右辺を 0 とする`（ゲージ条件 $\nabla\cdot\mathbf A=0$ を方程式として確定） |
| 入力 | `Az0 を z で偏微分して`（後で解くために $\partial A_z/\partial z$ を個別に用意） |
| 入力 | `Az_z = E22`（名前をつける） |
| 入力 | `E21 を Az_z について解いて` → $\partial A_z/\partial z = -\partial A_x/\partial x-\partial A_y/\partial y$ |
| 結果 | |

**$\nabla\cdot\mathbf E$ を組み立てて Poisson 形へ：**

| 項目 | 内容 |
|---|---|
| 入力 | `E_x を x で偏微分して` / `E_y を y で` / `E_z を z で` |
| 入力 | `divE = E25 + E26 + E27` |
| 入力 | `E28 の右辺を整理して` |
| 期待結果 | $\nabla\cdot\mathbf E=-\nabla^2\phi-\dfrac{\partial}{\partial t}\!\left(\dfrac{\partial A_x}{\partial x}+\dfrac{\partial A_y}{\partial y}+\dfrac{\partial A_z}{\partial z}\right)$（$-\nabla^2\phi$ に加え、ゲージ項の時間微分が残る） |
| 入力 | `gauge を t で偏微分して`（$\partial(\nabla\cdot\mathbf A)/\partial t$） |
| 入力 | `E30 の右辺を 0 とする`（ゲージ条件は恒等的に0なので、その時間微分も0） |
| 入力 | `E29 に E31 を代入して`（ゲージ条件の時間微分＝0 を $\nabla\cdot\mathbf E$ の展開式へ代入） |
| 期待結果 | $\nabla\cdot\mathbf E=-\nabla^2\phi$（Poisson 方程式） |
| 結果 | |

> **（履歴）BP 相当の制限 H-10 は 2026-07-27 に解消:** 上記の最後の「代入」は
> 以前は `substitute` がエラー（`IndexedSymbol index must resolve to a Symbol
> or IndexSymbol, got Pow`）になり、CAS では実行せず目視確認による結論として
> 記述していた。原因は3段で、(1) Maxima 自身の微分表示形
> $\frac{d^2}{d t\,d z}A_z$ に canonical への逆変換が無く、(2) 代入規則が
> その未変換 LaTeX をそのまま CAS に渡していて（`\frac{…}` がリゾルバに入って
> 落ちる）、(3) 仮に canonical に直っても Maxima の `subst` は構文的で、和の
> 形の LHS には一致せず**無変化**だった（`ratsubst` が正しい道具）。3点とも
> 修正済みで、この代入は CAS 上で実行され $-\nabla^2\phi$ を返す。
> `tests/test_derivative_substitution.py`。

---

#### EM-6: LaTeX エクスポート

| 項目 | 内容 |
|---|---|
| 操作 | 「⬇ LaTeX」ボタン |
| 確認 | 偏微分が `\frac{\partial}{\partial x}` 形式で出力されること |
| 確認 | `pdflatex` でコンパイル可能なこと |
| 結果 | |

---

### 検証チェックリスト（シナリオ 3）

- [x] `phi`, `A` の宣言（時間依存スカラー場・ベクトル場）が自然言語で通る
- [x] `\nabla \cdot` / `\nabla \times` による $\mathbf B$・ゲージ条件の組み立てが CAS 構文なしで通る
- [x] $\nabla \cdot \mathbf{B} = 0$ が `\nabla \cdot B` 一発で 0 に簡約される
- [x] クーロンゲージ条件を方程式として登録し、H-9 で対象の偏微分について解ける
- [x] Poisson 方程式の帰結が確認できる（機械的代入が CAS 上で実行される。H-10 解消・2026-07-27）
- [ ] LaTeX 出力がコンパイル可能

---

## 確認された挙動・注意点

- **`depends()` の二段階宣言は上書きされる（Maxima の仕様）。** 「スカラー場として定義して」
  （$x,y,z$）→「時間依存関数として定義する」（$t$）の順で別々に宣言すると、後者の
  `depends([name],[t])` が前者の $x,y,z$ 依存を**削除**してしまう（Maxima の `depends()` は
  同じ変数への再宣言をマージせず上書きする）。これは本シナリオの再構成中に発見した実バグで、
  `notebook/engine.py` の `declare_time_dependent()` を修正済み：既存の `natural_vars` を
  読み取り、時間パラメータと**和集合**にしてから単一の `depends()` 呼び出しで送信するように
  変更した（2026-07-01）。
- **「時間依存のスカラー場/ベクトル場として定義する」という結合表現を新設した。**
  `_infer_type_from_label`（`backend_handler.py`）はラベル中に「スカラー場」等のキーワードが
  あれば時間依存キーワードより優先してフィールド宣言を選び、`_preprocess_args`
  （`engine.py`）が時間パラメータを `natural_vars` に畳み込んでから単一の `depends()` を送る。
  これにより $x,y,z,t$ すべてへの依存を1ステップで宣言できる。
- **ベクトル場の成分（`A_x` 等）を再宣言すると曖昧確認ダイアログが誤発火していた。**
  添字1文字（`xi_k` 等）は「添字か名前の一部か」を確認するダイアログを出す設計だが、
  `define_vector_field` が**既に登録済みの**成分 `A_x` を「時間依存関数として定義する」で
  再宣言する際にも同じダイアログが発火し、非対話環境（generator）では失敗していた。
  既に `SymbolContext` に登録済みのシンボルは曖昧性判定をスキップするよう
  `backend_handler.py` の `extra_pre_phase1` を修正した（2026-07-01）。
- **H-9 の「について解いて」も同じエイリアス衝突ガードが必要だった。** `differentiate` の
  「を X で微分して」には、X が既にエイリアス登録済みシンボル（例: 別の式の LHS 名）のとき
  誤って E-id に置換されないようにするガードが既にあったが、`solve_equation` の
  「を X について解いて」には同じガードがなく、`Az_z について解いて` が
  `E23 について解いて`（E-id への誤爆）になっていた。`ui/interpret_pipeline.py` の
  シンボル→E-id 置換ガード正規表現に `について解い` を追加して修正した（2026-07-01）。
- **微分を含む式への `substitute` は失敗していた（H-10、上記 EM-5 の最終ステップで発見。
  2026-07-27 に修正）。** 3段の欠陥が重なっていた——(1) Maxima 自身の微分表示形
  $\frac{d^2}{d t\,d z}A_z$ に canonical への逆変換が無い、(2) 代入規則がその未変換
  LaTeX をそのまま CAS に渡す、(3) Maxima の `subst` は構文的で和の形の LHS には
  一致せず無変化（`ratsubst` が正しい道具）。EM-5 の代入は現在 CAS 上で実行される。
- **`\nabla \cdot (\nabla \times A)` のようなネストした `\nabla` 式は単一の式として入力できない。**
  `B` を明示的にベクトル場として登録し、curl の結果を成分ごとに束縛してから
  `\nabla \cdot B` を呼ぶ必要がある（本シナリオの EM-3〜EM-4 の手順どおり）。

**確認日:** 2026-07-01（実機 Maxima、generator 経由で全37ステップ完走）。

---

## generator コマンド（この `scenario` ブロックが生成の単一ソース）

```scenario domain=electromagnetism
@note: スカラーポテンシャル $\phi$（空間 $x,y,z$ と時間 $t$ に依存）と、ベクトルポテンシャル $\mathbf A$（成分 $A_x,A_y,A_z$、これも時間に依存）を宣言する。この2つから電場 $\mathbf E$・磁場 $\mathbf B$ を組み立てる。
phi を時間依存のスカラー場として定義する
A をベクトル場として定義する
A_x を時間依存関数として定義する
A_y を時間依存関数として定義する
A_z を時間依存関数として定義する
@note: 後で偏微分の計算に使うため、$\phi,A_x,A_y,A_z$ それぞれの値に別名（phi0, Ax0, Ay0, Az0）をつけて式番号として登録しておく。
phi0 = phi
Ax0 = A_x
Ay0 = A_y
Az0 = A_z
@note: 電場の定義 $\mathbf E=-\nabla\phi-\partial\mathbf A/\partial t$ の1項目 $\nabla\phi$ に使うため、$\phi$ の空間偏微分 $\partial\phi/\partial x,\partial\phi/\partial y,\partial\phi/\partial z$ を求める。
E1 を x で偏微分して
E1 を y で偏微分して
E1 を z で偏微分して
@note: 電場の定義の2項目 $\partial\mathbf A/\partial t$ に使うため、$\mathbf A$ の各成分の時間微分 $\partial A_x/\partial t,\partial A_y/\partial t,\partial A_z/\partial t$ を求める。
E2 を t で偏微分して
E3 を t で偏微分して
E4 を t で偏微分して
@note: 上で求めた偏微分を組み合わせ、電場 $\mathbf E=-\nabla\phi-\partial\mathbf A/\partial t$ の各成分を組み立てる。
E_x = -E5 - E8
E_y = -E6 - E9
E_z = -E7 - E10
@note: 磁場 $\mathbf B=\nabla\times\mathbf A$ を、$\nabla\times$ 演算子を $\mathbf A$ に直接適用して1ステップで求める（3成分がまとめて返る）。
\nabla \times A
B をベクトル場として定義する
@note: 得られた curl の3成分を、ベクトル場 $\mathbf B$ の $x,y,z$ 成分としてそれぞれ束縛する。
B_x = E14[1]
B_y = E14[2]
B_z = E14[3]
@note: $\mathbf B=\nabla\times\mathbf A$ から常に成り立つはずの恒等式 $\nabla\cdot\mathbf B=0$（磁荷が存在しないこと）を、$\mathbf B$ の発散を計算して確認する。
\nabla \cdot B
@note: クーロンゲージ条件 $\nabla\cdot\mathbf A=0$ を確認するため、まず $\mathbf A$ の発散を計算する。
\nabla \cdot A
@note: この発散の値に gauge という名前をつける。
gauge = E19
@note: クーロンゲージでは $\nabla\cdot\mathbf A=0$ という条件を課す。gauge の値を 0 と置いた方程式として確定する。
E20 の右辺を 0 とする
@note: 後でこのゲージ条件を解くために、$A_z$ の空間微分 $\partial A_z/\partial z$ を個別に用意し、名前をつける。
Az0 を z で偏微分して
Az_z = E22
@note: ゲージ条件の式を $\partial A_z/\partial z$ について解き、$\partial A_x/\partial x,\partial A_y/\partial y$ で表す。
E21 を Az_z について解いて
@note: 電場の発散 $\nabla\cdot\mathbf E$ を組み立てるため、$\mathbf E$ の各成分を対応する座標で偏微分する。
E11 を x で偏微分して
E12 を y で偏微分して
E13 を z で偏微分して
@note: 上の3つの偏微分の和として、電場の発散 $\nabla\cdot\mathbf E$ を組み立てる。
divE = E25 + E26 + E27
@note: 組み立てた $\nabla\cdot\mathbf E$ の式を整理する。
E28 の右辺を整理して
@note: ゲージ条件 $\nabla\cdot\mathbf A=0$ の時間微分を計算する（恒等的にゼロのはずであることを確認するため）。
E20 を t で偏微分して
@note: この時間微分の値を 0 と置いた方程式として確定する。
E30 の右辺を 0 とする
@note: ゲージ条件の時間微分＝0 を $\nabla\cdot\mathbf E$ の展開式へ代入すると、対応する3項が消えて $\nabla\cdot\mathbf E=-\nabla^2\phi$（Poisson 方程式）が得られる。
E29 に E31 を代入して
```
