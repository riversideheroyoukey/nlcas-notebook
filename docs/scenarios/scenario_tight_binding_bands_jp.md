# 実践ループ サンプルシナリオ：tight-binding バンド構造（分散関係）

**物理的テーマ:** 格子上のホッピング模型 → ブロッホ行列 → 分散関係（鎖・正方・三角・グラフェン・カゴメ・Lieb・SSH）
**バックエンド:** standard（Maxima）で可。分散の構成は CAS を経由せず決定論的に閉形式を組む（Julia 不要）
**所要時間:** 約 15 分
**検証目的:** F-1 3c-fourier が、名前付き格子・抽象結晶記述子から**厳密な閉形式バンド**（および N≥3 は永年方程式）を返すことを確認

---

## 物理的背景

並進不変な tight-binding 模型 $H=\sum_{ij}A_{ij}c^\dagger_i c_j$ のバンドは、単位胞のホッピング構造を
フーリエ変換したブロッホ行列 $\tilde H(k)=\sum_{\Delta R}t\,e^{ik\cdot\Delta R}$ の固有値で与えられる。
1 副格子なら $\varepsilon(k)=\sum_{\Delta R}2t\cos(k\cdot\Delta R)$、2 副格子（二部）なら
$\varepsilon_\pm=\tfrac{d_A+d_B}{2}\pm\sqrt{(\tfrac{d_A-d_B}{2})^2+|f|^2}$。本シナリオはこれを各種格子で辿る。
**交換子を経由せず**、ホッピング構造から直接閉形式を構成する（問題B を回避し自己完結）。

---

## セッション設定

- モード: **standard（Maxima）**（分散コマンドは backend 非依存・CAS 非経由）
- 検証ハーネス: `./scripts/debug_repl.sh --no-llm --lang ja`（ルールベース dispatch で再現可能）

---

## 入力ステップ

### ステップ 1：ホッピング模型の登録（シンボリック保持）

```
H = -t \sum_{\langle i j \rangle} c'_i c_j
```

**期待する出力:** `\sum_{⟨ij⟩}` の形のまま `H` として登録（展開しない）。

### ステップ 2：1次元鎖の分散

```
H の分散関係を求めて
```

**期待する出力:** `\varepsilon(k) = -2t \cos k`

### ステップ 3：正方格子

```
H の正方格子で分散関係を求めて
```

**期待する出力:** `\varepsilon(\mathbf{k}) = -2t \left( \cos k_x + \cos k_y \right)`

### ステップ 4：三角格子の t–t′（次近接シェル）

```
H の三角格子で次近接 -t' を含めて分散関係を求めて
```

**期待する出力:**
`\varepsilon(\mathbf{k}) = -2t \left( \cos k_x + \cos k_y + \cos(k_x + k_y) \right) -2t' \left( \cos(k_x - k_y) + \cos(2 k_x + k_y) + \cos(k_x + 2 k_y) \right)`

**検証ポイント:** 第2近接シェル $\pm(a_1-a_2),\pm(2a_1+a_2),\pm(a_1+2a_2)$ が加わる。

### ステップ 5：ハニカム＝グラフェンの2バンド

```
H のハニカム格子で分散関係を求めて
```

**期待する出力:**
`\varepsilon_\pm(\mathbf{k}) = \pm t \sqrt{ 3 + 2\cos k_1 + 2\cos k_2 + 2\cos(k_1 - k_2) }`

**検証ポイント:** $\Gamma$ 点で √内＝9 → $\varepsilon_\pm=\pm 3t$、Dirac 点でギャップレス。

### ステップ 6：質量項でギャップを開ける（ギャップ付き Dirac）

```
H のハニカム格子で質量 \Delta を含めて分散関係を求めて
```

**期待する出力:**
`\varepsilon_\pm(\mathbf{k}) = \pm \sqrt{ \Delta^2 + t^2 \left( 3 + 2\cos k_1 + 2\cos k_2 + 2\cos(k_1 - k_2) \right) }`

**検証ポイント:** スタガード副格子ポテンシャル $\Delta$ で $2\Delta$ のギャップ（h-BN 型）。

### ステップ 7：カゴメ格子（平坦バンド＋Dirac）

```
H のカゴメ格子で分散関係を求めて
```

**期待する出力:**
`\varepsilon_{\mathrm{flat}} = 2t, \quad \varepsilon_\pm(\mathbf{k}) = -t \left( 1 \pm \sqrt{ 3 + 2\cos k_x + 2\cos k_y + 2\cos(k_x + k_y) } \right)`

**検証ポイント:** 非二部 3 副格子だが、閉三角形恒等式で立方が因数分解 → 平坦バンド $2t$＋分散2本。$\Gamma$ 点で平坦バンドと上側バンドが接触。

### ステップ 8：Lieb 格子（star 構造の平坦バンド）

```
H の Lieb 格子で分散関係を求めて
```

**期待する出力:** `\varepsilon_0 = 0, \quad \varepsilon_\pm(\mathbf{k}) = \pm t \sqrt{ 4 + 2\cos k_x + 2\cos k_y }`

### ステップ 9：SSH 模型（重み付きボンド・自己完結）

```
```

**期待する出力:** `\varepsilon_\pm(k) = \pm \sqrt{ v^2 + w^2 + 2 v w \cos k }`

**検証ポイント:** セル内 $v$・セル間 $w$。`SSH模型 v=t_1, w=t_2 の…` で振幅記号も上書き可。

### ステップ 10：抽象結晶記述子（任意ボンド入力）

```
次のボンドで結晶の分散関係を求めて: (0,1,(0),v),(0,1,(-1),w)
```

**期待する出力:** Bloch 行列とバンドの両方
- `\tilde H(k) = \begin{pmatrix} 0 & v + w e^{-i k} \\ v + w e^{i k} & 0 \end{pmatrix}`
- `\varepsilon_\pm(k) = \pm \sqrt{ v^2 + w^2 + 2 v w \cos k }`（＝SSH の特殊例）

**検証ポイント:** 任意ボンド `(α,β,ΔR,振幅)` から統一的に構成。N≤2 は閉形式。

### ステップ 11：N≥3 任意結晶（永年方程式）

```
次のボンドで結晶の分散関係を求めて: (0,1,(0),a),(1,2,(0),b),(0,2,(0),c)
```

**期待する出力:** 3×3 ブロッホ行列とともに
`\det\!\left(\tilde H(k) - \varepsilon I\right) = 0`

**検証ポイント:** 一般 N×N は閉形式不可（Abel–Ruffini）なので、バンドが満たす**永年方程式**を解析的に表示する。

---

## カバレッジ（このシナリオで検証される機能）

| 機能 | 入力例 | 実装 |
|---|---|---|
| シンボリックボンド和の保持 | `H = -t \sum_{⟨ij⟩} …` | `register_bond_sum` |
| 単一バンド分散 / t–t′ | 鎖・正方・三角＋次近接 | `build_dispersion_single_band` |
| 二部 2 バンド（グラフェン） | ハニカム | `build_dispersion_multiband` |
| 質量ギャップ / 次近接対角 | 質量 Δ／次近接 t' | （同上の onsite/nnn） |
| star 平坦バンド（Lieb） | Lieb 格子 | `build_dispersion_nband` (star) |
| ring 平坦＋Dirac（カゴメ） | カゴメ格子 | `build_dispersion_nband` (ring) |
| 重み付きボンド（SSH） | SSH 模型 | `build_dispersion_ssh` |
| 抽象結晶記述子 / 永年方程式 | 次のボンドで結晶の… | `build_crystal_bloch` |

**関連設計文書:** [05_algebraic_index_architecture.md](../developer_manuals/05_algebraic_index_architecture.md)（k 空間）。分散構成は `notebook/runtime/backend_handler.py`。

## generator コマンド（この `scenario` ブロックが生成の単一ソース）

```scenario domain=quantum_mechanics
@note: 格子上のホッピング模型 $H=-t\sum_{\langle ij\rangle}c^\dagger_ic_j$（隣接サイト間のホッピング振幅 $t$）を、和を展開せずシンボリックなまま登録する。
H = -t \sum_{\langle i j \rangle} c'_i c_j
@note: この模型を1次元鎖として扱った場合の分散関係 $\varepsilon(k)$（ブロッホ行列の固有値。単一副格子の場合はフーリエ変換した和そのもの）を求める。
H の分散関係を求めて
@note: 同じホッピング模型を正方格子上で扱った場合の分散関係を求める。
H の正方格子で分散関係を求めて
@note: 三角格子上で、最近接に加えて次近接ホッピング $-t'$ も含めた分散関係を求める（第2近接シェルの寄与が加わる）。
H の三角格子で次近接 -t' を含めて分散関係を求めて
@note: ハニカム格子（グラフェンの構造）上で分散関係を求める。2つの副格子があるため2本のバンド $\varepsilon_\pm(k)$ が得られ、Dirac点でギャップレスになるはずである。
H のハニカム格子で分散関係を求めて
@note: ハニカム格子に、副格子ごとに符号の異なるオンサイトエネルギー（質量項 $\Delta$）を加えて分散関係を求める。これによりDirac点にギャップ $2\Delta$ が開くはずである（h-BN型）。
H のハニカム格子で質量 \Delta を含めて分散関係を求めて
@note: カゴメ格子（3副格子）上で分散関係を求める。非二部格子だが、平坦バンドと2本の分散バンドが得られるはずである。
H のカゴメ格子で分散関係を求めて
@note: Lieb格子上で分散関係を求める（star構造による平坦バンドが期待される）。
H の Lieb 格子で分散関係を求めて
@note: 特定の格子名を使わず、抽象的なボンドのリスト（副格子ペア・格子ベクトルの変位・振幅）を直接指定して結晶の分散関係を求める。このボンドの組が SSH（Su–Schrieffer–Heeger）二量化鎖そのもので、Bloch 行列と閉じた形のバンドの両方が出る。以前はここに「SSH 模型の分散関係を求めて」という専用の段があったが、**この一般経路が文字列まで同一の結果を出す**ことを実測したうえで削除した（SCENARIO-MACRO 第6号）——専用経路は Bloch 行列を出さないぶん、むしろ情報が少なかった。
次のボンドで結晶の分散関係を求めて: (0,1,(0),v),(0,1,(-1),w)
```
