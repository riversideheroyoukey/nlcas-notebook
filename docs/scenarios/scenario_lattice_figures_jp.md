# 実践ループ サンプルシナリオ：格子図の描画と TeX エクスポート（S-k）

実践ループ・シナリオメニュー **S-k（可視化）** の詳細化。格子和（F-1）を具体化したとき、その
**ボンド構造を図**にして「意図した格子になっているか」を確認し、論文へそのまま埋め込む。

**バックエンド:** standard（**Maxima**）／**ドメイン:** statistical_mechanics。
ただし**格子図生成は CAS 非経由**（純 Python の TikZ/SVG/PDF — [lattice_figure.py](../../notebook/runtime/lattice_figure.py)）なので、
図そのものはバックエンドに依らない（Julia セッションでも同一の図が出る）。Maxima を指定しているのは、
QA 演算子が不要なシナリオを Maxima 系ドメインに置くメニュー規約に従っているだけ。

## 背景

`expand_neighbor_sum`（F-1）は格子和 $\sum_{\langle ij\rangle}$ を具体化するとき、隣接リスト（ボンド生成器）から
サイト座標とボンドを得ている。同じ生成器を使ってボンド構造を**図に起こす**のがこのシナリオ：

- `lattice_to_tikz` — 論文用 `tikzpicture`（TeX エクスポートに自動挿入）。
- `lattice_to_svg` — ブラウザ／画像ビューア用のスタンドアロン SVG。
- 周期境界の**巻き込みボンド**（単位間隔の 1.5 倍より長い辺）は**破線**で描かれ、境界条件が一目で分かる。

## 入力ステップ

### ステップ 1：ハニカム格子 3×3 の最近接和

```
Sz_i Sz_j のハニカム格子 3x3 の最近接和を展開して
→ E1 = Sz(3)Sz(4) + Sz(1)Sz(3) + …（9 サイト・9 ボンド）
```

`expand_neighbor_sum`（pair_constraint=`honeycomb_2d`, lattice_shape=`3x3`, boundary=`open`）。

### ステップ 2：格子図を保存（スタンドアロン）

```
格子図を保存して
→ <session>_lattice.svg ＋ <session>_lattice.tex（standalone）
  ＋ <session>_lattice.pdf（latexmk/pdflatex/lualatex があれば）
```

直近の格子和（ステップ 1 のハニカム）の図を `sessions/` に書き出す。PDF は LaTeX エンジンが
PATH にあるときのみ。SVG はその場で確認でき、standalone TeX は単独でコンパイルできる。

### ステップ 3：正方格子 3×3 の周期境界（破線ラップ）

```
Sz_i Sz_j の正方格子 3x3 の周期境界で最近接和を展開して
→ E2 = …（9 サイト・18 ボンド。うち 6 本が巻き込み＝破線）
```

`boundary=periodic`。周期境界で生じる端↔端のボンドが図では**破線**になる。

### ステップ 4：TeX エクスポート（図の自動埋め込み）

セッションを TeX に書き出すと、各 `expand_neighbor_sum` ステップの**直後にその格子図**
（`tikzpicture`）が挿入される。本シナリオでは E1（ハニカム）と E2（正方・周期）の 2 図が
それぞれの展開ハミルトニアンの隣に並ぶ。プリアンブルは `\usepackage{tikz}`（dvipdfmx）を含み、
`uplatex && dvipdfmx` でコンパイル可。

## カバレッジ

- **格子図生成**（`lattice_layout`/`lattice_to_tikz`/`lattice_to_svg`）— ハニカム・正方。
- **境界条件の可視化** — 周期境界の巻き込みボンドを破線で表示。
- **スタンドアロン保存**（SVG / standalone TeX / PDF）と **TeX エクスポートへの自動埋め込み**。
- **CAS 非経由＝バックエンド非依存**（Maxima/Julia いずれでも同一の図）。

## 関連

- 実例：`domains/statistical_mechanics/examples/scenario_lattice_figures.{json,tex}`
  ＋ `scenario_lattice_figures_lattice.svg`（ハニカムのスタンドアロン図）。
- 格子和の物理利用：[scenario_ising_partition_function.md](scenario_ising_partition_function_jp.md)（S-g）、
  [scenario_tight_binding_bands.md](scenario_tight_binding_bands_jp.md)（分散関係）。
- 実装と残課題：[open_items.md](../../KNOWN_ISSUES.md) §F-1（格子図版）。

## generator コマンド（この `scenario` ブロックが生成の単一ソース）

```scenario domain=statistical_mechanics
@note: ハニカム格子（3×3、開放境界）上で、隣接サイト対 $\langle ij\rangle$ にわたる和 $\sum_{\langle ij\rangle}S_i^zS_j^z$ を、実際のボンド（隣接ペア）に展開する。
Sz_i Sz_j のハニカム格子 3x3 の最近接和を展開して
@note: 直前の格子和のボンド構造を図として保存する（SVGとスタンドアロンTeX、可能ならPDFも）。図を見ることで、意図した格子・境界条件になっているかを確認できる。
格子図を保存して
@note: 今度は正方格子（3×3）で、周期境界条件のもとで同じ最近接和を展開する。周期境界により生じる端と端をつなぐボンドは、図では破線として区別される。
Sz_i Sz_j の正方格子 3x3 の周期境界で最近接和を展開して
```
