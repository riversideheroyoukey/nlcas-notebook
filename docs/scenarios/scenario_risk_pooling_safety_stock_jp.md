# 実践ループ サンプルシナリオ：在庫配置（リスクプーリング）— 安全在庫の平方根則

実践ループ・シナリオメニュー **S-w（SCM・物流）** の詳細化。安全在庫の公式
$SS=z\sigma\sqrt{L}$ に、単独拠点の場合（$L$ のまま）と $n$ 拠点分の需要をプールした
中央倉庫の場合（$L\to nL$、分散の加法性より）をそれぞれ代入し、その比を取ることで
**リスクプーリングの平方根則** $SS_{\rm pool}/SS_{\rm sep}=1/\sqrt{n}$ を導く。

**バックエンド:** standard（Maxima）／**ドメイン:** SCM・物流（supply_chain_logistics）。

## モデルの背景

安全在庫は需要の標準偏差 $\sigma$、サービスレベル係数 $z$、リードタイム $L$ に対して
$SS=z\sigma\sqrt{L}$ で与えられる（基準在庫は $S=\mu L+z\sigma\sqrt{L}$）。

$n$ 拠点に分散していた需要を1箇所（中央倉庫）にプールすると、独立同分布の仮定の下で
分散が加法的に効き、実質的なリードタイム項が $L$ から $nL$ に置き換わる：

$$SS_{\rm pool}=z\sigma\sqrt{nL},\qquad SS_{\rm sep,total}=n\cdot z\sigma\sqrt{L}
\;（n\text{ 拠点それぞれの安全在庫の合計}）.$$

$$\frac{SS_{\rm pool}}{SS_{\rm sep,total}}=\frac{\sqrt{nL}}{n\sqrt{L}}=\frac{1}{\sqrt{n}}
\quad\text{（平方根則）}.$$

$n$ 拠点を集約すると必要な安全在庫の合計が $1/\sqrt{n}$ 倍で済む、という結果。

## セッション設定・記号

- モード **standard（Maxima）**、ドメイン **SCM・物流**。
- $z,\sigma,L,n$ は自由変数。$SS$ は H-1 で名前付き定義する。

## 入力ステップ

### ステップ 1：安全在庫の一般式（H-1）

```
SS = z * sigma * sqrt(L)
```

**期待する出力:** E1 として登録。$SS=z\sigma\sqrt{L}$。

### ステップ 2：$n$ 拠点プール（$L\to nL$、分散の加法性）

```
E1 の L に n * L を代入して
```

**期待する出力:** E2 $=SS_{\rm pool}=\sqrt{Ln}\,\sigma z$（元の $E1$ 自体は
$SS_{\rm sep}$ として引き続き参照できる）。

### ステップ 3：比を取る（H-2 の式参照展開）

```
ratio = E2 / (n * E1)
```

**期待する出力:** E3 $=\text{ratio}=\dfrac{\sqrt{Ln}}{\sqrt{L}\,n}$
（$=1/\sqrt{n}$ と代数的に同値だが、まだこの簡約形にはなっていない — ステップ4参照）。

### ステップ 4：教科書形式に簡約する（radical_simplify）

```
E3 の根号を簡約して
```

**期待する出力:** E4 $=\text{ratio}=\dfrac{1}{\sqrt{n}}$、見慣れた平方根則の形。

## 確認ポイント

- E3 の $\sqrt{Ln}/(\sqrt{L}\,n)$ は $1/\sqrt{n}$ と代数的に一致する（両辺を2乗すると
  $Ln/(L n^2)=1/n$ で一致）が、汎用の `simplify` 操作（Maxima の
  `ratsimp`/`trigsimp`）だけではこの約分は行われない — Maxima は分数をまたいだ
  別々の根号（$\sqrt{Ln}$ と $\sqrt{L}$）を、指示なしに自動でまとめない。専用の
  `radical_simplify` 操作（Maxima の `radcan`）がこの約分を行う。これは本シナリオ
  限定の場当たり的対応ではなく、他の操作と同様に一般公開されている操作である。
- `E1 の L に n * L を代入して`（LHS の記号 `L` が RHS の `n * L` にも文字どおり現れる
  「自己参照」代入）が正しく1回限りの単純な代数的置換として処理される（下記「確認された
  挙動」参照——循環代入ガードの誤検出を根治済み）。
- H-2（式参照展開）により、`ratio = E2 / (n * E1)` のように**複数の式番号を1つの新しい式へ
  組み合わせる**操作がそのまま計算できる。

## 確認された挙動・注意点（発見・根治済み）

- **循環代入ガードが、真に循環しない自己参照代入まで一律ブロックしていた
  （SUBST-LINEAR-SELFREF、発見・根治・2026-07-05）。** `E1 の L に n * L を代入して`
  （$L\to nL$、LHS `L` が RHS `n*L` にも文字どおり現れる）は、修正前は
  **循環代入エラー**（「L の右辺に L が含まれているため代入できません」）になっていた。
  このガードは元々、$\ddot q+\omega^2q=0$ を $q$ について解いた
  $q=-\ddot q/\omega^2$ を代入し直すような**真に危険なケース**（代入操作が複数の
  LHS 表記ゆれ — 素の `q` と微分形 `'diff(q,t,2)` — を連鎖 `subst()` するため、
  RHS 自身に含まれる導関数が互いを再度マッチし、入れ子の `subst()` が発散的に
  導関数を再挿入し続ける）を防ぐために存在する。しかし `L\to nL` のような**微分を
  含まない、単純な代数的自己参照**（スケーリング・改名代入）では、Maxima の
  `subst()` 自体は1パスで、実際には発散しないことを実機の Maxima セッションで
  直接確認した（`subst(L=n*L, z*sigma*sqrt(L))` は正しく `sqrt(L*n)*sigma*z` を
  返す）。ガードを「LHS が RHS の `'diff(...)` 呼び出しの中に現れるか」まで
  絞り込むよう精緻化し（`notebook/runtime/maxima_backend_handler.py`
  `_build_substitute`）、$\ddot q$ の真に循環するケースは従来どおりブロックしつつ、
  本シナリオのような単純な自己参照代入は正しく通るようにした
  （`tests/test_circular_substitution_guard.py`）。

## 実例ファイル

実行後のセッションとエクスポート文書：
[../../domains/supply_chain_logistics/examples/scenario_risk_pooling_safety_stock_jp.json](../../domains/supply_chain_logistics/examples/scenario_risk_pooling_safety_stock_jp.json)
／`.tex`（英語版は `_en.{json,tex}`）。

**再生成:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`

## generator コマンド（この `scenario` ブロックが生成の単一ソース）

```scenario domain=supply_chain_logistics
@note: 安全在庫の一般式 $SS=z\sigma\sqrt{L}$（$z$: サービスレベル係数、$\sigma$: 需要の標準偏差、$L$: リードタイム）を定義する。
SS = z * sigma * sqrt(L)
@note: $n$ 拠点分の需要を1つの中央倉庫にプールした場合を考える。独立同分布の仮定のもとで分散が加法的に効くため、実質的なリードタイム項が $L$ から $nL$ に置き換わる。この置き換えを元の式に代入する（元の式自体は単独拠点の場合として引き続き参照できる）。
E1 の L に n * L を代入して
@note: プールした場合の安全在庫と、$n$ 拠点それぞれの安全在庫の合計（元の式の $n$ 倍）との比を取り、リスクプーリングの効果（平方根則 $1/\sqrt n$ になるはず）を確認する。
ratio = E2 / (n * E1)
@note: $E3=\sqrt{Ln}/(\sqrt{L}\,n)$ は代数的には $1/\sqrt{n}$ だが、Maxima の汎用 `simplify` は2つの別々の根号を自動ではまとめない。専用の根号簡約操作を適用し、見慣れた閉形式に約分する。
E3 の根号を簡約して
```
