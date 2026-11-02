# 実践ループ サンプルシナリオ：M/M/1 待ち行列（利用率・平均系内人数・Little の法則）

実践ループ・シナリオメニュー **S-r（経営工学・OR）** の詳細化。到着率 $\lambda$・サービス率 $\mu$ の
M/M/1 待ち行列について、**利用率** $\rho=\lambda/\mu$ と **平均系内人数** $L=\rho/(1-\rho)$ を入力し、
$\rho$ を代入して $L=\lambda/(\mu-\lambda)$ を得、**両辺を $\lambda$ で割る**ことで Little の法則
$W=L/\lambda=1/(\mu-\lambda)$（平均滞在時間）を導く。

**バックエンド:** standard（Maxima）／**ドメイン:** 経営工学・OR（operations_research）。

> 数式モデルとしての M/M/1 は**代入と簡約だけ**で閉じ、CAS 支援ノートの最小サンプルとして見せやすい。
> 操作はすべてルールベース（LLM 不要）で解釈される。

## モデルの背景

到着がポアソン過程（率 $\lambda$）、サービス時間が指数分布（率 $\mu$）の単一窓口待ち行列（M/M/1）。
定常状態（$\rho=\lambda/\mu<1$）で、

- 利用率 $\rho=\dfrac{\lambda}{\mu}$
- 平均系内人数 $L=\dfrac{\rho}{1-\rho}=\dfrac{\lambda}{\mu-\lambda}$
- 平均滞在時間 $W=\dfrac{L}{\lambda}=\dfrac{1}{\mu-\lambda}$（**Little の法則** $L=\lambda W$）

本シナリオは $L$ の定義に $\rho=\lambda/\mu$ を代入し、Little の法則を**両辺を $\lambda$ で割る**操作で
適用して $W$ を得る。

## セッション設定・記号

- モード **standard（Maxima）**、ドメイン **経営工学・OR**。
- $\lambda,\mu,\rho$ は TeX 記法（`\lambda` 等）で入力する。

## 入力ステップ

### ステップ 1：利用率の定義（H-1）

```
\rho = \lambda/\mu
```

**期待する出力:** E1 として登録。$\rho=\lambda/\mu$。

### ステップ 2：平均系内人数の定義（H-1）

```
L = \rho/(1-\rho)
```

**期待する出力:** E2 として登録。$L=\rho/(1-\rho)$。

### ステップ 3：$\rho$ に $\lambda/\mu$ を代入（記号→値の代入）

```
E2 の \rho に \lambda/\mu を代入して
```

**期待する出力:** E3 $= L=-\dfrac{\lambda}{\lambda-\mu}\ \left(=\dfrac{\lambda}{\mu-\lambda}\right)$。

### ステップ 4：両辺を $\lambda$ で割る（Little の法則・H-8）

```
E3 の両辺を \lambda で割って
```

**期待する出力:** E4 $= \dfrac{L}{\lambda}=-\dfrac{1}{\lambda-\mu}\ \left(=\dfrac{1}{\mu-\lambda}=W\right)$。

> 符号は Maxima の正準形（$-1/(\lambda-\mu)$）で表示される。$\mu>\lambda$（定常条件）では正で、
> $W=1/(\mu-\lambda)$ に一致する。

## 確認ポイント

- $L=\rho/(1-\rho)$ に $\rho=\lambda/\mu$ を代入すると $\lambda/(\mu-\lambda)$ になる（ステップ 3）。
- Little の法則 $L=\lambda W$ は、$L$ の両辺を $\lambda$ で割れば $W=L/\lambda=1/(\mu-\lambda)$ として
  得られる（ステップ 4）。記号→値の代入（H-1 系）と両辺操作（H-8）が**ギリシャ文字オペランドでも**
  正しく動くことを確認する。

## 実例ファイル

実行後のセッションとエクスポート文書：
[../../domains/operations_research/examples/scenario_mm1_queue_jp.json](../../domains/operations_research/examples/scenario_mm1_queue_jp.json)
／`.tex`（英語版は `_en.{json,tex}`）。

**再生成:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`

## generator コマンド（この `scenario` ブロックが生成の単一ソース）

```scenario domain=operations_research
@note: 到着率 $\lambda$・サービス率 $\mu$ の M/M/1 待ち行列を扱う。利用率 $\rho=\lambda/\mu$（窓口がふさがっている割合）を定義する。
$\rho = \lambda/\mu$
@note: 平均系内人数 $L=\rho/(1-\rho)$ を定義する。
$L = \rho/(1-\rho)$
@note: $\rho$ を $\lambda/\mu$ に置き換え、$L$ を $\lambda,\mu$ だけで表す式にする。
E2 の \rho に \lambda/\mu を代入して
@note: Little の法則 $L=\lambda W$ を適用するため、両辺を到着率 $\lambda$ で割り、平均滞在時間 $W=L/\lambda$ を得る。
E3 の両辺を \lambda で割って
```
