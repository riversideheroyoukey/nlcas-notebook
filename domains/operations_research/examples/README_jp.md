# operations_research — 実例（M/M/1 待ち行列）

このフォルダは、オペレーションズ・リサーチ（OR）シナリオを実際に実行して得た**教育用の実例**です。
standard（**Maxima**）セッション・`operations_research` ドメインで再現できます。

| シナリオ | ファイル | 内容 |
|---|---|---|
| [scenario_mm1_queue.md](../../../docs/scenarios/scenario_mm1_queue_jp.md) | `scenario_mm1_queue_{jp,en}.{json,tex}` | M/M/1 待ち行列：利用率 $\rho=\lambda/\mu$・平均系内人数 $L=\rho/(1-\rho)$ に $\rho$ を代入して $L=\lambda/(\mu-\lambda)$、**両辺を $\lambda$ で割って** Little の法則 $W=L/\lambda=1/(\mu-\lambda)$ を導く |

各ファイル：`.json` は実行後のセッション（`sessions/` に置けばロード可能）、`.tex` は export 文書
（uplatex/dvipdfmx でコンパイル可）。**日本語版＝`_jp`／英語版＝`_en`** の両方を収録（入力 NL の i18n
対応により、英語版も同じパイプラインで実行して生成）。

**入力のポイント:**
- 利用率・平均系内人数は等式 `\rho = \lambda/\mu`・`L = \rho/(1-\rho)` として登録（H-1）。
- `E2 の \rho に \lambda/\mu を代入して`（記号→値の代入）で $L=\lambda/(\mu-\lambda)$。
- `E3 の両辺を \lambda で割って`（H-8）で Little の法則 $W=1/(\mu-\lambda)$。**代入・簡約だけ**で閉じる。

**再生成:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`
