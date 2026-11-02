# analytical_mechanics — 実例（Lagrangian → Euler-Lagrange）

このフォルダは、解析力学シナリオを実際に実行して得た**教育用の実例**です。standard（**Maxima**）
セッションで再現できます。

| シナリオ | ファイル | 内容 |
|---|---|---|
| [scenario_double_pendulum_lagrangian.md](../../../docs/scenarios/scenario_double_pendulum_lagrangian_jp.md) | `scenario_double_pendulum_lagrangian_jp.{json,tex}` | 二重振り子（速度結合あり）の Lagrangian → Euler-Lagrange 方程式 |
| [scenario_coupled_oscillations.md](../../../docs/scenarios/scenario_coupled_oscillations_jp.md) | `scenario_coupled_oscillations_jp.{json,tex}` | 連成振動：運動方程式 → 基準モード（和・差）→ 基準座標変換の逆変換（L-1） |

各ファイル：`.json` は実行後のセッション（`sessions/` に置けばロード可能）、`.tex` は export 文書
（uplatex/dvipdfmx でコンパイル可）。

$\partial L/\partial\dot q$ → $\frac{d}{dt}$ → $\partial L/\partial q$ → 運動方程式 の Euler-Lagrange ワークフローを辿る。
速度は `\dot{q}`（数式記法）で入力する。解析力学ドメインでは座標 $q$ と速度 $\dot q$ が独立アトムとして
扱われる（独立アトム・モデル＝[doc 19](../../../docs/developer_manuals/19_analytical_mechanics_independent_variables.md)）。

**再生成:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`
