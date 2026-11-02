# mathematical_biology — 実例（Lotka–Volterra の安定性解析）

このフォルダは、数理生物学シナリオを実際に実行して得た**教育用の実例**です。
standard（**Maxima**）セッション・`mathematical_biology` ドメインで再現できます。

| シナリオ | ファイル | 内容 |
|---|---|---|
| [scenario_lotka_volterra_stability.md](../../../docs/scenarios/scenario_lotka_volterra_stability_jp.md) | `scenario_lotka_volterra_stability_jp.{json,tex}` | 被食者·捕食者モデル $\dot x=ax-bxy,\ \dot y=dxy-cy$ の固定点（$(0,0)$ と共存平衡 $(c/d,a/b)$）→ ヤコビアン → 固有値 $\pm i\sqrt{ac}$（中立中心）。`solve_system`・偏微分・行列固有値の既存操作のみ |

各ファイル：`.json` は実行後のセッション（`sessions/` に置けばロード可能）、`.tex` は export 文書
（uplatex/dvipdfmx でコンパイル可）。

**入力のポイント:**
- ヌルクラインは等式 `a*x - b*x*y = 0` の形で登録し、`E1 と E2 の連立方程式を x と y について解いて`
  で固定点を求める（**複数の固定点**が解集合ごとに正しく登録される）。
- ヤコビアン成分は `a*x - b*x*y を x で偏微分して` のように偏微分で得る。
- 行列は `[0, -b*c/d; a*d/b, 0]`（`;` 行区切り・`,` 列区切り）で入力し、`固有値を求めて` で固有値。
