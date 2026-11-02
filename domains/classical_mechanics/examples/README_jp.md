# classical_mechanics — 実例（中心力の保存量・ポアソン括弧）

このフォルダは、古典力学（Newton 力学）シナリオを実際に実行して得た**教育用の実例**です。
standard（**Maxima**）セッションで再現できます。

| シナリオ | ファイル | 内容 |
|---|---|---|
| [scenario_central_force_conservation.md](../../../docs/scenarios/scenario_central_force_conservation_jp.md) | `scenario_central_force_conservation_jp.{json,tex}` | 中心力ポテンシャル中の運動の保存量：角運動量 $L_z=mr^2\dot\theta$ とエネルギー $E=\tfrac12 m(\dot r^2+r^2\dot\theta^2)-k/r$ を、$d/dt$ → 運動方程式の代入 → 0 で確認 |
| [scenario_poisson_bracket.md](../../../docs/scenarios/scenario_poisson_bracket_jp.md) | `scenario_poisson_bracket_jp.{json,tex}` | ポアソン括弧：正準対 $(r,p_r),(\theta,p_\theta)$ を宣言し、$\{r,p_r\}=1$・中心力ハミルトニアンの $\{H,L_z\}=0$（角運動量保存）・$\{H,r\}=-p_r/m$（Hamilton 運動方程式）を確認 |

各ファイル：`.json` は実行後のセッション（`sessions/` に置けばロード可能）、`.tex` は export 文書
（uplatex/dvipdfmx でコンパイル可）。

保存量 → $\frac{d}{dt}$ → 運動方程式（$\ddot\theta,\ddot r$）の代入 → $0$ のワークフローを辿る。
速度は `\dot{q}`（数式記法）で入力する。古典力学ドメインでは速度は従属量 $\dot q=dq/dt$ として
扱われ、Maxima の depends 形 `'diff(q,t,N)` で表現される（解析力学の独立アトム・モデル＝
[doc 19](../../../docs/developer_manuals/19_analytical_mechanics_independent_variables.md) とは対照的）。

**再生成:** `PATH="$HOME/.juliaup/bin:$PATH" .venv/bin/python scripts/generate_scenario_examples.py`
