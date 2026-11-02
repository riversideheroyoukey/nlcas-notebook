# electrical_engineering — 実例（交流インピーダンスと共振）

このフォルダは、電気工学シナリオを実際に実行して得た**教育用の実例**です。
standard（**Maxima**）セッション・`electrical_engineering` ドメインで再現できます。

| シナリオ | ファイル | 内容 |
|---|---|---|
| [scenario_ac_impedance.md](../../../docs/scenarios/scenario_ac_impedance_jp.md) | `scenario_ac_impedance_jp.{json,tex}` | 直列 RLC の複素インピーダンス $Z=R+j\omega L+\frac{1}{j\omega C}$ の実部 $R$・虚部 $\omega L-\frac{1}{\omega C}$ を分離し、$\operatorname{Im}Z=0$ を解いて共振 $\omega_0=\frac{1}{\sqrt{LC}}$ を導く |

各ファイル：`.json` は実行後のセッション（`sessions/` に置けばロード可能）、`.tex` は export 文書
（uplatex/dvipdfmx でコンパイル可）。

`electrical_engineering` ドメインでは **`j` が虚数単位**（C-4、Maxima では `%i`）。`実部を求めて` /
`虚部を求めて` は等式 `Z = …` の**右辺（値）**に作用し（`Re Z = R`・`Im Z = ωL−1/(ωC)`）、
リアクタンスを `w について解いて` 共振角周波数が出る（`solve_equation` は素の式を `=0` とみなす）。

**再生成:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`
