# statistical_mechanics — 実例（量子スピン系の統計力学）

このフォルダは、spin-algebra（CAS 非経由）で評価できる**量子スピン系の統計力学**シナリオを実際に実行して
得た**教育用の実例**です。いずれも standard（**Maxima**）セッションで再現できます。

| シナリオ | ファイル | 内容 |
|---|---|---|
| [scenario_spin_chain_spectrum.md](../../../docs/scenarios/scenario_spin_chain_spectrum_jp.md) | `scenario_spin_chain_spectrum_jp.{json,tex}` | 量子スピン鎖の**動的スペクトル**：交換子→モーメント→連分数→$S(\omega)$→高温展開（↔三角形数） |
| [scenario_ising_partition_function.md](../../../docs/scenarios/scenario_ising_partition_function_jp.md) | `scenario_ising_partition_function_jp.{json,tex}` | 2 次元イジングの**分配関数**（小格子・高温展開）：比熱の累積量展開、正方 vs 三角で奇数次 $\beta^3$ ↔ 三角形数 |
| [scenario_lattice_figures.md](../../../docs/scenarios/scenario_lattice_figures_jp.md) | `scenario_lattice_figures_jp.{json,tex}` ＋ `_lattice.svg` | **格子図の描画**（ハニカム／正方・周期境界）→ TeX エクスポートへ自動埋め込み。図生成は CAS 非経由（バックエンド非依存） |

各ファイル：`.json` は実行後のセッション（`sessions/` に置けばロード可能）、`.tex` は export 文書
（uplatex/dvipdfmx でコンパイル可）。

**ドメインについて:** これらは spin-algebra（モーメント・連分数・高温展開・$S(\omega)$）による**量子スピン系の
統計力学**で、CAS 非経由のため standard（Maxima）で評価される。第２量子化（多体・Julia）を前提とする
julia_full ドメイン `spin_systems` ではなく、バックエンドが Maxima で整合する `statistical_mechanics` に置いている。

**再生成:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`
