# quantum_mechanics — 実例（tight-binding バンド構造）

このフォルダは、シナリオ [docs/scenarios/scenario_tight_binding_bands_jp.md](../../../docs/scenarios/scenario_tight_binding_bands_jp.md)
を実際に実行して得た**教育用の実例**です。

| ファイル | 内容 |
|---|---|
| `scenario_tight_binding_bands_jp.json` | 実行後のセッション（`sessions/` に置けばロード可能） |
| `scenario_tight_binding_bands_jp.tex` | export された LaTeX 文書（uplatex/dvipdfmx でコンパイル可） |

鎖／正方／三角／ハニカム／カゴメ／Lieb／SSH／抽象結晶記述子の分散関係 $\varepsilon(k)$ を厳密に辿ります。

**ドメインについて:** tight-binding の分散は**非相互作用・単一粒子**の計算で、F-1 3c-fourier 機構（CAS 非経由）
により standard（**Maxima**）セッションで評価される。第２量子化（多体）を前提とする julia_full ドメイン
`condensed_matter` ではなく、バックエンドが Maxima で整合する `quantum_mechanics`（単一粒子格子）に置いている。

**再生成:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`
