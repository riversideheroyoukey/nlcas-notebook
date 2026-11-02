# linear_algebra — 実例（行列の対角化）

このフォルダは、線形代数シナリオを実際に実行して得た**教育用の実例**です。
standard（**Maxima**）セッションで再現できます。

| シナリオ | ファイル | 内容 |
|---|---|---|
| [scenario_matrix_diagonalization.md](../../../docs/scenarios/scenario_matrix_diagonalization_jp.md) | `scenario_matrix_diagonalization_jp.{json,tex}` | $2\times2$ 対称行列 $A$ の固有値・固有ベクトルを求め、正規化直交行列 $P$ と対角行列 $D$ で $P^{-1}AP=D=\mathrm{diag}(3,1)$ と対角化 |

各ファイル：`.json` は実行後のセッション（`sessions/` に置けばロード可能）、`.tex` は export 文書
（uplatex/dvipdfmx でコンパイル可）。

固有値 → 固有ベクトル → 直交変換行列 $P$ / 対角行列 $D$ → $P^{-1}AP=D$ のワークフローを辿る。
`対角化` は正規化した固有ベクトルを列に並べた**直交行列** $P$ を、`対角行列を求めて` は固有値を
並べた対角行列 $D$ を返す。行列は `[a, b; c, d]`（`;` = 行区切り）で入力する（`;` なしの
`[A, B]` は交換子）。行列操作（`eigenvalues` / `eigenvectors` / `diagonalize` /
`diagonal_matrix` / `inverse_matrix` / `matrix_multiply`）は Maxima・Julia 両バックエンドで
動作する。

**再生成:** `PATH="$HOME/.juliaup/bin:$PATH" .venv/bin/python scripts/generate_scenario_examples.py`
