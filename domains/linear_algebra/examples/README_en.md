# linear_algebra — worked example (matrix diagonalization)

This folder holds an **educational worked example** obtained by actually running the
linear-algebra scenario. It reproduces in a standard (**Maxima**) session.

| Scenario | Files | Content |
|---|---|---|
| [scenario_matrix_diagonalization](../../../docs/scenarios/scenario_matrix_diagonalization_jp.md) | `scenario_matrix_diagonalization_jp.{json,tex}` | Find the eigenvalues and eigenvectors of a $2\times2$ symmetric matrix $A$, then diagonalize as $P^{-1}AP=D=\mathrm{diag}(3,1)$ with the normalized orthogonal matrix $P$ and the diagonal matrix $D$ |

For each: `.json` is the session after execution (loadable from `sessions/`), `.tex`
is the exported document (compiles with uplatex/dvipdfmx).

Follows the workflow eigenvalues → eigenvectors → orthogonal transform matrix $P$ /
diagonal matrix $D$ → $P^{-1}AP=D$. `diagonalize` returns the **orthogonal matrix** $P$
whose columns are the normalized eigenvectors; `compute the diagonal matrix` returns the
diagonal matrix $D$ of eigenvalues. Enter a matrix as `[a, b; c, d]` (`;` = row
separator); `[A, B]` without `;` is a commutator. The matrix operations
(`eigenvalues` / `eigenvectors` / `diagonalize` / `diagonal_matrix` / `inverse_matrix`
/ `matrix_multiply`) work on both the Maxima and Julia backends.

**Regenerate:** `PATH="$HOME/.juliaup/bin:$PATH" .venv/bin/python scripts/generate_scenario_examples.py`
