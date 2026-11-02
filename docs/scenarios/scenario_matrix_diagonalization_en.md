# Practice-loop sample scenario: matrix diagonalization (S-i)

A detailed write-up of practice-loop scenario-menu item **S-i (linear algebra)**.
For the $2\times2$ symmetric matrix $A=\begin{pmatrix}2&1\\1&2\end{pmatrix}$: find the
eigenvalues and eigenvectors, build the normalized orthogonal matrix $P$ and the
diagonal matrix $D$, and verify $P^{-1}AP=D=\mathrm{diag}(3,1)$.

**Backend:** standard (Maxima) / **Domain:** linear algebra (linear_algebra).

## Steps and expected results

| # | Input | Expected result |
|---|---|---|
| 1 | `A = [2, 1; 1, 2]` | matrix $A$ (E1) |
| 2 | `Find the eigenvalues of E1` | $\{1, 3\}$ (E2) |
| 3 | `Find the eigenvectors of E1` | $(1,1)$ for 3, $(1,-1)$ for 1 (E3) |
| 4 | `Diagonalize E1` | orthogonal $P$ from the normalized eigenvectors (E4) |
| 5 | `Find the diagonal matrix of E1` | $D=\mathrm{diag}(3,1)$ (E5) |
| 6 | `Find the inverse of E4` | $P^{-1}$ (E6) |
| 7 | `Find the matrix product of E6 and E1` | $P^{-1}A$ (E7) |
| 8 | `Find the matrix product of E7 and E4` | $P^{-1}AP=D=\mathrm{diag}(3,1)$ (E8) |

Enter a matrix as `[a, b; c, d]` (`;` separates rows; `[A, B]` without `;` is a
commutator). **The TeX spellings read as the same matrix** (2026-09-10,
TEX-MATRIX-INPUT): `\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}` is
`[2, 1; 1, 2]` — so the matrix written in the background section above can be
copied and typed straight in (`bmatrix`, `matrix`, `Bmatrix`, `smallmatrix`,
`array` and plain TeX `\pmatrix{…}` too; vertical bars `\begin{vmatrix}` are
a DETERMINANT). The matrix operations (`eigenvalues` / `eigenvectors` / `diagonalize` /
`diagonal_matrix` / `inverse_matrix` / `matrix_multiply`) are general and work on both
the Maxima and Julia backends. → [scenario_matrix_diagonalization_jp.md](scenario_matrix_diagonalization_jp.md)

**Worked example:** [`domains/linear_algebra/examples/`](../../domains/linear_algebra/examples/)
(`scenario_matrix_diagonalization_{jp,en}.{json,tex}`).

## generator command block (the single source the generator reads)

```scenario domain=linear_algebra
@note: Register the symmetric matrix $A=\begin{pmatrix}2&1\\1&2\end{pmatrix}$.
A = [2, 1; 1, 2]
@note: Find the eigenvalues of $A$.
Find the eigenvalues of E1
@note: Find the eigenvectors of $A$.
Find the eigenvectors of E1
@note: Find the orthogonal transformation matrix $P$ whose columns are the normalized eigenvectors (the matrix that diagonalizes $A$).
Diagonalize E1
@note: Find the diagonal matrix $D$ with the eigenvalues on its diagonal.
Find the diagonal matrix of E1
@note: To compute the similarity transform $P^{-1}AP$, first find the inverse $P^{-1}$ of $P$ (since $P$ is orthogonal, $P^{-1}$ should equal $P^\top$).
Find the inverse of E4
@note: Find the matrix product $P^{-1}A$.
Find the matrix product of E6 and E1
@note: Multiply by $P$ to complete $P^{-1}AP$. This should match the diagonal matrix $D$ from earlier (the eigenvalues on the diagonal).
Find the matrix product of E7 and E4
```
