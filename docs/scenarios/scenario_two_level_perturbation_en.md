# Practice-loop scenario S-j: perturbation corrections in a two-level system

> Run 2026-08-06.  The Rayleigh–Schrödinger series for $H=H_0+\lambda V$ in a two-level system,
> obtained from **general operations only** — no dedicated "perturbation expansion" operation.

**Backend:** Maxima / **Domain:** quantum mechanics.

### The physics

$$H=\begin{pmatrix}E_a & \lambda V\\ \lambda V & E_b\end{pmatrix}$$

The perturbation is purely off-diagonal, so **the first-order correction vanishes**
($\langle n|V|n\rangle=0$).  The second order is

$$E^{(2)}_n=\frac{|\langle m|V|n\rangle|^2}{E_n-E_m}
\quad\Longrightarrow\quad
E_\pm = E_{a,b} \mp \frac{\lambda^2V^2}{E_b-E_a}$$

— the upper level moves up and the lower one down by the same amount (**level repulsion**).  So
**their sum does not move**: $E_++E_-=E_a+E_b=\operatorname{Tr}H$, and that holds exactly, which
makes it an independent check on the series.

### Why no dedicated operation (the point of this scenario)

The eigenvalues of a $2\times2$ are **exact in closed form**, so the perturbation series is nothing
but their Taylor expansion in the coupling $\lambda$.  Two operations that already exist are enough:

| Step | Operation | What it does |
|---|---|---|
| 1 | `eigenvalues` | the exact eigenvalues (a closed form with a square root) |
| 2 | `taylor_expand` | expand in $\lambda$ — that IS the Rayleigh–Schrödinger series |

The ticket listed this scenario as **waiting on C-Ext-4 (perturbation series)**, but C-Ext-4 Phase 3
is the **time-dependent Dyson series** for Green functions in Julia — a different thing.  As with S-c
and BP-2, **measuring showed there was nothing to wait for**.

---

### Steps

#### ST-1: the Hamiltonian

| Item | Content |
|---|---|
| Action | new session, field "quantum mechanics" (Maxima) |
| Input | `Define Ea, Eb, V, \lambda as scalars` |
| Input | `H = [Ea, \lambda*V; \lambda*V, Eb]` |
| Result | |

---

#### ST-2: the exact eigenvalues

| Item | Content |
|---|---|
| Input | `eigenvalues of E1` |
| Expected | $\dfrac{E_a+E_b\pm\sqrt{4V^2\lambda^2+(E_a-E_b)^2}}{2}$ (E2) |
| Check | a **closed form** with the square root — everything after this is expansion |
| Result | |

---

#### ST-3: the lower branch, to first order

| Item | Content |
|---|---|
| Input | `(Ea + Eb - sqrt(4*V^2*\lambda^2 + (Ea-Eb)^2))/2` (E3) |
| Input | `taylor expand E3 in \lambda around 0 to order 1` |
| Expected | **$E_a$** (E4) — **the first-order correction vanishes**, because the perturbation is off-diagonal ($\langle n|V|n\rangle=0$) |
| Result | |

---

#### ST-4: the lower branch, to second order

| Item | Content |
|---|---|
| Input | `taylor expand E3 in \lambda around 0 to order 2` |
| Expected | $E_a-\dfrac{V^2\lambda^2}{E_b-E_a}$ (E5) |
| Check | matches the textbook $\dfrac{|\langle m|V|n\rangle|^2}{E_n-E_m}$ |
| Result | |

---

#### ST-5: the upper branch, to second order

| Item | Content |
|---|---|
| Input | `(Ea + Eb + sqrt(4*V^2*\lambda^2 + (Ea-Eb)^2))/2` (E6) |
| Input | `taylor expand E6 in \lambda around 0 to order 2` |
| Expected | $E_b+\dfrac{V^2\lambda^2}{E_b-E_a}$ (E7) — the opposite sign, i.e. **level repulsion** |
| Result | |

---

#### ST-6: the check (the trace is preserved)

| Item | Content |
|---|---|
| Input | `E5 + E7` |
| Input | `simplify E8` |
| Expected | **$E_a+E_b$** (E9) — the second-order contributions cancel |
| Check | $\operatorname{Tr}H=E_a+E_b$ holds exactly, so this is an **independent check** on the series |
| Result | |

---

### Verification checklist

- [x] the exact $2\times2$ eigenvalues come out in closed form
- [x] the first-order correction vanishes (off-diagonal perturbation)
- [x] the second-order correction is $\pm V^2\lambda^2/(E_b-E_a)$, the textbook form
- [x] the two branches sum to the trace $E_a+E_b$ (an independent check)
- [x] no dedicated perturbation operation is used — `eigenvalues` and `taylor_expand` suffice

---

## What this run found

**The ticket's "waiting on" was a different thing.**  C-Ext-4 Phase 3 (perturbation series) is the
time-dependent Dyson series, a Green-function problem in Julia.  S-j does not need it.

**What it was really waiting on was NL field extraction (the BP-6 shape).**  `taylor_expand` has four
required fields (expr/var/point/order) and the positional mapper could not fill them:

| Input | What was extracted |
|---|---|
| `E1 を \lambda について 0 のまわりで 2 次までテイラー展開して` | point=**2**, order=**0** (swapped) |
| `taylor expand E1 in \lambda around 0 to order 2` | var=**expand**, point=**in** |

Both were returned as `needs_confirmation` **with those values already filled in**, so clicking
through gives an expansion about the wrong point at the wrong order, reported as a success.  The
definite integral (also four fields) had already been solved with a named-group pattern, so this
follows it (`taylor_expand_nl`, JA and EN).  **Fixed 2026-08-05.**

**"expand" alone means `expand`.**  `展開` is also an alias of `expand`, so a Taylor expansion has to
say so.  The new pattern **requires a stated variable and order**, so it cannot swallow a plain
`expand E1`.

**Verified:** 2026-08-06 (real Maxima, ST-1 through ST-6, all 10 steps).

---

## Generator commands (this `scenario` block is the single source)

```scenario domain=quantum_mechanics
@note: Perturbation theory for a two-level system. $H=H_0+\lambda V$, with $H_0$ eigenvalues $E_a, E_b$ and a purely off-diagonal $V$; $\lambda$ is the coupling.
Define Ea, Eb, V, \lambda as scalars
@note: Register the Hamiltonian as a $2\times2$ matrix.
H = [Ea, \lambda*V; \lambda*V, Eb]
@note: For a $2\times2$ the eigenvalues are exact, and the perturbation series is just that closed form expanded in $\lambda$ — no dedicated "perturbation expansion" operation is needed.
eigenvalues of E1
@note: Register the lower branch.
(Ea + Eb - sqrt(4*V^2*\lambda^2 + (Ea-Eb)^2))/2
@note: Expand to first order. The perturbation is off-diagonal ($\langle n|V|n\rangle=0$), so the first-order correction should vanish.
taylor expand E3 in \lambda around 0 to order 1
@note: Expand to second order. This should give the textbook $E^{(2)}_n=|\langle m|V|n\rangle|^2/(E_n-E_m)$.
taylor expand E3 in \lambda around 0 to order 2
@note: Do the same for the upper branch. The sign is opposite — the two levels repel.
(Ea + Eb + sqrt(4*V^2*\lambda^2 + (Ea-Eb)^2))/2
taylor expand E6 in \lambda around 0 to order 2
@note: The check. The sum of the two levels is the trace $E_a+E_b$, and that holds exactly, so the second-order contributions must cancel if the series is right.
E5 + E7
simplify E8
```
