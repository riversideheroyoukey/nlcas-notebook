# Practice-loop scenario: matrix elements over a spin-dependent basis

> Run 2026-08-09.  The coefficients that appear when a field operator is expanded
> in a **spin-dependent basis** — the one- and two-body matrix elements — must be
> writable in the notebook **as all-space integrals that stay definitions**.
> The first half of a pair; the second is
> [the spin-ful first-order perturbation scenario](scenario_spinful_perturbation_en.md).

**Backend:** Maxima / **Domain:** quantum mechanics.

### Why this is worth checking

Expanding a field operator in a spin-dependent basis,

$$\hat\psi_\sigma(\mathbf x)=\sum_m \varphi_{m\sigma}(\mathbf x)\,\hat c_{m\sigma}$$

makes the Hamiltonian's coefficients **all-space integrals by definition**:

$$h_{m\sigma,n\sigma'}=\int\! d^3x\;\varphi^*_{m\sigma}(\mathbf x)\,\hat T(\mathbf x)\,\varphi_{n\sigma'}(\mathbf x)$$

And that is the point: **this integral is not something to carry out.**  The basis
$\varphi_{m\sigma}$ is arbitrary, so there is nothing to evaluate.  The integral IS
the definition of $h_{m\sigma,n\sigma'}$, and what follows carries that coefficient
**as a symbol**.

So what the notebook has to offer is not "it computes" but **"it can be written
down, it shows what was written, and it can be used afterwards."**  Until
2026-08-08 it could not:

- $\int d^3x$ could only be lowered to a ONE-variable integral, and was **silently
  computed as a product**: `(d^3·int·x·f(x))/(8π³)` (open_items B-10 ②-g)
- and even after that, **an expansion that also evaluates falls over here**.
  Measured: asking Maxima to integrate an arbitrary-basis matrix element over all
  space does not merely fail to simplify — it returns `ok=False` with an EMPTY
  answer (a convergence question that cannot be answered).  So the expansion HOLDS
  and evaluation is a separate request.

This scenario checks both, on the real thing.

### The physics

1. Write the orthonormality integral $\int d^3x\,\varphi^*_{m\sigma}\varphi_{n\sigma'}$ and open it into components
2. Define the one-body matrix element $h_{m\sigma,n\sigma'}$ (H-1 equation form)
3. Open it into components → an iterated integral over the three Cartesian components
4. **Ask for it to be evaluated and watch it stay a definition** (an arbitrary basis has nothing to evaluate)
5. Define the two-body matrix element $V$ → a **six-dimensional** ($d^3x\,d^3y$) integral opens

Spin rides along throughout as the SECOND index of $\varphi_{m,s}$.  That mode and
spin stay SEPARATED in the display is also what makes the display re-readable.

---

### Steps

#### ST-1: orthonormality

| Item | Content |
|---|---|
| Action | New session, domain "quantum mechanics" → Maxima backend |
| Input | `\int d^3x \varphi^*_{m,s}(x) \varphi_{n,t}(x)` |
| Check | Shown as written ($d^3x$ does not degrade into a product) |
| Input | `expand into components` |
| Expected | $\int_{-\infty}^{\infty}\!\!\int_{-\infty}^{\infty}\!\!\int_{-\infty}^{\infty} \varphi_{n,t}\,\varphi^*_{m,s}\;dx_z dx_y dx_x$ |
| Check | The range comes in as ALL SPACE.  Mode and spin stay separated in $\varphi_{m,s}$ |
| Result | |

---

#### ST-2: the one-body matrix element

| Item | Content |
|---|---|
| Input | `h_{m,s,n,t} = \int d^3x \varphi^*_{m,s}(x) T(x) \varphi_{n,t}(x)` |
| Input | `expand into components` |
| Expected | A threefold iterated integral of $T\,\varphi_{n,t}\,\varphi^*_{m,s}$ |
| Check | $\varphi_{m,s}$ opens to the THREE arguments $\varphi(x_x,x_y,x_z)$ — unopened, Maxima reads it as a constant in each component and returns $x_x x_y x_z\cdot\varphi$ |
| Result | |

---

#### ST-3: a definition stays a definition

| Item | Content |
|---|---|
| Input | `evaluate E4` |
| Expected | **Unchanged** (the held iterated integral) |
| Check | Neither an error nor an invented value.  With an arbitrary basis, this is the right answer |
| Result | |

---

#### ST-4: the two-body matrix element (six-dimensional)

| Item | Content |
|---|---|
| Input | `V_{m,s,n,t,p,u,q,w} = \int d^3x \int d^3y \varphi^*_{m,s}(x) \varphi^*_{n,t}(y) v(x,y) \varphi_{p,u}(y) \varphi_{q,w}(x)` |
| Input | `expand into components` |
| Expected | Three components of $x$ and three of $y$ — a **sixfold** iterated integral |
| Check | $x$ and $y$ open into separate triples and do not mix.  All eight indices (four mode, four spin) survive |
| Result | |

---

### Verification checklist

- [x] $\int d^3x$ is read as a volume measure, not degraded into a product
- [x] The spin-dependent basis $\varphi_{m,s}$ opens to three arguments (unopened it is treated as a constant — a wrong answer)
- [x] With no range written, all space is used
- [x] **Asking to evaluate leaves the definition alone** (an arbitrary basis is not something to evaluate)
- [x] The two-body sixfold integral opens without mixing $x$ and $y$
- [x] Mode and spin are shown SEPARATED (a form that can be typed back in)

---

## Behaviour confirmed / notes

**Expanding and evaluating are separate requests.**  Not a preference but a
requirement: measured 2026-08-08, asking Maxima to evaluate an arbitrary-basis
matrix element over all space returns `ok=False` with an empty answer (an
unanswerable convergence question).  An expansion that also evaluated would take
this scenario down at ST-2.

**With no range written, it is all space.**  A volume measure is not an
antiderivative.  Read as one it returns
$\pi^{3/2}\mathrm{erf}(x_x)\mathrm{erf}(x_y)\mathrm{erf}(x_z)/8$ — an answer with the
integration variables still in it, and containing $\pi^{3/2}$, so easy to skim
past.  Filling the range in is only safe because the result is HELD: the range is
recorded as meaning, and nothing is evaluated.

**Complex conjugation goes in the physics spelling.** The star sits in front:
$\varphi^*_{m,s}(x)$.  Until 2026-08-09 that was REFUSED as a "multi-index applied
form" — the star was being counted as one of the indices, so single-index
$\psi^*_n(x)$ was refused too.  Worse, the braced spellings (`^{\ast}`, `^{*}` —
what a TeX editor emits, and what **this system's own display returns**) were not
refused but silently mangled into `psi^(\ast)*(x)`: **typing the display back in
gave a different expression.**  All of `^*`, `^{*}`, `^{\ast}`, `^\ast`,
`^{\star}` and trailing `f(x)^*` now reach the CAS as one thing,
`conjugate(...)`.

**The star must not change what the rest of the symbol means:**
$\varphi^*_{m,s}(x)$ and $\varphi_{m,s}(x)$ name the same `phim, s (reference omitted from the public snapshot)`.  Deciding
the subscript afresh on the starred side alone would leave the two factors of an
inner product permanently unable to pair up.  `\overline{f(x)}` reads as a
conjugate as well — while a bare `\overline{z}` stays the NAME `z_bar`, which is
H-1's naming convention and deliberately unchanged.

**The one-body operator's spelling.** `\hat{h}(x)` displays as `h_{hat}(x)` (an
applied subscript folds into the name), so it is not used here.  `T` is used
instead — it does not collide with the coefficient $h$ — and reads as the
one-body operator (kinetic plus external potential).

**What comes next.** With the coefficients defined, the operator algebra is a
Julia session's job (the backend is fixed per session, Core Principle #6).  See
[the spin-ful first-order perturbation scenario](scenario_spinful_perturbation_en.md).

**Confirmed:** 2026-08-09 (live Maxima, ST-1 through ST-4 completed).

---

## generator commands (this `scenario` block is the single source of the generated example)

```scenario domain=quantum_mechanics
@note: Expand a field operator in a **spin-dependent basis**: $\hat\psi_\sigma(\mathbf x)=\sum_m \varphi_{m\sigma}(\mathbf x)\hat c_{m\sigma}$. The coefficients are then all-space integrals BY DEFINITION. The basis is arbitrary, so there is nothing to carry out — the integral IS the definition. What the notebook has to offer is not "it computes" but "it can be written down and used afterwards".
@note: Orthonormality first. $\int d^3x$ is a volume measure, not a one-variable integral.
\int d^3x \varphi^*_{m,s}(x) \varphi_{n,t}(x)
@note: Open it into components. A threefold iterated integral, over all space even though no range was written. Spin ($s$, $t$) rides along separated from the mode — without the separator it becomes "one index called ms", and the display can no longer be typed back in.
expand into components
@note: The one-body matrix element. $T$ is the one-body operator (kinetic plus external potential). The $\varphi_{m,s}$ in the integrand has to open into three arguments too — unopened, Maxima reads it as a constant in each component and returns $x_x x_y x_z \cdot \varphi$.
h_{m,s,n,t} = \int d^3x \varphi^*_{m,s}(x) T(x) \varphi_{n,t}(x)
expand into components
@note: Now ask for it to be evaluated. **Unchanged is the right answer** — the basis is arbitrary, so this integral is a definition and stays one. An expansion that also evaluated would fall over here (measured: Maxima returns `ok=False` with an empty answer for an arbitrary-basis matrix element over all space).
evaluate E4
@note: The two-body matrix element: **six-dimensional** ($d^3x\,d^3y$), with four spins ($s,t,u,w$). Watch $x$ and $y$ open into separate triples without mixing.
V_{m,s,n,t,p,u,q,w} = \int d^3x \int d^3y \varphi^*_{m,s}(x) \varphi^*_{n,t}(y) v(x,y) \varphi_{p,u}(y) \varphi_{q,w}(x)
expand into components
```
