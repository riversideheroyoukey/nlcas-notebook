# Practice-loop sample scenario: checking Maxwell-equation consistency for given E, B

> (2026-07-03: an English translation of the Japanese scenario, verified against the
> real Maxima backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_maxwell_consistency_jp.md](scenario_maxwell_consistency_jp.md) for the
> Japanese version. No new i18n gaps were found — all the English NL patterns this
> scenario needs (vector field declaration, `\nabla \cdot`/`\nabla \times`, "take the
> partial derivative of ... with respect to ...", "solve En for X") were already fixed
> while building the earlier translations.)

Detail on **S-f (electromagnetism)** in the practice-loop scenario menu. Given concrete
electric and magnetic fields $\boldsymbol{E}$, $\boldsymbol{B}$, use vector calculus
(div/curl) to confirm **the Maxwell equations are mutually consistent**, and further
confirm that the consistency condition yields **the dispersion relation
$\omega=ck$**.

**Backend:** standard (Maxima) / **Domain:** electromagnetism (electromagnetism). Cartesian coordinates.

## Physical background

The vacuum, source-free Maxwell equations (SI, $\mu_0\varepsilon_0=1/c^2$):

$$\nabla\!\cdot\!\boldsymbol{E}=0,\quad
  \nabla\!\cdot\!\boldsymbol{B}=0,\quad
  \nabla\!\times\!\boldsymbol{E}=-\frac{\partial \boldsymbol{B}}{\partial t},\quad
  \nabla\!\times\!\boldsymbol{B}=\frac{1}{c^2}\frac{\partial \boldsymbol{E}}{\partial t}$$

**A linearly polarized plane wave traveling in the +x direction** ($E_0=1$):

$$\boldsymbol{E}=(0,\ \cos(kx-\omega t),\ 0),\qquad
  \boldsymbol{B}=\Bigl(0,\ 0,\ \tfrac{1}{c}\cos(kx-\omega t)\Bigr)$$

Both divergences are identically 0, and both curl equations give the same consistency
condition $\omega=ck$.

## How to supply components to a vector field (★ the heart of this scenario)

`define_vector_field` **only registers the components as abstract functions**; it does
not take component expressions as arguments. The concrete components are supplied
**after declaration, by assigning to each component via the H-1 equation form
(`E_y = ...`)** (see [doc 13 §2](../developer_manuals/13_vector_field_design_spec.md) and
[the user guide "vector calculus" §3](../user_guide.md)). Free symbols other than the
coordinates $(x,y,z)$ — such as $t,k,\omega,c$ — can also be used.

## Steps and expected results

### 1. Declare the fields and supply components

| Input (natural language / equation) | Effect |
|---|---|
| `E as a vector field` | abstractly registers the components `E_x,E_y,E_z` |
| `B as a vector field` | abstractly registers the components `B_x,B_y,B_z` |
| `E_x = 0` / `E_y = cos(k*x - w*t)` / `E_z = 0` | bind $\boldsymbol{E}$'s components |
| `B_x = 0` / `B_y = 0` / `B_z = cos(k*x - w*t)/c` | bind $\boldsymbol{B}$'s components |

### 2. Divergence (Gauss's law, no monopoles)

Entering nabla notation **as an expression** computes it on the spot (`div E` / `div(E)`
also work).

| Input | Expected result |
|---|---|
| `\nabla \cdot E` | $\nabla\!\cdot\!\boldsymbol{E}=0$ |
| `\nabla \cdot B` | $\nabla\!\cdot\!\boldsymbol{B}=0$ |

### 3. Curl (the left-hand sides of Faraday's and Ampère's laws)

| Input | Expected result |
|---|---|
| `\nabla \times E` (`curl E` / `rot E` also work) | $\nabla\!\times\!\boldsymbol{E}=[0,\,0,\,-k\sin(kx-\omega t)]$ |
| `\nabla \times B` | $\nabla\!\times\!\boldsymbol{B}=[0,\,k\sin(kx-\omega t)/c,\,0]$ |

### 4. Build $\partial\boldsymbol{B}/\partial t$ **component-wise** and reassemble it

> **A design note (important):** $\partial/\partial t$ is **not a vector operator**.
> Simply "differentiating the field name `B` with respect to t" does not work as a
> vector operation (no such shorthand is provided). **The correct approach is to take
> the time derivative of each component and reassemble the results into a new vector.**
> Not being able to handle all 3 components at once is an accepted design limitation —
> each component is processed individually.

> **Specify it as a partial derivative (important):** the component
> $B_z=\cos(kx-\omega t)/c$ is a multivariable function of $x$ and $t$. Just
> "differentiate with respect to t" is ambiguous between total and partial derivatives
> and triggers a confirmation prompt (the ∂/d disambiguation). Since $\partial
> \boldsymbol{B}/\partial t$ means the **partial** derivative at fixed $x$, always write
> **"take the partial derivative ... with respect to t."**

| Input | Expected result |
|---|---|
| `Take the partial derivative of B_x with respect to t` / `B_y` | $0$ / $0$ |
| `Take the partial derivative of B_z with respect to t` | $\dfrac{\partial B_z}{\partial t}=\dfrac{\omega}{c}\sin(kx-\omega t)$ |
| `dBdt as a vector field` + `dBdt_x = 0` / `dBdt_y = 0` / `dBdt_z = w*sin(k*x - w*t)/c` | **reassemble** $\partial\boldsymbol{B}/\partial t$ as a vector |
| (check) `\nabla \cdot dBdt` | $0$ (confirms the reassembled vector is usable in downstream calculations; div is computed directly from the expression) |

### 5. Faraday/Ampère consistency → the dispersion relation $\omega=ck$ (**compared component-wise**)

The vector equation $\nabla\!\times\!\boldsymbol{E}=-\partial\boldsymbol{B}/\partial t$
is solved as a **component-wise equality** (no combined comparison is performed — this
is done component-by-component as a matter of principle). The non-trivial $z$
component:

| Input | Expected result |
|---|---|
| register `-k*sin(k*x - w*t) = -w*sin(k*x - w*t)/c` (the $z$ component of $(\nabla\!\times\!\boldsymbol{E})_z=(-\partial\boldsymbol{B}/\partial t)_z$) | the $z$-component equality of Faraday's law |
| `Solve` the above equation `for w` | $\omega=ck$ (the dispersion relation; the other root $kx/t$ is the unphysical $\sin=0$ solution) |

Building and solving the corresponding $y$-component equation for Ampère's law,
$(\nabla\!\times\!\boldsymbol{B})_y=(\tfrac{1}{c^2}\partial\boldsymbol{E}/\partial t)_y$,
independently gives the same $\omega=ck$. Both curl equations are consistent, and all
four Maxwell equations are mutually consistent.

## Observed behaviour / notes

- **Completed entirely with the existing div / curl / differentiate / solve
  operations** (no additional functionality needed). Cartesian `nb_div`/`nb_curl_*`
  take the component expressions as arguments and expand the sum of partial derivatives
  using the values bound via H-1.
- **Components are supplied via equations** (they cannot be passed as arguments to
  `define_vector_field`). This is the most confusing point when reproducing this
  scenario.
- **$\partial/\partial t$ does not act directly on a vector.** Differentiate each
  component "with respect to t" and reassemble the results into a new vector field for
  further use (§4 above). The reassembled vector can be used in downstream `div`/`curl`
  calculations.
- **Vector-equation comparisons are done component-by-component** (register the
  $z$-component equality → "solve for w" → $\omega=ck$). Comparing/differentiating all
  3 components at once is not provided, but since each component is self-contained this
  is an acceptable design tradeoff.
- Time $t$, wavenumber $k$, angular frequency $\omega$, and $c$ are free symbols.

**Verified:** 2026-07-03 (real Maxima backend, generator run, all 20 steps completed
end-to-end: div E = div B = 0, curl E = $[0,0,-k\sin]$, curl B = $[0,k\sin/c,0]$, the
component time derivative $\partial B_z/\partial t=\omega\sin/c$ reassembled into
`dBdt` → `div(dBdt)=0`, and solving the Faraday $z$-component equality confirms
$\omega=ck$).

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=electromagnetism
@note: Consider the electromagnetic field. Prepare the electric field $\mathbf E$ and magnetic field $\mathbf B$ as vector fields (components registered abstractly, to be assigned concrete expressions next).
E as a vector field
B as a vector field
@note: Assign the components of the electric field for a linearly polarized plane wave traveling in the $+x$ direction (amplitude 1, wavevector $k$, angular frequency $\omega$).
E_x = 0
E_y = cos(k*x - w*t)
E_z = 0
@note: Assign the corresponding components of the magnetic field.
B_x = 0
B_y = 0
B_z = cos(k*x - w*t)/c
@note: Compute the divergence of the electric field $\nabla\cdot\mathbf E$. In vacuum (no charge), this should vanish identically (Gauss's law).
\nabla \cdot E
@note: Compute the divergence of the magnetic field $\nabla\cdot\mathbf B$. Since no magnetic charge exists, this should vanish identically.
\nabla \cdot B
@note: Compute the curl of the electric field $\nabla\times\mathbf E$ (the left-hand side of Faraday's law).
\nabla \times E
@note: Compute the curl of the magnetic field $\nabla\times\mathbf B$ (the left-hand side of Ampere's law).
\nabla \times B
@note: Since $\partial/\partial t$ cannot act directly on a vector field, take the time derivative component by component. Here, take the partial derivative of the nonzero component $B_z$ with respect to $t$.
Take the partial derivative of B_z with respect to t
@note: Assemble the derivative just found into a new vector field, which we name `dBdt` — a plain identifier chosen here to stand for $d\mathbf B/dt$ (it is not a CAS keyword or special syntax; any other name would work identically), assigning each component in turn.
dBdt as a vector field
dBdt_x = 0
dBdt_y = 0
dBdt_z = w*sin(k*x - w*t)/c
@note: Confirm that the assembled $d\mathbf B/dt$ can be used in further calculations (such as a divergence) by computing its divergence.
\nabla \cdot dBdt
@note: Register the $z$-components of Faraday's law $\nabla\times\mathbf E=-\partial\mathbf B/\partial t$ as an equality (comparing the $z$-component of each side).
-k*sin(k*x - w*t) = -w*sin(k*x - w*t)/c
@note: Solve this equality for the angular frequency $\omega$ to obtain the consistency condition (dispersion relation) $\omega=ck$.
Solve E16 for w
```
