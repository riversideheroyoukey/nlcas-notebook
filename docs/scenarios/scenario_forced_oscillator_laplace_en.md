# Practice-loop sample scenario: forced oscillator initial-value problem (Laplace transform)

Solve the **initial-value problem** for a driven harmonic oscillator all the way back to the
$t$ domain. The builtin `laplace` is not used. No scenario-specific operation is
added: everything runs on **general operations** (definite integration, E-id
references, solve, assumptions, inverse transform).

**Backend:** standard (Maxima) / **domain:** differential equations.

## Why this scenario is needed

There are three columns the existing
[RLC transfer function (S-e)](scenario_rlc_transfer_function_en.md) **never enters**:

| | RLC | this scenario |
|---|---|---|
| Right-hand side (drive) | time-dependent **unknown** $v$ → $V(s)$ | **explicit function** $F_0\sin(\omega t)$ → definition integral |
| Initial conditions | set to **zero** (that is what defines a transfer function) | solved at a **nonzero value** |
| Inverse transform | none (stops in the $s$ domain) | back to the $t$ domain, **closing the round trip** |

Each of the three produced a defect on the day this scenario was written (see the end).
**Looking for a column an existing scenario does not enter is faster than adding
another scenario.**

## Physical background

An undamped harmonic oscillator under a drive (mass $m$, spring constant $k$, drive
amplitude $F_0$, drive angular frequency $\omega$):

$$m\ddot x + kx = F_0\sin(\omega t),\qquad x(0)=1,\ \dot x(0)=0.$$

The transform is linear, so each term is replaced by its image — but the two images
here are **of different kinds**:

- the image of the unknown's second derivative is a **theorem given by integration by
  parts**,
  $$\int_0^\infty \ddot x\,e^{-st}dt = s^2X(s) - s\,x(0) - \dot x(0),$$
  in which $u=e^{-st}$ is fixed by the structure of the transform, so **there is no
  choice to make**. Being a theorem, the researcher writes it.
- the drive term contains no unknown, so its definition integral can actually be
  computed:
  $$\int_0^\infty \sin(\omega t)\,e^{-st}dt = \frac{\omega}{s^2+\omega^2}.$$
  That is a job for a general operation (definite integration).

State the initial conditions (released from rest at unit displacement), solve for $X$,
and invert under $m\omega^2 \ne k$ (off resonance): the result is a free term at the
natural frequency $\sqrt{k/m}$ plus a forced term at the drive frequency $\omega$.

## Steps and expected results

Each stage is its own YOU card. The boundary terms $x_0 \equiv x(0)$ and
$v_0 \equiv \dot x(0)$ are introduced as **ordinary symbols**, so any value can be
stated and solved for.

| Stage | Input (natural language / formula) | Result |
|---|---|---|
| 1 | `Declare F_0 as a scalar` | register $F_0$ **as a name** (its subscript is not a component index) |
| 2 | `Define x as a time-dependent function` | declare $x$ as a function of $t$ |
| 3 | `Let X be the Laplace transform of x` | definition card $X(s)=\int_0^\infty x(t)e^{-st}dt$ |
| 4 | `Assume s > 0` / `Assume \omega > 0` | the **convergence conditions** of the definition integral (without them the CAS asks about the sign and the step fails) |
| 5 | `$\sin(\omega t) e^{-s t}$` (registered as a formula) | the integrand of the definition integral |
| 6 | `Integrate E2 with respect to t from 0 to \infty` | $\displaystyle\int_0^\infty \sin(\omega t)e^{-st}\,dt=\frac{\omega}{s^2+\omega^2}$ (**the explicit function's image**) |
| 7 | `$m (s^2 X - s x_0 - v_0) + k X = F_0 E3$` (registered as a formula) | $m(s^2X - sx_0 - v_0)+kX=\dfrac{F_0\omega}{s^2+\omega^2}$ (the derivative rule written by the researcher; the drive's image composed in **by E-id**) |
| 8 | `$x_0 = 1$` / `$v_0 = 0$` (registered as formulas) | the **nonzero** initial conditions. An H-1 definition is sent to the CAS when registered, so these are in force from here on |
| 9 | `Solve E4 for X` | $X=\dfrac{ms^3+m\omega^2 s+F_0\omega}{ms^4+(m\omega^2+k)s^2+k\omega^2}$ |
| 10 | `Assume m > 0` / `Assume k > 0` | prerequisites of the inversion (an unknown sign makes the CAS ask an interactive question) |
| 11 | `Take the inverse Laplace transform of E7 from s to t` | $x(t)=\cos\!\sqrt{\tfrac km}\,t+\dfrac{F_0 m^{3/2}\omega}{m\sqrt k\,(m\omega^2-k)}\sin\!\sqrt{\tfrac km}\,t-\dfrac{F_0\sin(\omega t)}{m\omega^2-k}$ (at $t=0$, $x=1$ — consistent with the stated conditions) |

## Observed behaviour / notes

- **The two images come from two places, and that is correct.** The explicit function's
  image is computed by the CAS (stage 6); the derivative's image is written by the
  researcher (stage 7). The first needs a computation, the second involves no choice.
  That distinction is exactly why the `laplace_by_definition` macro could be deleted
  (open_items SCENARIO-MACRO, case 7).
- **Convergence conditions are visible steps.** Running stage 6 without them makes the
  CAS ask "Is s positive, negative or zero?", and the step **fails with that reason**
  (the session survives). The deleted macro did an assume/forget silently — an explicit
  step is also what the "no hidden modelling choices in shared code" rule asks for.
- **The initial conditions take effect on registration.** Registering $x_0=1$ hands it
  to the CAS, so stage 9's solve needs no separate substitution step — and adding one
  would be refused with "not present in the target", because it has already been
  applied. The E-id reference in stage 7 avoids the same double application.
- **The inversion needs sign assumptions.** With the signs of $m,k,\omega$ unknown,
  Maxima asks an interactive question and the operation returns as a "confirmation
  requested" failure. Stating the physical positivity **as an explicit assumption step**
  is the honest route; no default is baked into shared code.
- **Off resonance is assumed.** The solution above presumes $m\omega^2 \ne k$. At
  $\omega^2=k/m$ the denominator vanishes and resonance (a growing $t\sin\omega t$
  solution) must be treated separately.
- **$F_0$ must be declared as a name.** Undeclared, the rule for unregistered subscripts
  lowers it to $F(0)$, and the same quantity appears in one document under two
  spellings.
- **The inverse transform's result is labelled $x(t) =$.** That comes from stage 3's
  definition card, which records the $X \leftrightarrow x$ pairing. The CAS holds the
  pairing too (`ilt(laplace(x(t),t,s),s,t)` returns $x(t)$), but it does **not** know
  the researcher's name $X$ (`ilt(X,s,t)` → $X\delta(t)$). Anchoring to the CAS's noun
  form so no table is needed is open_items **LAPLACE-PAIR-ANCHOR**.

## Defects this scenario found (2026-08-13)

Four silent-wrongs, each returned **as a success**, on the day it was written. ①②③ are
fixed at the root; ④ is filed only, being larger.

1. **The drive term was not transformed.** The linearity split replaced only terms
   containing the unknown and **passed the rest through**, so an equation with a
   $t$-domain term sitting inside the $s$ domain was returned as a success. RLC's
   right-hand side is the unknown $v$, so it never hits this. **The split that had this
   defect was deleted with the macro on 2026-08-15** — there is no longer a code path
   that can pass a term through.
2. **$x_{ic1}$ became a double subscript and the `.tex` would not typeset**
   (`notebook/runtime/maxima_backend_handler.py`). Maxima's `tex1` splits the trailing
   digit run off an italicised name (`{\it x\_ic}_{1}`), so unwrapping produced
   `x_{ic}_{1}` — a LaTeX hard error. The same happens for `k_B1` or `a_b2`: a general
   defect, nothing to do with Laplace.
3. **The inverse transform transformed the equation's left side too**
   (`notebook/engine.py`). Applying `ilt` to $X=\cdots$ makes Maxima read $X$ as a
   constant and return **$X\delta(t)=\cdots$**. The decision to strip a bare-symbol left
   side was written as three separate membership sets and the transforms were in none of
   them (the remaining five operations are open_items EQ-SOURCE-BARE-LHS).
4. **English "inverse Laplace transform" resolved to the forward transform**
   (fixed at the root 2026-08-15, see below).

The rewrite (2026-08-15) turned up one more:

5. **An integral's upper bound displayed as `inf` rather than $\infty$**
   (`notebook/runtime/maxima_constants.py`). The constant table had `minf` → $-\infty$
   but no bare `inf`, so the researcher's own bound was about to reach the published
   `.tex` as the italic word "inf". The macro built its display itself and never took
   this path.

```scenario domain=differential_equations
@note: Solve the initial-value problem for an undamped driven harmonic oscillator $m\ddot x + kx = F_0\sin(\omega t)$. First declare the drive amplitude $F_0$ as a name (not as a component index).
Declare F_0 as a scalar
@note: Declare the displacement $x$ as a function of time $t$.
Define x as a time-dependent function
@note: Name the Laplace transform of $x$ as $X$ (an s-domain quantity) via a definition card. The inverse transform's result label reads this pairing.
Let X be the Laplace transform of x
@note: State the convergence conditions of the definition integral $\int_0^\infty e^{-st}dt$ as visible steps. Without them the CAS asks about the sign and the step fails (the session survives).
Assume s > 0
Assume \omega > 0
@note: The drive is an explicit function, so its definition integral can be computed. Register the integrand $\sin(\omega t)e^{-st}$ first.
$\sin(\omega t) e^{-s t}$
@note: Integrate from $0$ to $\infty$ by definition to get $\mathcal L\{\sin(\omega t)\} = \omega/(s^2+\omega^2)$. The builtin `laplace` is not used.
Integrate E2 with respect to t from 0 to \infty
@note: Assemble the s-domain equation of motion. The image of $\ddot x$, namely $s^2X - s x(0) - \dot x(0)$, is the theorem integration by parts gives (with $u=e^{-st}$ fixed by the structure, so no choice is made), so the researcher writes it; the drive's image is composed in by the E-id computed above.
$m (s^2 X - s x_0 - v_0) + k X = F_0 E3$
@note: State the initial conditions. Unlike RLC, which zeroes them to obtain a transfer function, release from rest at unit displacement: $x(0)=1,\ \dot x(0)=0$. An H-1 definition is sent to the CAS on registration, so these are in force from here on.
$x_0 = 1$
$v_0 = 0$
@note: Solve the s-domain equation for $X$ to get the closed form. The initial conditions are already in force.
Solve E4 for X
@note: Before inverting, explicitly assume the physical positivity of the quantities. With unknown signs the CAS asks an interactive question and the operation stops.
Assume m > 0
Assume k > 0
@note: Take the s-domain solution back to the $t$ domain, closing the round trip: a free term at the natural frequency $\sqrt{k/m}$ plus a forced term at the drive frequency $\omega$ (assuming $m\omega^2 \ne k$, i.e. off resonance). At $t=0$ it gives $x=1$, matching the stated conditions.
Take the inverse Laplace transform of E7 from s to t
```
