# Practice-loop sample scenario: normal modes of coupled oscillators and the inverse normal-coordinate transform

> (2026-07-02: an English translation of the Japanese scenario, verified against the
> real Maxima backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_coupled_oscillations_jp.md](scenario_coupled_oscillations_jp.md) for the
> Japanese version.)

Detailed write-up of practice-loop scenario-menu item **S-b (analytical mechanics)**.
For two spring-coupled point masses, derive the equations of motion from the
Lagrangian, **separate them into normal modes via the sum and difference of the
equations of motion**, and further find **the inverse of the normal-coordinate
transform** (L-1).

**Backend:** standard (Maxima) / **Domain:** analytical mechanics
(analytical_mechanics).

## Physical background

Two point masses of mass $m$, connected by outer springs $k$ and a coupling spring $c$
(displacements $x,y$):

$$L = \tfrac12 m\dot x^2 + \tfrac12 m\dot y^2 - \tfrac12 k x^2 - \tfrac12 k y^2 - \tfrac12 c(x-y)^2$$

Equations of motion:

$$m\ddot x + (k{+}c)x - c y = 0,\qquad m\ddot y + (k{+}c)y - c x = 0$$

Diagonalized in the **normal coordinates** $Q_1=x+y$ (symmetric), $Q_2=x-y$
(antisymmetric):

$$m\ddot Q_1 + k Q_1 = 0\ (\omega_1^2=k/m),\qquad m\ddot Q_2 + (k{+}2c)Q_2 = 0\ (\omega_2^2=(k{+}2c)/m)$$

## Session setup / symbol declarations

- Mode **standard (Maxima)**, domain **analytical mechanics**. Velocity is entered as
  **`\dot{x}`** (math notation). In the analytical-mechanics domain, $x,\dot x,\ddot x$
  are treated as independent atoms ([doc 19](../developer_manuals/19_analytical_mechanics_independent_variables.md)).

```
Define x as a time-dependent function
Define y as a time-dependent function
Define m, k, c as scalars
```

## Input steps

### Step 1: the Lagrangian

```
Lag = (1/2)*m*\dot{x}^2 + (1/2)*m*\dot{y}^2 - (1/2)*k*x^2 - (1/2)*k*y^2 - (1/2)*c*(x - y)^2
→ E1
```

### Step 2: the equation of motion for $x$

```
Take the partial derivative of E1 with respect to \dot{x}   → E2 = m\dot{x}
Take the total derivative of E2 with respect to t             → E3 = m\ddot{x}
Take the partial derivative of E1 with respect to x          → E4 = -c(x-y) - k x
EOMx = E3 - E4                                                 → E5 = m\ddot{x} - c y + (k+c)x
```

### Step 3: the equation of motion for $y$

```
Take the partial derivative of E1 with respect to \dot{y}   → E6 = m\dot{y}
Take the total derivative of E6 with respect to t             → E7 = m\ddot{y}
Take the partial derivative of E1 with respect to y          → E8 = c(x-y) - k y
EOMy = E7 - E8                                                 → E9 = m\ddot{y} + (k+c)y - c x
```

### Step 4: normal modes (sum and difference of the equations of motion)

```
sym  = E5 + E9   → E10 = m\ddot{y} + m\ddot{x} + k y + k x     (= mQ̈₁ + kQ₁ = 0,  ω₁²=k/m)
anti = E5 - E9   → E11 = m\ddot{x} - m\ddot{y} + (k+2c)x - (k+2c)y  (= mQ̈₂ + (k+2c)Q₂ = 0,  ω₂²=(k+2c)/m)
```

**Physical cross-check:** in the sum, the coupling $c$ cancels leaving only $k$ (the
in-phase mode, lower frequency); in the difference, $k+2c$ remains (the out-of-phase
mode, higher frequency).

### Step 5: inverse of the normal-coordinate transform (L-1)

Solve the forward transform $Q_1=x_1+x_2,\ Q_2=x_1-x_2$ for the old coordinates:

```
Q_1 = x_1 + x_2
Q_2 = x_1 - x_2
Invert E12, E13 for x_1, x_2
→ x_1 = (Q_1 + Q_2)/2
→ x_2 = (Q_1 - Q_2)/2
```

This gives the inverse transform from the normal coordinates $Q$ back to the original
coordinates $x$.

## Coverage

- The **Euler-Lagrange workflow** ($\partial_{\dot q}$ → $d/dt$ → $\partial_q$ → EOM)
  for two coordinates.
- **Normal-mode separation via a linear combination (sum/difference) of the equations
  of motion**. Correct sign handling when subtracting EOMs.
- **The inverse of an L-1 linear basis transform** (normal-coordinate transform
  $Q\to x$).
- Backend: Maxima, domain: analytical mechanics (independent-atom model —
  [doc 19](../developer_manuals/19_analytical_mechanics_independent_variables.md)).

## Related

- The extended version with velocity coupling:
  [scenario_double_pendulum_lagrangian.md](scenario_double_pendulum_lagrangian_jp.md) (S-d).
- L-1 in general: [user_guide.md](../user_guide.md) §線型変換,
  [18_linear_transform_design.md](../developer_manuals/18_linear_transform_design.md).
- Design and implementation: [doc 19](../developer_manuals/19_analytical_mechanics_independent_variables.md)
  (independent-atom model), [open_items.md](../../KNOWN_ISSUES.md) §AM-1 (background).

## Observed behaviour / notes

Building this English translation surfaced a genuine root-cause bug — not just a
missing pattern, but a case where the generic English differentiate path silently
produced a **wrong result** (not an error):

- **Differentiating with respect to a `\dot{...}` velocity via the generic English
  differentiate trigger sent the raw, unconverted LaTeX to Maxima, silently returning
  the input unchanged.** `ui/interpret_pipeline.py` has a *dedicated* short-circuit
  (2e2, the wrt-q̇ keyword detector — NL grammar now externalized to
  `nl_patterns.json["qdot_diff_kw"]`, 所見G stage 3) specifically for "differentiate with respect to
  a dotted velocity" requests — it resolves the underlying *quantity name* (e.g. `x`
  from `\dot{x}`) and passes it to the backend as `wrt_qdot=<name>`, letting the
  backend construct the correctly-dotted Maxima variable (`x_dot`) itself. This is
  necessary because the *generic* differentiate short-circuit (2e3, used for `E1 を x
  で偏微分して` / "differentiate E1 with respect to x") passes the captured variable
  text through mostly as-is, which is fine for a plain symbol like `x` but wrong for
  `\dot{x}` (Maxima doesn't understand raw LaTeX). The wrt-q̇ detector existed only in
  Japanese, so an English request like "take the partial derivative of E1 with respect
  to \dot{x}" fell through to the generic (2e3) English pattern I had added for the
  BCS/Ising scenarios, which sent `diff(..., \dot{x})` to Maxima verbatim — Maxima
  silently treated the malformed variable as if no differentiation variable were given
  and returned the input **completely unchanged**, with `status: completed` and no
  error. This is exactly the "wrong answer, not a crash" failure mode the project's
  scenario-driven bug hunting is meant to catch. Fixed by adding English patterns
  ("differentiate"/"derivative" co-occurring with a `\dot{...}` token) to
  the detector, and an English target-expression pattern ("of E<i>n</i>" / "E<i>n</i>
  with respect to") alongside the existing Japanese "En を", since without it the
  wrt_qdot short-circuit silently targeted the *most recently registered* expression
  instead of the intended one (harmless when there's only one candidate expression,
  but wrong — and undetected by a shallow test — as soon as a second expression
  exists, exactly as ST-3 revealed: differentiating $Lag$ again picked up $E5$, the
  most recent expression, instead of $E1$).

**Verified:** 2026-07-02 (real Maxima backend, generator run, all 17 steps completed
end-to-end with the same physics as the Japanese version, including the L-1 inverse
transform).

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=analytical_mechanics
@note: Consider two point masses of mass $m$ connected by two outer springs (constant $k$) and one coupling spring (constant $c$). Let $x,y$ be the displacements of the two masses.
Define x as a time-dependent function
Define y as a time-dependent function
Define m, k, c as scalars
@note: Define the Lagrangian $L=$ kinetic energy $-$ potential energy. The potential terms are the outer-spring energies $\frac12kx^2,\ \frac12ky^2$ for each mass, plus the coupling-spring energy $\frac12c(x-y)^2$, proportional to the square of the displacement difference $(x-y)$.
Lag = (1/2)*m*\dot{x}^2 + (1/2)*m*\dot{y}^2 - (1/2)*k*x^2 - (1/2)*k*y^2 - (1/2)*c*(x - y)^2
@note: Following the Euler-Lagrange equation $\frac{d}{dt}\frac{\partial L}{\partial\dot x}-\frac{\partial L}{\partial x}=0$, derive the equation of motion for $x$: differentiate $L$ with respect to $\dot x$, take the total time derivative of that result, differentiate $L$ with respect to $x$, then take the difference of the two.
Take the partial derivative of E1 with respect to \dot{x}
Take the total derivative of E2 with respect to t
Take the partial derivative of E1 with respect to x
EOMx = E3 - E4
@note: Derive the equation of motion for $y$ by the same procedure.
Take the partial derivative of E1 with respect to \dot{y}
Take the total derivative of E6 with respect to t
Take the partial derivative of E1 with respect to y
EOMy = E7 - E8
@note: Taking the sum and difference of the two equations of motion decouples them into normal modes. The sum cancels the coupling constant $c$, giving the symmetric mode (lower frequency, $\omega_1^2=k/m$); the difference brings in $k+2c$, giving the antisymmetric mode (higher frequency, $\omega_2^2=(k+2c)/m$).
sym = E5 + E9
anti = E5 - E9
@note: Define the normal-coordinate transformation $Q_1=x+y$ (the symmetric-mode coordinate) and $Q_2=x-y$ (the antisymmetric-mode coordinate).
Q_1 = x_1 + x_2
Q_2 = x_1 - x_2
@note: Invert this transformation to express the original displacements $x,y$ (written as $x_1,x_2$ here) in terms of the normal coordinates $Q_1,Q_2$.
Invert E12, E13 for x_1, x_2
```
