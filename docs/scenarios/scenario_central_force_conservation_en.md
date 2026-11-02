# Practice-loop sample scenario: conserved quantities of a central-force potential (angular momentum, energy)

> (2026-07-02: an English translation of the Japanese scenario, verified against the
> real Maxima backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_central_force_conservation_jp.md](scenario_central_force_conservation_jp.md)
> for the Japanese version.)

Detailed write-up of practice-loop scenario-menu item **S-h (classical mechanics)**.
For central-force motion in polar coordinates $(r,\theta)$, confirm that **angular
momentum** $L_z$ and **energy** $E$ are conserved quantities by differentiating each
with respect to time and showing it **vanishes once the equations of motion are
substituted in**.

**Backend:** standard (Maxima) / **Domain:** classical mechanics (classical_mechanics).

> While the analytical-mechanics domain ([doc 19](../developer_manuals/19_analytical_mechanics_independent_variables.md))
> treats $q,\dot q$ as independent atoms, the classical-mechanics (Newtonian) domain
> treats velocity as a **dependent quantity** $\dot q = dq/dt$, represented in Maxima's
> depends form `'diff(q,t,N)`. It is entered using math notation like `\dot{r}` (no CAS
> syntax needed).

## Physical background

A point mass $m$ moves in a plane under a central-force potential. In polar
coordinates $(r,\theta)$:

- Angular momentum $L_z = m r^2 \dot\theta$
- Energy $E = \tfrac12 m(\dot r^2 + r^2\dot\theta^2) + U(r)$ (here the Kepler potential
  $U=-k/r$)

Equations of motion (Euler-Lagrange):

$$\ddot\theta = -\frac{2\dot r\dot\theta}{r},\qquad
  \ddot r = r\dot\theta^2 - \frac{k}{m r^2}$$

For a central force, $\partial L/\partial\theta = 0$ so $L_z$ is conserved, and since the
force has no explicit time dependence, $E$ is conserved too.

## Session setup / symbol declarations

- Mode **standard (Maxima)**, domain **classical mechanics**. Velocity is entered as
  **`\dot{r}`** (math notation).

```
Define r as a time-dependent function
Define theta as a time-dependent function
Define m, k as scalars
```

## Input steps

### Step 1: conservation of angular momentum

```
L_z = m r^2 \dot{theta}                                     → E1
Take the total derivative of E1 with respect to t            → E2 = m r²θ̈ + 2 m r ṙ θ̇
In E2, substitute -2 \dot{r} \dot{theta} / r for \ddot{theta} → E3 = 0
```

Substituting $\theta$'s equation of motion $\ddot\theta = -2\dot r\dot\theta/r$ gives
$dL_z/dt = 0$. **Angular momentum is conserved.**

### Step 2: conservation of energy

```
E_tot = (1/2) m (\dot{r}^2 + r^2 \dot{theta}^2) - k / r        → E4
Take the total derivative of E4 with respect to t              → E5 = dE/dt
In E5, substitute -2 \dot{r} \dot{theta} / r for \ddot{theta}  → E6
In E6, substitute r \dot{theta}^2 - k / (m r^2) for \ddot{r}   → E7 = 0
```

Substituting both equations of motion ($\ddot\theta$, $\ddot r$) gives $dE/dt = 0$.
**Energy is conserved.**

> **Why say "total derivative" explicitly:** when differentiating a time-dependent
> expression like $L_z$ or $E$ with respect to $t$, a partial derivative
> $\partial/\partial t$ (holding everything else fixed) and a total derivative $d/dt$
> (chain rule) are different things. The system never guesses — it asks the user which
> one is meant (an interactive choice dialog). In non-interactive contexts (scripts,
> tests), one of "total derivative" / "partial derivative" must be stated explicitly.
> Here we want the chain-rule $d/dt$, so we write "total derivative".

## Coverage

- The **conserved-quantity verification workflow** (conserved quantity → $d/dt$ →
  substitute the equations of motion → 0) for two conserved quantities.
- Classical-mechanics (Newtonian) domain: velocity/acceleration are in depends form
  `'diff(q,t,N)`. `\dot{r}`/`\ddot{theta}` input → atom → unified into depends form, so
  the cancellation goes through.
- **Substituting a derivative term** (replacing `\ddot{theta}` via the equation of
  motion). Substitution of a multi-token RHS (`-2 \dot{r} \dot{theta} / r`).
- Implicit multiplication via a space before a parenthesis (`m (\dot{r}^2 + …)`,
  `k / (m r^2)`).
- Backend: Maxima, domain: classical mechanics.

## Related

- The analytical-mechanics (independent-atom model) version of Euler-Lagrange:
  [scenario_coupled_oscillations.md](scenario_coupled_oscillations_jp.md) (S-b),
  [scenario_double_pendulum_lagrangian.md](scenario_double_pendulum_lagrangian_jp.md) (S-d).
- Design: [doc 19](../developer_manuals/19_analytical_mechanics_independent_variables.md) (domain
  separation of classical vs. analytical mechanics).
- Input conventions: [user_guide.md](../user_guide.md) §数式入力の注意点 (space =
  multiplication, `\dot{}` notation).

## Observed behaviour / notes

Building this English translation surfaced two more English-language gaps in the
rule-based (no-LLM) NL interceptors, on top of the pattern already fixed for the BCS
Hamiltonian and transverse-field-Ising scenarios:

- **The total/partial-derivative disambiguation trigger recognized only the Japanese
  全微分/偏微分 literals.** `ui/interpret_pipeline.py`'s differentiate handler decides
  between `d/dt` and `∂/∂t` by searching `supervisor_text` for these two literal
  strings; there was no English equivalent, so an ambiguous request like
  "differentiate E1 with respect to t" (t already time-dependent) would either wait on
  the interactive disambiguation dialog or fail outright in non-interactive contexts.
  Fixed by also recognizing "total derivative" / "partial derivative" as plain English
  substrings, checked independently of the differentiate-trigger regex's own capture
  groups (so the fix doesn't disturb their group numbering, shared with the existing
  Japanese patterns).
- **The English differentiate trigger only recognized the "differentiate E1 with
  respect to k" phrasing**, not the equally natural "take the total derivative of E1
  with respect to t" / "the partial derivative of E1 w.r.t. t". Extended
  `_DIFF_RULE_RE_EN` to accept both forms (EID and the target variable stay in the same
  capture-group positions as the Japanese patterns either way).
- **Multi-symbol declaration only supports a comma separator in English** ("Define m,
  k as scalars"), not "and" ("Define m and k as scalars") — `_DEFINE_STMT_EN_RE`
  (`notebook/runtime/backend_handler.py`) has no "and"-splitting logic. This scenario's
  generator block uses the comma form, matching the convention already established by
  every other English scenario in this repository (`free_electron_fermi_surface`,
  `bcs_hamiltonian`, etc., which all declare multiple scalars via commas) — not treated
  as a bug requiring a fix, just documented so a comma is used consistently rather than
  "and".

**Verified:** 2026-07-02 (real Maxima backend, generator run, all 10 steps completed
end-to-end: $dL_z/dt=0$ and $dE/dt=0$, same as the Japanese version).

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=classical_mechanics
@note: Consider a point mass $m$ moving in a central-force potential in polar coordinates $(r,\theta)$. $k$ is a parameter giving the strength of the Kepler potential $U=-k/r$.
Define r as a time-dependent function
Define theta as a time-dependent function
Define m, k as scalars
@note: Define the angular momentum $L_z=mr^2\dot\theta$ (mass times the square of the radius times the angular velocity).
L_z = m r^2 \dot{theta}
@note: To check whether the angular momentum is conserved, take the total time derivative of $L_z$ (the total derivative includes the chain rule, accounting for the time dependence of both $r$ and $\theta$).
Take the total derivative of E1 with respect to t
@note: Substitute the equation of motion $\ddot\theta=-2\dot r\dot\theta/r$ into $dL_z/dt$. If the result is zero, the angular momentum is a conserved quantity.
In E2, substitute -2 \dot{r} \dot{theta} / r for \ddot{theta}
@note: Define the energy $E=\frac12 m(\dot r^2+r^2\dot\theta^2)-k/r$ (the sum of the kinetic energy and the Kepler potential).
E_tot = (1/2) m (\dot{r}^2 + r^2 \dot{theta}^2) - k / r
@note: To check whether the energy is conserved, take the total time derivative of $E$.
Take the total derivative of E4 with respect to t
@note: Substitute the equation of motion $\ddot\theta=-2\dot r\dot\theta/r$.
In E5, substitute -2 \dot{r} \dot{theta} / r for \ddot{theta}
@note: Substitute the other equation of motion $\ddot r=r\dot\theta^2-k/(mr^2)$ as well. If the result of substituting both is zero, the energy is a conserved quantity.
In E6, substitute r \dot{theta}^2 - k / (m r^2) for \ddot{r}
```
