# Practice-loop sample scenario: RLC circuit transfer function (S-e)

> (2026-07-02: an English translation of the Japanese scenario, verified against the
> real Maxima backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_rlc_transfer_function_jp.md](scenario_rlc_transfer_function_jp.md) for the
> Japanese version. Rewritten 2026-08-15 onto general operations only.)

Detailed write-up of practice-loop scenario-menu item **S-e (control / circuits)**.
Move a series RLC circuit's differential equation into the s-domain and derive the
**transfer function** $H(s)$. The builtin `laplace` is not used (it would bring
opaque `%at` initial-condition terms). No dedicated S-e operation is created —
everything is built from **general operations** (registering formulas, E-id
references, solve, substitute).

**Backend:** standard (Maxima) / **Domain:** control theory (control_theory).

## Physical background

Equation of motion of a series RLC circuit (charge $q(t)$, input voltage $v(t)$):

$$L\ddot q + R\dot q + \frac{q}{C} = v.$$

Since the Laplace transform is linear, each term maps to its image. The image of a
derivative of the unknown $q$ is what **integration by parts** gives:

$$\int_0^\infty \dot q\,e^{-st}dt = \bigl[q\,e^{-st}\bigr]_0^\infty + s\int_0^\infty q\,e^{-st}dt = sQ(s) - q(0),$$

and likewise $\ddot q \to s^2Q(s)-sq(0)-\dot q(0)$. Here
$Q(s)=\mathcal L\{q\}=\int_0^\infty q(t)e^{-st}dt$, $V(s)=\mathcal L\{v\}$
(**made explicit via a definition card**). So the ODE becomes

$$L\bigl(s^2Q - s\,q_0 - u_0\bigr) + R\bigl(sQ - q_0\bigr) + \frac{Q}{C} = V,
\qquad q_0 \equiv q(0),\quad u_0 \equiv \dot q(0).$$

The **transfer function** is a property of the system itself, defined under **zero
initial conditions** $q_0=u_0=0$. Stating those, then solving the s-domain equation
(in $Q,V$) for $Q$ and dividing by $V$:

$$Ls^2Q + RsQ + \frac{Q}{C} = V \;\Longrightarrow\; Q = \frac{V}{Ls^2+Rs+1/C}
  \;\Longrightarrow\; H(s) = \frac{Q}{V} = \frac{1}{Ls^2 + Rs + 1/C}.$$

## The image of a derivative is a THEOREM, not something the system derives

This scenario was rewritten on 2026-08-15. It used to call
`laplace_by_definition` — an operation **written for this scenario** — which
transformed the whole ODE in one step. That operation has been deleted and the
researcher now writes the transform rule itself (open_items SCENARIO-MACRO,
case 7).

The reason is that integration by parts here involves **no choice**: the Laplace
derivative rule fixes $u=e^{-st}$ and $dv=f^{(n)}dt$ by the structure of the
transform, unlike a general integration by parts where splitting $u$ and $dv$ is
a decision. What involves no choice is not a computation but a **theorem**, and a
theorem is something the researcher can write down.

If it can be written, there is no reason for the system to own it. Removing the
macro changes **not one character** of this scenario's results — the test's
expected values pass unchanged.

## Steps and expected results

Each stage is its own independent YOU card. All are general operations, acting only on
symbols that genuinely exist in the s-domain ($Q,V$).

| Stage | Input (natural language / formula) | Result |
|---|---|---|
| 1 | `Define q as a time-dependent function` / `Define v as a time-dependent function` | declare $q,v$ as functions of $t$ |
| 2 | `Let Q be the Laplace transform of q` / `Let V be the Laplace transform of v` | definition cards $Q(s)=\int_0^\infty q(t)e^{-st}dt$, $V(s)=\int_0^\infty v(t)e^{-st}dt$ |
| 3 | `$s Q - q_0$` (registered as a formula) | $\mathcal L\{\dot q\}$ — the rule, written by the researcher |
| 4 | `$s^2 Q - s q_0 - u_0$` (registered as a formula) | $\mathcal L\{\ddot q\}$ |
| 5 | `$L E4 + R E3 + Q/C = V$` (registered as a formula) | $L(s^2Q - sq_0 - u_0) + R(sQ - q_0) + \dfrac{Q}{C}=V$ (**E-id references** compose the images into the equation) |
| 6 | `$q_0 = 0$` / `$u_0 = 0$` (registered as formulas) | the zero initial conditions the transfer function is defined at |
| 7 | `$H = Q/V$` (registered as a formula) | $H=Q/V$ (defines the transfer function) |
| 8 | `Solve E5 for Q` | $Q=\dfrac{CV}{CLs^2+CRs+1}$ (the conditions are already in force) |
| 9 | `Substitute E9 into E8` | $H=\dfrac{C}{CLs^2+CRs+1}=\dfrac{1}{Ls^2+Rs+1/C}$ |

## Observed behaviour / notes

- **The images are composed by E-id reference (H-2).** Stage 5's
  `L E4 + R E3 + Q/C = V` expands the registered expressions into the canonical
  before registering. Writing the rules as separate *equations* and substituting them
  later is the obvious alternative, and it does not work: **an H-1 definition is sent
  to the CAS when it is registered** (P2), so by the time the substitution step runs
  the rule has already been applied, and the step is refused with "not present in the
  target". The E-id reference avoids that double application.
- **The initial conditions take effect the same way.** Registering $q_0=0$ hands it to
  the CAS, so stage 8's solve needs no separate substitution step — and adding one
  would be refused, for the same reason as above.
- **The transfer function comes from general operations.** The s-domain equation
  contains $Q,V$ and not $q,v$. A dedicated "transfer function q/v" operation would be
  mathematically dishonest, so none exists: `$H=Q/V$` (definition) → `Solve … for Q`
  → `Substitute` are all general operations on symbols that really are there.
- **The transform definition card stays.** That $Q$ is the image of $q$ is the
  researcher's *naming*, and the CAS does not know that name — measured, `ilt(Q,s,t)`
  reads $Q$ as a constant and answers $Q\delta(t)$. The pairing is what labels an
  inverse transform's result (see the forced-oscillator scenario). Anchoring to the
  CAS's own noun form instead, so no table is needed, is open_items
  **LAPLACE-PAIR-ANCHOR**.

**Verified:** 2026-07-02 (real Maxima backend, generator run, all steps completed
end-to-end: $H(s) = 1/(Ls^2+Rs+1/C)$); re-verified 2026-08-15 after the rewrite,
same result.

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=control_theory
@note: Consider a series RLC circuit (charge $q(t)$, input voltage $v(t)$). Prepare $q,v$ as time-dependent functions.
Define q as a time-dependent function
Define v as a time-dependent function
@note: Name the Laplace transform of $q$ as $Q$ (an s-domain quantity), via a definition card.
Let Q be the Laplace transform of q
@note: Likewise name the Laplace transform of $v$ as $V$.
Let V be the Laplace transform of v
@note: Write the Laplace transform rules for the derivatives. Integration by parts here fixes $u=e^{-st}$ by the structure of the transform, so there is no choice to make: this is a theorem, and the researcher writes it. First $\mathcal L\{\dot q\} = sQ - q(0)$.
$s Q - q_0$
@note: Then $\mathcal L\{\ddot q\} = s^2 Q - s q(0) - \dot q(0)$.
$s^2 Q - s q_0 - u_0$
@note: Assemble the s-domain form of the circuit equation $L\ddot q+R\dot q+q/C=v$ (inductance $L$, resistance $R$, capacitance $C$), referring to the two rules above by their E-ids.
$L E4 + R E3 + Q/C = V$
@note: The transfer function is defined under zero initial conditions, so state $q(0)=0$ and $\dot q(0)=0$. An H-1 definition is sent to the CAS when registered, so these are in force from here on.
$q_0 = 0$
$u_0 = 0$
@note: Define the transfer function $H=Q/V$ (the s-domain ratio of output to input).
$H = Q/V$
@note: Solve the s-domain equation for $Q$. The zero initial conditions are already in force.
Solve E5 for Q
@note: Substitute the resulting expression for $Q$ into the definition of $H$ to get the closed form of the transfer function.
Substitute E9 into E8
```
