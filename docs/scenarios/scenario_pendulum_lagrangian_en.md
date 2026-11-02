# Practice-loop sample scenario 2: analytical mechanics of the simple pendulum

> (2026-07-02: an English translation of the Japanese scenario, verified against the
> real Maxima backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_pendulum_lagrangian_jp.md](scenario_pendulum_lagrangian_jp.md) for the
> Japanese version. No new i18n gaps were found here — all the English NL patterns this
> scenario needs were already fixed while building the earlier translations.)

**Physical theme:** derive the Euler-Lagrange equation from the Lagrangian, then reduce
to simple harmonic motion under the small-angle approximation
**Backend:** Maxima (standard_symbolic session)
**Domain:** analytical mechanics (analytical_mechanics)

## Physical background

The Lagrangian of a simple pendulum ($l$ = pendulum length, $q$ = angle, $m$ = mass):

$$L = \frac{1}{2}m l^2 \dot{q}^2 - mgl(1-\cos q)$$

Euler-Lagrange equation:

$$\frac{d}{dt}\frac{\partial L}{\partial \dot{q}} - \frac{\partial L}{\partial q} = 0$$

$$\Rightarrow \quad ml^2\ddot{q} + mgl\sin q = 0$$

Small-angle approximation $\sin q \approx q$:

$$\ddot{q} + \omega^2 q = 0, \qquad \omega = \sqrt{g/l}$$

## Building it from general operations

1. Declare `q` as a time-dependent function and `m, l, g` as scalars *before* entering
   any formula, so `\dot{q}` display and d/dt operations work correctly from the start.
2. Register the Lagrangian in H-1 equation form.
3. Euler-Lagrange: partial derivative w.r.t. `\dot{q}` → total derivative w.r.t. `t` →
   partial derivative w.r.t. `q` → combine into the equation of motion.
4. H-8 both-sides division (`Divide both sides of E5 by m l^2`) simplifies the EOM.
5. The small-angle approximation is a plain substitution (`sin(q) → q`).

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=analytical_mechanics
@note: Consider a simple pendulum (length $l$, angle $q$, mass $m$). Prepare $q$ as a time-dependent angle, and $m,l,g$ (gravitational acceleration) as scalar parameters.
Define q as a time-dependent function
Define m, l, g as scalars
@note: Define the pendulum's Lagrangian $L=\frac12ml^2\dot q^2-mgl(1-\cos q)$ (kinetic energy minus potential energy).
$L = (1/2) m l^2 \dot{q}^2 - m g l * (1 - cos(q))$
@note: Following the Euler-Lagrange equation $\frac{d}{dt}\frac{\partial L}{\partial\dot q}-\frac{\partial L}{\partial q}=0$, first find the generalized momentum $\partial L/\partial\dot q$.
Take the partial derivative of E1 with respect to \dot{q}
@note: Take the total time derivative of the generalized momentum just found, to obtain $\frac{d}{dt}(\partial L/\partial\dot q)$.
Take the total derivative of E2 with respect to t
@note: Find the other term of the Euler-Lagrange equation, $\partial L/\partial q$.
Take the partial derivative of E1 with respect to q
@note: Subtract $\partial L/\partial q$ from $\frac{d}{dt}(\partial L/\partial\dot q)$ to assemble the equation of motion.
EOM = E3 - E4
@note: Divide both sides by $ml^2$ to drop the leading coefficient, moving toward the simple-harmonic-motion form.
Divide both sides of E5 by m l^2
@note: Apply the small-angle approximation $\sin q\approx q$, reducing the equation to simple harmonic motion $\ddot q+\omega^2q=0$ ($\omega=\sqrt{g/l}$).
In E6, substitute q for sin(q)
```

**Verified:** 2026-07-02 (real Maxima backend, generator run, all 9 steps completed
end-to-end, same physics as the Japanese version: $EOM/(ml^2) = \ddot q + (g/l)\sin q$,
then $\ddot q + (g/l) q$ after the small-angle substitution).
