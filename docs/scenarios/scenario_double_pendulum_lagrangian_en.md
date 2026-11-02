# Practice-loop sample scenario: analytical mechanics of the double pendulum (Euler-Lagrange)

> (2026-07-02: an English translation of the Japanese scenario, verified against the
> real Maxima backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_double_pendulum_lagrangian_jp.md](scenario_double_pendulum_lagrangian_jp.md)
> for the Japanese version. No new i18n gaps were found here — all the English NL
> patterns this scenario needs were already fixed while building the earlier
> translations.)

Detailed write-up of practice-loop scenario-menu item **S-d (analytical mechanics)**.
Derive the **Euler-Lagrange equation** from the double pendulum's Lagrangian. An
extension of the simple pendulum
([scenario_pendulum_lagrangian.md](scenario_pendulum_lagrangian_jp.md)) to 2 degrees of
freedom **with velocity coupling**, tracing $\partial L/\partial\dot q$ →
$\frac{d}{dt}$ → $\partial L/\partial q$ → equation of motion through Maxima.

**Backend:** standard (Maxima) / **Domain:** analytical mechanics
(analytical_mechanics).

## Physical background

The Lagrangian of a double pendulum (masses $m_1,m_2$, lengths $l_1,l_2$, angles
$\theta_1,\theta_2$):

$$L = \tfrac12(m_1{+}m_2)l_1^2\dot\theta_1^2 + \tfrac12 m_2 l_2^2\dot\theta_2^2
      + m_2 l_1 l_2\,\dot\theta_1\dot\theta_2\cos(\theta_1{-}\theta_2)
      + (m_1{+}m_2)g l_1\cos\theta_1 + m_2 g l_2\cos\theta_2$$

The Euler-Lagrange equation for $\theta_1$ (standard form):

$$(m_1{+}m_2)l_1^2\ddot\theta_1 + m_2 l_1 l_2\ddot\theta_2\cos(\theta_1{-}\theta_2)
  + m_2 l_1 l_2\dot\theta_2^2\sin(\theta_1{-}\theta_2) + (m_1{+}m_2)g l_1\sin\theta_1 = 0$$

We derive this.

## Session setup / symbol declarations

- Mode **standard (Maxima)**, domain **analytical mechanics**.
- Angles are time-dependent, coefficients are scalars — **declared before any
  formula**.
- Velocity is entered as **`\dot{a}`** (math notation). In the analytical-mechanics
  domain, $a,\dot a,\ddot a$ are treated as independent atoms, with no need to expose
  CAS syntax (`diff(a,t)`) ([doc 19](../developer_manuals/19_analytical_mechanics_independent_variables.md)).

```
Define a as a time-dependent function        # θ₁
Define b as a time-dependent function        # θ₂
Define M, m, p, q, g as scalars              # m₁,m₂,l₁,l₂,g
```

## Input steps

### Step 1: register the Lagrangian

```
Lag = (1/2)*(M + m)*p^2*\dot{a}^2 + (1/2)*m*q^2*\dot{b}^2 + m*p*q*\dot{a}*\dot{b}*\cos(a - b) + (M + m)*g*p*\cos(a) + m*g*q*\cos(b)
→ E1
```

### Step 2: generalized momentum $\partial L/\partial\dot\theta_1$

```
Take the partial derivative of E1 with respect to \dot{a}
→ E2 = m p q \dot{b}\cos(b-a) + (m+M)p^2\dot{a}
```

The velocity-coupling term $\dot\theta_1\dot\theta_2\cos$ produces the $\dot b\cos(b-a)$
term.

### Step 3: $\frac{d}{dt}\!\left(\partial L/\partial\dot\theta_1\right)$

```
Take the total derivative of E2 with respect to t
→ E3 = -m p q \dot{b}(\dot{b}-\dot{a})\sin(b-a) + m p q \ddot{b}\cos(b-a) + (m+M)p^2\ddot{a}
```

### Step 4: $\partial L/\partial\theta_1$

```
Take the partial derivative of E1 with respect to a
→ E4 = m p q \dot{a}\dot{b}\sin(b-a) - g(m+M)p\sin a
```

> The terms involving velocities $\dot a,\dot b$ correctly survive. In the
> independent-atom model, $\partial L/\partial a$ is a straightforward partial
> derivative with respect to $a$ ($\dot a,\dot b$ are independent, so they're held
> fixed).

### Step 5: the equation of motion $\frac{d}{dt}\partial_{\dot\theta_1}L - \partial_{\theta_1}L$

```
EOMa = E3 - E4
→ E5 = -m p q \dot{b}^2\sin(b-a) + m p q \ddot{b}\cos(b-a) + (m+M)p^2\ddot{a} + (m+M)g p\sin a
```

**Physical cross-check:** the $\dot a\dot b\sin(b-a)$ terms cancel between E3 and E4, and
since $\sin(b-a)=-\sin(a-b)$, $-mpq\dot b^2\sin(b-a)=+m_2l_1l_2\dot\theta_2^2\sin(\theta_1{-}\theta_2)$.
Matches the standard form in the physical background.

### Step 6 (optional): the equation of motion for $\theta_2$

Similarly, `\dot{b}` → derivative w.r.t. t → partial derivative w.r.t. `b` →
`EOMb = … - …` gives the EL equation for $\theta_2$ (the symmetric form with the sign
of the $\dot\theta_1^2\sin$ term flipped).

## Coverage

- **Partial derivative with respect to a velocity** $\partial L/\partial\dot\theta$ (a
  straightforward derivative with respect to the independent atom $\dot\theta$).
- **Total derivative with respect to time** $\frac{d}{dt}$ (the chain-rule sum over
  $\ddot\theta,\dot\theta\,$ products).
- **Partial derivative with respect to position** $\partial L/\partial\theta$
  (velocity-coupling terms are correctly preserved).
- **Assembling the equation of motion** $E3-E4$ (correct sign handling with negative
  terms in the subtraction).
- Backend: Maxima, domain: analytical mechanics (independent-atom model —
  [doc 19](../developer_manuals/19_analytical_mechanics_independent_variables.md)).

## Related

- The 1-degree-of-freedom version:
  [scenario_pendulum_lagrangian.md](scenario_pendulum_lagrangian_jp.md).
- Coupled oscillations (normal modes, inverse of the normal-coordinate transform):
  [scenario_coupled_oscillations.md](scenario_coupled_oscillations_jp.md) (S-b).
- Design and implementation: [doc 19](../developer_manuals/19_analytical_mechanics_independent_variables.md)
  (independent-atom model), [open_items.md](../../KNOWN_ISSUES.md) §AM-1 (background).

**Verified:** 2026-07-02 (real Maxima backend, generator run, all 8 steps completed
end-to-end, same result as the Japanese version).

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=analytical_mechanics
@note: Consider a double pendulum. To avoid subscripted symbols, the angles $\theta_1,\theta_2$ are written $a,b$, the masses $m_1,m_2$ as $M,m$, the pendulum lengths $l_1,l_2$ as $p,q$, and gravitational acceleration as $g$. $a,b$ are time-dependent angles.
Define a as a time-dependent function
Define b as a time-dependent function
Define M, m, p, q, g as scalars
@note: Define the double-pendulum Lagrangian $L=$ kinetic energy $+$ potential energy. The kinetic-energy term includes the rotational energy of each pendulum plus a velocity-coupling term $\dot a\dot b\cos(a-b)$, proportional to the product of the two angular velocities.
Lag = (1/2)*(M + m)*p^2*\dot{a}^2 + (1/2)*m*q^2*\dot{b}^2 + m*p*q*\dot{a}*\dot{b}*\cos(a - b) + (M + m)*g*p*\cos(a) + m*g*q*\cos(b)
@note: To obtain the generalized momentum $\partial L/\partial\dot\theta_1$, differentiate $L$ with respect to $\dot a$ ($\dot\theta_1$). The velocity-coupling term brings in a term involving $\dot b$ ($\dot\theta_2$).
Take the partial derivative of E1 with respect to \dot{a}
@note: Take the total time derivative of the generalized momentum just found, to obtain $\frac{d}{dt}(\partial L/\partial\dot\theta_1)$.
Take the total derivative of E2 with respect to t
@note: For the other term of the Euler-Lagrange equation, $\partial L/\partial\theta_1$, differentiate $L$ with respect to $a$ ($\theta_1$).
Take the partial derivative of E1 with respect to a
@note: Subtract $\partial L/\partial\theta_1$ from $\frac{d}{dt}(\partial L/\partial\dot\theta_1)$ to assemble the equation of motion for $\theta_1$.
EOMa = E3 - E4
```
