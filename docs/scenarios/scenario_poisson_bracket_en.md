# Practice-loop sample scenario: Poisson brackets (declaring canonical pairs → {f,g})

> (2026-07-02: an English translation of the Japanese scenario, verified against the
> real Maxima backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_poisson_bracket_jp.md](scenario_poisson_bracket_jp.md) for the Japanese
> version.)

## Goal

Compute the Hamiltonian-mechanics Poisson bracket

$$\{f, g\} = \sum_i\left(\frac{\partial f}{\partial q_i}\frac{\partial g}{\partial p_i} - \frac{\partial f}{\partial p_i}\frac{\partial g}{\partial q_i}\right)$$

in natural language, after **declaring the canonical coordinate-momentum pairs
$(q_i, p_i)$ once**. Verify the elementary bracket $\{r, p_r\}=1$, the conservation law
of a central-force Hamiltonian $\{H, L_z\}=0$, and Hamilton's equation of motion
$\{H, r\}=-p_r/m$.

> The Poisson bracket is the **classical analogue** of the quantum `commutator` — it
> needs no non-commutative algebra and is built purely as a **sum of partial
> derivatives**. So Maxima handles it directly (a sum of `diff`s).

## Declaring canonical pairs

The Poisson bracket needs to know which coordinate is conjugate to which momentum.
Rather than restating this every time, declare it once per session (stored in
`SymbolContext.canonical_pairs`, persisted across save/load):

```
Define p_r as the canonical momentum of r          ← canonical pair (r, p_r)
Define p_theta as the canonical momentum of theta   ← canonical pair (theta, p_theta)
```

The phrasing "define theta and p_theta as a canonical pair" (coordinate, momentum
order) also works.

## Input steps

### Step 1: the elementary Poisson bracket

```
Find the Poisson bracket of r and p_r        → 1     (canonical commutation {q,p}=1)
Find the Poisson bracket of r and p_theta    → 0     (different pairs commute)
```

### Step 2: the conservation law {H, L_z} = 0

```
H = p_r^2/(2*m) + p_theta^2/(2*m*r^2) + U(r)   → the central-force Hamiltonian (H-1)
L_z = p_theta                                  → angular momentum (= the conjugate momentum p_theta)
Find the Poisson bracket of H and L_z          → 0
```

$\{H, L_z\}=0$ shows **$L_z$ is a conserved quantity** ($H$ doesn't depend on
$\theta$ — rotational symmetry).

### Step 3: Hamilton's equation of motion

```
Find the Poisson bracket of H and r        → -p_r/m     ( \dot r = {r, H} = ∂H/∂p_r )
```

Since $\dot q = \{q, H\}$, we have $\{H, r\} = -\{r, H\} = -\partial H/\partial p_r = -p_r/m$.

## Specifying operands

`a` / `b` can be any of:

- An **E-id** (e.g. `E4`)
- A **name registered via H-1** (`H`, `L_z`)
- A **bare symbol** (`r`, `p_r`) — it doesn't need to be a registered expression

Requesting a bracket before the canonical pairs are declared returns a message asking
you to declare them.

## Coverage

- **Declaring and persisting canonical pairs** ("define ... as the canonical momentum
  of ..." / "... as a canonical pair", preserved across save/load).
- **The Poisson bracket** `poisson_bracket` (rule-based, no LLM, a sum of Maxima
  partial derivatives).
- The classical analogue of the quantum `commutator` (a different differential
  construction from doc 18's metric-preservation unification).
- Backend: Maxima.

## Observed behaviour / notes

- **The `poisson_bracket` operation and the canonical-pair declarations had no English
  rule-based trigger at all** (`ui/interpret_pipeline.py`'s `_try_poisson`) — both were
  matched by dedicated Japanese-only regexes (`_PB_PAIR_MOM_RE`, `_PB_PAIR_CANON_RE`,
  `_PB_BRACKET_RE`), not routed through the generic alias scan. Added English
  equivalents for all three ("X as the canonical momentum of Y", "X and Y as a
  canonical pair", "Poisson bracket of f and g").
- Incidentally, the schema alias for `poisson_bracket` itself (`"poisson bracket"`,
  with a space) can never match through the generic rule-based Phase-1 alias scan
  either way, for the same reason noted in the `ac_impedance` scenario: that scan
  tokenizes on whitespace and requires an exact single-token match. This didn't matter
  here since `_try_poisson` is a dedicated interceptor that runs independently of the
  generic alias scan.
- **(2026-07-08 re-audit) A canonical-pair declaration produced zero visible output in
  the exported TeX** — the confirmation message was returned in the API response but
  never recorded as a timeline step. Fixed by having the declaration call
  `engine.record_quantity_update(...)`, mirroring `declare_time_dependent`'s existing
  pattern, so the confirmation ("Registered canonical pair (r, p_r)...") now appears
  in the exported narrative between the input echo and the next step.
- **(2026-07-08 re-audit) `[X, Y]` bracket notation now also resolves to
  `poisson_bracket` on a Maxima session** (previously it only resolved to the quantum
  `commutator`, julia_full-only, and fell through to being registered as a literal
  Maxima list on Maxima). Root cause: `operands_provided()` — the hook that tells the
  "bare formula → just register it" fallback to defer to the rule-based operator
  parser instead — only recognized `[`/`{`-prefixed bracket syntax in
  `JuliaBackendHandler`. Since the check itself is not Julia-specific (it just
  recognizes that both operands are already explicit in bracket notation, regardless
  of backend), it was moved to the shared `BackendHandler` base class
  (`notebook/runtime/backend_handler.py`) per the Backend Encapsulation Rule, and
  `poisson_bracket` was added alongside `commutator` as a candidate for the same
  `[X, Y]` pattern in `llm_supervisor.py`'s Phase-0 bracket parser (the two operations
  are backend-exclusive, so there is no ambiguity). `Find the Poisson bracket of r and
  p_r` and `[r, p_r]` are now both valid, equivalent inputs; this scenario keeps using
  the natural-language form as the primary documented style, matching its Japanese
  counterpart.

**Verified:** 2026-07-02 (real Maxima backend, generator run, all 7 steps completed
end-to-end, same physics as the Japanese version: $\{r,p_r\}=1$, $\{H,L_z\}=0$,
$\{H,r\}=-p_r/m$).

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=classical_mechanics
@note: To compute a Poisson bracket, the system must first know which coordinate and momentum are canonically conjugate. Declare $p_r$ as the momentum conjugate to the radial coordinate $r$.
Define p_r as the canonical momentum of r
@note: Declare $p_\theta$ as the momentum conjugate to the angular coordinate $\theta$.
Define p_theta as the canonical momentum of theta
@note: Compute the basic Poisson bracket $\{r,p_r\}$. Since these are a canonical pair, the canonical commutation relation $\{q,p\}=1$ means this should equal 1.
Find the Poisson bracket of r and p_r
@note: Define the central-force Hamiltonian $H=p_r^2/(2m)+p_\theta^2/(2mr^2)+U(r)$ (the sum of the radial and angular kinetic energies and the potential $U(r)$).
H = p_r^2/(2*m) + p_theta^2/(2*m*r^2) + U(r)
@note: Define the angular momentum $L_z=p_\theta$ (which is simply the momentum conjugate to the angle).
L_z = p_theta
@note: Compute the Poisson bracket of the Hamiltonian $H$ and the angular momentum $L_z$. Since $H$ does not depend on the angle $\theta$ (rotational symmetry), this vanishing confirms that $L_z$ is conserved.
Find the Poisson bracket of H and L_z
@note: Using Hamilton's equation of motion $\dot q=\{q,H\}$, compute the Poisson bracket of $H$ and the radial coordinate $r$ to find its time evolution.
Find the Poisson bracket of H and r
```
