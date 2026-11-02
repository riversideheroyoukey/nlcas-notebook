# Practice-loop sample scenario: confirming the Coulomb gauge (electromagnetism)

> (2026-07-03: an English translation of the Japanese scenario, verified against the
> real Maxima backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_coulomb_gauge_jp.md](scenario_coulomb_gauge_jp.md) for the Japanese version.
> No new i18n gaps were found — all the English NL patterns this scenario needs
> (combined "time-dependent scalar/vector field" declaration, `\nabla \cdot`/`\nabla
> \times`, "take the partial derivative of ... with respect to ...", "set the
> right-hand side of En to X", "solve En for X", "simplify the right-hand side of En")
> were already fixed while building the earlier translations.)

> (Split off from the former "E-3 research-workflow verification scenario" into its own
> **1 scenario = 1 session** unit. 2026-07-01: the old version, which exposed CAS syntax
> (`diff(phi,x)` etc.), was rebuilt using only general operations — scalar/vector field
> declaration, partial differentiation, the `\nabla` operator, equation right-hand-side
> operations, and solving.)

**Backend:** Maxima / **Domain:** electromagnetism.

### Physical background

In electromagnetic gauge transformations, the Coulomb gauge is the gauge condition
$\nabla \cdot \mathbf{A} = 0$. Under this gauge the scalar potential satisfies a Poisson
equation, and the electrostatic field propagates instantaneously (non-retarded).

$$\nabla \cdot \mathbf{A} = 0 \quad \text{(gauge condition)}$$

$$\nabla^2 \phi = -\frac{\rho}{\varepsilon_0} \quad \text{(Poisson equation)}$$

$$\mathbf{E} = -\nabla\phi - \frac{\partial \mathbf{A}}{\partial t}, \quad \mathbf{B} = \nabla \times \mathbf{A}$$

Physics to verify:
1. $\nabla \cdot \mathbf{B} = 0$ (the divergence of B is zero: no magnetic charge)
2. Under the Coulomb gauge condition, confirm $\nabla \cdot \mathbf{E} = -\nabla^2\phi$
   (the Poisson equation)

---

### How this is built from general operations (★ the heart of this scenario)

Rather than entering CAS syntax such as `diff(phi,x)` directly, $\mathbf E,\mathbf B$ are
built up through the following chain of general operations:

1. **Declare $x,y,z,t$ dependence in one shot with "define as a time-dependent scalar
   field."** If the scalar-field declaration ($x,y,z$) and the time-dependence
   declaration ($t$) are issued separately, Maxima's `depends()` **overwrites** rather
   than accumulates a re-declaration of the same symbol, so the later call erases the
   earlier spatial dependence. This combined phrase internally merges both sets of
   variables into a single `depends()` call (see "Observed behaviour / notes" below).
2. **A vector field is computed simply by entering `\nabla \cdot` / `\nabla \times` as
   an expression.** Declaring $\mathbf A$ with `A as a vector field` and adding
   time-dependence to the components `A_x`,`A_y`,`A_z` is enough to get both the gauge
   condition $\nabla\cdot\mathbf A$ and the magnetic field $\mathbf B=\nabla\times\mathbf
   A$ in **one step each** (the same convention as
   [scenario_maxwell_consistency.md](scenario_maxwell_consistency_jp.md)).
3. Since $\partial/\partial t$ is **not a vector operator**, only the assembly of
   $\mathbf E$ requires differentiating component-by-component and combining via E-id
   references (e.g. `E_x = -E5 - E8`).
4. The gauge condition is registered **as an equation → solved for the target partial
   derivative via H-9** (a general solve operation), not via "assume." Maxima's
   `assume()` is for inequality/sign reasoning and does not act as a substitution rule
   for equalities.

---

### Steps

#### EM-1: start the session

| Item | Content |
|---|---|
| Action | New session, domain: "electromagnetism", backend: Maxima |
| Result | |

---

#### EM-2: declare the potentials and register as E-ids

| Item | Content |
|---|---|
| Input | `phi as a time-dependent scalar field` (depends on all of $x,y,z,t$) |
| Input | `A as a vector field` |
| Input | `A_x`, `A_y`, `A_z` `as a time-dependent function` (add $t$-dependence to the components) |
| Input | `phi0 = phi` / `Ax0 = A_x` / `Ay0 = A_y` / `Az0 = A_z` (register as differentiable E-ids) |
| Check | Maxima recognizes each symbol as depending on all of $x,y,z,t$ |
| Result | |

---

#### EM-3: assemble the electric and magnetic fields

**$\mathbf E$ via component-wise partial derivatives + E-id combination, $\mathbf B$ via
a single `\nabla \times A`:**

| Item | Content |
|---|---|
| Input | `Take the partial derivative of E1 with respect to x` / `y` / `z` ($\partial\phi/\partial x,y,z$) |
| Input | `Take the partial derivative of E2 with respect to t` / `E3` / `E4` ($\partial A_i/\partial t$) |
| Input | `E_x = -E5 - E8` / `E_y = -E6 - E9` / `E_z = -E7 - E10` |
| Input | `\nabla \times A` (get all 3 components of $\mathbf B$ at once) |
| Input | `B as a vector field` |
| Input | `B_x = E14[1]` / `B_y = E14[2]` / `B_z = E14[3]` (bind the curl result component-wise) |
| Check | KaTeX correctly displays $E_x=-\partial\phi/\partial x-\dot A_x$, $B_x=\partial A_z/\partial y-\partial A_y/\partial z$, etc. |
| Result | |

---

#### EM-4: confirm $\nabla \cdot \mathbf{B} = 0$

| Item | Content |
|---|---|
| Input | `\nabla \cdot B` |
| Expected result | 0 (`nb_div` differentiates and combines all 3 components together, so the mixed partials automatically cancel) |
| Physical meaning | an identity guaranteed automatically by $\mathbf{B} = \nabla \times \mathbf{A}$ |
| Result | |

---

#### EM-5: the Coulomb gauge condition and the Poisson equation

**The gauge condition is a single `\nabla \cdot A`, registered as an equation and solved
via H-9:**

| Item | Content |
|---|---|
| Input | `\nabla \cdot A` (get $\nabla\cdot\mathbf A$ in one step) |
| Input | `gauge = E19` (name it, register as an E-id) |
| Input | `Set the right-hand side of E20 to 0` (fix the gauge condition $\nabla\cdot\mathbf A=0$ as an equation) |
| Input | `Take the partial derivative of Az0 with respect to z` (prepare $\partial A_z/\partial z$ separately, for solving later) |
| Input | `Az_z = E22` (name it) |
| Input | `Solve E21 for Az_z` → $\partial A_z/\partial z = -\partial A_x/\partial x-\partial A_y/\partial y$ |
| Result | |

**Assemble $\nabla\cdot\mathbf E$ into Poisson form:**

| Item | Content |
|---|---|
| Input | `Take the partial derivative of E_x with respect to x` / `E_y` / `E_z` |
| Input | `divE = E25 + E26 + E27` |
| Input | `Simplify the right-hand side of E28` |
| Expected result | $\nabla\cdot\mathbf E=-\nabla^2\phi-\dfrac{\partial}{\partial t}\!\left(\dfrac{\partial A_x}{\partial x}+\dfrac{\partial A_y}{\partial y}+\dfrac{\partial A_z}{\partial z}\right)$ (in addition to $-\nabla^2\phi$, a time derivative of the gauge term remains) |
| Input | `Take the partial derivative of gauge with respect to t` ($\partial(\nabla\cdot\mathbf A)/\partial t$) |
| Input | `Set the right-hand side of E30 to 0` (the gauge condition is identically 0, so its time derivative is also 0) |
| Input | `Substitute E31 into E29` (substitute the gauge condition's time derivative = 0 into the expanded $\nabla\cdot\mathbf E$) |
| Expected | $\nabla\cdot\mathbf E=-\nabla^2\phi$ (the Poisson equation) |
| Result | |

> **(History) the limitation filed as H-10 was fixed on 2026-07-27:** the final
> substitution above used to error (`IndexedSymbol index must resolve to a Symbol
> or IndexSymbol, got Pow`), so it was not executed on the CAS and the conclusion
> was stated by visual inspection instead. Three defects stacked: (1) Maxima's own
> derivative display form $\frac{d^2}{d t\,d z}A_z$ had no inverse back to
> canonical, (2) the substitution rule handed that unconverted LaTeX straight to
> the CAS (`\frac{…}` reached the resolver and raised), and (3) even with a correct
> canonical LHS, Maxima's `subst` is purely syntactic and does not match a SUM-shaped
> LHS — it returned the target **unchanged** (`ratsubst` is the right tool). All
> three are fixed; the substitution now runs on the CAS and returns $-\nabla^2\phi$.
> See `tests/test_derivative_substitution.py`.

---

#### EM-6: LaTeX export

| Item | Content |
|---|---|
| Action | the "⬇ LaTeX" button |
| Check | partial derivatives are output in `\frac{\partial}{\partial x}` form |
| Check | the output compiles with `pdflatex` |
| Result | |

---

### Verification checklist (scenario 3)

- [x] declaring `phi`, `A` (time-dependent scalar/vector fields) works in natural language
- [x] assembling $\mathbf B$ / the gauge condition via `\nabla \cdot` / `\nabla \times` works without CAS syntax
- [x] $\nabla \cdot \mathbf{B} = 0$ reduces to 0 via a single `\nabla \cdot B`
- [x] the Coulomb gauge condition is registered as an equation and solved for the target partial derivative via H-9
- [x] the Poisson-equation consequence can be confirmed (the mechanical substitution runs on the CAS; H-10 fixed 2026-07-27)
- [ ] the LaTeX output compiles

---

## Observed behaviour / notes

- **A two-stage `depends()` declaration overwrites (a Maxima behaviour).** Declaring
  "define as a scalar field" ($x,y,z$) and then separately "define as a time-dependent
  function" ($t$) causes the latter's `depends([name],[t])` to **erase** the former's
  $x,y,z$ dependence (Maxima's `depends()` does not merge a re-declaration of the same
  variable, it overwrites it). This was a real bug discovered while rebuilding this
  scenario; `declare_time_dependent()` in `notebook/engine.py` was fixed to read the
  existing `natural_vars`, take the **union** with the time parameter, and send a single
  `depends()` call (2026-07-01).
- **A new combined phrase, "define as a time-dependent scalar/vector field," was
  introduced.** `_infer_type_from_label` (`backend_handler.py`) prioritizes a field
  declaration over the time-dependence keyword when the label contains a "scalar field"
  keyword, and `_preprocess_args` (`engine.py`) folds the time parameter into
  `natural_vars` before sending a single `depends()` call. This lets a single step
  declare dependence on all of $x,y,z,t$.
- **Re-declaring a vector-field component (e.g. `A_x`) was incorrectly triggering the
  disambiguation dialog.** A single-letter subscript (e.g. `xi_k`) is designed to show a
  confirmation dialog asking "is this an index or part of the name?", but the same
  dialog was firing when `define_vector_field`'s **already-registered** component `A_x`
  was re-declared via "define as a time-dependent function," which fails in the
  non-interactive generator environment. Fixed by making `extra_pre_phase1` in
  `backend_handler.py` skip the ambiguity check for symbols already registered in
  `SymbolContext` (2026-07-01).
- **H-9's "solve for" needed the same alias-collision guard.** The "differentiate with
  respect to X" pattern already guarded against X being misread as an already-registered
  alias symbol (e.g. another expression's LHS name) and wrongly substituted with an
  E-id, but `solve_equation`'s "solve for X" lacked the same guard, so "solve for Az_z"
  was misfiring as "solve for E23" (an accidental E-id substitution). Fixed by adding
  `について解い` to the symbol→E-id substitution guard regex in
  `ui/interpret_pipeline.py` (2026-07-01).
- **`substitute` used to fail on expressions containing derivatives (H-10, found at
  the final step of EM-5 above; fixed 2026-07-27).** Three defects stacked: (1)
  Maxima's own derivative display form $\frac{d^2}{d t\,d z}A_z$ had no inverse
  back to canonical, (2) the substitution rule handed that unconverted LaTeX to the
  CAS, and (3) Maxima's `subst` is syntactic and does not match a SUM-shaped LHS
  (`ratsubst` is the right tool). The EM-5 substitution now runs on the CAS.
- **A nested `\nabla` expression such as `\nabla \cdot (\nabla \times A)` cannot be
  entered as a single expression.** `B` must be explicitly registered as a vector field,
  with the curl result bound component-wise, before calling `\nabla \cdot B` (exactly the
  procedure in EM-3–EM-4 of this scenario).

**Verified:** 2026-07-03 (real Maxima backend, generator run, all 37 steps completed
end-to-end, same structure as the Japanese version).

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=electromagnetism
@note: Declare the scalar potential $\phi$ (depending on space $x,y,z$ and time $t$) and the vector potential $\mathbf A$ (components $A_x,A_y,A_z$, also time-dependent). The electric field $\mathbf E$ and magnetic field $\mathbf B$ are built from these two.
phi as a time-dependent scalar field
A as a vector field
Define A_x as a time-dependent function
Define A_y as a time-dependent function
Define A_z as a time-dependent function
@note: For use later in derivative calculations, give the values of $\phi,A_x,A_y,A_z$ the aliases phi0, Ax0, Ay0, Az0 and register them as expressions.
phi0 = phi
Ax0 = A_x
Ay0 = A_y
Az0 = A_z
@note: For the first term $\nabla\phi$ of the electric-field definition $\mathbf E=-\nabla\phi-\partial\mathbf A/\partial t$, take the spatial partial derivatives $\partial\phi/\partial x,\partial\phi/\partial y,\partial\phi/\partial z$.
Take the partial derivative of E1 with respect to x
Take the partial derivative of E1 with respect to y
Take the partial derivative of E1 with respect to z
@note: For the second term $\partial\mathbf A/\partial t$ of the electric-field definition, take the time derivative of each component of $\mathbf A$: $\partial A_x/\partial t,\partial A_y/\partial t,\partial A_z/\partial t$.
Take the partial derivative of E2 with respect to t
Take the partial derivative of E3 with respect to t
Take the partial derivative of E4 with respect to t
@note: Combine the derivatives just computed to assemble each component of the electric field $\mathbf E=-\nabla\phi-\partial\mathbf A/\partial t$.
E_x = -E5 - E8
E_y = -E6 - E9
E_z = -E7 - E10
@note: Obtain the magnetic field $\mathbf B=\nabla\times\mathbf A$ in a single step by applying the $\nabla\times$ operator directly to $\mathbf A$ (all three components are returned together).
\nabla \times A
B as a vector field
@note: Bind the three components of the curl result to the $x,y,z$ components of the vector field $\mathbf B$.
B_x = E14[1]
B_y = E14[2]
B_z = E14[3]
@note: Check the identity $\nabla\cdot\mathbf B=0$ (no magnetic charge), which must always hold since $\mathbf B=\nabla\times\mathbf A$, by computing the divergence of $\mathbf B$.
\nabla \cdot B
@note: To check the Coulomb-gauge condition $\nabla\cdot\mathbf A=0$, first compute the divergence of $\mathbf A$.
\nabla \cdot A
@note: Give this divergence the name gauge.
gauge = E19
@note: The Coulomb gauge imposes the condition $\nabla\cdot\mathbf A=0$. Fix this as an equation by setting the value of gauge to 0.
Set the right-hand side of E20 to 0
@note: To solve the gauge condition later, prepare the spatial derivative $\partial A_z/\partial z$ separately and give it a name.
Take the partial derivative of Az0 with respect to z
Az_z = E22
@note: Solve the gauge-condition equation for $\partial A_z/\partial z$, expressing it in terms of $\partial A_x/\partial x$ and $\partial A_y/\partial y$.
Solve E21 for Az_z
@note: To assemble the divergence $\nabla\cdot\mathbf E$, take the partial derivative of each component of $\mathbf E$ with respect to the corresponding coordinate.
Take the partial derivative of E11 with respect to x
Take the partial derivative of E12 with respect to y
Take the partial derivative of E13 with respect to z
@note: Assemble the divergence of the electric field $\nabla\cdot\mathbf E$ as the sum of the three partial derivatives above.
divE = E25 + E26 + E27
@note: Simplify the assembled expression for $\nabla\cdot\mathbf E$.
Simplify the right-hand side of E28
@note: Compute the time derivative of the gauge condition $\nabla\cdot\mathbf A=0$, to confirm that it must be identically zero.
Take the partial derivative of gauge with respect to t
@note: Fix this time derivative's value as an equation set to 0.
Set the right-hand side of E30 to 0
@note: Substituting the gauge condition's time derivative = 0 into the expanded $\nabla\cdot\mathbf E$ cancels the corresponding three terms, giving $\nabla\cdot\mathbf E=-\nabla^2\phi$ (the Poisson equation).
Substitute E31 into E29
```
