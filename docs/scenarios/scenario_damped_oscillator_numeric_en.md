# Practice-loop sample scenario: damped oscillator — ODE general solution → explicit assume → determined by initial conditions → numeric-layer fork (S-x)

> (2026-07-09: an English translation of the Japanese scenario, verified against the
> real Maxima backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_damped_oscillator_numeric_jp.md](scenario_damped_oscillator_numeric_jp.md) for
> the Japanese version.)

**Physical theme:** find the general solution of the damped-oscillator equation
$\ddot x+2\gamma\dot x+\omega_0^2x=0$, fix its arbitrary constants from initial
conditions, then fork the theory-layer expression as-is into the numeric layer (N-1) to
plot the decaying waveform. The first end-to-end scenario to confirm the theory ↔
numeric **round trip** with real data.
**Backend:** standard (Maxima)
**Time required:** about 10 minutes (theory layer) + numeric-layer fork (API/script
operations)
**Verification goal:**
1. A problem-dependent physical choice such as the sign of the discriminant must
   **never be hardcoded in shared code** — it must be recorded in the session as an
   explicit step that defines the discriminant and then `assume()`s its sign (see
   CLAUDE.md (reference omitted from the public snapshot)).
2. The full pipeline: general solution → fix arbitrary constants from initial
   conditions → extract the right-hand side → fork to the numeric layer → sandbox
   execution → provenance-carrying reflux.

---

## Physical background

Substituting the standard exponential ansatz $x=e^{rt}$ into the damped oscillator's
equation of motion $\ddot x+2\gamma\dot x+\omega_0^2 x=0$ ($\gamma$: damping rate,
$\omega_0$: natural angular frequency, both positive reals) gives the characteristic
equation $r^2+2\gamma r+\omega_0^2=0$. The sign of its discriminant
$(2\gamma)^2-4\omega_0^2$ determines the character of the solution (underdamped —
oscillating while decaying — / overdamped / critically damped). Here we explicitly
define the **sign-flipped quantity** $D=4\omega_0^2-(2\gamma)^2$ and assume $D>0$
(underdamped, the characteristic equation has complex roots). **This sign is a
physical choice that depends on the problem being treated, and the CAS-command-
generation code must never decide it on its own** (see
[feedback_no_hardcoded_modeling_choices]).

The general solution has the form
$x(t)=e^{-\gamma t}\left(k_1\sin(\omega t)+k_2\cos(\omega t)\right)$
($\omega=\sqrt{\omega_0^2-\gamma^2}=\sqrt{D}/2$) with arbitrary constants $k_1,k_2$
(FREE-PARAM). After fixing $k_1,k_2$ from the initial conditions
$x(0)=1,\ \dot x(0)=0$ to obtain a particular solution, its right-hand side is forked
into the numeric layer, sweeping $t$ to produce a plot of the decaying waveform.

---

## Session setup

- Mode: **standard (Maxima)**
- Domain: **differential_equations** (mechanics / differential equations)
- Symbol input: uses **TeX notation** (`\gamma`, `\omega_0`) — matching the natural
  input form when going through the handwriting-input layer (typing the bare ASCII
  spellings `gamma`/`omega_0` directly would be an unnatural input that bypasses this
  notation)
- Verification harness: `./scripts/debug_repl.sh --no-llm --lang en` (the theory layer
  is reproducible via rule-based dispatch)
- Numeric layer (fork onward): as of 2026-07-06 **neither an NL trigger nor a UI panel
  is implemented** — as intended by the CLAUDE.md "Numeric Layer" section, this is
  operated via API / engine methods (see "Numeric-layer steps" below)

---

## Input steps (theory layer — operable directly via NL / UI)

### Steps 1–3: prepare the variables

First, prepare the time-dependent displacement $x$ and the damping rate $\gamma$ /
natural angular frequency $\omega_0$ (both positive reals).

```
Define x as a time-dependent function
Define \gamma as a positive real number
Define \omega_0 as a positive real number
```

**Expected output:** `\gamma` and `\omega_0` are each confirmed with "registered as
positive real". At this point **`Quantity.sign="positive"` is retained and
`assume(gamma > 0)` / `assume(omega_0 > 0)` are automatically sent to Maxima**
(recorded on the timeline as ASM1/ASM2 — see the domain/sign-split implementation,
[project_domain_sign_split]). The internal identifiers become the ASCII-ized
`gamma`/`omega_0`, but display and subsequent math input can keep using
`\gamma`/`\omega_0` (TeX).

### Step 4: register the equation of motion

Consider the damped oscillator's equation of motion below ($\gamma$ is the damping
rate, $\omega_0$ is the natural angular frequency).

```math
\ddot{x} + 2 \gamma \dot{x} + \omega_0^2 x = 0
```

**Expected output:** `E1`: `\ddot{x} + 2 \gamma \dot{x} + \omega_0^2 x = 0`

### Step 5: register the characteristic equation (where the discriminant comes from)

Substituting the exponential ansatz $x=e^{rt}$ into this differential equation
($\dot x=re^{rt},\ \ddot x=r^2e^{rt}$, then dividing both sides by $e^{rt}\neq0$)
gives the characteristic equation of this differential equation below.

```math
r^2 + 2 \gamma r + \omega_0^2 = 0
```

**Expected output:** `E2`: `r^2 + 2 \gamma r + \omega_0^2 = 0`

**Check point:** keeping this as an actually-registered expression (`E2`), rather than
just explanatory prose, means the origin of the discriminant condition used below can
be traced within the scenario's own execution log.

### Step 6: define the discriminant $D$

Since the characteristic equation is a quadratic in $r$ ($ar^2+br+c=0$ with
$a=1,\ b=2\gamma,\ c=\omega_0^2$), its discriminant is
$b^2-4ac=(2\gamma)^2-4\omega_0^2$. To phrase the discussion below in the natural form
"positive means underdamped", we define the sign-flipped quantity
$D=4\omega_0^2-(2\gamma)^2=-(b^2-4ac)$.

```math
D = 4 \omega_0^2 - (2 \gamma)^2
```

**Expected output:** `E3`: `D = 4 \omega_0^2 - (2 \gamma)^2`

**Check point:** via H-1 (equation-form input), `D` is registered as a symbol and also
bound on the Maxima side as `D : 4*omega_0^2-(2*gamma)^2`. $D>0 \Leftrightarrow$
discriminant $b^2-4ac<0$ (complex roots = oscillatory solution = underdamped).

### Step 7: assume underdamped motion by explicitly stating the sign of $D$ (the researcher's choice)

Here we assume underdamped (oscillatory) motion, $D>0$. This choice is a physical
judgment that depends on the problem being treated, and is not something the tool
should decide on its own — so it is assumed explicitly, as follows.

```
Assume D > 0
```

**Expected output:** a blue assumption tile `ASM3`: `D > 0` is added.

**Check points (important):**
- Without this assumption, `ode2` cannot internally resolve the sign of the
  discriminant, and Maxima issues an interactive question asking for the sign —
  **hanging** the persistent adapter (`_build_solve_ode_general` automatically handles
  each individual symbol's positivity via `assume(sym>0)`, but that does not cover the
  sign of a **compound expression** such as a discriminant).
- The choice "we are treating the underdamped case ($D>0$)" is a premise of this
  scenario (**a choice made by the researcher**) and must not be hardcoded into
  `maxima_backend_handler.py`.
- **Since `D` is a quantity bound by `E3`, `assume(D > 0)` is expanded to `D`'s value
  (`4*omega_0^2-(2*gamma)^2`) internally in Maxima before being evaluated.** This does
  not contradict the previously-confirmed constraint that "Maxima's
  `assume()`/`is()` does not automatically normalize algebraically-equivalent
  expressions" — assuming `D` **as an already-bound quantity** produces a result that
  matches the form `ode2` checks internally (confirmed on a live Maxima session;
  conversely, writing a **different form** directly, such as
  `assume(omega_0^2 > gamma^2)`, without going through `D`, does not match and hangs).

### Step 8: the general solution (FREE-PARAM)

Under the assumption $D>0$, find the general solution of the equation of motion
(`E1`).

```
Find the general solution of E1
```

**Expected output:** `E4`:
`x=e^{-t\gamma}\left(k_{1}\sin\left(\dfrac{t\sqrt{4\omega_{0}^2-4\gamma^2}}{2}\right)+k_{2}\cos\left(\dfrac{t\sqrt{4\omega_{0}^2-4\gamma^2}}{2}\right)\right)`

**Check points:**
- The arbitrary constants `k_1, k_2` are held as `free_parameters` (consumed by a
  later step).
- `\sqrt{4\omega_0^2-4\gamma^2}/2` is equivalent to $\omega=\sqrt{D}/2$ (`E3`'s `D`)
  (Maxima does not automatically cancel the common factor 4 inside the radical — the
  output shown is exactly what comes back).
- **"solve E1 in general" / "solve E1 generally" / "solve E1 as a differential
  equation" / "solve the differential equation E1" / "find the general solution of
  E1" all route to the same `solve_ode_general`.** By contrast, "solve E1" (no
  qualifier) is unrecognized, and "solve E1 for x" **is** recognized but routes to the
  generic algebraic solver (`solve_equation`), which misreads the derivatives as
  ordinary algebraic variables and "succeeds" by returning the plausible-looking but
  meaningless algebraic rewrite $x = -(2\gamma\dot x+\ddot x)/\omega_0^2$. "in
  general"/"the general solution" is the effective trigger phrase that selects the
  dedicated general-solution ODE solver.
- **What actually happens:** Maxima's `ode2(equation, dependent_var, independent_var)`
  (the built-in solver for 1st/2nd-order linear ODEs) is called. Before the call, each
  scalar coefficient (`gamma`, `omega_0`) automatically receives `assume(sym>0)`; in
  addition, without step 7's assumption, an internal sign-confirmation interactive
  question is issued and the process hangs. The resulting `%k1,%k2` (Maxima's
  arbitrary-constant placeholders) are renamed to `k_1,k_2`.

### Step 9: fix the arbitrary constants from the initial conditions

The general solution (`E4`) contains arbitrary constants $k_1,k_2$. Give the initial
conditions $x(0)=1,\ \dot x(0)=0$ to fix them.

```
Set the initial conditions of E4 to x(0)=1, x'(0)=0
```

(Synonymous phrasings such as "…determine the constants in E4 from…" /
"…specialize E4 with…" give the same result; for a boundary-value problem,
"…set the boundary conditions of… to…" also works.)

**Expected output:** `E5`:
`x = e^{-t\gamma}\left(\cos\left(\dfrac{t\sqrt{4\omega_{0}^2-4\gamma^2}}{2}\right)-\dfrac{\gamma\sqrt{4\omega_{0}^2-4\gamma^2}\sin\left(\frac{t\sqrt{4\omega_{0}^2-4\gamma^2}}{2}\right)}{2\gamma^2-2\omega_{0}^2}\right)`

**Check points:**
- What is happening: each initial condition is substituted into the general solution
  (and, where needed, its derivatives) to form a system of equations in `k_1, k_2`,
  which Maxima's `solve` solves; those values are then substituted back into the
  general solution's right-hand side. This is the same operation as "applying initial
  conditions to fix the integration constants" in physics.
- `k_1, k_2` are consumed, so `free_parameters == []` (fully determined).

### Step 10: extract just the right-hand side for the numeric-layer fork

To fork into the numeric layer in the next step, extract and simplify just the
right-hand side of the determined solution (`E5`).

```
Simplify the right-hand side of E5
```

**Expected output:** `E6` (the raw RHS extraction, `evaluate_rhs`) is followed by `E7`
(after "simplify" = `ratsimp`). `E7`'s canonical is not the equality form `x = …` but
the **bare RHS expression** (the form required as a fork target).

**Check point:**
- The result of an ODE general solution / determination by initial conditions is
  stored in equality form `x = …` (unlike the H-1 convention "canonical is RHS only").
  The numeric layer's `fork_numeric_task` cannot parse an equality
  (`canonical_to_numpy` cannot treat `=` as an expression), so an H-7 operation such as
  "simplify the right-hand side" is needed before forking, to convert to a **bare RHS
  expression**. This is a step required by the scenario's own procedure, not a code
  defect.

---

## Numeric-layer steps (as of 2026-07-06: NL/UI trigger not yet implemented — API/engine operations)

Forking / code generation / sandbox execution in the numeric layer (N-1) can currently
only be driven via the REST endpoints (`/numeric/fork`, `/numeric/generate/{task_id}`,
`/numeric/run/{task_id}`) or the engine methods (`fork_numeric_task` /
`generate_numeric_code` / `run_numeric_task`) — neither a natural-language trigger nor
a clickable UI panel exists yet (as intended by the design in the CLAUDE.md N-1
section: the PoC scope is "generated code is the seed handed to a coding agent").
Below is the Python operation sequence, verified on the real backend (targeting `E7`,
fixing `gamma=0.3, omega_0=2.0` and sweeping `t`):

```python
task = engine.fork_numeric_task(
    "E7", problem_type="parameter_sweep", target_quantity="x",
    sweep_parameters={"t": {"min": 0, "max": 10, "points": 200}},
    fixed_values={"gamma": 0.3, "omega_0": 2.0},
)
engine.generate_numeric_code(task.task_id)
engine.save("<session_save_path>.json")   # numeric-layer output is tied to the save filename
execution = engine.run_numeric_task(task.task_id)
```

**Expected output (verified on the real backend, 2026-07-07, confirmed directly against `E7`):**
- `execution.return_status == 0` (sandbox execution succeeded)
- `NB_RESULT {"target": "x", "sweep_var": "t", "n_points": 200, "min": {"x": 1.608…, "y": -0.620…}, "max": {"x": 0.0, "y": 1.0}}`
- A `numeric_result` step reflux into the timeline:
  `Numeric computation N001: x(t) — min -0.620… (t=1.608…) / max 1.0 (t=0.0)`
- `max` being `y=1.0` at `t=0` is consistent with the initial condition $x(0)=1$.
  `min` is the first trough of the decaying oscillation.
- `figure.png` and `table.csv` are generated under
  `<session_name>_numeric/N001/<execution_id>/`.

**Check point:**
- `NumericTask.source_expression_ids == ["E7"]`, and `source_assumption_ids` snapshots
  `ASM1, ASM2, ASM3` — preserving the provenance of **which theory-layer assumptions**
  the numerical computation was performed under.

---

## Interactive UI fork verification (2026-07-18)

The above was verified via direct API/engine calls (2026-07-07). Since then the
"🔢 To numeric" button and fork modal have been implemented, so as part of the
practice loop we re-ran the same scenario **end-to-end through the browser UI**.

**Steps used this time (equivalent to the generator block above, but via English NL
and different numeric values):**
```
Define x as a time-dependent function
\ddot{x} + 2*gamma*\dot{x} + omega_0^2*x = 0
assume omega_0 > gamma
solve E1 generally
specialize E2 with x(0)=1, x'(0)=0
evaluate the right-hand side of E3
```
Select that expression (`E6`, the bare RHS) → "🔢 To numeric" → sweep variable=`t`,
Min=0, Max=20, Fixed values: `gamma=0.3, omega_0=2` → Generate code → ▶ Run.

**Result (real backend, browser, via `sandbox-exec`):**
- `min -0.6204120728921954 at t=1.6080402010050252` / `max 1.0 at t=0.0` —
  **identical** to the 2026-07-07 API-level verification (same `gamma=0.3,
  omega_0=2.0`), confirming reproducibility.
- The decaying waveform (oscillating from `x(0)=1` down toward 0) reflux into the
  timeline as a figure and was visually confirmed.
- The modal's guard against forking an Equation ("This expression construct
  (Equation) is not supported by the phase-1 numeric layer (scalar expressions
  only)") was also confirmed live — attempting to fork the equality form (`x = …`)
  directly grays out "Generate code" and shows that message. This is an intentional
  restriction; the correct workaround (already documented above) is to convert to a
  bare RHS via "evaluate the right-hand side of …" first.

**An additional caveat found (not a code bug — a known scenario-procedure
requirement):** this run skipped steps 1–3 of the documented procedure ("define
`\gamma`/`\omega_0` as positive real numbers") and instead only did `assume omega_0 >
gamma`. With `\gamma`/`\omega_0` left undeclared, running `evaluate the right-hand
side of E3` (which internally calls `simplify`) reproducibly corrupted the
digit-subscript symbol `omega_0` into `omega(0)` (a function call) — the generic
operation-dispatch path resolves an **undeclared** subscript as an index (see
CLAUDE.md's "Undefined Symbol Detection" section), unlike the FREE-PARAM
special-cased handlers (`solve_ode_general`/`specialize_by_conditions`), which are
unaffected. Declaring `\omega_0` (as a positive real number, or any scalar
declaration) before the generic operation avoids this entirely — which is exactly
what steps 1–3 of this scenario already do; this run deliberately skipped them to
confirm the edge case. No code fix is needed (declaring the symbol first works
correctly), but the requirement — "declare digit-subscript physics parameters
explicitly before passing them through a generic operation" — is recorded here since
it could resurface in other scenarios.

---

## Bugs found and fixed (code fixes, not scenario workarounds)

While running this scenario on the real backend, we found that passing the fork
target's canonical (long expressions Maxima pretty-prints get line-wrapped with
indented continuation lines) directly into the numeric layer's template produced
generated Python code that was a **syntax error (`IndentationError`)**.

The cause: `notebook/numeric/templates.py`'s one-line comment
`# --- theory expression: {task.source_canonical} ---` embedded the canonical
verbatim, including any embedded newlines, so a continuation line lost its `#` and was
interpreted as literal Python code (e.g. the second line `  +(...)` of
`sin(...)\n +(...)*cos(...)` looked like an indented constructing statement).

**Fix:** added a `_single_line()` helper that collapses newlines and consecutive
whitespace into a single space immediately before embedding into the comment
(`templates.py`); the multi-line provenance display in the docstring (where multiple
lines are intentional) was left unchanged. Added the regression test
`tests/test_numeric_layer.py::TestTemplates::test_multiline_canonical_stays_valid_python`.
The existing `_task()` fixture's canonical was a simple expression with no newlines, so
this had gone unnoticed until now.

## NL-side improvements found during manual scenario review (code fixes)

While manually walking through this scenario, the following points were raised and the
code was fixed accordingly:

- **TeX-notation support:** TeX notation such as `\gamma`/`\omega_0` was already
  correctly interpreted as text input (the natural form when going through the
  handwriting-input layer). The original document's use of the bare ASCII spellings
  `gamma`/`omega_0` directly was itself a documentation error, not a code bug.
- **The origin of the discriminant was unclear:** the original document had the reader
  assume the inequality `(2*\gamma)^2 < 4*\omega_0^2` directly, with no trace of where
  it came from as a registered expression. Added steps that actually register the
  characteristic equation (`E2`) and the discriminant `D` (`E3`, an H-1 registration),
  rephrased as "Assume D > 0" (separately from [project_domain_sign_split], `D`'s
  value itself simply flows through to Maxima's `assume()` via the general H-1
  bound-symbol behavior — no new code was required for this).
- **Need for "in general" / additional phrasings:** "solve E1" / "solve the
  differential equation of E1" were originally unrecognized; resolved by adding
  "solve the differential equation …" / "find the general solution of the differential
  equation of …" to `_ODE_GENERAL_RE` (and its English counterpart
  `_ODE_GENERAL_RE_EN`) in `ui/interpret_free_params.py`. "find the general solution
  of …" was already supported.
- **The term "specialize":** feedback that this CS-derived term was unfamiliar to
  physics researchers led to adding NL triggers "determine"/"fix"/"decide", and
  declarative phrasings that state the condition as a statement — "set the initial
  conditions of E5 to x(0)=1, x'(0)=0" / "set the boundary conditions of … to …" (the
  internal operation name `specialize_by_conditions` itself was left unchanged). Error
  messages (e.g. `engine.specialize_no_params`) were also updated to avoid
  "specialize"-derived wording. The boundary-condition form was also confirmed to work
  for a genuine two-point boundary-value problem not limited to $t=0$ (e.g.
  $q(0)=q_0,\ q(10)=q_1$).
- **Formulas appeared abruptly with no connection to their explanation:** the scenario
  document had introductory prose, but that prose lived only in the `.md` file — it was
  reflected in neither the executed `scenario` block nor the generated TeX output.
  Added support for `@note: <text>` lines to `scripts/generate_scenario_examples.py`
  (which simply calls the existing UI "📝 memo" mechanism, `engine.record_manual_note`),
  inserting an explanatory memo immediately before each formula. The memo is recorded
  as a genuine timeline step, and `export_latex`'s existing `manual_note` handling
  (`_note_to_latex`) renders it into the TeX as-is (no new rendering code was needed).
  This memo mechanism is not specific to this scenario — the same `@note:` syntax can
  be used in any scenario document.

---

## Coverage (features verified by this scenario)

| Feature | Example input | Implementation |
|---|---|---|
| Explicitly define the discriminant and assume it (a problem-dependent modeling choice) | `D = 4 \omega_0^2 - (2 \gamma)^2` → `Assume D > 0` | H-1 registration → `_try_add_assumption` → `engine.add_assumption` |
| A sign declaration is actually wired into assume() | `Define \gamma as a positive real number` | `Quantity.sign` + `BackendHandler.assume_sign` |
| ODE general solution (FREE-PARAM) | `Find the general solution of E1` | `solve_ode_general` |
| Fix arbitrary constants from initial conditions (declarative form) | `Set the initial conditions of E4 to x(0)=1, x'(0)=0` | `specialize_by_conditions` |
| Extract a bare RHS from an equality result | `Simplify the right-hand side of E5` | `evaluate_rhs` (H-5/H-7) |
| Theory expression → numeric-layer fork | `fork_numeric_task("E7", …)` | N-1 phase 1 (deterministic template) |
| Sandbox execution → provenance-carrying reflux | `run_numeric_task(task_id)` | `numeric_result` step |

**Related design docs:** [12_symbol_domain_design.md](../developer_manuals/12_symbol_domain_design.md),
[22_numeric_layer_design.md](../developer_manuals/22_numeric_layer_design.md)

**Verified:** 2026-07-09 (real Maxima backend, generator run, all 10 theory-layer steps
completed end-to-end, same results as the Japanese version).

## generator commands (theory layer only; this `scenario` block is the single source used to generate the example)

The numeric-layer fork onward is not NL-driven, so it is not part of this
auto-generated block (see "Numeric-layer steps" above; run and verify separately via
script).

```scenario domain=differential_equations
@note: First, prepare the time-dependent displacement $x$ and the damping rate $\gamma$ / natural angular frequency $\omega_0$ (both positive reals).
Define x as a time-dependent function
Define \gamma as a positive real number
Define \omega_0 as a positive real number
@note: Consider the damped oscillator's equation of motion below ($\gamma$ is the damping rate, $\omega_0$ is the natural angular frequency).
$\ddot{x} + 2 \gamma \dot{x} + \omega_0^2 x = 0$
@note: Substituting the exponential ansatz $x=e^{rt}$ into this differential equation ($\dot x=re^{rt},\ \ddot x=r^2e^{rt}$, then dividing both sides by $e^{rt}\neq0$) gives the characteristic equation of this differential equation below.
$r^2 + 2 \gamma r + \omega_0^2 = 0$
@note: Since the characteristic equation is a quadratic in $r$ ($ar^2+br+c=0$ with $a=1,\ b=2\gamma,\ c=\omega_0^2$), its discriminant is $b^2-4ac=(2\gamma)^2-4\omega_0^2$. To phrase the discussion below in the natural form "positive means underdamped", we define the sign-flipped quantity $D=4\omega_0^2-(2\gamma)^2=-(b^2-4ac)$.
$D = 4 \omega_0^2 - (2 \gamma)^2$
@note: Here we assume underdamped (oscillatory) motion, $D>0$. This choice is a physical judgment that depends on the problem being treated, and is not something the tool should decide on its own, so it is assumed explicitly, as follows.
Assume D > 0
@note: Under the assumption $D>0$, find the general solution of the equation of motion (E1).
Find the general solution of E1
@note: The general solution (E4) contains arbitrary constants $k_1,k_2$. Give the initial conditions $x(0)=1,\ \dot x(0)=0$ to fix them.
Set the initial conditions of E4 to x(0)=1, x'(0)=0
@note: To fork into the numeric layer in the next step, extract and simplify just the right-hand side of the determined solution (E5).
Simplify the right-hand side of E5
```
