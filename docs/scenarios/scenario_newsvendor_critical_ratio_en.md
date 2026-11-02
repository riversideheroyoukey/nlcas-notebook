# Practice-loop sample scenario: the newsvendor model (critical ratio, optimal order quantity under uniform demand)

Detailed version of practice-loop scenario menu **S-v (supply chain / logistics)**. Given
underage cost $c_u$ and overage cost $c_o$, with demand following a uniform distribution
$U[a,b]$, this scenario solves the expected-cost minimization problem via
**differentiate → marginal analysis → solve** to find the optimal order quantity $Q^*$.

**Backend:** standard (Maxima) / **Domain:** supply chain / logistics (supply_chain_logistics).

## Background

When demand $D$ follows a known distribution, the expected cost as a function of order
quantity $Q$ is the sum of the expected "overage" cost (when $D<Q$, the leftover
$Q-D$ costs $c_o$ per unit) and the expected "underage" cost (when $D>Q$, the shortfall
$D-Q$ costs $c_u$ per unit). In general, differentiating this expected cost with
respect to $Q$ and setting it to zero (marginal analysis) gives the **critical ratio**
formula in terms of the demand's CDF $F$:

$$F(Q^*)=\frac{c_u}{c_u+c_o}$$

(a distribution-independent result).

For demand uniform on $U[a,b]$ (density $f=1/(b-a)$), the expected cost has a closed form:

$$C(Q)=\underbrace{c_o\,\frac{(Q-a)^2}{2(b-a)}}_{\text{expected overage}}
      +\underbrace{c_u\,\frac{(b-Q)^2}{2(b-a)}}_{\text{expected underage}}.$$

Differentiating $C$ with respect to $Q$, setting it to zero, and solving:

$$\frac{dC}{dQ}=\frac{c_o(Q-a)}{b-a}-\frac{c_u(b-Q)}{b-a}=0
\;\Longrightarrow\; Q^*=\frac{c_ub+c_oa}{c_u+c_o}
=a+(b-a)\frac{c_u}{c_u+c_o}.$$

(the latter form matches substituting the uniform CDF $F(Q)=(Q-a)/(b-a)$ into the
critical-ratio formula).

## Session setup / symbols

- Mode **standard (Maxima)**, domain **supply chain / logistics**.
- $Q,a,b$ are free variables. $c_u,c_o$ (underage/overage cost) carry a **single-letter
  alphabetic subscript** (`u`, `o`), so declare them as real before use (see caveat
  below).
- **Product notation:** writing `c_o (Q-a)^2` (a symbol followed directly by a space
  then a parenthesis) fails to parse (`c_o` is neither a known function name nor a
  declared callable). Use an explicit `*` (`c_o * (Q - a)^2`).

## Input steps

### Step 1: declare the cost coefficients

```
Declare c_u, c_o as real.
```

**Expected output:** a confirmation dialog ("u is a component index" vs. "u is part
of the name — c_u is a distinct quantity"); pick "part of the name" (the default
option) — both $c_u$ and $c_o$ register correctly.

> **Caveat (single-letter alphabetic subscript branching):** `c_u` and `c_o` have a
> single-letter alphabetic subscript, not a digit, so the confirmation dialog fires
> (per the project's documented subscript-classification rule). Choosing "part of the
> name" correctly treats $c_u$/$c_o$ as distinct cost parameters.

### Step 2: define the expected cost function (H-1)

```
C = c_o * (Q - a)^2 / (2 * (b - a)) + c_u * (b - Q)^2 / (2 * (b - a))
```

**Expected output:** registered as E1.

### Step 3: total derivative with respect to $Q$

```
Take the total derivative of E1 with respect to Q
```

**Expected output:** E2 $=\dfrac{dC}{dQ}=\dfrac{(Q-a)c_o}{b-a}-\dfrac{(b-Q)c_u}{b-a}$.

### Step 4: solve for $Q$ (the optimal order quantity)

```
Solve E2 for Q
```

**Expected output:** E3 $=Q^*=\dfrac{bc_u+ac_o}{c_u+c_o}$ (algebraically equivalent to
$a+(b-a)\,c_u/(c_u+c_o)$ — see "Confirmation points" below).

## Confirmation points

- E3's $Q^*=(bc_u+ac_o)/(c_u+c_o)$ is algebraically equal to the uniform-demand
  specialization of the critical ratio, $Q^*=a+(b-a)c_u/(c_u+c_o)$
  ($a+(b-a)c_u/(c_u+c_o)=(a(c_u+c_o)+(b-a)c_u)/(c_u+c_o)=(ac_o+bc_u)/(c_u+c_o)$).
- Starting from a closed-form expected cost function for a given demand distribution,
  the general differentiate → solve pattern derives the optimal order quantity directly
  (the same procedure extends to other demand distributions).

## Observed behavior / caveats

- **A single-letter alphabetic subscript triggers a declaration confirmation
  dialog.** `c_u`/`c_o`'s `u`/`o` are not digits, so the existing subscript
  classification rule (see the "Subscript Classification in Symbol Declaration"
  section of CLAUDE.md) cannot resolve them automatically and asks the user —
  unlike digit subscripts (`a_1`, etc.), which always resolve to "part of the name"
  automatically (see S-s).
- **Bulk-declaring multiple symbols needing confirmation in one command only
  processed the first (found and root-caused, 2026-07-05).** Writing
  `"Declare c_u, c_o as real."` (declare both **in one comma-separated command**)
  used to apply the confirmation answer to `c_u` only — `c_o` stayed undeclared
  (the derivative result then showed a corrupted function-call form, `c(o)`).
  Root cause: `extra_pre_phase1` in `notebook/runtime/backend_handler.py` only
  applied the chosen interpretation to the first ambiguous name found. Fixed by
  applying the choice (component index vs. part of the name) to every ambiguous
  name in the list (`tests/test_batch_symbol_declaration.py`). Distinct from the
  English bulk-declaration bug (J-4, only first/last of a comma list registers) —
  this one was specific to the confirmation-dialog declaration path, in both
  languages. Both J-4 and J-5 are now closed in open_items.md.

## Example files

The executed session and exported document:
[../../domains/supply_chain_logistics/examples/scenario_newsvendor_critical_ratio_en.json](../../domains/supply_chain_logistics/examples/scenario_newsvendor_critical_ratio_en.json)
/ `.tex` (the Japanese version is `_jp.{json,tex}`).

**Regenerate:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`

## Generator commands (this `scenario` block is the single source for generation)

```scenario domain=supply_chain_logistics
@note: Suppose demand $D$ follows a uniform distribution on $[a,b]$ ($a,b$ are the lower and upper bounds of demand). For an order quantity $Q$, demand below $Q$ leaves excess inventory (overage cost $c_o$ per unit), while demand above $Q$ causes a stockout (underage cost $c_u$ per unit). Prepare $c_u,c_o$ as real parameters. $D$ itself will not appear symbolically in the expressions below — it is a random variable, integrated out (over the uniform density $1/(b-a)$) to obtain the closed-form EXPECTED cost $C(Q)$ in the next step, which depends only on $Q,a,b,c_u,c_o$.
Declare c_u, c_o as real.
@note: With this setup, define the expected total cost $C(Q)=E_D[\text{overage}+\text{underage}]$, as a function of the order quantity $Q$, as the sum of the expected overage and underage costs — already integrated over $D$'s uniform density (this is the closed form one gets after evaluating $\int_a^Q c_o(Q-D)\frac1{b-a}\,dD+\int_Q^b c_u(D-Q)\frac1{b-a}\,dD$ by hand), so $D$ does not appear here; only its distribution's parameters $a,b$ do.
C = c_o * (Q - a)^2 / (2 * (b - a)) + c_u * (b - Q)^2 / (2 * (b - a))
@note: To find the order quantity that minimizes the expected total cost, take the total derivative of $C$ with respect to $Q$.
Take the total derivative of E1 with respect to Q
@note: Set the derivative to zero and solve for $Q$ to find the optimal order quantity $Q^*$ that minimizes the expected total cost.
Solve E2 for Q
```
