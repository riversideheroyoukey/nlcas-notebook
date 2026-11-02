# Practice-loop sample scenario: risk pooling in inventory placement — the square-root law for safety stock

Detailed version of practice-loop scenario menu **S-w (supply chain / logistics)**.
Substituting into the safety stock formula $SS=z\sigma\sqrt{L}$ for the single-location
case ($L$ unchanged) and the $n$-location pooled central-warehouse case ($L\to nL$, from
the additivity of variance) and taking their ratio derives the **risk-pooling
square-root law** $SS_{\rm pool}/SS_{\rm sep}=1/\sqrt{n}$.

**Backend:** standard (Maxima) / **Domain:** supply chain / logistics (supply_chain_logistics).

## Background

Safety stock is given by $SS=z\sigma\sqrt{L}$ in terms of the demand standard deviation
$\sigma$, the service-level factor $z$, and lead time $L$ (base stock is
$S=\mu L+z\sigma\sqrt{L}$).

Pooling demand from $n$ locations into one central warehouse makes variance additive
(under the iid assumption), replacing the effective lead-time term $L$ with $nL$:

$$SS_{\rm pool}=z\sigma\sqrt{nL},\qquad SS_{\rm sep,total}=n\cdot z\sigma\sqrt{L}
\;(\text{the sum of the } n \text{ separate locations' safety stocks}).$$

$$\frac{SS_{\rm pool}}{SS_{\rm sep,total}}=\frac{\sqrt{nL}}{n\sqrt{L}}=\frac{1}{\sqrt{n}}
\quad\text{(the square-root law)}.$$

Pooling $n$ locations means only $1/\sqrt{n}$ times the total safety stock is needed.

## Session setup / symbols

- Mode **standard (Maxima)**, domain **supply chain / logistics**.
- $z,\sigma,L,n$ are free variables. $SS$ is a named H-1 definition.

## Input steps

### Step 1: the general safety stock formula (H-1)

```
SS = z * sigma * sqrt(L)
```

**Expected output:** registered as E1. $SS=z\sigma\sqrt{L}$.

### Step 2: pool $n$ locations ($L\to nL$, additivity of variance)

```
In E1, substitute n * L for L
```

**Expected output:** E2 $=SS_{\rm pool}=\sqrt{Ln}\,\sigma z$ (the original E1 is still
available as $SS_{\rm sep}$).

### Step 3: take the ratio (H-2 formula-reference expansion)

```
ratio = E2 / (n * E1)
```

**Expected output:** E3 $=\text{ratio}=\dfrac{\sqrt{Ln}}{\sqrt{L}\,n}$ (algebraically
equivalent to $1/\sqrt{n}$, but not yet in that reduced form — see Step 4).

### Step 4: reduce to the textbook form (radical_simplify)

```
radical_simplify E3
```

**Expected output:** E4 $=\text{ratio}=\dfrac{1}{\sqrt{n}}$, the familiar square-root law.

## Confirmation points

- E3's $\sqrt{Ln}/(\sqrt{L}\,n)$ is algebraically equal to $1/\sqrt{n}$ (squaring both
  sides gives $Ln/(Ln^2)=1/n$ on both), but the general-purpose `simplify` operation
  (`ratsimp`/`trigsimp` in Maxima) does not reduce this on its own — Maxima does not
  automatically combine separate radicals ($\sqrt{Ln}$ and $\sqrt{L}$) across a
  fraction unless asked.  The dedicated `radical_simplify` operation (Maxima's
  `radcan`) performs that reduction and is a general operation like any other — not a
  one-off fix for this scenario.
- `In E1, substitute n * L for L` (a **self-referential** substitution — the LHS symbol
  `L` also appears literally in the RHS `n * L`) is correctly processed as a one-shot
  algebraic substitution (see "Observed behavior" below — the circular-substitution
  guard's false positive here has been root-caused and fixed).
- H-2 (formula-reference expansion) lets `ratio = E2 / (n * E1)` — combining **multiple
  expression ids into one new expression** — go straight through to a computed result.

## Observed behavior / caveats (found and root-caused)

- **The circular-substitution guard used to block every self-referential
  substitution, even ones that are not actually circular
  (SUBST-LINEAR-SELFREF, found and root-caused 2026-07-05).**
  `In E1, substitute n * L for L` ($L\to nL$, where the LHS `L` also appears literally in
  the RHS `n * L`) used to fail with a **circular substitution error** ("cannot
  substitute into L because L appears on the right-hand side"). That guard exists to
  prevent a genuinely dangerous case: substituting $q=-\ddot q/\omega^2$ (solved from
  $\ddot q+\omega^2q=0$) back into an expression chains multiple LHS spellings (bare `q`
  and the derivative form `'diff(q,t,2)`) through nested `subst()` calls — since the
  RHS itself contains the derivative, each nested call re-inserts a fresh copy for the
  other to match again, diverging. But a **plain algebraic self-reference with no
  derivative involved** (a scaling/renaming substitution like $L\to nL$) is not
  circular — Maxima's `subst()` is single-pass and handles it correctly in one step,
  confirmed directly against a live Maxima session
  (`subst(L=n*L, z*sigma*sqrt(L))` correctly returns `sqrt(L*n)*sigma*z`). The guard was
  narrowed to only fire when the LHS appears inside a `'diff(...)` call in the RHS
  (`notebook/runtime/maxima_backend_handler.py`, `_build_substitute`), so the true
  $\ddot q$ circular case is still blocked while a plain self-referential scaling
  substitution like this scenario's now goes through
  (`tests/test_circular_substitution_guard.py`).

## Example files

The executed session and exported document:
[../../domains/supply_chain_logistics/examples/scenario_risk_pooling_safety_stock_en.json](../../domains/supply_chain_logistics/examples/scenario_risk_pooling_safety_stock_en.json)
/ `.tex` (the Japanese version is `_jp.{json,tex}`).

**Regenerate:** `PYTHONPATH=. .venv/bin/python scripts/generate_scenario_examples.py`

## Generator commands (this `scenario` block is the single source for generation)

```scenario domain=supply_chain_logistics
@note: Define the general safety-stock formula $SS=z\sigma\sqrt{L}$ ($z$: service-level factor, $\sigma$: demand standard deviation, $L$: lead time).
SS = z * sigma * sqrt(L)
@note: Consider pooling the demand of $n$ locations into a single central warehouse. Under the i.i.d. assumption, variances add, so the effective lead-time term becomes $nL$ instead of $L$. Substitute this replacement into the original expression (the original expression itself remains available as the single-location case).
In E1, substitute n * L for L
@note: Take the ratio of the pooled safety stock to the total safety stock of the $n$ separate locations ($n$ times the original expression), to confirm the risk-pooling effect (the square-root law $1/\sqrt n$).
ratio = E2 / (n * E1)
@note: $E3=\sqrt{Ln}/(\sqrt{L}\,n)$ is algebraically $1/\sqrt{n}$, but Maxima's general `simplify` does not combine the two separate radicals on its own. Apply the dedicated radical-simplification operation to reduce it to the familiar closed form.
radical_simplify E3
```
