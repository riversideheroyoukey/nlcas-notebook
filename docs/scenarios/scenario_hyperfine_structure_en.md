# Practice-loop scenario S-p: hyperfine structure ($H_{\rm hf}=A\,\mathbf I\cdot\mathbf S$)

> Run 2026-08-04.  The hyperfine coupling of an electron spin $S=1/2$ to a nuclear spin $I=3/2$,
> taken along **two independent routes** — angular-momentum coupling (F-9) and a spin-algebra trace
> (F-13).

**Backend:** Maxima (though both routes bypass the CAS) / **Domain:** magnetic resonance.

### The physics

The hyperfine interaction is $H_{\rm hf}=A\,\mathbf I\cdot\mathbf S$.  In terms of the total angular
momentum $\mathbf F=\mathbf I+\mathbf S$,

$$\mathbf I\cdot\mathbf S=\tfrac12\left[F(F+1)-I(I+1)-S(S+1)\right]$$

For $I=3/2,\,S=1/2$ this splits into $F=1,2$, giving

$$E_{F=2}=\tfrac34 A,\qquad E_{F=1}=-\tfrac54 A,\qquad \Delta E=2A=A\,F_{\rm max}$$

— the **Landé interval rule**.  Separately, the infinite-temperature moments are
$\langle\mathbf I\cdot\mathbf S\rangle=0$ and
$\langle(\mathbf I\cdot\mathbf S)^2\rangle=\tfrac13 x_1x_2$ with $x_i=S_i(S_i+1)$, so for
$S_1=1/2,\,S_2=3/2$ that is $\tfrac13\cdot\tfrac34\cdot\tfrac{15}4=\tfrac{15}{16}$.

**The two are an independent check on each other**: one gives where the levels sit, the other how the
distribution spreads, and both come out of the same $A\,\mathbf I\cdot\mathbf S$.

### The two routes (the point of this scenario)

| | Route A: angular-momentum coupling (F-9) | Route B: spin-algebra trace (F-13) |
|---|---|---|
| What it produces | allowed $F$, the $\mathbf I\cdot\mathbf S$ identity, the levels, the interval rule | $\langle\mathbf I\cdot\mathbf S\rangle$, $\langle(\mathbf I\cdot\mathbf S)^2\rangle$ |
| Implementation | `angular_momentum.py` (exact, self-contained, no SymPy) | `spin_algebra.py` (likewise) |
| Composability | the identity keeps $F$ **symbolic**, so it rides ordinary operations — multiply both sides by $A$, substitute $F$, … | the result registers as a **substitutable CAS value** via `Coeff.to_canonical()` |

Both are worked examples of the "a result must not be a dead end" rule that F-9 and F-13 were built on.

---

### Steps

#### ST-1: setup

| Item | Content |
|---|---|
| Action | new session, field "magnetic resonance" (Maxima) |
| Input | `Define A as a scalar` |
| Result | |

---

#### ST-2 [route A]: couple the angular momenta

| Item | Content |
|---|---|
| Input | `couple I=3/2 and S=1/2` |
| Expected | E1: allowed $F=1,\,2$ / E2: $\mathbf I\cdot\mathbf S=\frac12[F(F+1)-\frac{15}4-\frac34]$ |
| Check | $F$ stays **symbolic** — collapsed to a number it would not ride the general operations that follow |
| Result | |

---

#### ST-3 [route A]: turn it into the Hamiltonian

| Item | Content |
|---|---|
| Input | `multiply both sides of E2 by A` |
| Expected | E3: $A\,\mathbf I\cdot\mathbf S=\dfrac{2AF^2+2AF-9A}{4}$ |
| Check | an identity built without the CAS rides an ordinary both-sides operation (H-8) unchanged |
| Result | |

---

#### ST-4 [route A]: the two levels

| Item | Content |
|---|---|
| Input | `in E3, substitute 2 for F` |
| Expected | E4: $A\,\mathbf I\cdot\mathbf S=\dfrac{3A}{4}$ |
| Input | `in E3, substitute 1 for F` |
| Expected | E5: $A\,\mathbf I\cdot\mathbf S=-\dfrac{5A}{4}$ |
| Result | |

---

#### ST-5 [route A]: the Landé interval rule

| Item | Content |
|---|---|
| Input | `E4 - E5` |
| Input | `simplify E6` |
| Expected | E7: $2A$ — i.e. $\Delta E=A\,F_{\rm max}$, the **Landé interval rule** |
| Result | |

---

#### ST-6 [route B]: declare the site spins

| Item | Content |
|---|---|
| Input | `site 1 has spin 1/2 and site 2 has spin 3/2` |
| Check | the declaration is recorded as a **timeline step** — which spins a trace was taken with is part of what the result means (F-13's design) |
| Result | |

---

#### ST-7 [route B]: the moments

| Item | Content |
|---|---|
| Input | `S^x_1 S^x_2 + S^y_1 S^y_2 + S^z_1 S^z_2` ($\mathbf I\cdot\mathbf S$, E8) |
| Input | `expectation value of E8` |
| Expected | E9: $\langle\mathbf I\cdot\mathbf S\rangle=0$ |
| Input | `2nd moment of E8` |
| Expected | E10: $\langle(\mathbf I\cdot\mathbf S)^2\rangle=\dfrac{15}{16}$ |
| Check | matches $\frac13 x_1x_2$ with $x_1=\frac34,\,x_2=\frac{15}4$ — the **per-site** spins are doing the work; a uniform spin would give $\frac13x^2$ |
| Result | |

---

### Verification checklist

- [x] the coupling returns the allowed $F=1,2$ and keeps $F$ symbolic in the identity
- [x] a CAS-free identity rides both-sides (H-8) and substitution unchanged
- [x] levels $\frac34A$ and $-\frac54A$, interval $2A$ (Landé)
- [x] the site-spin declaration is a timeline step
- [x] $\langle\mathbf I\cdot\mathbf S\rangle=0$ and $\langle(\mathbf I\cdot\mathbf S)^2\rangle=\frac{15}{16}$
- [x] the two routes reach the same physics independently

---

## What this run found

**No new defect.**  This run confirmed that the "a result must not be a dead end" design of F-9
(2026-07-28) and F-13 (2026-07-29) actually carries into a chain of general operations — ST-3 through
ST-5 are only writable because the identity keeps $F$ symbolic.

**An internal intermediate consumes an E-number.**  At the time, `multiply both sides of E2 by A`
created E3 (an inactive intermediate that never reaches the timeline) and E4 (the result), so the
next thing to reference was **E4**.  This is the same shape as the inverse transform's gap
(open_items INTERNAL-INTERMEDIATE-EID-GAP) and **it was misread twice while writing this scenario**.
Since more than one operation family does it, the ticket was widened from "the inverse transform" to
"operations that create internal intermediates".
**[Resolved 2026-08-06]** Intermediates are keyed in a separate namespace (`_T1`) from birth and
take no E-number, so the next thing to reference is simply **E3** — and the numbers in the tables
above each moved down by one.

**Spin components are spelled `S^x_1`** (superscript plus subscript).  The suffix form `Sz_1` is
accepted by design elsewhere (CLAUDE.md, "Julia-specific: Spin component notation"), but this
scenario runs the CAS-free path in a Maxima session, where
`spin_algebra.parse_spin_expression` takes the `S^a_i` form.  Ladder operators (`Sp`/`Sm`) are not
accepted, so $\mathbf I\cdot\mathbf S$ is written as the sum of its $x,y,z$ components.

**`E4 - E5` displays as typed.**  The canonical is expanded (`((3*A)/4) - (-((5*A)/4))`) while the
latex keeps `E4 - E5`.  As provenance that is reasonable — it says which two results were
subtracted — though an expanded form would read better in a paper.  Here the next `simplify` turns it
into $2A$, so nothing is lost.

**Verified:** 2026-08-04 (real Maxima, ST-1 through ST-7, all 11 steps).

---

## Generator commands (this `scenario` block is the single source)

```scenario domain=magnetic_resonance
@note: The hyperfine interaction $H_{\rm hf}=A\,\mathbf I\cdot\mathbf S$, taken along two independent routes. $A$ is the hyperfine coupling constant.
Define A as a scalar
@note: [Route A] Couple the nuclear spin $I=3/2$ to the electron spin $S=1/2$. This gives the allowed total angular momenta $F$ and an identity expressing $\mathbf I\cdot\mathbf S$ through $F$. $F$ stays symbolic, so the identity can be carried on by ordinary operations.
couple I=3/2 and S=1/2
@note: Multiply both sides by the coupling constant $A$ to get the hyperfine eigenvalues — and to check that an identity built without the CAS rides an ordinary both-sides operation.
multiply both sides of E2 by A
@note: Substitute the upper level $F=2$.
in E3, substitute 2 for F
@note: Substitute the lower level $F=1$.
in E3, substitute 1 for F
@note: Take the interval between the two levels. The Landé interval rule $\Delta E = A\,F_{\rm max}$ says it should be $2A$.
E4 - E5
simplify E6
@note: [Route B] Now the same $\mathbf I\cdot\mathbf S$ through a spin-algebra trace at infinite temperature. First declare each site's spin — which spins the trace was taken with is part of what the result means, so this is recorded as a timeline step.
site 1 has spin 1/2 and site 2 has spin 3/2
@note: Register $\mathbf I\cdot\mathbf S$ as the sum of its $x,y,z$ components.
S^x_1 S^x_2 + S^y_1 S^y_2 + S^z_1 S^z_2
@note: The first moment (expectation value), which should vanish at infinite temperature.
expectation value of E8
@note: The second moment. With $\frac13 x_1x_2$ and $x_i=S_i(S_i+1)$, putting in $x_1=3/4$ and $x_2=15/4$ should give $15/16$ — confirmation that the per-site spins are doing the work.
2nd moment of E8
```
