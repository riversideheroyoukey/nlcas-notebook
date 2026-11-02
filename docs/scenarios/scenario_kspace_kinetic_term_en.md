# Practice-loop scenario: the k-space kinetic term $\xi_k c^\dagger_k c_k$ (an indexed scalar)

> Run 2026-08-03.  The most ordinary thing written in condensed matter — **an indexed scalar
> multiplying operators** — checked through normal ordering and a commutator in a Julia session.

**Backend:** Julia (QuantumAlgebra.jl) / **Domain:** condensed matter.

### Why this is worth checking

$\xi_k c^\dagger_k c_k$ is on the first page of the textbook, and **until 2026-08-02 it could not be
written**.  To QuantumAlgebra `xi_k` is a c-number carrying an index, and Julia had no spelling for
that, so the CAS answered `Sym xi is not callable` (open_items JULIA-INDEXED-SCALAR).

The resolution is to route it through `param(:xi, k)`, and that was **chosen by measurement**:
`param` multiplies QA operators, survives `normal_form`, combines with other params, multiplies a
Symbolics variable — and its latex is `\xi_{k}`, **the researcher's own subscript**, so no display
bridge is needed.  `Symbolics.@syms` displays as `xi(k)` (CAS syntax leaking out) and cannot
multiply a QA operator at all.

**The declaration is the warrant.**  An undeclared `V_i` is still refused with a reason: routing
every unbound index to `param` turned the structural token in `\sum_k xi_k c'_k c_k` into
`param(:sum, k)`, so **a refusal became silent nonsense** (measured 2026-08-02).

### The physics

1. Write the kinetic term $\xi_k c^\dagger_k c_k$ and normal order it (already ordered, so unchanged)
2. Write a Hamiltonian containing the deliberately reversed $\xi_k c_k c^\dagger_k$ and normal order
   it → the anticommutator $c c^\dagger = 1 - c^\dagger c$ produces a constant term
3. Take $[H, c_k]$ → $\xi_k c_k - \Delta c^\dagger_{-k}$

Step 3 is checkable by hand: $[c^\dagger_k c_k, c_k] = -c_k$ and
$[c^\dagger_k c^\dagger_{-k}, c_k] = -c^\dagger_{-k}$ for $k\neq0$.

---

### Steps

#### ST-1: session and declarations

| Item | Content |
|---|---|
| Action | new session, field "condensed matter" → the Julia backend is selected automatically |
| Input | `c as the annihilation operator for electron` |
| Input | `Define xi_k as a scalar` |
| Check | if the subscript-role dialog appears, choose **"part of the name"** ($\xi_k$ is one quantity, "xi at k") |
| Input | `Define Delta as a scalar` |
| Result | |

---

#### ST-2: the kinetic term

| Item | Content |
|---|---|
| Input | `xi_k c'(k) c(k)` |
| Input | `normal_order E1` |
| Expected | $\xi_k c^\dagger_k c_k$ (already normal-ordered, so unchanged) |
| Check | the display is **`\xi_{k}`**, the researcher's own subscript — no `xi(k)` CAS syntax |
| Result | |

---

#### ST-3: a reversed product → normal ordering

| Item | Content |
|---|---|
| Input | `H = xi_k c(k) c'(k) + Delta c'(k) c'(-k)` |
| Input | `normal_order E3` |
| Expected | $H = \xi_k - \xi_k c^\dagger_k c_k + \Delta c^\dagger_k c^\dagger_{-k}$ |
| Check | $c c^\dagger = 1 - c^\dagger c$ has been applied and **the constant $\xi_k$ appears** — the indexed scalar passed through the anticommutator intact |
| Result | |

---

#### ST-4: the commutator

| Item | Content |
|---|---|
| Input | `commutator of E3 and c(k)` |
| Expected | $\xi_k c_k - \Delta c^\dagger_{-k}$ |
| Check | the indexed scalars $\xi_k$, $\Delta$ survive as coefficients |
| Result | |

---

### Verification checklist

- [x] `xi_k` can be declared as "part of the name" and `xi_k c'(k) c(k)` reaches the CAS
- [x] the display keeps the researcher's own subscript $\xi_k$ (no `xi(k)` leak)
- [x] normal ordering applies the anticommutator and the constant $\xi_k$ appears
- [x] the commutator $[H,c_k]=\xi_k c_k-\Delta c^\dagger_{-k}$ matches the hand calculation

---

## What this run found

**The one thing `param` cannot do is differentiation.**  `Differential(::QuExpr)` has no method on the
Julia side.  That is translated into words rather than left as a MethodError
(`error.julia_quexpr_not_differentiable` — redeclare without the index, or hand the scalar part to a
Maxima session).  The latter is the road taken in the
[S-c scenario](scenario_bcs_gap_equation_en.md).

**An undeclared indexed scalar is refused.**  A base with no declaration, like `V_i`, gets
`parser.unsupported_notation.julia_indexed_scalar` with the reason and the fix (declare `V_i`; the
subscript dialog's "part of the name").  **That is the design, not a defect** — as above, letting it
through turns a refusal into a silent wrong answer.

**Verified:** 2026-08-03 (real Julia, ST-1 through ST-4).

---

## Generator commands (this `scenario` block is the single source)

```scenario domain=condensed_matter
@note: Check that the most ordinary condensed-matter kinetic term $\xi_k c^\dagger_k c_k$ can be written. $\xi_k$ is one quantity ("the kinetic energy at $k$"), so its subscript is declared as part of the name, not as a component index.
c as the annihilation operator for electron
Define xi_k as a scalar
Define Delta as a scalar
@note: Write the kinetic term. An indexed scalar multiplying QA operators reaches the CAS as written.
xi_k c'(k) c(k)
@note: Normal order it. The expression is already ordered, so nothing changes — the point is that the display keeps the researcher's own subscript $\xi_k$ and no CAS syntax leaks out.
normal_order E1
@note: Now a Hamiltonian with the product deliberately reversed, plus a pair term to commute with later.
H = xi_k c(k) c'(k) + Delta c'(k) c'(-k)
@note: Normal order it. The anticommutator $c c^\dagger = 1 - c^\dagger c$ produces the constant term $\xi_k$ — evidence that the indexed scalar passes through the anticommutator intact.
normal_order E3
@note: Take the commutator $[H, c_k]$, the backbone of the equation of motion. By hand $[c^\dagger_k c_k, c_k]=-c_k$ and $[c^\dagger_k c^\dagger_{-k}, c_k]=-c^\dagger_{-k}$, so the answer should be $\xi_k c_k - \Delta c^\dagger_{-k}$.
commutator of E3 and c(k)
```
