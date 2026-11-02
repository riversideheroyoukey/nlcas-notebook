# Practice-loop scenario: the first-order perturbation term for spin-ful fermions

> Run 2026-08-09.  Write a **spin-ful general Hamiltonian** (arbitrary basis,
> spin-dependent) and take it through the commutator with the interaction — the
> shape that appears at first order in a perturbation expansion.
> The second half of a pair; the first is
> [the matrix-element scenario](scenario_spinful_matrix_elements_en.md).

**Backend:** Julia (QuantumAlgebra.jl) / **Domain:** condensed matter.

### Why this is worth checking

With the coefficients defined in the first half, the Hamiltonian reads

$$\hat H_0=\sum_{m\sigma,n\sigma'} h_{m\sigma,n\sigma'}\,\hat c^\dagger_{m\sigma}\hat c_{n\sigma'},
\qquad
\hat V=\sum V_{m\sigma_1,n\sigma_2,p\sigma_3,q\sigma_4}\,
\hat c^\dagger_{m\sigma_1}\hat c^\dagger_{n\sigma_2}\hat c_{p\sigma_3}\hat c_{q\sigma_4}$$

— the general form over an **arbitrary, spin-dependent** basis, not specialised to
$k$-space.  First order needs $[\hat H_0,\hat V]$, and every contraction there
introduces a **new bound index**.

That is where the display is decided, and two things were broken until 2026-08-08:

- indices came back **with no separator**, as $\sum_{mnpq}$ and $V_{mnpq}$.  By
  this project's own rule that is one index named $mnpq$ — so **typing the display
  back in gives a different expression**
- an index introduced by a contraction was given a **neutral name like $k_{5}$**,
  which standing beside $m,n,p,q$ is not a line anyone would publish

Go up in order and the factors multiply and the indices multiply with them.
**Whether that TeX can go straight into a paper** is what this scenario asks.

### The physics

1. Define the electron annihilation operator $\hat c$
2. Write the one-body term $\hat H_0$ and normal-order it (already normal-ordered, so unchanged)
3. Write the interaction $\hat V$ and normal-order it
4. Take $[\hat H_0,\hat V]$ → the first-order term

The structure of 4 can be followed by hand:
$[\hat c^\dagger_a\hat c_b,\ \hat c^\dagger_c\hat c^\dagger_d\hat c_e\hat c_f]$ gives **four
terms** — from contracting $\hat c_b$ against $\hat c^\dagger_c,\hat c^\dagger_d$ and
$\hat c^\dagger_a$ against $\hat c_e,\hat c_f$.  Each term keeps four operators, and the
index consumed by the contraction is replaced by **one new bound pair (mode and
spin)**.

---

### Steps

#### ST-1: session and declarations

| Item | Content |
|---|---|
| Action | New session, domain "condensed matter" → the Julia backend is selected automatically |
| Input | `c as the annihilation operator for electron` |
| Result | |

---

#### ST-2: the one-body term (spin-dependent, arbitrary basis)

| Item | Content |
|---|---|
| Input | `\sum_{m,s,n,t} h_{m,s,n,t} c'_{m,s} c_{n,t}` |
| Input | `normal_order E1` |
| Expected | $\sum_{m,s,n,t} h_{m,s,n,t}\,c^\dagger_{m,s}c_{n,t}$ (already normal-ordered, so unchanged) |
| Check | Mode ($m,n$) and spin ($s,t$) are shown SEPARATED |
| Result | |

---

#### ST-3: the interaction (eight indices)

| Item | Content |
|---|---|
| Input | `\sum_{m,s,n,t,p,u,q,w} V_{m,s,n,t,p,u,q,w} c'_{m,s} c'_{n,t} c_{p,u} c_{q,w}` |
| Input | `normal_order E3` |
| Expected | All eight indices (four mode, four spin) preserved |
| Check | Eight indices survive and their order is not permuted |
| Result | |

---

#### ST-4: the first-order term $[\hat H_0,\hat V]$

| Item | Content |
|---|---|
| Input | `commutator of E1 and E3` |
| Expected | Four terms, each with $V$, $h$, four operators and one new bound pair |
| Check | The new indices are **$m_1,s_1$** — the researcher's own letters plus a subscript — not a foreign series like `k_{5}` |
| Check | The $\sum$ subscript and the coefficient indices are SEPARATED, i.e. this TeX types back in as the same expression |
| Result | |

---

### Verification checklist

- [x] A spin-dependent, arbitrary-basis general Hamiltonian can be written
- [x] The eight-index interaction survives normal ordering
- [x] The first-order term $[\hat H_0,\hat V]$ comes out as four terms
- [x] New bound indices are the researcher's letters plus a subscript ($m_1,s_1$) and spend no plain letter
- [x] $\sum$ and coefficient indices are separated, so the display types back in

---

## Behaviour confirmed / notes

**A bound index spends no letter.**  It is a throwaway label living inside one
sum, yet taking $r$, then $s$, then $t$ for successive contractions shrinks the
alphabet the researcher still has for actual quantities.  A declared $r$ — a
radius — really was taken by a dummy (measured 2026-08-08).  So the extras are
**the researcher's own letters, cycled and subscripted** ($m_1,s_1,n_1,t_1,\dots$):
a mode next to a mode, a spin next to a spin, which is also how it reads.

**Which letters are "free" is not something the display layer can know.**  The
ones missing from THIS expression are missing by accident, and researchers use
$i$, $j$, $k$ as ordinary indices everywhere else.  A subscript removes the need
to decide.

**The expansion of the field operator itself cannot be written here.**
$\hat\psi_\sigma(\mathbf x)=\sum_m \varphi_{m\sigma}(\mathbf x)\hat c_{m\sigma}$ multiplies a
basis function (a Maxima-side function) by a QA operator, and in a Julia session
$\varphi_{m,s}(x)$ is refused with a reason
(`parser.unsupported_notation.julia_applied_indexed` — a QuantumAlgebra operator
takes indices, but the result cannot be called).  **The backend is fixed per
session** (Core Principle #6), so the basis side belongs to
[the first half](scenario_spinful_matrix_elements_en.md).

**Confirmed:** 2026-08-09 (live Julia, ST-1 through ST-4 completed).

---

## generator commands (this `scenario` block is the single source of the generated example)

```scenario domain=condensed_matter
@note: Using the coefficients defined in the first half, write the general Hamiltonian over an **arbitrary, spin-dependent** basis — not specialised to $k$-space — and take it to first order.
c as the annihilation operator for electron
@note: The one-body term $\hat H_0=\sum_{m\sigma,n\sigma'} h_{m\sigma,n\sigma'} c^\dagger_{m\sigma}c_{n\sigma'}$, with mode ($m,n$) and spin ($s,t$) written separated.
\sum_{m,s,n,t} h_{m,s,n,t} c'_{m,s} c_{n,t}
@note: Normal-order it. It is already normal-ordered so the expression does not change, but mode and spin must stay separated in the display — without the separator it becomes "one index called $ms$", and typing the display back in gives a different expression.
normal_order E1
@note: The interaction $\hat V$: eight indices, four mode and four spin.
\sum_{m,s,n,t,p,u,q,w} V_{m,s,n,t,p,u,q,w} c'_{m,s} c'_{n,t} c_{p,u} c_{q,w}
normal_order E3
@note: The first-order term $[\hat H_0,\hat V]$. Every contraction introduces a new bound index — watch it come out as the researcher's own letter plus a subscript ($m_1,s_1$). Spending the plain $r$, $s$, $t$ would shrink the alphabet left for actual quantities, for labels that live inside one sum.
commutator of E1 and E3
```
