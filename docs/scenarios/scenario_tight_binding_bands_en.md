# Practice-loop sample scenario: tight-binding band structure (dispersion relations)

> (2026-07-03: an English translation of the Japanese scenario, verified against the
> real Maxima backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_tight_binding_bands_jp.md](scenario_tight_binding_bands_jp.md) for the
> Japanese version.
>
> **One genuine i18n gap was found and fixed here**: the phrase "next-nearest-neighbor"
> (hyphenated, as in "including next-nearest-neighbor -t'") broke `_NNN_RE` in
> `ui/interpret_f1.py` — the regex alternative `next[\s-]*nearest` matched only
> "next-nearest," leaving the trailing "-neighbor" unconsumed, which then got
> mis-captured by the amplitude group, producing `-2neighbor` instead of `-2t'` in the
> dispersion result (a silent-wrong-answer bug, not a crash). Fixed by extending the
> alternative to `next[\s-]*nearest(?:[\s-]*neighbou?r)?` so an optional
> "-neighbor"/" neighbor" suffix is swallowed by the keyword match itself. The Japanese
> alternatives were not touched.)

**Physical theme:** a hopping model on a lattice → the Bloch matrix → dispersion
relations (chain, square, triangular, graphene, kagome, Lieb, SSH)
**Backend:** standard (Maxima) works fine. Dispersion is built deterministically in
closed form without going through the CAS (Julia is not needed)
**Time required:** about 15 minutes
**Verification goal:** confirm that F-1 stage 3c-fourier returns an **exact closed-form
band** (and, for N≥3, a secular equation) from a named lattice or an abstract crystal
descriptor.

---

## Physical background

For a translation-invariant tight-binding model $H=\sum_{ij}A_{ij}c^\dagger_i c_j$, the
bands are given by the eigenvalues of the Bloch matrix
$\tilde H(k)=\sum_{\Delta R}t\,e^{ik\cdot\Delta R}$, the Fourier transform of the unit
cell's hopping structure. For a single sublattice, $\varepsilon(k)=\sum_{\Delta
R}2t\cos(k\cdot\Delta R)$; for two sublattices (bipartite),
$\varepsilon_\pm=\tfrac{d_A+d_B}{2}\pm\sqrt{(\tfrac{d_A-d_B}{2})^2+|f|^2}$. This scenario
walks through this for various lattices. **Without going through commutators**, the
closed form is constructed directly from the hopping structure (avoiding Problem B,
self-contained).

---

## Session setup

- Mode: **standard (Maxima)** (the dispersion commands are backend-independent and
  bypass the CAS)
- Verification harness: `./scripts/debug_repl.sh --no-llm --lang en` (reproducible via
  rule-based dispatch)

---

## Input steps

### Step 1: register the hopping model (kept symbolic)

```
H = -t \sum_{\langle i j \rangle} c'_i c_j
```

**Expected output:** registered as `H` in the `\sum_{⟨ij⟩}` form as-is (not expanded).

### Step 2: the 1D chain's dispersion

```
Find the dispersion relation of H
```

**Expected output:** `\varepsilon(k) = -2t \cos k`

### Step 3: the square lattice

```
Find the dispersion relation of H on a square lattice
```

**Expected output:** `\varepsilon(\mathbf{k}) = -2t \left( \cos k_x + \cos k_y \right)`

### Step 4: the triangular lattice's t–t′ (a next-nearest shell)

```
Find the dispersion relation of H on a triangular lattice including next-nearest-neighbor -t'
```

**Expected output:**
`\varepsilon(\mathbf{k}) = -2t \left( \cos k_x + \cos k_y + \cos(k_x + k_y) \right) -2t' \left( \cos(k_x - k_y) + \cos(2 k_x + k_y) + \cos(k_x + 2 k_y) \right)`

**Verification point:** the second-nearest shell $\pm(a_1-a_2),\pm(2a_1+a_2),\pm(a_1+2a_2)$ is added.

### Step 5: honeycomb = graphene's two bands

```
Find the dispersion relation of H on a honeycomb lattice
```

**Expected output:**
`\varepsilon_\pm(\mathbf{k}) = \pm t \sqrt{ 3 + 2\cos k_1 + 2\cos k_2 + 2\cos(k_1 - k_2) }`

**Verification point:** at the $\Gamma$ point the value inside the square root is 9 →
$\varepsilon_\pm=\pm 3t$; gapless at the Dirac point.

### Step 6: opening a gap with a mass term (gapped Dirac)

```
Find the dispersion relation of H on a honeycomb lattice including a mass \Delta
```

**Expected output:**
`\varepsilon_\pm(\mathbf{k}) = \pm \sqrt{ \Delta^2 + t^2 \left( 3 + 2\cos k_1 + 2\cos k_2 + 2\cos(k_1 - k_2) \right) }`

**Verification point:** a staggered sublattice potential $\Delta$ opens a gap of
$2\Delta$ (the h-BN type).

### Step 7: the kagome lattice (a flat band + Dirac)

```
Find the dispersion relation of H on a kagome lattice
```

**Expected output:**
`\varepsilon_{\mathrm{flat}} = 2t, \quad \varepsilon_\pm(\mathbf{k}) = -t \left( 1 \pm \sqrt{ 3 + 2\cos k_x + 2\cos k_y + 2\cos(k_x + k_y) } \right)`

**Verification point:** a non-bipartite 3-sublattice lattice, but a closed-triangle
identity factors the cubic → a flat band $2t$ plus two dispersive bands. The flat band
touches the upper band at the $\Gamma$ point.

### Step 8: the Lieb lattice (a star-structure flat band)

```
Find the dispersion relation of H on a Lieb lattice
```

**Expected output:** `\varepsilon_0 = 0, \quad \varepsilon_\pm(\mathbf{k}) = \pm t \sqrt{ 4 + 2\cos k_x + 2\cos k_y }`

### Step 9: the SSH model (weighted bonds, self-contained)

```
```

**Expected output:** `\varepsilon_\pm(k) = \pm \sqrt{ v^2 + w^2 + 2 v w \cos k }`

**Verification point:** intracell $v$, intercell $w$. Saying "the SSH model with v=t_1,
w=t_2" also lets you override the amplitude symbols.

### Step 10: an abstract crystal descriptor (arbitrary bond input)

```
Find the dispersion relation of the crystal with the following bonds: (0,1,(0),v),(0,1,(-1),w)
```

**Expected output:** both the Bloch matrix and the bands
- `\tilde H(k) = \begin{pmatrix} 0 & v + w e^{-i k} \\ v + w e^{i k} & 0 \end{pmatrix}`
- `\varepsilon_\pm(k) = \pm \sqrt{ v^2 + w^2 + 2 v w \cos k }` (a special case of SSH)

**Verification point:** uniformly built from an arbitrary bond `(α,β,ΔR,amplitude)`. N≤2
gives a closed form.

### Step 11: an arbitrary N≥3 crystal (a secular equation)

```
Find the dispersion relation of the crystal with the following bonds: (0,1,(0),a),(1,2,(0),b),(0,2,(0),c)
```

**Expected output:** along with the 3×3 Bloch matrix,
`\det\!\left(\tilde H(k) - \varepsilon I\right) = 0`

**Verification point:** a general N×N has no closed form (Abel–Ruffini), so the
**secular equation** the bands satisfy is displayed analytically instead.

---

## Coverage (functionality verified by this scenario)

| Functionality | Example input | Implementation |
|---|---|---|
| keeping a symbolic bond sum | `H = -t \sum_{⟨ij⟩} …` | `register_bond_sum` |
| single-band dispersion / t–t′ | chain, square, triangular + next-nearest | `build_dispersion_single_band` |
| bipartite 2-band (graphene) | honeycomb | `build_dispersion_multiband` |
| mass gap / next-nearest diagonal | mass Δ / next-nearest t' | (same as above, onsite/nnn) |
| star flat band (Lieb) | Lieb lattice | `build_dispersion_nband` (star) |
| ring flat + Dirac (kagome) | kagome lattice | `build_dispersion_nband` (ring) |
| weighted bonds (SSH) | SSH model | `build_dispersion_ssh` |
| abstract crystal descriptor / secular equation | "the crystal with the following bonds…" | `build_crystal_bloch` |

**Related design document:** [05_algebraic_index_architecture.md](../developer_manuals/05_algebraic_index_architecture.md) (k-space). Dispersion construction lives in `notebook/runtime/backend_handler.py`.

**Verified:** 2026-07-03 (real Maxima backend, generator run, all 10 steps completed
end-to-end, same closed-form results as the Japanese version, including the
next-nearest-neighbor `-t'` fix confirmed in step 4).

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=quantum_mechanics
@note: Register the lattice hopping model $H=-t\sum_{\langle ij\rangle}c^\dagger_ic_j$ (hopping amplitude $t$ between neighboring sites) symbolically, without expanding the sum.
H = -t \sum_{\langle i j \rangle} c'_i c_j
@note: Find the dispersion relation $\varepsilon(k)$ (the eigenvalue of the Bloch matrix; for a single sublattice, simply the Fourier-transformed sum) treating this model as a 1D chain.
Find the dispersion relation of H
@note: Find the dispersion relation for the same hopping model on a square lattice.
Find the dispersion relation of H on a square lattice
@note: On a triangular lattice, find the dispersion relation including next-nearest-neighbor hopping $-t'$ in addition to nearest-neighbor hopping (adding the contribution of the second shell of neighbors).
Find the dispersion relation of H on a triangular lattice including next-nearest-neighbor -t'
@note: Find the dispersion relation on a honeycomb lattice (the graphene structure). With two sublattices, this should give two bands $\varepsilon_\pm(k)$ that are gapless at the Dirac point.
Find the dispersion relation of H on a honeycomb lattice
@note: Add a sublattice-staggered on-site energy (a mass term $\Delta$) to the honeycomb lattice and find the dispersion relation. This should open a gap of $2\Delta$ at the Dirac point (the h-BN type).
Find the dispersion relation of H on a honeycomb lattice including a mass \Delta
@note: Find the dispersion relation on a kagome lattice (3 sublattices). Although not bipartite, this should yield a flat band plus two dispersive bands.
Find the dispersion relation of H on a kagome lattice
@note: Find the dispersion relation on a Lieb lattice (a flat band is expected from the star structure).
Find the dispersion relation of H on a Lieb lattice
@note: Without using a named lattice, find the dispersion relation of a crystal specified directly by an abstract list of bonds (sublattice pair, lattice-vector displacement, amplitude). This should be a special case with the same structure as the SSH model.
Find the dispersion relation of the crystal with the following bonds: (0,1,(0),v),(0,1,(-1),w)
```
