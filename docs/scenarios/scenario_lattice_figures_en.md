# Practice-loop sample scenario: rendering lattice figures and exporting to TeX (S-k)

> (2026-07-03: an English translation of the Japanese scenario, verified against the
> real Maxima backend through the same rule-based, no-LLM generator pipeline. See
> [scenario_lattice_figures_jp.md](scenario_lattice_figures_jp.md) for the Japanese
> version.
>
> **One genuine i18n gap was found and fixed here** (not present in the earlier
> translations): the `expand_neighbor_sum` (F-1) trigger phrase "の最近接和を...展開して"
> had **no English equivalent at all** — `_NS_TRIGGER_RE` and `_NS_BODY_RE` in
> `ui/interpret_f1.py` only recognized the Japanese body-then-suffix word order ("<body>
> の最近接和"). English reverses this ("the nearest-neighbor sum **of** <body> over a 3x3
> honeycomb lattice" — the sum-type noun comes first, body follows "of"), so a purely
> additive fix was needed: `_NS_TRIGGER_RE` gained an English alternative
> (`nearest[\s-]*neighbor(?:\s*-?\s*sum)?|lattice\s+sum`), and a new
> `_NS_BODY_OF_EN_RE` pattern extracts the body after "sum of" up to the first
> `over`/`on`/`for` boundary marker. Both the 2D-lattice-keyword branch and the 1D
> nearest-neighbor/all-pairs branch of `_parse_neighbor_sum` now fall back to this
> English extraction when the Japanese `の` marker is absent. The Japanese patterns were
> not touched.)

Detail on **S-k (visualization)** in the practice-loop scenario menu. When concretizing
a lattice sum (F-1), **draw its bond structure as a figure** to confirm "is this the
lattice I intended?", and embed it directly into the paper.

**Backend:** standard (**Maxima**) / **Domain:** statistical_mechanics.
**Lattice-figure generation does not go through the CAS** (pure-Python TikZ/SVG/PDF —
[lattice_figure.py](../../notebook/runtime/lattice_figure.py)), so the figure itself is
backend-independent (a Julia session produces the identical figure). Maxima is
specified here only because this scenario is placed under a Maxima-family domain per
the menu convention for scenarios that don't need QA operators.

## Background

When `expand_neighbor_sum` (F-1) concretizes a lattice sum $\sum_{\langle ij\rangle}$, it
obtains site coordinates and bonds from an adjacency list (a bond generator). This
scenario uses the same generator to **turn the bond structure into a figure**:

- `lattice_to_tikz` — a `tikzpicture` for papers (automatically inserted into the TeX export).
- `lattice_to_svg` — a standalone SVG for browsers/image viewers.
- A **wrap-around bond** from periodic boundaries (an edge longer than 1.5× the unit
  spacing) is drawn as a **dashed line**, making the boundary condition visible at a
  glance.

## Input steps

### Step 1: nearest-neighbor sum over a 3×3 honeycomb lattice

```
Expand the nearest-neighbor sum of Sz_i Sz_j over a 3x3 honeycomb lattice
→ E1 = Sz(3)Sz(4) + Sz(1)Sz(3) + …  (9 sites, 9 bonds)
```

`expand_neighbor_sum` (pair_constraint=`honeycomb_2d`, lattice_shape=`3x3`, boundary=`open`).

### Step 2: save the lattice figure (standalone)

```
Save the lattice figure
→ <session>_lattice.svg + <session>_lattice.tex (standalone)
  + <session>_lattice.pdf (if latexmk/pdflatex/lualatex is available)
```

Writes out the figure for the most recent lattice sum (the honeycomb from step 1) into
`sessions/`. The PDF is produced only when a LaTeX engine is on PATH. The SVG can be
checked immediately, and the standalone TeX compiles on its own.

### Step 3: periodic boundary on a 3×3 square lattice (dashed wrap-around)

```
Expand the nearest-neighbor sum of Sz_i Sz_j over a 3x3 square lattice with periodic boundary conditions
→ E2 = …  (9 sites, 18 bonds, 6 of which are wrap-around → dashed)
```

`boundary=periodic`. The edge↔edge bonds arising from the periodic boundary are drawn
as **dashed lines** in the figure.

### Step 4: TeX export (figures embedded automatically)

When the session is exported to TeX, **the lattice figure** for each
`expand_neighbor_sum` step (a `tikzpicture`) is inserted **immediately after** it. In
this scenario, the two figures for E1 (honeycomb) and E2 (square, periodic) each appear
next to their respective expanded Hamiltonian. The preamble includes
`\usepackage{tikz}` (dvipdfmx) and compiles with `uplatex && dvipdfmx`.

## Coverage

- **Lattice-figure generation** (`lattice_layout`/`lattice_to_tikz`/`lattice_to_svg`) — honeycomb, square.
- **Boundary-condition visualization** — periodic wrap-around bonds shown as dashed lines.
- **Standalone save** (SVG / standalone TeX / PDF) and **automatic embedding into the TeX export**.
- **Bypasses the CAS = backend-independent** (the same figure regardless of Maxima/Julia).

## Related

- Real example: `domains/statistical_mechanics/examples/scenario_lattice_figures.{json,tex}`
  + `scenario_lattice_figures_lattice.svg` (the standalone honeycomb figure).
- Physical use of lattice sums:
  [scenario_ising_partition_function.md](scenario_ising_partition_function_jp.md) (S-g),
  [scenario_tight_binding_bands.md](scenario_tight_binding_bands_jp.md) (dispersion relation).
- Implementation and open items: [open_items.md](../../KNOWN_ISSUES.md) §F-1 (lattice-figure version).

**Verified:** 2026-07-03 (real Maxima backend, generator run, all 3 steps completed
end-to-end, same 9-site/9-bond and 9-site/18-bond structure as the Japanese version).

## generator commands (this `scenario` block is the single source used to generate the example)

```scenario domain=statistical_mechanics
@note: On a $3\times3$ honeycomb lattice (open boundaries), expand the sum $\sum_{\langle ij\rangle}S_i^zS_j^z$ over nearest-neighbor site pairs $\langle ij\rangle$ into the actual bonds (adjacent pairs).
Expand the nearest-neighbor sum of Sz_i Sz_j over a 3x3 honeycomb lattice
@note: Save the bond structure of the lattice sum just computed as a figure (SVG and standalone TeX, plus PDF if available). The figure lets you confirm that the intended lattice and boundary conditions were used.
Save the lattice figure
@note: Now expand the same nearest-neighbor sum on a $3\times3$ square lattice, this time under periodic boundary conditions. The bonds that wrap around due to the periodic boundary are drawn as dashed lines in the figure to distinguish them.
Expand the nearest-neighbor sum of Sz_i Sz_j over a 3x3 square lattice with periodic boundary conditions
```
