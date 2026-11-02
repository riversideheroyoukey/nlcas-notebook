# Declarative Cartesian plots

`presentation.kind = "cartesian"` selects deterministic, domain-independent
plotting. The Adviser proposes a `NumericSpec`; the researcher reviews its
axes, layers and expression references before execution. No model-written
Python is needed for this vocabulary. Other numeric intents retain their
existing code-generation and review paths.

## Layers and input meaning

- `line`: a registered scalar value y=f(x). One axis varies.
- `contour`: level sets of a registered scalar expression. An equality is
  lowered as lhs-rhs; level zero draws its boundary.
- `region`: intersection of registered strict or non-strict inequalities.
  Separate layers represent alternative constraint sets. Relations are
  preserved, including their direction and strictness.
- `point`: two real scalar coordinates, literal numbers or references to
  computed expressions. Plotting does not solve or certify an optimum.

Each x/y axis names a `sweep` input with explicit finite min/max/points.
Other parameters are `fixed` inputs. Per-layer `fixed_values` overrides allow
parameter comparisons without copying a formula or changing its definition.
Parameters outside the axis set must be bound; point coordinates and contour
levels must be scalar after parameter substitution, independent of plot axes.

`presentation.options` declares `x`, optional `y`, `layers`, optional
`title/xlabel/ylabel`, `formats` (PNG/SVG/PDF), `dpi`, and `legend`.
Layers accept `label`, `color`, `linestyle`, `alpha`, `fixed_values`; points
also accept `marker` and `markersize`. Unknown options are rejected rather
than silently ignored. There is no raw Python, eval, filesystem path, or
arbitrary matplotlib-argument escape hatch in this spec.

Example (expression numbers are placeholders for registered expressions):

```json
{
  "intent": {"kind": "cartesian_plot", "goal": "Compare two parameter settings"},
  "inputs": [
    {"name": "x", "role": "sweep", "spec": {"min": -2, "max": 2, "points": 201}},
    {"name": "k", "role": "fixed", "spec": {"value": 1}}
  ],
  "outputs": [{"name": "comparison", "shape": "series", "over": ["x"]}],
  "method": {"name": "direct_eval"},
  "presentation": {"kind": "cartesian", "options": {
    "x": "x", "layers": [
      {"kind": "line", "expr_id": "E1", "label": "k=1"},
      {"kind": "line", "expr_id": "E1", "label": "k=3", "fixed_values": {"k": 3}}
    ], "formats": ["png", "svg", "pdf"]
  }}
}
```

## Validation and reproducibility

`plot_spec.py` validates bounded grids, layer shapes and references.
`plot_codegen.py` lowers every referenced expression through the existing
scalar numpy whitelist, then substitutes axis arguments and finite parameter
literals in the generated AST. The generated standalone renderer runs in the
existing numeric sandbox. Unsupported constructs fail explicitly; they are
never passed through as raw code.

All referenced expressions are frozen into `NumericTask.plot_sources` at
fork time, including canonical/LaTeX and source branch. They become graph
source edges and are included in stale detection and persistence. Regenerating
a task uses these snapshots even if the notebook expressions later change.
The LLM fallback now receives the full faceted spec and these snapshots as
well, rather than losing method/presentation details in flat projection.

PNG previews and SVG/PDF exports are accompanied by per-layer CSV files and
`plot.json` diagnostics. Downloads are session-confined by resolved path,
including symlinks, and limited to known product suffixes. File links survive
reload through the timeline serializer.

## Numerical limits

Regions are **sampled**, including nonlinear and disconnected sets; they are
not exact polygons. Their edges have grid discretization error. View bounds
clip the displayed domain and are not extra mathematical constraints.
Strict inequalities remain strict in the exported sample mask; a raster image
alone cannot distinguish an included boundary from an excluded one. Compare
CSV masks or the original inequalities when this distinction matters.

All-invalid domains fail; partially undefined values stay transparent.
Complex-valued results are refused unless the researcher first chooses an
explicit real quantity. Empty sampled regions and levels without crossings
are recorded as diagnostics, not invented boundaries. A region narrower than
the grid can be missed; increase the explicit resolution to investigate it.
The budget is two million samples across all layers, at most 2000 samples per
axis. This facility is not a symbolic geometry prover or optimizer.

## Adviser path

The spec proposer receives the current expression catalog, including later
computations, rather than only history preceding the selected expression.
Its response budget supports layered specs. Proposal parsing and connection
failures are distinct stages, with an error reason and a bounded response
excerpt available from the API. A parse failure never silently falls back to
an unrelated single-value calculation. The existing approval step remains.

## Complete two-axis example

Register the following expressions in a **new** Maxima notebook; in an existing
notebook, replace the example IDs with the IDs actually assigned. Register
scalar expressions, not named assignments, for the last three entries.

| ID in this example | Canonical input | Purpose |
|---|---|---|
| E1 | `x^2+y^2<=r^2` | Outer disk |
| E2 | `x^2+y^2>=1/4` | Exclude the central disk |
| E3 | `x^2+y^2=1` | Outer boundary |
| E4 | `x^2+y^2` | Scalar field |
| E5 | `1/4` | Computed or registered scalar contour level |
| E6 | `0` | Reference x coordinate |
| E7 | `1/2` | Reference y coordinate |

This spec overlays an annulus, a larger alternative region, an equality
boundary, a scalar-field level and a reference point. It demonstrates
nonlinear constraints independently of the production-planning example.

```json
{
  "intent": {"kind": "cartesian_plot", "goal": "Compare two annular regions"},
  "inputs": [
    {"name": "x", "role": "sweep", "spec": {"min": -2, "max": 2, "points": 201}},
    {"name": "y", "role": "sweep", "spec": {"min": -2, "max": 2, "points": 201}},
    {"name": "r", "role": "fixed", "spec": {"value": 1}}
  ],
  "outputs": [{"name": "regions", "shape": "field", "over": ["x", "y"]}],
  "method": {"name": "direct_eval"},
  "presentation": {"kind": "cartesian", "options": {
    "x": "x", "y": "y", "xlabel": "x", "ylabel": "y",
    "title": "Two annular regions", "legend": true, "dpi": 200,
    "formats": ["png", "svg", "pdf"],
    "layers": [
      {"kind": "region", "constraints": ["E1", "E2"], "label": "r=1", "color": "blue", "alpha": 0.25},
      {"kind": "region", "constraints": ["E1", "E2"], "label": "r=1.5", "color": "green", "alpha": 0.2, "fixed_values": {"r": 1.5}},
      {"kind": "contour", "expr_id": "E3", "levels": [0], "label": "outer boundary", "color": "blue"},
      {"kind": "contour", "expr_id": "E4", "levels": [{"expr_id": "E5"}], "label": "inner boundary", "linestyle": "--"},
      {"kind": "point", "x": {"expr_id": "E6"}, "y": {"expr_id": "E7"}, "label": "reference", "marker": "*", "markersize": 12}
    ]
  }}
}
```

A point's coordinates and each contour level accept a finite number or an
object such as `{"expr_id":"E5"}`. A bare string such as `"E5"` is invalid
there. By contrast, a layer's `expr_id` and a region's `constraints` contain
ID strings. An equality in a contour is evaluated as lhs-rhs, so its boundary
has level **zero**, regardless of the constant on the original right side.
A `line` requires a scalar expression: extract/register the RHS first if the
notebook entry is an equation. A region accepts only `<`, `<=`, `>`, `>=`.
Nonnegativity must be an explicit registered inequality if it is intended as
a mathematical constraint; a nonnegative view bound only clips the picture.

## Supported controls and validation bounds

| Field | Contract |
|---|---|
| `inputs` | Unique names; `sweep` or `fixed` roles only. Sweep names exactly match the plot axes |
| Sweep `spec` | Finite `min < max`; integer `points` from 2 to 2000; uniformly spaced linear samples |
| `method.name` | Empty or `direct_eval`; a plot does not execute a solver |
| `layers` | 1–32; a region has 1–32 inequality references; a contour has 1–32 levels |
| Total grid budget | Product of axis point counts × number of layers ≤ 2,000,000 (counts every layer conservatively) |
| `fixed_values` | Finite per-layer parameter overrides; cannot override an axis; unused parameters are rejected |
| `formats` | Nonempty unique subset of `png`, `svg`, `pdf`; default `png,svg`; PNG added for preview |
| `title`, `xlabel`, `ylabel`, layer `label` | At most 300 characters each |
| `color`, `linestyle` | Matplotlib color value (max 40 characters); `-`, `--`, `:`, `-.` |
| `alpha` | Region fill opacity, greater than 0 and at most 1; default 0.25 |
| Point `marker`, `markersize` | `o`, `s`, `*`, `x`, `+`, `^`, `D`, `.`; size greater than 0 and at most 40 |
| `dpi`, `legend` | 72–600 (default 200); boolean (default true) |

Contour/region/point layers require two axes. Lines vary only the x axis.
Region constraints are intersected within each layer; alternative scenarios
are separate layers. There is no dedicated union/difference operator.
Axis units are scaled independently to fit the figure (automatic aspect),
so a mathematical circle may look elliptical; equal-aspect control is not
currently exposed. Use axis coordinates/CSV for geometric measurements.
The supported presentation options are a closed vocabulary: logarithmic
axes, 3-D surfaces, animations, arbitrary Python and solver calls are not
implemented by this renderer. Other numeric specifications retain their
existing routes; an unsupported plot is never silently treated as scalar
evaluation. Cartesian syntax can validate while expression lowering still
fails, for example for an unsupported function or an unbound parameter.

## UI and API sequence

1. Register the source expressions, compute any special point/level in the
   theory or numeric layer, and save the notebook session.
2. Select an expression and open **💬 対話で仕様を確定 / Set up via dialogue**.
   State reference IDs, axes/ranges/resolution, parameter settings and formats.
3. Review/refine the proposed spec; check every layer reference and coordinate.
   Approve the spec, generate the deterministic code, then run it. A valid
   Cartesian spec does not require model-written code or manual code edits.
4. Inspect the figure and diagnostics; download the products and save the
   session again to retain task/execution provenance.

The existing **🔢 To numeric** form is unchanged. The header's **🕸 Numeric
graph** visualizes task dependencies, not Cartesian function data.

All paths below have prefix `/api/notebook`. Use the complete JSON examples
above as the `spec` object, replacing placeholder IDs first.

| Request | Body / response |
|---|---|
| `POST /numeric/propose_spec` | `{"expr_id":"E1","goal":"..."}`; refinement adds `prior_spec` and `refinement` |
| `POST /numeric/fork` | `{"expr_id":"E1","spec":{...}}`; returns task and code-generation route |
| `POST /numeric/generate/{task_id}` | Generate from the task's full spec and source snapshots; metadata source is `cartesian_template` |
| `POST /numeric/run/{task_id}` | Execute; response includes `downloads:[{"artifact_id":"...","name":"figure.svg"},...]` |
| `GET /numeric/file/{artifact_id}` | Download a known current-session product |
| `GET /numeric/figure/{artifact_id}` | Existing PNG preview |

Do not infer failure ownership from “specification reading failed”: the
Adviser supplies a reply and the application's parser checks it. On a failed
proposal, `stage:"adviser"` reports a call error, while `stage:"parse"`
reports an unusable spec response. `detail` and `response_excerpt` (at most
2000 characters) support diagnosis. An unavailable Adviser is reported
separately. The API does not guarantee automatic repair. One spec candidate
is accepted; multiple candidates require refinement. Point/level references
must use the object form shown above even if an Adviser suggests a bare ID.

## Products, provenance and implementation map

| Product | Content / interpretation |
|---|---|
| `figure.png` | Always available as the timeline preview |
| `figure.svg`, `figure.pdf` | When selected in `formats`; publication exports use the same renderer |
| `layer_NN.csv` for a line | `x,value` |
| `layer_NN.csv` for a contour | `x,y,value` of the scalar field; not traced contour vertices |
| `layer_NN.csv` for a region | `x,y,inside,valid`; inside/valid are 0/1 sample masks |
| `layer_NN.csv` for a point | `x,y` |
| `plot.json` | `summary` (target, layer count, grid) and `diagnostics` list |

`plot.json` warning entries use zero-based layer indices; the timeline
message displays one-based layer numbers. `empty_sampled_region` is not a
proof that the mathematical feasible set is empty. `levels_without_crossing`
means at least one level did not lie strictly between the sampled finite
minimum and maximum; an extremal tangent level may therefore be omitted.
The actual geometry also depends on the grid and invalid-value mask.

Each layer CSV is a typed `series` manifest artifact. Current artifact
categories are PNG=`figure`, CSV=`table`, SVG/PDF/JSON=`log`; these file formats
do not add new artifact types. Download paths are resolved beneath the
session's numeric output root, including symlink checks, and only known
PNG/SVG/PDF/CSV/JSON products are served. Product files remain beside the
saved session under `<stem>_numeric/<task>/<execution>/`; moving only the
session JSON does not carry those files with it.

| Component | Responsibility |
|---|---|
| `notebook/numeric/plot_spec.py` | Typed options, reference collection, source snapshots, diagnostics text |
| `notebook/numeric/plot_codegen.py` | Whitelist lowering and bound AST substitution; standalone renderer generation |
| `notebook/numeric/plot_runtime.py` | Sampling, drawing, CSV and manifest/diagnostics emission |
| `notebook/numeric/plot_prompt.py` | Shared declarative vocabulary for the Adviser |
| `intent_registry.py`, `templates.py`, `spec_projection.py`, `spec_validate.py` in `notebook/numeric/` | Route, dispatch, compatibility projection and validation |
| `notebook/engine_numeric_mixin.py`, `notebook/models/numeric.py` | Task provenance, persistence and stale detection |
| `ui/numeric_products.py`, `ui/routes/numeric.py`, `ui/timeline_serializer.py`, `ui/notebook_numeric.js` | Confined downloads, proposal/API flow, reload and UI |
| `tests/test_cartesian_plot.py` | Nonlinear holes, parameter comparisons, scalar references, invalid domains, snapshots/stale, API and download boundaries |

The task model accepts older saves without `plot_sources` (default empty).
This compatibility does not retroactively supply snapshots for old plots.
Function domains and strict comparison masks are numerical; the renderer
refuses imaginary parts above its 1e-12 tolerance instead of silently dropping
them. It does not infer or validate optimality from a marker label.

The post-load partial-substitution observation from the production-planning
workflow is tracked as **unresolved** in [N-1 open items](../../KNOWN_ISSUES.md).
Its cause is unconfirmed; plotting support does not establish correctness of
that separate CAS operation after replay.
