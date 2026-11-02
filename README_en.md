# Digital Research Notebook for Theorists, with CAS Support

Public project name: **NLCAS Notebook** (`nlcas-notebook`), the Research Notebook
implementation associated with APIEMS2026. Its successor is
[Out-of-Order CAS (OOCAS)](https://github.com/riversideheroyoukey/oocas);
access depends on that repository's visibility and permissions.
See [publication preparation](docs/github_publication.md) and
[third-party notices](THIRD_PARTY_NOTICES.md). The original code's license is not yet selected.

A system that lets theoretical researchers operate a computational notebook with
natural language and handwritten mathematics — without having to learn CAS syntax.

**Core features:**
- **Equation-form input (H-1)**: register an expression in `LHS = RHS` form, e.g.
  `H = J * x * y`; `H` then becomes a reusable symbol
- **Expression-ID references (H-2)**: use expression IDs inside a formula, e.g.
  `E1 * E4 + E3`, expanded automatically
- **P1 auto-update**: when a symbol is defined, the undefined-symbol lists of
  existing expressions are refreshed automatically
- **P2 auto-substitution**: Maxima receives `H : RHS`; Julia requires an explicit
  substitute
- **Equation-operation series (H-5–H-9)**: follow-up operations on a registered
  equation (rule-based, no LLM needed)
  - **H-5** evaluate the RHS: `compute the RHS of E3`
  - **H-5z** set the RHS to a value: `set the RHS of E3 to 0` → registers a new
    equation `RHS = 0`
  - **H-7** operate on the RHS: `expand / simplify / factor the RHS of E3`
  - **H-8** operate on both sides: `divide both sides of E3 by m`
  - **H-9** solve an equation: `solve E3 for x` / simultaneous equations supported
- **Multi-step solution plans (C-Ext-5)**: fixed workflows (e.g. "find the
  eigenstates of E1"), Adviser-proposed candidates ("solution candidates"),
  free-form planning ("plan this"), and Supervisor auto-selection
  ("auto-select the best") — all gated by a confirmation step before
  execution. Failure stops safely with partial results kept; a stopped plan
  card offers 🔁 Retry and 🔄 Replan (see
  [user_guide.md](docs/user_guide.md))
- **General plotting (numeric layer)**: configure overlaid curves, contours,
  inequality regions and reference points through dialogue. A common renderer
  generates PNG/SVG/PDF and sampled CSV data from expression references and a
  declarative spec ([worked examples](docs/scenarios/scenario_cartesian_plots.md),
  [specification](docs/developer_manuals/36_cartesian_plot_layers.md)).
- Handwriting input → OCR → notebook with expression IDs
- Natural-language instruction → LLM supervisor → common operation → CAS
  (Maxima / Julia) → result
- History is a replayable DAG (branch / resume / fork)
- CAS syntax is normally hidden from the UI

**Features specific to Julia sessions (second quantization):**
- `operator_substitute`: replace a quantum operator by a linear combination
  (Bogoliubov transformation, etc.)
- `pair_substitute`: operator pair → c-number substitution (number density
  `c†(k)c(k)→n_k`, BCS gap, etc.)
- `wick_contract`: Wick contraction (pull a chosen pair out of an n-th-order
  operator product and replace it by a c-number, leaving the (n−2)-th product)
  - HF/BCS decoupling via 4-index contraction values `ρ(k,σ,p,τ)` / `κ(k,σ,p,τ)`
  - `declare_symmetry`: declare antisymmetry of κ such as `κ(j,i)=-κ(i,j)`
- `re_evaluate_expr`: update parameters of a contracted expression and re-evaluate
  (self-consistent iteration)
- `extract_order`: extract the λⁿ term from a scalar/operator polynomial
  (e.g. `H=H₀+λV` → n=1 → `V`)
- `truncate_series`: drop terms of order ≥ n (scalar and operator)
- `operator_taylor_expand`: `f(H) ≈ Σ aₖ Hᵏ` — apply an analytic function to an
  operator (high-temperature expansion, resolvent, etc.)
- 2-index operators `c(k, σ)`: definition and substitution of operators carrying a
  spin/polarization index
- automatic spin-notation normalization: `↑→u`, `↓→d`, `σ→s`
  (QuantumAlgebra.jl QuIndex)
- domain declaration of scalar parameters: `@variables X::Real` / `@variables X::Integer`
- automatic imaginary-unit conversion: an undeclared `i` → Julia `im` (declared
  variables and bound indices are left untouched)
- Heisenberg equation: input in the form `Heisenberg equation of a_1, E1`

**Input assistance:**
- Unicode Greek letters (`ω`, `α`, `Δ`, …) are converted automatically to TeX
  notation (`\omega`, …)
- confirmation dialog for subscripted symbol declarations: `declare xi_k as a scalar`
  → choose "component index" or "part of the name"

**Analytical mechanics (independent-atom model):**
- in the `analytical_mechanics` domain the coordinate $q$ and velocity $\dot q$ are
  treated as **independent variables** (as the Euler-Lagrange formulation requires).
  Entering only `\dot{q}` (no CAS syntax) produces $\partial L/\partial\dot q$,
  $d/dt$ (total time derivative) and the equations of motion cleanly
- a separate `classical_mechanics` domain is provided for Newtonian mechanics
  (dependent $\dot q=dq/dt$) (design = [docs/19](docs/developer_manuals/19_analytical_mechanics_independent_variables.md))

**F-1: multi-site lattice Hamiltonians (pure Python, no CAS):**
- **lattice sums**: `Σ_{<ij>}` both finitely expanded (concretized) and kept
  symbolic. Pluggable generators (chain, square, triangular, honeycomb, kagome,
  Lieb, arbitrary graph; open/periodic/cylinder boundaries)
- **lattice figures**: render the bond structure as TikZ/SVG/PDF and embed it
  automatically into the TeX export (periodic boundaries dashed)
- **analytic band structure**: tight-binding dispersion ε(k) (single/multi-band,
  t–t′, mass term, SSH, abstract crystal descriptor)
- **synthetic spin algebra**: nested commutators (EOM / moment hierarchy),
  infinite-temperature moments, continued fraction → spectral function S(ω),
  high-temperature expansion of the specific heat, general spin S

**L-1: general linear basis transformations:**
- generate the inverse `old = M⁻¹ new` automatically from N forward transforms
  (Bogoliubov, spin raising/lowering, coordinate change, etc.)
- derive / apply / verify the canonical/metric constraint $M^\dagger G M = G$
  (para-unitarity), for both Maxima and Julia (design = [docs/18](docs/developer_manuals/18_linear_transform_design.md))

---

## Requirements

- Python 3.14 (inside `.venv/`) — pinned in `requirements.txt` / `requirements-numeric.txt`
- Maxima 5.49 (the `maxima` command on PATH)
- Julia 1.12 (the `julia` command on PATH, for second-quantization sessions) —
  packages pinned in `Project.toml` / `Manifest.toml`
- Ollama (for the LLM supervisor; the app also starts without it)

---

## Install & Quick start

**macOS (recommended — one shot):**

```bash
./setup.sh          # installs system tools + both venvs + the pinned Julia env
./scripts/start.sh  # → http://localhost:8000/notebook.html
```

**Linux (Docker; Ollama runs host-native):**

```bash
docker compose up --build   # → http://localhost:8000/notebook.html
```

**Windows (native — recommended, verified on real hardware, 2026-07):**

```powershell
.\setup_windows.ps1
.\scripts\start_windows.ps1   # → http://localhost:8000/notebook.html
```

**Windows (Docker Desktop — only if you specifically need Docker; unverified, low priority):**

```powershell
docker compose -f docker-compose.windows.yml up --build   # → http://localhost:8000/notebook.html
```

Full instructions — including host-native Ollama install/connection for Linux
and Windows, and what `docker-compose.windows.yml` overrides and why — are in
**[docs/setup_guide_en.md](docs/setup_guide_en.md)**.
The numeric layer's (N-1) sandbox is a real default-deny sandbox on macOS
(sandbox-exec), bare-metal Linux (bwrap), and native Windows (dedicated
low-privilege account + Task Scheduler, verified on real hardware); inside a
Docker container it falls back to a lighter Landlock + dedicated-user
isolation (details:
[docs/developer_manuals/29](docs/developer_manuals/29_deployment_and_sandbox_design.md)).

The local inference server (Ollama by default) starts automatically when
`ui/app.py` launches. Provider, command and environment variables are set under
`local_llm_server` in `config/default_settings.json`; switching Ollama → vLLM /
LocalAI is config-only (see `docs/developer_manuals/16_llm_model_setup.md`).

---

## CLI (for testing and debugging)

```bash
source .venv/bin/activate
python cli.py --title "session name"
```

| Command | Description |
|---|---|
| `reg <expr>` | register an expression (e.g. `reg (x^2-1)/(x-1)`) |
| `simplify E1` | rational simplification |
| `diff E1 x` | differentiate with respect to x |
| `taylor E1 x 0 3` | Taylor expansion (x=0, 3rd order) |
| `solve E1 x` | solve an equation |
| `fork S3 label` | branch from S3 |
| `switch <branch_id>` | switch branch |
| `replay` | replay the current branch |
| `history` | show the step history |
| `exprs` | show the expression table |
| `save <path>` | save the session |
| `help` | command list |

---

## Tests

```bash
source .venv/bin/activate

# daily iteration: fast subset (pure unit/parser/schema tests that never
# spin up julia/maxima, ~1m40s / 2,773 tests. tests/conftest.py auto-detects
# start_engine()-style calls in each test's own source and applies
# @pytest.mark.slow -- no manual tagging needed; the marker is registered
# in pytest.ini)
pytest tests/ -m "not slow"

# pre-merge full suite (~13m40s / 4,192 tests. CAS round-trips to Julia are
# 57% of it and Julia process STARTUP is still 20%; see docs/open_items.md's
# T-2 entry for the breakdown and what is left to recover. Exact count:
# `pytest tests/ --collect-only -q`)
pytest tests/

# a specific module: Engine tests
pytest tests/test_engine.py -v

# a specific module: new-operation tests
pytest tests/test_new_operations.py -v

# a single file
pytest tests/test_engine.py -v
```

## Debug REPL (`ui/debug_repl.py`)

Test the interpretation pipeline, save and load over stdin/stdout without a
browser. With `--no-llm` only rule-based interpretation runs, and confirmation
dialogs auto-select the first candidate.

```bash
# interactive (with LLM)
./scripts/debug_repl.sh --lang ja

# interactive (no LLM, rule-based only)
./scripts/debug_repl.sh --no-llm --lang ja

# pipe mode (non-interactive, for automated tests)
printf "x^2 + 1\ndifferentiate E1 by x\nquit\n" | ./scripts/debug_repl.sh --no-llm --lang en

# load a session
./scripts/debug_repl.sh --no-llm --session sessions/mysession.json

# specify a domain (inject domain-specific hints to the LLM)
./scripts/debug_repl.sh --lang ja --domains analytical_mechanics
```

For detailed sample input/output see section 8 of PROGRESS.md (reference omitted from the public snapshot).

---

## Directory layout

```
Research_Notebook/
│
├── notebook/               # core engine
│   ├── engine.py           # NotebookEngine (wires all components)
│   ├── engine_history_mixin.py  # EngineHistoryMixin — history/step management (record, delete, numbering)
│   ├── engine_persistence_mixin.py  # EnginePersistenceMixin — session persistence (save/load, CAS snapshot)
│   ├── models/             # Pydantic data models
│   │   ├── session.py
│   │   ├── expression.py   # Expression, Assumption (holds the IR tree)
│   │   ├── quantity.py     # Quantity, MathType
│   │   ├── symbol_context.py # SymbolContext (semantic layer)
│   │   ├── ir.py           # abstract syntax tree AST (formal layer)
│   │   ├── step.py         # Step, StepKind, StepStatus
│   │   └── branch.py
│   └── runtime/            # execution layer (3-layer compiler architecture)
│       ├── shared_parser.py        # SharedParser — Surface AST (Juxtaposition / BracketExpr, …)
│       ├── semantic_resolver.py    # BaseSemanticResolver — semantic-resolution base class
│       ├── julia_semantic_resolver.py  # JuliaSemanticResolver (Commutator / Mul elevation)
│       ├── maxima_semantic_resolver.py # MaximaSemanticResolver
│       ├── code_generator.py       # BaseCodeGenerator — code-generation base class
│       ├── julia_code_generator.py     # JuliaCodeGenerator
│       ├── maxima_code_generator.py    # MaximaCodeGenerator
│       ├── backend_handler.py      # BackendHandler ABC (abstract methods + no-op hooks)
│       ├── maxima_backend_handler.py   # MaximaBackendHandler — Maxima-specific handling
│       ├── julia_backend_handler.py    # JuliaBackendHandler — Julia-specific handling, particle-operator parser
│       ├── julia_handler_common.py     # module-level IR/operator helpers, regexes, constants for the Julia handler
│       ├── julia_sympy_mixin.py        # JuliaSympyMixin — SymPy chain and routing
│       ├── julia_build_mixin.py        # JuliaBuildMixin — CAS command construction (build_command + _build_*)
│       ├── julia_undefined_mixin.py    # JuliaUndefinedSymbolMixin — undefined-symbol detection
│       ├── julia_latex_mixin.py        # JuliaLatexMixin — LaTeX display/formatting
│       ├── lhs_formatting.py       # LHS / derivative LaTeX formatting, equation parsing (split from engine.py)
│       ├── backend_factory.py      # BackendFactory — make_backend(mode) builds the adapter set
│       ├── command_builder.py      # CommandBuilder shared base class (template-driven)
│       ├── maxima_command_builder.py   # MaximaCommandBuilder
│       ├── julia_command_builder.py    # JuliaCommandBuilder
│       ├── builder.py              # re-export shim (BuildResult / MaximaCommandBuilder / JuliaCommandBuilder)
│       ├── latex_converter.py      # LatexConverter shared base class (structural rules)
│       ├── maxima_latex_converter.py   # MaximaLatexConverter (%pi / %e, …)
│       ├── julia_latex_converter.py    # JuliaLatexConverter (pi / Inf / im, …)
│       ├── maxima_ir_compiler.py   # MaximaIRCompiler (recursive-descent parser)
│       ├── ir_compiler.py          # re-export shim → MaximaIRCompiler
│       ├── expr_generator.py       # ExprGenerator — IR AST → string (backend-independent)
│       ├── maxima_adapter.py       # persistent Maxima REPL
│       ├── julia_adapter.py        # persistent Julia process (for second-quantization sessions)
│       ├── _julia_driver.jl        # Julia driver script (loads QuantumAlgebra.jl + Symbolics.jl)
│       ├── operation_validator.py  # operation validation
│       ├── replay.py               # history replay
│       ├── branch_manager.py       # branch management
│       ├── llm_supervisor.py       # LLM 3-phase interpretation (extra_pre_phase1 hook)
│       ├── llm_client.py           # LLM API client
│       ├── llm_adviser.py          # LLM commentary (adviser)
│       ├── index_contractor.py     # static index contraction, Kronecker-δ application
│       ├── auto_einstein.py        # Einstein contraction (automatic sum over up/down indices)
│       ├── notation_profile_loader.py  # per-domain notation-profile loading
│       ├── lattice_figure.py       # lattice-figure generation (TikZ/SVG/PDF, F-1, no CAS)
│       ├── spin_algebra.py         # synthetic spin algebra (nested commutators, ∞T moments, continued fraction, S(ω), F-1 3c, no CAS)
│       └── i18n.py                 # t(key, lang) translation helper
│
│   (notebook/ also has engine_numeric_mixin.py, engine_plan_mixin.py, and
│    engine_laplace/fourier/finance/game/optimization/free_params/deferred_forms/wick_mixin.py
│    -- one mixin per feature family -- plus numeric/ below)
│
├── notebook/numeric/       # numeric layer (N-1; evaluates expressions the CAS can't solve analytically)
│   ├── sandbox.py          # SandboxRunner -- per-OS sandbox implementation (see below)
│   ├── templates.py        # deterministic templates (parameter_sweep/plotting/eigen, etc.)
│   ├── plot_spec.py / plot_codegen.py / plot_runtime.py / plot_prompt.py  # declarative Cartesian plots
│   ├── expr_to_numpy.py    # 3-layer-compiler-style function-whitelist lowering
│   ├── llm_codegen.py      # vibe-coding path (mandatory review gate)
│   └── spec_adviser.py / spec_projection.py / spec_validate.py / graph.py / intent_registry.py
│
├── ui/                     # web UI
│   ├── app.py              # FastAPI app (server entry point)
│   ├── notebook_bridge.py  # /api/notebook/* endpoints (thin routing layer)
│   ├── interpret_pipeline.py # natural language → operation pipeline (parsers split out below in AUD-5)
│   ├── interpret_common.py # low-level constants shared by interpret_* (_EIDS_RE, …)
│   ├── interpret_equation_ops.py # H-5/H-7/H-8/H-9 parsers (solve, both-sides ops, RHS→operation)
│   ├── interpret_l1.py     # L-1 linear-basis-transform / canonical-metric-constraint parsers
│   ├── interpret_f1.py     # F-1 lattice-sum / dispersion / crystal-descriptor / spin-algebra parsers
│   ├── session_io.py       # session save / load / autosave
│   ├── vlm_client.py       # OCR pipeline (domain-hint injection)
│   ├── apfel_client.py     # Apple Vision Framework bridge (macOS NPU OCR)
│   ├── image_utils.py      # handwriting-image preprocessing (margin trimming, …)
│   ├── katex_utils.py      # KaTeX local-asset management
│   ├── debug_repl.py       # stdin/stdout debug REPL (--no-llm)
│   ├── llm_server_launcher.py  # config-driven launch of the local inference server (Ollama/vLLM/LocalAI)
│   ├── routes/              # APIRouters split out of bridge (numeric/history/persistence/
│   │                        #  export/operations/session/advisory; all 49 endpoints)
│   ├── interpret_plan.py / interpret_numeric.py  # NL triggers for the C-Ext-5 plan layer / N-1 numeric fork
│   ├── notebook.html       # main notebook screen
│   ├── notebook_main.js    # DOM/state hub, split into ES modules by feature (notebook_numeric.js /
│   │                        #  notebook_plan.js / notebook_branch.js / notebook_timeline.js, etc.)
│   ├── notebook_main.css   # main notebook CSS
│   ├── index.html          # handwriting-input screen
│   └── main.js             # handwriting-input JS
│
├── schemas/
│   └── common_operations.schema.json   # 109 operation definitions
│
├── backends/
│   ├── maxima.full.json    # Maxima command templates (93 operations)
│   └── julia.full.json     # Julia command templates (50 operations, QuantumAlgebra.jl + Symbolics.jl)
│
├── setup.sh / setup_windows.ps1   # macOS / native-Windows one-shot installer
├── scripts/                # launch / build / provisioning scripts
│   ├── start.sh / start_windows.ps1  # launch the UI server (auto-activates the venv; mac / Windows)
│   ├── setup_windows_numeric_sandbox.ps1  # admin-elevated provisioning for the G-4 numeric sandbox
│   ├── debug_repl.sh       # launch the debug REPL
│   ├── build_sysimage.sh / build_julia_sysimage.jl  # build a custom Julia system image
│   └── sysimage_precompile.jl
│
├── models/                 # LLM endpoint configs
│   ├── default/config.json         # default model routing
│   ├── ollama_qwen25_coder_32b/    # supervisor (recommended, active)
│   ├── ollama_gpt_oss_20b/         # supervisor (alternate, active)
│   ├── ollama_qwen3_vl/            # OCR + supervisor (active)
│   ├── cloud_openai_gpt54/         # adviser (API key required)
│   ├── lmstudio_remote_gpt-oss-120b/ # adviser (LM Studio remote)
│   ├── lmstudio_gpt_oss_20b/       # alternate via LM Studio
│   ├── engines/                    # OCR-specialized engine configs
│   ├── refiners/                   # OCR post-processing configs (vision_input)
│   ├── particles.json              # particle table (32 particles: 18 fermions + 14 bosons)
│   └── domains/                    # per-field prompt hints (32 fields)
│       ├── categories.json         # category definitions (order, default_backend)
│       └── <domain_name>/          # meta.json / supervisor_hint.txt / ocr_hint.txt
│
├── i18n/                   # translation tables
│   ├── messages.json       # backend messages (engine / supervisor / adviser)
│   ├── ui_messages.json    # frontend UI strings
│   └── nl_patterns.json    # natural-language INPUT grammar patterns (per language)
│
├── config/                 # application settings
│   └── default_settings.json  # "language" — system default language
│
├── sessions/               # saved sessions (JSON / .mac snapshots)
│   └── autosave/           # autosave rotation (slot_1–slot_5)
│
├── tests/                  # pytest suite (exact count: `pytest --collect-only -q`)
│
├── docs/                   # design documents. See developer_manual.md for the full index (docs 01-29).
│   ├── developer_manual.md                    # developer-manual index (cover page)
│   ├── user_guide.md                          # user guide (input rules, all features)
│   ├── operation_reference.md                 # common-operation reference
│   ├── setup_guide.md                         # setup-guide cover page (jp/en, per-OS instructions)
│   ├── open_items.md                          # single source of truth for unresolved/deferred items
│   ├── scenarios.md / scenarios/               # practice-loop scenario cover + docs (scenario_*_jp.md / _en.md)
│   └── developer_manuals/                      # numbered technical docs (01-29, linked from developer_manual.md)
│
├── cli.py                  # CLI entry point
├── PROGRESS.md             # implementation progress
└── CLAUDE.md               # development policy (for AI)
```

---

## Installing dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn pydantic httpx pytest anyio
```

Installing Maxima (macOS):

```bash
brew install maxima
```

For the numeric layer (N-1, optional), create its dedicated interpreter
environment (kept separate from the app venv):

```bash
python3 -m venv .venv-numeric
source .venv-numeric/bin/activate
pip install -r notebook/numeric/requirements.txt
```

---

## Saving and loading sessions

In the UI: the "💾 Save" button → overwrite, or save under a chosen filename.

Autosave rotates every 5 minutes to `sessions/autosave/slot_N.json` (N=1–5).

Directly from code:

```python
from notebook.engine import NotebookEngine

# save
with NotebookEngine(title="experiment") as e:
    e.save("sessions/my_session.json")

# load
with NotebookEngine.load("sessions/my_session.json") as e:
    result = e.replay_current_branch()
```

---

## Usage modes

| Mode | primary backend | Use |
|---|---|---|
| standard_symbolic | Maxima | classical/analytical mechanics, field theory, ordinary symbolic computation |
| second_quantization | Julia (QuantumAlgebra.jl) | condensed matter, quantum optics, QFT, nuclear physics, spin systems — any field using second quantization |

Selecting even one second-quantization-category domain in the new-session modal
switches automatically to `second_quantization` mode (Julia backend).

---

## Related documents

- [Developer manual (index of technical docs)](docs/developer_manual.md) — cover and table of contents for design / architecture / internals
- [User guide](docs/user_guide.md)
- [Operation reference](docs/operation_reference.md)
- [Unimplemented / open items](KNOWN_ISSUES.md)
- Independent scenarios split out of the old E-3 workflow verification: [Jaynes-Cummings](docs/scenarios/scenario_jaynes_cummings_jp.md), [Heisenberg 2-site](docs/scenarios/scenario_heisenberg_two_site_jp.md), [Coulomb gauge](docs/scenarios/scenario_coulomb_gauge_jp.md), [ideal gas](docs/scenarios/scenario_ideal_gas_jp.md)
- [Single-pendulum analytical-mechanics scenario (Maxima)](docs/scenarios/scenario_pendulum_lagrangian_jp.md)
- [Symbol-domain / subscript-classification design](docs/developer_manuals/12_symbol_domain_design.md)
- [Vector-field / coordinate-system design spec](docs/developer_manuals/13_vector_field_design_spec.md)
- [Linear basis transformation (L-1) design](docs/developer_manuals/18_linear_transform_design.md)
- [Analytical mechanics: coordinates and velocities as independent variables (doc 19)](docs/developer_manuals/19_analytical_mechanics_independent_variables.md)
- [Notation Bridge catalog](docs/developer_manuals/08_notation_bridge_catalog.md)
- [Index-notation / Einstein-contraction design](docs/developer_manuals/06_index_notation_design.md)
- [LLM model-setup guide](docs/developer_manuals/16_llm_model_setup.md)
- [i18n design](docs/developer_manuals/15_i18n_design.md)
- [Whole-system specification](docs/developer_manuals/01_theoretical_research_notebook_spec.md)
- [History design spec](docs/developer_manuals/02_history_spec.md)
- [Numeric layer (N-1) design](docs/developer_manuals/22_numeric_layer_design.md) — theory expression → sandboxed execution → provenance-carrying reflux (phase 1 implemented; UI implemented — "🔢 To numeric")
- [Cartesian plot layers](docs/developer_manuals/36_cartesian_plot_layers.md) — declarative layers, API, source snapshots and sampling limits

## Precompiling Julia (solving the time-to-first-X problem)

To shorten the Julia backend's first-launch and module-load times, a script is
provided to build a custom system image (`sysimage.dylib`). After adding or
updating packages, rebuild the system image with:

```bash
# build the custom Julia system image (takes a few minutes)
./scripts/build_sysimage.sh
```

Once built, the system loads this `sysimage.dylib` automatically on startup, and
even complex computations return in milliseconds.
