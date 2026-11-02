# Backend Paradigm Unification

**Date:** 2026-06-04  
**Context:** H-1 (equation-form input) feature implementation revealed fundamental architectural tension between Maxima and Julia backends.

## Problem Statement

When symbol definitions are updated, the handling differs drastically between backends:

### Maxima Behavior (Evaluate-on-Assignment)

```
H = H_0 + H_1          → Sends: H : H_0 + H_1 to CAS
                       → RHS evaluated immediately at definition time
                       → H becomes fixed value: H : p^2/(2*m) - k*x^2

H_1 = -k*(x-a)^2       → Sends: H_1 : -k*(x-a)^2 to CAS
                       → H remains unchanged (already fixed)
                       → Problem: H does not reflect updated H_1
```

**Root cause:** Maxima evaluates and fixes composite expressions at definition time. Subsequent changes to component symbols (H_0, H_1) do not propagate to dependent composite expressions.

**User workaround:** Re-register H to force re-evaluation with latest symbol values.

### Julia Behavior (Structure-Preserving)

```
H = H_0 + H_1          → Stores: canonical "H = H_0 + H_1"
                       → Symbolic structure preserved (not evaluated)

H_1 = -k*(x-a)^2       → Stores: canonical "-k*(x-a)^2" 
                       → Does not affect H (no auto-substitution)

simplify(H)            → Returns "H_0 + H_1" (unevaluated)
                       → User must explicitly call:
                          substitute(H, [(H_0, p^2/(2*m)), (H_1, -k*(x-a)^2)])
```

**Root cause:** Julia intentionally separates definition (storing symbols) from substitution (using values). User controls when symbolic values are deployed.

## Paradigm Comparison

| Aspect | Maxima | Julia |
|--------|--------|-------|
| **Definition** | `H : expr` (assign + evaluate) | `H = expr` (store structure) |
| **Evaluation** | At definition time | On-demand (user explicit) |
| **Value propagation** | Automatic (fixed at definition) | Explicit (via `substitute`) |
| **Re-evaluation** | Requires re-registration | Requires re-call to `substitute` |
| **Symbol updates** | Don't propagate to composites | User controls propagation |
| **Use case fit** | Algebraic transformation | Expression manipulation + control |

## Why This Matters: Physics Use Cases

The tension becomes critical for advanced physics operations:

### 1. Hamiltonian Approximations

```
H = H_0 + λ*H_1  (perturbation Hamiltonian)

Operation: "Take λ → 0 limit"
  Maxima: Re-register H with H : H_0
  Julia:  substitute(H, λ=0)

Operation: "Vary λ from 0 to 1 in steps"
  Maxima: Register 11 different H expressions
  Julia:  Loop: for λ in [0, 0.1, ..., 1.0]: substitute(H, λ=value)
```

### 2. Perturbation Series Truncation

```
H = H_0 + λ*H_1 + λ²*corr_1 + λ³*corr_2 + ...

Operation: "Keep terms up to 2nd order in λ"
  Maxima: Extract specific terms, re-register
  Julia:  Delete λ³ onwards, substitute, evaluate

Operation: "Systematically explore orders"
  Maxima: 1st, 2nd, 3rd order → 3 separate registrations
  Julia:  Evaluate same structure with different truncations
```

### 3. Time-Dependent Protocols

```
H(t) = H_0 + t*H_1  (adiabatic sweep)

Operation: "Solve dynamics at t=0, 0.5, 1.0"
  Maxima: Register 3 different H expressions
  Julia:  Single structure, substitute(H, t=value) three times
```

**Conclusion:** Julia's structure-preserving paradigm is fundamentally better suited to systematic exploration of parameter spaces and approximation hierarchies. Maxima's evaluate-on-assignment requires re-registration for each variation.

## Current Implementation Status

### P1: undefined_symbols Auto-Update ✅
- When symbol defined, existing expressions' `undefined_symbols` refreshed
- Works for both backends
- **No paradigm issue here**

### P2: Maxima Auto-Substitution ✅ (with limitation)
- `H : H_0 + H_1` sent to Maxima on definition
- Problem: composite expressions fix at definition time
- Julia: No equivalent (design-intentional)

### P2 Enhancement: Re-Evaluation Workaround ✅
- User can manually re-evaluate H by re-registration
- But: **operation differs between backends**
  - Maxima: Re-register the composite expression
  - Julia: Call `substitute` with explicit pairs

## Design Solution: Backend Encapsulation

**Principle:** Unify user-facing operations while allowing backend-specific implementations.

```
【Layer 1: User Interface】
"Apply approximation to Hamiltonian"
"Re-evaluate composite expression with latest symbols"
"Explore perturbation series up to order N"
        ↓ [Unified API]
        
【Layer 2: NotebookEngine】
def re_evaluate_expr(expr_id: str, substitutions: dict) -> Expression:
    """Re-evaluate expression with substituted symbol values.
    
    Args:
        expr_id: Expression to re-evaluate
        substitutions: {symbol_name: new_canonical, ...}
    
    Returns:
        New expression with substitutions applied
    
    Backend-agnostic; delegates to handler.
    """
    return self._handler.re_evaluate_expr(expr_id, substitutions)

def apply_approximation(expr_id: str, approx_spec: str) -> Expression:
    """Apply physical approximation (e.g., λ → 0, truncate to order 2).
    
    Args:
        expr_id: Composite expression (e.g., H = H_0 + λ*H_1 + ...)
        approx_spec: Approximation description
    
    Returns:
        Expression with approximation applied
    
    Backend: Handles parsing and realization of approximation.
    """
    return self._handler.apply_approximation(expr_id, approx_spec)
        ↓ [Backend-specific implementation]
        
【Layer 3: BackendHandler (Abstract)】
@abstractmethod
def re_evaluate_expr(self, expr_id: str, substitutions: dict) -> str:
    """Re-evaluate composite expression.
    
    Backend implements using its own strategy:
    - Maxima: Extract canonical, subst([...], canonical), eval
    - Julia:  call substitute(expr, pairs)
    """
    pass

@abstractmethod
def apply_approximation(self, expr_id: str, approx_spec: str) -> str:
    """Apply approximation (λ → 0, truncate series, etc.)."""
    pass
        ↓
        
【Layer 4: Concrete Implementations】
MaximaBackendHandler.re_evaluate_expr():
    # Strategy 1: Extract canonical, use subst()
    # subst([H_0=..., H_1=...], H_0 + H_1) → evaluate
    # Return: p^2/(2*m) - k*(x-a)^2

MaximaBackendHandler.apply_approximation():
    # Extract terms, apply approximation rules, register

JuliaBackendHandler.re_evaluate_expr():
    # Strategy 2: Use substitute() on canonical
    # substitute(expr, [(H_0, value), (H_1, value)])
    # Return: result

JuliaBackendHandler.apply_approximation():
    # Parse approximation spec, call substitute, simplify
```

## Implementation Status（実装済み）

Phase 1-4 は実装完了（2026-06-06）。Phase 5 テストは `tests/test_algebraic_index_routing.py` に含まれる。

### Julia `re_evaluate_expr` — QA + SymPy 両対応

Julia バックエンドの `re_evaluate_expr` は QA 式・SymPy 式の両方に対応：

| 式の種類 | 実装 | 備考 |
|---|---|---|
| QA 式（`_nb_E2` = QuExpr） | Symbolics.substitute（係数のみ） | 実装済み（C-Ext Phase 1-A） |
| SymPy 式（`sympy_canonical` あり） | `_re_evaluate_sympy` → `_nb_sympy_substitute` を順次適用 | 実装済み（A-10, 2026-06-06） |

**バグ修正（2026-06-06）:** `_nb_sympy_substitute` のスカラーパースが `sp.sympify` だったため `nonzero=True` の assumptions が失われ、スカラー代入（`xi→1` 等）が効かなかった。`builtins.eval(..., ctx)` に修正。

---

## Implementation Roadmap（参照用）

### Phase 1: Abstract Interface Definition ✅
**File:** `notebook/runtime/backend_handler.py`

Add to `BackendHandler` (ABC):
```python
@abstractmethod
def re_evaluate_expr(self, expr_id: str, substitutions: dict) -> str:
    """Re-evaluate expression with current symbol values."""
    
@abstractmethod
def apply_approximation(self, expr_id: str, approx_type: str, params: dict) -> str:
    """Apply approximation to Hamiltonian/composite expression."""
    
@abstractmethod
def expand_perturbation_series(self, expr_id: str, order: int) -> str:
    """Expand perturbation series up to specified order."""
```

Add to `NotebookEngine`:
```python
def re_evaluate_expr(self, expr_id: str, substitutions: dict = None) -> Expression:
    """Public API: re-evaluate composite expression with latest symbol values."""
    if substitutions is None:
        # Auto-detect substitutions from lhs_alias_map
        substitutions = self._auto_detect_substitutions(expr_id)
    
    canonical_result = self._handler.re_evaluate_expr(expr_id, substitutions)
    return self.register_expression(canonical_result, ...)

def apply_approximation(self, expr_id: str, approx_type: str, params: dict) -> Expression:
    """Public API: apply approximation to composite expression."""
    canonical_result = self._handler.apply_approximation(expr_id, approx_type, params)
    return self.register_expression(canonical_result, ...)
```

### Phase 2: Maxima Implementation ✅
**File:** `notebook/runtime/maxima_backend_handler.py`

```python
def re_evaluate_expr(self, expr_id: str, substitutions: dict) -> str:
    """
    Extract canonical, build subst() list, evaluate.
    
    Example:
        expr_id = "E3", canonical = "H_0 + H_1"
        substitutions = {"H_0": "p^2/(2*m)", "H_1": "-k*(x-a)^2"}
        → subst([H_0=p^2/(2*m), H_1=-k*(x-a)^2], H_0 + H_1)
        → p^2/(2*m) - k*(x-a)^2
    """

def apply_approximation(self, expr_id: str, approx_type: str, params: dict) -> str:
    """
    Apply approximation: extract terms, filter, re-register.
    
    approx_type values:
    - "limit": {"symbol": "λ", "value": 0}
      → subst(λ=0, expr)
    - "truncate_perturbation": {"symbol": "λ", "order": 2}
      → Keep λ^0, λ^1, λ^2; drop higher
    - "neglect_term": {"term": "corr_2"}
      → Delete specified term
    """

def expand_perturbation_series(self, expr_id: str, order: int) -> str:
    """Expand perturbation series (typically in small parameter λ)."""
```

### Phase 3: Julia Implementation ✅（QA + SymPy 両対応）
**File:** `notebook/runtime/julia_backend_handler.py`

```python
def re_evaluate_expr(self, expr_id: str, substitutions: dict) -> str:
    """
    Call substitute() with substitution pairs.
    
    Example:
        expr canonical = "H = H_0 + H_1"
        substitutions = {"H_0": "p^2/(2*m)", "H_1": "-k*(x-a)^2"}
        → substitute(H_0 + H_1, [(H_0, p^2/(2*m)), (H_1, -k*(x-a)^2)])
        → p^2/(2*m) - k*(x-a)^2
    """

def apply_approximation(self, expr_id: str, approx_type: str, params: dict) -> str:
    """
    Apply approximation via substitute and simplify.
    
    approx_type same as Maxima, but implementation uses Julia's
    substitute + symbolic manipulation.
    """

def expand_perturbation_series(self, expr_id: str, order: int) -> str:
    """Expand perturbation series in Julia (series() or Taylor expansion)."""
```

### Phase 4: UI Integration ✅
**File:** `ui/notebook_bridge.py` (or new module)

Expose operations:
```python
{
  "re_evaluate": {
    "expr_id": "E3",
    "explicit_substitutions": {"H_0": "...", "H_1": "..."}  # optional
  },
  "apply_approximation": {
    "expr_id": "E3",
    "type": "limit",  # or "truncate_perturbation", "neglect_term"
    "params": {"symbol": "λ", "value": 0}
  },
  "explore_perturbation": {
    "expr_id": "E3",
    "max_order": 2
  }
}
```

### Phase 5: Tests ✅
`tests/test_algebraic_index_routing.py` に含まれる（30 テスト）。
```python
@pytest.mark.parametrize("mode", [SessionMode.standard_symbolic, SessionMode.second_quantization])
def test_re_evaluate_with_updated_symbols(mode):
    """Both Maxima and Julia should re-evaluate identically."""
    # Setup: H = H_0 + H_1, define H_0, H_1
    # Update: H_1 = new_value
    # Operation: re_evaluate(H)
    # Assert: result contains new H_1 value
```

## Key Design Principles

1. **Backend Encapsulation:** Backend-specific behavior isolated in `BackendHandler` subclasses. Shared layers (`NotebookEngine`, `notebook_bridge`) never inspect backend identity.

2. **Structure Preservation:** Canonical forms track symbolic structure (H = H_0 + H_1), allowing systematic exploration of approximations.

3. **Explicit User Control:** Operations like `re_evaluate` or `apply_approximation` are explicit; no silent propagation.

4. **Backward Compatibility:** Existing P1/P2 behavior unchanged. New methods are additive.

## Future Extensions

Once Phase 1-5 complete, support these physics-specific operations:

```
- Adiabatic theorem sweep: vary(H(t), t=[0, 1], steps=10)
- Mean-field approximation: assume_decoupling(expr, variables=[...])
- Bogoliubov transformation: apply_unitary_transform(expr, transform_expr)
- Second-order perturbation energy: compute_correction_order(expr, order=2)
```

All leverage the unified `apply_approximation` and `re_evaluate` infrastructure.

## Second Quantization: QA → SymPy ルーティング原則（2026-06-06）

第２量子化セッション（Julia full バックエンド）では、さらに内部で
QA（QuantumAlgebra）と SymPy の二層構造が存在する：

| 条件 | バックエンド | 備考 |
|---|---|---|
| 整数サイト・単純記号インデックス | QA | `c(1)`, `c(i)` など |
| 代数インデックス（`-k`, `k+q`）出現後 | SymPy | `c(-k)`, `c(k+q)` など |

**QA → SymPy は一方通行。SymPy → QA への逆行はない。**

これは Julia バックエンドの「パラダイム統一」の延長として理解できる：
Maxima と Julia が固定である（CLAUDE.md §5）のと同様に、
QA と SymPy の切り替えも一方向かつ不可逆である。

詳細: `docs/developer_manuals/05_algebraic_index_architecture.md` §3

## Related Documents

- Backend Encapsulation Rule (CLAUDE.md) (reference omitted from the public snapshot)
- [Algebraic Index Architecture](./05_algebraic_index_architecture.md)
- [Symbol Domain Feature](./12_symbol_domain_design.md)
- Math Input Conventions (CLAUDE.md) (reference omitted from the public snapshot)
