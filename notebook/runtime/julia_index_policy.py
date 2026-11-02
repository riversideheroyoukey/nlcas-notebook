"""How an indexed quantity is lowered in a Julia session — the DECISION only.

Split out of ``JuliaCodeGenerator.generate`` for the reason
SUBSCRIPT-POLICY-SPLIT gives for its own split: the branch kept growing because
the material for the decision kept growing (declaration, explicit lowering
style, the second-quantization letter convention, bound vs unbound, and now the
`param` route), and each addition raised the function's hygiene ceiling again.
Rather than raise it a fourth time, the decision moves here and the generator
keeps the rendering — it alone owns ``generate``.

The rule, in order:

1. **The whole subscripted name is declared** (`omega_0` as a scalar) → it is an
   ATOM.  The researcher named that quantity, subscript and all.
2. **An explicit ``index_lowering_style``** on the quantity wins next (docs/07).
3. **An operator** — declared, or one of the second-quantization letters — takes
   its index as a QuantumAlgebra index.
4. **Otherwise it is a c-number with an index**, which QuantumAlgebra spells
   ``param(:xi, k)``.  This is where a bound index has always gone; since
   2026-08-02 an unbound one goes there too **when the base is declared**.
5. **No declaration and no binding → refuse.**  There is no warrant for reading
   `V_i` as a parameter, and measured 2026-08-02, routing those as well turned a
   REFUSAL into silent nonsense: `\\sum_k xi_k c'_k c_k` resolves to a Mul in
   this backend, so the structural `sum` itself became `param(:sum, k)` and the
   whole expression a meaningless product.

Why `param` and not ``Symbolics.@syms``, measured 2026-08-02: `param` multiplies
QA operators (`xi(k) c†(k) c(k)`), survives `normal_form`, adds/multiplies/powers
with other params, takes scalar factors, and even multiplies a Symbolics
variable (`t xi(k)`); its latex is `xi_{k}`, the researcher's own subscript, so
no display bridge is needed.  `@syms xi(..)` displays as `xi(k)` — CAS syntax in
the researcher's face — and cannot multiply a QA operator at all, which is the
one thing a Julia session exists to do.  The single thing `param` cannot do is
DIFFERENTIATION (`Differential(::QuExpr)`), and that is translated into words by
``JuliaBackendHandler.classify_cas_error`` rather than left as a MethodError.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IndexedPlan:
    """What the generator should emit for one ``IndexedSymbol``."""
    kind: str                       # atom_full | quantum_index | array | atom | param | refuse
    value: str = ""                 # atom_full: the declared full name
    i18n_key: str = ""              # refuse: why
    fmt: dict = field(default_factory=dict)


def classify_indexed(base: str, index_names: "list[str]", context,
                     bound_indices: "list[str]",
                     operator_letters: "frozenset[str]") -> IndexedPlan:
    """Decide how ``base_{index_names}`` lowers.  No IR, no rendering."""
    clean = base.rstrip("'†")

    # 1. the whole subscripted name is itself declared
    if context and index_names:
        full_name = base + "_" + "_".join(index_names)
        full_prop = context.get(full_name)
        if full_prop and getattr(full_prop.quantity, "kind", None) == "scalar":
            return IndexedPlan(kind="atom_full", value=full_name)

    prop = context.get(clean) if context else None
    style = "function_call"
    is_operator = False
    if prop:
        explicit = getattr(prop.quantity, "index_lowering_style", None)
        if explicit and explicit != "backend_default":
            style = explicit
        elif prop.quantity.kind == "operator":
            style, is_operator = "quantum_index", True

    # 3. the second-quantization letter convention
    if clean in operator_letters and style == "function_call":
        style, is_operator = "quantum_index", True

    if style == "quantum_index" or is_operator:
        return IndexedPlan(kind="quantum_index")
    if style == "array_access":
        return IndexedPlan(kind="array")
    if style == "atom":
        return IndexedPlan(kind="atom")

    # 4/5. c-number with an index: `param`, if there is a warrant for it
    has_bound = any(n in bound_indices
                    or (n.startswith("-") and n[1:] in bound_indices)
                    for n in index_names)
    if has_bound or prop is not None:
        return IndexedPlan(kind="param")
    return IndexedPlan(
        kind="refuse",
        i18n_key="parser.unsupported_notation.julia_indexed_scalar",
        fmt={"name": f"{base}_{'_'.join(index_names)}",
             "base": base, "index": ", ".join(index_names)})
