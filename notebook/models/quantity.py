from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, model_validator


class MathType(BaseModel):
    algebra: Literal[
        "scalar", "vector", "matrix", "operator", "tensor", "spinor", "form"
    ] = "scalar"
    shape: Optional[tuple[int, ...]] = None
    field: Literal["real", "complex"] = "complex"
    properties: list[str] = []
    space: Optional[str] = None


class Quantity(BaseModel):
    """Semantic description of a named mathematical quantity.

    Tracks how a quantity was defined (kind), its mathematical type (math_type),
    and the variables it depends on (natural_vars).  Populated automatically by
    NotebookEngine.register_expression when a defining equation is registered.
    """
    name: str
    kind: Literal["anonymous", "scalar", "function", "conjugate", "operator", "vector_field", "scalar_field"]
    math_type: MathType = MathType()
    index_lowering_style: Optional[Literal["backend_default", "function_call", "array_access", "atom", "quantum_index", "index_alias"]] = None
    """How an indexed quantity is lowered (docs/07 §4).

    ``index_alias`` (docs/07 §8.2, added 2026-07-30) lowers ``V_i`` under a
    DISTINCT CAS name — ``V__idx(i)`` — so it cannot share CAS space with a
    genuine function call ``V(x)``.  Opt-in per symbol, NOT the default: making
    it the default was measured (47 failures in 9 test files alone) and, worse,
    would not match the ``V(i)`` already stored in saved sessions — a new
    ``subst(V__idx(i)=5, …)`` silently misses an old file's ``V(i)``, which is
    the very failure mode this project is repairing.  Not applicable to
    Julia/QuantumAlgebra, where an index is a typed ``QuIndex`` and no collision
    exists in the first place.
    """
    natural_vars: list[str] = []
    # vector_field / scalar_field metadata
    components: list[str] = []          # e.g. ["A_x", "A_y", "A_z"]
    coordinate_system: Optional[str] = None  # "cartesian" | "spherical" | "cylindrical"
    algebraic_context: Optional[str] = None
    defining_expr_id: Optional[str] = None
    conjugate_of: Optional[tuple[str, str]] = None
    held_fixed: list[str] = []
    time_dependent: bool = False
    time_parameter: str = "t"
    notation: Literal["dot", "leibniz"] = "dot"
    latex_name: Optional[str] = None
    # Number domain (kind): "real" | "complex" | "integer" | "natural"
    # None = complex (default, backend-dependent)
    domain: Optional[str] = None
    # Sign constraint, independent of domain kind, so compound declarations
    # like "positive integer" keep BOTH ("integer", "positive") instead of
    # collapsing to one or the other:
    # "positive" | "negative" | "nonneg" | "nonpos"
    sign: Optional[str] = None

    # Index symmetry of a 2-argument function f(i,j):
    #   "symmetric":     f(i,j) =  f(j,i)   e.g. distance matrix
    #   "antisymmetric": f(i,j) = -f(j,i)   e.g. BCS κ, magnetic field tensor
    #   "hermitian":     f(i,j) =  f(j,i)*  e.g. Hamiltonian matrix
    # None = no symmetry constraint
    symmetry: Optional[Literal["symmetric", "antisymmetric", "hermitian"]] = None

    @model_validator(mode="before")
    @classmethod
    def migrate_display_format(cls, data: dict) -> dict:
        if isinstance(data, dict):
            if "display_format" in data and "notation" not in data:
                data["notation"] = data.pop("display_format")
        return data

