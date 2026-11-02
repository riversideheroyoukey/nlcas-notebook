"""Tests for the characteristic_polynomial operation (A-10).

Supports the matrix eigenvalue workflow: given a matrix A and an eigenvalue
variable λ, compute det(λI - A) in one step, register it as an equation and
solve (H-9).  Julia uses LinearAlgebra's UniformScaling `I`; Maxima uses native
charpoly.
"""
from notebook.models.ir import CasOperation
from notebook.runtime.builder import MaximaCommandBuilder, JuliaCommandBuilder
from notebook.runtime.operation_validator import OperationValidator


class TestCharacteristicPolynomialBuild:
    def test_julia_command(self):
        b = JuliaCommandBuilder()
        r = b.build(CasOperation(operation="characteristic_polynomial",
                                 args={"expr": "H_mat", "var": "E"}))
        assert r.ok
        # det(λI - A); I is LinearAlgebra UniformScaling (auto-sized)
        assert r.command == "det(E*I - H_mat)"

    def test_maxima_command(self):
        b = MaximaCommandBuilder()
        r = b.build(CasOperation(operation="characteristic_polynomial",
                                 args={"expr": "H_mat", "var": "E"}))
        assert r.ok
        assert r.command == "expand(charpoly(H_mat,E))"


class TestCharacteristicPolynomialValidation:
    def test_known_operation_both_backends(self):
        v = OperationValidator()
        for backend in ("julia_full", "maxima"):
            r = v.validate("characteristic_polynomial",
                           {"expr": "H_mat", "var": "E"}, backend, set())
            assert r.valid, f"{backend}: {r.error}"

    def test_missing_var_is_rejected(self):
        v = OperationValidator()
        r = v.validate("characteristic_polynomial", {"expr": "H_mat"}, "julia_full", set())
        assert not r.valid
        assert "var" in r.missing_fields

    def test_aliases_present(self):
        # the schema must expose Japanese/English aliases so the supervisor can
        # map "特性多項式" / "characteristic polynomial" to this operation
        import json
        from pathlib import Path
        schema = json.loads((Path(__file__).resolve().parent.parent
                             / "schemas" / "common_operations.schema.json").read_text(encoding="utf-8"))
        entry = next(o for o in schema["operations"]
                     if o["operation"] == "characteristic_polynomial")
        assert "特性多項式" in entry["aliases"]["ja"]
        assert "characteristic polynomial" in entry["aliases"]["en"]
        assert entry["required_fields"] == ["expr", "var"]
