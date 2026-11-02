"""Matrix literal notation: "[a, b; c, d]" (';' → matrix, no ';' → commutator).

A bracket with a ';' row separator is a 2-D matrix; without ';' it stays a
commutator/list (the agreed convention).  The matrix flows through the shared
parser to a matrix(...) IR node and each backend's CodeGenerator renders its own
literal — Maxima matrix([a,b],[c,d]); Julia [a b; c d] — with the rows kept as
lists, never collapsed to commutators (the bug that broke 2×2 matrices).
"""
from notebook.runtime.shared_parser import SharedParser
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
from notebook.runtime.julia_semantic_resolver import JuliaSemanticResolver
from notebook.runtime.julia_code_generator import JuliaCodeGenerator
from notebook.models.symbol_context import SymbolContext


def _mx(s):
    return MaximaCodeGenerator().generate(
        MaximaSemanticResolver().resolve(SharedParser().parse(s), SymbolContext()))


def _jl(s):
    return JuliaCodeGenerator().generate(
        JuliaSemanticResolver().resolve(SharedParser().parse(s), SymbolContext()))


class TestMatrixLiteralCodegen:
    def test_maxima_semicolon_is_matrix(self):
        assert _mx("[2, 1; 1, 2]") == "matrix([2, 1], [1, 2])"
        assert _mx("[a, b; c, d]") == "matrix([a, b], [c, d])"
        assert _mx("[1, 2, 3; 4, 5, 6]") == "matrix([1, 2, 3], [4, 5, 6])"

    def test_julia_semicolon_is_matrix(self):
        assert _jl("[2, 1; 1, 2]") == "[2 1; 1 2]"
        assert _jl("[a, b; c, d]") == "[a b; c d]"
        assert _jl("[1, 2, 3; 4, 5, 6]") == "[1 2 3; 4 5 6]"

    def test_no_semicolon_stays_commutator(self):
        # 2-element bracket WITHOUT ';' is still a commutator on both backends.
        assert _mx("[A, B]") == "((A)*(B) - (B)*(A))"
        assert _jl("[A, B]") == "comm(A, B)"

    def test_explicit_matrix_call_rows_are_lists(self):
        # A 2-element row [2, 1] inside matrix(...) must NOT become a commutator
        # (the bug that collapsed 2×2 matrices to matrix(0, 0)).
        assert _mx("matrix([2,1],[1,2])") == "matrix([2, 1], [1, 2])"
        assert _jl("matrix([2,1],[1,2])") == "[2 1; 1 2]"


class TestMatrixLiteralNormalization:
    def test_pre_process_converts_to_neutral_matrix(self):
        from notebook.engine import NotebookEngine
        h = NotebookEngine()._handler
        assert h.normalize_matrix_literal("[2, 1; 1, 2]") == "matrix([2, 1], [1, 2])"
        assert h.normalize_matrix_literal("A = [2, 1; 1, 2]") == "A = matrix([2, 1], [1, 2])"
        # No ';' → untouched (commutator/list).
        assert h.normalize_matrix_literal("[A, B]") == "[A, B]"


class TestMatrixOperationsMaxima:
    def test_eigenvalues_and_inverse_e2e(self):
        from notebook.engine import NotebookEngine
        with NotebookEngine() as e:
            e.session.domains = ["linear_algebra"]
            e.start()
            a = e.register_expression("[2, 1; 1, 2]")
            assert e.expressions[a.display_id].canonical == "matrix([2, 1], [1, 2])"
            # eigenvalues 1 and 3 (symmetric matrix)
            s, _ = e.run_operation("eigenvalues", {"expr": a.display_id})
            ev = e.expressions[s.output_refs[0]].latex
            assert "3" in ev and "1" in ev
            # inverse is a 2×2 matrix (renders as pmatrix)
            s2, _ = e.run_operation("inverse_matrix", {"expr": a.display_id})
            assert "pmatrix" in e.expressions[s2.output_refs[0]].latex


class TestMatrixPipeline:
    """NL-pipeline behaviour: clean registration (no duplicate) and pmatrix display."""

    def _run(self, eng, sup, text):
        import asyncio
        from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
        return asyncio.run(run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))

    def _session(self):
        import ui.notebook_bridge as nb
        from notebook.runtime.llm_supervisor import LLMSupervisor
        from notebook.runtime.operation_validator import OperationValidator
        nb.start_engine(title="mtest", mode="standard", language="ja",
                        domains=["linear_algebra"])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["linear_algebra"], primary_backend="maxima")
        return nb, eng, sup

    def test_labeled_matrix_registers_once(self):
        nb, eng, sup = self._session()
        try:
            self._run(eng, sup, "A = [2, 1; 1, 2]")
            # exactly one expression (no duplicate RHS copy)
            assert list(eng.expressions.keys()) == ["E1"]
            assert eng.expressions["E1"].canonical == "matrix([2, 1], [1, 2])"
            assert r"\begin{pmatrix}" in eng.expressions["E1"].latex
        finally:
            nb.stop_engine()

    def test_bare_matrix_eigenvalues(self):
        nb, eng, sup = self._session()
        try:
            self._run(eng, sup, "[2, 1; 1, 2]")
            assert eng.expressions["E1"].canonical == "matrix([2, 1], [1, 2])"
            assert r"\begin{pmatrix}" in eng.expressions["E1"].latex
            r = self._run(eng, sup, "E1 の固有値を求めて")
            assert r["status"] == "completed"
            assert "3" in r["latex"] and "1" in r["latex"]
        finally:
            nb.stop_engine()

    def test_diagonalization_sequential_ids(self):
        nb, eng, sup = self._session()
        try:
            self._run(eng, sup, "A = [2, 1; 1, 2]")
            self._run(eng, sup, "E1 の固有値を求めて")
            self._run(eng, sup, "E1 の固有ベクトルを求めて")
            # A=E1, eigenvalues=E2, eigenvectors=E3 — no gaps / duplicates
            assert list(eng.expressions.keys()) == ["E1", "E2", "E3"]
        finally:
            nb.stop_engine()

    def test_diagonalize_builds_modal_matrix(self):
        # "対角化して" builds P (eigenvector columns) automatically — no manual P.
        nb, eng, sup = self._session()
        try:
            self._run(eng, sup, "A = [2, 1; 1, 2]")
            r = self._run(eng, sup, "E1 を対角化して")
            assert r["status"] == "completed"
            # P is built from NORMALIZED eigenvectors -> orthogonal matrix:
            # columns (1,1)/sqrt(2) and (1,-1)/sqrt(2)
            assert eng.expressions["E2"].canonical == (
                "matrix([1/sqrt(2),1/sqrt(2)],[1/sqrt(2),-(1/sqrt(2))])"
            )
        finally:
            nb.stop_engine()

    def test_diagonalization_similarity_transform(self):
        # Scenario S-i end-to-end: 対角化 → P, then P⁻¹ A P = diag(eigenvalues).
        nb, eng, sup = self._session()
        try:
            self._run(eng, sup, "A = [2, 1; 1, 2]")
            self._run(eng, sup, "E1 の固有値を求めて")          # 1, 3
            self._run(eng, sup, "E1 の固有ベクトルを求めて")
            self._run(eng, sup, "E1 を対角化して")              # P = E4 (auto)
            self._run(eng, sup, "E4 の逆行列を求めて")          # P⁻¹ = E5
            self._run(eng, sup, "E5 と E1 の行列積を求めて")     # P⁻¹A = E6
            r = self._run(eng, sup, "E6 と E4 の行列積を求めて")  # P⁻¹AP = E7
            assert r["status"] == "completed"
            assert eng.expressions["E7"].canonical == "matrix([3,0],[0,1])"
        finally:
            nb.stop_engine()

    def test_diagonal_matrix_of_eigenvalues(self):
        # "対角行列を求めて" builds D = diag(eigenvalues), distinct from 対角化 (→ P).
        nb, eng, sup = self._session()
        try:
            self._run(eng, sup, "A = [2, 1; 1, 2]")
            r = self._run(eng, sup, "E1 の対角行列を求めて")
            assert r["status"] == "completed"
            assert eng.expressions["E2"].canonical == "matrix([3,0],[0,1])"
        finally:
            nb.stop_engine()


class TestResultNamingAndAssignment:
    """Name/assign an operation result to a variable, then reuse it (P = En /
    En を P とする).  The named variable is bound in the CAS for later formulas."""

    def _run(self, eng, sup, text):
        import asyncio
        from ui.interpret_pipeline import InterpretRequest, InputToken, run_interpret_pipeline
        return asyncio.run(run_interpret_pipeline(
            InterpretRequest(tokens=[InputToken(type="text", value=text)]), eng, sup))

    def _session(self):
        import ui.notebook_bridge as nb
        from notebook.runtime.llm_supervisor import LLMSupervisor
        from notebook.runtime.operation_validator import OperationValidator
        nb.start_engine(title="nm", mode="standard", language="ja",
                        domains=["linear_algebra"])
        eng = nb.get_engine()
        sup = LLMSupervisor(validator=OperationValidator(), llm_client=None,
                            domains=["linear_algebra"], primary_backend="maxima")
        return nb, eng, sup

    def test_name_matrix_result_equation_form(self):
        nb, eng, sup = self._session()
        try:
            self._run(eng, sup, "A = [2, 1; 1, 2]")
            self._run(eng, sup, "E1 を対角化して")   # E2 = P
            r = self._run(eng, sup, "P = E2")        # name it P
            assert r["status"] == "registered"
            # P is bound to the (normalized, orthogonal) modal matrix in the CAS
            assert eng._adapter.execute("P;").value == (
                "matrix([1/sqrt(2),1/sqrt(2)],[1/sqrt(2),-(1/sqrt(2))])"
            )
            assert eng._adapter.execute("ratsimp(invert(P) . A . P)").value == "matrix([3,0],[0,1])"
        finally:
            nb.stop_engine()

    def test_assign_command_form(self):
        nb, eng, sup = self._session()
        try:
            self._run(eng, sup, "A = [2, 1; 1, 2]")
            self._run(eng, sup, "E1 を対角化して")   # E2
            r = self._run(eng, sup, "E2 を P とする")  # assignment command
            assert r["status"] == "registered"
            assert eng._adapter.execute("P;").value == (
                "matrix([1/sqrt(2),1/sqrt(2)],[1/sqrt(2),-(1/sqrt(2))])"
            )
        finally:
            nb.stop_engine()

    def test_name_scalar_result(self):
        nb, eng, sup = self._session()
        try:
            self._run(eng, sup, "f = x^2")
            r = self._run(eng, sup, "g = E1")
            assert r["status"] == "registered"
            assert eng._adapter.execute("g;").value == "x^2"
        finally:
            nb.stop_engine()

    def test_type_declaration_not_caught(self):
        # "b をスカラーとする" must remain a type declaration, not a name-assign.
        nb, eng, sup = self._session()
        try:
            r = self._run(eng, sup, "b をスカラーとする")
            assert r["status"] in ("confirmed", "declared")
            assert eng.symbol_context.get("b") is not None
        finally:
            nb.stop_engine()
