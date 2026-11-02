import pytest
from notebook.runtime.operation_validator import OperationValidator, ValidationResult


@pytest.fixture(scope="module")
def v():
    return OperationValidator()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_operations_loaded(self, v):
        assert len(v.known_operations) > 0

    def test_all_mvp_operations_present(self, v):
        for op in ["expand", "factor", "simplify", "rational_simplify",
                   "substitute", "differentiate", "integrate_indefinite",
                   "solve_equation", "taylor_expand", "to_latex",
                   "commutator", "normal_order"]:
            assert op in v.known_operations, f"'{op}' missing from registry"

    def test_get_spec_returns_dict(self, v):
        spec = v.get_spec("expand")
        assert spec is not None
        assert spec["feature"] == "algebraic_rewrite"
        assert "maxima" in spec["supported_backends"]

    def test_get_spec_unknown_returns_none(self, v):
        assert v.get_spec("nonexistent") is None

    def test_operations_for_maxima(self, v):
        ops = v.operations_for_backend("maxima")
        assert "expand" in ops
        assert "commutator" not in ops

    def test_operations_for_julia(self, v):
        ops = v.operations_for_backend("julia_full")
        assert "commutator" in ops
        assert "normal_order" in ops


# ---------------------------------------------------------------------------
# Unknown operation
# ---------------------------------------------------------------------------

class TestUnknownOperation:
    def test_unknown_op_name(self, v):
        r = v.validate("frobnicate", {}, "maxima")
        assert not r.valid
        assert r.error_code == "UNKNOWN_OPERATION"
        assert "frobnicate" in r.error

    def test_result_fields_on_unknown(self, v):
        r = v.validate("bad_op", {}, "maxima")
        assert r.operation is None
        assert r.backend is None


# ---------------------------------------------------------------------------
# Missing required fields
# ---------------------------------------------------------------------------

class TestMissingFields:
    def test_differentiate_missing_var(self, v):
        r = v.validate("differentiate", {"expr": "E1"}, "maxima")
        assert not r.valid
        assert r.error_code == "MISSING_REQUIRED_FIELDS"
        assert "var" in r.missing_fields

    def test_substitute_missing_lhs_and_rhs(self, v):
        r = v.validate("substitute", {"expr": "E1"}, "maxima")
        assert not r.valid
        assert r.error_code == "MISSING_REQUIRED_FIELDS"
        assert "lhs" in r.missing_fields
        assert "rhs" in r.missing_fields

    def test_taylor_expand_missing_multiple(self, v):
        r = v.validate("taylor_expand", {"expr": "E1"}, "maxima")
        assert not r.valid
        missing = set(r.missing_fields)
        assert {"var", "point", "order"}.issubset(missing)

    def test_expand_all_fields_present(self, v):
        r = v.validate("expand", {"expr": "E1"}, "maxima", known_expr_ids={"E1"})
        assert r.valid

    def test_optional_fields_not_required(self, v):
        # assign_to is optional — omitting it must not fail
        r = v.validate("expand", {"expr": "E1"}, "maxima", known_expr_ids={"E1"})
        assert r.valid


# ---------------------------------------------------------------------------
# Unknown / misspelled fields
# ---------------------------------------------------------------------------

class TestUnknownFields:
    def test_misspelled_field_rejected(self, v):
        # "varl" instead of "var"
        r = v.validate("differentiate", {"expr": "E1", "var": "x", "varl": "t"}, "maxima")
        assert not r.valid
        assert r.error_code == "UNKNOWN_FIELDS"
        assert "varl" in r.unknown_fields

    def test_completely_foreign_field(self, v):
        r = v.validate("expand", {"expr": "E1", "foo": "bar"}, "maxima")
        assert not r.valid
        assert r.error_code == "UNKNOWN_FIELDS"
        assert "foo" in r.unknown_fields

    def test_optional_field_accepted(self, v):
        # assign_to is optional for expand
        r = v.validate("expand", {"expr": "E1", "assign_to": "result"},
                       "maxima", known_expr_ids={"E1"})
        assert r.valid

    def test_all_required_only_accepted(self, v):
        r = v.validate("differentiate", {"expr": "E1", "var": "x"},
                       "maxima", known_expr_ids={"E1"})
        assert r.valid

    def test_unknown_fields_lists_all_offenders(self, v):
        r = v.validate("expand", {"expr": "E1", "foo": "a", "bar": "b"}, "maxima")
        assert not r.valid
        assert set(r.unknown_fields) == {"foo", "bar"}

    def test_error_message_includes_allowed(self, v):
        r = v.validate("expand", {"expr": "E1", "typo": "x"}, "maxima")
        assert "Allowed:" in r.error
        assert "expr" in r.error

    def test_internal_fields_not_seen_by_validator(self):
        """_is_total_derivative must be stripped by _preprocess_args, not reach validate."""
        from notebook.engine import NotebookEngine
        with NotebookEngine() as e:
            e.start()
            expr = e.register_expression("f(x) = x^2")
            # Passing _is_total_derivative must not raise UNKNOWN_FIELDS
            step, _ = e.run_operation(
                "differentiate",
                {"expr": expr.display_id, "var": "x", "_is_total_derivative": True},
            )
            assert step is not None


# ---------------------------------------------------------------------------
# Expression reference existence
# ---------------------------------------------------------------------------

class TestExpressionExists:
    def test_unknown_expr_ref_fails(self, v):
        r = v.validate("expand", {"expr": "E99"}, "maxima", known_expr_ids={"E1", "E2"})
        assert not r.valid
        assert r.error_code == "UNKNOWN_EXPRESSION"
        assert "E99" in r.unknown_exprs

    def test_known_expr_ref_passes(self, v):
        r = v.validate("expand", {"expr": "E1"}, "maxima", known_expr_ids={"E1"})
        assert r.valid

    def test_no_check_when_known_expr_ids_is_none(self, v):
        # Caller opted out of expression checking
        r = v.validate("expand", {"expr": "E99"}, "maxima", known_expr_ids=None)
        assert r.valid

    def test_inline_expr_not_checked_against_registry(self, v):
        # Values not matching E\d+ are treated as inline expressions
        r = v.validate("expand", {"expr": "x^2+1"}, "maxima", known_expr_ids=set())
        assert r.valid

    def test_commutator_both_refs_checked(self, v):
        r = v.validate("commutator", {"a": "E1", "b": "E99"},
                       "julia_full", known_expr_ids={"E1"})
        assert not r.valid
        assert "E99" in r.unknown_exprs

    def test_commutator_all_known(self, v):
        r = v.validate("commutator", {"a": "E1", "b": "E2"},
                       "julia_full", known_expr_ids={"E1", "E2"})
        assert r.valid

    def test_solve_system_list_of_equations(self, v):
        r = v.validate("solve_system",
                       {"equations": ["E1", "E2"], "vars": ["x", "y"]},
                       "maxima", known_expr_ids={"E1", "E2"})
        assert r.valid

    def test_solve_system_unknown_in_list(self, v):
        r = v.validate("solve_system",
                       {"equations": ["E1", "E99"], "vars": ["x", "y"]},
                       "maxima", known_expr_ids={"E1"})
        assert not r.valid
        assert "E99" in r.unknown_exprs


# ---------------------------------------------------------------------------
# Backend support
# ---------------------------------------------------------------------------

class TestBackendSupport:
    def test_commutator_not_in_maxima(self, v):
        r = v.validate("commutator", {"a": "E1", "b": "E2"},
                       "maxima", known_expr_ids={"E1", "E2"})
        assert not r.valid
        assert r.error_code == "BACKEND_NOT_SUPPORTED"
        assert "julia_full" in r.error

    def test_fallback_confirmed_resolves_backend(self, v):
        r = v.validate("commutator", {"a": "E1", "b": "E2"},
                       "maxima", known_expr_ids={"E1", "E2"},
                       fallback_confirmed=True)
        assert r.valid
        assert r.backend == "julia_full"

    def test_collect_only_in_maxima(self, v):
        r = v.validate("collect", {"expr": "E1", "var": "x"},
                       "maxima", known_expr_ids={"E1"})
        assert r.valid
        assert r.backend == "maxima"

    def test_collect_not_in_julia(self, v):
        r = v.validate("collect", {"expr": "E1", "var": "x"},
                       "julia_full", known_expr_ids={"E1"})
        assert not r.valid
        assert r.error_code == "BACKEND_NOT_SUPPORTED"

    def test_expand_both_backends(self, v):
        for backend in ["maxima", "julia_full"]:
            r = v.validate("expand", {"expr": "E1"}, backend, known_expr_ids={"E1"})
            assert r.valid
            assert r.backend == backend

    def test_numeric_solve_only_julia(self, v):
        r = v.validate("numeric_solve",
                       {"function": "E1", "initial_guess": "0"},
                       "maxima")
        assert not r.valid
        assert r.error_code == "BACKEND_NOT_SUPPORTED"


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------

class TestAliasResolution:
    # --- canonical names resolve to themselves ---
    def test_canonical_resolves_to_itself(self, v):
        assert v.resolve_alias("factor") == "factor"
        assert v.resolve_alias("differentiate") == "differentiate"
        assert v.resolve_alias("taylor_expand") == "taylor_expand"

    # --- English short aliases ---
    def test_en_alias_diff(self, v):
        assert v.resolve_alias("diff") == "differentiate"

    def test_en_alias_integrate(self, v):
        assert v.resolve_alias("integrate") == "integrate_indefinite"

    def test_en_alias_taylor(self, v):
        assert v.resolve_alias("taylor") == "taylor_expand"

    def test_en_alias_solve(self, v):
        assert v.resolve_alias("solve") == "solve_equation"

    def test_en_alias_factorize(self, v):
        assert v.resolve_alias("factorize") == "factor"

    def test_en_alias_factorise(self, v):
        assert v.resolve_alias("factorise") == "factor"

    def test_en_alias_ratsimp(self, v):
        assert v.resolve_alias("ratsimp") == "rational_simplify"

    def test_en_alias_det(self, v):
        assert v.resolve_alias("det") == "determinant"

    def test_en_alias_lim(self, v):
        assert v.resolve_alias("lim") == "limit"

    def test_en_alias_subst(self, v):
        assert v.resolve_alias("subst") == "substitute"

    def test_en_alias_vev(self, v):
        assert v.resolve_alias("vev") == "vacuum_expectation"

    def test_en_alias_latex(self, v):
        assert v.resolve_alias("latex") == "to_latex"

    # --- Japanese aliases ---
    def test_ja_alias_factor(self, v):
        assert v.resolve_alias("因数分解") == "factor"
        assert v.resolve_alias("因数分解する") == "factor"

    def test_ja_alias_differentiate(self, v):
        assert v.resolve_alias("微分") == "differentiate"
        assert v.resolve_alias("微分する") == "differentiate"

    def test_ja_alias_expand(self, v):
        assert v.resolve_alias("展開") == "expand"
        assert v.resolve_alias("展開する") == "expand"

    def test_ja_alias_integrate_indefinite(self, v):
        assert v.resolve_alias("積分") == "integrate_indefinite"
        assert v.resolve_alias("不定積分") == "integrate_indefinite"

    def test_ja_alias_taylor(self, v):
        assert v.resolve_alias("テイラー展開") == "taylor_expand"

    def test_ja_alias_limit(self, v):
        assert v.resolve_alias("極限") == "limit"

    def test_ja_alias_solve(self, v):
        assert v.resolve_alias("方程式を解く") == "solve_equation"
        assert v.resolve_alias("解く") == "solve_equation"

    def test_ja_alias_rational_simplify(self, v):
        assert v.resolve_alias("有理簡約") == "rational_simplify"
        assert v.resolve_alias("約分") == "rational_simplify"

    def test_ja_alias_commutator(self, v):
        assert v.resolve_alias("交換子") == "commutator"

    def test_ja_alias_normal_order(self, v):
        assert v.resolve_alias("正規順序") == "normal_order"

    def test_ja_alias_boson_variants(self, v):
        assert v.resolve_alias("ボゾン演算子") == "define_boson_operator"
        assert v.resolve_alias("ボーズ演算子") == "define_boson_operator"
        assert v.resolve_alias("フォノン演算子") == "define_boson_operator"
        assert v.resolve_alias("ボソン演算子") == "define_boson_operator"

    def test_ja_alias_fermion_variants(self, v):
        assert v.resolve_alias("フェルミ演算子") == "define_fermion_operator"
        assert v.resolve_alias("電子演算子") == "define_fermion_operator"

    # --- Case insensitivity ---
    def test_case_insensitive_en(self, v):
        assert v.resolve_alias("FACTOR") == "factor"
        assert v.resolve_alias("Diff") == "differentiate"
        assert v.resolve_alias("Taylor") == "taylor_expand"

    # --- Unknown ---
    def test_unknown_returns_none(self, v):
        assert v.resolve_alias("frobnicate") is None
        assert v.resolve_alias("") is None
        assert v.resolve_alias("zzz") is None

    # --- known_aliases property ---
    def test_known_aliases_nonempty(self, v):
        assert len(v.known_aliases) > 30


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_valid_result_has_all_fields(self, v):
        r = v.validate("expand", {"expr": "E1"}, "maxima", known_expr_ids={"E1"})
        assert isinstance(r, ValidationResult)
        assert r.valid is True
        assert r.operation == "expand"
        assert r.args == {"expr": "E1"}
        assert r.backend == "maxima"
        assert r.error is None
        assert r.error_code is None
        assert r.missing_fields == []
        assert r.unknown_exprs == []

    def test_invalid_result_preserves_args(self, v):
        args = {"expr": "E1"}
        r = v.validate("differentiate", args, "maxima")
        assert r.args == args
        assert r.operation == "differentiate"


class TestAnUnknownExpressionIdIsRefusedEverywhereItIsExpanded:
    r"""Found running S-l (squeezed states) 2026-08-04.

    The backend expander (`_make_build_ctx`) replaces ANY string arg matching
    `^E\d+$` with that expression's canonical text — including strings nested
    inside `op_rules` dicts.  The validator checked a whitelist of top-level
    field names that did not include `rhs`, so an id that did not exist was
    never refused: it simply failed to expand and was parsed as a SYMBOL.

        E1 の a(1) に E99 を代入する
        → a^{\dagger}(1) a(1) = E99^{2}          status=completed

    An internal identifier standing in the researcher's notebook as if it were
    mathematics — the H1-ALIAS-EID-LEAK shape, and a wrong value with no error.

    The invariant these pin: **whatever the expander will look up, the validator
    must have checked exists.**
    """

    def _validator(self):
        from notebook.runtime.operation_validator import OperationValidator
        return OperationValidator()

    def test_unknown_rhs_id_is_refused(self):
        v = self._validator()
        res = v.validate("operator_substitute",
                         {"expr": "E1", "op_name": "a", "rhs": "E99",
                          "formal_index": "1"},
                         known_expr_ids={"E1"}, primary_backend="julia_full")
        assert not res.valid
        assert res.error_code == "UNKNOWN_EXPRESSION"
        assert "E99" in res.error

    def test_known_rhs_id_is_accepted(self):
        v = self._validator()
        res = v.validate("operator_substitute",
                         {"expr": "E1", "op_name": "a", "rhs": "E2",
                          "formal_index": "1"},
                         known_expr_ids={"E1", "E2"}, primary_backend="julia_full")
        assert res.valid, res.error

    def test_a_raw_expression_rhs_is_not_mistaken_for_an_id(self):
        """`rhs` usually holds a formula; only an exact `E<digits>` is a ref."""
        v = self._validator()
        res = v.validate("operator_substitute",
                         {"expr": "E1", "op_name": "a", "rhs": "u*b(1) - v*b'(1)",
                          "formal_index": "1"},
                         known_expr_ids={"E1"}, primary_backend="julia_full")
        assert res.valid, res.error

    def test_unknown_id_inside_op_rules_is_refused(self):
        """The multi-rule form nests its rhs inside dicts — the shape the
        expander reaches into, so the checker must too."""
        v = self._validator()
        res = v.validate("operator_substitute",
                         {"expr": "E1", "op_rules": [
                             {"op_name": "a", "formal_index": "1", "rhs": "E2"},
                             {"op_name": "b", "formal_index": "1", "rhs": "E99"}]},
                         known_expr_ids={"E1", "E2"}, primary_backend="julia_full")
        assert not res.valid
        assert "E99" in res.error and "E2" not in res.error

    def test_a_symbol_legitimately_named_E1_is_not_a_reference(self):
        """`name` is an IDENTIFIER field, so declaring a symbol called `E1`
        still works.  That exception is the whole reason the classification is a
        denylist rather than "anything shaped like an id"."""
        v = self._validator()
        res = v.validate("define_scalar_parameter", {"name": "E1"},
                         known_expr_ids=set(), primary_backend="maxima")
        assert res.valid, res.error


class TestOneAnswerForWhichFieldsHoldAnId:
    r"""EXPR-REF-FIELD-TYPES (2026-08-04).

    Measured, the codebase held three different answers to "may this field hold
    an expression id?" — Maxima's builder used a denylist (`_IDENTIFIER_KEYS`),
    Julia's builder had no notion of identifier fields at all, and the validator
    used an allowlist of 13 names.  The allowlist is what let EXPR-REF through.

    The invariant is not about which fields *ought* to accept a reference; it is
    that the expander and the validator agree, so behaviour cannot depend on
    whether the id someone wrote happens to exist.  `var` makes the point: it
    holds an integration variable, and the expander substitutes it anyway
    (measured: `integrate {expr: E1, var: E1}` builds
    `integrate(x^2 + 1, x^2 + 1)`).  Because the expander does, the validator
    must check it.
    """

    def _validator(self):
        from notebook.runtime.operation_validator import OperationValidator
        return OperationValidator()

    def test_all_three_read_the_same_set(self):
        from notebook.runtime.expr_ref_fields import IDENTIFIER_KEYS
        from notebook.runtime.maxima_backend_handler import _IDENTIFIER_KEYS as mx
        from notebook.runtime.julia_backend_handler import _IDENTIFIER_KEYS as jl
        assert mx is IDENTIFIER_KEYS and jl is IDENTIFIER_KEYS

    def test_a_field_outside_the_old_allowlist_is_now_checked(self):
        """`var` was in neither the old allowlist nor the denylist — the exact
        gap shape that EXPR-REF was."""
        v = self._validator()
        res = v.validate("integrate_indefinite", {"expr": "E1", "var": "E99"},
                         known_expr_ids={"E1"}, primary_backend="maxima")
        assert not res.valid and "E99" in res.error

    def test_an_identifier_field_is_still_exempt(self):
        v = self._validator()
        res = v.validate("define_scalar_parameter", {"name": "E99"},
                         known_expr_ids=set(), primary_backend="maxima")
        assert res.valid, res.error

    def test_iter_expr_refs_walks_what_the_builders_walk(self):
        from notebook.runtime.expr_ref_fields import iter_expr_refs
        got = iter_expr_refs({
            "expr": "E1",                       # string
            "vars": ["E2", "x"],                # list of strings
            "op_rules": [{"op_name": "a", "rhs": "E3"}],   # list of dicts
            "name": "E4",                       # identifier — exempt
            "order": "2",                       # not an id at all
        })
        assert sorted(eid for _f, eid in got) == ["E1", "E2", "E3"]
