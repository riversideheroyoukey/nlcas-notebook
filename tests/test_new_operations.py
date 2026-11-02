from notebook.models.ir import CasOperation
"""Tests for newly added operations.

Covers: schema completeness, alias resolution, builder templates,
and Maxima integration for each new operation category.
"""
import pytest
from notebook.engine import NotebookEngine
from notebook.runtime.operation_validator import OperationValidator
from notebook.runtime.builder import MaximaCommandBuilder
from notebook.runtime.maxima_adapter import MaximaAdapter


@pytest.fixture(scope="module")
def v():
    return OperationValidator()


@pytest.fixture(scope="module")
def b():
    return MaximaCommandBuilder()


@pytest.fixture(scope="module")
def maxima():
    with MaximaAdapter() as m:
        yield m


# ---------------------------------------------------------------------------
# Schema / alias consistency: every new operation has both files in sync
# ---------------------------------------------------------------------------

NEW_OPS = [
    "use_relation", "radical_simplify", "partial_fraction", "log_contract",
    "trig_expand", "trig_reduce", "trig_to_exponential",
    "real_part", "imaginary_part", "complex_conjugate",
    "to_rectangular", "to_polar", "residue",
    "summation", "product_expr", "laplace_transform", "inverse_laplace",
    "add_assumption", "remove_assumption",
    "solve_ode",
    "transpose", "trace", "matrix_multiply",
    "numerical_eval", "numerical_eval_hp",
    "declare_time_dependent",
    "set_notation",
    "set_latex_name",
]


class TestSchemaCompleteness:
    def test_all_new_ops_in_schema(self, v):
        for op in NEW_OPS:
            assert op in v.known_operations, f"'{op}' missing from schema"

    def test_all_new_ops_have_template(self, b):
        for op in NEW_OPS:
            assert op in b.supported_operations, f"'{op}' missing from maxima.full.json"

    def test_use_relation_backend_is_maxima_only(self, v):
        spec = v.get_spec("use_relation")
        assert "maxima" in spec["supported_backends"]
        assert "julia_full" not in spec["supported_backends"]

    def test_no_maxima_op_missing_template(self, v, b):
        """Every operation listed as maxima-supported must have a template."""
        for op in v.known_operations:
            spec = v.get_spec(op)
            if "maxima" in spec["supported_backends"]:
                assert op in b.supported_operations, (
                    f"'{op}' has maxima in supported_backends but no template in maxima.full.json"
                )


# ---------------------------------------------------------------------------
# Alias resolution for new operations
# ---------------------------------------------------------------------------

class TestNewAliases:
    def test_assume(self, v):
        assert v.resolve_alias("assume") == "add_assumption"
        assert v.resolve_alias("仮定する") == "add_assumption"

    def test_forget(self, v):
        assert v.resolve_alias("forget") == "remove_assumption"
        assert v.resolve_alias("仮定を削除") == "remove_assumption"

    def test_radcan(self, v):
        assert v.resolve_alias("radcan") == "radical_simplify"
        assert v.resolve_alias("根号を簡約") == "radical_simplify"

    def test_partfrac(self, v):
        assert v.resolve_alias("partfrac") == "partial_fraction"
        assert v.resolve_alias("部分分数分解") == "partial_fraction"

    def test_trigexpand(self, v):
        assert v.resolve_alias("trigexpand") == "trig_expand"
        assert v.resolve_alias("三角展開") == "trig_expand"

    def test_exponentialize(self, v):
        assert v.resolve_alias("exponentialize") == "trig_to_exponential"
        assert v.resolve_alias("オイラー形式") == "trig_to_exponential"

    def test_realpart(self, v):
        assert v.resolve_alias("realpart") == "real_part"
        assert v.resolve_alias("実部") == "real_part"

    def test_imagpart(self, v):
        assert v.resolve_alias("imagpart") == "imaginary_part"
        assert v.resolve_alias("虚部") == "imaginary_part"

    def test_conjugate(self, v):
        assert v.resolve_alias("conjugate") == "complex_conjugate"
        assert v.resolve_alias("複素共役") == "complex_conjugate"

    def test_sum_alias(self, v):
        assert v.resolve_alias("sum") == "summation"
        assert v.resolve_alias("総和") == "summation"

    def test_product_alias(self, v):
        assert v.resolve_alias("product") == "product_expr"
        assert v.resolve_alias("総積") == "product_expr"

    def test_laplace(self, v):
        assert v.resolve_alias("laplace") == "laplace_transform"
        assert v.resolve_alias("ラプラス変換") == "laplace_transform"

    def test_ilt(self, v):
        assert v.resolve_alias("ilt") == "inverse_laplace"
        assert v.resolve_alias("逆ラプラス変換") == "inverse_laplace"

    def test_ode2(self, v):
        assert v.resolve_alias("ode2") == "solve_ode"
        assert v.resolve_alias("微分方程式を解く") == "solve_ode"

    def test_transpose(self, v):
        assert v.resolve_alias("transpose") == "transpose"
        assert v.resolve_alias("転置") == "transpose"

    def test_trace(self, v):
        assert v.resolve_alias("trace") == "trace"
        assert v.resolve_alias("トレース") == "trace"

    def test_float_alias(self, v):
        assert v.resolve_alias("float") == "numerical_eval"
        assert v.resolve_alias("数値評価") == "numerical_eval"

    def test_bfloat_alias(self, v):
        assert v.resolve_alias("bfloat") == "numerical_eval_hp"
        assert v.resolve_alias("高精度数値評価") == "numerical_eval_hp"


# ---------------------------------------------------------------------------
# Builder: template rendering for new operations
# ---------------------------------------------------------------------------

class TestNewTemplates:
    EXPR = {"E1": "x^2-1", "E2": "sin(x)"}

    def test_use_relation(self, b):
        r = b.build(CasOperation(operation="use_relation", args={"expr": "x^2-1", "relation_ref": "y=x+1"}))
        assert r.ok
        assert "subst" in r.command
        assert "y=x+1" in r.command

    def test_radical_simplify(self, b):
        r = b.build(CasOperation(operation="radical_simplify", args={"expr": "x^2-1"}))
        assert r.ok
        assert r.command == "radcan(x^2-1)"

    def test_partial_fraction(self, b):
        r = b.build(CasOperation(operation="partial_fraction", args={"expr": "x^2-1", "var": "x"}))
        assert r.ok
        assert "partfrac" in r.command

    def test_trig_expand(self, b):
        r = b.build(CasOperation(operation="trig_expand", args={"expr": "sin(x)"}))
        assert r.ok
        assert "trigexpand" in r.command

    def test_trig_to_exponential(self, b):
        r = b.build(CasOperation(operation="trig_to_exponential", args={"expr": "sin(x)"}))
        assert r.ok
        assert "exponentialize" in r.command

    def test_real_part(self, b):
        r = b.build(CasOperation(operation="real_part", args={"expr": "x^2-1"}))
        assert r.ok
        assert "realpart" in r.command

    def test_complex_conjugate(self, b):
        r = b.build(CasOperation(operation="complex_conjugate", args={"expr": "x^2-1"}))
        assert r.ok
        assert "conjugate" in r.command

    def test_summation(self, b):
        r = b.build(CasOperation(operation="summation", args={"expr": "x^2-1", "var": "k", "lower": "1", "upper": "n"}))
        assert r.ok
        assert r.command == "sum(x^2-1,k,1,n)"

    def test_product_expr(self, b):
        r = b.build(CasOperation(operation="product_expr", args={"expr": "x^2-1", "var": "k", "lower": "1", "upper": "n"}))
        assert r.ok
        assert r.command == "product(x^2-1,k,1,n)"

    def test_laplace_transform(self, b):
        r = b.build(CasOperation(operation="laplace_transform", args={"expr": "sin(x)", "t": "x", "s": "s"}))
        assert r.ok
        assert "laplace" in r.command

    def test_inverse_laplace(self, b):
        r = b.build(CasOperation(operation="inverse_laplace", args={"expr": "x^2-1", "s": "s", "t": "t"}))
        assert r.ok
        assert "ilt" in r.command

    def test_add_assumption(self, b):
        r = b.build(CasOperation(operation="add_assumption", args={"condition": "x>0"}))
        assert r.ok
        assert r.command == "assume(x>0)"

    def test_remove_assumption(self, b):
        r = b.build(CasOperation(operation="remove_assumption", args={"condition": "x>0"}))
        assert r.ok
        assert r.command == "forget(x>0)"

    def test_solve_ode(self, b):
        r = b.build(CasOperation(operation="solve_ode", args={"equation": "'diff(y,x)+y=0", "y": "y", "x": "x"}))
        assert r.ok
        assert "ode2" in r.command

    def test_transpose(self, b):
        r = b.build(CasOperation(operation="transpose", args={"expr": "x^2-1"}))
        assert r.ok
        assert "transpose" in r.command

    def test_matrix_multiply(self, b):
        r = b.build(CasOperation(operation="matrix_multiply", args={"a": "x^2-1", "b": "sin(x)"}))
        assert r.ok
        assert "." in r.command

    def test_numerical_eval(self, b):
        r = b.build(CasOperation(operation="numerical_eval", args={"expr": "sin(x)"}))
        assert r.ok
        assert "float" in r.command

    def test_numerical_eval_hp(self, b):
        r = b.build(CasOperation(operation="numerical_eval_hp", args={"expr": "sin(x)"}))
        assert r.ok
        assert "bfloat" in r.command


# ---------------------------------------------------------------------------
# Maxima integration: new operations actually work
# ---------------------------------------------------------------------------

class TestNewOpsMaxima:
    def test_radical_simplify(self, maxima):
        r = maxima.execute("radcan(sqrt(x^4))")
        assert r.ok
        assert "x^2" in r.value or "x" in r.value

    def test_partial_fraction(self, maxima):
        r = maxima.execute("partfrac(1/(x^2-1),x)")
        assert r.ok
        assert "x" in r.value

    def test_trig_expand(self, maxima):
        r = maxima.execute("trigexpand(sin(2*x))")
        assert r.ok
        assert "sin" in r.value or "cos" in r.value

    def test_trig_to_exponential(self, maxima):
        r = maxima.execute("exponentialize(sin(x))")
        assert r.ok
        assert "%e" in r.value or "exp" in r.value or "%i" in r.value

    def test_real_part(self, maxima):
        r = maxima.execute("realpart(3+2*%i)")
        assert r.ok
        assert "3" in r.value

    def test_imaginary_part(self, maxima):
        r = maxima.execute("imagpart(3+2*%i)")
        assert r.ok
        assert "2" in r.value

    def test_complex_conjugate(self, maxima):
        r = maxima.execute("conjugate(3+2*%i)")
        assert r.ok
        assert "3" in r.value

    def test_summation(self, maxima):
        r = maxima.execute("sum(k,k,1,5)")
        assert r.ok
        assert "15" in r.value

    def test_product_expr(self, maxima):
        r = maxima.execute("product(k,k,1,4)")
        assert r.ok
        assert "24" in r.value

    def test_laplace_transform(self, maxima):
        r = maxima.execute("laplace(exp(-t),t,s)")
        assert r.ok
        assert "s" in r.value

    def test_add_and_remove_assumption(self, maxima):
        r1 = maxima.execute("assume(x>0)")
        assert r1.ok
        r2 = maxima.execute("forget(x>0)")
        assert r2.ok

    def test_assumption_affects_integrate(self, maxima):
        maxima.execute("assume(x>0)")
        r = maxima.execute("integrate(1/x,x)")
        assert r.ok
        assert "log" in r.value
        maxima.execute("forget(x>0)")

    def test_numerical_eval(self, maxima):
        r = maxima.execute("float(%pi)")
        assert r.ok
        assert "3.14" in r.value

    def test_numerical_eval_hp(self, maxima):
        r = maxima.execute("bfloat(%pi)")
        assert r.ok
        assert "3." in r.value

    def test_use_relation_via_engine(self):
        with NotebookEngine() as e:
            e1 = e.register_expression("x+y")
            e2 = e.register_expression("y=x^2")
            _, val = e.run_operation("use_relation",
                                     {"expr": e1.display_id,
                                      "relation_ref": e2.display_id})
            assert "x^2" in val or "x" in val


class TestDeclareTimeDependent:
    def test_alias_ja(self, v):
        assert v.resolve_alias("時間依存を宣言") == "declare_time_dependent"
        assert v.resolve_alias("時間依存にする") == "declare_time_dependent"

    def test_alias_en(self, v):
        assert v.resolve_alias("declare_time_dependent") == "declare_time_dependent"
        assert v.resolve_alias("time_dependent") == "declare_time_dependent"

    def test_required_field(self, v):
        spec = v.get_spec("declare_time_dependent")
        assert "name" in spec["required_fields"]

    def test_optional_time_parameter(self, v):
        spec = v.get_spec("declare_time_dependent")
        assert "time_parameter" in spec["optional_fields"]

    def test_has_maxima_template(self, b):
        assert "declare_time_dependent" in b.supported_operations


class TestSetNotation:
    def test_alias_leibniz_ja(self, v):
        assert v.resolve_alias("Leibniz記法にする") == "set_notation"
        assert v.resolve_alias("d/dt記法にする") == "set_notation"
        assert v.resolve_alias("ライプニッツ記法にする") == "set_notation"

    def test_alias_dot_ja(self, v):
        assert v.resolve_alias("dot記法にする") == "set_notation"
        assert v.resolve_alias("ドット記法に戻す") == "set_notation"

    def test_alias_en(self, v):
        assert v.resolve_alias("set_notation") == "set_notation"
        assert v.resolve_alias("use_leibniz_notation") == "set_notation"
        assert v.resolve_alias("use_dot_notation") == "set_notation"

    def test_required_fields(self, v):
        spec = v.get_spec("set_notation")
        assert "name" in spec["required_fields"]
        assert "format" in spec["required_fields"]

    def test_has_maxima_template(self, b):
        assert "set_notation" in b.supported_operations


class TestSetLatexName:
    def test_alias_ja(self, v):
        assert v.resolve_alias("LaTeX名を設定") == "set_latex_name"
        assert v.resolve_alias("LaTeX名を変更") == "set_latex_name"
        assert v.resolve_alias("ラテックス名を設定") == "set_latex_name"

    def test_alias_en(self, v):
        assert v.resolve_alias("set_latex_name") == "set_latex_name"
        assert v.resolve_alias("set latex name") == "set_latex_name"

    def test_required_fields(self, v):
        spec = v.get_spec("set_latex_name")
        assert "name" in spec["required_fields"]
        assert "latex" in spec["required_fields"]

    def test_has_maxima_template(self, b):
        assert "set_latex_name" in b.supported_operations


class TestDefineSpinOperator:
    def test_alias_ja(self, v):
        assert v.resolve_alias("define_spin_operator") == "define_spin_operator"

    def test_required_fields(self, v):
        spec = v.get_spec("define_spin_operator")
        assert "name" in spec["required_fields"]

    def test_supported_backends(self, v):
        r"""Maxima も受ける（2026-08-22）。

        以前は julia_full 限定で、`magnetic_resonance`（Maxima）セッションでは
        「スピン演算子として定義して」がどの綴りでも既定のラベル推論に落ち、
        **スカラーとして登録されて成功が返っていた**（実測）。Maxima 側の
        テンプレートは no-op である——spin_algebra は CAS 非経由なので CAS に
        伝えるものが無い。それでも受けるのは「書けることと計算できることは
        別の層」だからで、断ると NMR セッションで自分のハミルトニアンを
        書き始められない。
        """
        spec = v.get_spec("define_spin_operator")
        assert "julia_full" in spec["supported_backends"]
        assert "maxima" in spec["supported_backends"]
