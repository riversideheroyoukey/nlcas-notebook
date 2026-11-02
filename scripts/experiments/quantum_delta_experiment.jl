# scripts/experiments/quantum_delta_experiment.jl
# Automatic experiment script for Kronecker delta / Dirac delta,
# index sums, integrals, and LaTeX output in QuantumAlgebra.jl

using QuantumAlgebra
using Symbolics
using Latexify
using Pkg

println("=========================================================================")
println("QuantumAlgebra.jl Delta / Sum Behavior Experiment")
println("=========================================================================")
println("Julia version: ", VERSION)
println("QuantumAlgebra version: (see status)")
println("Symbolics version: (see status)")
println("Latexify version: (see status)")
println("=========================================================================\n")

# Helper to print header
function print_header(title)
    println("\n-------------------------------------------------------------")
    println("### ", title)
    println("-------------------------------------------------------------")
end

# Helper to report test cases
function report_case(label, raw_expr, eval_func=normal_form)
    res = eval_func(raw_expr)
    println("* Case: ", label)
    println("  - Input raw: ", raw_expr)
    println("  - Output string(res): ", string(res))
    println("  - Output typeof(res): ", typeof(res))
    println("  - Output latexify(res): ", latexify(res))
end

# =========================================================================
# 実験 A: ボソン交換子と Kronecker delta
# =========================================================================
print_header("A. Bosonic commutators")

@boson_ops a
i = QuantumAlgebra.QuIndex(:i)
j = QuantumAlgebra.QuIndex(:j)

println("\n#### A1. Symbolic indices")
report_case("[a(i), a†(j)]", comm(a(i), a'(j)))
report_case("[a(i), a(j)]", comm(a(i), a(j)))
report_case("[a†(i), a†(j)]", comm(a'(i), a'(j)))

println("\n#### A2. Numeric indices")
report_case("[a(1), a†(1)]", comm(a(1), a'(1)))
report_case("[a(1), a†(2)]", comm(a(1), a'(2)))
report_case("[a(2), a†(1)]", comm(a(2), a'(1)))

# =========================================================================
# 実験 B: フェルミオン反交換子と Kronecker delta
# =========================================================================
print_header("B. Fermionic anticommutators")

@fermion_ops c
i_f = QuantumAlgebra.QuIndex(:i)
j_f = QuantumAlgebra.QuIndex(:j)
# Anticommutator logic helper
anticomm(x, y) = x*y + y*x

println("\n#### B1. Symbolic indices")
report_case("{c(i), c†(j)}", anticomm(c(i_f), c'(j_f)))
report_case("{c(i), c(j)}", anticomm(c(i_f), c(j_f)))
report_case("{c†(i), c†(j)}", anticomm(c'(i_f), c'(j_f)))

println("\n#### B2. Numeric indices")
report_case("{c(1), c†(1)}", anticomm(c(1), c'(1)))
report_case("{c(1), c†(2)}", anticomm(c(1), c'(2)))
report_case("{c(2), c†(1)}", anticomm(c(2), c'(1)))

# =========================================================================
# 実験 C: Kronecker delta と添字和
# =========================================================================
print_header("C. Kronecker delta and index sums")

s1 = QuantumAlgebra.sumindex(1)
s2 = QuantumAlgebra.sumindex(2)
i_idx = QuantumAlgebra.QuIndex(:i)
j_idx = QuantumAlgebra.QuIndex(:j)
k_idx = QuantumAlgebra.QuIndex(:k)

# Helplers to build QuExpr manually
function make_sum_expr(nsuminds, δs, params, bares_op=nothing)
    bares = isnothing(bares_op) ? QuantumAlgebra.BaseOpProduct(QuantumAlgebra.BaseOperator[]) : first(keys(bares_op.terms)).bares
    term = QuantumAlgebra.QuTerm(nsuminds, δs, params, QuantumAlgebra.ExpVal[], QuantumAlgebra.Corr[], bares)
    return QuantumAlgebra.QuExpr(Dict(term => 1))
end

# C1. Delta and general parameter sums
println("\n#### C1. Delta and parameter sums")

# 1. Σ_j δ_{ij} f_j
d_ij = QuantumAlgebra.δ(i_idx, s1)
p_f1 = first(keys(param(:f, s1).terms)).params[1]
sum_expr1 = make_sum_expr(1, [d_ij], [p_f1])
report_case("Σ_j δ_{ij} f_j", sum_expr1)

# 2. Σ_i δ_{ij} f_i
d_ji = QuantumAlgebra.δ(s1, j_idx)
sum_expr2 = make_sum_expr(1, [d_ji], [p_f1])
report_case("Σ_i δ_{ij} f_i", sum_expr2)

# 3. Σ_k δ_{ik} δ_{kj}
d_ik = QuantumAlgebra.δ(i_idx, s1)
d_kj = QuantumAlgebra.δ(s1, j_idx)
sum_expr3 = make_sum_expr(1, [d_ik, d_kj], QuantumAlgebra.Param[])
report_case("Σ_k δ_{ik} δ_{kj}", sum_expr3)

# C2. Sums inside commutators
println("\n#### C2. Sums inside commutators")

# 1. [a_i, Σ_j f_j a†_j]
op_adag_s1 = a'(s1)
sum_op1 = make_sum_expr(1, QuantumAlgebra.δ[], [p_f1], op_adag_s1)
report_case("[a_i, Σ_j f_j a†_j]", comm(a(i_idx), sum_op1))

# 2. [Σ_i f_i a_i, a†_j]
op_a_s1 = a(s1)
sum_op2 = make_sum_expr(1, QuantumAlgebra.δ[], [p_f1], op_a_s1)
report_case("[Σ_i f_i a_i, a†_j]", comm(sum_op2, a'(j_idx)))

# 3. [a_i, Σ_j A_{jk} a†_j]
p_A = first(keys(param(:A, s1, k_idx).terms)).params[1]
sum_op3 = make_sum_expr(1, QuantumAlgebra.δ[], [p_A], op_adag_s1)
report_case("[a_i, Σ_j A_{jk} a†_j]", comm(a(i_idx), sum_op3))

# =========================================================================
# 実験 D: 演算子積を含む交換子
# =========================================================================
print_header("D. Commutators with operator products")

i_op = a(i_idx)
j_op = a'(j_idx)
k_op = a(k_idx)
l_idx = QuantumAlgebra.QuIndex(:l)
l_op = a(l_idx)

println("\n#### D1. Simple products")
report_case("[a(i), a†(j) a(k)]", comm(i_op, j_op * k_op))
report_case("[a(i), a†(j) a†(k)]", comm(i_op, j_op * a'(k_idx)))
report_case("[a†(i) a(j), a†(k)]", comm(j_op * a(i_idx), a'(k_idx))) # Note: correct order
report_case("[a†(i) a(j), a(k)]", comm(a'(i_idx) * a(j_idx), k_op))

println("\n#### D2. Nested commutators")
report_case("[a(i), [a†(j), a(k)]]", comm(i_op, comm(j_op, k_op)))
report_case("[[a(i), a†(j)], a(k)]", comm(comm(i_op, j_op), k_op))
report_case("[a(i), [a†(j), a†(k) a(l)]]", comm(i_op, comm(j_op, a'(k_idx) * l_op)))

# =========================================================================
# 実験 E: Dirac delta 的な連続変数の扱い
# =========================================================================
print_header("E. Dirac delta and integrals")

@boson_ops ψ
x = QuantumAlgebra.QuIndex(:x)
y = QuantumAlgebra.QuIndex(:y)

println("\n#### E1. Field-like commutation relations")
report_case("[ψ(x), ψ†(y)]", comm(ψ(x), ψ'(y)))
report_case("{ψ(x), ψ†(y)}", anticomm(ψ(x), ψ'(y)))

println("\n#### E2. Continuous integration with Dirac delta (Conceptual checking)")
println("Checking if Symbolics.jl or SymbolicNumericIntegration.jl can support continuous Dirac delta:")
try
    @variables x_var y_var f_var(..)
    # Note: Dirac delta is not natively defined in Symbolics.jl, checking for integration of standard functions
    println("Symbolics.jl Integration capability:")
    @variables x_sym
    # Testing Symbolics integration
    # SymbolicNumericIntegration needs integrate()
    integ_expr, err, solved = integrate(x_sym^2, x_sym)
    println("  - Integral of x^2 wrt x: ", integ_expr, " (solved: ", solved, ")")
catch e
    println("  - Error or not supported natively: ", e)
end

# =========================================================================
# 実験 F: Latexify / notebook 表示への影響
# =========================================================================
print_header("F. Display implications")

println("\nSelected LaTeX strings for verification:")
println("1. Kronecker Delta: ", latexify(normal_form(comm(a(i), a'(j)))))
println("2. Double Delta Sum: ", latexify(sum_expr3))
println("3. Index Sum with Param: ", latexify(sum_expr1))
println("4. Nested Commutator Result: ", latexify(normal_form(comm(i_op, comm(j_op, a'(k_idx) * l_op)))))

println("\n=========================================================================")
println("Experiment Completed Successfully")
println("=========================================================================")
