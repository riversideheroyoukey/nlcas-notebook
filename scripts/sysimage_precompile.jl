using QuantumAlgebra
using Symbolics, Latexify, LinearAlgebra
using SymbolicNumericIntegration
import SymPy
import JSON3

# Exercise JSON3 so its serialize/deserialize paths are precompiled into the
# sysimage (the driver round-trips terms/rules/stats through JSON3).
JSON3.read(JSON3.write(Dict("x" => 1, "y" => [1, 2, 3])))

@boson_ops a
@fermion_ops f

expr = a(1) * a'(1)
normal_form(expr)
comm(a(1), a'(1))
normal_form(comm(a(1), a'(1)))

@syms x y
x^2 + y^2

SymPy.integrate(SymPy.sympify("x^2"), SymPy.sympify("x"))
SymPy.integrate(SymPy.sympify("exp(-x^2)"), (SymPy.sympify("x"), -SymPy.oo, SymPy.oo))
