# _julia_driver.jl — persistent Julia worker process for JuliaAdapter.
# Launched by JuliaAdapter.start() as:
#   julia --startup-file=no /path/to/_julia_driver.jl
#
# Protocol
# --------
# Startup  : loads packages, defines helpers, prints ready markers to stdout.
# Commands : one command per line from stdin; multi-statement lines use ";".
#            Each command is evaluated with eval(Meta.parse(cmd)).
# Response : per command —
#              <<<RESULT_BEGIN>>>
#              <string(result)>
#              <<<TEX_SEP>>>
#              <latex(result)>   (may be empty)
#              <<<RESULT_END>>>
#
# Auto-declare: when a command raises UndefVarError for a simple identifier,
# the driver automatically declares it as a Symbolics.jl @variables symbol
# and retries once.  This allows Maxima-style canonicals (e.g. "x^2 + 1")
# to work in Julia sessions without explicit variable declarations.

const _NB_BEGIN = "<<<RESULT_BEGIN>>>"
const _NB_END   = "<<<RESULT_END>>>"
const _NB_SEP   = "<<<TEX_SEP>>>"
const _NB_ERR   = "<<<JULIA_ERROR>>>"

using QuantumAlgebra
using Symbolics, Latexify, LinearAlgebra
using SymbolicNumericIntegration
using JSON3
import SymPy
import QuantumAlgebra as _QA
import Latexify as _Lfy

# Overloads to allow algebraic operations on QuantumAlgebra.δ directly.
import Base: *
function _nb_wrap_delta(d::QuantumAlgebra.δ)
    term = QuantumAlgebra.QuTerm(0, [d], QuantumAlgebra.Param[], QuantumAlgebra.ExpVal[], QuantumAlgebra.Corr[], QuantumAlgebra.BaseOpProduct(QuantumAlgebra.BaseOperator[]))
    return QuantumAlgebra.QuExpr(Dict(term => 1))
end
*(d::QuantumAlgebra.δ, x::QuantumAlgebra.QuExpr) = _nb_wrap_delta(d) * x
*(x::QuantumAlgebra.QuExpr, d::QuantumAlgebra.δ) = x * _nb_wrap_delta(d)
*(d1::QuantumAlgebra.δ, d2::QuantumAlgebra.δ) = _nb_wrap_delta(d1) * _nb_wrap_delta(d2)
*(d::QuantumAlgebra.δ, x::Number) = _nb_wrap_delta(d) * x
*(x::Number, d::QuantumAlgebra.δ) = x * _nb_wrap_delta(d)

# Pre-declare vector calculus functions so they are callable out-of-the-box
Symbolics.@syms grad(x) div(x) rot(x) curl(x)

function _nb_latex(x::_QA.QuExpr)
    _QA.latex(x)
end
function _nb_latex(x::Nothing)
    ""
end
function _nb_latex(x::AbstractMatrix)
    try
        string(_Lfy.latexify(x, env=:raw))
    catch
        string(x)
    end
end
function _nb_latex(x)
    try
        string(_Lfy.latexify(x, env=:raw))
    catch
        string(x)
    end
end

# Julia keywords and known built-in names that must not be declared as @variables.
const _NB_KEYWORDS = Set([
    "begin", "end", "for", "in", "if", "else", "elseif", "while", "return",
    "true", "false", "nothing", "using", "import", "function", "let", "try",
    "catch", "finally", "do", "struct", "mutable", "abstract", "type",
    "module", "baremodule", "macro", "quote", "local", "global", "const",
    "break", "continue", "export", "where", "isa", "typeof", "and", "or",
    # QuantumAlgebra / Symbolics builtins
    "normal_form", "comm", "vacExpVal", "heisenberg_eom",
    "expand", "simplify", "factor", "real", "imag", "conj",
    "string", "println", "print", "show", "param", "params",
    "det", "tr", "transpose", "inv", "sum", "prod", "map",
    "substitute", "Differential", "integrate", "limit", "taylor",
    "expand_derivatives", "solve_for", "Float64", "BigFloat", "big",
    "Dict", "Vector", "Matrix", "Symbolics", "LinearAlgebra",
    # Module names — must never be auto-declared as @variables
    "QuantumAlgebra", "_QA", "SymPy", "Latexify", "JSON3",
    # QuantumAlgebra index/param helpers
    "sumindex", "QuIndex", "acomm", "QuExpr", "QuTerm",
])

# Domain registry: maps variable name → Julia type annotation string.
# Populated by _nb_register_domain() when define_scalar_parameter is called
# with a domain argument.  _nb_autodeclare() consults this so that
# re-declarations after session restart use the correct type.
_NB_DOMAIN_REGISTRY = Dict{String, String}()

function _nb_register_domain(name::String, julia_type::String)
    _NB_DOMAIN_REGISTRY[name] = julia_type
end

# Auto-declare an undeclared identifier as a Symbolics @variables symbol.
# Consults _NB_DOMAIN_REGISTRY for typed declarations (e.g. ::Real, ::Integer).
# Only acts on simple identifiers (letters/digits/underscore, not Julia operators).
function _nb_autodeclare(var_name::Symbol)
    name_str = string(var_name)
    if name_str in _NB_KEYWORDS
        return false
    end
    if !isnothing(match(r"^[A-Za-z][A-Za-z0-9_]*$", name_str))
        try
            julia_type = get(_NB_DOMAIN_REGISTRY, name_str, "")
            decl = isempty(julia_type) ? "@variables $name_str" : "@variables $name_str::$julia_type"
            eval(Meta.parse(decl))
            return true
        catch
            return false
        end
    end
    return false
end

# Evaluate a command, auto-declaring Symbolics variables on UndefVarError.
# Recursively retries after each successful auto-declaration so arbitrarily many
# unknown variables (e.g. DeltaStar, u, v in a Bogoliubov substitution) can
# each be declared one by one without an artificial retry cap.
function _nb_eval(cmd::String, expr=Meta.parse(cmd))
    try
        return eval(expr)
    catch ex
        if isa(ex, UndefVarError) && _nb_autodeclare(ex.var)
            return _nb_eval(cmd, expr)  # retry with the same parsed expr
        end
        rethrow(ex)
    end
end

# Higher-order symbolic differentiation: apply Differential(var) `order` times,
# expanding derivatives at each step.  The template-based differentiate operation
# only emitted a single Differential, ignoring the order argument; this helper
# makes "N 回微分" (order > 1) compute the correct N-th derivative.
function _nb_differentiate(expr, var, order::Int=1)
    result = expr
    D = Differential(var)
    for _ in 1:order
        result = expand_derivatives(D(result))
    end
    result
end

# Result wrapper for solve: carries both a canonical string and a LaTeX string.
# This lets _nb_latex() produce proper LaTeX even when the value came from SymPy.
struct _NbSolveResult
    var::String
    value_str::String   # canonical (Julia-style) string
    latex_str::String   # LaTeX string
end
Base.string(r::_NbSolveResult) = r.var * " = " * r.value_str
function _nb_latex(r::_NbSolveResult)
    r.var * " = " * r.latex_str
end

# Solve a single-variable equation expressed as two canonical strings.
# Tries Symbolics.solve_for first (linear only); falls back to SymPy.solve
# for nonlinear equations (quadratic, etc.).
# Returns _NbSolveResult so that the LaTeX output is always well-formed.
function _nb_solve_equation(lhs_str::String, rhs_str::String, var_str::String)
    var_sym = eval(Meta.parse(var_str))
    lhs_expr = eval(Meta.parse(lhs_str))
    rhs_expr = eval(Meta.parse(rhs_str))
    sol_symbolics = try
        Symbolics.solve_for(lhs_expr ~ rhs_expr, var_sym)
    catch
        nothing
    end
    if sol_symbolics !== nothing
        val_str = string(sol_symbolics)
        lat_str = try string(_Lfy.latexify(sol_symbolics, env=:raw)) catch; val_str end
        return _NbSolveResult(var_str, val_str, lat_str)
    end
    # Nonlinear: delegate to SymPy
    _sp_var = SymPy.symbols(var_str)
    _sp_expr = SymPy.sympify(lhs_str) - SymPy.sympify(rhs_str)
    _sp_sols = SymPy.solve(_sp_expr, _sp_var)
    isempty(_sp_sols) && error("no solution for " * var_str)
    # Pick the most positive (last) real solution as the principal value.
    _sp_sol = _sp_sols[end]
    # Combine product of sqrts at the SymPy level: sqrt(2)*sqrt(m)*sqrt(mu) → sqrt(2*m*mu)
    _sp_sol_simp = try SymPy.powsimp(_sp_sol, force=true) catch; _sp_sol end
    val_str = string(_sp_sol_simp)
    # Leave lat_str empty; Python-side postprocess_result_latex will convert
    # SymPy canonical (sqrt(X), a/b) to LaTeX via _sympy_canonical_to_latex.
    _NbSolveResult(var_str, val_str, "")
end

# Signal ready.
println(_NB_BEGIN)
println("julia_ready")
println(_NB_SEP)
println("")
println(_NB_END)
flush(stdout)

# ------------------------------------------------------------------
# Hyper-efficient hybrid helpers for continuous QFT integration
# and discrete index sum contraction.
# ------------------------------------------------------------------

function _nb_parse_limit(s::String)
    s_clean = strip(s)
    if s_clean == "-Inf" || s_clean == "-oo"
        return -SymPy.oo
    elseif s_clean == "Inf" || s_clean == "oo"
        return SymPy.oo
    else
        return SymPy.sympify(s_clean)
    end
end

function _nb_eval_continuous_integral(expr_str::String, var_str::String, lower_str::String="", upper_str::String="")
    # Normalise "delta(x-y)" and "SymPy.sympy.DiracDelta" to "DiracDelta(x-y)" for SymPy compatibility
    clean_expr = replace(expr_str, "SymPy.sympy.DiracDelta" => "DiracDelta")
    clean_expr = replace(clean_expr, r"\bdelta\b" => "DiracDelta")
    expr = SymPy.sympify(clean_expr)
    var = SymPy.sympify(var_str)
    if isempty(lower_str) || isempty(upper_str)
        res = SymPy.integrate(expr, var)
    else
        lower = _nb_parse_limit(lower_str)
        upper = _nb_parse_limit(upper_str)
        res = SymPy.integrate(expr, (var, lower, upper))
    end
    return res
end

function _nb_eval_index_sum(expr_str::String, dummy_syms_str::String)
    dummy_syms = [strip(s) for s in split(dummy_syms_str, ",")]
    expr_mut = expr_str
    for (idx, sym) in enumerate(dummy_syms)
        # Replace :sym with _QA.sumindex(idx) — use the module alias to avoid
        # any shadowing of the QuantumAlgebra name in the Main scope.
        expr_mut = replace(expr_mut, ":" * sym => "_QA.sumindex($idx)")
        # Static-index-mode generates _QA.QuIndex("sym") for bound indices.
        # Replace those too so the sum is properly evaluated.
        expr_mut = replace(expr_mut, "_QA.QuIndex(\"$sym\")" => "_QA.sumindex($idx)")
        # Cleanup any legacy QuIndex(_QA.sumindex(...)) wrapping
        expr_mut = replace(expr_mut, "QuIndex(_QA.sumindex($idx))" => "_QA.sumindex($idx)")
    end

    result = _nb_eval(expr_mut)
    return normal_form(result)
end

# Evaluate Kronecker deltas whose both indices are concrete QuIndex labels (not sumindices).
# QuantumAlgebra keeps δ(QuIndex("1"), QuIndex("2")) symbolic even though 1 ≠ 2.
# This helper drops terms where two distinct concrete labels appear in a delta factor.
#
# Safety: sumindex has .num > 0 (concrete integer run-time index), while a named
# QuIndex from QuIndex("k") has .num == typemin(Int32) = -2147483648.
# Only concrete-label pairs are evaluated; sumindex pairs are left symbolic.
function _nb_eval_site_deltas(expr::QuantumAlgebra.QuExpr)
    # Only evaluate δ(i,j) = 0 when BOTH indices are concrete INTEGER labels
    # (e.g. QuIndex("1"), QuIndex("2")).  Abstract letter labels like QuIndex("r"),
    # QuIndex("k") are deliberately excluded: δ(r,s) inside Σ_r Σ_s must remain
    # symbolic so the sifting property is not prematurely applied.
    #
    # Discrimination:
    #   sumindex(n)    → idx.num > 0          → skip (runtime sum variable)
    #   QuIndex("k")   → idx.num == typemin(Int32), sym = :k, NOT parseable as Int
    #   QuIndex("1")   → idx.num == typemin(Int32), sym = :1, parseable as Int → target
    _is_named(idx) = idx.num == typemin(Int32)
    _is_integer_site(idx) =
        _is_named(idx) && tryparse(Int, string(idx.sym)) !== nothing

    # Negation partners: a momentum -k is encoded as the UPPER-CASE single-letter
    # label "K" (see JuliaCodeGenerator._gen_index: inner.upper()).  Hence a δ
    # whose two indices are the same letter differing only in case — e.g. δ(k,K) —
    # is δ(k,-k) = δ(2k,0), which vanishes away from the zone centre.  Dropping it
    # encodes the standard k-space assumption k≠0 and makes the QA path agree with
    # the SymPy routing, which already evaluates δ(-k,k)=0.  Only abstract named
    # labels qualify (sumindex dummies and integer sites are handled separately),
    # and different letters (δ(r,s) in Σ_r Σ_s) are deliberately untouched.
    # Two DIFFERENT projection labels are two different values.  Such a label
    # cannot be a bound dummy — nobody writes Σ_↑ or Σ_⁺ — so δ(↑,↓) = 0 carries
    # exactly the warrant that δ(1,2) = 0 does above, and NOT the one δ(r,s)
    # lacks.  Without this the simplest Hubbard commutator came back carrying
    # δ_{↑↓} terms the researcher knows are zero (measured 2026-08-22).
    #
    # This asks a PROPERTY, not a list (2026-09-12).  It used to be the literal
    # tuple ("↑","↓") — a second copy of the Python-side projection table, which
    # would have to grow in step every time a value set gained a spelling
    # (integer spin is spelled ⁻ ⁰ ⁺).  The property is the warrant itself: a
    # single character that is neither a LETTER (k, r, σ — those can be bound by
    # a sum, and δ(r,s) must keep its sifting property) nor a DIGIT (an integer
    # site, handled by _is_integer_site above).  Nothing else can be written
    # under a Σ, so nothing else is a variable.
    _is_projection_label(s) =
        length(s) == 1 && !isletter(first(s)) && !isdigit(first(s))
    _is_distinct_spin(a, b) = begin
        _is_named(a) && _is_named(b) || return false
        sa = string(a.sym); sb = string(b.sym)
        sa != sb && _is_projection_label(sa) && _is_projection_label(sb)
    end

    _is_sign_partner(a, b) = begin
        _is_named(a) && _is_named(b) || return false
        sa = string(a.sym); sb = string(b.sym)
        length(sa) == 1 && length(sb) == 1 &&
            isletter(sa[1]) && isletter(sb[1]) &&
            sa != sb && lowercase(sa) == lowercase(sb)
    end

    new_terms = Dict{QuantumAlgebra.QuTerm, Number}()
    for (term, coeff) in expr.terms
        keep = true
        for d in term.δs
            if (_is_integer_site(d.iA) && _is_integer_site(d.iB) && d.iA != d.iB) ||
               _is_distinct_spin(d.iA, d.iB) ||
               _is_sign_partner(d.iA, d.iB)
                keep = false; break
            end
        end
        if keep
            new_terms[term] = get(new_terms, term, 0) + coeff
        end
    end
    normal_form(QuantumAlgebra.QuExpr(new_terms))
end

# Fallback: non-QuExpr values (scalars, numbers) pass through unchanged.
_nb_eval_site_deltas(expr) = expr

# Vacuum expectation value ⟨0|…|0⟩ computed directly from the normal-ordered
# QuExpr as the sum of its operator-free terms (empty `bares`).
#
# Why not QuantumAlgebra's vacExpVal: vacExpVal throws "non-boolean (Num) used
# in boolean context" whenever a Symbolics scalar parameter is present — i.e. on
# essentially every physics expression (u, v, ξ, Δ, …).  Selecting operator-free
# terms ourselves keeps symbolic coefficients intact and returns a QuExpr (so the
# result still formats/chains like any other expression).
function _nb_vac_expect(expr::QuantumAlgebra.QuExpr)
    nf = normal_form(expr)
    keep = Dict{QuantumAlgebra.QuTerm, Any}()
    for (term, coeff) in nf.terms
        isempty(term.bares) || continue   # any creation/annihilation op ⇒ vanishes in vacuum
        keep[term] = get(keep, term, 0) + coeff
    end
    normal_form(QuantumAlgebra.QuExpr(keep))
end

# Fallback: scalars / numbers are their own vacuum expectation.
_nb_vac_expect(expr) = expr

# Substitute scalar parameters in a QuantumAlgebra QuExpr.
# Iterates over .terms dict and applies Symbolics.substitute() to each
# coefficient, then reconstructs the QuExpr using the same dict type.
# Substitute scalar parameters in a QuExpr.
# In QuantumAlgebra, scalar parameters (param(:xi, ind)) are stored as Param
# objects in QuTerm.params, NOT as Symbolics variables in the numeric coefficient.
# Usage: _nb_scalar_subst(expr, param(:xi, QuIndex("K")), param(:xi, QuIndex("k")))
function _nb_scalar_subst(expr::QuantumAlgebra.QuExpr, old_qu::QuantumAlgebra.QuExpr, new_qu::QuantumAlgebra.QuExpr)
    old_p = only(keys(old_qu.terms)).params[1]
    new_p = only(keys(new_qu.terms)).params[1]
    new_terms = Dict{QuantumAlgebra.QuTerm, Number}()
    for (term, coeff) in expr.terms
        new_ps = [p == old_p ? new_p : p for p in term.params]
        new_term = QuantumAlgebra.QuTerm(term.nsuminds, term.δs, new_ps, term.expvals, term.corrs, term.bares)
        new_terms[new_term] = get(new_terms, new_term, 0) + coeff
    end
    return normal_form(QuantumAlgebra.QuExpr(new_terms))
end


# ==========================================================================
# SymPy-based operator algebra engine for algebraic momentum indices.
#
# QuantumAlgebra.jl uses categorical single-char QuIndex labels and cannot
# handle arithmetic index expressions such as -k, k+q, or a*k+b.  These
# helpers implement the same commutator/normal-order logic using SymPy
# (already available via SymPy.PyCall) so that condensed-matter, BEC, and
# lattice-physics computations work correctly.
#
# Note: _nb_sympy_sum_fermion_comm (below) handles IndexSum-wrapped H.
#   The Python side (JuliaCodeGenerator._generate_sympy_sum_commutator) serializes
#   the IndexSum IR into JSON format {"coeff": "...", "ops": "..."} and calls it.
#
# Design
# ------
# _nb_sympy_fermion_comm  –  [A₁A₂…Aₙ, C] for fermionic operators
# _nb_sympy_boson_comm    –  [A₁A₂…Aₙ, C] for bosonic operators
#
# Operator notation used in string arguments
#   cd(p)  = c†(p)  fermionic creation
#   c(p)   = c(p)   fermionic annihilation
#   ad(p)  = a†(p)  bosonic creation
#   a(p)   = a(p)   bosonic annihilation
#
# All operators must be written as  <name>(<index>)  with no spaces inside
# parentheses, and separated by  *  in products.  The index may be any
# arithmetic expression built from the declared variables, e.g. -k, a*k+b.
# ==========================================================================

# Build a Python evaluation context from a comma-separated variable list.
# The returned Dict maps each name to a SymPy Symbol and includes
# __builtins__ so that Python arithmetic (-k, a*k+b, …) works in eval().
function _nb_sympy_ctx(vars_str::String)
    sp  = SymPy.PyCall.pyimport("sympy")
    ctx = Dict{String, Any}(
        "__builtins__" => SymPy.PyCall.pyimport("builtins"),
        # SymPy built-ins that may appear in serialized scalar coefficients.
        "conjugate" => sp.conjugate,
        "conj"      => sp.conjugate,
        "re"        => sp.re,
        "im"        => sp.im,
    )
    for v in split(vars_str, ",")
        v = strip(v)
        isempty(v) && continue
        # All user-defined names → plain SymPy Symbols.
        # Previously multi-char names were made into SymPy Functions (for
        # "xi(k)" call syntax), but scalar coeff strings now use underscore
        # notation ("xi_k") so no Function objects are needed in the context.
        ctx[v] = sp.Symbol(v)
    end
    return ctx
end

# ---------------------------------------------------------------------------
# _nb_sum_comm helpers
# ---------------------------------------------------------------------------

"""Extract the index name from a target string like "c(k)" → "k", "c(-k)" → "-k"."""
function _nb_extract_target_idx(target_str::String)::String
    m = match(r"[A-Za-z_]\w*\(([^)]+)\)", target_str)
    m === nothing ? "" : String(strip(String(m.captures[1])))
end

"""
Replace whole-word occurrences of `old` with `new_` in string `s`.
Uses PCRE look-around so that e.g. "j" in "cd(j)" is replaced but "jk" is not.
`old` must be a plain identifier name (no regex metacharacters).
"""
function _nb_word_replace(s::String, old::String, new_::String)
    replace(s, Regex("(?<![A-Za-z_0-9])" * old * "(?![A-Za-z_0-9])") => new_)
end

# ---------------------------------------------------------------------------

# Parse "cd(k)*c(-k)" into an ordered list of (op_type, SymPy_index) tuples.
# For single-index operators only (used by normal_order helpers).
function _nb_parse_op_str(expr_str::String, ctx::Dict)
    builtins = SymPy.PyCall.pyimport("builtins")
    ops = Tuple{String, Any}[]
    for m in eachmatch(r"(cd|c|ad|a)\(([^)]+)\)", replace(expr_str, " " => ""))
        op_type = String(m.captures[1])
        idx_str = String(m.captures[2])
        idx = builtins.eval(idx_str, ctx)
        push!(ops, (op_type, idx))
    end
    return ops
end

# Parse "ad(k,u)*a(-k,d)" into (op_type, [idx0, idx1, ...]) — multi-index aware.
# Used by _nb_sympy_wick to support spin+momentum operators like a'(k,σ).
# Each index is evaluated independently so INDEX1_0, INDEX1_1 work correctly.
function _nb_parse_op_str_multi(expr_str::String, ctx::Dict)
    builtins = SymPy.PyCall.pyimport("builtins")
    ops = Tuple{String, Vector{Any}}[]
    for m in eachmatch(r"(cd|c|ad|a)\(([^)]+)\)", replace(expr_str, " " => ""))
        op_type = String(m.captures[1])
        idx_str = String(m.captures[2])
        # Split by commas to evaluate each index independently
        idx_parts = [strip(s) for s in split(idx_str, ",")]
        indices = Any[]
        for part in idx_parts
            idx = try builtins.eval(part, ctx) catch; SymPy.PyCall.pyimport("sympy").Symbol(part) end
            push!(indices, idx)
        end
        push!(ops, (op_type, indices))
    end
    return ops
end

"""
    _nb_sympy_fermion_comm(prod_str, target_str, vars_str)

Compute the fermionic commutator [A₁A₂…Aₙ, C] using the identity

    [A₁…Aₙ, C] = Σᵢ (−1)^(n−i) {Aᵢ, C} ∏_{j≠i} Aⱼ

where anti-commutation rules are
    {c†(p), c(q)} = {c(p), c†(q)} = δ(p, q)
    {c(p),  c(q)} = {c†(p), c†(q)} = 0

Arguments
---------
prod_str   : ordered product of fermionic operators, e.g. `"cd(k)*cd(-k)"`
target_str : single target operator,               e.g. `"c(q)"`
vars_str   : comma-separated SymPy symbol names,   e.g. `"k,q"`

Returns
-------
Tuple (value_str, latex_str) — SymPy string and LaTeX representations.

Notes
-----
Unlike QuantumAlgebra, this helper supports algebraic index expressions
(-k, k+q, a*k+b, …).  The KroneckerDelta terms in the result are left
symbolic and can be evaluated further with substitute() operations.
"""
function _nb_sympy_fermion_comm(prod_str::String, target_str::String, vars_str::String)
    sp  = SymPy.PyCall.pyimport("sympy")
    ctx = _nb_sympy_ctx(vars_str)
    result = _nb_sympy_fermion_comm_inner(prod_str, target_str, sp, ctx)
    result = sp.expand(result)
    return (sp.sstr(result), sp.latex(result))
end

"""
    _nb_sympy_boson_comm(prod_str, target_str, vars_str)

Compute the bosonic commutator [A₁A₂…Aₙ, C] using the identity

    [A₁…Aₙ, C] = Σᵢ (∏_{j<i} Aⱼ) [Aᵢ, C] (∏_{j>i} Aⱼ)

where commutation rules are
    [a(p), a†(q)] = δ(p, q)
    [a†(p), a(q)] = −δ(p, q)
    [a(p), a(q)]  = [a†(p), a†(q)] = 0

Operator notation: `ad(p)` = a†(p),  `a(p)` = a(p).
Same argument format as _nb_sympy_fermion_comm.
"""
function _nb_sympy_boson_comm(prod_str::String, target_str::String, vars_str::String)
    sp  = SymPy.PyCall.pyimport("sympy")
    ctx = _nb_sympy_ctx(vars_str)
    result = _nb_sympy_boson_comm_inner(prod_str, target_str, sp, ctx)
    result = sp.expand(result)
    return (sp.sstr(result), sp.latex(result))
end


"""
    _nb_sympy_sum_fermion_comm(terms_json, target_str, vars_str)

Compute the commutator [Σᵢ cᵢ Aᵢ, C] = Σᵢ cᵢ [Aᵢ, C] where each term has a
scalar coefficient cᵢ and an operator product Aᵢ.

Arguments
---------
terms_json : JSON-encoded array of objects {"coeff": "...", "ops": "..."}
             coeff is a SymPy-evaluable expression (e.g. "xi(k)", "Delta", "-1")
             ops   is an operator product string like "cd(k)*cd(-k)"
target_str : single target operator, e.g. "c(k)"
vars_str   : comma-separated SymPy symbol names, e.g. "k,q,xi,Delta"

Returns
-------
Tuple (value_str, latex_str)
"""
function _nb_sympy_sum_fermion_comm(terms_json::String, target_str::String, vars_str::String)
    sp  = SymPy.PyCall.pyimport("sympy")
    ctx = _nb_sympy_ctx(vars_str)
    builtins = SymPy.PyCall.pyimport("builtins")

    # ── Dummy-variable identification ──────────────────────────────────────
    # The target index (e.g. "k" from "c(k)") is a fixed external momentum.
    # Any variable in vars_str that does NOT appear in target_str is a
    # summation dummy (e.g. "j" after alphatize renames the pairing sum).
    # We evaluate the sum at the target momentum by substituting j → target_idx.
    tgt_idx_name = _nb_extract_target_idx(target_str)
    all_vars     = [String(strip(v)) for v in split(vars_str, ",") if !isempty(strip(v))]
    tgt_chars    = Set{String}(m.match for m in eachmatch(r"[A-Za-z_]\w*", target_str))
    dummy_vars   = [v for v in all_vars if v ∉ tgt_chars]

    # Make the target-index symbol nonzero so KroneckerDelta(-k, k) = 0.
    if !isempty(tgt_idx_name) && haskey(ctx, tgt_idx_name)
        ctx[tgt_idx_name] = sp.Symbol(tgt_idx_name, nonzero=true)
    end
    # ───────────────────────────────────────────────────────────────────────

    total = sp.Integer(0)
    terms = JSON3.read(terms_json)

    for term in terms
        ops_str   = String(term["ops"])
        coeff_str = String(term["coeff"])

        # Substitute each dummy variable → target index name.
        # This selects the j = target_idx term from Σ_j coeff(j)·[ops(j), target],
        # which is the physically relevant contribution in a BCS-style sum.
        subst_ops   = ops_str
        subst_coeff = coeff_str
        for dummy in dummy_vars
            isempty(tgt_idx_name) && continue
            subst_ops   = _nb_word_replace(subst_ops,   dummy, tgt_idx_name)
            subst_coeff = _nb_word_replace(subst_coeff, dummy, tgt_idx_name)
        end

        coeff = try
            builtins.eval(subst_coeff, ctx)
        catch
            # Compound coeff strings like "-1*Delta" or "-1*conjugate_Delta"
            # cannot be eval'd when the parameter names are not in ctx (which
            # only contains index variables like j, k).  Create SymPy Symbols
            # for every unknown identifier in subst_coeff and re-evaluate so
            # that e.g. "-1*Delta" → -Symbol("Delta") rather than the incorrect
            # single-symbol Symbol("-1*Delta").
            extra_ctx = copy(ctx)
            for m in eachmatch(r"[A-Za-z_]\w*", subst_coeff)
                name = String(m.match)
                if !haskey(extra_ctx, name)
                    extra_ctx[name] = sp.Symbol(name)
                end
            end
            try
                builtins.eval(subst_coeff, extra_ctx)
            catch
                sp.Symbol(subst_coeff)  # absolute last resort
            end
        end

        comm_result = _nb_sympy_fermion_comm_inner(subst_ops, target_str, sp, ctx)
        # sp.expand per term ensures sign products like (-coeff)*(-op) → coeff*op
        # are fully reduced even on older SymPy versions that keep them as-is.
        total = total + sp.expand(coeff * comm_result)
    end

    result = sp.expand(total)
    return (sp.sstr(result), sp.latex(result))
end


"""
    _nb_sympy_sum_boson_comm(terms_json, target_str, vars_str)

Bosonic version of _nb_sympy_sum_fermion_comm.
"""
function _nb_sympy_sum_boson_comm(terms_json::String, target_str::String, vars_str::String)
    sp  = SymPy.PyCall.pyimport("sympy")
    ctx = _nb_sympy_ctx(vars_str)
    builtins = SymPy.PyCall.pyimport("builtins")

    tgt_idx_name = _nb_extract_target_idx(target_str)
    all_vars     = [String(strip(v)) for v in split(vars_str, ",") if !isempty(strip(v))]
    tgt_chars    = Set{String}(m.match for m in eachmatch(r"[A-Za-z_]\w*", target_str))
    dummy_vars   = [v for v in all_vars if v ∉ tgt_chars]
    if !isempty(tgt_idx_name) && haskey(ctx, tgt_idx_name)
        ctx[tgt_idx_name] = sp.Symbol(tgt_idx_name, nonzero=true)
    end

    terms = JSON3.read(terms_json)
    total = sp.Integer(0)
    for term in terms
        ops_str   = String(term["ops"])
        coeff_str = String(term["coeff"])
        subst_ops   = ops_str
        subst_coeff = coeff_str
        for dummy in dummy_vars
            isempty(tgt_idx_name) && continue
            subst_ops   = _nb_word_replace(subst_ops,   dummy, tgt_idx_name)
            subst_coeff = _nb_word_replace(subst_coeff, dummy, tgt_idx_name)
        end
        coeff = try
            builtins.eval(subst_coeff, ctx)
        catch
            # Same robust fallback as fermion version: add unknown identifiers
            # as SymPy Symbols so "-1*Delta" → -Symbol("Delta").
            extra_ctx = copy(ctx)
            for m in eachmatch(r"[A-Za-z_]\w*", subst_coeff)
                name = String(m.match)
                if !haskey(extra_ctx, name)
                    extra_ctx[name] = sp.Symbol(name)
                end
            end
            try
                builtins.eval(subst_coeff, extra_ctx)
            catch
                sp.Symbol(subst_coeff)
            end
        end
        comm_result = _nb_sympy_boson_comm_inner(subst_ops, target_str, sp, ctx)
        total = total + sp.expand(coeff * comm_result)
    end
    result = sp.expand(total)
    return (sp.sstr(result), sp.latex(result))
end


# ---------------------------------------------------------------------------
# _nb_sympy_normal_order(expr_str, stats, vars_str)
#
# Compute the normal-ordered form of a fermionic/bosonic operator expression
# using SymPy's physics.secondquant module (NO() operator).
#
# Arguments:
#   expr_str : SymPy-evaluatable product string, e.g. "c(-k)*cd(k)"
#              Uses same cd/c (or ad/a for bosons) notation as _nb_sympy_fermion_comm.
#   stats    : "fermion" or "boson"
#   vars_str : comma-separated symbolic variable names, e.g. "k,q,p"
#
# Returns: (value_str, latex_str)  ← same format as _nb_sympy_*_comm
#
# Key feature: KroneckerDelta(k, -k) is correctly evaluated to 0 when k is
# declared nonzero, which QA's large-K workaround cannot achieve.
# ---------------------------------------------------------------------------
function _nb_sympy_normal_order(expr_str::String, stats_json::String, vars_str::String)
    sp = SymPy.PyCall.pyimport("sympy")
    ctx = _nb_sympy_ctx(vars_str)

    # Declare all momentum variables as nonzero so KroneckerDelta(k, -k) → 0.
    # This is the key feature: δ(-k, k) = 0 because k ≠ 0.
    for v in split(vars_str, ",")
        vt = strip(v)
        isempty(vt) && continue
        ctx[vt] = sp.Symbol(vt, nonzero=true)
    end

    # Parse per-operator statistics from JSON dict:
    #   '{"c":"fermion","a":"boson"}' → stats_map["c"]="fermion", etc.
    # Backward compat: plain "fermion"/"boson" string → all operators use that stats.
    stats_map = Dict{String,String}()
    if startswith(strip(stats_json), "{")
        try
            stats_map = JSON3.read(stats_json, Dict{String,String})
        catch
            stats_map = Dict{String,String}()
        end
    else
        # Legacy single-string: treat all operators as that stats
        st = strip(stats_json)
        stats_map["__default__"] = st
    end
    # Helper: look up stats for a base operator name
    _stats_of(base::String) = get(stats_map, base, get(stats_map, "__default__", "fermion"))

    # Split into additive terms and process each separately.
    # This handles multi-term inputs like "Delta*cd(-k) - xi*c(k)"
    # as well as single products like "-xi*c(-k)*cd(k)".
    sum_terms = _nb_split_sum_str(String(strip(expr_str)))

    all_pieces_val   = String[]
    all_pieces_latex = String[]

    for sum_term in sum_terms
        ops = _nb_parse_op_str(sum_term, ctx)  # [(op_type, idx), ...]

        # Extract scalar coefficient by replacing operator calls with 1.
        no_ops_str = replace(sum_term, r"(?:cd|c|ad|a)\([^)]+\)" => "1")
        # A term this helper cannot READ is not a scalar coefficient of 1.
        # Swallowing the parse failure here is exactly how the braced
        # multi-index spelling `c'_{k+q,u} c_{k,u}` — the one CLAUDE.md
        # documents and the QA index error itself recommends — came back as
        # the literal 1, ok=true, with no error anywhere (measured 2026-08-07).
        # The caller is expected to send SymPy operator syntax; if it did not,
        # say so instead of inventing a coefficient.
        scalar_coeff = try
            sp.sympify(no_ops_str)
        catch err
            error("_nb_sympy_normal_order: cannot read \"$(sum_term)\" as a " *
                  "SymPy expression (operator calls removed: \"$(no_ops_str)\"). " *
                  "Expected operator syntax such as cd(k+q,u)*c(k,u). " *
                  "Underlying error: $(err)")
        end

        if isempty(ops)
            # Pure scalar term (no operators)
            c_val = sp.sstr(sp.expand(scalar_coeff))
            c_tex = string(SymPy.PyCall.pycall(sp.latex, String, sp.expand(scalar_coeff)))
            c_val != "0" && push!(all_pieces_val, c_val)
            c_tex != "0" && push!(all_pieces_latex, c_tex)
            continue
        end

        terms = _nb_sympy_no_mixed(ops, stats_map, sp)

        # Format output manually to preserve operator ordering (SymPy.Mul reorders).
        # Each term: (scalar_coeff::SymPy, ops_list::[(String, Any)])
        pieces_val   = String[]
        pieces_latex = String[]

        for (coeff, tops) in terms
            coeff_s = sp.expand(scalar_coeff * coeff)  # include physical scalar
            coeff_s == sp.Integer(0) && continue  # δ(k,-k) = 0 drops the term

        op_val_str   = join(["$(t)($(_nb_idx_val_str(idx, sp)))"   for (t, idx) in tops], "*")
        op_latex_str = join(["$(t)($(_nb_idx_latex_str(idx, sp)))" for (t, idx) in tops], " ")

        c_str = sp.sstr(coeff_s)
        c_tex = sp.latex(coeff_s)

        if isempty(tops)
            push!(pieces_val,   c_str)
            push!(pieces_latex, c_tex)
        elseif c_str == "1"
            push!(pieces_val,   op_val_str)
            push!(pieces_latex, op_latex_str)
        elseif c_str == "-1"
            push!(pieces_val,   "-" * op_val_str)
            push!(pieces_latex, "-" * op_latex_str)
        else
            push!(pieces_val,   c_str * "*" * op_val_str)
            push!(pieces_latex, "(" * c_tex * ")" * op_latex_str)
        end
    end

        append!(all_pieces_val, pieces_val)
        append!(all_pieces_latex, pieces_latex)
    end  # end sum_term loop

    isempty(all_pieces_val) && return ("0", "0")

    value_str = _nb_join_terms(all_pieces_val)

    # Rebuild latex_str using SymPy-format operator notation
    # (re-run over all sum terms to get proper \\operatorname{cd}{...} format)
    function _op_to_latex(t::String, idx_sym)::String
        idx_latex = _nb_idx_latex_str(idx_sym, sp)
        t in ("cd", "ad") ? "\\operatorname{$t}{\\left($idx_latex \\right)}" :
                             "$t{\\left($idx_latex \\right)}"
    end

    all_pieces_latex2 = String[]
    for sum_term in sum_terms
        ops2 = _nb_parse_op_str(sum_term, ctx)
        no_ops2 = replace(sum_term, r"(?:cd|c|ad|a)\([^)]+\)" => "1")
        sc2 = try sp.sympify(no_ops2) catch; sp.Integer(1) end

        if isempty(ops2)
            c_tex2 = string(SymPy.PyCall.pycall(sp.latex, String, sp.expand(sc2)))
            c_tex2 != "0" && push!(all_pieces_latex2, c_tex2)
            continue
        end

        terms2 = _nb_sympy_no_mixed(ops2, stats_map, sp)
        for (coeff2, tops2) in terms2
            coeff_s2 = sp.expand(sc2 * coeff2)
            coeff_s2 == sp.Integer(0) && continue
            op_latex2 = join([_op_to_latex(t, idx) for (t, idx) in tops2], " ")
            c_tex2 = string(SymPy.PyCall.pycall(sp.latex, String, coeff_s2))
            if isempty(tops2)
                push!(all_pieces_latex2, c_tex2)
            elseif c_tex2 == "1"
                push!(all_pieces_latex2, op_latex2)
            elseif c_tex2 == "-1"
                push!(all_pieces_latex2, "-" * op_latex2)
            else
                push!(all_pieces_latex2, c_tex2 * " " * op_latex2)
            end
        end
    end

    latex_str2 = isempty(all_pieces_latex2) ? "0" : _nb_join_terms(all_pieces_latex2)
    return (value_str, latex_str2)
end

# ---------------------------------------------------------------------------
# _nb_sympy_substitute(expr_str, lhs_str, rhs_str, vars_str)
#
# Substitute lhs → rhs in a SymPy operator expression.
#
# SymPy's subs() propagates into function arguments automatically:
#   cd(k)*c(-k)  with k→p  →  cd(p)*c(-p)
#   cd(k)*c(-k)  with k→-k →  cd(-k)*c(k)
#
# Both lhs and rhs are parsed via sympify in the vars context, so
# algebraic expressions like "-k" or "k+q" work as rhs.
#
# Returns: (value_str, latex_str)
# ---------------------------------------------------------------------------
function _nb_sympy_substitute(expr_str::String, lhs_str::String, rhs_str::String,
                               stats_json::String, vars_str::String)
    sp       = SymPy.PyCall.pyimport("sympy")
    builtins = SymPy.PyCall.pyimport("builtins")
    ctx      = _nb_sympy_ctx(vars_str)
    for v in split(vars_str, ",")
        vt = strip(v); isempty(vt) && continue
        ctx[vt] = sp.Symbol(vt, nonzero=true)
    end
    # Also add lhs and rhs symbols to context so sympify can resolve them
    for tok in [lhs_str, rhs_str]
        for m in eachmatch(r"[A-Za-z_]\w*", tok)
            nm = String(m.match)
            haskey(ctx, nm) || (ctx[nm] = sp.Symbol(nm, nonzero=true))
        end
    end

    # Parse lhs using the vars context (nonzero=true assumptions match the expression).
    # Parse rhs using the same context, but also allow new symbols (e.g. "p" in k→p).
    for tok in [lhs_str, rhs_str]
        for m in eachmatch(r"[A-Za-z_]\w*", tok)
            nm = String(m.match)
            haskey(ctx, nm) || (ctx[nm] = sp.Symbol(nm, nonzero=true))
        end
    end
    lhs_sym = try builtins.eval(lhs_str, ctx) catch; sp.sympify(lhs_str) end
    rhs_sym = try builtins.eval(replace(rhs_str, " "=>""), ctx) catch; sp.sympify(rhs_str) end

    # Build SymPy context: operators are abstract Functions so subs() can
    # reach into their arguments.
    op_ctx = copy(ctx)
    for op_name in ("cd", "c", "ad", "a")
        op_ctx[op_name] = sp.Function(op_name)
    end

    # Evaluate the full expression
    result_expr = try
        builtins.eval(replace(expr_str, " " => ""), op_ctx)
    catch
        sp.sympify(expr_str)
    end

    # Apply substitution term-by-term, preserving operator ordering.
    # SymPy Mul is commutative so subs(expr) reorders operators; instead we
    # apply subs only to indices and scalars while keeping the operator sequence.

    function _apply_subs_to_idx(idx)
        isa(idx, Tuple) && return Tuple(_apply_subs_to_idx(i) for i in idx)
        isa(idx, Integer) && return idx   # plain integer: subs has no effect
        try; return idx.subs(lhs_sym, rhs_sym); catch; return idx; end
    end

    pieces_val   = String[]
    pieces_latex = String[]

    function op_to_latex_s(t, idx)
        il = _nb_idx_latex_str(idx, sp)
        t in ("cd","ad") ? "\\operatorname{$t}{\\left($il \\right)}" : "$t{\\left($il \\right)}"
    end

    for sum_term in _nb_split_sum_str(String(strip(expr_str)))
        ops = _nb_parse_op_str(sum_term, ctx)
        no_ops_s = replace(sum_term, r"(?:cd|c|ad|a)\([^)]+\)" => "1")
        # Parse scalar using ctx so symbol assumptions (nonzero=True) match lhs_sym.
        scalar = try builtins.eval(replace(no_ops_s, " " => ""), ctx) catch
            try sp.sympify(no_ops_s) catch; sp.Integer(1) end
        end

        # Substitute lhs→rhs in the scalar coefficient
        new_scalar = try scalar.subs(lhs_sym, rhs_sym) catch; scalar end
        new_scalar = sp.expand(new_scalar)
        new_scalar == sp.Integer(0) && continue

        # Substitute in each operator's index (operator order preserved)
        new_ops = [(_t, _apply_subs_to_idx(_idx)) for (_t, _idx) in ops]

        op_val = join(["$(_t)($(_nb_idx_val_str(_idx, sp)))" for (_t, _idx) in new_ops], "*")
        op_tex = join([op_to_latex_s(_t, _idx)               for (_t, _idx) in new_ops], " ")

        c_str = sp.sstr(new_scalar)
        c_tex = string(SymPy.PyCall.pycall(sp.latex, String, new_scalar))

        if isempty(new_ops)
            push!(pieces_val, c_str)
            push!(pieces_latex, c_tex)
        elseif c_str == "1"
            push!(pieces_val, op_val)
            push!(pieces_latex, op_tex)
        elseif c_str == "-1"
            push!(pieces_val, "-" * op_val)
            push!(pieces_latex, "-" * op_tex)
        else
            push!(pieces_val, c_str * "*" * op_val)
            push!(pieces_latex, c_tex * " " * op_tex)
        end
    end

    isempty(pieces_val) && return ("0", "0")
    return (_nb_join_terms(pieces_val), _nb_join_terms(pieces_latex))
end


# ---------------------------------------------------------------------------
# _nb_sympy_vacuum_expectation(expr_str, stats_json, vars_str)
#
# Compute ⟨0|expr|0⟩ for expressions with algebraic indices.
#
# Algorithm:
#   1. Split expr into additive terms (via _nb_split_sum_str)
#   2. For each term: extract scalar coefficient + operator product
#   3. Normal-order the operator product using _nb_sympy_no_mixed
#   4. Keep only the operator-free pieces from the normal ordering
#      (those with isempty(tops)) — these are the contraction constants
#   5. Multiply each by the term's scalar and accumulate
#
# Physical basis: ⟨0|c†...c†c...c|0⟩ = 0 for any non-empty operator product
# after normal ordering (c|0⟩ = 0 and ⟨0|c† = 0).
# Only the scalar remnants from anticommutation (KroneckerDelta products)
# survive as the vacuum expectation value.
#
# Returns: (value_str, latex_str)
# ---------------------------------------------------------------------------
function _nb_sympy_vacuum_expectation(expr_str::String, stats_json::String, vars_str::String)
    sp = SymPy.PyCall.pyimport("sympy")
    ctx = _nb_sympy_ctx(vars_str)
    for v in split(vars_str, ",")
        vt = strip(v); isempty(vt) && continue
        ctx[vt] = sp.Symbol(vt, nonzero=true)
    end

    stats_map = Dict{String,String}()
    if startswith(strip(stats_json), "{")
        try stats_map = JSON3.read(stats_json, Dict{String,String}) catch end
    else
        stats_map["__default__"] = strip(stats_json)
    end

    scalar_sum = sp.Integer(0)

    for sum_term in _nb_split_sum_str(String(strip(expr_str)))
        ops = _nb_parse_op_str(sum_term, ctx)

        no_ops_str = replace(sum_term, r"(?:cd|c|ad|a)\([^)]+\)" => "1")
        term_scalar = try sp.sympify(no_ops_str) catch; sp.Integer(1) end

        if isempty(ops)
            # Already a scalar — contributes directly
            scalar_sum = sp.expand(scalar_sum + term_scalar)
        else
            # Normal-order the operator product; collect operator-free pieces
            no_terms = _nb_sympy_no_mixed(ops, stats_map, sp)
            for (coeff, tops) in no_terms
                if isempty(tops)
                    contrib = sp.expand(term_scalar * coeff)
                    scalar_sum = sp.expand(scalar_sum + contrib)
                end
                # Terms with operators → vanish in vacuum, skip
            end
        end
    end

    val_str = sp.sstr(scalar_sum)
    lat_str = string(SymPy.PyCall.pycall(sp.latex, String, scalar_sum))
    return (val_str, lat_str)
end


# ---------------------------------------------------------------------------
# _nb_sympy_collect_by_operators(expr_str, stats_json, vars_str)
#
# Group additive terms by their operator sequence and sum scalar coefficients.
# This is the SymPy-path equivalent of QA's collect_by_operators.
#
# Example:
#   "-u**2*xi*cd(k)*c(-k) + v**2*xi*cd(k)*c(-k)"
#   → group key "cd(k)*c(-k)", scalar sum = (-u**2 + v**2)*xi
#   → sp.factor((-u**2 + v**2)*xi) = -(u**2 - v**2)*xi = xi*(v-u)*(v+u)
#   → return "(v - u)*(v + u)*xi*cd(k)*c(-k)"
#
# The operator ORDER is preserved (no normal ordering applied).
# Two terms with the same operators in different order have different keys.
# ---------------------------------------------------------------------------
function _nb_sympy_collect_by_operators(expr_str::String, stats_json::String, vars_str::String)
    sp = SymPy.PyCall.pyimport("sympy")
    ctx = _nb_sympy_ctx(vars_str)
    for v in split(vars_str, ",")
        vt = strip(v); isempty(vt) && continue
        ctx[vt] = sp.Symbol(vt, nonzero=true)
    end

    # Parse terms and group by operator-sequence key
    sum_terms = _nb_split_sum_str(String(strip(expr_str)))

    # groups_scalar: op_key_str → accumulated SymPy scalar
    # groups_ops:   op_key_str → [(op_type, idx), ...] (first occurrence)
    groups_scalar = Dict{String, Any}()
    groups_ops    = Dict{String, Vector}()
    key_order     = String[]  # preserve insertion order

    for sum_term in sum_terms
        ops = _nb_parse_op_str(sum_term, ctx)
        no_ops = replace(sum_term, r"(?:cd|c|ad|a)\([^)]+\)" => "1")
        scalar = try sp.sympify(no_ops) catch; sp.Integer(1) end

        # Key: ordered list of "type(idx_str)" — order matters for non-commutative ops
        op_key = join(["$(_t)($(_nb_idx_val_str(_idx, sp)))" for (_t, _idx) in ops], "*")

        if haskey(groups_scalar, op_key)
            groups_scalar[op_key] = sp.expand(groups_scalar[op_key] + scalar)
        else
            groups_scalar[op_key] = scalar
            groups_ops[op_key]    = ops
            push!(key_order, op_key)
        end
    end

    function _op_to_latex_c(t, idx)
        il = _nb_idx_latex_str(idx, sp)
        t in ("cd","ad") ? "\\operatorname{$t}{\\left($il \\right)}" :
                           "$t{\\left($il \\right)}"
    end

    pieces_val   = String[]
    pieces_latex = String[]

    for op_key in key_order
        coeff_raw = groups_scalar[op_key]
        coeff_raw == sp.Integer(0) && continue

        # Simplify / factor the accumulated scalar coefficient
        coeff_s = try sp.factor(coeff_raw) catch; sp.expand(coeff_raw) end
        coeff_s == sp.Integer(0) && continue

        ops = groups_ops[op_key]
        op_val = join(["$(_t)($(_nb_idx_val_str(_idx, sp)))" for (_t, _idx) in ops], "*")
        op_tex = join([_op_to_latex_c(_t, _idx)                for (_t, _idx) in ops], " ")
        c_str  = sp.sstr(coeff_s)
        c_tex  = string(SymPy.PyCall.pycall(sp.latex, String, coeff_s))

        if isempty(ops)
            push!(pieces_val, c_str)
            push!(pieces_latex, c_tex)
        elseif c_str == "1"
            push!(pieces_val, op_val);  push!(pieces_latex, op_tex)
        elseif c_str == "-1"
            push!(pieces_val, "-"*op_val);  push!(pieces_latex, "-"*op_tex)
        else
            push!(pieces_val, c_str * "*" * op_val)
            # Wrap coefficient in braces to prevent Display Bridge from
            # treating "coeff \left(op\right)" as "coeff_{op}" subscript.
            # This is necessary because scalar parameters like xi are
            # registered in SymbolContext and the Display Bridge will
            # re-interpret "xi \left(...\right)" as an indexed symbol.
            if contains(c_tex, "\\left(") || contains(c_tex, "\\left [")
                push!(pieces_latex, "\\left(" * c_tex * "\\right)" * " " * op_tex)
            else
                push!(pieces_latex, c_tex * " " * op_tex)
            end
        end
    end

    isempty(pieces_val) && return ("0", "0")
    return (_nb_join_terms(pieces_val), _nb_join_terms(pieces_latex))
end


# ---------------------------------------------------------------------------
# _nb_sympy_general_comm(a_str, b_str, stats_json, vars_str)
#
# Compute [A, B] for arbitrary operator expressions A and B.
#
# Uses the derivation property:
#   [A, B₁B₂...Bₙ] = Σᵢ B₁...Bᵢ₋₁ [A, Bᵢ] Bᵢ₊₁...Bₙ
#
# where [A, single_op] is computed using the existing bubble-sort
# anticommutation logic via _nb_sympy_no_mixed.
#
# Both A and B may be sums of operator products (via _nb_split_sum_str).
# ---------------------------------------------------------------------------
function _nb_sympy_general_comm(a_str::String, b_str::String,
                                stats_json::String, vars_str::String)
    sp = SymPy.PyCall.pyimport("sympy")
    ctx = _nb_sympy_ctx(vars_str)
    for v in split(vars_str, ",")
        vt = strip(v); isempty(vt) && continue
        ctx[vt] = sp.Symbol(vt, nonzero=true)
    end

    stats_map = Dict{String,String}()
    if startswith(strip(stats_json), "{")
        try stats_map = JSON3.read(stats_json, Dict{String,String}) catch end
    else
        stats_map["__default__"] = strip(stats_json)
    end

    # Split A and B into additive terms
    a_terms = _nb_split_sum_str(String(strip(a_str)))
    b_terms = _nb_split_sum_str(String(strip(b_str)))

    all_val   = String[]
    all_latex = String[]

    function _op_to_latex_g(t, idx_sym)
        il = _nb_idx_latex_str(idx_sym, sp)
        t in ("cd","ad") ? "\\operatorname{$t}{\\left($il \\right)}" : "$t{\\left($il \\right)}"
    end

    for a_term in a_terms, b_term in b_terms
        # Parse each term into (scalar_coeff, ops_list)
        a_ops  = _nb_parse_op_str(a_term, ctx)
        b_ops  = _nb_parse_op_str(b_term, ctx)
        a_coeff = try sp.sympify(replace(a_term, r"(?:cd|c|ad|a)\([^)]+\)" => "1")) catch; sp.Integer(1) end
        b_coeff = try sp.sympify(replace(b_term, r"(?:cd|c|ad|a)\([^)]+\)" => "1")) catch; sp.Integer(1) end
        ab_coeff = sp.expand(a_coeff * b_coeff)
        ab_coeff == sp.Integer(0) && continue

        # [a_ops, b_ops] via derivation: [A, B₁B₂...Bₙ] = Σᵢ B₁...Bᵢ₋₁·[A,Bᵢ]·Bᵢ₊₁...Bₙ
        # For each position i in b_ops, compute commutator_terms = normal_order([a_ops, b_ops[i]])
        # then sandwich with b_ops[1..i-1] on left and b_ops[i+1..n] on right.

        for i in 1:length(b_ops)
            b_i_type, b_i_idx = b_ops[i]
            left_ops  = b_ops[1:i-1]   # B₁...Bᵢ₋₁
            right_ops = b_ops[i+1:end]  # Bᵢ₊₁...Bₙ

            # Compute [a_ops, b_ops[i]] using anticommutation:
            # [A, Bᵢ] = A*Bᵢ - Bᵢ*A  →  normal_order(A*Bᵢ) - normal_order(Bᵢ*A)
            # = _nb_sympy_no_mixed([a_ops..., (b_i,b_i_idx)], stats_map)
            #  - _nb_sympy_no_mixed([(b_i,b_i_idx), a_ops...], stats_map)

            ab_product = vcat(a_ops, [(b_i_type, b_i_idx)])
            ba_product = vcat([(b_i_type, b_i_idx)], a_ops)

            ab_normal = _nb_sympy_no_mixed(ab_product, stats_map, sp)
            ba_normal = _nb_sympy_no_mixed(ba_product, stats_map, sp)

            # Combine: ab_normal - ba_normal, prepend left_ops, append right_ops
            for (sign, norm_terms) in [(sp.Integer(1), ab_normal), (sp.Integer(-1), ba_normal)]
                for (c_inner, inner_ops) in norm_terms
                    total_coeff = sp.expand(ab_coeff * sign * c_inner)
                    total_coeff == sp.Integer(0) && continue

                    full_ops = vcat(left_ops, inner_ops, right_ops)
                    op_val = join(["$(t)($(_nb_idx_val_str(idx, sp)))" for (t,idx) in full_ops], "*")
                    c_str  = sp.sstr(total_coeff)
                    c_tex  = string(SymPy.PyCall.pycall(sp.latex, String, total_coeff))
                    op_tex = join([_op_to_latex_g(t, idx) for (t,idx) in full_ops], " ")

                    if isempty(full_ops)
                        push!(all_val,   c_str)
                        push!(all_latex, c_tex)
                    elseif c_str == "1"
                        push!(all_val,   op_val)
                        push!(all_latex, op_tex)
                    elseif c_str == "-1"
                        push!(all_val,   "-" * op_val)
                        push!(all_latex, "-" * op_tex)
                    else
                        push!(all_val,   c_str * "*" * op_val)
                        push!(all_latex, c_tex * " " * op_tex)
                    end
                end
            end
        end
    end

    isempty(all_val) && return ("0", "0")
    return (_nb_join_terms(all_val), _nb_join_terms(all_latex))
end


# Join term strings into a sum, handling leading "-" signs correctly.
function _nb_join_terms(pieces::Vector{String})
    isempty(pieces) && return "0"
    result = pieces[1]
    for p in pieces[2:end]
        if startswith(p, "-")
            result *= " - " * p[2:end]
        else
            result *= " + " * p
        end
    end
    return result
end


# ---------------------------------------------------------------------------
# _nb_sympy_wick(expr_str, rules_json, include_constant, stats, vars_str)
#
# Wick contraction for expressions with algebraic indices (-k, k+q, …).
# Uses the same cd/c operator notation as _nb_sympy_fermion_comm.
#
# Arguments:
#   expr_str         : SymPy-evaluatable operator product, e.g. "cd(-k)*c(k)"
#   rules_json       : JSON array of {op1_name, op1_create, op2_name, op2_create, value}
#                      value: plain scalar, or "f(INDEX1,INDEX2)" template
#   include_constant : if true, also return double-contracted scalar terms
#   stats            : "fermion" or "boson"
#   vars_str         : comma-separated SymPy variable names, e.g. "k,q,p"
#
# Returns: JSON string — array of [value_str, latex_str] pairs, one per contraction.
#
# The Python caller (engine.wick_contract) parses the JSON and registers each
# [value_str, latex_str] as a separate Expression with sympy_canonical set.
# ---------------------------------------------------------------------------
_nb_wick_no_match = false

function _nb_sympy_wick(expr_str::String, rules_json::String,
                         include_constant::Bool, stats_json::String, vars_str::String)
    sp       = SymPy.PyCall.pyimport("sympy")
    builtins = SymPy.PyCall.pyimport("builtins")
    ctx      = _nb_sympy_ctx(vars_str)

    # Declare momentum variables nonzero so δ(-k,k)=0
    for v in split(vars_str, ",")
        vt = strip(v)
        isempty(vt) && continue
        ctx[vt] = sp.Symbol(vt, nonzero=true)
    end

    # Parse per-operator statistics (backward compat: plain "fermion"/"boson")
    stats_map_w = Dict{String,String}()
    if startswith(strip(stats_json), "{")
        try stats_map_w = JSON3.read(stats_json, Dict{String,String}) catch end
    else
        stats_map_w["__default__"] = strip(stats_json)
    end
    _wstats(base::String) = get(stats_map_w, base, get(stats_map_w, "__default__", "fermion"))

    rules = JSON3.read(rules_json, Vector{Dict{String,Any}})
    # Use multi-index parser to support (k,σ) spin+momentum operators.
    # Each element: (op_type, [idx0, idx1, ...])
    ops = _nb_parse_op_str_multi(expr_str, ctx)

    all_terms = Tuple{Any, Vector{Tuple{String, Vector{Any}}}}[
        (sp.Integer(1), [(t, idxv) for (t, idxv) in ops])
    ]

    result_terms = Tuple{Any, Vector{Tuple{String, Vector{Any}}}}[]

    for (coeff, term_ops) in all_terms
        n_ops = length(term_ops)
        n_ops < 2 && continue

        for rule in rules
            op1_name   = String(rule["op1_name"])
            op1_create = Bool(rule["op1_create"])
            op2_name   = String(rule["op2_name"])
            op2_create = Bool(rule["op2_create"])
            value_tmpl = String(rule["value"])

            for i in 1:n_ops, j in (i+1):n_ops
                t_i, idxv_i = term_ops[i]
                t_j, idxv_j = term_ops[j]

                name_i   = endswith(t_i, "d") ? t_i[1:end-1] : t_i
                name_j   = endswith(t_j, "d") ? t_j[1:end-1] : t_j
                create_i = t_i in ("cd", "ad")
                create_j = t_j in ("cd", "ad")

                name_i   == op1_name   || continue
                name_j   == op2_name   || continue
                create_i == op1_create || continue
                create_j == op2_create || continue

                # Sign: fermion operators anticommute; boson or mixed → +1
                _base_i_w = endswith(t_i, "d") ? t_i[1:end-1] : t_i
                _base_j_w = endswith(t_j, "d") ? t_j[1:end-1] : t_j
                _both_fermion = (_wstats(_base_i_w) == "fermion" && _wstats(_base_j_w) == "fermion")
                sign_val = (_both_fermion && isodd(j - i - 1)) ? sp.Integer(-1) : sp.Integer(1)

                # Pass index VECTORS to value builder
                cval = _nb_sympy_wc_value(value_tmpl, idxv_i, idxv_j, sp, ctx, builtins)

                rem_ops = Tuple{String,Vector{Any}}[(term_ops[k]) for k in 1:n_ops if k != i && k != j]

                if isempty(rem_ops)
                    include_constant || continue
                    push!(result_terms, (sp.expand(coeff * sign_val * cval), Tuple{String,Vector{Any}}[]))
                else
                    push!(result_terms, (sp.expand(coeff * sign_val * cval), rem_ops))
                end
            end
        end
    end

    # "no rule reached any pair" and "the contractions evaluated to zero" both
    # end with an empty answer, and the researcher cannot tell them apart —
    # measured 2026-08-15, a rule naming operators the expression does not
    # contain returned the same empty list as the physically correct
    # δ(-k,k)=0 vanishing (open_items NOOP-REWRITE-FAMILY).  The difference is
    # visible right here and nowhere else: `result_terms` collects one entry per
    # pair a rule MATCHED, before any coefficient is evaluated.
    global _nb_wick_no_match
    _nb_wick_no_match = isempty(result_terms)

    out = Any[]
    for (coeff, rem_ops) in result_terms
        coeff_s = sp.expand(coeff)
        coeff_s == sp.Integer(0) && continue
        val_str = _nb_format_sympy_term_val_multi(coeff_s, rem_ops, sp)
        lat_str = _nb_format_sympy_term_lat_multi(coeff_s, rem_ops, sp)
        push!(out, [val_str, lat_str])
    end

    return JSON3.write(out)
end


# Build SymPy contraction value from rule template.
#
# Supported formats:
#   "V_HF"                       → plain SymPy scalar (no INDEX)
#   "n(INDEX1)"                  → sp.Function("n")(idx1)
#   "n(INDEX1,INDEX2)"           → sp.Function("n")(idx1, idx2)
#   "delta(INDEX1,INDEX2)"       → sp.KroneckerDelta(idx1, idx2)
#   "KroneckerDelta(INDEX1,INDEX2)" → sp.KroneckerDelta(idx1, idx2)
#   "delta(INDEX1,INDEX2)*n(INDEX1)"  → product of two terms
#   "delta(INDEX1,INDEX2)*V_HF"       → KroneckerDelta × scalar
#
# KEY: with momentum variables declared nonzero (e.g. k ≠ 0),
#   delta(-k, k) = KroneckerDelta(-k, k) = 0 ← δ(-k,k)=0 correctly evaluated
#
# _nb_sympy_wc_value: idx1 and idx2 are now Vector{Any} (multi-index support).
# idx1[1] = first index of op1 (momentum), idx1[2] = second index (spin), etc.
function _nb_sympy_wc_value(value_tmpl::String, idx1::Vector{Any}, idx2::Vector{Any}, sp, ctx, builtins)
    parts = _nb_split_product_template(String(strip(value_tmpl)))
    if length(parts) > 1
        result = sp.Integer(1)
        for part in parts
            result = result * _nb_sympy_wc_single_factor(String(strip(part)), idx1, idx2, sp, ctx, builtins)
        end
        return sp.expand(result)
    end
    return _nb_sympy_wc_single_factor(String(strip(value_tmpl)), idx1, idx2, sp, ctx, builtins)
end

# Split an expression string at top-level `+` and `-` (additive decomposition).
# Inner `-` signs inside parentheses (e.g. c(-k)) are NOT split.
# e.g. "Delta*cd(-k) - xi*c(k)" → ["Delta*cd(-k)", "- xi*c(k)"]
function _nb_split_sum_str(s::String)
    terms = String[]
    depth = 0
    buf   = IOBuffer()
    for (i, ch) in enumerate(s)
        if ch in ('(', '[', '{')
            depth += 1
            write(buf, ch)
        elseif ch in (')', ']', '}')
            depth -= 1
            write(buf, ch)
        elseif ch in ('+', '-') && depth == 0 && i > 1
            t = strip(String(take!(buf)))
            # Skip bare sign-only tokens (e.g. "+" from "A + -B" splitting)
            !isempty(t) && t != "+" && t != "-" && push!(terms, t)
            write(buf, ch)  # start new term with sign
        else
            write(buf, ch)
        end
    end
    last = strip(String(take!(buf)))
    !isempty(last) && push!(terms, last)
    isempty(terms) && return [s]
    return terms
end

# Split "A*B*C(X,Y)" at top-level `*` only.
function _nb_split_product_template(s::String)
    parts = String[]
    depth = 0
    buf   = IOBuffer()
    for ch in s
        if ch == '('
            depth += 1
            write(buf, ch)
        elseif ch == ')'
            depth -= 1
            write(buf, ch)
        elseif ch == '*' && depth == 0
            push!(parts, String(take!(buf)))
        else
            write(buf, ch)
        end
    end
    last_part = String(take!(buf))
    !isempty(last_part) && push!(parts, last_part)
    return parts
end

# Build one factor of the contraction value template.
# idx1, idx2 are now Vector{Any}.
# INDEX1   / INDEX1_0  → idx1[1]   (first index, 0-based in template = 1-based in Julia)
# INDEX1_N (N≥1)       → idx1[N+1]
# INDEX2_* similarly.
function _nb_sympy_wc_single_factor(tmpl::String, idx1::Vector{Any}, idx2::Vector{Any}, sp, ctx, builtins)
    if !occursin("INDEX1", tmpl) && !occursin("INDEX2", tmpl)
        return try
            extra = copy(ctx)
            for m in eachmatch(r"[A-Za-z_]\w*", tmpl)
                nm = String(m.match)
                haskey(extra, nm) || (extra[nm] = sp.Symbol(nm))
            end
            builtins.eval(tmpl, extra)
        catch
            sp.Symbol(tmpl)
        end
    end

    m = match(r"^(\w+)\((.*)\)$", tmpl)
    m === nothing && return sp.Symbol(tmpl)

    fn_name  = String(m[1])
    arg_toks = [strip(s) for s in split(String(m[2]), ",")]

    function _nth_idx(idxv::Vector{Any}, n::Int)  # n is 0-based
        j = n + 1  # 1-based
        j <= length(idxv) ? idxv[j] : sp.Symbol("?")
    end

    args = Any[]
    for tok in arg_toks
        if tok == "INDEX1" || tok == "INDEX1_0"
            push!(args, _nth_idx(idx1, 0))
        elseif tok == "INDEX2" || tok == "INDEX2_0"
            push!(args, _nth_idx(idx2, 0))
        elseif startswith(tok, "INDEX1_")
            n = parse(Int, tok[8:end])
            push!(args, _nth_idx(idx1, n))
        elseif startswith(tok, "INDEX2_")
            n = parse(Int, tok[8:end])
            push!(args, _nth_idx(idx2, n))
        else
            push!(args, try builtins.eval(tok, ctx) catch; sp.Symbol(tok) end)
        end
    end

    if fn_name in ("delta", "KroneckerDelta")
        return SymPy.PyCall.pycall(sp.KroneckerDelta, SymPy.PyCall.PyObject, args...)
    end

    return SymPy.PyCall.pycall(sp.Function(fn_name), SymPy.PyCall.PyObject, args...)
end


# Format one Wick result term (multi-index) as value string.
# rem_ops: [(op_type, [idx0, idx1, ...])]
function _nb_format_sympy_term_val_multi(coeff, rem_ops, sp)
    if isempty(rem_ops)
        return string(SymPy.PyCall.pystr(sp.expand(coeff)))
    end
    op_str = join([
        let idxv = idxv_raw
            idx_strs = [string(SymPy.PyCall.pystr(_nb_to_sympy_idx(idx, sp))) for idx in idxv]
            "$t($(join(idx_strs, ",")))"
        end
        for (t, idxv_raw) in rem_ops
    ], "*")
    c_str = SymPy.PyCall.pystr(coeff)
    c_str == "1" && return op_str
    c_str == "-1" && return "-" * op_str
    return string(c_str) * "*" * op_str
end

# Format one Wick result term (multi-index) as LaTeX string.
function _nb_format_sympy_term_lat_multi(coeff, rem_ops, sp)
    c_tex = string(SymPy.PyCall.pycall(sp.latex, String, coeff))
    if isempty(rem_ops)
        return c_tex
    end
    op_parts = String[]
    for (t, idxv_raw) in rem_ops
        idx_texs = [string(SymPy.PyCall.pycall(sp.latex, String, _nb_to_sympy_idx(idx, sp))) for idx in idxv_raw]
        idx_tex = join(idx_texs, ", ")
        if t in ("cd", "ad")
            push!(op_parts, "\\operatorname{$t}{\\left($idx_tex \\right)}")
        else
            push!(op_parts, "$t{\\left($idx_tex \\right)}")
        end
    end
    op_tex = join(op_parts, " ")
    c_tex == "1"  && return op_tex
    c_tex == "-1" && return "-" * op_tex
    # No parentheses: coefficient directly before operator (QA style)
    return c_tex * " " * op_tex
end


# Coerce index to a SymPy object for string/latex conversion.
# Julia integers and floats come through PyCall as native types;
# sp.sympify() converts them to SymPy Integers/Floats.
function _nb_to_sympy_idx(idx, sp)
    try
        # If already a Python/SymPy object, use as-is
        SymPy.PyCall.pystr(idx)  # probe
        return idx
    catch
        return sp.sympify(idx)
    end
end

# Format one Wick result term as a value string (SymPy sstr format).
function _nb_format_sympy_term_val(coeff, rem_ops, sp)
    isempty(rem_ops) && return string(SymPy.PyCall.pystr(sp.expand(coeff)))
    op_str = join(["$(t)($(SymPy.PyCall.pystr(_nb_to_sympy_idx(idx, sp))))" for (t, idx) in rem_ops], "*")
    c_str  = SymPy.PyCall.pystr(coeff)
    c_str == "1" && return op_str
    c_str == "-1" && return "-" * op_str
    return string(c_str) * "*" * op_str
end

# Format one Wick result term as a LaTeX string (SymPy operatorname format).
function _nb_format_sympy_term_lat(coeff, rem_ops, sp)
    c_tex = string(SymPy.PyCall.pycall(sp.latex, String, coeff))
    if isempty(rem_ops)
        return c_tex
    end
    op_parts = String[]
    for (t, idx) in rem_ops
        idx_tex = string(SymPy.PyCall.pycall(sp.latex, String, _nb_to_sympy_idx(idx, sp)))
        if t in ("cd", "ad")
            push!(op_parts, "\\operatorname{$t}{\\left($idx_tex \\right)}")
        else
            push!(op_parts, "$t{\\left($idx_tex \\right)}")
        end
    end
    op_tex = join(op_parts, " ")
    c_tex == "1"  && return op_tex
    c_tex == "-1" && return "-" * op_tex
    return c_tex * " " * op_tex
end


# Compute anticommutation delta for single or multi-index operators.
# Single index: KroneckerDelta(idx1, idx2)
# Multi-index (Tuple): product KroneckerDelta(k1,k2)*KroneckerDelta(s1,s2)*...
# e.g. {c(k,σ), c†(q,τ)} = δ(k,q)δ(σ,τ)
function _nb_multi_delta(idx1, idx2, sp)
    if isa(idx1, Tuple)
        result = sp.Integer(1)
        for (i1, i2) in zip(idx1, idx2)
            result = result * sp.KroneckerDelta(i1, i2)
        end
        return result
    end
    return sp.KroneckerDelta(idx1, idx2)
end

# Format an operator index for value string output.
# Single index: sp.sstr(idx) → "k" or "-k"
# Multi-index Tuple: join → "k,s" (omit outer parens)
# Plain Julia integers (e.g. from builtins.eval("1", ctx) → Int64): stringify directly.
function _nb_idx_val_str(idx, sp)
    if isa(idx, Tuple)
        return join([_nb_idx_val_str(i, sp) for i in idx], ",")
    end
    isa(idx, Integer) && return string(idx)
    try; return string(SymPy.PyCall.pystr(idx)); catch; return string(idx); end
end

# Format an operator index for LaTeX output.
function _nb_idx_latex_str(idx, sp)
    if isa(idx, Tuple)
        return join([_nb_idx_latex_str(i, sp) for i in idx], ", ")
    end
    isa(idx, Integer) && return string(idx)
    try
        return string(SymPy.PyCall.pycall(sp.latex, String, _nb_to_sympy_idx(idx, sp)))
    catch
        return string(idx)
    end
end


# Normal-order a fermionic operator product using bubble sort + anticommutation:
#   {c(i), cd(j)} = δ(i,j)  →  c(i)*cd(j) = δ(i,j) - cd(j)*c(i)
# Each term is a (coeff, ops_vec) pair; ops_vec = [(type, idx), ...]
# ---------------------------------------------------------------------------
# _nb_sympy_no_mixed — general normal ordering with per-operator statistics.
#
# stats_map : Dict{"c"=>"fermion", "a"=>"boson", ...}
#   Determines the (anti-)commutation relation for each swap:
#
#   fermion × fermion (same name)  : {c(i),c†(j)} = δ(i,j)  →  δ term + swap with sign -1
#   fermion × fermion (diff name)  : {c,d†} = 0              →  swap with sign -1, no δ
#   boson   × boson   (same name)  : [a(i),a†(j)] = δ(i,j)  →  δ term + swap with sign +1
#   boson   × boson   (diff name)  : [a,b†] = 0              →  swap with sign +1, no δ
#   fermion × boson   (any)        : [c,a†] = 0              →  swap with sign +1, no δ
#   boson   × fermion (any)        : [a,c†] = 0              →  swap with sign +1, no δ
#
# Multi-index operators (e.g. c(k,σ)) use _nb_multi_delta for the delta product.
# ---------------------------------------------------------------------------
function _nb_sympy_no_mixed(initial_ops::Vector, stats_map::Dict{String,String}, sp)
    terms = Tuple{Any, Vector{Tuple{String,Any}}}[
        (sp.Integer(1), [(t, idx) for (t, idx) in initial_ops])
    ]

    max_iter = length(initial_ops)^2 + length(initial_ops) + 1
    for _ in 1:max_iter
        changed = false
        next_terms = Tuple{Any, Vector{Tuple{String,Any}}}[]

        for (coeff, tops) in terms
            # Find first adjacent (annihilation, creation) pair that needs swapping
            swap_i = 0
            for i in 1:(length(tops)-1)
                t_i    = tops[i][1]
                t_next = tops[i+1][1]
                is_ann  = !(t_i   in ("cd","ad"))   # "c","a",etc. = annihilation
                is_cre  = t_next  in ("cd","ad")    # "cd","ad"    = creation
                if is_ann && is_cre
                    swap_i = i; break
                end
            end

            if swap_i == 0
                push!(next_terms, (coeff, tops))   # already normal ordered
            else
                changed = true
                t_i, idx_i = tops[swap_i]
                t_j, idx_j = tops[swap_i+1]

                # Base names (strip creation suffix "d")
                base_i = endswith(t_i, "d") ? t_i[1:end-1] : t_i   # "c" or "a"
                base_j = endswith(t_j, "d") ? t_j[1:end-1] : t_j   # "c" or "a"

                # Per-operator statistics; default to "fermion"
                st_i = get(stats_map, base_i, "fermion")
                st_j = get(stats_map, base_j, "fermion")

                # Sign for the swap term
                same_stats  = (st_i == st_j)
                all_fermion = (st_i == "fermion" && st_j == "fermion")
                swap_sign   = (all_fermion) ? sp.Integer(-1) : sp.Integer(1)

                # Delta term: only when same base name (same operator species)
                if base_i == base_j
                    delta_val = _nb_multi_delta(idx_i, idx_j, sp)
                    rem = Tuple{String,Any}[tops[k] for k in 1:length(tops) if k != swap_i && k != swap_i+1]
                    push!(next_terms, (sp.expand(coeff * delta_val), rem))
                end

                # Swapped term
                swapped = copy(tops)
                swapped[swap_i]   = (t_j, idx_j)
                swapped[swap_i+1] = (t_i, idx_i)
                push!(next_terms, (swap_sign * coeff, swapped))
            end
        end

        terms = next_terms
        !changed && break
    end
    return terms
end


function _nb_sympy_no_fermion(initial_ops::Vector, sp)
    delta = sp.KroneckerDelta
    terms = Tuple{Any, Vector{Tuple{String,Any}}}[
        (sp.Integer(1), [(t, idx) for (t, idx) in initial_ops])
    ]

    max_iter = (length(initial_ops))^2 + length(initial_ops) + 1
    for _ in 1:max_iter
        changed = false
        next_terms = Tuple{Any, Vector{Tuple{String,Any}}}[]
        for (coeff, tops) in terms
            # Find first c(i)*cd(j) pair (annihilation before creation)
            swap_i = 0
            for i in 1:(length(tops) - 1)
                if tops[i][1] == "c" && tops[i+1][1] == "cd"
                    swap_i = i; break
                end
            end
            if swap_i == 0
                push!(next_terms, (coeff, tops))  # already normal ordered
            else
                changed = true
                t1, idx1 = tops[swap_i]      # c(idx1)
                t2, idx2 = tops[swap_i + 1]  # cd(idx2)

                # Delta term: coeff * δ(idx1,idx2) * remaining product
                remaining = vcat(tops[1:swap_i-1], tops[swap_i+2:end])
                d_coeff   = sp.expand(coeff * delta(idx1, idx2))
                push!(next_terms, (d_coeff, remaining))

                # Swapped term: -coeff * [..., cd(idx2), c(idx1), ...]
                swapped = copy(tops)
                swapped[swap_i]     = ("cd", idx2)
                swapped[swap_i + 1] = ("c",  idx1)
                push!(next_terms, (-coeff, swapped))
            end
        end
        terms = next_terms
        !changed && break
    end

    return terms
end

# Normal-order a bosonic operator product: [a(i), ad(j)] = δ(i,j)
#   a(i)*ad(j) = δ(i,j) + ad(j)*a(i)   (positive sign, unlike fermions)
function _nb_sympy_no_boson(initial_ops::Vector, sp)
    delta = sp.KroneckerDelta
    terms = Tuple{Any, Vector{Tuple{String,Any}}}[
        (sp.Integer(1), [(t, idx) for (t, idx) in initial_ops])
    ]

    max_iter = (length(initial_ops))^2 + length(initial_ops) + 1
    for _ in 1:max_iter
        changed = false
        next_terms = Tuple{Any, Vector{Tuple{String,Any}}}[]
        for (coeff, tops) in terms
            swap_i = 0
            for i in 1:(length(tops) - 1)
                if tops[i][1] == "a" && tops[i+1][1] == "ad"
                    swap_i = i; break
                end
            end
            if swap_i == 0
                push!(next_terms, (coeff, tops))
            else
                changed = true
                t1, idx1 = tops[swap_i]
                t2, idx2 = tops[swap_i + 1]

                # Delta term: +coeff * δ(idx1,idx2) * remaining
                remaining = vcat(tops[1:swap_i-1], tops[swap_i+2:end])
                d_coeff   = sp.expand(coeff * delta(idx1, idx2))
                push!(next_terms, (d_coeff, remaining))

                # Swapped term: +coeff * [..., ad(idx2), a(idx1), ...]  (same sign)
                swapped = copy(tops)
                swapped[swap_i]     = ("ad", idx2)
                swapped[swap_i + 1] = ("a",  idx1)
                push!(next_terms, (coeff, swapped))
            end
        end
        terms = next_terms
        !changed && break
    end

    return terms
end


# ------------------------------------------------------------------
# Inner helpers: fermion/boson commutator kernel (no JSON overhead)
# ------------------------------------------------------------------

function _nb_sympy_fermion_comm_inner(
        prod_str::String, target_str::String,
        sp, ctx::Dict)
    delta   = sp.KroneckerDelta
    factors = _nb_parse_op_str(prod_str, ctx)
    isempty(factors) && return sp.Integer(0)
    tgt_ops = _nb_parse_op_str(target_str, ctx)
    length(tgt_ops) != 1 && error("target must be a single operator; got: $target_str")
    tgt_type, tgt_idx = tgt_ops[1]
    n      = length(factors)
    result = sp.Integer(0)
    for i in 1:n
        op_type, op_idx = factors[i]
        if (op_type == "cd" && tgt_type == "c") || (op_type == "c" && tgt_type == "cd")
            anticomm = delta(op_idx, tgt_idx)
        else
            continue
        end
        sign = iseven(n - i) ? sp.Integer(1) : sp.Integer(-1)
        rem_expr = sp.Integer(1)
        for j in 1:n
            j == i && continue
            rem_type, rem_idx = factors[j]
            rem_expr = rem_expr * sp.Function(rem_type)(rem_idx)
        end
        result = result + sign * anticomm * rem_expr
    end
    return result
end


function _nb_sympy_boson_comm_inner(
        prod_str::String, target_str::String,
        sp, ctx::Dict)
    delta   = sp.KroneckerDelta
    factors = _nb_parse_op_str(prod_str, ctx)
    isempty(factors) && return sp.Integer(0)
    tgt_ops = _nb_parse_op_str(target_str, ctx)
    length(tgt_ops) != 1 && error("target must be a single operator; got: $target_str")
    tgt_type, tgt_idx = tgt_ops[1]
    n      = length(factors)
    result = sp.Integer(0)
    for i in 1:n
        op_type, op_idx = factors[i]
        if op_type == "a" && tgt_type == "ad"
            comm_val = delta(op_idx, tgt_idx)
        elseif op_type == "ad" && tgt_type == "a"
            comm_val = -delta(op_idx, tgt_idx)
        else
            continue
        end
        left_expr  = sp.Integer(1)
        right_expr = sp.Integer(1)
        for j in 1:n
            j == i && continue
            rem_type, rem_idx = factors[j]
            if j < i
                left_expr  = left_expr  * sp.Function(rem_type)(rem_idx)
            else
                right_expr = right_expr * sp.Function(rem_type)(rem_idx)
            end
        end
        result = result + left_expr * comm_val * right_expr
    end
    return result
end


"""
    _nb_pair_substitute(expr, pairs_json)

Mean-field / decoupling approximation: replace operator-pair products
with c-number (scalar) values.

Arguments
---------
- `expr`       : QuExpr to transform
- `pairs_json` : JSON array of substitution rules, each element is a
                 two-element array ["bares_pattern", "scalar_expr"].
                 `bares_pattern` is matched against `string(term.bares)`.
                 `scalar_expr`  is evaluated as a Julia expression and used
                 as the new coefficient contribution.

Example
-------
    _nb_pair_substitute(_nb_E1,
        "[[\"c†(k) c(k)\",\"n_k\"],[\"c†(K) c(K)\",\"n_k\"]]")

For each term in `expr`:
- If `string(term.bares)` matches a pattern, replace the *entire* QuTerm
  with a scalar contribution: `coefficient * scalar_value`.
  The bare-operator content is stripped (becoming the identity, i.e. a
  pure c-number in the QuExpr representation).
- Otherwise the term is kept unchanged.

The result is returned as a `normal_form`-ed QuExpr.
"""
# Order-independent canonical key for a bares string.
# `string(term.bares)` is space-separated base operators (e.g. "c†(k) c(k)").
# The order in which operators appear depends on how the term was built
# (the engine's `expand` order vs a fresh `normal_form` can differ), so we
# sort the base-operator tokens to obtain a representation that matches
# regardless of factor order.  Identity terms map to "".
_nb_canon_bares(bares_str::AbstractString) =
    join(sort(filter(!isempty, split(bares_str))), " ")

# Derive the canonical match key from a pattern, which may be either:
#   - a Julia operator expression produced by the Python code generator
#     (e.g. `c'(:k)*c(:k)` or `c'(_QA.QuIndex("K"))*c(_QA.QuIndex("K"))`), or
#   - a legacy raw bares string (e.g. "c†(k) c(k)").
# For the former we evaluate it to a QuExpr and read the bares of its operator
# term (the one with the most base operators, skipping any identity term that
# normal_form may introduce).  For the latter `Meta.parse`/`eval` fails and we
# canonicalize the raw string directly — preserving backward compatibility with
# callers that pass raw bares strings.
function _nb_pattern_key(pat::String)
    try
        nf = normal_form(eval(Meta.parse(pat)))
        best = ""
        best_len = -1
        for (op, _) in nf.terms
            n = length(op.bares.v)
            if n > best_len
                best_len = n
                best = string(op.bares)
            end
        end
        return best_len >= 0 ? _nb_canon_bares(best) : _nb_canon_bares(pat)
    catch
        return _nb_canon_bares(pat)
    end
end

_nb_pair_no_match = false

function _nb_pair_substitute(expr::_QA.QuExpr, pairs_json::String)
    rules = JSON3.read(pairs_json, Vector{Vector{String}})
    rule_map = Dict{String,Any}(_nb_pattern_key(r[1]) => Meta.parse(r[2]) for r in rules)
    _nb_nt = Dict{_QA.QuTerm, Number}()
    # Count the matches so "no rule applied to any term" can be told from "the
    # substitution happened".  Measured 2026-08-15: a pattern naming operators
    # the expression does not contain returned the expression UNCHANGED, as a
    # success — indistinguishable from a rule that did apply
    # (open_items NOOP-REWRITE-FAMILY).  The count lives here because this is
    # where the matching happens; deciding it again on the Python side would be
    # a second answer to one question.
    _nb_hits = 0
    for (op, coeff) in expr.terms
        bares_str = _nb_canon_bares(string(op.bares))
        if haskey(rule_map, bares_str)
            _nb_hits += 1
            scalar_val = eval(rule_map[bares_str])
            # Strip operator content: keep nsuminds, δs, params, expvals, corrs;
            # set bares to identity (empty BaseOpProduct).
            bare_id = _QA.QuTerm(op.nsuminds, op.δs, op.params,
                                  op.expvals, op.corrs,
                                  _QA.BaseOpProduct())
            _nb_nt[bare_id] = get(_nb_nt, bare_id, 0) + coeff * scalar_val
        else
            _nb_nt[op] = get(_nb_nt, op, 0) + coeff
        end
    end
    # No rule reached any term.  Reported as a WARNING, not a refusal: this
    # operation takes SEVERAL rules at once, and "this Hamiltonian has no such
    # pair" is information a researcher may legitimately act on — unlike
    # `operator_substitute`, which names ONE operator and can only mean it.
    # Decided 2026-08-15; the expression comes back unchanged either way, which
    # is what `tests/test_pair_substitute.py::test_no_matching_pattern_unchanged`
    # has asserted since the operation was written.
    global _nb_pair_no_match
    _nb_pair_no_match = (_nb_hits == 0)
    return normal_form(_QA.QuExpr(_nb_nt))
end

# ---------------------------------------------------------------------------
# _nb_collect_spin_pm(expr, sp_x, sp_y, sm_x, sm_y, sp_label, sm_label)
#
# Re-express a QuExpr (in Sx/Sy/Sz basis) using user-defined ladder operators
# Sp and Sm, where:
#   Sp = sp_x * Sx + sp_y * Sy  (e.g. sp_x=1, sp_y=im  →  Sp = Sx + i·Sy)
#   Sm = sm_x * Sx + sm_y * Sy  (e.g. sm_x=1, sm_y=-im →  Sm = Sx - i·Sy)
#
# Algorithm:
#   1. Group terms by (non-spin operator key, spin site).
#   2. For each group, sum Sx/Sy/Sz coefficients.
#   3. Apply the inverse linear transform to find c_Sp, c_Sm, c_Sz.
#   4. Return a display string using sp_label/sm_label notation.
#
# sp_x_str / sp_y_str / sm_x_str / sm_y_str: Julia-eval-able strings (e.g. "1", "im", "-im").
# sp_label / sm_label: output labels (e.g. "Sp", "Sm").
#
# Returns: (value_str, latex_str)  — display strings, NOT a new QuExpr.
# ---------------------------------------------------------------------------
function _nb_collect_spin_pm(expr::_QA.QuExpr,
                              sp_x_str::String, sp_y_str::String,
                              sm_x_str::String, sm_y_str::String,
                              sp_label::String, sm_label::String)
    # Parse transform coefficients (plain Julia scalars, e.g. 1, im, -im)
    sp_x = eval(Meta.parse(sp_x_str))
    sp_y = eval(Meta.parse(sp_y_str))
    sm_x = eval(Meta.parse(sm_x_str))
    sm_y = eval(Meta.parse(sm_y_str))

    # Inverse transform (Cramer's rule):
    # Forward: c_Sp*[sp_x; sp_y] + c_Sm*[sm_x; sm_y] = [cSx; cSy]
    # det(M) = sp_x*sm_y - sm_x*sp_y
    # c_Sp = (sm_y*cSx - sm_x*cSy) / det
    # c_Sm = (-sp_y*cSx + sp_x*cSy) / det
    det_val = sp_x * sm_y - sm_x * sp_y   # e.g. 1*(-im) - 1*(im) = -2im

    # -----------------------------------------------------------------------
    # Group QuExpr terms by (spin_position_in_product, non_spin_key, site)
    # Accumulate Sx/Sy/Sz coefficients per group.
    # -----------------------------------------------------------------------
    # Key tuple; value: vector of (component_sym, coeff)
    # component_sym ∈ {:Sx, :Sy, :Sz}
    groups = Dict{Tuple{Int,String,String},
                  Dict{Symbol,Any}}()
    group_meta = Dict{Tuple{Int,String,String},
                      NamedTuple{(:non_spin,:site),Tuple{Vector,Any}}}()
    leftover = Dict{_QA.QuTerm,Any}()

    for (term, coeff) in expr.terms
        bares_v = collect(term.bares.v)

        # Classify each operator: Sx/Sy are "collectible", Sz and others are "context"
        function _is_collect(bop)
            string(bop.name) == "S" &&
            !occursin("Z", uppercase(string(bop.t)))
        end

        collect_pos = [i for (i, b) in enumerate(bares_v) if _is_collect(b)]

        if length(collect_pos) != 1
            # 0 collectible (Sz-only or no spin) or 2+ Sx/Sy → leftover
            leftover[term] = get(leftover, term, 0) + coeff
            continue
        end

        sidx     = only(collect_pos)
        spin_bop = bares_v[sidx]
        # Everything except the one Sx/Sy is context (including Sz at other sites)
        context  = [b for (i, b) in enumerate(bares_v) if i != sidx]

        site     = spin_bop.inds
        site_key = join(string.(site), ",")
        # Key includes position so Sx(1)*Sz(2) ≠ Sz(2)*Sx(1)
        ctx_key  = join(["$(b.name)($(join(b.inds, ",")))|$(b.t)" for b in context], "__")
        gkey     = (sidx, ctx_key, site_key)

        if !haskey(groups, gkey)
            groups[gkey]     = Dict(:Sx => 0, :Sy => 0, :Sz => 0)
            group_meta[gkey] = (non_spin=context, site=site)
        end

        t_up = uppercase(string(spin_bop.t))
        sym  = occursin("X", t_up) ? :Sx : :Sy
        groups[gkey][sym] = groups[gkey][sym] + coeff
    end

    # -----------------------------------------------------------------------
    # Build output string
    # -----------------------------------------------------------------------
    parts = String[]

    for gkey in sort(collect(keys(groups)), by=k->k[1])
        g    = groups[gkey]
        meta = group_meta[gkey]
        cSx, cSy, cSz = g[:Sx], g[:Sy], g[:Sz]
        site = meta.site
        non  = meta.non_spin

        # Apply inverse transform; use expand() to normalise (fast, no simplify)
        _c_Sp_raw = (sm_y * cSx - sm_x * cSy) / det_val
        _c_Sm_raw = (-sp_y * cSx + sp_x * cSy) / det_val
        c_Sp = try Symbolics.expand(_c_Sp_raw) catch; _c_Sp_raw end
        c_Sm = try Symbolics.expand(_c_Sm_raw) catch; _c_Sm_raw end

        site_str = join(string.(site), ",")
        # Display each context operator with proper component name (Sx/Sy/Sz for spins)
        function _ctx_display(b)
            nm = string(b.name)
            idx = join(string.(b.inds), ",")
            if nm == "S"
                t_u = uppercase(string(b.t))
                comp = occursin("X",t_u) ? "x" : (occursin("Y",t_u) ? "y" : "z")
                return "S$(comp)($(idx))"
            end
            # Creation operators: show with dagger
            endswith(string(b.t), "Create_") ? "$(nm)†($(idx))" : "$(nm)($(idx))"
        end
        non_str  = join(_ctx_display.(non), " ")
        sep      = isempty(non_str) ? "" : " "

        for (c, lbl) in [(c_Sp, "$(sp_label)($(site_str))"),
                          (c_Sm, "$(sm_label)($(site_str))"),
                          (cSz,  "Sz($(site_str))")]
            # Zero check (handles 0, 0.0, 0//1, symbolic zero)
            _is_zero = try iszero(Symbolics.expand(c)) catch; false end
            _is_zero && continue
            cs = string(Symbolics.expand(c))
            # Clean up Julia rational literals: n//1 → n, n//m → n/m
            cs = replace(cs, r"\((-?\d+)//1\)" => s"\1")   # (2//1)  → 2
            cs = replace(cs, r"(-?\d+)//1\b"   => s"\1")   # -2//1   → -2
            cs = replace(cs, r"\((-?\d+)//(\d+)\)" => s"(\1/\2)")  # (-1//2) → (-1/2)
            cs = replace(cs, r"\b(-?\d+)//(\d+)\b"  => s"\1/\2")   # 1//2    → 1/2
            push!(parts, "($(cs)) $(lbl)$(sep)$(non_str)")
        end
    end

    # Leftover terms (0 or 2+ spin operators per term)
    if !isempty(leftover)
        push!(parts, string(_QA.QuExpr(leftover)))
    end

    val = isempty(parts) ? "0" : join(parts, " + ")
    return (val, val)   # same string for value and latex (display only)
end

# ---------------------------------------------------------------------------
# _nb_operator_identity(nb_var, lhs_name, rhs_template)
#
# Replace all occurrences of operator `lhs_name` in a QuExpr with a linear
# combination of operators given by `rhs_template`.
#
# This is the QA-layer implementation of expression-pattern substitution for
# non-commutative operators.  The same user operation ("E2 の Sx を E3 で
# 置き換えて") routes here when the LHS symbol is a registered operator.
#
# Arguments
# ---------
#   nb_var       : Julia session variable name, e.g. "_nb_E2"
#   lhs_name     : base operator name to replace, e.g. "Sx"
#   rhs_template : Julia expression string with "IDX" as placeholder for the
#                  matched operator's index tuple, e.g. "(Sp(IDX)+Sm(IDX))/2"
#                  For multi-index operators use IDX_0, IDX_1, ...
#
# Algorithm
# ---------
# For each (QuTerm, coeff) in expr.terms:
#   Decompose bares into individual BaseOperators.
#   For each op that matches lhs_name, substitute with rhs(same_index).
#   Rebuild using QA multiplication (left * rhs_QuExpr * right) so that
#   non-commutativity is preserved.
#   Sum all terms, then call normal_form.
#
# Backends other than Julia/QA (Maxima, SymPy) use their own substitute
# implementations -- this function is QA-only.
# ---------------------------------------------------------------------------
function _nb_operator_identity(expr::_QA.QuExpr, lhs_name::String,
                                rhs_template::String)

    # Helper: given a BaseOperator's index tuple, instantiate the rhs QuExpr.
    # Replaces IDX (or IDX_0, IDX_1, ...) with the actual index values.
    function _instantiate_rhs(bop)
        inds = bop.inds
        rhs_str = rhs_template
        if length(inds) == 1
            # Single-index operator: replace IDX and IDX_0
            idx_str = string(inds[1])
            rhs_str = replace(rhs_str, "IDX_0" => idx_str)
            rhs_str = replace(rhs_str, "IDX"   => idx_str)
        elseif length(inds) > 1
            # Multi-index: replace IDX_0, IDX_1, ...
            for (n, idx) in enumerate(inds)
                rhs_str = replace(rhs_str, "IDX_$(n-1)" => string(idx))
            end
            rhs_str = replace(rhs_str, "IDX" => string(inds[1]))
        end
        # Evaluate: result may be a QuExpr (operator) or a scalar (Symbolics.Num / Number).
        # Scalars are promoted to a scalar QuExpr (identity operator × scalar).
        rhs_val = eval(Meta.parse(rhs_str))
        if isa(rhs_val, _QA.QuExpr)
            return rhs_val
        else
            # Scalar expression — multiply identity QuExpr by the scalar
            return rhs_val * one(_QA.QuExpr)
        end
    end

    # Helper: wrap a single BaseOperator as a length-1 QuExpr.
    function _single_op_quexpr(bop, term_meta)
        single_bares = _QA.BaseOpProduct([bop])
        qt = _QA.QuTerm(term_meta.nsuminds, term_meta.δs, term_meta.params,
                        term_meta.expvals, term_meta.corrs, single_bares)
        return _QA.QuExpr(Dict(qt => 1))
    end

    # Build a matcher: lhs_name matches if bop.name == lhs_name
    # OR if lhs_name encodes a spin component (e.g. "Sz" matches name="S", t contains "TLSz").
    # This handles @tlsxyz_ops operators where all components share name "S" but differ in .t
    function _bop_matches_lhs(bop)
        bname = string(bop.name)
        bname == lhs_name && return true
        # Spin component: lhs_name starts with bname and remainder appears in bop.t
        if startswith(lhs_name, bname) && length(lhs_name) > length(bname)
            suffix = lhs_name[length(bname)+1:end]  # e.g. "z" from "Sz"
            return occursin(uppercase(suffix), uppercase(string(bop.t)))
        end
        return false
    end

    result = zero(_QA.QuExpr)

    for (term, coeff) in expr.terms
        bares_v = term.bares.v

        # Start with scalar (metadata part of this term, operators stripped).
        # We rebuild the operator content by sequential QA multiplication.
        scalar_term = _QA.QuExpr(Dict(
            _QA.QuTerm(term.nsuminds, term.δs, term.params,
                       term.expvals, term.corrs,
                       _QA.BaseOpProduct()) => coeff
        ))

        acc = scalar_term
        for bop in bares_v
            if _bop_matches_lhs(bop)
                # Replace this operator with the rhs QuExpr
                acc = acc * _instantiate_rhs(bop)
            else
                # Keep this operator as-is
                acc = acc * _single_op_quexpr(bop, term)
            end
        end
        result = result + acc
    end

    result = normal_form(result)
    latex  = _nb_latex(result)
    return (string(result), latex)
end

"""
    _nb_wick_contract(expr, rules_json, include_constant)

Wick contraction: extract a specific operator pair from an n-th order
operator product, replace it with a c-number (contraction value), and
keep the remaining (n-2)-th order operator product.

Arguments
---------
- `expr`             : QuExpr to transform
- `rules_json`       : JSON array of contraction rules.
                       Each rule is a dict with keys:
                         "op1_name"   : operator name for first operator (e.g. "a")
                         "op1_create" : true if op1 is a creation operator
                         "op2_name"   : operator name for second operator
                         "op2_create" : true if op2 is a creation operator
                         "value"      : contraction value expression (string,
                                        with INDEX1 and INDEX2 as placeholders
                                        for the indices of op1 and op2)
- `include_constant` : if true, also return doubly-contracted constant terms

Returns
-------
A JSON array of strings, one per contraction term.
Each element is a Julia expression string for the contracted QuExpr.
The caller (Python) registers each as a separate expression.

Physics
-------
For fermions, contracting operators at positions i < j in the bares vector:
  sign = (-1)^(j - i - 1)
This comes from anticommuting op[j] leftward past (j-i-1) operators to make
it adjacent to op[i].

For a rule matching op1 at position i and op2 at position j (i < j):
  result += sign * contraction_value(inds[i], inds[j]) * remaining_QuTerm

where remaining_QuTerm uses BaseOpProduct with ops at all positions except i, j.
"""
# _nb_wc_build_value(value_tmpl, bop1, bop2, symmetry_map)
# Build contraction value from template + two BaseOperators.
# Plain scalar (no INDEX) → eval; funcname(INDEX1_N,...) → param with all inds.
# INDEX1/INDEX2 alone are aliases for INDEX1_0/INDEX2_0.
# symmetry_map: Dict{String,String} mapping funcname → symmetry type.
# When symmetry is "antisymmetric", κ(j,i) is normalized to -κ(i,j) by
# checking if (idx1,idx2) < (idx2,idx1) lexicographically and negating if so.
function _nb_wc_build_value(value_tmpl::String, bop1, bop2,
                             symmetry_map::Dict{String,String}=Dict{String,String}())
    if !occursin("INDEX1", value_tmpl) && !occursin("INDEX2", value_tmpl)
        # Phase 1: plain scalar
        return eval(Meta.parse(value_tmpl))
    end

    # Phase 2: parse "funcname(arg0, arg1, ...)"
    _tmpl_m = match(r"^(\w+)\((.*)\)$", strip(value_tmpl))
    _fn_name = isnothing(_tmpl_m) ?
                   Symbol(strip(split(value_tmpl,"(")[1])) :
                   Symbol(strip(_tmpl_m[1]))
    _arg_tokens = isnothing(_tmpl_m) ?
                      ["INDEX1", "INDEX2"] :
                      [strip(s) for s in split(_tmpl_m[2], ",")]

    _qa_inds = _QA.QuIndex[]
    for _tok in _arg_tokens
        if _tok == "INDEX1" || _tok == "INDEX1_0"
            push!(_qa_inds, length(bop1.inds)>0 ? bop1.inds[1] : _QA.QuIndex("1"))
        elseif startswith(_tok, "INDEX1_")
            _n = parse(Int, _tok[8:end]) + 1
            push!(_qa_inds, length(bop1.inds)>=_n ? bop1.inds[_n] : _QA.QuIndex("?"))
        elseif _tok == "INDEX2" || _tok == "INDEX2_0"
            push!(_qa_inds, length(bop2.inds)>0 ? bop2.inds[1] : _QA.QuIndex("1"))
        elseif startswith(_tok, "INDEX2_")
            _n = parse(Int, _tok[8:end]) + 1
            push!(_qa_inds, length(bop2.inds)>=_n ? bop2.inds[_n] : _QA.QuIndex("?"))
        else
            push!(_qa_inds, _QA.QuIndex(string(eval(Meta.parse(_tok)))))
        end
    end

    # Symmetry normalization: for antisymmetric functions κ(j,i) → -κ(i,j).
    # Canonical order: lexicographic on index strings.
    # If current order > canonical, swap and negate (antisymmetric) or just swap (symmetric).
    _fn_str = string(_fn_name)
    _sym = get(symmetry_map, _fn_str, "")
    if _sym in ("antisymmetric", "symmetric") && length(_qa_inds) >= 2
        # Split indices into op1-group and op2-group.
        # For 2-arg: [idx1] and [idx2]; for 4-arg (k,σ,p,τ): [idx1_0,idx1_1] and [idx2_0,idx2_1]
        _half = length(_qa_inds) ÷ 2
        _inds1 = _qa_inds[1:_half]
        _inds2 = _qa_inds[_half+1:end]
        _str1  = join(string(idx) for idx in _inds1)
        _str2  = join(string(idx) for idx in _inds2)
        if _str1 > _str2
            # Swap: κ(j,i) → (-1) × κ(i,j) for antisymmetric, κ(i,j) for symmetric
            _qa_inds = vcat(_inds2, _inds1)
            if _sym == "antisymmetric"
                return -1 * param(_fn_name, _qa_inds...)
            else
                return param(_fn_name, _qa_inds...)
            end
        end
    end

    return param(_fn_name, _qa_inds...)
end

function _nb_wick_contract(expr::_QA.QuExpr, rules_json::String,
                            include_constant::Bool=false,
                            symmetry_json::String="{}")
    rules        = JSON3.read(rules_json, Vector{Dict{String,Any}})
    symmetry_map = JSON3.read(symmetry_json, Dict{String,String})

    # Collect all single-contraction results as Julia expression strings.
    # Key: string description, Value: QuExpr
    results = Vector{_QA.QuExpr}()

    for (op, coeff) in expr.terms
        bares_v = op.bares.v
        n = length(bares_v)
        n < 2 && continue  # nothing to contract

        for rule in rules
            op1_name   = rule["op1_name"]
            op1_create = rule["op1_create"]   # true = creation (†), false = annihilation
            op2_name   = rule["op2_name"]
            op2_create = rule["op2_create"]
            value_tmpl = rule["value"]        # e.g. "V_HF(INDEX1,INDEX2)"

            # Scan all (i, j) pairs with i < j
            for i in 1:n, j in (i+1):n
                bop_i = bares_v[i]
                bop_j = bares_v[j]

                # Check operator names (name is QuOpName, compare via string())
                string(bop_i.name) == op1_name || continue
                string(bop_j.name) == op2_name || continue

                # Check create/destroy type using suffix of type string
                _i_is_create = endswith(string(bop_i.t), "Create_")
                _j_is_create = endswith(string(bop_j.t), "Create_")
                _i_is_create == op1_create || continue
                _j_is_create == op2_create || continue

                # Fermionic sign: (-1)^(j - i - 1)
                _sign = (-1)^(j - i - 1)

                # Build contraction value.
                # Two strategies depending on whether INDEX1/INDEX2 appear in value_tmpl:
                #
                # Phase 1 (no INDEX): evaluate as plain scalar expression.
                #   e.g. "V_HF" → eval(:V_HF) → Symbolics.Num scalar
                #
                # Phase 2 (with INDEX): build a QuantumAlgebra param QuExpr.
                #   e.g. "V_HF(INDEX1,INDEX2)"
                #        → param(:V_HF, inds_i[1], inds_j[1])          [single-index ops]
                #   e.g. "V_HF(INDEX1_0,INDEX1_1,INDEX2_0,INDEX2_1)"
                #        → param(:V_HF, inds_i[1], inds_i[2], inds_j[1], inds_j[2])
                #                                                        [multi-index ops, e.g. (k,σ)]
                #   INDEX1   / INDEX2   are aliases for INDEX1_0 / INDEX2_0 (backward compat).
                #   INDEX1_N / INDEX2_N address the N-th index (0-based) of op1 / op2.

                # Build remaining BaseOpProduct (all positions except i and j)
                _rem_indices = [k for k in 1:n if k != i && k != j]
                _rem_bares   = _QA.BaseOpProduct([bares_v[k] for k in _rem_indices])

                # Build contraction value via shared helper (_nb_wc_build_value).
                # Returns QuExpr (Phase 2: INDEX template) or Number (Phase 1: plain scalar).
                _cval = _nb_wc_build_value(value_tmpl, bop_i, bop_j, symmetry_map)

                if isempty(_rem_indices)
                    include_constant || continue
                    if isa(_cval, _QA.QuExpr)
                        push!(results, normal_form(coeff * _sign * _cval))
                    else
                        _new_term = _QA.QuTerm(op.nsuminds, op.δs, op.params,
                                               op.expvals, op.corrs,
                                               _QA.BaseOpProduct(Vector{_QA.BaseOperator}()))
                        push!(results, normal_form(_QA.QuExpr(Dict(_new_term => coeff * _sign * _cval))))
                    end
                else
                    _rem_term  = _QA.QuTerm(op.nsuminds, op.δs, op.params,
                                            op.expvals, op.corrs, _rem_bares)
                    _rem_qexpr = _QA.QuExpr(Dict(_rem_term => 1))
                    if isa(_cval, _QA.QuExpr)
                        push!(results, normal_form(coeff * _sign * _cval * _rem_qexpr))
                    else
                        push!(results, normal_form(_QA.QuExpr(Dict(_rem_term => coeff * _sign * _cval))))
                    end
                end
            end
        end
    end

    # Double contractions (include_constant=true): produce scalar (c-number) terms.
    # Scan all ordered combinations of two non-overlapping pairs (i1,j1),(i2,j2)
    # where i1 < j1, i2 < j2, and {i1,j1} ∩ {i2,j2} = ∅.
    if include_constant
        for (op, coeff) in expr.terms
            bares_v = op.bares.v
            n = length(bares_v)
            n < 4 && continue

            # All rule pairs (possibly the same rule twice)
            for rule1 in rules, rule2 in rules
                for i1 in 1:n, j1 in (i1+1):n
                    # Pair 1 must match rule1
                    bop_i1 = bares_v[i1]; bop_j1 = bares_v[j1]
                    string(bop_i1.name) == rule1["op1_name"] || continue
                    string(bop_j1.name) == rule1["op2_name"] || continue
                    endswith(string(bop_i1.t),"Create_") == rule1["op1_create"] || continue
                    endswith(string(bop_j1.t),"Create_") == rule1["op2_create"] || continue

                    sign1 = (-1)^(j1 - i1 - 1)
                    val1  = _nb_wc_build_value(rule1["value"], bop_i1, bop_j1, symmetry_map)

                    for i2 in 1:n, j2 in (i2+1):n
                        (i2 == i1 || i2 == j1 || j2 == i1 || j2 == j1) && continue
                        # Avoid double-counting: require i2 > i1 (canonical order)
                        i2 > i1 || continue

                        bop_i2 = bares_v[i2]; bop_j2 = bares_v[j2]
                        string(bop_i2.name) == rule2["op1_name"] || continue
                        string(bop_j2.name) == rule2["op2_name"] || continue
                        endswith(string(bop_i2.t),"Create_") == rule2["op1_create"] || continue
                        endswith(string(bop_j2.t),"Create_") == rule2["op2_create"] || continue

                        # Adjusted positions after removing (i1, j1)
                        adj_i2 = i2 - (i2 > i1 ? 1 : 0) - (i2 > j1 ? 1 : 0)
                        adj_j2 = j2 - (j2 > i1 ? 1 : 0) - (j2 > j1 ? 1 : 0)
                        sign2 = (-1)^(adj_j2 - adj_i2 - 1)

                        val2 = _nb_wc_build_value(rule2["value"], bop_i2, bop_j2, symmetry_map)

                        # All 4 positions contracted → scalar (empty bares)
                        _rem_indices = [k for k in 1:n if k ∉ (i1, j1, i2, j2)]
                        if isempty(_rem_indices)
                            # val1 and val2 may be QuExpr (Phase 2) or Number (Phase 1)
                            if isa(val1, _QA.QuExpr) || isa(val2, _QA.QuExpr)
                                push!(results, normal_form(coeff * sign1 * sign2 * val1 * val2))
                            else
                                _scalar_term = _QA.QuTerm(op.nsuminds, op.δs, op.params,
                                                           op.expvals, op.corrs,
                                                           _QA.BaseOpProduct(Vector{_QA.BaseOperator}()))
                                push!(results, normal_form(_QA.QuExpr(Dict(_scalar_term => coeff * sign1 * sign2 * val1 * val2))))
                            end
                        end
                    end
                end
            end
        end
    end

    return results
end

# Helper: coefficient of var^n in expr using repeated differentiation.
# coeff(f, x, n) = (d^n f/dx^n)|_{x=0} / n!
# Used by _nb_extract_order and _nb_truncate_series.
function _nb_poly_coeff(se, sv_sym, n::Int)
    _d = Symbolics.expand(se)
    for _ in 1:n
        _d = Symbolics.expand_derivatives(Symbolics.Differential(sv_sym)(_d))
    end
    _at_zero = Symbolics.substitute(_d, Dict(sv_sym => 0))
    # Evaluate concrete numeric calls like exp(0)=1, sin(0)=0, etc.
    _at_zero = Symbolics.simplify(Symbolics.substitute(_at_zero,
        Dict(Symbolics.exp(Symbolics.Num(0)) => 1,
             Symbolics.sin(Symbolics.Num(0)) => 0,
             Symbolics.cos(Symbolics.Num(0)) => 1,
             Symbolics.log(Symbolics.Num(1)) => 0)))
    _fac = factorial(n)
    return Symbolics.simplify(_at_zero / _fac)
end


# ---------------------------------------------------------------------------
# _nb_extract_order / _nb_truncate_series
#
# Two dispatch paths (selected automatically by Julia's type system):
#
#   String input  → re-evaluate to get a Symbolics.Num or QuExpr, then dispatch
#   QuExpr input  → operator path: inspect each QuTerm's Symbolics coefficient,
#                   extract the var^n component, reassemble QuExpr preserving
#                   operator ordering
#   Num input     → scalar path: _nb_poly_coeff directly
# ---------------------------------------------------------------------------

# Internal: extract var^n coefficient from a Symbolics scalar.
function _nb_scalar_coeff(coeff, sv_sym, n::Int)
    # Plain numbers (Int, Rational, Complex) need to be wrapped first.
    _c = isa(coeff, Number) ? Symbolics.Num(coeff) : coeff
    return Symbolics.simplify(_nb_poly_coeff(_c, sv_sym, n))
end

# Internal QuExpr path: collect terms whose coeff has a var^n component.
function _nb_extract_order_qexpr(expr::_QA.QuExpr, sv_sym, n::Int)
    _result_terms = Dict{_QA.QuTerm, Any}()
    for (op, coeff) in expr.terms
        _c_n = _nb_scalar_coeff(coeff, sv_sym, n)
        iszero(_c_n) && continue
        _result_terms[op] = get(_result_terms, op, Symbolics.Num(0)) + _c_n
    end
    isempty(_result_terms) && return normal_form(_QA.QuExpr(Dict{_QA.QuTerm,Any}()))
    return normal_form(_QA.QuExpr(_result_terms))
end

# Internal QuExpr path: keep terms up to var^order.
function _nb_truncate_qexpr(expr::_QA.QuExpr, sv_sym, order::Int)
    _result_terms = Dict{_QA.QuTerm, Any}()
    for (op, coeff) in expr.terms
        _c_trunc = Symbolics.Num(0)
        for k in 0:order
            _c_k = _nb_scalar_coeff(coeff, sv_sym, k)
            iszero(_c_k) && continue
            _c_trunc = _c_trunc + _c_k * sv_sym^k
        end
        iszero(Symbolics.simplify(_c_trunc)) && continue
        _result_terms[op] = get(_result_terms, op, Symbolics.Num(0)) +
                            Symbolics.expand(_c_trunc)
    end
    isempty(_result_terms) && return normal_form(_QA.QuExpr(Dict{_QA.QuTerm,Any}()))
    return normal_form(_QA.QuExpr(_result_terms))
end

# ---------------------------------------------------------------------------
# _nb_operator_taylor_expand
#
# Compute the truncated Taylor expansion of a scalar function f applied to
# an operator H:
#
#   f(H) ≈ Σ_{k=0}^{order}  a_k · H^k
#
# where a_k = k-th Taylor coefficient of f(var) at var=0 (scalar).
#
# H^k is computed via QuantumAlgebra's operator multiplication, so:
#   - Anti-commutation relations are applied correctly (normal_form)
#   - Operator ordering is preserved
#   - Projector properties (n_k² = n_k) are exploited automatically
#
# Arguments:
#   f_str   : scalar function of var, e.g. "exp(-beta*x)" or "1/(1-x)"
#   op_expr : the operator H (QuExpr)
#   var_str : the dummy variable in f_str that H replaces (e.g. "x")
#   order   : truncation order
#
# Example:
#   H = xi * c†(1)c(1)
#   _nb_operator_taylor_expand("exp(-beta*x)", H, "x", 3)
#   → 1 - beta·H + (beta²/2)·H² - (beta³/6)·H³
#   → 1 - beta·xi·n + (beta²xi²/2)·n - (beta³xi³/6)·n   (n² = n applied)
# ---------------------------------------------------------------------------
function _nb_operator_taylor_expand(f_str::String, op_expr::_QA.QuExpr,
                                     var_str::String, order::Int)
    sv     = Symbolics.variable(Symbol(var_str))
    f_sym  = _nb_eval(f_str)          # scalar Symbolics expression for f(var)
    _result = normal_form(op_expr^0)  # start with identity (QuExpr 1)
    _result = normal_form(_nb_scalar_coeff(f_sym, sv, 0) * _result)

    _Hpow = op_expr                   # H^1
    for k in 1:order
        _ak = _nb_scalar_coeff(f_sym, sv, k)
        if !iszero(Symbolics.simplify(_ak))
            _result = normal_form(_result + _ak * _Hpow)
        end
        k < order && (_Hpow = normal_form(_Hpow * op_expr))
    end
    return _result
end

# Public API — String input (scalar expression or re-parseable canonical)
function _nb_extract_order(expr_str::String, var_str::String, n::Int)
    _sv_sym = Symbolics.variable(Symbol(var_str))
    _se     = _nb_eval(expr_str)
    isa(_se, _QA.QuExpr) && return _nb_extract_order_qexpr(_se, _sv_sym, n)
    return string(Symbolics.simplify(_nb_poly_coeff(_se, _sv_sym, n)))
end

# Public API — QuExpr input (direct Julia variable, no string round-trip)
function _nb_extract_order(expr::_QA.QuExpr, var_str::String, n::Int)
    _sv_sym = Symbolics.variable(Symbol(var_str))
    return _nb_extract_order_qexpr(expr, _sv_sym, n)
end

# Public API — String input
function _nb_truncate_series(expr_str::String, var_str::String, order::Int)
    _sv_sym = Symbolics.variable(Symbol(var_str))
    _se     = _nb_eval(expr_str)
    isa(_se, _QA.QuExpr) && return _nb_truncate_qexpr(_se, _sv_sym, order)
    _result = Symbolics.Num(0)
    for k in 0:order
        _c = _nb_poly_coeff(_se, _sv_sym, k)
        iszero(Symbolics.simplify(_c)) && continue
        _result = Symbolics.expand(_result + _c * _sv_sym^k)
    end
    return string(_result)
end

# Public API — QuExpr input
function _nb_truncate_series(expr::_QA.QuExpr, var_str::String, order::Int)
    _sv_sym = Symbolics.variable(Symbol(var_str))
    return _nb_truncate_qexpr(expr, _sv_sym, order)
end

# Command loop.
while !eof(stdin)
    cmd = readline(stdin)
    isempty(cmd) && continue
    (cmd == "exit()" || cmd == "quit()") && break
    try
        result = _nb_eval(cmd)
        println(_NB_BEGIN)
        if isa(result, Tuple{String,String})
            # SymPy engine result: (value_str, latex_str) — unpack directly.
            # Produced by _nb_sympy_fermion_comm / _nb_sympy_boson_comm.
            println(result[1])
            println(_NB_SEP)
            println(result[2])
        else
            println(string(result))
            println(_NB_SEP)
            try
                println(_nb_latex(result))
            catch
                println("")
            end
        end
        println(_NB_END)
    catch ex
        println(_NB_BEGIN)
        println(_NB_ERR)
        println(sprint(showerror, ex))
        println(_NB_SEP)
        println("")
        println(_NB_END)
    end
    flush(stdout)
end




