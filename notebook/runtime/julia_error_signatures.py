"""What a raw Julia failure means, as patterns.

Beside the handler rather than inside it, for the reason
``maxima_error_classify`` is: error signatures are a list that only grows, and
the handler is already one of the largest files in the runtime.  Still
backend-specific knowledge — the Backend Encapsulation Rule is about which
backend owns it, not which file holds it.
"""
import re

#: Julia answers an applied-but-undeclared name with
#: "Sym G_0 is not callable. Use @syms G_0(var1, var2,...) to create it…".
#: Translating that AFTER the fact — rather than refusing before the CAS —
#: is deliberate: measured 2026-08-01, a pre-CAS guard had to allow-list every
#: name the system itself emits, and two rounds of the full suite showed it
#: still refused legitimate ones (`c(1)`, and `adjoint(c(1))` from the
#: generator's OWN output re-entering as canonical text).  The defect here is
#: message quality, not a wrong answer, so a fix whose blast radius is every
#: function call in a Julia session is the wrong trade.
UNDECLARED_CALL_RE = re.compile(r"Sym\s+(\S+?)\s+is not callable")

#: QuantumAlgebra restricts index names to one character; `param(:G_0, :k, :omega)`
#: answers "ArgumentError: index names must be single character, got omega".
MULTICHAR_INDEX_RE = re.compile(
    r"index names must be single character,\s*got\s+(\S+)")

#: Symbolics cannot differentiate a QuantumAlgebra value.  Measured 2026-08-02:
#: an indexed scalar reaches the CAS as `param(:xi, QuIndex("k"))`, which
#: multiplies operators and Symbolics variables happily but answers
#: "MethodError: no method matching (::Differential)(::QuExpr)" under
#: `derivative`.  That is the ONE thing the param route cannot do
#: (JULIA-INDEXED-SCALAR), so it gets said in words rather than left as a
#: Julia MethodError.
QUEXPR_DIFF_RE = re.compile(r"Differential\)\(::QuExpr")


