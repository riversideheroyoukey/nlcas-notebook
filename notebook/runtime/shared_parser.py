from __future__ import annotations
import re
from typing import List, Optional, Tuple, Union

from notebook.models.ir import (
    Expr, Add, Mul, Pow, Symbol, Number, FunctionCall, Derivative, Integral,
    Commutator, Equation, OpaqueExpr, SurfaceExpr, Juxtaposition, BracketExpr, Assignment,
    Subscript, Superscript, DotProduct, CrossProduct, Decorated
)

# Decoration names produced by LatexConverter._decorations().
# When an identifier is subscripted by one of these names (e.g. H_hat),
# SharedParser produces Decorated(base, decoration) instead of Subscript.
# ``dagger`` is not written as an accent — it arrives as the SUPERSCRIPT
# `X^\dagger` and is turned into a decorated name by the resolver (a Maxima
# session has no operator algebra, so the adjoint of an undeclared symbol is a
# NAME).  It is listed here because the CAS returns that name as `psi_dagger`,
# and re-reading it as "psi subscripted by dagger" is how a creation operator
# turned into `psi(dagger)`.  Kept in step with latex_converter and
# backend_handler by tests/test_field_operator_notation.py.
from notebook.runtime.script_label import is_label_name  # noqa: E402


def subscript_name_node(base, sub):
    r"""`base` subscripted by `sub` — `Decorated` for a decoration, else `Subscript`.

    One question, two askers: the parser builds the name chain with it, and the
    resolver rebuilds the head when a SPACED parenthesis turns out to be
    multiplication (`\dot{a} \left(m+M\right)`).  Asked in two places with two
    answers, the rebuilt head came out as `a(dot)` while the same text written
    without the parenthesis is the name `a_dot` (measured 2026-09-25).
    """
    if isinstance(sub, Symbol) and (sub.name in _DECORATION_NAMES
                                    or is_label_name(sub.name)):
        return Decorated(base=base, decoration=sub.name)
    return Subscript(base=base, sub=sub)

_DECORATION_NAMES: frozenset[str] = frozenset({
    "hat", "tilde", "bar", "dot", "ddot", "check", "breve", "acute", "grave",
    "dagger", "cal", "scr",
})

class UnknownNotationError(ValueError):
    """Raised when the input contains notation the parser cannot represent.

    Before 2026-07-03 the tokenizer silently SKIPPED characters it did not
    know — ``|A|`` parsed as plain ``A``, silently dropping the absolute
    value / determinant / norm.  Unsupported notation must fail loudly.

    Carries an i18n key (like PrimeDerivativeError) so callers that know the
    session language can translate before surfacing to the user.
    """
    def __init__(self, i18n_key: str, **fmt_kwargs: object) -> None:
        super().__init__(i18n_key)
        self.i18n_key = i18n_key
        self.fmt_kwargs = fmt_kwargs


# Characters that are mathematically meaningful but NOT representable by the
# parser.  Each maps to the i18n key explaining what to write instead.
# NOTE: characters the tokenizer legitimately ignores (``\`` from TeX commands
# like ``\sum``/``\delta``) must NOT be listed here.
_UNSUPPORTED_CHARS: dict[str, str] = {
    "|": "parser.unsupported_notation.pipe",
    # `#` is where every spelling of "not equal" arrives (normalize_relational_ops
    # folds \neq / \ne / != into it).  The IR has no not-equal relation, and
    # inventing a half-supported one would be worse than saying so: what the CAS
    # can actually consume is `assume(notequal(a,b))`, an ASSUMPTION, not an
    # expression.  Until that path exists this refuses with a reason — which is
    # already an improvement on what each spelling silently became (`a * neq * b`
    # and `factorial(a) = b`, both successful).
    "#": "parser.unsupported_notation.not_equal",
}

_T_SPECIAL  = "SPECIAL"
_T_ID       = "ID"
_T_FLOAT    = "FLOAT"
_T_INT      = "INT"
_T_OP       = "OP"
_T_SUB      = "SUB"
_T_LPAREN   = "LP"
_T_RPAREN   = "RP"
_T_COMMA    = "COM"
_T_SEMI     = "SEMI"
_T_QUOTE    = "QUOTE"
_T_LBRACKET = "LBRACKET"
_T_RBRACKET = "RBRACKET"
_T_LBRACE   = "LBRACE"
_T_RBRACE   = "RBRACE"
_T_BANG     = "BANG"
_T_EQ       = "EQ"

Token = Tuple[str, str]

_TOKEN_RE = re.compile(
    r"(?P<SPECIAL>%[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<QUOTE>')"
    r"|(?P<ID>[A-Za-z][A-Za-z0-9]*)"
    r"|(?P<FLOAT>\d+\.\d+)"
    r"|(?P<INT>\d+)"
    r"|(?P<POW>\*\*|\^)"
    r"|(?P<SUB>_)"
    r"|(?P<BANG>!)"
    r"|(?P<EQ>=)"
    r"|(?P<OP>[+\-*/.~])"
    r"|(?P<LP>\()"
    r"|(?P<RP>\))"
    r"|(?P<COM>,)"
    r"|(?P<SEMI>;)"
    r"|(?P<LB>\[)"
    r"|(?P<RB>\])"
    r"|(?P<LBR>\{)"
    r"|(?P<RBR>\})"
    r"|(?P<WS>\s+)"
)

#: The only characters `_TOKEN_RE` is ALLOWED to leave unconsumed.
#:
#: ``\`` introduces a TeX command whose name the ID rule then takes (``\sum`` →
#: ``sum``); ``†`` is the adjoint mark, carried by the operator name around it.
#: Everything else that the regex cannot match used to be dropped in silence,
#: which is how a pasted en dash turned `x – 1` into the PRODUCT `x * 1`, a
#: typographic prime turned `f′(x)` into the undifferentiated `f(x)`, and an
#: accented name was truncated (`café` → `caf`, `x温` → `x`, the latter then
#: COLLIDING with a variable named x) — every one of them reported as a success.
#:
#: The guard this replaces named the two characters it knew about (``|`` and
#: ``#``), so any character nobody had thought of was still dropped.  Inverting
#: it to "anything not consumed is refused" is what makes the check complete
#: rather than a list that the next paste outruns.
#:
#: Measured over the full suite before inverting: these two are the ONLY
#: characters the corpus relies on being skipped (733 and 80 occurrences); the
#: other two cases it turned up were a Maxima ``pmatrix`` blob's ``&`` and a
#: lattice name in Japanese, both of which were already parsing to something
#: other than what they say.
_SKIPPABLE_CHARS = frozenset("\\†")


def _tokenize(s: str) -> Tuple[List[Token], set]:
    """Tokenize *s*, dropping whitespace, and also return the set of token
    indices that were immediately preceded by whitespace in the source.  That
    ``ws_before`` set lets the parser tell ``f (x)`` (space → possible product)
    apart from ``f(x)`` (function call) without otherwise tracking positions.

    Raises :class:`UnknownNotationError` when a character is neither tokenised
    nor in :data:`_SKIPPABLE_CHARS` — see that constant for why."""
    _pos = 0
    for _m in _TOKEN_RE.finditer(s):
        for _ch in s[_pos:_m.start()]:
            if _ch not in _SKIPPABLE_CHARS and not _ch.isspace():
                raise UnknownNotationError(
                    "parser.unsupported_notation.unreadable_char", token=_ch)
        _pos = _m.end()
    for _ch in s[_pos:]:
        if _ch not in _SKIPPABLE_CHARS and not _ch.isspace():
            raise UnknownNotationError(
                "parser.unsupported_notation.unreadable_char", token=_ch)

    tokens: List[Token] = []
    ws_before: set = set()
    _pending_ws = False
    for m in _TOKEN_RE.finditer(s):
        kind = m.lastgroup
        if kind == "WS":
            _pending_ws = True
            continue
        if _pending_ws:
            ws_before.add(len(tokens))
            _pending_ws = False
        if kind == "POW":
            tokens.append((_T_OP, "^"))
        elif kind == "LP":
            tokens.append((_T_LPAREN, "("))
        elif kind == "RP":
            tokens.append((_T_RPAREN, ")"))
        elif kind == "COM":
            tokens.append((_T_COMMA, ","))
        elif kind == "SEMI":
            tokens.append((_T_SEMI, ";"))
        elif kind == "QUOTE":
            tokens.append((_T_QUOTE, "'"))
        elif kind == "SPECIAL":
            tokens.append((_T_SPECIAL, m.group()))
        elif kind == "ID":
            tokens.append((_T_ID, m.group()))
        elif kind == "SUB":
            tokens.append((_T_SUB, "_"))
        elif kind == "FLOAT":
            tokens.append((_T_FLOAT, m.group()))
        elif kind == "INT":
            tokens.append((_T_INT, m.group()))
        elif kind == "OP":
            tokens.append((_T_OP, m.group()))
        elif kind == "LB":
            tokens.append((_T_LBRACKET, "["))
        elif kind == "RB":
            tokens.append((_T_RBRACKET, "]"))
        elif kind == "LBR":
            tokens.append((_T_LBRACE, "{"))
        elif kind == "RBR":
            tokens.append((_T_RBRACE, "}"))
        elif kind == "BANG":
            tokens.append((_T_BANG, "!"))
        elif kind == "EQ":
            tokens.append((_T_EQ, "="))
    return tokens, ws_before


def _as_noun(node: Expr) -> Expr:
    """Carry a CAS noun mark on the NAME: `sum(…)` → `'sum(…)`.

    Kept rather than dropped so the generator writes back exactly what the CAS
    sent — an unquoted `integrate(f,x)` is a different request from the
    `'integrate(f,x)` it answered with.  Idempotent.
    """
    if isinstance(node, FunctionCall) and isinstance(node.func, Symbol):
        if node.func.name.startswith("'"):
            return node
        return FunctionCall(func=Symbol(name="'" + node.func.name),
                            args=node.args, spaced=node.spaced)
    if isinstance(node, Symbol) and not node.name.startswith("'"):
        return Symbol(name="'" + node.name)
    return node


def _binds_a_range(node) -> bool:
    r"""A BINDER's subscript (``\\sum_{p \\sigma}``), not a name.

    A bracket glued to a NAME is that name's subscript; glued to a
    binder's subscript it is the SUMMAND —
    ``\\sum_{p \\sigma}\\left[V - \\ldots\\right]``.  Measured 2026-08-21:
    read as a subscript, the Hartree-Fock self-energy stopped being an
    expression at all and the whole line was taken for an instruction.
    The binder names are ``script_label``'s own list, which exists for the
    same reason (a superscript on ``\\sum`` is a BOUND, not a label).
    """
    from notebook.runtime.script_label import _BINDERS
    while isinstance(node, (Decorated, Subscript)):
        node = node.base
    return isinstance(node, Symbol) and node.name.lstrip("\\\\") in _BINDERS


def find_top_level_relation(s: str) -> Optional[Tuple[int, str]]:
    """Find the first top-level relational operator in *s*.

    Recognizes ``<=``, ``>=``, ``<``, ``>``, ``=``, ``#`` at bracket-depth 0,
    still excluding ``:=`` and ``==`` exactly as the equation-only scanner did.
    Returns ``(index, op)`` or ``None``.  This is the SINGLE place that
    defines "what counts as a top-level relation" — the inequality NL
    interceptor (``ui/interpret_inequality.py``) reuses this function rather
    than re-deriving its own detection, so the two can never disagree on
    where a relation splits.

    ``#`` is where every spelling of "not equal" arrives (``normalize_relational_ops``
    folds ``\neq``/``\ne``/``!=`` into it).  It is listed here rather than
    scanned again somewhere else, which is this function's whole point — the
    Maxima handler needs the split to write ``assume(notequal(a,b))``, and a
    second bracket-aware scan for one operator is how two answers to one
    question begin.  Callers that whitelist ops are unaffected: an expression
    carrying ``#`` is refused before it is parsed (``_UNSUPPORTED_CHARS``), and
    the H-1 split scans RAW text, where the spelling is still ``\neq``.
    """
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c in ("(", "[", "{"):
            depth += 1
        elif c in (")", "]", "}"):
            depth -= 1
        elif depth == 0 and c == "#":
            return i, "#"
        elif depth == 0 and c == "≈":
            # Where every spelling of "approximately equal" arrives
            # (`normalize_relational_ops`).  It splits like any other relation
            # so the two sides are READ; what it does not have is a CAS form,
            # which the generator says at the point of use.
            return i, "≈"
        elif depth == 0 and c in ("<", ">", "="):
            if c == "=" and i > 0 and s[i - 1] == ":":
                pass
            elif c == "=" and i + 1 < len(s) and s[i + 1] == "=":
                i += 1
            elif c in ("<", ">") and i + 1 < len(s) and s[i + 1] == "=":
                return i, c + "="
            elif c in ("<", ">"):
                return i, c
            else:
                return i, "="
        i += 1
    return None

def strip_lhs(text: str) -> str:
    r"""``LHS = RHS`` → ``RHS``; anything that is not a top-level equation is
    returned unchanged.

    This is the question five call sites were each answering with
    ``if "=" in t: t.split("=", 1)[1]`` — a test that reads
    ``\sum_{i=1}^{3} i^2`` as the equation ``\sum_{i`` = ``1}^{3} i^2`` and
    throws the head away.  Measured 2026-08-11, still live in the equation-op
    interceptors long after the same defect was repaired in ``engine.py`` and
    ``lhs_formatting``:

        \sum_{i=1}^{3} i^2 = 14   →  trigsimp(ratsimp(1}^3*i^2 = 14))

    Maxima cannot read that, so its scratch name ``_res_`` came back and was
    registered as the result.  Bracket-aware because
    :func:`find_top_level_relation` is; ``:=`` is excluded there too.
    """
    rel = find_top_level_relation(text)
    if rel is None or rel[1] != "=":
        return text
    return text[rel[0] + 1:].strip()


def split_relation(text: str) -> "Optional[Tuple[str, str, str]]":
    """``(lhs, op, rhs)`` at the top level, or None when *text* is not a relation.

    The companion to :func:`strip_lhs` for callers that need both sides.
    """
    rel = find_top_level_relation(text)
    if rel is None:
        return None
    at, op = rel
    return text[:at].strip(), op, text[at + len(op):].strip()


class _Parser:
    def __init__(self, tokens: List[Token], ws_before: Optional[set] = None) -> None:
        self.toks = tokens
        self.pos = 0
        self.ws_before = ws_before or set()

    def peek_ws_before(self) -> bool:
        """True when the current token was preceded by whitespace in the source."""
        return self.pos in self.ws_before

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.toks):
            return self.toks[self.pos]
        return None

    def peek_type(self) -> Optional[str]:
        t = self.peek()
        return t[0] if t else None

    def peek_val(self) -> Optional[str]:
        t = self.peek()
        return t[1] if t else None

    def consume(self) -> Token:
        tok = self.toks[self.pos]
        self.pos += 1
        return tok

    def expect(self, typ: str) -> Token:
        tok = self.peek()
        if tok is None or tok[0] != typ:
            got = tok[1] if tok else "EOF"
            raise SyntaxError(f"Expected {typ}, got {got!r}")
        return self.consume()

    def at_end(self) -> bool:
        return self.pos >= len(self.toks)

    def parse_expr(self) -> Expr:
        # A nested expression starts a fresh context: the "do not attach a
        # bracket here" flag belongs to the ONE subscript operand that set it,
        # and an argument list inside that operand is not part of it.  Left
        # standing, it suppressed the attachment inside every nested call, and
        # `time_order(psi_hat_dagger[I, sigma](x))` — whose own name contains an
        # underscore, so its ARGUMENT is parsed as a subscript operand — read
        # `[I, sigma]` as a COMMUTATOR again (measured 2026-08-21).
        # The same is true of "do not attach a SPACED parenthesis here", which
        # an EXPONENT's operand sets: `a^2 (b)` is `a^2 · b`, but the braces of
        # `a^{2 (b)}` say the product IS the exponent, and the brace group is
        # parsed through here (JUXT-POW-PRECEDENCE).
        _prev = getattr(self, "_no_attached_bracket", False)
        _prev_nsc = getattr(self, "_no_spaced_call", False)
        self._no_attached_bracket = False
        self._no_spaced_call = False
        try:
            return self.parse_add()
        finally:
            self._no_attached_bracket = _prev
            self._no_spaced_call = _prev_nsc

    def parse_add(self) -> Expr:
        left = self.parse_vector_op()
        args: List[Expr] = [left]
        while self.peek_type() == _T_OP and self.peek_val() in ("+", "-"):
            op = self.consume()[1]
            right = self.parse_vector_op()
            if op == "-":
                right = Mul(args=[Number(value=-1), right])
            args.append(right)
        if len(args) == 1:
            return args[0]
        return Add(args=args)

    def parse_vector_op(self) -> Expr:
        left = self.parse_mul()
        while self.peek_type() == _T_OP and self.peek_val() in (".", "~"):
            op = self.consume()[1]
            right = self.parse_mul()
            if op == ".":
                left = DotProduct(left=left, right=right)
            elif op == "~":
                left = CrossProduct(left=left, right=right)
        return left

    def parse_mul(self) -> Expr:
        left = self.parse_pow()
        factors: List[Expr] = [left]
        while True:
            next_type = self.peek_type()
            if next_type == _T_OP and self.peek_val() in ("*", "/"):
                op = self.consume()[1]
                right = self.parse_pow()
                if op == "/":
                    right = Pow(base=right, exponent=Number(value=-1))
                factors.append(right)
            elif next_type in (_T_ID, _T_INT, _T_FLOAT, _T_LPAREN, _T_LBRACKET, _T_LBRACE, _T_SPECIAL, _T_QUOTE):
                if next_type == _T_ID and self.peek_val() in ("cdot", "times"):
                    break
                right = self.parse_pow()
                prev_factor = factors.pop()
                factors.append(Juxtaposition(left=prev_factor, right=right))
            else:
                break
        if len(factors) == 1:
            return factors[0]
        return Mul(args=factors)

    def parse_pow(self) -> Expr:
            # The precedence level a product's factors sit at.  UNARY, not the
            # script level: a leading minus binds LOOSER than `^`, so `-x^2` is
            # `-(x^2)` and not `(-x)^2`.  Read the other way round — which is
            # what this did until 2026-08-08 — `e^{-x^2}` reached the CAS as
            # `%e^(-x)^2`, the Gaussian with its sign flipped, silently:
            #   -x^2                      → x^2
            #   e^{-x^2}      simplify    → %e^x^2
            #   \int_{-\infty}^{\infty} e^{-x^2} dx → a divergent integral
            # Gaussians, Boltzmann factors `e^{-E/(k_B T)}` and every other
            # `-<base>^<exp>` were affected.  It stayed hidden because the
            # operations that pass the CANONICAL TEXT through (the transform
            # family, which is where the Gaussian scenarios live) never build
            # this AST — Maxima's own parser reads `-x^2` correctly — so the
            # published examples are right and only the IR path was wrong.
            return self.parse_unary()

    def parse_script(self) -> Expr:
        base = self.parse_primary()
        while True:
            if self.peek_type() == _T_OP and self.peek_val() == "^":
                self.consume()
                # `^` is RIGHT-associative, as in both target CAS languages and
                # in ordinary notation: a^b^c means a^(b^c).  Reading it
                # left-associatively made the IR disagree with the input
                # (2^3^2 became 64, not 512) — which stayed hidden only because
                # the code generator used to re-emit `a^b^c` unparenthesised, so
                # the CAS re-read it right-associatively and got the right answer
                # from a wrong AST.  Parenthesising a Pow base (required: `k/r^2`
                # must emit `k*(r^2)^-1`, never `k*r^2^-1` = k*sqrt(r)) removed
                # that accidental cover, so the associativity is fixed here.
                # Subscripts stay LEFT-associative: H_0_hat is Decorated(H_0).
                base = Superscript(base=base, script=self._parse_exponent())
            elif self.peek_type() == _T_SUB:
                self.consume()
                _prev_nab = getattr(self, "_no_attached_bracket", False)
                self._no_attached_bracket = True
                try:
                    sub = self._parse_signed_primary()
                finally:
                    self._no_attached_bracket = _prev_nab
                # If the subscript is a known decoration name, produce Decorated
                # rather than a plain Subscript.  E.g.:
                #   H _ hat  → Decorated(Symbol("H"), "hat")
                #   H_0 _ hat → Decorated(Subscript(H, 0), "hat")
                base = subscript_name_node(base, sub)
            else:
                break
        # The name chain is complete, so a bracket GLUED to it is that whole
        # name's subscript: `c_hat[n, sigma](t)`, `omega_0[i]`.  This is the
        # spelling the Maxima generator itself writes for `\hat c_{n\sigma}(t)`,
        # and re-reading it used to attach the bracket to the LAST underscore
        # piece — `d_hat[i](t)` parsed as "d subscripted by (hat subscripted by
        # [i](t))", which the resolver then tried to NAME as one algebraic index
        # and died with `generator.unsupported_ir_node` (measured 2026-08-21,
        # substituting a mode expansion `\psi = \sum_n \varphi_n c_n`).  The
        # round trip of our own output is an invariant, and it was measured
        # through Maxima's `tex1` only — never back through this parser.
        if isinstance(base, (Decorated, Subscript)):
            # `allow_chain` ONLY here: this is a NAME chain built from `_`
            # pieces, so a bracket glued to it is that name's subscript.  The
            # `parse_primary` call site must stay Symbol-only — a bracket after
            # ANY subscripted node includes `\sum_{p \sigma''}\left[…\right]`,
            # whose bracket is the summand, not a subscript (measured 2026-08-21:
            # the Hartree–Fock self-energy stopped registering at all).
            attached = self._parse_attached_subscript(base, allow_chain=True)
            if attached is not None:
                return attached
        return base

    def _spaced_call_suppressed(self) -> bool:
        r"""Is a SPACE-separated ``(`` here the end of the operand being read?

        A space before the parenthesis means multiplication (`f (x)` is `f·x`),
        and whether a given one is a product or a call is the RESOLVER's
        question — it needs to know whether the callee is a function.  But
        **where the operand ENDS is grammar**, and in an exponent the space
        answers it: `a^2 (b)` is `a^2 · b`, never `a^{2b}`.  Read the other way
        round — which is what this did until 2026-09-25 — the exponent
        SWALLOWED the following factor::

            a^2 (b)          →  a^(2 * b)
            a (b)^2 (c)^3    →  (a * b)^(2 * c)^3

        A researcher who means the product as the exponent writes the braces,
        and a brace group is parsed through `parse_expr`, which clears this.
        """
        return bool(getattr(self, "_no_spaced_call", False)
                    and self.peek_ws_before())

    def _parse_exponent(self) -> Expr:
        """Parse the right-hand side of ``^``, itself right-associative.

        Only the ``^`` chain recurses here — a following subscript is left to
        ``parse_script``, so ``a^b_c`` keeps its existing reading.
        """
        _prev = getattr(self, "_no_spaced_call", False)
        self._no_spaced_call = True
        try:
            base = self._parse_signed_primary()
        finally:
            self._no_spaced_call = _prev
        if self.peek_type() == _T_OP and self.peek_val() == "^":
            self.consume()
            return Superscript(base=base, script=self._parse_exponent())
        return base

    def _parse_signed_primary(self) -> Expr:
        r"""A leading sign applied to a PRIMARY — what a subscript and an
        exponent each want as their operand.

        Neither may descend into `parse_script`, and both must still admit a
        sign.  Measured 2026-08-08 while moving the unary minus below `^`: a
        subscript that reached the script level swallowed the exponent
        (`x_1^2` → `x_(1^2)`, rejected outright by IndexedSymbol), and an
        exponent that did swallowed the subscript (`a^b_c` → `a^(b_c)`).
        Dropping the sign instead broke every negative index there is —
        `c'_{-k}`, `\int_{-\infty}^{\infty}`, the einstein summation tests.
        This is exactly the density `parse_unary` had before the move.
        """
        if self.peek_type() == _T_OP and self.peek_val() == "-":
            self.consume()
            operand = self._parse_signed_primary()
            if isinstance(operand, Number):
                return Number(value=-operand.value)
            return Mul(args=[Number(value=-1), operand])
        if self.peek_type() == _T_OP and self.peek_val() == "+":
            self.consume()
        return self.parse_primary()

    def parse_unary(self) -> Expr:
        """A leading sign, applied to a whole POWER — `-x^2` is `-(x^2)`.

        `-2^2` is therefore `-4`, which is the ordinary reading of the notation
        (and Maxima's own).  A bare `-2` still folds into the Number, because
        `parse_script` hands back the plain Number when no `^` follows.
        """
        if self.peek_type() == _T_OP and self.peek_val() == "-":
            self.consume()
            operand = self.parse_unary()
            if isinstance(operand, Number):
                return Number(value=-operand.value)
            return Mul(args=[Number(value=-1), operand])
        if self.peek_type() == _T_OP and self.peek_val() == "+":
            self.consume()
        return self.parse_script()

    def parse_item(self) -> Expr:
        r"""Parse ONE comma-separated item of a bracket group, allowing `=`.

        A sum's lower bound is written inside the subscript — ``\sum_{i=1}^{N}``
        — and until 2026-08-08 the lexer had no `=` token at all, so a
        non-top-level `=` was silently skipped by the tokenizer and `{i=1}`
        arrived as `Juxtaposition(i, 1)`.  The bound was gone before any
        resolver could see it, which is why `\sum_{i=1}^{N}` reached Maxima as
        the two-argument `sum(x(i), i)` that Maxima rejects (open_items B-10).

        Top-level relations are still split at the STRING level in `parse()`
        (`find_top_level_relation`), which also handles `<`, `>`, `<=`, `>=`;
        this covers only the bracketed case, where that scan does not look.
        """
        node = self.parse_expr()
        if self.peek_type() == _T_EQ:
            self.consume()
            return Equation(lhs=node, rhs=self.parse_expr(), op="=")
        return node

    def parse_primary(self) -> Expr:
        node = self._parse_base_primary()

        from notebook.models.ir import PrimeExpr
        while self.peek_type() == _T_QUOTE:
            self.consume()
            node = PrimeExpr(expr=node)

        attached = self._parse_attached_subscript(node)
        if attached is not None:
            return attached

        if self.peek_type() == _T_LPAREN and not self._spaced_call_suppressed():
            node = self.parse_call_on_node(node, spaced=self.peek_ws_before())

        # Postfix factorial: N! -> factorial(N).  Maps onto the same
        # FunctionCall("factorial", ...) form the SemanticResolver already
        # recognizes for the prefix call form factorial(N), so no separate
        # IR node or resolver/generator case is needed.  Loop to also accept
        # ((N-1)!)! etc.
        while self.peek_type() == _T_BANG:
            self.consume()
            node = FunctionCall(func=Symbol(name="factorial"), args=[node])

        return node

    def _parse_attached_subscript(self, node: Expr,
                                  allow_chain: bool = False) -> "Expr | None":
        r"""`f[i](x)`, `D[i,j](x)`, `x[1]` — a bracket GLUED to a name is a
        subscript, spelled the way the backends spell it.  Returns None when
        this is not that shape, and then nothing above changes.

        This is the READ side of what the Maxima generator writes: an applied
        indexed quantity is emitted as Maxima's own subscripted function
        (`D[i,j](x)`), and a declared vector/matrix/tensor as `V[i]`.  That
        generator's comment says "the round trip closes" — measured against
        Maxima's `tex1`, which does give back `D_{i,j}(x)`.  It was never
        measured against THIS parser, and here the round trip did not close at
        all (2026-08-16):

            f[i](x)     → f * [i](x)                       a product with a LIST
            A[i, j](x)  → A * ((i)*(j) - (j)*(i))(x)       a COMMUTATOR
            x[1]        → x * [1]

        all three silently — so `simplify` on `\int \sum_i f_i(x) dx` came back
        as `f*'sum('integrate([i](x),x),i,nbSumLo1,nbSumHi1)`, a success.  The
        flat two-element `[A, B]` really is a commutator in this notation
        (`_square_bracket_role`), which is why only a bracket with NO space
        before it and a plain name in front of it is taken as a subscript:
        `A [B, C]` and a bare `[A, B]` are untouched.

        The node handed back is the SAME shape the `_{…}` spelling produces —
        one element becomes the element, several become a curly group, and a
        following argument list goes INSIDE the subscript exactly as LaTeX's
        `D_{i,j}(x)` puts it there.  Sameness is the point: whether a subscript
        is an index or part of a name, how many indices a group holds, and how
        an applied form is rebuilt are all decided downstream (subscript_policy,
        `_parse_indices_from_symbol`), and this spelling must not get a second
        set of answers.
        """
        if getattr(self, "_no_attached_bracket", False):
            # Parsing a SUBSCRIPT's operand.  In `c_hat[n, sigma](t)` the
            # bracket belongs to the whole name `c_hat`, not to `hat` — see
            # `parse_script`, which re-offers it once the name chain is done.
            return None
        if not (isinstance(node, Symbol)
                or (allow_chain and isinstance(node, (Decorated, Subscript))
                    and not _binds_a_range(node))):
            return None
        if self.peek_type() != _T_LBRACKET or self.peek_ws_before():
            return None
        start = self.pos
        self.consume()
        args: List[Expr] = []
        if self.peek_type() != _T_RBRACKET:
            args.append(self.parse_item())
            while self.peek_type() == _T_COMMA:
                self.consume()
                args.append(self.parse_item())
        self.expect(_T_RBRACKET)
        if not args:
            self.pos = start        # `f[]` — the EMPTY list is still a list, and
            return None             # declining after consuming would drop it

        sub: Expr = args[0] if len(args) == 1 else BracketExpr(args=args, kind="curly")
        if self.peek_type() == _T_LPAREN and not self.peek_ws_before():
            sub = self.parse_call_on_node(sub, spaced=False)
        return Subscript(base=node, sub=sub, bracketed=True)

    def _parse_base_primary(self) -> Expr:
        tok = self.peek()
        if tok is None:
            raise SyntaxError("Unexpected end of expression")

        typ, val = tok

        if typ == _T_LPAREN:
            self.consume()
            node = self.parse_expr()
            self.expect(_T_RPAREN)
            return node

        if typ == _T_LBRACKET:
            self.consume()
            # Elements separated by ',' within a row; ';' starts a new row.  A
            # ';' anywhere makes this a MATRIX ("[2, 1; 1, 2]" — the agreed
            # convention: ';' → matrix, no ';' → commutator/list).  Emitted as
            # matrix(...) of per-row square BracketExpr; the SemanticResolver
            # keeps those rows as lists (not commutators) and the CodeGenerator
            # renders the backend matrix literal.
            rows: List[List[Expr]] = []
            row: List[Expr] = []
            if self.peek_type() != _T_RBRACKET:
                row.append(self.parse_item())
                while self.peek_type() in (_T_COMMA, _T_SEMI):
                    if self.consume()[0] == _T_SEMI:
                        rows.append(row)
                        row = []
                    row.append(self.parse_item())
            self.expect(_T_RBRACKET)
            if rows:                       # at least one ';' → matrix
                rows.append(row)
                return FunctionCall(
                    func=Symbol(name="matrix"),
                    args=[BracketExpr(args=r, kind="square") for r in rows],
                )
            return BracketExpr(args=row, kind="square")

        if typ == _T_LBRACE:
            self.consume()
            args = []
            if self.peek_type() != _T_RBRACE:
                args.append(self.parse_item())
                while self.peek_type() == _T_COMMA:
                    self.consume()
                    args.append(self.parse_item())
            self.expect(_T_RBRACE)
            return BracketExpr(args=args, kind="curly")

        if typ == _T_QUOTE:
            _nxt = self.toks[self.pos + 1] if len(self.toks) > self.pos + 1 else None
            if _nxt is not None and _nxt[1] == "diff":
                return self.parse_uneval_diff()
            if _nxt is not None and _nxt[0] == _T_ID:
                # A LEADING quote is the CAS noun form — "this call, unevaluated"
                # — and it is what a CAS hands back for anything it would not or
                # could not evaluate.  Only `'diff` was read; every other noun
                # (`'sum`, `'integrate`, `'limit`) failed here, and the failure
                # took the WHOLE expression down to one Symbol.  That mattered
                # as soon as such a result was referenced back into a formula:
                # `E1 - E2` re-registers the spliced text, so `(\sum_i x_i) -
                # ('sum(x(i),i,nbSumLo1,nbSumHi1))` stopped being parseable and
                # the left half never reached the CAS as a sum — returned as a
                # SUCCESS (open_items B-10 ②-e, measured 2026-08-08).
                #
                # The mark is kept ON THE NAME rather than dropped, because the
                # generator has to write back what the CAS sent: dropping it
                # would re-evaluate, and `'integrate(f,x)` unquoted is a
                # different request.  It also keeps the noun out of the Σ prefix
                # reader — `'sum(…)` is already a complete four-argument call,
                # not a prefix waiting for a body.
                self.consume()
                return _as_noun(self.parse_primary())
            raise SyntaxError("Unexpected quote mark")

        if typ == _T_FLOAT:
            self.consume()
            return Number(value=float(val))
        if typ == _T_INT:
            self.consume()
            return Number(value=int(val))

        if typ == _T_SPECIAL:
            self.consume()
            return Symbol(name=val)

        if typ == _T_ID:
            self.consume()
            if self.peek_type() == _T_LPAREN and not self._spaced_call_suppressed():
                return self.parse_call(val, spaced=self.peek_ws_before())
            return Symbol(name=val)

        raise SyntaxError(f"Unexpected token {val!r}")

    def parse_call(self, func_name: str, spaced: bool = False) -> Expr:
        self.expect(_T_LPAREN)
        args: List[Expr] = []
        if self.peek_type() != _T_RPAREN:
            args.append(self.parse_item())
            while self.peek_type() == _T_COMMA:
                self.consume()
                args.append(self.parse_item())
        self.expect(_T_RPAREN)
        return FunctionCall(func=Symbol(name=func_name), args=args, spaced=spaced)

    def parse_call_on_node(self, func_node: Expr, spaced: bool = False) -> Expr:
        self.expect(_T_LPAREN)
        args: List[Expr] = []
        if self.peek_type() != _T_RPAREN:
            args.append(self.parse_item())
            while self.peek_type() == _T_COMMA:
                self.consume()
                args.append(self.parse_item())
        self.expect(_T_RPAREN)
        return FunctionCall(func=func_node, args=args, spaced=spaced)

    def parse_uneval_diff(self) -> Expr:
        self.consume()
        name_tok = self.peek()
        if name_tok is None or name_tok[1] != "diff":
            raise SyntaxError("Expected 'diff'")
        self.consume()
        self.expect(_T_LPAREN)

        expr_node = self.parse_expr()
        var_orders: List[Tuple[Symbol, int]] = []
        while self.peek_type() == _T_COMMA:
            self.consume()
            var_node = self.parse_expr()
            if not isinstance(var_node, Symbol):
                raise SyntaxError("'diff variable must be a symbol")
            self.consume()
            order_tok = self.expect(_T_INT)
            var_orders.append((var_node, int(order_tok[1])))

        self.expect(_T_RPAREN)
        return Derivative(expr=expr_node, vars=var_orders)

class SharedParser:
    """Shared Frontend Parser.
    
    Parses handwritten/LaTeX OCR canonical math strings into standard Surface AST.
    Guarantees pure syntax structure without algebraic semantics.
    """
    def parse(self, canonical: str) -> Optional[Expr]:
        canonical = canonical.strip()
        if not canonical:
            return None

        # Fail loudly on notation we cannot represent (silently dropping a
        # character changes the mathematical meaning — |A| must not become A).
        for ch, key in _UNSUPPORTED_CHARS.items():
            if ch in canonical:
                raise UnknownNotationError(key, token=ch)

        # Normalise ^* to ^{\ast} to avoid tokenization errors with the binary * operator
        canonical = canonical.replace("^*", "^{\\ast}")

        # Assignment/Definition check (LHS := RHS)
        if ":=" in canonical:
            parts = canonical.split(":=", 1)
            lhs = self.parse(parts[0])
            rhs = self.parse(parts[1])
            if lhs is not None and rhs is not None:
                return Assignment(lhs=lhs, rhs=rhs)

        # Equation/relation: lhs <op> rhs, op in {"=", "<=", ">=", "<", ">"}
        # (excluding := and ==)
        rel = find_top_level_relation(canonical)
        if rel is not None:
            rel_pos, rel_op = rel
            lhs_str = canonical[:rel_pos]
            rhs_str = canonical[rel_pos + len(rel_op):]
            # Chained relation ("0 <= x <= 100") — reject loudly rather than
            # silently nesting an Equation inside another Equation's side.
            if (find_top_level_relation(lhs_str) is not None
                    or find_top_level_relation(rhs_str) is not None):
                raise UnknownNotationError("parser.unsupported_notation.chained_relation")
            lhs = self.parse(lhs_str)
            rhs = self.parse(rhs_str)
            if lhs is not None and rhs is not None:
                return Equation(lhs=lhs, rhs=rhs, op=rel_op)
        elif "<" in canonical or ">" in canonical:
            # No top-level relation, yet a stray '<'/'>' remains (e.g. nested
            # inside a function call or bracket group) — the tokenizer has no
            # token for these characters and would otherwise silently drop
            # them (the same failure class |A| used to hit).
            raise UnknownNotationError("parser.unsupported_notation.relational_char")

        tokens, ws_before = _tokenize(canonical)
        if not tokens:
            return Symbol(name=canonical)

        try:
            parser = _Parser(tokens, ws_before)
            result = parser.parse_expr()
            if parser.at_end():
                return result
        except SyntaxError:
            pass

        return Symbol(name=canonical)
