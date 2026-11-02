"""LaTeX → CAS canonical expression converter — backend-agnostic base class.

Structural rules (fractions, square roots, superscripts, operators, …) are
implemented here.  Maxima- and Julia-specific constant/function name maps live
in the concrete subclasses:
  MaximaLatexConverter  (notebook.runtime.maxima_latex_converter)
  JuliaLatexConverter   (notebook.runtime.julia_latex_converter)

A Maxima-only backward-compatibility shim ``latex_to_canonical()`` lives in
``maxima_latex_converter`` (not here — this module must not import a
backend).  New code should call ``backend_handler.latex_to_canonical(latex)``.
"""
from __future__ import annotations

import re

from notebook.runtime.function_application import apply_all
from notebook.runtime.integral_form import expand_multi_int_commands
from notebook.runtime.matrix_environment import rewrite as _matrix_environments
from notebook.runtime.number_words import number_literal
def _has_top_level_comma(text: str) -> bool:
    """括弧の外にカンマがあるか。``psi(x,t)`` の内側のカンマは数えない。"""
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            return True
    return False


from notebook.runtime.time_order import rewrite as _time_order
from notebook.runtime.script_label import (
    rewrite as _label_scripts, split_label as _split_script_label)
from notebook.runtime.latex_typography import (
    braced_group, normalize_command_aliases, strip_typography,
)


#: TeX Greek command → plain identifier.  CAS-NEUTRAL: every CAS this project
#: targets spells these letters the same way, which is why both concrete
#: converters used to hold byte-identical copies of this table.  Only the
#: CONSTANTS differ (``\pi`` → ``%pi``/``pi``, ``\infty`` → ``inf``/``Inf``,
#: ``\mathrm{i}`` → ``%i``/``im``), so those stay in each subclass's
#: ``_GREEK_MAP`` and are deliberately absent here.
#:
#: Shared layers that need TeX→name (e.g. the NL declaration parser in
#: ``backend_handler``) must use THIS table.  Reaching into
#: ``MaximaLatexConverter._GREEK_MAP`` made a shared module depend on one CAS
#: and made that CAS's canonicalization the default for every session
#: (open_items AUDIT-SHARED-CAS-LEAK).
TEX_GREEK_TO_NAME: dict[str, str] = {
    r"\alpha": "alpha", r"\beta": "beta", r"\gamma": "gamma",
    r"\delta": "delta", r"\epsilon": "epsilon", r"\varepsilon": "epsilon",
    r"\zeta": "zeta", r"\eta": "eta", r"\theta": "theta",
    r"\iota": "iota", r"\kappa": "kappa", r"\lambda": "lambda",
    r"\mu": "mu", r"\nu": "nu", r"\xi": "xi",
    r"\rho": "rho", r"\sigma": "sigma", r"\tau": "tau",
    r"\upsilon": "upsilon", r"\phi": "phi", r"\varphi": "phi",
    r"\chi": "chi", r"\psi": "psi", r"\omega": "omega",
    r"\Gamma": "Gamma", r"\Delta": "Delta", r"\Theta": "Theta",
    r"\Lambda": "Lambda", r"\Xi": "Xi", r"\Pi": "Pi",
    r"\Sigma": "Sigma", r"\Upsilon": "Upsilon", r"\Phi": "Phi",
    r"\Psi": "Psi", r"\Omega": "Omega",
    r"\hbar": "hbar",
}


def replace_tex_command(s: str, cmd: str, repl: str) -> str:
    r"""Replace the TeX command *cmd* with *repl*, but only where *cmd* ENDS.

    A plain ``str.replace`` matches a command that is merely a PREFIX of a
    longer one.  Measured 2026-08-01: ``\i`` → ``%i`` (Maxima's imaginary
    unit) rewrote ``\int_0^1 x^2 dx`` into ``%int_0^1*x^2*dx``, and since
    ``%int`` is an ordinary symbol to Maxima, the integral silently became a
    PRODUCT that evaluates without error (open_items EQ-EMBEDDED-NOTATION).
    Sorting the table by length only protects the commands that are IN the
    table — ``\int`` is not, and neither is any command a future backend map
    forgets.

    **The OPERATORS were not asking this** until 2026-09-10 (TEX-MATRIX-INPUT):
    ``_operators`` spelled its own ``str.replace``/prefix regexes, so four
    ordinary spellings were rewritten into something else, every one of them
    silently::

        \pmatrix{2 & 1 \\ 1 & 2}   → +atrix2 & 1 1 & 2
        \pmb{x}                    → +bx
        \pmod{3}                   → +od3
        x_1 + \cdots + x_n         → x_1 + . s + x_n

    The last is the one written most often: ``\cdots`` was eaten by ``\cdot``
    (the dot product) and the leftover ``s`` entered the calculation as a symbol.
    One question, two answers — and the ad-hoc one was the wrong answer.
    """
    if cmd and cmd[-1].isalpha():
        return re.sub(re.escape(cmd) + r"(?![A-Za-z])", lambda _m: repl, s)
    return s.replace(cmd, repl)


def normalize_relational_ops(s: str) -> str:
    r"""Rewrite LaTeX relational macros to their ASCII form (``\leq``/``\le``
    -> ``<=``, ``\geq``/``\ge`` -> ``>=``).

    Public (not underscore-prefixed): the inequality NL interceptor
    (``ui/interpret_inequality.py``) calls this directly to normalize raw
    input before handing it to ``find_top_level_relation`` for detection,
    so the same substitution is used for both detection and canonical
    generation.  ``\le``/``\ge`` are substituted after ``\leq``/``\geq`` so
    ``\leq`` is not half-consumed into ``<=q``.
    """
    # `\neq`/`\ne`/`!=` are made ONE spelling (`#`, Maxima's own not-equal) for
    # the same reason as their `\leq` siblings — and because each of the three
    # was silently something else: `a \neq b` lowered to the PRODUCT
    # `a * neq * b`, and `a != b` to `factorial(a) = b`, both reporting success
    # (measured 2026-08-10).  `#` is not representable in the IR, so the parser
    # refuses it with a reason; that refusal is the point.  A `\ne` inside a sum
    # SUBSCRIPT never reaches here — `_subscript_constraint` has already made it
    # the bare marker `ne`, which is why that marker carries no backslash.
    s = re.sub(r"\\neq\b|\\ne\b|!=", "#", s)
    # Every spelling of "approximately equal" becomes ONE marker, for the same
    # reason as its `\leq` siblings and with the same measured failure: `A
    # \approx B + C` reached Maxima as the PRODUCT `A * approx * B + C` and
    # `simplify` returned `A*B*approx+C` as a SUCCESS (measured 2026-08-21).  The
    # marker is the character itself — no later rule in this pass touches it, so
    # it cannot be flipped back and forth the way a `\le` spelling was.  Unlike
    # `#`, it is NOT added to the parser's unsupported-character list: an
    # approximation must stay WRITABLE (it is the step a derivation turns on),
    # so it is read as a relation here and refused where it would be COMPUTED.
    s = re.sub(r"\\approx\b|\\simeq\b|\\cong\b|\\thickapprox\b", "≈", s)
    s = s.replace(r"\leq", "<=").replace(r"\geq", ">=")
    s = re.sub(r"\\le\b", "<=", s)
    s = re.sub(r"\\ge\b", ">=", s)
    return s


class LatexConverter:
    """Backend-agnostic LaTeX → CAS canonical converter.

    Subclasses override :attr:`_TRIG_MAP`, :attr:`_GREEK_MAP`,
    :attr:`_KNOWN_FUNCTIONS`, and :attr:`_FN_ALT` as class-level attributes
    to produce backend-specific output.
    """

    _TRIG_MAP: dict[str, str] = {}
    _GREEK_MAP: dict[str, str] = {}
    _KNOWN_FUNCTIONS: frozenset[str] = frozenset({"sqrt"})
    _FN_ALT: str = "sqrt"
    #: What this backend calls the natural logarithm — the one function whose
    #: SUBSCRIPT has a reading (``\log_b X`` is a change of base).  Maxima and
    #: Julia both spell it `log`; a backend that does not overrides this.
    _LOG_FN: str = "log"

    def to_canonical(self, latex: str, max_passes: int = 6) -> str:
        r"""Convert *latex* to CAS canonical form, applying rules until stable.

        The matrix ENVIRONMENTS are read first, before anything else touches the
        string.  `\\` (the row separator) is otherwise gone by the time any rule
        could use it and `&` (the column separator) survives all the way to the
        parser, where it is reported as an unreadable character — both are
        symptoms of the researcher's ROW STRUCTURE having already been lost
        (measured 2026-09-10, TEX-MATRIX-INPUT).  It runs before
        `_strip_delimiters` too, because that removes `\[` and `\]` wherever
        they stand and a row's spacing hint is written `\\[2pt]`.
        Why the environments do not all mean the same thing — `vmatrix` is a
        DETERMINANT — is in `notebook/runtime/matrix_environment.py`.

        **文全体が語で書かれた数なら、それは数である**（`number_literal`——
        なぜ「全体が」だけを訊くのか、どの綴りが黙って壊れていたのかはそちら）。
        値の欄はどれもこの扉を通るので、ここ1箇所で足りる。
        """
        _literal = number_literal(latex)
        if _literal is not None and _literal != latex.strip():
            return _literal
        s = self._strip_delimiters(_matrix_environments(latex.strip()))
        for _ in range(max_passes):
            prev = s
            s = self._apply_rules(s)
            if s == prev:
                break
        return s.strip()

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _strip_delimiters(self, s: str) -> str:
        for delim in (r"\[", r"\]", r"\(", r"\)", "$$", "$"):
            s = s.replace(delim, "")
        return s.strip()

    #: Operators whose two scripts are LIMITS, and may therefore be written in
    #: either order.  A plain symbol is excluded on purpose: this rule reorders
    #: text, and it should only do so where both orders are the same statement.
    _LIMIT_OPS = ("sum", "prod", "coprod", "int", "iint", "iiint", "oint",
                  "bigcup", "bigcap", "bigoplus", "bigotimes", "bigwedge")

    def _script_order(self, s: str) -> str:
        r"""``\int^{1}_{0}`` → ``\int_{0}^{1}`` — lower limit first, always.

        Both orders are ordinary LaTeX and mean the same integral, but only one
        of them was read.  Measured 2026-08-10, the superscript-first spelling
        silently LOST ITS BOUNDS and answered as though none had been written:

            \int^{1}_{0} x dx   → integrate(x, x)   → simplify gave `x^2/2`
            \int_{0}^{1} x dx   → the definite integral, whose value is 1/2

        The sum has the same shape (``\sum^{3}_{i=1} i^2`` reached Maxima with
        no range at all).  Making the two orders ONE spelling here, at the
        converter's entrance, is the same move as ``_subscript_separator`` and
        ``_conjugate_mark``: downstream keeps a single spelling to work on
        instead of two to keep in step.
        """
        def _script(text: str, i: int) -> "tuple[str, int] | None":
            """Read one ``^…``/``_…`` script at *i*, braced or bare."""
            if i >= len(text) or text[i] not in "^_":
                return None
            k = i + 1
            group = braced_group(text, k)
            if group is not None:
                return text[i] + "{" + group[0] + "}", group[1]
            m = re.match(r"\\?[A-Za-z0-9]+|.", text[k:])   # bare: \pi, 3, N
            if not m:
                return None
            return text[i] + "{" + m.group(0) + "}", k + m.end()

        for op in self._LIMIT_OPS:
            needle = "\\" + op
            i = 0
            while True:
                j = s.find(needle, i)
                if j < 0:
                    break
                k = j + len(needle)
                if k < len(s) and s[k].isalpha():        # \integer is not \int
                    i = k
                    continue
                first = _script(s, k)
                second = _script(s, first[1]) if first else None
                if not (first and second) or first[0][0] != "^":
                    i = k
                    continue
                s = f"{s[:k]}{second[0]}{first[0]}{s[second[1]:]}"
                i = k
        return s

    def _apply_rules(self, s: str) -> str:
        s = self._typography(s)
        s = self._script_order(s)
        # `\iint`/`\iiint` → repeated `\int`: the abbreviation is made one
        # spelling here so the general route reads it, and route ① (which works
        # on the raw text, before any of this) calls the SAME function.
        s = expand_multi_int_commands(s)
        s = self._conjugate_mark(s)
        s = self._dagger_mark(s)
        # 上付きの摂動次数ラベルを名前へ畳む（順序非依存）。
        # 記法だけで決まる部分なので、宣言を見る層より手前でよい
        # ——`_conjugate_mark` / `_dagger_mark` と同じ入口の移動。
        s = _label_scripts(s)
        s = self._separate_adjacent_commands(s)
        # 極限の添字は `_subscript_separator` が空白をコンマにする前に開く。
        s = self._limit_direction(s)
        s = self._subscript_constraint(s)
        # 時間順序は期待値の**手前**——`\langle T[\ldots]\rangle` は
        # `expectation(time_order(...))` の順に畳まれる。
        s = _time_order(s)
        # 期待値の括弧。`_subscript_constraint` の**後**に置く——`\sum_{\langle
        # i,j \rangle}`（最近接対）はそちらが先に食べており、あれは平均ではない。
        s = self._expectation_bracket(s)
        s = self._subscript_separator(s)
        s = self._frac(s)
        s = self._sqrt(s)
        s = self._vector(s)
        s = self._superscript(s)
        s = self._trig_and_functions(s)
        s = self._greek_and_constants(s)
        # _decorations runs AFTER _greek_and_constants so that e.g.
        # \tilde{\omega} is first converted to \tilde{omega} and then
        # processed by _decorations as "omega" → "omega_tilde".
        s = self._decorations(s)
        s = self._operators(s)
        s = self._left_right(s)
        s = self._function_application(s)
        s = self._remove_remaining_braces(s)
        s = self._implicit_multiplication(s)
        s = self._whitespace(s)
        return s

    # ------------------------------------------------------------------
    # Structural rules (backend-agnostic)
    # ------------------------------------------------------------------

    def _typography(self, s: str) -> str:
        """Delegate to :mod:`notebook.runtime.latex_typography`.

        The backend's own tables are handed over so that a wrapper which NAMES
        something is left for that table to read rather than unwrapped here.
        Two kinds of name qualify: a CONSTANT (``\\mathrm{i}`` is the imaginary
        unit) and an upright OPERATOR (``\\text{tr}`` is the trace).  Only the
        constants were passed, so every wrapped operator name was unwrapped to a
        bare identifier and the function table — which spells it
        ``\\text{tr}`` — never matched it (measured 2026-09-06: ``\\text{tr}(M)``
        registered as the undefined ``tr(M)`` and nothing downstream computed).
        A key that is not a ``\\wrapper{...}`` spelling can never match the
        keep test, so merging the whole function table costs nothing.
        """
        return strip_typography(s, {**self._GREEK_MAP, **self._TRIG_MAP})

    #: Every spelling of the COMPLEX-CONJUGATE superscript.  A TeX editor emits
    #: the braced ones, and so does this system's own display — which is the
    #: point: ``\psi^{\ast}(x)`` is what a result is written down as, so it must
    #: type back in.  Measured 2026-08-09, it did not:
    #:
    #:     \psi^*(x)        → psi^*(x)          ← the one spelling that worked
    #:     \psi^{*}(x)      → psi^(*)*(x)       ← braces gone, star read as a factor
    #:     \psi^{\ast}(x)   → psi^(\ast)*(x)
    #:     \psi^\ast(x)     → psi^\ast(x)       ← a backslash reaching the CAS
    #:
    #: Only a LONE star: ``^{*2}`` is an exponent, and ``^\dagger`` is the
    #: ADJOINT (a different IR node), so neither is touched.
    _CONJUGATE_MARK_RE = re.compile(
        r"\^\s*(?:\{\s*(?:\*|\\ast|\\star)\s*\}|\\ast\b|\\star\b|\*)")

    #: The ADJOINT superscript, in every spelling.  Same move as the conjugate
    #: mark above and for the same measured reason: only ``^\dagger`` (bare)
    #: survived the pass, while the braced form — **the spelling this system's
    #: own display produces** — came out as a PRODUCT:
    #:
    #:     \psi^\dagger(x)     → psi^\dagger(x)      ← the one spelling that worked
    #:     \psi^{\dagger}(x)   → psi^(\dagger)*(x)   ← adjoint and x, multiplied
    #:
    #: (measured 2026-08-20).  ``\dag`` is the same command shortened, and the
    #: bare Unicode ``†`` is the one non-ASCII character the input layer already
    #: admits.
    _DAGGER_MARK_RE = re.compile(
        r"\^\s*(?:\{\s*(?:\\dagger|\\dag|†)\s*\}|\\dagger\b|\\dag\b|†)")

    def _dagger_mark(self, s: str) -> str:
        r"""Make every adjoint superscript one spelling: ``^\dagger``.

        What that spelling MEANS is not decided here — a declared matrix gets
        the adjoint, anything else a decorated name — because the reading
        depends on the declaration and canonical must not freeze it (CLAUDE.md,
        "What `Expression.canonical` Is").  This pass only makes the four
        spellings one, so the resolver has a single shape to read.
        """
        return self._DAGGER_MARK_RE.sub(r"^\\dagger", s)

    #: 期待値の開き括弧。`\left` 付きも同じもの。
    _LANGLE_RE = re.compile(r"(?:\\left\s*)?\\langle")

    def _expectation_bracket(self, s: str) -> str:
        r"""``\langle X \rangle`` → ``expectation(X)``、``\langle X \rangle_0`` →
        ``expectation(X, 0)``。

        括弧が読まれていなかったので、中身と一緒に ``langle``・``rangle`` という
        **未定義の記号2つの積**として CAS に渡っていた（実測 2026-08-20、
        ``simplify`` は成功を返す）。添字付きの ``\langle A \rangle_0`` では
        ``rangle`` が**添字付き関数** ``rangle(0)`` になっていた。

        落とし先は R-9 の bracket functional と**同じ** ``expectation(...)`` —— 
        ``E[X]`` が既にそこへ落ちており、実測で行き止まりにならない
        （``expectation(X)+expectation(Y)`` も ``diff(expectation(X)*t, t)`` も通る）。
        添字は**第2引数**にする: どの状態で平均を取ったかを CAS 側で置換・比較できる。

        **トップレベルにカンマがあるものは対象外**。``\langle a, b \rangle`` は内積
        または対の記法で、平均ではない（最近接対 ``\sum_{\langle i,j \rangle}`` は
        この手前の ``_subscript_constraint`` が既に食べている）。
        """
        out: list[str] = []
        i = 0
        while i < len(s):
            m = self._LANGLE_RE.match(s, i)
            if not m:
                out.append(s[i])
                i += 1
                continue
            j = m.end()
            depth = 0
            body_end = None
            k = j
            while k < len(s):
                if s[k] in "([{":
                    depth += 1
                elif s[k] in ")]}":
                    depth -= 1
                elif depth == 0:
                    rm = re.compile(r"(?:\\right\s*)?\\rangle").match(s, k)
                    if rm:
                        body_end = k
                        k = rm.end()
                        break
                    if self._LANGLE_RE.match(s, k):
                        body_end = None
                        break
                k += 1
            if body_end is None:
                out.append(s[i:m.end()])
                i = m.end()
                continue
            body = s[j:body_end].strip()
            _braket = split_state_braket(body) if "|" in body else None
            if _braket is not None:
                # `\langle \Phi_0 | A | \Phi_0 \rangle` —— 両側が同じ状態。
                # `\langle A \rangle_{\Phi_0}` と同じ量なので、同じ対象に落とす。
                _op, _state = _braket
                out.append(f"expectation({_op}, {_state})")
                i = k
                continue
            if not body or _has_top_level_comma(body) or "|" in body:
                # 対・内積、および状態が2つのブラケットはここの担当ではない。
                out.append(s[i:m.end()])
                i = m.end()
                continue
            state, k = self._read_bracket_state(s, k)
            args = body if state is None else f"{body}, {state}"
            out.append(f"expectation({args})")
            i = k
        return "".join(out)

    @staticmethod
    def _read_bracket_state(s: str, i: int) -> "tuple[str | None, int]":
        r"""``\rangle`` の直後の下付き（``_0``、``_{H_0}``）を読む。"""
        if i >= len(s) or s[i] != "_":
            return None, i
        group = braced_group(s, i + 1)
        if group is not None:
            return group[0].strip(), group[1]
        m = re.match(r"\\?[A-Za-z0-9]+", s[i + 1:])
        if not m:
            return None, i
        return m.group(0), i + 1 + m.end()

    def _conjugate_mark(self, s: str) -> str:
        r"""Make every conjugate superscript one spelling: ``^*``.

        The one `*` in a formula that is not multiplication, settled at the
        entrance so nothing downstream has to know about the other spellings —
        the same move as `_subscript_separator`.  `_resolve_pow` turns ``^*``
        into `Conjugate`; that end already worked.
        """
        return self._CONJUGATE_MARK_RE.sub("^*", s)

    #: A LaTeX command that follows something it is NOT part of — another
    #: command (``\mu\nu``), a name (``m\omega``), a digit, or a closing
    #: bracket (``(a+b)\alpha``).  The backslash is the boundary in every case.
    #:
    #: ``_`` and ``^`` are deliberately absent from the preceding set: there the
    #: command IS part of what precedes it (``x_\alpha``, ``x^\alpha``), and a
    #: space would break the subscript or the exponent.
    #:
    #: The DELIMITER commands are excluded too.  They are syntax, not factors:
    #: ``G_{0}\left(k,\omega\right)`` is a function CALL, and a space before the
    #: ``\left`` turns it into ``G_0 (k, omega)`` — which the space-then-paren
    #: rule reads as multiplication, the very silent-wrong that
    #: `test_a_digit_subscript_reads_back_as_a_call_not_a_product` pins.
    #: A PRIME counts as the end of a name for this purpose (``c_{k'\sigma}``):
    #: it marks a different quantity of the same name, so what follows it is a
    #: new token.  Without it the glued command was left alone and the pair
    #: reached the CAS as the product ``c(k') * sigma`` (measured 2026-08-20).
    #: Maxima's noun-form quote PRECEDES its name (``'diff``) and carries no
    #: backslash, so it never matches here.
    _COMMAND_NEEDS_SPACE_RE = re.compile(
        r"(?<=[A-Za-z0-9')\]}])(?=\\(?!left\b|right\b|[Bb]igg?[lrm]?\b)[A-Za-z])")

    def _separate_adjacent_commands(self, s: str) -> str:
        r"""Put a space in front of a command that is glued to what precedes it.

        Measured 2026-08-04: commands were replaced by their ASCII names with
        nothing inserted, so two symbols MERGED INTO ONE, silently:

            \alpha\beta       → `alphabeta`      (the product αβ, as one symbol)
            \sin\theta        → `sintheta`
            \alpha\beta\gamma → `alpha\betagamma`    (not even consistently)
            m\omega^2         → `momega^2`       ← ordinary physics notation
            x\sin\theta       → `xsin(theta)`
            )\alpha           → `)alpha`

        The merged name is then an undeclared symbol that differentiates to 0
        and ignores substitution, with no error anywhere.  `to_backend_code`
        reads the same input correctly because it PARSES the LaTeX instead of
        rewriting it — the same split that hid the subscript-separator bug
        below.

        A space is the right thing to insert because the machinery to read one
        already exists on both sides of the ambiguity: inside a subscript
        `_subscript_separator` turns it into a comma (two indices), outside
        `_implicit_multiplication` turns it into `*` (a product), and
        `_trig_and_functions` turns "name space arg" into a call.  Nothing new
        has to decide which context it is in — which is why `\sin\theta` starts
        working as well without a rule about trig functions being added.
        """
        return self._COMMAND_NEEDS_SPACE_RE.sub(" ", s)

    #: A CONSTRAINT written among a sum's indices: `\sum_{i<j}`,
    #: `\sum_{i \ne j}`, `\sum_{\langle i,j \rangle}` (nearest neighbours).
    #: Made comma-separated tokens here so all four spellings arrive at the
    #: resolver the same way — the `_subscript_separator` move again.  `\ne` and
    #: `\neq` already reached it separated (a space, then a comma); `<` and the
    #: angle brackets did not, so `\sum_{i<j}` escaped the braces entirely and
    #: came back as the INEQUALITY `sum(i)<j * V(i, j)` (measured 2026-08-09).
    #: Longest alternative first: `le|leq` would match `\leq` as `\le` and leave
    #: a stray `q` behind.
    #: `<=` / `>=` are listed BEFORE the bare `<`/`>`, and matter even though no
    #: researcher types them: `_operators` rewrites `\le` to `<=` further down
    #: the same pass, and this rule runs again on that output.  Without the
    #: two-character alternatives it cut `<=` in half and `\sum_{i \le j}`
    #: stabilised as `{i,\lt,=,j}`.
    _SUBSCRIPT_RELATION_RE = re.compile(
        r"\\(?:leq|geq|neq|le|ge|ne|lt|gt)\b|<=|>=|[<>]")
    #: …and every spelling becomes a BARE WORD, with no backslash.  A marker
    #: that still looks like a LaTeX relation is rewritten again further down
    #: the same pass (`_operators` turns `\le` into `<=`), and since
    #: `to_canonical` re-applies its rules until the text stops changing, the
    #: two rules flipped it back and forth and it never settled.  A bare word is
    #: a Symbol to the parser and invisible to every other rule.
    _RELATION_AS_COMMAND = {
        r"\leq": "le", r"\le": "le", "<=": "le",
        r"\geq": "ge", r"\ge": "ge", ">=": "ge",
        r"\neq": "ne", r"\ne": "ne",
        r"\lt": "lt", "<": "lt", r"\gt": "gt", ">": "gt",
    }

    #: 極限の矢印。綴りは1つに畳む——`\to` / `\rightarrow` / `\longrightarrow` /
    #: Unicode の `→` は同じ矢印で、下流に4つ覚えさせる理由が無い
    #: （`_conjugate_mark` / `_dagger_mark` と同じ入口の移動）。
    _LIMIT_ARROWS = (r"\longrightarrow", r"\rightarrow", "→", r"\to")

    #: 片側極限の印。`0^+` の `^+` はトークナイザが読めない（`+` は上付きの
    #: 中身になれない）ので、**添字の中で語に開く**。開いた語を読むのは
    #: `sum_integral_prefix._read_limit_spec`——Maxima の綴りを選ぶのは生成器で、
    #: ここは「読める形にする」だけである。
    _LIMIT_SIDE_RE = re.compile(r"\^\s*\{?\s*([+-])\s*\}?\s*$")

    def _limit_direction(self, s: str) -> str:
        r"""``\lim_{t' \to t^+}`` → ``\lim_{t' to t plus}``。

        **読めない綴りは、読める綴りに開いてから下流へ渡す。** 直す前は
        `\lim_{t' \to t^+} G(t,t')` の添字がトークナイズできず、
        **右辺まるごとが1個の未定義記号**として登録され `completed` が返っていた
        （GF-NOTATION の最後の1族、実測 2026-09-11）。`\sum_{i \ne j}` の
        `_subscript_constraint` と同じ移動——綴りを入口で1つにし、下流には
        「読める形」だけを見せる。

        **`\liminf` / `\limsup` に明示の除外は要らない。** 変異試験で
        「別の束縛子を弾く」枝を消しても何も落ちなかったので実測したところ、
        直後が `_` かどうかの検査が既に同じ答えを出していた（`\liminf_{n}` の
        `\lim` の次は `i`）。**生き残った変異は冗長の証拠**——固定せず消した。
        """
        out = []
        i = 0
        while True:
            j = s.find(r"\lim", i)
            if j < 0:
                out.append(s[i:]); break
            k = j + len(r"\lim")
            if s[k:k + len(r"\limits")] == r"\limits":
                k += len(r"\limits")
            if k >= len(s) or s[k] != "_":
                out.append(s[i:k]); i = k; continue
            body_start = k + 1
            if body_start >= len(s) or s[body_start] != "{":
                out.append(s[i:k]); i = k; continue
            depth, m = 0, body_start
            while m < len(s):
                if s[m] == "{": depth += 1
                elif s[m] == "}":
                    depth -= 1
                    if depth == 0: break
                m += 1
            if m >= len(s):
                out.append(s[i:]); break
            inner = s[body_start + 1:m]
            for arrow in self._LIMIT_ARROWS:
                inner = inner.replace(arrow, " to ")
            side = self._LIMIT_SIDE_RE.search(inner)
            if side:
                inner = (inner[:side.start()]
                         + (" plus" if side.group(1) == "+" else " minus"))
            out.append(s[i:body_start + 1] + inner)
            i = m
        return "".join(out)

    def _subscript_constraint(self, s: str) -> str:
        r"""Separate a constraint written inside a subscript from the indices.

        The subscript of a Σ is read as an INDEX LIST, so a condition written
        there became an index: `\sum_{i \ne j}` summed over a dummy named `ne`,
        and `\sum_{\langle i,j \rangle}` over one named `langle` — nonsense,
        returned as a success.  Separating the tokens here does not decide what
        a constrained sum MEANS; it makes the constraint visible to the layer
        that has to answer that (`_sum_dummy_indices`, then a refusal with a
        reason at code generation, the same path `\int d^3k` takes).
        """
        out, i = [], 0
        while True:
            j = s.find("_{", i)
            if j < 0:
                out.append(s[i:])
                break
            k, depth = j + 2, 1
            while k < len(s) and depth:
                depth += (s[k] == "{") - (s[k] == "}")
                k += 1
            if depth:                       # unbalanced — leave it alone
                out.append(s[i:])
                break
            inner = s[j + 2:k - 1]
            if self._SUBSCRIPT_RELATION_RE.search(inner) or "\\langle" in inner:
                inner = self._SUBSCRIPT_RELATION_RE.sub(
                    lambda m: "," + self._RELATION_AS_COMMAND.get(m.group(0),
                                                                  m.group(0)) + ",",
                    inner)
                inner = inner.replace("\\langle", "langle,").replace("\\rangle", "")
                # `_apply_rules` runs until the text stops changing, so this has
                # to be IDEMPOTENT.  Without the collapse it fired on its own
                # output every pass and produced `{i,,,,,,\ne,,,,,,j}`.
                # Inside a CONSTRAINED subscript every token is either a
                # marker or an index, so a space is a separator like the comma
                # — the same rule `_subscript_separator` applies, restated here
                # only because the marker it now contains stops that pass from
                # recognising the brace.  `\sum_{\langle i j \rangle}` (the
                # spacing the tight-binding scenario uses) came out as
                # `{langle,i*j}` without it.
                inner = re.sub(r"\s+", ",", inner.strip())
                inner = re.sub(r"\s*,\s*", ",", inner)
                inner = re.sub(r",{2,}", ",", inner).strip().strip(",")
            out.append(s[i:j + 2])
            out.append(inner)
            out.append("}")
            i = k
        return "".join(out)

    #: A subscript whose braces hold two or more INDEX-LIKE tokens separated by
    #: spaces: ``A_{i j}``, ``T_{i j k}``, ``g_{\mu \nu}``.  An index-like token
    #: is a name, a number, or a Greek command — requiring every token to look
    #: like one keeps this away from a brace holding anything else.
    #: An index token may carry PRIMES (``\sigma'``, ``k'``): a prime marks a
    #: different quantity of the same name, and a Green's function's second
    #: spin/momentum label is written that way as a matter of course.  Without
    #: them in this pattern the separator did not fire at all, so
    #: ``\varphi_{m\sigma'}`` fell through to implicit multiplication and
    #: reached the CAS as ``phi(m) * sigma'`` — a PRODUCT — while the unprimed
    #: ``\varphi_{m\sigma}`` was read correctly as two indices (2026-08-20).
    _SPACED_SUBSCRIPT_RE = re.compile(
        r"(?<=_)\{(\\?[A-Za-z]\w*'*|\d+)((?:\s+(?:\\?[A-Za-z]\w*'*|\d+))+)\}")

    def _subscript_separator(self, s: str) -> str:
        r"""Space-separated subscript indices → the comma spelling.

        "Multiple indices must be separated by a comma **or a space**"
        (AUDIT-SUBSCRIPT-SPLIT, CLAUDE.md).  Only the comma half was reaching
        the CAS: measured 2026-08-04, ``A_{i j}(x)`` arrived as
        ``A(i) * j(x)`` and ``A_{i j}`` as ``A(i) * j`` — a tensor component
        silently turned into a product, with no error.  Two passes here were
        undoing it in turn: ``_remove_remaining_braces`` dropped the braces
        (a comma is in its exclusion class, a space was not), and then
        ``_implicit_multiplication`` read the bare ``i j`` as multiplication.

        The 3-layer pipeline was never at fault — handed the original text it
        answers ``A[i, j](x)``.  So rather than protect the space form through
        both passes, the two spellings are made ONE here, at the entrance: the
        comma form is already correct end to end, and downstream then has a
        single spelling to keep working rather than two to keep in step.  That
        is the same lesson this rule has already taught twice (open_items
        AUDIT-INDEXED-CALL: "再掲せず呼ぶ").

        Only inside a subscript.  Elsewhere a space between letters really is
        multiplication (``\frac{a b}{c}``), which must keep working.
        """
        return self._BRACED_SUBSCRIPT_RE.sub(self._separate_subscript_group, s)

    #: 下付きの中括弧ひとかたまり。中身の区切りは下の関数が決める。
    _BRACED_SUBSCRIPT_RE = re.compile(r"(?<=_)\{([^{}]*)\}")

    @classmethod
    def _separate_subscript_group(cls, m: "re.Match[str]") -> str:
        r"""下付きの中身を、**コンマ区切りの区画ごとに**空白で割る。

        規則は「添字はコンマ**または空白**で区切る」だが、実装は
        **全体が空白区切りの場合にしか効いていなかった**。両方が混ざる形——
        ``V_{n\sigma,m\sigma'}`` のように、対になったラベルを並べる標準的な
        書き方——では空白側が読まれず、各区画が**積**になっていた（実測
        2026-08-20、シナリオを書いていて発見）:

            V_{n\sigma}          → V(n, sigma)        ← 正しい
            V_{n\sigma,m\sigma'} → V(n*sigma, m*sigma') ← 積
            \sum_{n\sigma,m\sigma'} → 和の添字まで積

        区画ごとに同じ規則を当てれば、両方の綴りが1つの読みに揃う。
        """
        parts = [p for p in m.group(1).split(",")]
        out: list[str] = []
        for part in parts:
            toks = part.split()
            out.extend(toks if toks else ([part] if part else []))
        return "{" + ",".join(out) + "}" if out else m.group(0)

    def _decorations(self, s: str) -> str:
        r"""Convert LaTeX decoration commands to canonical form.

        Each decoration creates a distinct symbol via a suffix so that
        decorated forms remain distinguishable from their undecorated base:

          \hat{X}       → X_hat      (operator/unit vector notation)
          \widehat{X}   → X_hat
          \tilde{X}     → X_tilde    (renormalised / Fourier-transformed)
          \widetilde{X} → X_tilde
          \bar{X}       → X_bar      (complex conjugate / mean value)
          \overline{X}  → X_bar
          \overbar{X}   → X_bar
          \dot{X}       → X_dot      (first time derivative — NOT the same as X)
          \ddot{X}      → X_ddot     (second time derivative)
          \check{X}     → X_check
          \breve{X}     → X_breve
          \acute{X}     → X_acute
          \grave{X}     → X_grave
          \underline{X} → X          (text emphasis only — no physics meaning)

        The suffix is always appended, regardless of whether the inner
        content is a single symbol or a compound expression.  This preserves
        the physics distinction (e.g. ``\hat{H}`` ≠ ``H``,
        ``\dot{x}`` ≠ ``x``).

        Called AFTER _greek_and_constants so that e.g. \\tilde{\\omega} is
        first converted to \\tilde{omega} and then processed as
        "omega" → "omega_tilde".
        """
        def _apply(cmd: str, suffix: str, s: str) -> str:
            """Replace \\cmd{inner} → inner<suffix>, \\cmd X → X<suffix>."""
            def _replace_brace(m: re.Match) -> str:
                inner = m.group(1).strip()
                return f"{inner}{suffix}"
            s = re.sub(rf"\\{cmd}\{{([^{{}}]*)\}}", _replace_brace, s)
            # Bare form: `\cmd X`.  The argument is the whole IDENTIFIER, not
            # its first character.  A Greek letter is one LaTeX token but a
            # multi-letter name by the time it reaches here (_decorations runs
            # after _greek_and_constants), so a one-character match decorated
            # the first letter and left the rest standing: `\hat\psi` became
            # `p_hatsi` — a symbol nobody wrote, registered without complaint
            # (measured 2026-08-20, on the most common object in second
            # quantization).  Matching the identifier is also what this
            # codebase's own no-auto-split rule says a letter run is.
            s = re.sub(rf"\\{cmd}\s+([A-Za-z][A-Za-z0-9]*|[0-9])", rf"\1{suffix}", s)
            return s

        # Hat-family → _hat suffix
        for cmd in ("hat", "widehat"):
            s = _apply(cmd, "_hat", s)

        # Tilde-family → _tilde suffix
        for cmd in ("tilde", "widetilde"):
            s = _apply(cmd, "_tilde", s)

        # Bar-family → _bar suffix, but only over a NAME (see _bar_over_compound)
        s = self._bar_over_compound(s)
        for cmd in ("bar", "overline", "overbar"):
            s = _apply(cmd, "_bar", s)

        # Time derivative: \dot → _dot, \ddot → _ddot
        s = _apply("ddot", "_ddot", s)   # ddot before dot to avoid partial match
        s = _apply("dot",  "_dot",  s)

        # Script letters.  `\mathcal{H}` (Hamiltonian density) and
        # `\mathcal{L}` (Lagrangian density) are everyday notation and were
        # becoming the mangled names `mathcalH` / `mathcalL` — the braces were
        # dropped later in the pass and the command name glued to the letter
        # (measured 2026-08-20, the same shape as the bare `\hat\psi` →
        # `p_hatsi` defect).  They are a DECORATION: `\mathcal{H}` is a
        # different quantity from `H`, which is exactly why physics writes it.
        for cmd, suf in (("mathcal", "_cal"), ("mathscr", "_scr")):
            s = _apply(cmd, suf, s)

        # Miscellaneous accent commands → meaningful suffix
        for cmd, suf in (("check", "_check"), ("breve", "_breve"),
                         ("acute", "_acute"), ("grave", "_grave")):
            s = _apply(cmd, suf, s)

        # Underline: text emphasis only — strip decoration
        s = re.sub(r"\\underline\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\underline\s+([A-Za-z0-9])", r"\1", s)

        return s

    #: What the bar family may name: an identifier, optionally subscripted.
    #: `\bar{\phi}` and `\bar{\phi_{n}}` are both NAMES (H-1 stores them as
    #: `phi_bar` / `phi_n_bar`); the two spellings of the subscript must not
    #: disagree about that.
    _BARE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:_\{?[A-Za-z0-9,]+\}?)*$")

    def _bar_over_compound(self, s: str) -> str:
        r"""``\overline{f(x)}`` → ``(f(x))^*`` — a bar over a COMPOUND is
        complex conjugation, not a name.

        H-1 folds a decoration into the name (`\bar{H}` → `H_bar`,
        CLAUDE.md "Decorated forms") and that stays: over a bare identifier the
        bar is how the researcher SPELLS a quantity.  But the same rule applied
        to a compound produced a name out of an expression, and the
        conjugation vanished with no error at all (measured 2026-08-09):

            \overline{\psi(x)}  → psi(x)_bar     ← conjugate silently dropped
            \overline{a+b}      → a+b_bar        ← and only `b` was decorated

        Rewritten to `(inner)^*`, it goes through the conjugate machinery every
        other spelling uses rather than a second one of its own.  Braces are
        matched BALANCED — `\overline{\psi_{n}(x)}` holds braces of its own,
        and the regex form silently skipped exactly those.
        """
        for cmd in ("overline", "overbar", "bar"):
            head = "\\" + cmd + "{"
            out, i = [], 0
            while True:
                j = s.find(head, i)
                if j < 0:
                    out.append(s[i:])
                    break
                k, depth = j + len(head), 1
                while k < len(s) and depth:
                    depth += (s[k] == "{") - (s[k] == "}")
                    k += 1
                if depth:                      # unbalanced — leave it alone
                    out.append(s[i:])
                    break
                inner = s[j + len(head):k - 1].strip()
                out.append(s[i:j])
                out.append(f"{head}{inner}}}" if self._BARE_NAME_RE.match(inner)
                           else f"({inner})^*")
                i = k
            s = "".join(out)
        return s

    #: ベクトルの命令と、その引数の書かれ方。波括弧つき（`\\vec{A}`）と素
    #: （`\\vec A`）の両方で、直後に**空白なく**開き括弧が来たときだけ適用形。
    _APPLIED_VEC_RE = re.compile(
        r"\\(?P<cmd>vec|mathbf)\s*(?:\{(?P<braced>[^{}]+)\}|\s+(?P<bare>[A-Za-z][A-Za-z0-9]*))(?=\()")

    def _applied_vector(self, s: str) -> str:
        r"""``\vec{A}(t)`` → ``vec(A(t))``——包みは**適用の外**に付く。"""
        from notebook.runtime.latex_brackets import balanced
        out, i = [], 0
        while True:
            m = self._APPLIED_VEC_RE.search(s, i)
            if m is None:
                out.append(s[i:])
                break
            name = m.group("braced") or m.group("bare")
            read = balanced(s, m.end(), "(", ")")
            if read is None:                     # 閉じていない——触らない
                out.append(s[i:m.end()])
                i = m.end()
                continue
            args, after = read
            out.append(s[i:m.start()])
            out.append(f"vec({name}({args}))")
            i = after
        return "".join(out)

    def _vector(self, s: str) -> str:
        r"""Convert vector notations:
        - \vec{F} -> vec(F)
        - \vec F -> vec(F)
        - \mathbf{F} -> vec(F)
        """
        # **A VECTOR THAT IS APPLIED IS STILL ONE QUANTITY.**  `\vec A(t)` is the
        # vector field A at time t, and the wrapper has to go around the
        # APPLICATION — written the other way the argument list is left
        # standing beside `vec(A)` and becomes a FACTOR.  Measured 2026-09-11,
        # the whole of electromagnetism was unwritable and answered as success::
        #
        #     \vec{A}(t)                        → vec(A) * t
        #     \nabla \times \vec{A}(t)          → curl(vec(A) * t)   ← B = ∇×A
        #     \frac{\partial\vec{A}(t)}{\partial t} → 'diff(vec(A)*t,t,1) = A
        #
        # Only the GLUED form is an application — the same rule the parser uses
        # for `f(x)` vs `f (x)`, where a space means a product.
        s = self._applied_vector(s)
        s = re.sub(r"\\vec\{([^{}]+)\}", r"vec(\1)", s)
        s = re.sub(r"\\mathbf\{([^{}]+)\}", r"vec(\1)", s)
        # Bare forms.  The argument is the whole IDENTIFIER, not its first
        # character — the same rule the decorations follow, and for the same
        # measured reason: `\mathbf k` (no braces) was left alone and became
        # the PRODUCT `mathbf * k`, so inside a sum `mathbf` was even taken for
        # a second dummy index (`\sum_{\mathbf k'} v(\mathbf k-\mathbf k')`
        # lowered to `sum(sum(v(j*k - j*k'), k'), j)`, measured 2026-08-20).
        s = re.sub(r"\\vec\s+([A-Za-z][A-Za-z0-9]*|[0-9])", r"vec(\1)", s)
        s = re.sub(r"\\mathbf\s+([A-Za-z][A-Za-z0-9]*|[0-9])", r"vec(\1)", s)
        return s

    def _frac(self, s: str) -> str:
        r"""Replace \frac{num}{den} with (num)/(den), nesting and all.

        Both operands are scanned with balanced braces (``braced_group``) and
        converted recursively, so an operand carrying its own braced command
        works: ``\frac{1}{\sqrt{x^2+1}}``.  With the old non-nested regex that
        fraction was left unconverted, and the very same pass then stripped its
        braces (``_remove_remaining_braces``), so the next pass could not
        recover it either — it reached the CAS as ``frac1*sqrt(x^2+1)``, a
        PRODUCT of an undefined symbol, which Maxima evaluates without
        complaint.  Confirmed wrong by difference, not by eye:
        ``ratsimp(1/sqrt(x^2+1) - frac1*sqrt(x^2+1)) ≠ 0``.

        ``\frac`` without two braced operands (e.g. TeX's ``\frac12``) is left
        alone, as before — this rule converts what it can read, and never
        guesses.

        **The result is wrapped as a whole**, ``((num)/(den))``.  A fraction is
        ONE operand, and ``(num)/(den)`` is not: anything binding tighter than
        ``/`` on either side takes a piece of it, and every such case was
        silently wrong rather than an error —

        =========================== ==================== =====================
        input                       was                  means
        =========================== ==================== =====================
        ``\log \frac{1}{6}``        ``log (1)/(6)``      ``log(1)/6`` = **0**
        ``\sin \frac{\pi}{2}``      ``sin (%pi)/(2)``    ``sin(π)/2`` = **0**
        ``\log \frac{p(x)}{q(x)}``  ``log (p)/(q)``      ``log(p)/q``
        ``\frac{a}{b}^2``           ``(a)/(b)^2``        ``a/b²``
        ``x^\frac{1}{2}``           ``x^(1)/(2)``        ``x/2``
        =========================== ==================== =====================

        Found while rewriting the information-theory scenario onto general
        operations (open_items SCENARIO-MACRO): the entropy of a fair die,
        written as ``-\sum_{x=1}^{6}\frac16\log\frac16``, evaluated to **0** and
        came back a SUCCESS.  The three ``sum``-template macros there had been
        hiding this — they build the ``log`` call themselves and never send the
        researcher's own spelling through this converter.
        """
        out: list[str] = []
        i = 0
        while True:
            j = s.find(r"\frac", i)
            if j < 0:
                out.append(s[i:])
                return "".join(out)
            k = j + len(r"\frac")
            while k < len(s) and s[k] == " ":
                k += 1
            num = braced_group(s, k)
            den = braced_group(s, num[1]) if num else None
            if den is None:
                out.append(s[i:j + len(r"\frac")])
                i = j + len(r"\frac")
                continue
            out.append(s[i:j])
            out.append(f"(({self._frac(num[0])})/({self._frac(den[0])}))")
            i = den[1]

    def _sqrt(self, s: str) -> str:
        r"""Replace \sqrt[n]{x} → x^(1/n) and \sqrt{x} → sqrt(x)."""
        s = re.sub(r"\\sqrt\[([^\]]+)\]\{([^{}]*)\}", r"(\2)^(1/(\1))", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
        s = re.sub(r"\\sqrt\s+([A-Za-z0-9])", r"sqrt(\1)", s)
        return s

    def _superscript(self, s: str) -> str:
        r"""Replace x^{expr} with x^(expr) for multi-char exponents."""
        def replace_exp(m: re.Match) -> str:
            inner = m.group(1)
            if re.fullmatch(r"[A-Za-z0-9]", inner):
                return f"^{inner}"
            return f"^({inner})"
        return re.sub(r"\^\{([^{}]*)\}", replace_exp, s)

    def _trig_and_functions(self, s: str) -> str:
        for latex_cmd, fn_name in sorted(self._TRIG_MAP.items(),
                                         key=lambda x: -len(x[0])):
            s = replace_tex_command(s, latex_cmd, fn_name)
        return s

    def _greek_and_constants(self, s: str) -> str:
        for latex_cmd, canonical in sorted(self._GREEK_MAP.items(), key=lambda x: -len(x[0])):
            s = replace_tex_command(s, latex_cmd, canonical)
        return s

    def _operators(self, s: str) -> str:
        r"""演算子のコマンドを正準形へ —— **置換は ``replace_tex_command`` に訊く。**

        ここは長く ``str.replace`` / 前方一致の正規表現で書かれており、同じ問いの
        答えが同じファイルの ``replace_tex_command`` に既にあった。何が壊れて
        いたかはその docstring にある。
        """
        s = replace_tex_command(s, r"\cdot", " . ")
        s = replace_tex_command(s, r"\times", " ~ ")
        s = replace_tex_command(s, r"\div", "/")
        s = replace_tex_command(s, r"\nabla", "nabla")
        s = replace_tex_command(s, r"\pm", "+")
        s = replace_tex_command(s, r"\mp", "-")
        s = normalize_relational_ops(s)
        return s

    def _left_right(self, s: str) -> str:
        r"""Remove \left and \right size hints (keep the actual bracket)."""
        s = re.sub(r"\\left\s*", "", s)
        s = re.sub(r"\\right\s*", "", s)
        return s

    def _function_application(self, s: str) -> str:
        r"""``known-function{arg}`` / ``known-function<space>arg`` → ``fn(arg)``.

        The reading itself lives in
        :mod:`notebook.runtime.function_application` — it grew past what a
        regex can say once the argument was allowed to be a function too
        (``\\log \\sqrt{x}`` used to come out as ``log(sqrt) * x``).  Which
        names are functions stays here, in the backend's ``_FN_ALT``.
        """
        return apply_all(s, self._FN_ALT, self._LOG_FN)

    #: A subscript whose content is an ALGEBRAIC expression — `c_{k+q}`,
    #: `c_{k-q}`, `A_{i+1}`, `A_{2i}`.  Standard many-body and lattice notation,
    #: and both target CAS take it natively (`c(k+q)`).  Dropping the braces
    #: made the parser read only the first token as the subscript and the rest
    #: as separate terms: `c_{k+q}` reached Maxima as **`c(k) + q`** and
    #: `x_{n+1} = 2 x_n` as the false equation `x(n) + 1 = 2 x(n)` — silently
    #: (measured 2026-08-20).  A LEADING minus is not compound (`c_{-k}` is a
    #: negated index and has its own reading), and a comma-separated list is
    #: already excluded below.
    #: A digit run followed by a letter is an implicit PRODUCT (`A_{2i}` is A
    #: at 2i), which the brace-drop turned into `A(2) * i`.
    _COMPOUND_SUBSCRIPT = re.compile(r"[+*/^]|(?<=.)-|(?<=\d)(?=[A-Za-z])")

    #: A VECTOR-valued subscript, in this system's own spelling for a bold
    #: letter (`\\sum_{\\mathbf k}` becomes `{vec(k)}` by the time the braces
    #: are dropped).  Without them the subscript was only `vec` and the `(k)`
    #: became an application, so the sum ran over an index called `vec`
    #: (measured 2026-08-20).  Deliberately NOT any parenthesised content:
    #: `A_{f(x)}` is the function `A_f` at `x`, which is its own reading.
    _VEC_SUBSCRIPT = re.compile(r"vec\([^()]*\)'*")

    def _remove_remaining_braces(self, s: str) -> str:
        def _drop(m: "re.Match[str]") -> str:
            inner = m.group(1)
            if (m.start() > 0 and s[m.start() - 1] == "_"
                    and (self._COMPOUND_SUBSCRIPT.search(inner.strip())
                         or self._VEC_SUBSCRIPT.fullmatch(inner.strip()))):
                return m.group(0)
            return inner

        return re.sub(r"\{([^{}\\=,]*)\}", _drop, s)

    def _implicit_multiplication(self, s: str) -> str:
        """Insert explicit * for implicit multiplication patterns."""
        # A digit run before "(" is a NUMERIC LITERAL times a group — `2(x+1)`.
        # It is not one when the digits are part of an identifier: `G_0(k)`,
        # `H_1(t)`, `A_12(x)` are calls on the names `G_0`/`H_1`/`A_12`.  Without
        # the lookbehind this rule fired on the last digit of the NAME, so a
        # displayed `G_{0}\left(k,\omega\right)` came back as `G_0*(k, omega)` —
        # a product, silently, which is what the researcher gets for copying a
        # result out of the timeline.  Anchoring on the whole digit run matters:
        # matching one digit would still fire on the `2` of `A_12(`.
        s = re.sub(r"(?<![A-Za-z0-9_])(\d+)\s*\(", r"\1*(", s)
        s = re.sub(r"\)\s*\(", r")*(", s)
        s = re.sub(r"(\d)([A-Za-z])", r"\1*\2", s)
        s = re.sub(r"(\d)(%)", r"\1*\2", s)
        s = re.sub(r"(\d)\s+([A-Za-z%])", r"\1*\2", s)
        s = re.sub(r"(%[a-z]+)\s+([A-Za-z%])", r"\1*\2", s)
        s = re.sub(r"\)\s+([A-Za-z%])", r")*\1", s)
        s = re.sub(r"([A-Za-z])\s+([A-Za-z%])", r"\1*\2", s)
        # Standalone letter (not part of a multi-letter name) followed by
        # space + open-paren is multiplication, not a function call.
        # e.g. "m (l*omega^2)" → "m*(l*omega^2)"  (occurs after _expand_symbol_refs
        # substitutes a named symbol like g → (l*omega^2))
        # Guard: (?<![A-Za-z0-9_]) ensures we only match a standalone letter,
        # not the last letter of a function name like "sin" or "cos".
        s = re.sub(r"(?<![A-Za-z0-9_])([A-Za-z])\s+\(", r"\1*(", s)
        # A CAS noun form (`'diff(...)`, `'integrate(...)`) glued to what
        # precedes it.  The quote there is Maxima's hold operator, not a prime,
        # but with nothing between them the parser reads the name as PRIMED and
        # the hold disappears: `i\hbar\frac{\partial}{\partial t}G` — the
        # equation of motion — lowered to `i * hbar' * diff(G, t, 1)`, i.e. a
        # primed ħ times an EVALUATED derivative (measured 2026-08-20).  An
        # applied prime (`a'(1)`) has no name after the quote and is untouched.
        # 空白があっても同じこと（`S^2 'diff(V,S,2)`）。空白ごと `*` にする ——
        # 空白のある綴りを外していたため、Black–Scholes 方程式（プライムを1つも
        # 含まない）で `S^2` にプライムが付いた形として読まれ、拒否に落ちた。
        # 実例の再生成が捕まえた（EXAMPLES-DRIFT の検査が働いた形）。それまでは
        # 「プライムを黙って落とす」既定がこの綴りを通していた —— 落とす挙動は
        # 有害なだけではなく、ここでは保留を失う代わりに動いてもいた。
        s = re.sub(r"(?<=[A-Za-z0-9_)])\s*(?=\'[A-Za-z]+\s*\()", "*", s)
        return s

    def _whitespace(self, s: str) -> str:
        for cmd in (r"\,", r"\;", r"\:", r"\!", r"\ ", r"\quad", r"\qquad"):
            s = s.replace(cmd, " ")
        s = re.sub(r" {2,}", " ", s)
        return s.strip()



# ---------------------------------------------------------------------------
# Display Bridge: canonical → display LaTeX
# ---------------------------------------------------------------------------

# Decoration suffixes in the order they should be checked (longest first to
# avoid partial matches, e.g. "_ddot" before "_dot").
_DECORATION_SUFFIXES: list[tuple[str, str]] = [
    ("_cal",   r"\mathcal"),
    ("_scr",   r"\mathscr"),
    ("_ddot",  r"\ddot"),
    ("_dot",   r"\dot"),
    ("_hat",   r"\hat"),
    ("_tilde", r"\tilde"),
    ("_bar",   r"\bar"),
    ("_check", r"\check"),
    ("_breve", r"\breve"),
    ("_acute", r"\acute"),
    ("_grave", r"\grave"),
]

# Canonical identifier → LaTeX command for Greek letters.
# (Reverse of the _GREEK_MAP entries that map 1-to-1.)
_CANONICAL_TO_LATEX: dict[str, str] = {
    "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma",
    "delta": r"\delta", "epsilon": r"\epsilon", "zeta": r"\zeta",
    "eta": r"\eta", "theta": r"\theta", "iota": r"\iota",
    "kappa": r"\kappa", "lambda": r"\lambda", "mu": r"\mu",
    "nu": r"\nu", "xi": r"\xi", "rho": r"\rho", "sigma": r"\sigma",
    "tau": r"\tau", "upsilon": r"\upsilon", "phi": r"\phi",
    "chi": r"\chi", "psi": r"\psi", "omega": r"\omega",
    "Gamma": r"\Gamma", "Delta": r"\Delta", "Theta": r"\Theta",
    "Lambda": r"\Lambda", "Xi": r"\Xi", "Pi": r"\Pi",
    "Sigma": r"\Sigma", "Upsilon": r"\Upsilon", "Phi": r"\Phi",
    "Psi": r"\Psi", "Omega": r"\Omega",
    "hbar": r"\hbar",
}

# Sorted longest-first so that e.g. "omega" is tried before "om"
_CANONICAL_TO_LATEX_SORTED = sorted(
    _CANONICAL_TO_LATEX.items(), key=lambda kv: -len(kv[0])
)


#: Decorations that display as a SUPERSCRIPT instead of an accent.  There is no
#: `\dagger{X}` in LaTeX: the adjoint is written `X^{\dagger}`, which is also
#: the spelling the researcher typed.  (The accent table above is the
#: `\hat{X}` → `X_hat` direction; the dagger has no such input form — it comes
#: from the superscript, so it appears only in this, the restore, direction.)
_SCRIPT_DECORATIONS: dict[str, str] = {"dagger": "^{\\dagger}"}

#: Decoration NAME (without the underscore) → its LaTeX command.  Same table as
#: `_DECORATION_SUFFIXES`, keyed the way a name's pieces are read.
_DECO_CMD_BY_NAME: dict[str, str] = {suf[1:]: cmd for suf, cmd in _DECORATION_SUFFIXES}


def _split_decoration(name: str) -> tuple[str, str] | tuple[None, None]:
    """If *name* ends with a known decoration suffix, return (base, decoration).

    Returns ``(None, None)`` when no decoration suffix is found.
    """
    for suffix in ([s for s, _ in _DECORATION_SUFFIXES]
                   + ["_" + d for d in _SCRIPT_DECORATIONS]):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if base and not base.endswith("_"):
                return base, suffix[1:]  # strip leading underscore
    return None, None


#: 研究者が**名前として**書きうる綴り。`\bar{y}` / `y'` / `x_1` / `\alpha_k` /
#: `H` ——式ではなく1つの名前であるもの。`i18n` の `symbol` 断片と同じ形だが、
#: ここは NL の文法ではなく**変換の入口**なので、表そのものは持たない。
_NAME_TOKEN_RE = re.compile(
    r"^(?:\\[A-Za-z]+\{[^{}]*\}"                 # \bar{y}, \mathcal{H}
    r"|\\[A-Za-z]+"                              # \alpha
    r"|[A-Za-z_][A-Za-z0-9_]*)"                  # y, x_1
    r"(?:[_^](?:\{[^{}]*\}|[A-Za-z0-9]+))*'*$"   # 添字・上付き・プライム
)


def latex_name_to_canonical(token: str) -> "str | None":
    r"""**名前1つ**を正準の綴りへ。名前でないものには `None` を返す。

    `\bar{y}` → ``y_bar``、`\hat{H}` → ``H_hat``、`\alpha` → ``alpha``、
    `\alpha_k` → ``alpha_k``、`y'` → ``y'``、`x_1` → ``x_1``、`H` → ``H``。
    `a+b` や `f(x)` のような**式**は名前ではないので `None`。

    **逆向きは既にここに在る**（`restore_decorations` / `_split_decoration`）ので、
    表は1つ、方向が2つという形になる。装飾の畳み込みはバックエンドに依らない
    共有の約束であり（Maxima も Julia も同じ `y_bar` を使う）、バックエンドごとに
    違うのは定数名と関数名のほうである——だから完全な `latex_to_canonical`
    （ハンドラの仕事）ではなく、**名前だけ**をここで答える。

    なぜ要るか（DECORATED-VAR-NOT-TOKENIZED、実測 2026-09-19）: 別名走査が欄を
    埋める経路には**素の識別子と `\alpha` のような単純なコマンドしか通らない**
    手書きの規則があり、`\bar{y}` / `y'` / `x_1` を変数に書いた指示が
    「必要な引数が不足しています」で落ちていた（不定積分・collect・部分分数・
    留数・極限、日英とも。`x_1` は**普通の変数**である）。
    """
    tok = (token or "").strip()
    if not tok or not _NAME_TOKEN_RE.match(tok):
        return None
    # `\deco{base}` は名前の接尾辞へ畳む——`restore_decorations` の逆。
    m = re.match(r"^\\([A-Za-z]+)\{([^{}]*)\}(?P<rest>.*)$", tok)
    # 綴りは**コマンド**（`\\mathcal`）、接尾辞は**名前**（`cal`）——表は
    # `{名前: コマンド}` なので、引くのはその逆である。
    _by_cmd = {cmd[1:]: name for name, cmd in _DECO_CMD_BY_NAME.items()}
    if m:
        # **知らないコマンドは名前ではない。** `\\vec{A}` は名前の飾りではなく
        # **包み**で、変換器は `vec(A)` を返す（実測——ここで `vec{A}` を返して
        # いた）。表に無いものは答えない。
        if m.group(1) not in _by_cmd:
            return None
        inner = latex_name_to_canonical(m.group(2))
        if inner is None:
            return None
        return f"{inner}_{_by_cmd[m.group(1)]}{m.group('rest')}"
    if tok.startswith("\\"):
        return tok[1:]
    return tok


def _base_to_display_latex(base: str) -> str:
    """Convert a canonical base identifier to its display LaTeX form.

    E.g. ``"omega"`` → ``"\\\\omega"``, ``"H_0"`` → ``"H_0"``,
    ``"xi_k"`` → ``"\\\\xi_k"``.
    """
    if base in _CANONICAL_TO_LATEX:
        return _CANONICAL_TO_LATEX[base]
    for greek, cmd in _CANONICAL_TO_LATEX_SORTED:
        if base.startswith(greek):
            rest = base[len(greek):]
            if not rest or rest[0] == "_":
                return cmd + rest
    return base


#: A superscript label as it appears in DISPLAY text — bare (`G_sup0`) or in
#: the braced subscript spelling Maxima's `tex1` produces (`G_{sup0}`).
_LABEL_IN_LATEX_RE = re.compile(
    r"(?<![\\A-Za-z])(?P<b>[A-Za-z][A-Za-z0-9]*)_\{?(?P<p>p?)sup"
    r"(?P<v>[A-Za-z0-9]+)\}?(?![A-Za-z0-9])"
)


def restore_decorations(s: str) -> str:
    r"""Convert canonical decoration suffixes back to LaTeX decoration commands.

    This is the **inverse** of :meth:`LatexConverter._decorations`.  It is
    applied by the Display Bridge inside ``postprocess_result_latex()`` so
    that CAS output containing decorated symbol names is rendered correctly.

    Examples::

        restore_decorations("H_hat")          → "\\hat{H}"
        restore_decorations("omega_tilde")    → "\\tilde{\\omega}"
        restore_decorations("H_0_hat")        → "\\hat{H_0}"
        restore_decorations("H_hat + V_hat")  → "\\hat{H} + \\hat{V}"

    The function scans identifier tokens (``[A-Za-z][A-Za-z0-9_]*``) and
    replaces those whose trailing suffix matches a known decoration.
    Non-identifier content (numbers, operators, LaTeX commands already
    present) is passed through unchanged.
    """
    def _restore_name(ident: str) -> "str | None":
        r"""``psi_hat_dagger_sigma`` → ``\hat{\psi}^{\dagger}_{\sigma}``, or None.

        A decorated name that also carries INDICES had no reading at all: the
        Maxima bridge braced every underscore piece, so a CAS result mentioning
        the most ordinary object in this corpus came back as
        ``psi_{hat}_{sigma}`` — a DOUBLE SUBSCRIPT, which is not valid LaTeX and
        does not typeset (measured 2026-08-21; ``psi_hat`` alone was fine, which
        is why it went unseen).

        The pieces are read left to right: everything BEFORE the first
        decoration belongs to the base name (so ``H_0_hat`` stays
        ``\hat{H_0}``, as documented), the decorations wrap it in the order
        written, and whatever follows a decoration is an index.
        """
        pieces = ident.split("_")
        # A TRAILING underscore means the LaTeX continues with a braced
        # subscript (`c_hat_{n,\sigma}` — the token scan stops at the brace).
        # It is not an empty index: read as one it printed `\hat{c}_{}{n,\sigma}`.
        trailing = pieces[-1] == "" and len(pieces) > 1
        if trailing:
            pieces = pieces[:-1]
        base_parts, decos, indices = [pieces[0]], [], []
        for piece in pieces[1:]:
            if piece in _DECO_CMD_BY_NAME or piece in _SCRIPT_DECORATIONS:
                decos.append(piece)
            elif decos:
                indices.append(piece)
            else:
                base_parts.append(piece)
        if not decos:
            return None
        out = _base_to_display_latex("_".join(base_parts))
        for deco in decos:
            if deco in _SCRIPT_DECORATIONS:
                out = f"{out}{_SCRIPT_DECORATIONS[deco]}"
            else:
                out = f"{_DECO_CMD_BY_NAME[deco]}{{{out}}}"
        if indices:
            out += "_{" + ", ".join(_base_to_display_latex(i) for i in indices) + "}"
        return out + "_" if trailing else out

    def _replace_tok(m: re.Match) -> str:
        ident = m.group(0)
        # A LEADING BACKSLASH means the base is already a command — Maxima's
        # `tex1` writes the canonical `theta_dot` as `\theta_dot`, converting
        # the Greek prefix and leaving the decoration suffix behind.  Matching
        # only the identifier left that backslash standing in front of the
        # restoration, so the result was `\\dot{\theta}` — and `\\` is a LINE
        # BREAK in LaTeX, not a decoration (measured 2026-08-23 on the
        # substitution label of an analytical-mechanics scenario, where
        # `\dot\theta` is the most ordinary quantity there is; `\tilde\omega`
        # and `\hat\Sigma` are the same shape).  The docstring above already
        # claimed commands pass through unchanged; for a command whose NAME
        # carries a decoration suffix it was not true.
        #
        # The backslash is consumed only when a restoration actually happens —
        # `_base_to_display_latex` puts the command form back.  Otherwise the
        # whole match is returned untouched, which is what keeps `\frac`,
        # `\left`, `\sin` and a bare `\alpha` intact.
        _had_backslash = ident.startswith("\\")
        if _had_backslash:
            ident = ident[1:]
        _label = _split_script_label(ident)
        if _label is not None:
            # `G_sup0` → `G^{0}` / `Sigma_psup2` → `\Sigma^{(2)}`.  The label is
            # part of the NAME on the CAS side; display puts it back where the
            # researcher wrote it — and the braced-vs-parenthesised distinction
            # has to survive, because re-typing `G^{2}` and `G^{(2)}` gives two
            # different readings.
            _b, _v, _paren = _label
            _v = f"({_v})" if _paren else _v
            return f"{_base_to_display_latex(_b)}^{{{_v}}}"
        restored = _restore_name(ident)
        if restored is not None:
            return restored
        base, deco = _split_decoration(ident)
        if base is None:
            return ("\\" + ident) if _had_backslash else ident
        base_latex = _base_to_display_latex(base)
        if deco in _SCRIPT_DECORATIONS:
            return f"{base_latex}{_SCRIPT_DECORATIONS[deco]}"
        cmd = {s[1:]: c for s, c in _DECORATION_SUFFIXES}[deco]  # type: ignore[index]
        return f"{cmd}{{{base_latex}}}"

    # Maxima's `tex1` renders `G_sup0` as the BRACED subscript `G_{sup0}`, so the
    # plain-identifier scan below never sees it (the same shape the lagrangian
    # `q_{dot}` fix in the Maxima bridge already had to handle).  Labels are put
    # back first, in either spelling.
    s = _LABEL_IN_LATEX_RE.sub(
        lambda m: (f"{_base_to_display_latex(m.group('b'))}"
                   f"^{{{'(' + m.group('v') + ')' if m.group('p') else m.group('v')}}}"),
        s)
    return re.sub(r"\\?[A-Za-z][A-Za-z0-9_]*", _replace_tok, s)


# ---------------------------------------------------------------------------
# (The Maxima-specific ``latex_to_canonical`` shim used to live here; it moved
# to ``maxima_latex_converter`` so this shared base no longer imports a backend
# module — open_items AUDIT-SHARED-CAS-LEAK.)


def split_state_braket(body: str) -> "tuple[str, str] | None":
    r"""``\Phi_0 | A | \Phi_0`` → ``("A", "\Phi_0")``、読めなければ None。

    ブラケットの縦棒は**区切り**であって囲いではない —— その前提は正しく、だから
    絶対値や行列式の読みは当たらない。だが「したがって入力としては読まない」という
    結論は、⟨…⟩ が読めるようになった時点でずれた: **両側が同じ状態なら、それは
    ``\langle A \rangle_{\Phi}`` と同じ量**であり、この系はもうその対象
    (``expectation(A, state)``) を持っている。外側2つが文字どおり同一であることが
    判断材料なので、推測は入らない。

    状態が異なる形（行列要素 ``\langle\phi|A|\psi\rangle``）と、縦棒が1本の形
    （内積 ``\langle\phi|\psi\rangle``）は**状態を2つ持つ量**で、名前を決める必要が
    あるため、ここでは読まない（open_items BRAKET-TWO-STATES）。
    """
    parts, depth, buf = [], 0, []
    for ch in body:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    if len(parts) != 3:
        return None
    bra, op, ket = (p.strip() for p in parts)
    if not bra or not op or bra != ket:
        return None
    return op, bra


def unreadable_braket(text: str) -> bool:
    r"""*text* に、この系が読まないブラケットがあるか。

    読むのは「両側が同じ状態」の形だけ（`split_state_braket`）。行列要素と内積は
    状態を2つ持つ量で、名前が決まっていない。片側だけの ``|\psi\rangle`` / 
    ``\langle\psi|`` も読まない —— 状態そのもので、CAS 値が無い。

    **判定はここ一箇所** —— 記法確認のインターセプタは変換器より手前で走り、縦棒を
    含む式を先に断つので、両者が別々に「読めるか」を決めると、変換器が読める形を
    インターセプタが断つことになる。読める部分を伏せてから縦棒が残るかを見るので、
    片側だけの形も残った縦棒として捕まる（実測 2026-08-20: 先に読める形だけを
    探す書き方にしたら、素のケットが一般の縦棒エラー——絶対値や行列式の話——に
    落ちた）。
    """
    rest = []
    i = 0
    while True:
        j = text.find("\\langle", i)
        if j < 0:
            rest.append(text[i:])
            break
        k = text.find("\\rangle", j)
        if k < 0:
            rest.append(text[i:])
            break
        body = text[j + len("\\langle"):k]
        rest.append(text[i:j])
        if "|" in body and split_state_braket(body) is None:
            rest.append(body)          # 読めない中身は残す
        i = k + len("\\rangle")
    return "|" in "".join(rest)
