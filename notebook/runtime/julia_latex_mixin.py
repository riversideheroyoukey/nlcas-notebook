"""Julia backend mixin: LaTeX display / output formatting.

Extracted from JuliaBackendHandler (AUD-4 stage 2b).  Converts CAS output and
canonical strings into KaTeX-ready LaTeX (postprocess_result_latex, clean_latex,
normalize_display_latex, SymPy-canonical and spin-component handling).  Mixed
into JuliaBackendHandler; base restore helpers (_restore_*) and siblings resolve
via self / MRO.  The display-only regex/table class attributes travel with it.
"""
from __future__ import annotations

import re

from notebook.runtime.backend_handler import GREEK_CMD
from notebook.runtime.julia_handler_common import _GREEK_CMD


# ---------------------------------------------------------------------------
# I-1: imaginary-unit ordering in a coefficient.
# QuantumAlgebra/Symbolics emit coefficient factors in internal order, so the
# imaginary unit lands at the end, e.g. "-2 B \gamma i S_{y_1}".  Move it right
# after the numeric coefficient: "-2 i B \gamma S_{y_1}".
#
# Safety: a "factor" is only a single ASCII letter (optionally subscripted) or a
# Greek-letter command — NOT a structural command (\frac, \sqrt, \left, …), so a
# factor run never crosses into a fraction/group and complex expressions are
# left untouched.  Fires only when exactly one imaginary unit is present and it
# is not already first.
# ---------------------------------------------------------------------------
_IMAG_TOKEN = r"(?:\\mathtt\{im\}|\\mathrm\{i\}|im|i)"
_GREEK_NAMES = sorted(
    (v.lstrip("\\") for v in _GREEK_CMD.values()), key=len, reverse=True
)
_GREEK_ALT = "|".join(re.escape(n) for n in _GREEK_NAMES)
_SYM_FACTOR = (
    r"(?:\\(?:" + _GREEK_ALT + r")|[A-Za-z])"
    r"(?:_\{[^{}]*\}|_[A-Za-z0-9])?"
)
_COEFF_RUN_RE = re.compile(
    r"(?<![A-Za-z}\\\d_^])(-?\d+(?:\.\d+)?)"
    r"((?:\s+(?:" + _IMAG_TOKEN + r"|" + _SYM_FACTOR + r"))+)"
)
_RUN_TOK_RE = re.compile(r"(\s+)(" + _IMAG_TOKEN + r"|" + _SYM_FACTOR + r")")
_IMAG_FULL_RE = re.compile(_IMAG_TOKEN + r"\Z")


def _reorder_imaginary_unit(s: str) -> str:
    """Move a trailing imaginary unit to the front of its monomial coefficient."""
    def _repl(m: "re.Match") -> str:
        num, run = m.group(1), m.group(2)
        toks = _RUN_TOK_RE.findall(run)          # [(sep, token), …]
        imag_idx = [i for i, (_, t) in enumerate(toks) if _IMAG_FULL_RE.fullmatch(t)]
        if len(imag_idx) != 1 or imag_idx[0] == 0:
            return m.group(0)                    # none / multiple / already first
        i0 = imag_idx[0]
        sep0 = toks[0][0]
        imag_tok = toks[i0][1]
        rest = [toks[k] for k in range(len(toks)) if k != i0]
        return num + sep0 + imag_tok + "".join(sep + tok for sep, tok in rest)
    return _COEFF_RUN_RE.sub(_repl, s)


class JuliaLatexMixin:
    """LaTeX display/output formatting helpers mixed into JuliaBackendHandler."""

    def _restore_latexify_subscripts(self, s: str) -> str:
        r"""Undo Latexify.jl's two spellings of a subscripted name.

        It wraps a multi-character Symbolics name in ``\mathtt{}`` and escapes
        the underscore, and on some paths (the Heisenberg equation) emits the
        escaped form WITHOUT the wrapper.  Inside either, ``\_`` renders as a
        literal underscore, not a subscript, so a declared ``omega_0`` printed as
        the word "omega" followed by an underscore.

        The ORDER matters and is the whole reason these two live together: undo
        the escape first and the wrapper's unwrap loses the very mark that tells
        it where the subscript is — `\mathtt{xi\_k}` stopped becoming `\xi_{k}`
        (measured 2026-08-09, a regression this arrangement exists to prevent).
        """
        from notebook.runtime.backend_handler import GREEK_CMD

        # (a) Unwrap \mathtt{X\_Y} → X_{Y} (Latexify subscript convention).
        # Latexify.jl wraps multi-char Symbolics variable names in \mathtt{} and
        # escapes underscores as \_, e.g. \mathtt{u\_k} or \mathtt{\xi\_k}.
        # Inside \mathtt{}, \_ renders as a literal underscore, not a subscript.
        # Restore to proper LaTeX subscript form: u_{k}, \xi_{k}, etc.
        def _unwrap_mathtt(m: re.Match) -> str:
            content = m.group(1)
            # \mathtt{\omega_0} or \mathtt{\omega_{0}} — already a TeX command inside
            if content.startswith("\\"):
                return content
            if r"\_" in content:
                base, sub = content.split(r"\_", 1)
                # base may itself be a Greek name (e.g. "omega" → \omega)
                base_tex = GREEK_CMD.get(base, base)
                return f"{base_tex}_{{{sub}}}"
            # Plain multi-char name with no subscript — convert Greek names
            return GREEK_CMD.get(content, content)
        s = re.sub(r'\\mathtt\{([^}]+)\}', _unwrap_mathtt, s)

        # Latexify escapes the underscore in a name it did NOT wrap in \mathtt{}
        # either — the Heisenberg-equation path yields `omega\_0`, where `\_` is a
        # literal underscore rather than a subscript, so a declared `omega_0`
        # printed as the word "omega" followed by an underscore.  Same convention
        # as above, same repair, for where the wrapper is absent.
        #
        # AFTER the unwrap, never before: a blanket unescape removes the very
        # `\_` that tells the unwrap where the subscript is, and `\mathtt{xi\_k}`
        # stopped becoming `\xi_{k}` (measured 2026-08-09 — the regression this
        # ordering exists to prevent).
        def _unescape_subscript(m: re.Match) -> str:
            base, sub = m.group(1), m.group(2)
            return f"{GREEK_CMD.get(base, base)}_{{{sub}}}"
        s = re.sub(r'(?<![A-Za-z])(\\?[A-Za-z]+)\\_([A-Za-z0-9]+)',
                   _unescape_subscript, s)
        return s

    def postprocess_result_latex(
        self,
        canonical: str,
        raw_latex: str,
        registry: SymbolContext,
        sum_indices: str = "",
    ) -> str:
        import re

        # §1  Shared restorations (kron_delta, matrix ops, IndexSum)
        raw_latex = self._restore_kron_delta(raw_latex)
        raw_latex = self._restore_matrix_ops(raw_latex)
        raw_latex = self._restore_index_sum(raw_latex)

        # 1.9. Normalize SymPy quantum operator LaTeX to subscript notation.
        # SymPy renders function calls with \left/\right parens:
        #   creation operators (cd, ad, …): \operatorname{cd}{\left(ARGS\right)}
        #   annihilation operators (c, a, …): c{\left(ARGS\right)}
        # Map these directly to subscript form (c^{\dagger}_{ARGS}, c_{ARGS})
        # before the Display Bridge runs, so the Bridge's prime_pat = "(...)*"
        # regex cannot accidentally apply annihilation formatting to creation ops.
        _is_quantum_ctx = (
            "second_quantization" in registry.domains or
            "quantum_mechanics" in registry.domains or
            registry.session_mode == "second_quantization"
        )
        for _sq_name, _sq_prop in registry.registry.items():
            _sq_qty = _sq_prop.quantity
            if _sq_qty.kind != "operator":
                continue
            _sq_primes = _sq_name.count("'") + _sq_name.count("†")
            _sq_base = _sq_name.rstrip("'†")
            if _sq_primes > 0:
                # Creation operator: Julia SymPy uses BASE+"d" (cd, ad, bd, …)
                _julia_op = _sq_base + "d"
                _sq_display = (f"{_sq_base}^{{\\dagger}}" if _is_quantum_ctx
                               else (_sq_base + "'" * _sq_primes))
                _sq_d = _sq_display
                # \operatorname{cd}{\left(ARGS\right)} → c^{\dagger}_{ARGS}
                raw_latex = re.sub(
                    r'\\operatorname\{' + re.escape(_julia_op) + r'\}'
                    r'\s*\{\s*\\left\s*\(\s*(.*?)\s*\\right\s*\)\s*\}',
                    lambda m, _d=_sq_d: f"{_d}_{{{m.group(1).strip().replace(' ', '')}}}",
                    raw_latex,
                )
            else:
                # Annihilation operator: Julia SymPy uses BASE name unchanged
                _sq_d2 = _sq_base
                # c{\left(ARGS\right)} → c_{ARGS}
                raw_latex = re.sub(
                    r'\b' + re.escape(_sq_base)
                    + r'\{\s*\\left\s*\(\s*(.*?)\s*\\right\s*\)\s*\}',
                    lambda m, _d=_sq_d2: f"{_d}_{{{m.group(1).strip().replace(' ', '')}}}",
                    raw_latex,
                )

        # Clean up SymPy double-negative coefficient artifacts:
        # "- -1*EXPR" = "+(−1)·(−1)·EXPR" = "+EXPR".
        # This can arise from older SymPy versions not fully collapsing the
        # sign product (-coeff) * (-operator_term).
        raw_latex = re.sub(r'-\s*-\s*1\s*\*\s*', '+ ', raw_latex)
        raw_latex = raw_latex.strip()
        if raw_latex.startswith('+ '):
            raw_latex = raw_latex[2:].strip()
        elif raw_latex.startswith('+') and not raw_latex.startswith('++'):
            raw_latex = raw_latex[1:].strip()

        # §2  Display Bridge (shared; _display_bridge_scalar_guard adds Guard 2)
        raw_latex = self._apply_display_bridge(raw_latex, registry)

        s = raw_latex.replace(r"\,", " ")
        # Latexify.jl uses ~ (LaTeX tie/non-breaking space) between coefficients
        # and variables.  Normalise to a single space for clean display/export.
        s = re.sub(r'\s*~\s*', ' ', s)

        # Name QuantumAlgebra's dummy summation indices (`#₁`, `#₂`, …) back
        # to what the RESEARCHER wrote.
        #
        # QA discards the written name and numbers what is left by FIRST
        # OCCURRENCE IN THE BODY — measured 2026-08-07 with marker parameters,
        # and preserved across `normal_form`.  `sum_indices` is that order (the
        # generator computes it from its own output; see `_sum_order`), NOT the
        # binding order, which is what made `\sum_{m,n} c'_{n,u} c_{m,u}` come
        # back as `c†_m c_n`.
        #
        # With no recorded order — a sum the backend introduced itself — no
        # name is invented: neutral `k_{1}, k_{2}, …` from the one place that
        # defines them, which the Σ-prefix path in engine.py reads too, so the
        # sign and its body cannot disagree.  The previous table
        # `["j","k","l","m","n"]` was indexed MODULO 5, so a sixth dummy came
        # back as `j` again — two summation indices under one name.
        _names = [n.strip() for n in (sum_indices or "").split(",") if n.strip()]
        _fresh = self._fresh_dummy_names(s, _names, registry)

        def _name_for(num: int) -> str:
            if 1 <= num <= len(_names):
                return _names[num - 1]
            # A dummy the BACKEND introduced — the researcher never wrote it, so
            # there is no name to recover.  Give it the next unused letter of the
            # family they ARE using (`m,n,p,q` → `r`) rather than a number from
            # another series: a first-order commutator of two general terms puts
            # one of these in every sum, and `k_{5}` beside `m,n,p,q` is not a
            # line anyone would publish (open_items SUM-DUMMY-NAMES).
            offset = num - len(_names) - 1
            if 0 <= offset < len(_fresh):
                return _fresh[offset]
            return self.sum_dummy_display_names(num)[num - 1]

        def replace_run(match):
            run = match.group(0)
            names = [_name_for(int(n)) for n in re.findall(r"\d+", run)]
            # SEPARATE them.  QuantumAlgebra writes a term's indices with no
            # separator at all (`V(#₁#₂#₃#₄)`), and substituting names into that
            # produced `V_{mnpq}` and `\sum_{mnpqr}` — which this project's own
            # rule reads as ONE index named `mnpq` (CLAUDE.md, AUDIT-SUBSCRIPT-
            # SPLIT).  So the display did not round-trip: feeding `\sum_{mn}
            # h_{mn}` back in gave a SINGLE sum over an index called `mn`
            # (measured 2026-08-08).  A comma is what the input side requires,
            # so it is what the output side must write.
            body = ",".join(names)
            before = s[:match.start()]
            after = s[match.end():]
            # A literal neighbour is another index (the spin label in `c_{#₁u}`),
            # so it needs the separator too.  This also subsumes the Greek-fusion
            # trap it used to be braced against: `m` next to `u` cannot become
            # `\mu` once there is a comma between them.
            if re.search(r"[A-Za-z]$", before):
                body = "," + body
            if re.match(r"[A-Za-z]", after):
                body = body + ","
            return body

        # `\#_{1}`, `\#1`, `\#_{\{1\}}` (including optionally escaped braces),
        # matched as a RUN so consecutive indices are separated in one place.
        _marker = (r"(?:\\?#_{\\?\{\\?\d+\\?\\?\}}"
                   r"|\\?#_{\d+}"
                   r"|\\?#\d+)")
        s = re.sub(r"(?:%s)+" % _marker, replace_run, s)

        # 3. Convert registered parameter names that are Greek letters to proper
        # LaTeX commands.  QuantumAlgebra.latex() outputs parameter names as plain
        # ASCII text (e.g. "Delta", "xi") without the leading backslash.  This step
        # restores \Delta, \xi, etc. so that KaTeX can rasterize them correctly.
        #
        # Pattern: not preceded by a letter or `\` (to avoid touching already-escaped
        # commands like `\Delta`), and not followed by a letter (to avoid partial
        # matches inside compound identifiers like "DeltaE" or "xistar").
        # `_` and `^` are allowed as the next character so that `xi_{k}` → `\xi_{k}`.
        for reg_name, prop in registry.registry.items():
            qty = prop.quantity
            if qty.kind not in ("scalar", "anonymous"):
                continue
            base = reg_name.rstrip("'†")
            # Prefer an explicit latex_name stored on the quantity, then Greek table.
            latex_cmd = getattr(qty, "latex_name", None) or _GREEK_CMD.get(base)
            if not latex_cmd:
                continue
            # Skip if the LaTeX command would be the same as the original (no-op).
            if latex_cmd == base:
                continue
            # Replace bare occurrences: not preceded by letter/backslash,
            # not followed by a letter (but underscore/caret/digit are allowed).
            # Use a callable replacement to avoid Python regex escape-sequence
            # errors with backslash-prefixed LaTeX commands (e.g. \Delta).
            _cmd = latex_cmd
            s = re.sub(
                rf'(?<![A-Za-z\\]){re.escape(base)}(?![A-Za-z])',
                lambda _m, _c=_cmd: _c,
                s,
            )

        s = self._restore_latexify_subscripts(s)
        # Also add braces to any remaining bare letter subscripts (e.g. v_k → v_{k}).
        s = re.sub(r'(?<=[A-Za-z])_([A-Za-z0-9]+)(?!\{)', r'_{\1}', s)

        # 4. QuantumAlgebra uppercase index → negated lowercase conversion.
        # QuantumAlgebra represents -k as uppercase K (and -j as J, etc.) in its
        # LaTeX output.  Convert back to standard physics notation.
        # Rule: apply only when the lowercase version appears elsewhere as a
        # subscript in the same expression — avoids touching standalone uppercase
        # labels such as BZ K-points that have no negative-k interpretation.
        #
        # Handles both:
        #   _{K}   → _{-k}        (single-letter subscript, e.g. c^†_{K})
        #   _{Kk}  → _{-k,k}      (two-letter subscript in Kronecker delta)
        #   _{Kq}  → _{-k,q}      (mixed, e.g. δ_{Kq} → δ_{-k,q})
        _lower_subs = set(re.findall(r'_\{([a-z])\}', s))
        _lower_subs |= set(re.findall(r'_([a-z])(?![A-Za-z])', s))
        # Also collect lowercase from inside multi-char subscripts like _{Kk}
        for _cs in re.findall(r'_\{([A-Za-z]{2,})\}', s):
            _lower_subs |= {c for c in _cs if c.islower()}

        def _negate_upper_in_sub(m: re.Match) -> str:
            content = m.group(1)
            # Only transform pure-letter subscripts (no +, -, *, digits, spaces).
            # Complex subscripts like _{k+q} are left untouched.
            if not re.fullmatch(r'[A-Za-z]+', content):
                return m.group(0)
            parts = []
            any_negated = False
            for ch in content:
                if ch.isupper() and ch.lower() in _lower_subs:
                    parts.append('-' + ch.lower())
                    any_negated = True
                else:
                    parts.append(ch)
            # Only modify the subscript when at least one uppercase was negated.
            # Pure-lowercase subscripts like _{ij} must be returned unchanged to
            # avoid inserting spurious commas (_{i,j} ≠ _{ij} in KaTeX).
            if not any_negated:
                return m.group(0)
            # Re-join: use comma separator when the converted parts expand
            # (e.g. "Kk" → ["-k", "k"] → "-k,k")
            if len(parts) > 1:
                return '_{' + ','.join(parts) + '}'
            return '_{' + parts[0] + '}'

        s = re.sub(r'_\{([^}]+)\}', _negate_upper_in_sub, s)

        # 4b. Normalize QA spin-component LaTeX to match the user's input convention.
        # QA renders Sy(1) as S_{1}^{y} (site subscript, component superscript),
        # but users type Sx_1 (component-then-site).  Convert to Sc_{k} form.
        s = self._normalize_spin_component_latex(s, registry)

        # 5. Convert QuantumAlgebra operator names to LaTeX Greek commands.
        s = self.clean_latex(s)

        # 6. Normalize SymPy function-call notation to subscript notation for
        # scalar physics parameters.  SymPy renders `rho(-k, k)` as
        # `\rho{\left(- k, k \right)}` but QA renders the same contraction as
        # `\rho_{-k,k}`.  This step makes both backends display consistently.
        #
        # Pattern: \name{\left( args \right)} → \name_{args}
        #   where name is a registered scalar parameter (not a math function).
        #
        # Standard math functions (sin, cos, exp, …) are excluded; only
        # user-registered physics parameters are converted.
        _MATH_BUILTINS = frozenset({
            "sin", "cos", "tan", "cot", "sec", "csc",
            "sinh", "cosh", "tanh", "exp", "log", "ln",
            "sqrt", "operatorname", "text", "mathrm", "mathbf", "mathtt",
            "frac", "partial", "int", "sum", "prod",
            "lim", "inf", "sup", "det", "tr",
        })
        # Collect LaTeX names for registered scalars (including Greek variants).
        _scalar_latex_names: set[str] = set()
        for _rn, _rp in registry.registry.items():
            if _rp.quantity.kind in ("scalar", "anonymous"):
                _base = _rn.rstrip("'†")
                _scalar_latex_names.add(_base)
                _lat = getattr(_rp.quantity, "latex_name", None) or _GREEK_CMD.get(_base)
                if _lat:
                    _scalar_latex_names.add(_lat.lstrip("\\"))

        def _fn_to_subscript(m: re.Match) -> str:
            fn_name = m.group(1)    # e.g. "rho" from \rho
            args    = m.group(2)    # e.g. "- k, k" from \left(- k, k \right)
            if fn_name.lower() in _MATH_BUILTINS:
                return m.group(0)
            if _scalar_latex_names and fn_name not in _scalar_latex_names:
                return m.group(0)   # not a registered parameter, leave untouched
            clean_args = re.sub(r'\s+', '', args)
            return f"\\{fn_name}_{{{clean_args}}}"

        s = re.sub(
            r'\\([A-Za-z_]+)\{\\left\((.+?)\\right\s*\)\}',
            _fn_to_subscript,
            s,
        )

        # B-2: restore decoration suffixes to LaTeX commands
        from notebook.runtime.latex_converter import restore_decorations
        s = restore_decorations(s)

        # I-1: put the imaginary unit right after the numeric coefficient.
        s = _reorder_imaginary_unit(s)

        return s

    @staticmethod
    def _sympy_canonical_to_latex(s: str) -> str:
        """Convert SymPy canonical string to LaTeX.

        Applied when the display string came from SymPy.solve (not Latexify),
        e.g. "sqrt(2*m*mu)/hbar" → "\\frac{\\sqrt{2 m \\mu}}{\\hbar}".
        Only fires when the string contains bare 'sqrt(' (no backslash).
        """
        import re as _re
        if "sqrt(" not in s:
            return s

        # sqrt(X) → \sqrt{X}  (handles nested parens by walking brackets)
        def _replace_sqrt(text: str) -> str:
            result = []
            i = 0
            while i < len(text):
                if text[i:i+5] == "sqrt(":
                    depth = 0
                    j = i + 4  # points at '('
                    while j < len(text):
                        if text[j] == '(':
                            depth += 1
                        elif text[j] == ')':
                            depth -= 1
                            if depth == 0:
                                inner = text[i+5:j]
                                result.append("\\sqrt{" + _replace_sqrt(inner) + "}")
                                i = j + 1
                                break
                        j += 1
                    else:
                        result.append(text[i])
                        i += 1
                else:
                    result.append(text[i])
                    i += 1
            return "".join(result)

        s = _replace_sqrt(s)
        # a/b  →  \frac{a}{b}  for simple numerator/denominator
        m = _re.fullmatch(r'(.+)/(\w+)', s)
        if m:
            s = f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}"
        # Convert known physics names (hbar, mu, etc.) to LaTeX commands.
        # Done here while names are still plain ASCII (before super() runs).
        for plain, cmd in GREEK_CMD.items():
            if plain in s:
                s = _re.sub(r'(?<![A-Za-z\\])' + plain + r'(?![A-Za-z])',
                            cmd.replace('\\', r'\\'), s)
        return s

    def clean_latex(self, latex: str) -> str:
        import re
        s = re.sub(r'\s*~\s*', r'\\,', latex)
        # Expand \mathtt{name} emitted by Latexify.jl for Symbolics scalar variables.
        # Handles three cases:
        #   \mathtt{\omega}      → \omega      (already a TeX command inside)
        #   \mathtt{\omega_{0}}  → \omega_{0}  (TeX command with subscript)
        #   \mathtt{omega\_0}    → \omega_{0}  (plain name with escaped underscore)
        #   \mathtt{omega}       → \omega      (plain Greek name)
        _MATHTT_GREEK = {
            "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma",
            "delta": r"\delta", "epsilon": r"\epsilon", "zeta": r"\zeta",
            "eta": r"\eta", "theta": r"\theta", "iota": r"\iota",
            "kappa": r"\kappa", "lambda": r"\lambda", "mu": r"\mu",
            "nu": r"\nu", "xi": r"\xi", "pi": r"\pi", "rho": r"\rho",
            "sigma": r"\sigma", "tau": r"\tau", "upsilon": r"\upsilon",
            "phi": r"\phi", "chi": r"\chi", "psi": r"\psi", "omega": r"\omega",
            "Gamma": r"\Gamma", "Delta": r"\Delta", "Theta": r"\Theta",
            "Lambda": r"\Lambda", "Xi": r"\Xi", "Pi": r"\Pi",
            "Sigma": r"\Sigma", "Phi": r"\Phi", "Psi": r"\Psi", "Omega": r"\Omega",
        }
        def _expand_mathtt(m: re.Match) -> str:
            content = m.group(1)
            if content.startswith("\\"):
                return content          # already a TeX command
            if r"\_" in content:
                base, sub = content.split(r"\_", 1)
                base_tex = _MATHTT_GREEK.get(base, base)
                return f"{base_tex}_{{{sub}}}"
            return _MATHTT_GREEK.get(content, content)
        s = re.sub(r'\\mathtt\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', _expand_mathtt, s)
        # QuantumAlgebra emits operator names as {alpha}, {beta}, etc. (plain text groups).
        # Convert to proper LaTeX Greek commands so they render as α, β, etc.
        _GREEK = {
            "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma",
            "delta": r"\delta", "epsilon": r"\epsilon", "zeta": r"\zeta",
            "eta": r"\eta", "theta": r"\theta", "iota": r"\iota",
            "kappa": r"\kappa", "lambda": r"\lambda", "mu": r"\mu",
            "nu": r"\nu", "xi": r"\xi", "pi": r"\pi", "rho": r"\rho",
            "sigma": r"\sigma", "tau": r"\tau", "upsilon": r"\upsilon",
            "phi": r"\phi", "chi": r"\chi", "psi": r"\psi", "omega": r"\omega",
            "Alpha": r"\Alpha", "Beta": r"\Beta", "Gamma": r"\Gamma",
            "Delta": r"\Delta", "Theta": r"\Theta", "Lambda": r"\Lambda",
            "Xi": r"\Xi", "Pi": r"\Pi", "Sigma": r"\Sigma",
            "Phi": r"\Phi", "Psi": r"\Psi", "Omega": r"\Omega",
        }
        for name, cmd in _GREEK.items():
            # Replace {name} → {cmd} (e.g. {alpha} → {\alpha}) so it renders as Greek.
            # Use a function replacement to avoid backslash interpretation in repl string.
            replacement = '{' + cmd + '}'
            s = re.sub(r'\{' + re.escape(name) + r'\}',
                       lambda m, r=replacement: r, s)

        # Convert XStar → X^{\ast} (complex conjugate emitted by JuliaCodeGenerator
        # to avoid Julia's conj() simplifying real variables to themselves).
        # Apply Greek conversion so e.g. DeltaStar → \Delta^{\ast}.
        def _restore_conj(m: re.Match) -> str:
            base = m.group(1)
            display = _GREEK.get(base, base)
            return f"{display}^{{\\ast}}"
        s = re.sub(r'([A-Z][A-Za-z]*)Star\b', _restore_conj, s)

        return s

    def format_constant_for_latex(self, token: str) -> str:
        return token

    _DAGGER_INPUT_RE = re.compile(r'(\w)\s*\^\{?\\(?:dagger|dag)\}?')
    #: A NAME carrying a trailing prime.  The name is captured whole (with its
    #: leading LaTeX backslash, if any) because whether the prime is an ADJOINT
    #: is a question about the NAME — see `_prime_to_dagger`.
    _PRIME_ADJOINT_RE = re.compile(r"(?P<name>\\?[A-Za-z][A-Za-z0-9_]*)'")


    #: Unicode arrows → their LaTeX command, made ONE spelling at the entrance.
    #: The converter reads a backslash as an index boundary; a bare ``↑`` is not
    #: one, so ``c_{1↑}`` would come out as the single index ``c_1↑``.
    _ARROW_GLYPHS: "dict[str, str]" = {'↑': r'\uparrow', '↓': r'\downarrow'}

    def latex_to_canonical(self, latex: str) -> str:
        from notebook.runtime.julia_latex_converter import JuliaLatexConverter
        # **The arrow is replaced AFTER the LaTeX conversion, not before.**
        #
        # Measured 2026-08-22: replacing it first erased the separator it WAS.
        # `c_{1\uparrow}` — the most ordinary spelling in condensed matter —
        # became `c_{1u}`, one index named `1u`, which implicit multiplication
        # then read as `1 * u`, and the operator was refused with a message
        # about multi-character index names.  A backslash is an index boundary
        # (CLAUDE.md, the `g_{\mu\nu}` rule); substituting a bare letter for a
        # command destroys the boundary, exactly as `_separate_adjacent_commands`
        # was written to prevent in the other direction.
        for glyph, cmd in self._ARROW_GLYPHS.items():
            latex = latex.replace(glyph, cmd)          # ↑ → \uparrow (a boundary)
        # **整数スピンの射影は、矢印と同じ扱いをここで受ける。** 矢印はコマンドが
        # 境界なので変換器に任せられるが、素の符号は境界を持たないので、名前へ
        # 置き換えるのはここ——変換器の手前——である（同じ位置、同じ理由）。
        latex = self._projection_signs_to_names(latex)
        # Normalise dagger notation to prime before general LaTeX conversion
        latex = self._DAGGER_INPUT_RE.sub(r"\1'", latex)
        canonical = JuliaLatexConverter().to_canonical(latex)
        from notebook.runtime.spin_label_canon import canon_spin_arrows
        canonical = canon_spin_arrows(canonical)       # \uparrow → the ASCII name
        return self._apply_spin_component_norm(canonical)

    def _integer_spin_operator_names(self) -> "set[str]":
        """整数スピン（＝ボソン）として宣言された演算子の名前。

        兄弟の `quantum_operator_names` の隣に在る（同じ問いの一段狭い版）。
        """
        from notebook.runtime.julia_handler_common import integer_spin_operator_names
        return integer_spin_operator_names(self._context)

    def _projection_signs_to_names(self, latex: str) -> str:
        """コンマ区切り／末尾の**素の符号**を射影の名前にする。

        規則そのものは表の隣（`spin_label_canon.projection_signs_to_names`）に在る
        ——「further index-spelling discriminators go here, beside the table they
        read」。ここが持っているのは**このセッションの宣言**だけである。
        """
        from notebook.runtime.spin_label_canon import projection_signs_to_names
        return projection_signs_to_names(latex, self._integer_spin_operator_names())

    _SPIN_COMP_SUFFIX: "dict[str, str]" = {
        'x': 'x', 'y': 'y', 'z': 'z',
        'X': 'x', 'Y': 'y', 'Z': 'z',
        '+': 'p', '-': 'm',
    }

    def _apply_spin_component_norm(self, s: str) -> str:
        """Convert S^z → Sz etc. for registered spin operator components.

        Only fires when the compound name (e.g. Sz) is already registered in
        SymbolContext, so unrelated X^y patterns are never touched.
        Two orderings handled:
          S^z(1)  — superscript-first (from S^z_1)
          S(1)^z  — subscript-first  (from S_1^z)
        """
        if not self._context:
            return s
        ctx = self._context
        sfx = self._SPIN_COMP_SUFFIX

        def _record_style(base: str, style: str) -> None:
            """Record the user's spin display style on first use (= notation declaration)."""
            if base not in ctx.spin_display_styles:
                ctx.spin_display_styles[base] = style

        def _sup(m: "re.Match[str]") -> str:
            base, char = m.group(1), m.group(2)
            suf = sfx.get(char)
            if suf and ctx.get(base + suf):
                _record_style(base, "superscript")
                return base + suf
            return m.group(0)

        def _sub_first(m: "re.Match[str]") -> str:
            base, idx, char = m.group(1), m.group(2), m.group(3)
            suf = sfx.get(char)
            if suf and ctx.get(base + suf):
                _record_style(base, "superscript")
                return f"{base}{suf}({idx})"
            return m.group(0)

        # Subscript-style S_x / S_y / S_z — flat underscore component notation.
        # Convert to suffix form (Sx, Sy, Sz) so Julia receives the correct identifier.
        # Only fires when the compounded name (e.g. Sx) is registered.
        def _underscore_comp(m: "re.Match[str]") -> str:
            base, char = m.group(1), m.group(2)
            suf = sfx.get(char)
            if suf and ctx.get(base + suf):
                _record_style(base, "suffix")
                return base + suf  # S_x → Sx
            return m.group(0)
        s = re.sub(r'([A-Za-z][A-Za-z0-9_]*)_([xyzmXYZ])(?=[^A-Za-z0-9_]|$)', _underscore_comp, s)

        # bare: S^y_1  or  S^y (no site)
        s = re.sub(r'([A-Za-z][A-Za-z0-9_]*)\^([xyzmXYZ+\-])', _sup, s)
        # braced component: S^{y}_1  or  S^{y}_{1}
        s = re.sub(r'([A-Za-z][A-Za-z0-9_]*)\^\{([xyzmXYZ+\-])\}', _sup, s)
        # function-call site: S(1)^y
        s = re.sub(r'([A-Za-z][A-Za-z0-9_]*)\(([^)]+)\)\^([xyzmXYZ+\-])', _sub_first, s)
        s = re.sub(r'([A-Za-z][A-Za-z0-9_]*)\(([^)]+)\)\^\{([xyzmXYZ+\-])\}', _sub_first, s)

        # Detect suffix-style usage (Sy_k, Sz(1) etc.) — no conversion needed,
        # but record style so the display bridge uses suffix form.
        for name in ctx.registry:
            for char in sfx:
                if (name.endswith(char) and len(name) > len(char)
                        and re.search(r'\b' + re.escape(name) + r'[\(_]', s)):
                    _record_style(name[:-len(char)], "suffix")

        return s

    def _normalize_spin_component_latex(self, s: str, registry: "SymbolContext") -> str:
        r"""Convert spin-component LaTeX to match the user's input convention.

        Both the *user's input* and *QA's output* can express spin operators in
        two orderings of superscript and subscript.  All eight combinations
        (braced/bare × two orderings) are normalised to ``Sc_{site}`` form,
        where ``c`` is the component letter (x/y/z) appearing as a suffix::

            Site-first  (QA default output):
              S_1^y   →  Sy_1       S_{1}^y  →  Sy_1
              S_1^{y} →  Sy_1       S_{1}^{y} → Sy_{1}

            Component-first (user's S^y_1 input style):
              S^y_1   →  Sy_1       S^y_{1}  →  Sy_{1}
              S^{y}_1 →  Sy_1       S^{y}_{1} → Sy_{1}

        Only converts when the compound name (e.g. ``Sy``) is registered in
        *registry* so that unrelated patterns are never touched.
        Works for any spin operator base name (not just ``S``).
        """
        if not registry:
            return s
        sfx = self._SPIN_COMP_SUFFIX  # {"x": "x", "y": "y", "z": "z", ...}

        def _make_pattern(base: str) -> "re.Pattern[str]":
            b = re.escape(base)
            # QuantumAlgebra wraps single-letter operator names in braces:
            # e.g. Sy(1) → {S}_{1}^{y}.  Accept both {BASE} and BASE.
            bp = rf'(?:\{{{b}\}}|{b})'

            # Two orderings, each with braced and bare site/component variants.
            # Ordering 1: {BASE}_{site}^{comp} or BASE_{site}^{comp}  (QA LaTeX output)
            p_site_first = (
                rf'{bp}_\{{([^}}]+)\}}\^\{{([xyz])\}}'  # {B}_{site}^{c}
                rf'|{bp}_\{{([^}}]+)\}}\^([xyz])'        # {B}_{site}^c
                rf'|{bp}_(\w+)\^\{{([xyz])\}}'           # {B}_site^{c}
                rf'|{bp}_(\w+)\^([xyz](?![A-Za-z0-9_]))'# {B}_site^c
            )
            # Ordering 2: {BASE}^{comp}_{site} or BASE^{comp}_{site} (user's S^y_1 style)
            p_comp_first = (
                rf'{bp}\^\{{([xyz])\}}_\{{([^}}]+)\}}'  # {B}^{c}_{site}
                rf'|{bp}\^\{{([xyz])\}}_(\w+)'           # {B}^{c}_site
                rf'|{bp}\^([xyz])_\{{([^}}]+)\}}'        # {B}^c_{site}
                rf'|{bp}\^([xyz])_(\w+)'                  # {B}^c_site (bare)
            )
            return re.compile(p_site_first + '|' + p_comp_first)

        # Collect unique base names from registered spin-component operators
        bases_seen: set[str] = set()
        for name in registry.registry:
            for char in sfx:
                if name.endswith(char):
                    base = name[:-len(char)]
                    if base:
                        bases_seen.add(base)

        for base in bases_seen:
            pat = _make_pattern(base)
            style = registry.spin_display_styles.get(base, "suffix")

            def _reorder(m: re.Match, _base: str = base, _style: str = style) -> str:
                g = m.groups()
                n = len(g)
                for i in range(0, n, 2):
                    a, b_ = g[i], g[i + 1]
                    if a is None or b_ is None:
                        continue
                    # First 8 groups: (site, comp); last 8: (comp, site)
                    site, comp = (a, b_) if i < 8 else (b_, a)
                    suf = sfx.get(comp)
                    if suf and registry.get(_base + suf):
                        site_tex = f"{{{site}}}" if len(site) > 1 else site
                        if _style == "superscript":
                            # S^y_1 form: component as superscript, site as subscript
                            return f"{_base}^{comp}_{site_tex}"
                        else:
                            # Sy_1 form: component as suffix (default)
                            return f"{_base}{suf}_{site_tex}"
                return m.group(0)

            s = pat.sub(_reorder, s)

        return s

    #: Letters a summation dummy may be given, in the order physics writes them.
    #: Used only for a dummy the BACKEND introduced — the researcher's own names
    #: are recovered from the generator's output and never invented.
    _DUMMY_NAME_POOL = ("i", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t")

    def _fresh_dummy_names(self, text: str, taken: "list[str]",
                           registry=None) -> "list[str]":
        r"""Names for a dummy the BACKEND introduced — **spending no new letters**.

        A bound index is a label that lives inside one sum and means nothing
        outside it.  Handing it a fresh letter each time therefore costs the
        researcher something real for nothing: after `m, n, p, q` came `r`, then
        `s`, then `t`, and the alphabet they still had available for actual
        quantities shrank with every commutator.  (Raised by the researcher,
        2026-08-08, after seeing a declared `r` — a radius — taken by a dummy.)

        So the extra indices are the researcher's OWN first index letter,
        subscripted: `m_1`, `m_2`, …  It reads as "more of these", stays inside
        their notation, cannot run out, and consumes nothing.  It round-trips: a
        digit subscript is part of the name (CLAUDE.md) and QuantumAlgebra takes
        a multi-digit subscript on a single-character base (measured, open_items
        SUM-DUMMY-NAMES).

        Letters are used only when there is no written index to build on at all
        — a sum the backend produced from nothing.  Even then, anything already
        spoken for is skipped: the SESSION's declarations, and every index
        visible in a subscript (the spin label `u` in `c_{m,u}` is an index too).
        Letters inside LaTeX commands are not scanned, so `\sum` does not consume
        `s`, `u` and `m`.
        """
        used = set(taken)
        if registry is not None:
            used.update(getattr(registry, "registry", None) or {})
        # Every single letter standing on its own is an index in play — the spin
        # label `u` in `c_{m,u}`, the free `i` in `\delta_{i\#_{1}}`.  Scanned on
        # the text with LaTeX COMMANDS removed first, so `\sum` does not consume
        # `s`, `u` and `m` while `\delta_{i…}` still yields its `i`.  Scanning
        # brace groups instead missed the nested case entirely (measured).
        _scrubbed = re.sub(r"\\[A-Za-z]+", " ", text)
        used.update(re.findall(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", _scrubbed))

        bases = [b for b in taken if re.fullmatch(r"[A-Za-z]", b)]
        if not bases:
            # Nothing written to build on — a sum the backend produced from
            # nothing (a δ contraction).  A letter is chosen as the BASE, but it
            # is still subscripted: `\sum_{k_1}`, not `\sum_k`.  The researcher's
            # call (2026-08-08), and the reason is the same frugality —
            # `\sum_k \delta_{ik}\delta_{kj}` is what a paper prints, but it
            # would take the plain `k` away from them, and which letters are
            # "free" is not something this layer can know: the ones missing from
            # THIS expression are missing by accident, and researchers use
            # `i`, `j`, `k` as ordinary indices everywhere else.  A subscript
            # says "bound index" on sight and leaves every plain letter alone.
            free = [c for c in self._DUMMY_NAME_POOL if c not in used]
            bases = free[:1] or ["k"]
        # CYCLE the written letters before raising the subscript — `m_1, n_1,
        # p_1, q_1, m_2, …` rather than `m_1, m_2, m_3`.  Keeps the numbers
        # small, which is what a reader has to hold in their head.
        return [f"{base}_{{{order}}}"
                for order in range(1, 40)
                for base in bases
                if f"{base}_{order}" not in used]

    def _prime_to_dagger(self, m: "re.Match") -> str:
        r"""A prime is an ADJOINT only on this session's operator names.

        `a'(k)` is a creation operator; `t'`, `\sigma'` and `\mathbf r'` are the
        researcher's mark for a DISTINCT VARIABLE — the reading
        ``latex_converter`` already gives a primed index.  This substitution
        made no distinction, so in a Julia session every prime came back a
        dagger and the most ordinary spellings in the Green's-function family
        were shown as something else (PRIME-DISPLAY-DAGGER, 2026-09-10)::

            G(t, t')                  →  G(t, t^{\dagger})
            \psi^\dagger_{\sigma'}(x')  →  \psi^{\dagger}_{\sigma^{\dagger}}(x^{\dagger})

        **The declaration is the discriminator**, and the published Julia
        sessions say so: every primed operator base there (`c`, `g`, `a`, `b`)
        carries ``kind == "operator"`` and `sigma` is declared nowhere.  Where
        nothing is declared, the researcher's own spelling stands — a display
        that invents an adjoint is worse than one that shows the prime typed.
        """
        name = m.group("name")
        if name.lstrip("\\") in self._display_operator_names():
            return f"{name}^{{\\dagger}}"
        return m.group(0)

    def _display_operator_names(self) -> "set[str]":
        """Operator names for the display, asked through the one helper."""
        from notebook.runtime.julia_handler_common import quantum_operator_names
        return quantum_operator_names(getattr(self, "_context", None))

    def normalize_display_latex(self, latex: str) -> str:
        # Convert SymPy canonical form FIRST (while hbar/mu are still plain
        # text), then let super() replace hbar→\hbar, mu→\mu, etc.
        latex = self._sympy_canonical_to_latex(latex)
        # The adjoint is decided WHILE THE NAME IS STILL A NAME — before super()
        # rewrites `psi_hat'` into `\hat{\psi}'`, after which the prime no
        # longer follows an identifier and nothing could read it.  Measured: a
        # DECORATED creation operator (`\hat\psi^\dagger`, the ordinary field
        # operator) displayed with a bare prime for exactly that reason.  Same
        # move as `_subscript_separator` and `_conjugate_mark`, one door earlier.
        latex = self._PRIME_ADJOINT_RE.sub(self._prime_to_dagger, latex)
        latex = super().normalize_display_latex(latex)
        # A spin projection is a GLYPH in QuantumAlgebra's own printing, and a
        # bare `↑` / `⁺` is not LaTeX — published through uplatex it does not
        # compile. The glyph is the CANONICAL's business, never the display's.
        #
        # **Latexify writes a superscript glyph as `\^<char>`** (measured
        # 2026-09-12: `⁺` comes back as `{\^+}`), and `\^+` in math mode is the
        # circumflex ACCENT applied to a plus — not a superscript, and not
        # something that typesets. Integer projections therefore go back to the
        # plain sign, which is how physics writes them ($a_{1+}$). The spellings
        # live in ONE table beside the labels themselves (`PROJECTION_LATEX`).
        from notebook.runtime.spin_value_set import PROJECTION_LATEX
        for glyph, cmd in PROJECTION_LATEX.items():
            if not cmd.startswith("\\"):        # 素の符号は `\^+` の形で来る
                latex = latex.replace("\\^" + cmd, cmd)
            latex = latex.replace(glyph, cmd)
        # **区切りは、入力側が境界を見つけられないときにだけ印刷する。** 矢印は
        # コマンドが境界なので要らない。素の符号には要る——無いと m=0 が `b_{10}`
        # と印刷されて**サイト10と曖昧**になり、打ち返すと生の `ParseError:` が
        # 返っていた（実測 2026-09-12、CLAUDE.md「A PRINTED SPELLING MUST TYPE
        # BACK」。読む側は `spin_label_canon.projection_signs_to_names`）。
        latex = self._PROJECTION_SEP_RE.sub(r"\1{,\2}", latex)
        return latex

    #: 添字の中で、何かに続く**素の符号**（`_{1{+}}` の `{+}`）。先頭に立つとき
    #: （`_{{+}}`）は区切る相手がいないので触らない。
    _PROJECTION_SEP_RE = re.compile(r"(_\{[^{}]+)\{([+0-])\}")

    def cas_repr(self, cas_value: str, canonical: str) -> str | None:
        # When canonical is a _nb_ variable reference, preserve the Julia
        # string(result) as a human-readable record of the actual expression.
        if canonical.startswith("_nb_") and cas_value and cas_value != canonical:
            return cas_value
        return None

    # ------------------------------------------------------------------
    # Particle operator definitions (Phase 0.5 hook)
    # ------------------------------------------------------------------

