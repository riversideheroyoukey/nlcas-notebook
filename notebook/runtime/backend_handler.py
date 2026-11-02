"""Abstract base class and shared types for backend-specific logic.

``BackendHandler`` is an ABC that each backend implements.
``NotebookEngine`` holds ``self._handler: BackendHandler`` and delegates all
backend-specific decisions here, keeping ``engine.py`` free of
Maxima/Julia conditionals.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

from notebook.models.quantity import Quantity
from notebook.runtime.nl_patterns import nlp, nlp_any, nlp_search
from notebook.runtime.number_words import number_literal
from notebook.runtime.operand_boundary import split_name_list
from notebook.runtime.type_label import (
    declaration_operation_for_label, declaration_unsupported_here,
    symmetry_value_for_label)
from notebook.runtime.operand_guards import TYPE_DECL_ALGEBRA

#: 宣言された名前の綴り（区切りで切ったあとの1片から拾う形）。
_DECL_NAME_RE = r"[A-Za-z_Α-ω][A-Za-z0-9_Α-ω]*(?:_[A-Za-z0-9]+)*"
from notebook.models.symbol_context import SymbolContext
from notebook.runtime.latex_converter import (
    TEX_GREEK_TO_NAME, _SCRIPT_DECORATIONS)
from notebook.runtime.indexed_display import (
    record_greek_glyphs, restore_greek_glyphs, restore_indexed_bases,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from notebook.models.ir import Expr
    from notebook.runtime.expr_lowering import Lowered



# ---------------------------------------------------------------------------
# Lattice / graph pure functions moved to lattice_ops.py (R-10a, 2026-07-04).
# Re-exported here so existing `from notebook.runtime.backend_handler import
# build_dispersion, ...` call sites keep working unchanged.
# ---------------------------------------------------------------------------
from notebook.runtime.lattice_ops import (  # noqa: F401
    _BOND_GENERATORS,
    register_bond_generator,
    _shape_or_square,
    _bonds_nearest_neighbor,
    _bonds_all_pairs,
    _periodic_xy,
    _bonds_square_2d,
    _bonds_triangular_2d,
    _bonds_honeycomb_2d,
    _parse_edges,
    _CRYSTAL_1BAND,
    _CRYSTAL_MB,
    _CRYSTAL_NBAND,
    _SINGLE_TOKEN_COEFF,
    _k_dot_delta,
    _fold_two_coeff,
    build_dispersion_single_band,
    _abs_coeff,
    _negate_disp,
    _sq,
    _cos_of_diff,
    _shell_term,
    _link_inner,
    _format_abs_sq,
    _combine_coeffs,
    build_dispersion_multiband,
    build_dispersion_nband,
    build_dispersion,
    _exp_term,
    _join_signed,
    _diag_entry,
    _offdiag_entry,
    _weighted_abs_sq,
    _KVARS,
    build_crystal_bloch,
    build_adjacency_matrix,
    adjacency_to_canonical,
    adjacency_to_latex,
    _matmul,
    graph_degree_sequence,
    graph_triangle_count,
    graph_walk_trace,
    graph_edge_count,
    graph_four_cycle_count,
    _GRAPH_IDENTITIES,
    graph_identity_statements,
    graph_identity_check,
    build_neighbor_sum_terms,
)



# Unicode Greek letter → TeX command mapping (for input normalization).
_UNICODE_GREEK_TO_TEX: dict[str, str] = {
    "α": r"\alpha",   "β": r"\beta",    "γ": r"\gamma",   "Γ": r"\Gamma",
    "δ": r"\delta",   "Δ": r"\Delta",   "ε": r"\epsilon", "ζ": r"\zeta",
    "η": r"\eta",     "θ": r"\theta",   "Θ": r"\Theta",   "ι": r"\iota",
    "κ": r"\kappa",   "λ": r"\lambda",  "Λ": r"\Lambda",  "μ": r"\mu",
    "ν": r"\nu",      "ξ": r"\xi",      "Ξ": r"\Xi",      "π": r"\pi",
    "Π": r"\Pi",      "ρ": r"\rho",     "σ": r"\sigma",   "Σ": r"\Sigma",
    "τ": r"\tau",     "υ": r"\upsilon", "Υ": r"\Upsilon", "φ": r"\phi",
    "Φ": r"\Phi",     "χ": r"\chi",     "ψ": r"\psi",     "Ψ": r"\Psi",
    "ω": r"\omega",   "Ω": r"\Omega",
    # Spin arrows travel with the Greek letters for the same reason: the LaTeX
    # converter reads a BACKSLASH as an index boundary, so `c_{1↑}` written with
    # the bare glyph came out as the single index `c_1↑` (and was then refused
    # as an unreadable character), while `c_{1\uparrow}` split correctly.
    # Measured 2026-08-22.
    "↑": r"\uparrow", "↓": r"\downarrow",
}

_UNICODE_GREEK_RE = re.compile("|".join(re.escape(c) for c in _UNICODE_GREEK_TO_TEX))


def normalize_unicode_greek(text: str) -> str:
    """Replace Unicode Greek letters with their TeX commands.

    e.g. ``"ω を実数パラメータとして定義して"`` → ``"\\omega を実数パラメータとして定義して"``

    Unicode subscript digits (₀₁₂…) are NOT converted — users should use
    TeX subscript notation (e.g. ``\\omega_0``) instead.

    The spin arrows ``↑``/``↓`` ride along: they are input glyphs with a LaTeX
    command spelling, exactly like the Greek letters.
    """
    return _UNICODE_GREEK_RE.sub(lambda m: _UNICODE_GREEK_TO_TEX[m.group()], text)


# Greek letter canonical name → LaTeX command mapping.
# Shared by all backends that render CAS-defined symbols in the timeline.
GREEK_CMD: dict[str, str] = {
    "alpha":      r"\alpha",
    "beta":       r"\beta",
    "gamma":      r"\gamma",    "Gamma":    r"\Gamma",
    "delta":      r"\delta",    "Delta":    r"\Delta",
    "epsilon":    r"\epsilon",  "varepsilon": r"\varepsilon",
    "zeta":       r"\zeta",
    "eta":        r"\eta",
    "theta":      r"\theta",    "Theta":    r"\Theta",
    "vartheta":   r"\vartheta",
    "iota":       r"\iota",
    "kappa":      r"\kappa",
    "lambda":     r"\lambda",   "Lambda":   r"\Lambda",
    "mu":         r"\mu",
    "nu":         r"\nu",
    "xi":         r"\xi",       "Xi":       r"\Xi",
    "pi":         r"\pi",       "Pi":       r"\Pi",
    "varpi":      r"\varpi",
    "rho":        r"\rho",      "varrho":   r"\varrho",
    "sigma":      r"\sigma",    "Sigma":    r"\Sigma",
    "tau":        r"\tau",
    "upsilon":    r"\upsilon",  "Upsilon":  r"\Upsilon",
    "phi":        r"\phi",      "Phi":      r"\Phi",
    "varphi":     r"\varphi",
    "chi":        r"\chi",
    "psi":        r"\psi",      "Psi":      r"\Psi",
    "omega":      r"\omega",    "Omega":    r"\Omega",
    "hbar":       r"\hbar",
}


def canonical_name_to_latex(name: str, context=None) -> str:
    """Convert a bare canonical name (e.g. 'xi', 'omega_0') to its LaTeX form.

    Pure, no backend-instance state required — takes the SymbolContext (or
    None) directly, so non-handler callers (e.g. the numeric layer's
    templates.py, which only has a SymbolContext, not a BackendHandler) can
    render the same Greek/subscript display the rest of the UI uses instead
    of re-deriving their own name→display mapping.  Checks the registered
    quantity's explicit *latex_name* first, then falls back to the
    module-level Greek letter table ``GREEK_CMD``.  Single-letter names and
    unknown names are returned unchanged.
    """
    qty = context.get_quantity(name) if context is not None else None
    if qty and getattr(qty, "latex_name", None):
        return qty.latex_name
    # Spin component suffixes: Sx→S^x, Sy→S^y, Sz→S^z, Sp→S^+, Sm→S^-
    # Gated on BOTH the compound name and its stem being registered as
    # operators (kind=="operator", as set by define_spin_operator's
    # base + component auto-registration) — a bare name-shape match
    # (e.g. a scalar parameter "mz") must never be reinterpreted as a
    # spin component just because it happens to end in x/y/z/p/m.
    _SPIN_SUFFIX: dict[str, str] = {"x": "x", "y": "y", "z": "z", "p": "+", "m": "-"}
    if (len(name) >= 2 and name[-1] in _SPIN_SUFFIX and name[:-1].isalpha()
            and context is not None and qty is not None and qty.kind == "operator"):
        stem = name[:-1]
        stem_qty = context.get_quantity(stem)
        if stem_qty is not None and stem_qty.kind == "operator":
            suf = name[-1]
            stem_latex = GREEK_CMD.get(stem, stem)
            return f"{stem_latex}^{{{_SPIN_SUFFIX[suf]}}}"

    # 装飾（`H_hat_0`）と上付きラベル（`G_sup0`）は、**結果の表示が既に持っている
    # 復元**にそのまま任せる。ここが独自に答えていたため、宣言だけが内部の綴りを
    # 出していた: 公開 TeX に `\[ H_hat_0 \]` ——TeX は二重下付きで**組版に失敗する**
    # （実測 2026-08-22、利用者報告）。`psi_hat_sigma` は `\psi_{hat_sigma}` になり、
    # 装飾が添字に化けていた（CLAUDE.md が `phi_bar` で記録しているのと同じ形）。
    from notebook.runtime.latex_converter import restore_decorations
    restored = restore_decorations(name)
    if restored != name:
        return restored
    base = name.split("_", 1)[0]
    if base in GREEK_CMD:
        suffix = name[len(base):]
        if suffix.startswith("_"):
            return f"{GREEK_CMD[base]}_{{{suffix[1:]}}}"
        return GREEK_CMD[base]
    return name


# (?<!\\) excludes names already behind a backslash (e.g. "\Gamma" typed
# directly by the user must not become "\\Gamma").  The optional "(_\d+)?"
# tail catches a DIGIT-subscripted Greek name (e.g. "omega_0") as ONE token
# — without it, "\bomega\b" never matches inside "omega_0" (there is no
# word-boundary between the letter "a" and the underscore, since "_" is a
# \w character), so gamma rendered but omega_0 silently didn't (found via
# the numeric-layer figure-caption work, 2026-07-09).  Digit-only, deliberately
# NOT "\w+": an alpha suffix ("sigma_x") is registration-gated elsewhere
# because it's ambiguous (component index vs. part of the name — see
# CLAUDE.md "Subscript Classification"); only a digit suffix is UNAMBIGUOUSLY
# part of the name in every case, so this pure-text beautifier can convert it
# without consulting the symbol registry.  Matches canonical_name_to_latex's
# own base+subscript split for the digit case.
#: A Greek name typed in ASCII, so the display can give it its LaTeX command.
#: Three endings: a subscript, a BRACED subscript, or nothing.  Only `phi` and
#: `phi_1` used to convert — `phi_{m,s}` and `xi_k` did not, `phi` and `_` having
#: no word boundary between them — so a spin-dependent basis function reached the
#: published `.tex` as the italic WORD "phi" (measured 2026-08-09, the same shape
#: as the `\dot{theta}` EXAMPLES-DRIFT finding).
#:
#: The subscript may be written with an underscore or GLUED ON (`omega0`), the
#: second being safe only because both doors that name a symbol — the
#: declaration and an H-1 left-hand side — split it first
#: (`split_greek_digit_name`), so what is displayed is a name that can be typed
#: back.  Display alone would have shown ω₀ for a symbol still called `omega0`.
#:
#: `phi_bar` — H-1's spelling of `\bar{\phi}` — is safe here only because
#: `restore_decorated_name` runs FIRST and has already turned it into
#: `\bar{\phi}`; the backslash then stops this pass.  Reorder those two and φ
#: comes back subscripted with the word "bar".
_BARE_GREEK_RE = re.compile(
    r"(?<!\\)\b(" + "|".join(sorted(GREEK_CMD, key=len, reverse=True))
    + r")(?:_(?P<sub>[A-Za-z0-9]+)\b(?P<sub_primes>\u0027*)|(?P<glued>\d+)\b|(?=_\{)|\b)"
)


#: A Greek name used as a SUBSCRIPT — `p_theta` is p_θ.  The pass above cannot
#: see it: `_` is a word character, so there is no boundary before the name.
#: Only the bare form converts (`p_theta2` is left alone), because a second
#: subscript on the letter would nest two `_` in one group.
#: A PRIME IS PART OF THE INDEX NAME, so it is swallowed into the group rather
#: than left outside the braces.  `latex_converter` already reads it that way —
#: "an index token may carry primes (`\sigma'`, `k'`): a prime marks a distinct
#: variable" — and the display did not, so the field operator
#: `\psi^\dagger_{\sigma'}(x')` came back as `\psi^{\dagger}_{\sigma}'(x')`,
#: the prime detached from the index it belongs to (PRIME-DISPLAY-DAGGER,
#: 2026-09-10).
_GREEK_SUBSCRIPT_RE = re.compile(
    r"(?<=_)(" + "|".join(sorted(GREEK_CMD, key=len, reverse=True))
    + r")(?![A-Za-z0-9])(?P<primes>'*)"
)


def _greek_subscript_sub(m: "re.Match[str]") -> str:
    r"""`_sigma` → `_{\sigma}`, keeping any prime inside the braces."""
    return '{' + GREEK_CMD[m.group(1)] + (m.group("primes") or "") + '}'


def _bare_greek_sub(m: "re.Match[str]") -> str:
    r"""`psi` → `\psi`, and `psi_sigma` → `\psi_{\sigma}`.

    **THE SUBSCRIPT IS A NAME TOO.**  This pass reads "a Greek base with a glued
    subscript" itself, and emitted the subscript RAW — so the very next pass,
    `_GREEK_SUBSCRIPT_RE`, never saw it and the answer depended on the BASE:
    `x_\sigma` displayed as `x_{\sigma}` and `\psi_\sigma` as `\psi_{sigma}`,
    the italic word (measured 2026-09-11).  One question, two places, and the
    one that fires first said no.  `\psi_\sigma` is the ordinary field-operator
    spelling, and this is the `\dot{theta}` shape: it typesets, as prose.
    """
    tex = GREEK_CMD[m.group(1)]
    sub = m.group("sub") or m.group("glued")
    if not sub:
        return tex
    # A prime is part of the index NAME, so it stays inside the braces — the
    # same rule `_GREEK_SUBSCRIPT_RE` follows (PRIME-DISPLAY-DAGGER).
    primes = m.group("sub_primes") or "" if m.group("sub") else ""
    return f"{tex}_{{{GREEK_CMD.get(sub, sub)}{primes}}}"


def _sqrt_call_to_radical(text: str) -> str:
    """Convert bare ``sqrt(...)`` function-call syntax to ``\\sqrt{...}``.

    Handles arbitrarily nested parens in the argument (recursing on the
    argument itself so a nested ``sqrt(...)`` or Greek name also converts).
    Leaves ``\\sqrt(...)`` / already-backslashed forms untouched.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if (text.startswith("sqrt(", i)
                and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] in "_\\"))):
            j = i + len("sqrt(")
            depth = 1
            start = j
            while j < n and depth:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1
            arg = text[start:j - 1] if depth == 0 else text[start:j]
            out.append(r"\sqrt{" + _sqrt_call_to_radical(arg) + "}")
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


#: ``^*`` / ``^{*}`` — COMPLEX CONJUGATION, the one `*` in a formula that is not
#: multiplication.  Every place with a `*` rule settles this first; one did not,
#: and its `*` → space substitution ate the star, writing an H-1 definition down
#: as ``… phi_{m,s}(x)^ T(x) …`` — a bare `^`, not LaTeX at all (measured
#: 2026-08-09, docs/scenarios/scenario_spinful_matrix_elements_jp.md).
_SUPERSCRIPT_STAR_RE = re.compile(r"\^\{\s*\*\s*\}|\^\*")


def normalize_superscript_star(text: str) -> str:
    r"""``x^*`` → ``x^{\ast}``.  Only a lone star: ``^{*k}`` is left alone."""
    return _SUPERSCRIPT_STAR_RE.sub(r"^{\\ast}", text)


def _conjugate_star_onto_name(latex: str) -> str:
    r"""``\varphi_{m,s}(x)^{\ast}`` → ``\varphi^{\ast}_{m,s}(x)``.

    Maxima's ``tex1`` writes a conjugated call with the star after the argument
    list.  Physics writes it on the FUNCTION: $\varphi^*_{m\sigma}(\mathbf x)$,
    and a matrix element is where that shows up most
    (docs/scenarios/scenario_spinful_matrix_elements_jp.md).

    A LaTeX convention, not a backend quirk, so it applies to what the
    RESEARCHER wrote as well: run only over Maxima's own output it moved the
    star on a result and left the input echo beside it reading
    `\phi_{m,s}(x)^{\ast}` — the same session publishing the same quantity two
    ways (measured 2026-08-09).

    Only a NAMED call moves.  ``(A+B)^{\ast}`` is a conjugated expression, not a
    function, and is left exactly as written — which is also why the argument
    list is read with BALANCED delimiters and both spellings (`\left(…\right)`
    and `(…)`, which Maxima mixes) rather than matched: `f(g(x))^{\ast}` has
    parentheses of its own and a regex would cut at the first one.
    """
    from notebook.runtime.latex_brackets import PAREN_PAIRS, balanced
    _NAME = r"(?:\\[A-Za-z]+|[A-Za-z][A-Za-z0-9]*)"
    _HEAD = re.compile(r"(?P<name>" + _NAME + r")(?P<sub>_\{[^{}]*\}|_[A-Za-z0-9]+)?$")
    _STAR = r"^{\ast}"
    out: list = []
    i = 0
    while i < len(latex):
        hit = None
        for open_tok, close_tok in PAREN_PAIRS:
            at = latex.find(open_tok, i)
            if at >= 0 and (hit is None or at < hit[0]):
                hit = (at, open_tok, close_tok)
        if hit is None:
            break
        at, open_tok, close_tok = hit
        span = balanced(latex, at, open_tok, close_tok)
        head = _HEAD.search(latex[i:at])
        if span is None or head is None or not latex[span[1]:].startswith(_STAR):
            out.append(latex[i:at + len(open_tok)])
            i = at + len(open_tok)
            continue
        out.append(latex[i:i + head.start()])
        out.append(head.group("name") + _STAR + (head.group("sub") or ""))
        out.append(latex[at:span[1]])
        i = span[1] + len(_STAR)
    out.append(latex[i:])
    return "".join(out)


#: The decorations H-1 folds into a NAME: `\bar{\phi}` is stored as `phi_bar`
#: (CLAUDE.md, "Decorated forms").  Display has to fold it back, or the accent
#: the researcher wrote returns as a subscript spelling out the word — a result
#: containing `phi_bar` printed as φ subscripted "bar" (measured 2026-08-09).
#: **波括弧は対で読む。** `\{?` と `\}?` を独立に任意にすると、閉じ波括弧が
#: 隣に在るだけで**それを食う**——`\sqrt{H_hat}` → `\sqrt{\hat{H}`、
#: `\frac{\partial H_hat}{\partial x}` → `\frac{\partial \hat{H}{\partial x}`
#: で、どちらも TeX として通らない（実測 2026-09-19。飾りのある名前を微分の変数に
#: 書けるようにしたら公開物の `.tex` が組版に失敗し、`test_tex_compiles` が捕まえた
#: ——静的な走査ではなく**実際に組む**検査が在る理由そのもの）。
_DEC_NAMES = r"hat|tilde|bar|dot|ddot|check|breve|acute|grave|dagger|cal|scr"
_DECORATED_NAME_RE = re.compile(
    r"(?<![\\A-Za-z])(?P<base>[A-Za-z][A-Za-z0-9]*)_"
    r"(?:\{(?P<decb>" + _DEC_NAMES + r")\}|(?P<dec>" + _DEC_NAMES + r"))"
    r"(?![A-Za-z])"
)


#: A Greek name with digits glued on — `omega0`, `phi0`.  Split at declaration
#: so the registered NAME carries the subscript (`omega_0`) and the display can
#: show ω₀ without the two disagreeing.
_GREEK_DIGIT_RE = re.compile(
    r"^(" + "|".join(sorted(GREEK_CMD, key=len, reverse=True)) + r")(\d+)$"
)


def split_greek_digit_name(name: str) -> str:
    """``omega0`` → ``omega_0``; anything else unchanged."""
    return _GREEK_DIGIT_RE.sub(r"\1_\2", name)


#: A superscript LABEL folded into the name (`G_sup0`, `Sigma_psup2`).  Open
#: ended — the value is any run of letters/digits — so it cannot join the fixed
#: decoration alternation above.
#: 波括弧は対で読む——理由は `_DECORATED_NAME_RE` の註のとおり（`{G_sup0}` →
#: `{G^{0}` になっていた）。
_LABELLED_NAME_RE = re.compile(
    r"(?<![\\A-Za-z])(?P<lbase>[A-Za-z][A-Za-z0-9]*)_"
    r"(?:\{(?P<lparenb>p?)sup(?P<lvalb>[A-Za-z0-9]+)\}"
    r"|(?P<lparen>p?)sup(?P<lval>[A-Za-z0-9]+))"
    r"(?![A-Za-z0-9])"
)


def restore_decorated_name(text: str) -> str:
    r"""``phi_bar`` / ``phi_{bar}`` → ``\bar{\phi}``; ``H_hat`` → ``\hat{H}``.

    The inverse of the H-1 registration convention, applied to what is written
    down.  Run BEFORE the bare-Greek pass so the stem is decorated rather than
    subscripted: `\bar{\phi}`, not `\phi_{bar}`.
    """
    def _sub(m: "re.Match[str]") -> str:
        base = GREEK_CMD.get(m.group("base"), m.group("base"))
        dec = m.group("dec") or m.group("decb")
        # The adjoint is written as a SUPERSCRIPT, not an accent — there is no
        # `\dagger{\psi}`.  Without this branch a creation operator came back
        # from the CAS displayed as `\psi_{dagger}`: the word, spelled out in
        # the subscript (measured 2026-08-20, the same shape as the `phi_bar`
        # case this function was written for).
        if dec in _SCRIPT_DECORATIONS:
            return f"{base}{_SCRIPT_DECORATIONS[dec]}"
        return f"\\{dec}{{{base}}}"

    def _label(m: "re.Match[str]") -> str:
        base = GREEK_CMD.get(m.group("lbase"), m.group("lbase"))
        value = m.group("lval") or m.group("lvalb")
        paren = m.group("lparen") or m.group("lparenb")
        return f"{base}^{{{'(' + value + ')' if paren else value}}}"

    # 上付きラベル（`G_sup0` → `G^{0}`）。装飾と違って値が開いているので別の走査。
    text = _LABELLED_NAME_RE.sub(_label, text)
    return _DECORATED_NAME_RE.sub(_sub, text)


#: 極限の添字の内部綴り。`_limit_direction`（変換器の入口）と
#: `_subscript_separator` が作る `{v,to,p}` / `{v,to,p,plus}` で、**この系の綴り
#: であって研究者の記法ではない**（原則#1）。表示の手前で書かれた形へ戻す。
_LIMIT_SUBSCRIPT_RE = re.compile(r"\\lim_\{([^{}]*)\}")
#: 片側の印 → 研究者の綴り。
_LIMIT_SIDE_MARK = {"plus": "^{+}", "minus": "^{-}"}


#: 極限の点として書かれる無限大。**この欄でだけ**戻す——素の `inf` は一般には
#: 研究者の変数名でありうるが、極限の点に `inf` という名前を付ける人はいない。
_LIMIT_POINT_WORDS = {"inf": "\\infty", "minf": "-\\infty", "und": "\\text{und}"}


def _limit_atom(text: str) -> str:
    r"""極限の添字に立つ1原子を、**書かれた綴り**に戻す。

    `t\'` は CAS がプライムを識別子に含めるための綴りであって研究者の記法では
    ないので、逃がしの `\` を落とす（原則#1）。
    """
    out = text.replace("\\'", "'")
    return _LIMIT_POINT_WORDS.get(out, out)


def limit_subscript_latex(var: str, point: str, direction: "str | None") -> str:
    r"""`(x, 0, None)` → `x \to 0`；`(t', t, "plus")` → `t' \to t^{+}`.

    ONE place says how a limit's subscript is written, because two restores need
    it: the canonical's internal `{v,to,p}` spelling and Maxima's own `tex1`,
    which writes a one-sided limit as `x\downarrow a`.  Written twice they would
    drift, which is the shape this area keeps producing.
    """
    return (f"{_limit_atom(var)} \\to {_limit_atom(point)}"
            f"{_LIMIT_SIDE_MARK.get(direction or '', '')}")


def restore_limit_subscript(s: str) -> str:
    r"""`\lim_{x,to,0}` → `\lim_{x \to 0}`；`\lim_{t',to,t,plus}` → `\lim_{t' \to t^{+}}`."""
    def _sub(m: "re.Match[str]") -> str:
        parts = [p.strip() for p in m.group(1).split(",")]
        direction = None
        if len(parts) == 4 and parts[3] in _LIMIT_SIDE_MARK:
            direction = parts.pop()
        if len(parts) != 3 or parts[1] != "to":
            return m.group(0)          # not this system's spelling — leave it
        return "\\lim_{" + limit_subscript_latex(parts[0], parts[2], direction) + "}"
    return _LIMIT_SUBSCRIPT_RE.sub(_sub, s)


def beautify_bare_canonical_latex(text: str) -> str:
    """Convert CAS-canonical-syntax text into displayable LaTeX WITHOUT
    invoking the CAS: ``*``-multiplication becomes a space, bare Greek names
    (``sigma`` -> ``\\sigma``) get their LaTeX macro, and ``sqrt(...)``
    function calls become ``\\sqrt{...}``.  Term order, parens, and every
    other token are left exactly as typed.

    This is deliberately a pure text substitution, not a CAS round-trip:
    Maxima's own renderer (``tex1``) re-orders terms into its internal
    canonical polynomial order even without simplifying, which would corrupt
    the user's own term ordering for raw, unevaluated input such as an H-1
    RHS or a YOU-card echo of the literal command typed.
    """
    out = re.sub(r"\s*\*\s*", " ", normalize_superscript_star(text))
    out = _sqrt_call_to_radical(out)
    out = restore_decorated_name(out)
    out = _BARE_GREEK_RE.sub(_bare_greek_sub, out)
    out = _GREEK_SUBSCRIPT_RE.sub(_greek_subscript_sub, out)
    # `vec(...)` はこの系がベクトルを包む綴りであって研究者の記法ではないので、
    # 表示に出してはいけない（原則#1）。復元は CAS 非依存なので、canonical を
    # 表示にする入口で当てる——ラベルだけこの復元を通っておらず、公開 TeX に
    # `G_{\sigma}(vec(r)t,vec(r)'t')` が出ていた（利用者報告、2026-08-22、E18）。
    # **入口はここだけではない。** この註は 2026-09-11 まで自分を「共通の入口」と
    # 名乗っていたが、`BackendHandler.normalize_display_latex`（`register_expression`
    # と操作経路が通る）は訊いておらず、`\mathbf r` が `vec(r)` と表示されていた。
    # 註が「唯一の扉」を名乗っていたら、扉を数えること。
    out = restore_limit_subscript(out)
    return BackendHandler._restore_vector(out)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Symbol type-declaration NL patterns (shared across all backends)
#
# Recognized forms:
#   Japanese: <sym> を <type-keyword> とする / と定義する / と宣言する
#   English:  define <sym> as (a) <type-keyword>
#             <sym> is (a) <type-keyword>
#
# Each type produces a different operation name so the engine can register
# the correct Quantity.kind and the backend can emit the right CAS command.
# ---------------------------------------------------------------------------

#: 名前のうち**装飾**である部分（`H_hat_0` の `hat`）。添字ではないので、添字の
#: 個数を数える規則から外す。表示側の復元（`latex_converter._restore_name`）と
#: 同じ表を読む——ここで数え方が割れると、宣言だけが別の量を登録する。
def _decoration_pieces() -> frozenset:
    from notebook.runtime.latex_converter import (
        _DECO_CMD_BY_NAME, _SCRIPT_DECORATIONS)
    return frozenset(set(_DECO_CMD_BY_NAME) | set(_SCRIPT_DECORATIONS))


_DECORATION_PIECES = _decoration_pieces()


def _without_decoration_pieces(name: str) -> str:
    """``H_hat_0`` → ``H_0``: the name with its decoration marks removed."""
    pieces = name.split("_")
    return "_".join([pieces[0]] + [p for p in pieces[1:]
                                   if p not in _DECORATION_PIECES])


#: 宣言文の中の「量の名前」。研究者が式の側で書く綴りをそのまま受ける ——
#: 先頭の装飾コマンド（`\hat H`）、ギリシャ（`\Sigma`）、添字（`G_0`）、
#: 上付きラベル（`G^0`・`G^{(1)}`）、プライム（`k'`）。捕まえた綴りは
#: `declared_name` が**式の側と同じ扉**（`latex_to_canonical`）で名前に直す。
#: ここが式と別の読みを持っていたのが DECL-NAME-READING（2026-08-21）:
#: `^` がクラスに無いので `G^0 を行列とする` は文ごと不成立になり、装飾は
#: 綴りごと落ちて `\hat H` が `H`（別の量）を宣言していた。
#: 宣言の動詞。英語の文では名前より前に立ち、**それ自身が識別子として読める**ので、
#: 剥がさないと名前として拾われる —— 実測 2026-08-21、`declare G^0 as a matrix` は
#: 成功を返して `declare` という名前の行列を作っていた。
_DECL_VERB_RE = re.compile(r"^\s*(?:define|declare|let|set|treat|consider)\s+", re.I)
#: 英語の繋辞。`decl_stmt[en]` の中にも同じものが書かれているが、こちらは
#: **マッチ位置を知るため**に単独で使う（名前の並びがどこで終わるか）。
_EN_COPULA = r"\s+(?:as|is|are)\s+(?:a\s+)?"


# TeX Greek commands → canonical names (e.g. \gamma → gamma), for the NL
# declaration parser below.  Built from the CAS-NEUTRAL table: a shared module
# must not depend on one backend's converter, and must not make that backend's
# canonicalization the default for every session (open_items
# AUDIT-SHARED-CAS-LEAK).  The constants (\pi, \infty, \mathrm{i}) live in each
# converter's own _GREEK_MAP and are absent from the neutral table by design —
# which is why the previous version had to filter %pi/inf/%i back out.
_TEX_GREEK_TO_CANONICAL: dict[str, str] = {
    tex.lstrip("\\"): name for tex, name in TEX_GREEK_TO_NAME.items()
}

# Regex to match \cmd in text (e.g. \gamma, \omega, \Delta)
_TEX_CMD_RE = re.compile(r"\\([A-Za-z]+)")


def _nl_any(key: str, text: str) -> bool:
    r"""Does *text* match ``key`` in EITHER language?

    Every reader of a declaration sentence in this file asks it this way, and
    that is deliberate: these are not READINGS chosen after the session language
    is known — they are the questions "is this a declaration at all?" and "what
    TYPE does it name?", asked before anything else, by code that has no reliable
    language (a handler with no engine bound answers ``DEFAULT_LANG``).

    The two mistakes are not symmetric.  A false positive costs a redundant
    check; a false NEGATIVE falls through to ``define_scalar_parameter`` — a
    sentence that says OPERATOR registers a SCALAR and reports success, which is
    the defect recorded in CLAUDE.md for the nuclear-spin declaration and, in the
    same shape, for `Define k, m, \mu, \hbar as scalars`.

    Measured 2026-08-23 while migrating this file's patterns to
    ``i18n/nl_patterns.json``: narrowing the same question by language in the
    Julia handler dropped `の消滅演算子をbとする`, and the instruction corpus did
    NOT see it (its scenarios run each language in its own session, so the
    crossing never occurs there).

    The implementation is ``nl_patterns.nlp_any``: the supervisor's own lexicon
    asks the same question (2026-08-27), and one question gets one answer.
    """
    return nlp_any(key, text)


def _nl_search(key: str, text: str):
    """The first match of *key* in either language, or None (same rule as above)."""
    return nlp_search(key, text)


#: 型語彙 → 操作。**優先順位と対応づけは構造**なのでここにあり、語そのものは
#: `i18n/nl_patterns.json` にある。場（`ベクトル場`/`vector field`）は素の
#: `ベクトル`/`vector` より先に見なければならない——後にすると、div/curl/grad の
#: 対象になる**場**ではなく、ただのベクトル量が宣言される。
_TYPE_KEYWORD_OPS: "list[tuple[str, str]]" = [
    ("type_kw_tensor",       "define_tensor"),
    ("type_kw_matrix",       "define_matrix"),
    ("type_kw_vector_field", "define_vector_field"),
    ("type_kw_scalar_field", "define_scalar_field"),
    ("type_kw_vector",       "define_vector"),
]


def wants_redeclare(text: str) -> bool:
    """研究者が**宣言し直す**と明示したか（型の変更を許す唯一の合図）。"""
    return _nl_any("redeclare", text)


def read_spin_operator_declaration(text: str) -> "tuple[str, str] | None":
    r"""``(主語, 説明)`` を返す。スピン演算子の宣言でなければ None。

    共有層に置くのは、これが**宣言文の読み方**であって CAS の知識ではないから
    ——`_infer_type_from_label` のすぐ隣が定位置である。Julia だけが読めていた
    ため、`magnetic_resonance`（Maxima）セッションではどの綴りもスカラーに
    なっていた。
    """
    m = _nl_search("spin_operator_decl", text)
    if not m:
        return None
    return m.group(1).strip(), (m.group("desc") or "").strip()


def names_particle_operator_declaration(text: str) -> "str | None":
    r"""この文は**粒子演算子の宣言**か。そうなら `define_fermion_operator` /
    `define_boson_operator` のどちらかを返す。

    **読みは文法なので共有層に置く**（`read_spin_operator_declaration` と同じ
    理由・同じ隣）。実装は Julia ハンドラの中にあり、Maxima セッションでは
    誰も訊いていなかったので、`c を電子の消滅演算子とする` が型ラベルの既定に
    落ちて **`define_scalar_parameter` を登録し成功を返していた**
    （実測 2026-09-06、日英7綴り）。CLAUDE.md が「核スピン演算子」で記録した
    defect と同じ形で、スピンについては直され粒子演算子については残っていた。

    **統計は物理であって言語ではない**ので、両言語を無条件に見る
    （`should_bypass_auto_register_formula` と同じ理由——guard は絞らない）。
    """
    for _l in ("ja", "en"):
        m = nlp("particle_operator_decl", _l).search(text)
        if m:
            # 粒子名からは統計を決めない（それは Julia 側の粒子表の仕事）。
            # ここが答えるのは「粒子演算子の宣言である」ことだけなので、
            # フェルミオン側を代表として返す——**どちらであっても
            # `supported_backends` は同じ**（julia_full のみ）で、この関数の
            # 用途は「このバックエンドで動かせるか」を訊くことである。
            return "define_fermion_operator"
    for _l in ("ja", "en"):
        m = nlp("bare_quantum_op_decl", _l).search(text)
        if m:
            return ("define_fermion_operator"
                    if _nl_any("fermion_kind_kw", m.group("kind"))
                    else "define_boson_operator")
    # 粒子も統計も書かれていない「c を消滅演算子とする」。**どちらとも決めない**
    # ——ここが答えるのは「粒子演算子の宣言である」ことだけで、統計を要求するのは
    # この先の仕事である。
    if _nl_any("bare_ladder_op_decl", text):
        return "define_fermion_operator"
    return None


def _infer_type_from_label(label: str, backend: "str | None" = None) -> str:
    """Return the define_* operation that best matches the type description *label*.

    **最初に訊くのは操作スキーマ自身の別名**（`type_label`）である。同じ問いに
    2つの表が答えていて、実パイプラインで11綴りが別の操作に届き `confirmed` を
    返していた——`A を対称行列とする` が `define_matrix`（対称性が落ちる）、
    `A をスカラー関数とする` が `define_scalar_parameter`（場であることが落ちる）
    など。測定と理由は `notebook/runtime/type_label.py` の docstring にある。

    別名に完全一致しなかったときだけ、以下の優先順位に落ちる。これは
    **組み合わせラベル**（`時間依存のスカラー場`）のためのもので、そこは
    どの別名とも一致しないので構造として残っている。

    Scans for type keywords in priority order:
      scalar_field/vector_field > time_dependent > tensor > matrix > vector >
      scalar (default).  A field keyword takes priority over "時間依存" so that
      combined labels like "時間依存のスカラー場" still declare the field (with
      spatial dependence) rather than falling through to a bare
      ``declare_time_dependent`` that would lose the spatial vars — see
      ``_label_wants_time_dependent`` for how the time-dependence is folded
      back in for this combined case.
    Falls back to ``define_scalar_parameter`` when no keyword is found — this
    covers physics labels like "１電子エネルギー", "BCS 秩序パラメータ", "運動量" etc.
    """
    named = declaration_operation_for_label(label, backend)
    if named is not None:
        return named
    return _type_from_keywords(label)


def _type_from_keywords(label: str) -> str:
    """優先順位表だけで答える（別名表は見ない）。

    切り出してあるのは、`対称行列` のように**型と対称性を同時に言うラベル**で、
    別名表の答え（`declare_symmetry`）に加えて**基底の型**も要るからである
    ——研究者は「行列」と「対称」の両方を書いており、どちらを捨てても損失になる。
    """
    for key, op in _TYPE_KEYWORD_OPS:
        if op in ("define_scalar_field", "define_vector_field") and _nl_any(key, label):
            return op
    if _nl_any("time_dependent", label):
        return "declare_time_dependent"
    for key, op in _TYPE_KEYWORD_OPS:
        if _nl_any(key, label):
            return op
    return "define_scalar_parameter"


def _label_wants_time_dependent(label: str) -> bool:
    """True when *label* also asks for time-dependence (e.g. "時間依存のスカラー場").

    Only meaningful when ``_infer_type_from_label`` returned a field operation
    (scalar_field/vector_field); combine with that check at the call site.
    """
    return _nl_any("time_dependent", label)


def _looks_like_type_label(label: str) -> bool:
    """Return True if *label* looks like a mathematical/physical type description.

    Guards against false positives such as "x を 0 とする" (assignment) or
    "x を a とする" (substitution), where the label is a number or bare identifier.

    **その guard は数字の綴りしか知らなかった。** 語で書かれた数（`ゼロ`・
    `零`・`zero`）は下の2つの総当たり——「日本語なら型の説明」「ラテン文字4字
    以上なら物理の名詞」——に当たって型になり、`x をゼロとする` が
    `define_scalar_parameter` を **confirmed** で返していた（実測 2026-09-10）。
    総当たりは消せない（`１電子エネルギー` をスカラーとして受けるのが正しい）
    ので、**数詞だけを先に外す**。`一定` は数詞ではないので型のまま。
    """
    s = label.strip()
    # 語で書かれた数は値であって型ではない（表は notebook/runtime/number_words.py）
    if number_literal(s) is not None:
        return False
    # Contains Japanese text → definitely a type description
    if re.search(r"[぀-ヿ一-鿿！-ﾟ]", s):
        return True
    # Contains a Latin word of 4+ characters → likely a physics/math noun
    if re.search(r"[A-Za-z]{4,}", s):
        return True
    # Matches a known short type keyword directly (e.g. "vector", "tensor")
    for key, _ in _TYPE_KEYWORD_OPS:
        if _nl_any(key, s):
            return True
    return False

#: Standard math function names that should not be treated as user variables.
MATH_FUNC_NAMES: frozenset[str] = frozenset({
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh",
    "exp", "log", "sqrt", "abs",
    "floor", "ceiling", "round", "sign",
    "max", "min", "gcd", "lcm",
    "diff", "integrate", "limit", "sum",
    # Display-only functionals this system emits itself (R-9's bracket
    # functional; the expectation bracket lowers to the same one).  Measured
    # 2026-08-20: `\langle H \rangle = E_0` looked like a definition of a
    # FUNCTION called `expectation` at the point `H`, so a quantity by that
    # name was registered — the researcher defined the average of H, not the
    # averaging operation.  Same reason `\sin(x) = y` does not define `sin`.
    "expectation", "variance",
})


# ---------------------------------------------------------------------------
# Shared display-normalisation helpers
# ---------------------------------------------------------------------------

def _convert_caret_parens(latex: str) -> str:
    """Convert Maxima-style ^(expr) superscripts to LaTeX ^{expr}.

    Handles nested parentheses correctly.  Only converts ^( patterns; leaves
    ^{...} and ^x (single-char) untouched.
    """
    result: list[str] = []
    i = 0
    n = len(latex)
    while i < n:
        if latex[i] == '^' and i + 1 < n and latex[i + 1] == '(':
            result.append('^{')
            depth = 1
            i += 2  # skip '^('
            while i < n and depth > 0:
                ch = latex[i]
                if ch == '(':
                    depth += 1
                    result.append(ch)
                elif ch == ')':
                    depth -= 1
                    result.append('}' if depth == 0 else ch)
                else:
                    result.append(ch)
                i += 1
        else:
            result.append(latex[i])
            i += 1
    return ''.join(result)


# ---------------------------------------------------------------------------
# BuildResult (shared)
# ---------------------------------------------------------------------------

class BuildResult:
    """Thin result wrapper for :py:meth:`BackendHandler.build_command`."""

    __slots__ = ("ok", "command", "error", "sum_index_hint", "input_canonical",
                 "reason_code")

    def __init__(self, ok: bool, command: str, error: str | None = None,
                 sum_index_hint: str = "",
                 input_canonical: str | None = None,
                 reason_code: str | None = None) -> None:
        self.ok = ok
        self.command = command
        self.error = error
        #: WHY a refusal happened, for a caller that must tell one refusal from
        #: another.  `error` is the researcher's message: translated, and worded
        #: for a person — keying on it means matching prose, in whichever
        #: language the session happens to use.  Set it only where a caller has
        #: a REASON to distinguish; the default None keeps every existing
        #: refusal opaque, which is the honest state for most of them.
        self.reason_code = reason_code
        self.sum_index_hint = sum_index_hint  # e.g. "k" → prepend \sum_{k} to latex
        self.input_canonical = input_canonical  # override for new expression's input_canonical


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class UnsupportedAssumption(Exception):
    """This backend has no assumption mechanism.

    Carries an i18n KEY, not a message: the caller knows the session language,
    this layer does not (see the i18n rule's "a backend handler must not guess
    the language").
    """

    def __init__(self, i18n_key: str, **fmt):
        super().__init__(i18n_key)
        self.i18n_key = i18n_key
        self.fmt = fmt


class BackendHandler(ABC):
    """Protocol for backend-specific engine logic."""

    # -- Command building ---------------------------------------------------

    @abstractmethod
    def build_command(self, operation: str, args: dict, expr_lookup: dict) -> BuildResult:
        """Translate *operation* + *args* into a ready-to-execute CAS string.

        *expr_lookup* maps expression IDs (``"E1"`` …) to their canonical strings.
        Returns a :class:`BuildResult` with ``ok=True`` and ``command`` set on
        success, or ``ok=False`` with ``error`` on failure.
        """

    # declare_symmetry: record index symmetry of a 2-argument function.  The
    # name/symmetry validation is identical across backends; only the emitted
    # command differs.  ``_SYMMETRY_COMMANDS`` maps symmetry → a command
    # template (``{name}`` placeholder).  A backend with no CAS-level symmetry
    # declaration (e.g. Julia/QuantumAlgebra — symmetry is stored Python-side
    # in SymbolContext and used at Wick-contraction time) leaves this ``None``
    # and gets a no-op ``"true"`` command instead.
    _SYMMETRY_COMMANDS: "dict[str, str] | None" = None

    #: 対称性と**非スカラー性**を1つのコマンドで宣言する形。`対称行列` は型と
    #: 対称性の両方を言っており、片方だけ宣言すると他方が黙って落ちる。
    #: Maxima の `declare` は (原子, 特性) の**対**を取るので
    #: `declare(A, nonscalar, symmetric)` は拒否され、
    #: `declare(A, nonscalar, A, symmetric)` が通る（実測 2026-09-05）。
    _SYMMETRY_COMMANDS_NONSCALAR: "dict[str, str] | None" = None

    #: スキーマの ``supported_backends`` に載っている自分の綴り。宣言の型ラベルを
    #: 操作スキーマの別名に照合するとき、**そのバックエンドが動かせない操作へ
    #: 送らない**ために要る（`define_fermion_operator` は julia_full だけ）。
    #: 共有コードがバックエンドを見分けるのではなく、バックエンドが自分を名乗る
    #: ——`_MATRIX_OP_RESTORE` などと同じクラス属性の形（Backend Encapsulation）。
    _SCHEMA_BACKEND: "str | None" = None

    def _refuse_declaration_this_backend_cannot_make(
        self, label: str, text: str
    ) -> "object | None":
        """このバックエンドが動かせない宣言なら、**理由つきの断り**を返す。

        **「動かせない」を既定のスカラーに落とさない。** 直す前は
        `c を電子の消滅演算子とする` が Maxima セッションで
        `define_scalar_parameter` になり、**「演算子」と書いた文がスカラーを
        登録して成功を返していた**（実測 2026-09-06、日英7綴り）。フェルミオン／
        ボソン代数は Julia/QuantumAlgebra にしかなく、受け入れると `c'(k) c(k)`
        がただの積になってその後の計算が全部おかしくなる。スピンは
        `spin_algebra` が CAS-bypass で実装されているので Maxima でも受ける——
        その違いが `supported_backends` にそのまま出ている（利用者の判断:
        受けるのではなく断る）。

        **研究者が書いていない統計を断り文で名指さない。** 粒子も統計も書かない
        `c を消滅演算子とする` では、系が代表として `define_fermion_operator` を
        選んでいるだけである（利用者の指摘、2026-09-06）。断る理由はバックエンド
        であって統計ではないので、名指すのは**族**にする。
        """
        op = (declaration_unsupported_here(label, self._SCHEMA_BACKEND)
              or self._particle_op_unsupported_here(text))
        if op is None:
            return None
        from notebook.runtime.i18n import t as _t
        from notebook.runtime.llm_supervisor import InterpretResult
        from notebook.runtime.operation_validator import OperationValidator
        if (op in ("define_fermion_operator", "define_boson_operator")
                and not _nl_any("bare_quantum_op_decl", text)
                and not _nl_any("particle_operator_decl", text)):
            display = _t("interpreter.decl_family.second_quantization", self.lang)
        else:
            spec = OperationValidator().get_spec(op) or {}
            display = (spec.get("display") or {}).get(self.lang) or op
        return InterpretResult(
            ok=False, operation=None, args={}, confidence=1.0, source="rule",
            error=_t("interpreter.declaration_backend_unsupported",
                     self.lang, display=display))

    def _particle_op_unsupported_here(self, text: str) -> "str | None":
        """粒子演算子の宣言だが、このバックエンドでは動かせない——ならその操作名。"""
        op = names_particle_operator_declaration(text)
        if op is None or self._SCHEMA_BACKEND is None:
            return None
        from notebook.runtime.operation_validator import OperationValidator
        spec = OperationValidator().get_spec(op) or {}
        if self._SCHEMA_BACKEND in (spec.get("supported_backends") or []):
            return None
        return op

    def _build_declare_symmetry(self, args: dict) -> BuildResult:
        name     = str(args.get("name", ""))
        symmetry = str(args.get("symmetry", ""))
        if not name:
            return BuildResult(ok=False, command="",
                               error="declare_symmetry: missing name")
        if symmetry not in ("symmetric", "antisymmetric", "hermitian"):
            return BuildResult(ok=False, command="",
                               error=f"declare_symmetry: unknown symmetry '{symmetry}'")
        if self._SYMMETRY_COMMANDS is None:
            return BuildResult(ok=True, command="true")
        algebra = str(args.get("algebra", "") or "")
        table = self._SYMMETRY_COMMANDS
        if algebra in ("matrix", "vector", "tensor") and self._SYMMETRY_COMMANDS_NONSCALAR:
            table = self._SYMMETRY_COMMANDS_NONSCALAR
        return BuildResult(ok=True, command=table[symmetry].format(name=name))

    @abstractmethod
    def wrap_command_for_storage(self, command: str, output_id: str | None, operation: str) -> str:
        """Optionally wrap *command* so that its result is stored in the CAS session.

        For Julia, non-void operations are assigned to ``_nb_<output_id>`` so
        that subsequent operations can reference the result without re-parsing
        the display text.  Maxima keeps assignment state differently and does
        not need this wrapping.
        """

    @abstractmethod
    def canonical_for_result(self, cas_value: str, output_id: str | None, operation: str) -> str:
        """Return the canonical string that should be stored in ``Expression.canonical``.

        For Julia, this is ``_nb_<output_id>`` (the variable name in the Julia
        session).  For Maxima it is just *cas_value*.
        """

    @abstractmethod
    def postprocess_result_latex(
        self,
        canonical: str,
        raw_latex: str,
        registry: SymbolContext,
        sum_indices: str = "",
    ) -> str:
        """Apply backend-specific post-processing to the LaTeX output.

        *sum_indices* is the comma-separated list of summation dummy names in
        the order the backend numbers them (see `_sum_order`), or "" when the
        result is not a sum.

        *canonical* is the raw text result from the CAS adapter.
        *raw_latex* is the LaTeX string (may equal *canonical* when no LaTeX
        rendering is available).
        Returns the final LaTeX string ready to store in ``Expression.latex``.
        """

    def scalar_simplify_command(self, expr: str) -> str:
        """CAS text that simplifies one SCALAR and returns ``value|latex``.

        The coefficient layer of a pair representation is scalar algebra, and
        which engine does it — and how a result is spelled back — is the
        BACKEND's knowledge (Backend Encapsulation Rule).
        """
        raise NotImplementedError(
            "scalar_simplify_command is not implemented for this backend")

    def undeclare_command(self, name: str, algebra: str) -> "str | None":
        """CAS text withdrawing a previous TYPE declaration, or None if the
        backend has nothing to withdraw.

        How a declaration is undone is the backend's knowledge (Backend
        Encapsulation Rule): Maxima's `declare(x, scalar)` is undone by
        `remove(x, scalar)`, and measured 2026-08-22, `declare` after `remove`
        succeeds where `declare` alone is refused.
        """
        return None

    def pair_exponential_command(self, entries: "list[str]", size: int,
                                 time_var: str) -> str:
        r"""CAS text for the FIRST ROW of ``exp(i t M)`` (PAIR-REP).

        A time-evolved operator's coefficients are the first row of a matrix
        exponential — the closed form of the linear ODE the ad-chain is.  Which
        scalar engine computes it, and how the row is spelled back, is the
        BACKEND's knowledge; a backend without one says so here rather than
        letting a shared layer assemble somebody's syntax (BP-5).
        """
        raise NotImplementedError(
            "pair_exponential_command is not implemented for this backend")

    # ------------------------------------------------------------------
    # Shared postprocessing helpers (used by both Maxima and Julia)
    # ------------------------------------------------------------------

    #: True when a summed RESULT needs the Σ pasted onto its latex by the
    #: caller.  False for a backend that renders its own — and rendering its
    #: own is also what tells the caller the result is STILL a sum: a sum that
    #: evaluated away (`\sum_{i=1}^{3} i^2` → 14) must not be given one back,
    #: which is what produced `\sum_{i}[14]` (measured 2026-08-08).
    needs_sum_prefix: bool = False

    @staticmethod
    def sum_dummy_display_names(count: int) -> "list[str]":
        r"""Neutral display names for *count* summation dummies.

        NOT the names the researcher wrote.  A backend may discard those —
        QuantumAlgebra does, deliberately ("summed indices have no semantic
        meaning"), replacing them with `#₁`, `#₂`, … numbered by first
        occurrence IN THE BODY.  Measured 2026-08-07, that is not the order the
        indices were bound in: `∑(:m, ∑(:n, c'(:n) c(:m)))` and
        `∑(:n, ∑(:m, c'(:n) c(:m)))` produce the identical `∑₁₂ c†(#₁) c(#₂)`.
        Mapping position-to-position therefore prints a sum the researcher did
        not write — a legal relabelling for one sum over one range, but a claim
        we cannot check once terms carry different numbers of dummies (which
        `normal_form` produces: `∑₁ - ∑₁₂ …`).

        So this does not guess.  It gives distinct, unambiguous names and says
        nothing about which one the researcher called what.  Recovering that
        correspondence needs information from upstream — open_items SUM-DUMMY-NAMES.

        Braced (`k_{1}`) so the subscript survives being concatenated with the
        next index in a backend that writes them without a separator.
        """
        return [f"k_{{{i}}}" for i in range(1, max(0, count) + 1)]

    @staticmethod
    def result_latex_has_sum(latex: str) -> bool:
        r"""True if *latex* already carries a summation sign.

        The caller wraps a result in `\sum_{…}[…]` when the backend could not
        express the sum itself.  A backend whose sum IS an object (QuantumAlgebra's
        `∑`) renders its own, and the wrap then produced `\sum_{k}[\sum_{j} …]`
        (measured 2026-08-07).  Asking the rendered RESULT — rather than which
        backend produced it — keeps the shared layer free of backend names
        (Backend Encapsulation Rule).
        """
        return bool(re.search(r"\\sum(?![A-Za-z])", latex or ""))

    @staticmethod
    def result_carries_index(latex: str, names: "list[str]") -> bool:
        r"""True if any of *names* still appears in *latex* AS AN INDEX.

        The Σ prefix exists for a result that CARRIES a summation index without
        printing a Σ.  Whether it does was being decided from the INPUT — the
        source expression's own LaTeX — so a sum the CAS had contracted away got
        a Σ pasted back on: measured 2026-08-22,
        `[U \sum_i n_{i↑}n_{i↓}, c_{1↑}]` came back as
        `\sum_{i}\left[U c†_{1↓} c_{1↑} c_{1↓}\right]` — **a sum over an index
        the summand does not contain**, which read literally is a different
        quantity, and it was a success.

        Position is what makes this precise rather than a substring search: an
        index sits after `_`, `{`, `(` or `,`.  The imaginary unit in
        `e^{- i U t}` does not, so a hint named `i` cannot fire on it.
        """
        for name in names:
            if not name:
                continue
            if re.search(r"[_(,{]\s*\\?" + re.escape(name) + r"(?![A-Za-z0-9])",
                         latex or ""):
                return True
        return False

    # Backend-specific matrix-op names → LaTeX notation.
    # Each entry: (mathrm_name, replacement_string).
    # replacement_string may use \1 for the captured argument.
    _MATRIX_OP_RESTORE: list[tuple[str, str]] = []

    #: LaTeX commands a backend may wrap a bare symbol name in before handing a
    #: result back, as regex fragments (e.g. ``r"\\mathrm"``).  The display
    #: bridge strips them so the researcher sees their own notation rather than
    #: the CAS's.  Empty here — a backend that wraps says so itself
    #: (Backend Encapsulation Rule).
    _DISPLAY_NAME_WRAPPERS: list[str] = []

    @staticmethod
    def _restore_kron_delta(s: str) -> str:
        """kron_delta(i, j) → \\delta_{i, j}"""
        return re.sub(
            r"kron\\?_delta\s*(?:\\left\s*\(|\()([^,)]+),\s*([^)]+)(?:\\right\s*\)|\))",
            r"\\delta_{\1, \2}",
            s,
        )

    @staticmethod
    def _restore_functional(s: str, name: str, wrap) -> str:
        r"""One display-only functional back to the researcher's notation.

        ``expectation(X)`` → ``\langle X \rangle``、``expectation(X, 0)`` →
        ``\langle X \rangle_{0}``; ``time_order(X)`` → ``T[X]``.  Parameterised
        rather than copied per functional — the scan below is the whole
        difficulty and a second copy is how two spellings drift apart.

        The CAS spelling is a function (so the value composes — that is R-9's
        rule for `E[X]`), but `tex1` renders it as the English word, so the
        display has to put the researcher's brackets back.

        The scan is balanced and knows two things a regex character class does
        not (both measured while writing it): ``\left(`` / ``\right)`` are
        delimiters whose parenthesis must not be captured into the argument,
        and ``\,`` is a THIN SPACE — splitting on its comma cut
        ``T\,\psi(x,t)`` in half.  The state, when the researcher wrote one, is
        the LAST top-level argument.
        """
        _OPEN = re.compile(r"\\left\s*([([{])|([([{])")
        _CLOSE = re.compile(r"\\right\s*([)\]}])|([)\]}])")
        # Maxima's `tex1` renders a name with an underscore as a BRACED
        # subscript (`time_order` → `time_{order}`), so the head pattern has to
        # accept both spellings — the same shape the label restore needed.
        _parts = name.split("_")
        _np = r"\\?_\{?".join(re.escape(_p) for _p in _parts)
        if len(_parts) > 1:
            _np += r"\}?"
        _HEAD = re.compile(rf"{_np}\s*(?:\\left\s*\(|\()")
        out: list[str] = []
        i = 0
        while i < len(s):
            m = _HEAD.match(s, i)
            if not m:
                out.append(s[i])
                i += 1
                continue
            depth, k, parts, start = 1, m.end(), [], m.end()
            closed_at = None
            while k < len(s):
                cm = _CLOSE.match(s, k)
                if cm:
                    depth -= 1
                    if depth == 0:
                        parts.append(s[start:k])
                        closed_at = cm.end()
                        break
                    k = cm.end()
                    continue
                om = _OPEN.match(s, k)
                if om:
                    depth += 1
                    k = om.end()
                    continue
                if s[k] == "," and depth == 1 and s[k - 1] != "\\":
                    parts.append(s[start:k])
                    start = k + 1
                k += 1
            if closed_at is None:
                out.append(s[i])
                i += 1
                continue
            body = (", ".join(p.strip() for p in parts[:-1])
                    if len(parts) > 1 else parts[0].strip())
            state = parts[-1].strip() if len(parts) > 1 else None
            out.append(wrap(body, state))
            i = closed_at
        return "".join(out)

    @classmethod
    def _restore_expectation(cls, s: str) -> str:
        r"""``expectation(X[, state])`` → ``\langle X \rangle[_{state}]``."""
        return cls._restore_functional(
            s, "expectation",
            lambda body, st: rf"\langle {body} \rangle" + (rf"_{{{st}}}" if st else ""))

    @classmethod
    def _restore_time_order(cls, s: str) -> str:
        r"""``time_order(X[, var])`` → ``T[X]`` / ``T_{var}[X]``.

        The researcher may have written ``\mathcal{T}`` or ``T\{…\}``; the
        display returns ONE spelling, and it types back in as the same object
        (the bracketed form with an applied content is read as the ordering).
        """
        return cls._restore_functional(
            s, "time_order",
            lambda body, st: (rf"T_{{{st}}}" if st else "T") + rf"\left[{body}\right]")

    @classmethod
    def _restore_vector(cls, s: str) -> str:
        r"""``vec(r)`` → ``\mathbf{r}``；``vec(r_x, r_y, r_z)`` → その成分の並び。

        ``vec`` はこの系がベクトルを包む綴りであって、研究者が書いた記法ではない。
        CAS を通った結果には ``{\it vec}\left(…\right)`` として出てくるので、
        表示では外す（原則#1: 通常の UI に CAS 構文を出さない）。

        **成分に開いたあとは包みが要らない** —— 利用者の指摘（2026-08-20、公開 TeX を
        レンダリングして発見）: ``\varphi_{m,\sigma}(vec(r_x, r_y, r_z))`` は
        3成分の関数と書けば足りる。包んだままだと ``vec`` という語が本文の数式に
        居座り、成分も読みにくい。
        """
        return cls._restore_functional(
            s, "vec",
            lambda body, st: (rf"\mathbf{{{body}}}" if st is None
                              else f"{body}, {st}"))

    def _restore_matrix_ops(self, s: str) -> str:
        r"""Restore \\mathrm{OP}(EXPR) → EXPR^{SYMBOL} for each backend's matrix ops.

        The mapping is defined in the class attribute ``_MATRIX_OP_RESTORE``.
        """
        for op_name, replacement in self._MATRIX_OP_RESTORE:
            s = re.sub(
                rf"\\mathrm\{{{op_name}\}}\s*(?:\\left\s*\(|\()(.+?)(?:\\right\s*\)|\))",
                replacement,
                s,
            )
        return s

    @staticmethod
    def _restore_index_sum(s: str) -> str:
        r"""\\mathrm{sum}\\left( expr, idx \\right) → \\sum_{idx} expr"""
        pat = r"\\mathrm\{sum\}\s*(\\left\s*\(|\()"
        while True:
            m = re.search(pat, s)
            if not m:
                break
            start_idx = m.start()
            inner_start = m.end()
            open_str = m.group(1)
            close_str = r"\right)" if "left" in open_str else ")"
            depth = 1
            i = inner_start
            while i < len(s) and depth > 0:
                if s[i:].startswith(open_str):
                    depth += 1
                    i += len(open_str)
                elif s[i:].startswith(close_str):
                    depth -= 1
                    if depth == 0:
                        break
                    i += len(close_str)
                else:
                    i += 1
            if depth == 0:
                inner = s[inner_start:i]
                parts = inner.rsplit(",", 1)
                if len(parts) == 2:
                    expr_part, idx_part = parts[0].strip(), parts[1].strip()
                    replacement = rf"\sum_{{{idx_part}}} {expr_part}"
                else:
                    replacement = r"\sum " + inner
                s = s[:start_idx] + replacement + s[i + len(close_str):]
            else:
                break
        return s

    # Characters/operators that mark *args_str* as an arithmetic expression
    # (e.g. ``l \left(1 - \cos q\right)``) rather than a simple index, for
    # ``_display_bridge_scalar_guard``.  Maxima additionally treats ``^`` as
    # arithmetic (exponents never appear inside a bare index argument there).
    _SCALAR_GUARD_ARITH_CHARS: tuple[str, ...] = ("+", "*", "/", r"\left", r"\right")

    def _display_bridge_scalar_guard(self, args_str: str, qty: "Quantity") -> bool:
        """Skip subscript-lowering for scalars with arithmetic arguments.

        e.g. ``l \\left(1 - \\cos q\\right)`` is multiplication, not a
        subscripted form.  Only simple index arguments (a single variable or
        digit) are lowered; the arithmetic-character set is backend-specific
        via ``_SCALAR_GUARD_ARITH_CHARS``.
        """
        if qty.kind not in ("scalar", "anonymous"):
            return False
        _is_arith = (
            any(c in args_str for c in self._SCALAR_GUARD_ARITH_CHARS)
            or any(op in args_str for op in (r"\cdot", r"\times"))
        )
        _has_binary_minus = bool(re.search(r"[A-Za-z0-9]-", args_str))
        _is_multi = "," in args_str
        return (_is_arith or _has_binary_minus) and not _is_multi

    def _apply_display_bridge(self, s: str, registry: SymbolContext) -> str:
        """Lower indexed-symbol function calls to subscript notation.

        Converts ``a(k)`` → ``a_{k}``, ``c†(k)`` → ``c^{\\dagger}_{k}`` etc.
        for all symbols registered in *registry* that carry index information,
        then the same for undeclared bases the researcher wrote subscripted
        (``indexed_bases``, docs/32 Phase 4).
        """
        s = restore_indexed_bases(s, registry)
        s = restore_greek_glyphs(s, registry)
        is_quantum = (
            "second_quantization" in registry.domains
            or "quantum_mechanics" in registry.domains
            or registry.session_mode == "second_quantization"
        )
        for name, prop in registry.registry.items():
            qty = prop.quantity
            if not (
                qty.kind in ("scalar", "operator", "anonymous")
                or (qty.kind == "function"
                    and getattr(qty, "index_lowering_style", None) is not None)
            ):
                continue
            base_name = name.rstrip("'†")
            primes = name.count("'") + name.count("†")
            escaped_base = re.escape(base_name)
            # The adjoint marks are CAPTURED, not just consumed.  Counting them
            # from the registered NAME alone was wrong: operators are registered
            # under the bare base (`b`), so `_primes` was always 0 and every
            # matched form lost its dagger — `b'(k) b(k)` displayed as
            # `b_{k} b_{k}`, i.e. the number operator shown as something else.
            # Measured 2026-08-04: unreachable at the time (QuantumAlgebra hands
            # back subscript forms and Latexify hands back `\mathrm{}`-wrapped
            # ones, neither of which matched), but teaching this pattern the
            # wrapped shape below is exactly what makes it reachable.
            marks_pat = r"(?P<marks>(?:'|\\prime|\^\{\\dagger\}|†)*)"
            call_pat = (r"\s*(?:\\left\s*\(|\(|\\left\s*\[|\[)"
                        r"(?P<args>.*?)(?:\\right\s*\)|\)|\\right\s*\]|\])")
            # One pattern for the bare name, plus one per wrapper this backend
            # may put around it.  Julia's Latexify hands back
            # `\mathrm{g'}\left( k, u \right)` — CAS spelling, not the
            # researcher's — so the wrapper is declared per backend
            # (`_DISPLAY_NAME_WRAPPERS`) rather than assumed here.  Separate
            # patterns, not one alternation: named groups keep each readable,
            # and a mis-numbered group is how the dagger got dropped above.
            patterns = [rf"\b(?P<base>{escaped_base}){marks_pat}{call_pat}"]
            patterns += [
                rf"{w}\{{(?P<base>{escaped_base}){marks_pat}\}}{call_pat}"
                for w in self._DISPLAY_NAME_WRAPPERS
            ]

            def _replace(m: re.Match,
                         _qty=qty, _primes=primes,
                         _is_q=is_quantum) -> str:
                matched_base = m.group("base")
                # Adjoint marks written in the TEXT count too — the operator is
                # registered under its bare name, so the registered count alone
                # is always 0 and every dagger would be dropped.
                _primes = max(_primes, len(re.findall(
                    r"'|\\prime|\^\{\\dagger\}|†", m.group("marks") or "")))
                args_str = m.group("args").strip().replace(" ", "")
                args_list = args_str.replace(r"\,", ",").split(",")
                # Guard 1: time-dependent var or natural_vars
                if any(
                    arg in _qty.natural_vars
                    or (_qty.time_dependent and arg == _qty.time_parameter)
                    for arg in args_list
                ):
                    return m.group(0)
                # Guard 2 (backend-specific): e.g. arithmetic index guard
                if self._display_bridge_scalar_guard(args_str, _qty):
                    return m.group(0)
                display_base = matched_base
                if _primes > 0:
                    display_base = (
                        f"{matched_base}^{{\\dagger}}" if _is_q
                        else matched_base + "'" * _primes
                    )
                return f"{display_base}_{{{args_str}}}"

            for _pat in patterns:
                s = re.sub(_pat, _replace, s)
        return s

    @abstractmethod
    def format_constant_for_latex(self, token: str) -> str:
        """Convert a backend canonical constant token to its LaTeX form.

        E.g. Maxima's ``%pi`` → ``\\pi``, ``%inf`` → ``\\infty``.
        For Julia or already-LaTeX tokens the string is returned unchanged.
        """

    @abstractmethod
    def parse_quantity(self, canonical: str, defining_expr_id: str) -> Quantity | None:
        """Parse a defining equation canonical string into a :class:`Quantity`.

        Returns ``None`` when the canonical does not represent a definition.
        """

    @abstractmethod
    def canonical_variables(self, canonical: str) -> set[str]:
        """Return variable names appearing in *canonical*, ignoring math functions.

        Scans only the RHS when an equation ``=`` is present.
        """

    @abstractmethod
    def qty_name_from_origin(self, origin: str) -> str:
        """Extract the bare quantity name from an *origin* string.

        E.g. ``"f(x,y)"`` → ``"f"``, ``"q"`` → ``"q"``.
        """

    @abstractmethod
    def wrt_qdot_canonical(self, wrt_name: str, t: str,
                           source_canonical: str = "") -> str:
        """Return the canonical form of ``dq/dt`` used as a differentiation variable.

        It must be spelled the way *source_canonical* — the expression being
        differentiated — spells the same quantity, or the CAS answers 0; see the
        Maxima override's measurement.
        """

    @abstractmethod
    def declare_depends(self, adapter: object, var: str, depends_on: list[str]) -> None:
        """Declare a CAS-side variable dependency (e.g. ``depends([q],[t])``).

        *adapter* is the live CAS adapter instance.  For backends that do not
        support this concept (Julia) the method should be a no-op.
        """

    @property
    @abstractmethod
    def supports_equation_source(self) -> bool:
        """``True`` when the backend can resolve equation-source transformations.

        Maxima canonicals are rich enough to detect ``f(x) = rhs`` patterns
        and extract an equation LHS.  Julia canonicals are variable references
        (``_nb_E3``) and do not support this.
        """

    @abstractmethod
    def latex_to_canonical(self, latex: str) -> str:
        """Convert a LaTeX expression string to backend-compatible canonical form.

        Maxima and Julia use different constant names (``%pi`` vs ``pi``,
        ``inf`` vs ``Inf``, ``%i`` vs ``im``) and different function names
        (``determinant`` vs ``det``, ``mattrace`` vs ``tr``).  Each backend
        handler performs the conversion appropriate to its CAS.
        """

    @abstractmethod
    def parse_ir(self, canonical: str, before_alphatize=None) -> "Expr | None":
        """Parse a canonical string into an IR AST node, or return ``None``.

        For Maxima, delegates to :class:`MaximaIRCompiler`.  For Julia the
        canonical is a variable reference (``_nb_E3``) that cannot be parsed
        as a mathematical expression, so implementations should return ``None``.

        The result is stored in ``Expression.ir_ast`` at runtime only — it is
        not persisted (canonical is the ground truth).

        *before_alphatize* is passed through to the resolver, which calls it
        while the dummy indices still carry the names the researcher wrote (see
        :meth:`BaseSemanticResolver.resolve`).  A backend that does no resolving
        ignores it — there is nothing yet to hand over.
        """

    def parse_for_undefined_scan(self, canonical: str) -> "Expr | None":
        r"""Which AST the undefined-symbol scan should read for *canonical*.

        Only reached where `parse_ir` returns `None` (Julia). The default parses
        without resolving — what the engine did inline — and a backend whose
        resolver can read the canonical overrides this. Unresolved, a binder
        looks like symbols (`\lim_{x,to,0}` → `lim`, `to`); measurement and the
        rejected alternatives are in
        `tests/test_a_binder_is_not_an_undefined_symbol.py`.
        """
        from notebook.runtime.shared_parser import SharedParser
        return SharedParser().parse(canonical)

    def cas_repr(self, cas_value: str, canonical: str) -> str | None:
        """Return a human-readable secondary representation, or ``None``.

        For backends where ``canonical_for_result`` replaces the actual CAS
        output with an opaque variable reference (e.g. Julia's ``_nb_E3``),
        this method preserves the original ``cas_value`` (e.g. Julia's
        ``string(result)``) so it can be stored in ``Expression.cas_repr``.

        The default returns ``None`` (Maxima canonicals are already readable).
        Override in ``JuliaBackendHandler``.
        """
        return None

    def apply_time_dep_map(self, s: str) -> str:
        """Replace dot-notation placeholders with CAS-native time-derivative forms.

        The default is a no-op.  MaximaBackendHandler overrides this to convert
        q_dot → diff(q(t),t,1) so that P2 assignments store time-dependent
        expressions in a form Maxima can differentiate.
        """
        return s

    def ensure_explicit_mult(self, s: str) -> str:
        """Promote space-juxtaposition to an explicit multiplication sign.

        Users may type a space in place of `*` (``k r_ddot``, ``m r^2``).  CAS
        backends that lack implicit multiplication (Maxima) must receive an
        explicit ``*`` or the source is mis-parsed.  The default is a no-op;
        MaximaBackendHandler overrides it.  The ambiguous symbol-space-paren case
        (``f (x)``) is deliberately left to the SharedParser, which has the
        SymbolContext needed to decide product vs function call.
        """
        return s

    def lhs_latex_to_canonical(self, lhs: str) -> str:
        """Convert a LaTeX LHS string to backend canonical form for substitution.

        Used by the substitution pipeline when an lhs_alias_map key is in LaTeX
        form (e.g. ``\\ddot{q}``) and needs to be converted to the canonical
        form that the CAS understands (e.g. ``'diff(q(t),t,2)``).
        The default is a no-op; override in backends where LaTeX dot-notation
        maps to derivative expressions.
        """
        return lhs

    # D-7 (open_items.md): an H-1 LHS containing literal-division derivative
    # notation (``dy/dx``) is a compound expression, not the single simple
    # atom engine._build_expr_lookup's fast path (``_SIMPLE_LHS_RE``) accepts
    # — so it was silently skipped and the operation only ever saw the bare
    # RHS canonical.  restore_derivative_lhs is the narrower, opt-in repair:
    # each top-level +/- term must be either a literal derivative or a simple
    # atom, or the whole thing aborts (returns None) rather than guess.
    _LITERAL_DERIV_TERM_RE = re.compile(
        r'^d(?P<dep>[A-Za-z_]\w*)/d(?P<indep>[A-Za-z_]\w*)$'
    )
    _SIMPLE_LHS_TERM_RE = re.compile(
        r'^(?:'
        r'[A-Za-z_Α-ω][A-Za-z0-9_Α-ω]*(?:_[A-Za-z0-9]+)?(?:\([^()]*\))?'  # plain id
        r'|\\[A-Za-z]+(?:\^[0-9]+)?(?:_[A-Za-z0-9]+)?'  # \cmd, \cmd^n, \cmd_n
        r')$'
    )

    def restore_derivative_lhs(self, lhs: str) -> "str | None":
        """Best-effort restoration of an H-1 LHS containing literal-division
        derivative notation (``dy/dx``, ``dx/dt``, …) into backend-ready CAS
        syntax.  Splits *lhs* on top-level +/- into terms; each term must be
        either a literal derivative (converted via ``literal_derivative_canonical``)
        or a plain/backslash atom (converted via ``lhs_latex_to_canonical``) —
        any other term aborts the whole restoration (returns None), and so
        does finding no literal derivative at all (nothing for this method to
        add over the existing simple-atom path).  Default-safe on backends
        that don't override ``literal_derivative_canonical`` (returns None
        for every term, so the first non-atom term aborts immediately).
        """
        terms = re.findall(r'[+-]?[^+-]+', lhs.replace(' ', ''))
        if not terms:
            return None
        found_derivative = False
        out_parts: list[str] = []
        for raw in terms:
            sign, body = ('', raw)
            if body and body[0] in '+-':
                sign, body = body[0], body[1:]
            if not body:
                return None
            m = self._LITERAL_DERIV_TERM_RE.match(body)
            if m:
                canon = self.literal_derivative_canonical(m.group('dep'), m.group('indep'))
                if canon is None:
                    return None
                found_derivative = True
                out_parts.append(f"{sign}{canon}")
            elif self._SIMPLE_LHS_TERM_RE.match(body):
                out_parts.append(f"{sign}{self.lhs_latex_to_canonical(body)}")
            else:
                return None
        return "".join(out_parts) if found_derivative else None

    def literal_derivative_canonical(self, dep: str, indep: str) -> "str | None":
        """Backend-specific first-derivative CAS syntax for literal-division
        notation (``dy/dx`` → e.g. Maxima's ``'diff(y,x)``).  Default:
        unsupported (None) — override where the backend has ODE-style scalar
        derivatives (Maxima)."""
        return None

    def to_time_function_form(self, canonical: str, names: "list[str]", t: str) -> str:
        """Rewrite time-dependent symbols to explicit function form for transforms
        that need it (e.g. Maxima's ``laplace`` ignores ``depends`` on a bare atom
        and treats ``q`` as a constant — it needs ``q(t)``).  Default: no-op."""
        return canonical

    #: How this backend spells ∞ in an integration bound.  Used when a volume
    #: measure carries no written range and the all-space reading is filled in
    #: (notebook/runtime/vector_components).  Maxima and Julia agree on "inf"
    #: today; a backend that does not overrides this rather than having the
    #: shared rewrite guess (Backend Encapsulation Rule).
    infinity_name: str = "inf"

    def normalize_bound(self, bound: str) -> str:
        """Normalize an integration bound (lower/upper) for this backend.

        Used by the rule-based definite-integral intercept so a user-typed limit
        such as ``2π`` or ``2\\pi`` becomes a backend-canonical constant.  The base
        converts unicode Greek to TeX commands; backends with a special π/∞ spelling
        (e.g. Maxima ``%pi``) override this.
        """
        return normalize_unicode_greek(str(bound))

    def localize_imaginary_unit(self, result, adapter, registry):
        """Render the imaginary unit using the domain's symbol (electrical
        engineering writes it ``j``).  The compute side maps the domain symbol to the
        backend's internal unit; this converts a result back at the VALUE level —
        where the unit is an unambiguous token — so a subscript / superscript /
        exponent ``i`` is never disturbed.  Default: no change."""
        return result

    def is_free_solve_const(self, value: str) -> bool:
        """True if *value* is a backend artifact standing for a STILL-FREE unknown
        in an under-determined solve result (Maxima introduces ``%r1``, ``%r2``, …).

        Used by FREE-PARAM specialization: when fewer conditions than free
        parameters are given, the unconstrained constants come back as these
        placeholders, which the engine maps back to the original constant names.
        Backends without this concept keep the default (never)."""
        return False

    def laplace_definition_command(self, f: str, t: str, s: str) -> str:
        """Build the definition integral ``∫₀^∞ f e^{-st} dt`` for the Laplace
        transform of an explicit function, without any builtin transform.  Only
        backends with a definite integrator over ``[0,∞)`` implement this."""
        raise NotImplementedError

    def declaration_reading_ops(self, kind=None, algebra=None) -> "tuple[str, ...]":
        """**値の無い宣言だけから計算できる操作**（その種別に対して）。

        どの操作がそうかは**そのバックエンドの知識**なので既定は空で、
        サブクラスが名乗る（Backend Encapsulation）。種別を取るのは操作名だけ
        では足りないから——理由と実測は
        `operand_guards.reject_declared_object_without_value`。
        """
        return ()

    def declaration_computing_ops(self, kind=None, algebra=None) -> "tuple[str, ...]":
        """そのうち**実際に計算できる**もの（断り文が案内する相手）。

        免除（`declaration_reading_ops`）は型エラーを出す組も含むので、
        そのまま案内すると**効かない回避策**を勧めることになる。
        """
        return ()

    def reads_declared_object(self, operation: str, kind=None, algebra=None) -> bool:
        return operation in self.declaration_reading_ops(kind, algebra)

    def clean_latex(self, latex: str) -> str:
        """Remove backend-specific display artifacts from a stored latex string.

        Called by the export layer so that backend quirks in already-stored
        ``expr.latex`` values are hidden from common code.  The default is a
        no-op; override in backends that produce non-standard spacing tokens.
        """
        return latex

    def hold_aggregates(self, code: str) -> str:
        """Keep a bounded ``sum``/``product`` from being EVALUATED, if the backend
        needs telling.  Default: nothing to do.

        Maxima evaluates one the moment it reads it, which turns the
        researcher's `\\sum_{i=1}^{3} c_i` into three terms in any result that
        merely CARRIES it — see `maxima_constants.quote_aggregates`.  A backend
        whose aggregates are ordinary objects overrides nothing.
        """
        return code

    def result_warnings(self, operation: str, adapter) -> "list[str]":
        """i18n KEYS to attach to an operation's result expression, if any.

        Asked after a SUCCESSFUL execution, so a backend can report something
        the researcher should see without failing the step — the
        「判別不能なら警告する」 shape.  Default: nothing.

        It takes the operation name and the adapter, not the RESULT: what counts
        as "worth saying" is the backend's own knowledge (it may have to ask the
        CAS for a flag its driver set), and passing the result in would invite a
        second, shallower judgement here in shared code.  The adapter is passed
        rather than held, following `localize_imaginary_unit`.
        """
        return []

    def assumption_canonical(self, canonical: str) -> str:
        """Spell *canonical* the way THIS backend's ``assume`` accepts it.

        Default: unchanged.  A relation the CAS writes one way in an expression
        and another way in an assumption is a backend quirk, so it belongs in
        the subclass (Backend Encapsulation Rule) rather than in
        ``add_assumption``, which must not know a CAS's spelling.
        """
        return canonical

    #: How this backend spells "hold this condition true" / "drop it".  Empty
    #: means the backend has no assumption mechanism at all — the base
    #: implementations below then refuse with a reason instead of sending
    #: anything.  Data rather than an override because the two commands differ
    #: only in their NAME; everything around them (rendering the condition,
    #: recording the step) is identical.
    _ASSUME_FN: str = ""
    _FORGET_FN: str = ""
    #: Which reason to give when this backend has none.  A subclass names its
    #: own, because WHY a backend cannot hold an assumption — and what the
    #: researcher should do instead — is that backend's knowledge, not this
    #: layer's (Backend Encapsulation Rule).  The default says only what is
    #: true of any such backend.
    _ASSUME_UNSUPPORTED_KEY: str = "error.assumption_unsupported"

    def assumption_command(self, canonical: str) -> str:
        """The command that asserts *canonical*, in this backend's dialect.

        ``assume(...)`` is a MAXIMA function, and ``add_assumption`` was
        building that string itself — a backend-specific command assembled in
        the backend-independent layer, which is the one thing the Backend
        Encapsulation Rule forbids.  In a Julia session the researcher's
        "X > 0 と仮定する" therefore came back as
        ``Sym assume is not callable. Use @syms assume(var1, var2,...)`` —
        raw CAS text in their face (Core Principle #1), naming a function they
        never wrote (open_items BP-5, measured again 2026-08-06).

        Raises :class:`UnsupportedAssumption` when the backend has no mechanism,
        so the caller can say so in the researcher's language rather than let
        the CAS answer for it.
        """
        if not self._ASSUME_FN:
            raise UnsupportedAssumption(self._ASSUME_UNSUPPORTED_KEY)
        return f"{self._ASSUME_FN}({self.assumption_canonical(canonical)})"

    def forget_command(self, canonical: str) -> "str | None":
        """The command that withdraws *canonical*, or None if there is nothing
        to send.  Withdrawing is best-effort by design (the CAS may warn that it
        was never held), so an absent mechanism is None rather than a refusal —
        an assumption that was never sent has nothing to take back."""
        if not self._FORGET_FN:
            return None
        return f"{self._FORGET_FN}({self.assumption_canonical(canonical)})"

    def assume_sign(self, engine, sym_name: str, sign: str, kind: "str | None" = None) -> None:
        """Send a CAS-level sign assumption for a just-declared scalar parameter.

        Called by ``NotebookEngine`` right after ``define_scalar_parameter``
        registers a symbol with a sign qualifier (positive/negative/nonneg/nonpos),
        e.g. "n を正の整数とする".  The sign is a real constraint, not just
        display metadata, so it should reach the CAS the same way an explicit
        "X > 0 を仮定する" step would (see engine.add_assumption).  Backends
        without an assume()-style predicate mechanism (e.g. Julia/Symbolics has
        no sign predicate, only ``::Real``/``::Integer`` type annotations) no-op.
        """
        return None

    def declare_domain(self, engine, sym_name: str, domain: str) -> None:
        """Tell the CAS a just-declared scalar's NUMBER DOMAIN (real / complex /
        integer / natural).

        The sibling of :meth:`assume_sign`, and for the same reason: a domain is
        a real constraint, not display metadata.  Measured 2026-08-16, all three
        were recorded on the Quantity and none reached the CAS, so a researcher
        could declare a symbol, see the system accept it, and get a different
        answer than they asked for:

            「z を複素数とする」  ratsimp(conjugate(z)*z) → z^2   (conjugate GONE)
            「n を整数とする」    ratsimp(sin(n*%pi))     → sin(%pi*n)  (it is 0)

        Backends without a predicate mechanism no-op, exactly as with
        ``assume_sign``.
        """
        return None

    def classify_cas_error(self, error: str) -> str | None:
        """Map a raw backend error string to an i18n message key, when the
        backend recognizes the failure as a known user mistake.

        Keeps backend-specific error signatures (e.g. Julia's opaque
        ``MethodError``) out of shared code: the caller looks up the returned
        key with ``t(key, lang)`` and raises a friendly message instead of the
        raw backend text.  The default returns ``None`` (no translation);
        override in backends that can recognize specific failures.
        """
        return None

    def cas_holds_value(self, adapter, name: str) -> bool:
        """この名前に**いま値が束縛されているか**を CAS に訊く。

        H-1 の P2 が送った束縛（`y : x^3`）の有無である。帳簿側の
        `defining_expr_id` は上位集合で、解の結果は id を持つのに束縛しない
        （実測 2026-09-19）ので、変数の欄を守る判定はこちらが決める
        （`operand_guards.translate_value_bearing_variable_failure`）。

        綴りは**そのバックエンドの知識**なので、既定は `False` ——
        `classify_cas_error` と同じで、答えられないバックエンドは何もしない。
        """
        return False

    def cas_error_args(self, error: str) -> dict:
        """Format arguments for the message ``classify_cas_error`` selected.

        Kept separate from the key so the caller can translate without
        knowing which backend produced the failure, and so messages that need
        no substitution stay unaffected (the default is an empty dict, which
        makes ``t()`` behave exactly as it does with no arguments at all)."""
        return {}

    # A-11 case B: complete site-unspecified operators when the session uses a
    # single, unambiguous site (e.g. a one-spin Larmor model).  Backends that need
    # explicit operator sites opt in by setting this flag and overriding the hooks.
    completes_operator_sites: bool = False

    def spin_sites_in(self, canonical: str) -> "set[str]":
        """The distinct site indices used by sited spin operators in *canonical*."""
        return set()

    def complete_bare_spin_sites(self, canonical: str, site: str) -> str:
        """Append *site* to any site-unspecified registered spin operator."""
        return canonical

    def normalize_display_latex(self, latex: str) -> str:
        """Normalize a user-supplied LaTeX string into the canonical display form.

        Called at expression-registration time.  Applies universal KaTeX-compat
        fixes and user-input shorthands that must NOT live in JS.  Subclasses
        should call ``super().normalize_display_latex(latex)`` first.
        """
        import re
        # --- 1. Strip surrounding math-mode delimiters ---
        latex = re.sub(r'^\s*\$\$\s*|\s*\$\$\s*$', '', latex.strip())
        latex = re.sub(r'^\s*\$\s*|\s*\$\s*$', '', latex.strip())
        latex = re.sub(r'^\s*\\\[\s*|\s*\\\]\s*$', '', latex.strip())


        # --- 2. KaTeX-incompatible LaTeX commands ---
        # \rm is obsolete; KaTeX requires \mathrm
        latex = re.sub(r'\{\\rm\s+([^}]+)\}', r'\\mathrm{\1}', latex)
        latex = re.sub(r'\\rm\b', r'\\mathrm', latex)
        # \mbox is not supported by KaTeX; use \text
        latex = re.sub(r'\\mbox\{([^{}]*)\}', r'\\text{\1}', latex)

        # --- 2b. A Greek name typed in ASCII gets its LaTeX command ---
        # The same answer `beautify_bare_canonical_latex` gives.  Without it the
        # two display paths disagreed on the SAME input: an H-1 definition showed
        # `\phi_{m,s}` while a plain expression showed the italic word "phi", and
        # the second one is what reached the published `.tex`
        # (measured 2026-08-09; the `\dot{theta}` EXAMPLES-DRIFT finding again).
        latex = restore_decorated_name(latex)
        latex = _BARE_GREEK_RE.sub(_bare_greek_sub, latex)
        latex = _GREEK_SUBSCRIPT_RE.sub(_greek_subscript_sub, latex)

        # --- 3. Superscript asterisk: ^* or ^{*} → ^{\ast} ---
        # Must be done BEFORE the * → \cdot step below — the same order, and now
        # the same rule, `beautify_bare_canonical_latex` needs (see
        # normalize_superscript_star).
        latex = normalize_superscript_star(latex)
        # …and the star belongs on the FUNCTION, whoever wrote it.  Run after
        # the step above so `f(x)^*` and `f(x)^{\ast}` are one spelling by the
        # time this scans for it.
        latex = _conjugate_star_onto_name(latex)

        # --- 4. Maxima-style superscript parens: ^(expr) → ^{expr} ---
        latex = _convert_caret_parens(latex)

        # --- 5. Bare function names → LaTeX commands (user typing sin(...)) ---
        latex = re.sub(
            r'(?<!\\)\b(sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan'
            r'|sinh|cosh|tanh|log|ln|exp|det|lim)\b(?=\s*[({])',
            r'\\\1', latex
        )

        # --- 7. sqrt(x) → \sqrt{x}  (skip if already preceded by backslash) ---
        # NOTE: the same conversion is mirrored on the frontend in
        # ui/notebook_main.js (prepareLaTeX).  The `(?<!\\)` guard prevents an
        # already-escaped \sqrt from becoming \\sqrt (KaTeX renders \\ as a line
        # break).  Keep both copies in sync when changing this rule.
        latex = re.sub(r'(?<!\\)\bsqrt\{([^{}]*)\}', r'\\sqrt{\1}', latex)
        # Balanced scan, not `[^()]*`: the regex could not see past a nested
        # paren, so `sqrt(4*V^2*\lambda^2 + (Ea-Eb)^2)` — an ordinary quadratic
        # discriminant — stayed CAS-spelled in the published TeX beside a result
        # that read `\frac{\sqrt{…}}{2}` (measured 2026-08-09).  The other half
        # of this project already had the scanning version; this is that same
        # answer rather than a third one.
        latex = _sqrt_call_to_radical(latex)

        # --- 8. `vec(...)` is the system's own wrapper, not the researcher's ---
        # `beautify_bare_canonical_latex` calls this restore and names itself
        # "the shared entrance for turning a canonical into a display" — but
        # THIS is the other shared entrance (`engine.register_expression` and
        # the operation route both come here), and it did not ask.  So a
        # researcher who wrote `\mathbf r` was shown `vec(r)`: raw CAS spelling
        # on screen, which is Core Principle #1 (measured 2026-09-10, found
        # displaying `v(\mathbf r - \mathbf r')`, a GF-NOTATION family line).
        latex = BackendHandler._restore_vector(latex)
        latex = restore_limit_subscript(latex)

        return latex

    def treats_bracket_as_commutator(self, text: str = "") -> bool:
        """`[X, Y]` / `{X, Y}` を**括弧の操作**として扱うか（式として登録せず）。

        **既定は「このバックエンドに二項括弧の操作があるか」を訊く。** 直す前は
        無条件に False で、註は「Maxima では `[a,b]` はリスト」と述べていたが、
        Maxima には `poisson_bracket` があり **Phase 0 は `[A, B]` を読んでいる**
        ——同じ問いに2つの答えがあり、`[A, B] を計算して` は
        `operands_provided` が True（括弧操作である）と答える一方でここが False
        と答えたため、**先頭の式として登録されていた**（実測 2026-09-06、
        canonical `[A, B]`）。

        **行列リテラルは括弧の操作ではない。** `[0, -b*c/d; a*d/b, 0]` は
        Maxima の行列で、実在する（コーパスの `lotka_volterra_stability`）。
        本物の二項括弧は `;` を含まない——この判別は Phase 0 が既に持っており、
        ここでも同じものを使う。
        """
        if ";" in (text or ""):
            return False
        from notebook.runtime.operation_validator import OperationValidator
        if self._SCHEMA_BACKEND is None:
            return False
        for op in ("commutator", "poisson_bracket", "anticommutator"):
            spec = OperationValidator().get_spec(op) or {}
            if self._SCHEMA_BACKEND in (spec.get("supported_backends") or []):
                return True
        return False

    def operands_provided(self, text: str) -> "bool | None":
        """Backend hook for E-id injection decision.

        Return True  — operands are already in the text; skip injection.
        Return False — operands are missing; injection is needed.
        Return None  — no opinion; defer to the shared LLMSupervisor logic.

        ``[X, Y]`` / ``{X, Y}`` bracket notation (commutator on julia_full,
        Poisson bracket on maxima — same syntax, backend-exclusive operations)
        always carries both operands explicitly, regardless of backend — this
        check is shared, not backend-specific behavior (Backend Encapsulation
        Rule), so it lives here rather than duplicated per subclass.  Without
        it, interpret_pipeline.py's "bare formula → just register it" fallback
        (which has no opinion on operator syntax) claims `[X, Y]` as a literal
        list/matrix to register before the rule-based bracket-commutator
        parser ever sees it.

        Override further in backends with additional special input syntaxes
        where operands are embedded in notation rather than listed as E-ids.
        """
        t = text.strip()
        # Exclude ";" (a Maxima matrix-literal row separator, e.g.
        # "[2, 1; 1, 2]") — that is a bare formula to auto-register, not a
        # two-operand bracket operation; the pure-formula fallback must still
        # claim it (regression found via test_matrix_literal.py /
        # test_lotka_volterra_scenario.py, 2026-07-09).
        from notebook.runtime.operand_boundary import is_bracket_operation_text
        if is_bracket_operation_text(t) and ';' not in t:
            return True
        return None

    def uneval_diff_match(self, canonical: str) -> "re.Match[str] | None":
        """Match *canonical* against the backend unevaluated-diff pattern.

        Returns a ``re.Match`` when the canonical encodes an unevaluated
        derivative (e.g. Maxima's ``'diff(f(t),t,1)``), otherwise ``None``.
        The default returns ``None``; ``MaximaBackendHandler`` overrides this.
        """
        return None

    def _declared_names_source(self, sym_part: str) -> str:
        r"""宣言文の「名前が並んでいる部分」を、**式の側と同じ読み**に直す。

        宣言はこの系のほとんどの読みの判別材料である（``^T``・``^\dagger``・
        ``^{-1}`` が転置／随伴／逆行列になるか、添字が index か名前か）。その名前を
        宣言だけが別の規則で読んでいたのが DECL-NAME-READING で、実測 2026-08-21:

            \hat H を行列とする        → `H` を宣言（`H_hat` ではない。`\hat H` と
                                          `H` は別の量で、`\hat H^T` は転置に
                                          ならないまま）
            declare G^0 as a matrix   → **成功を返して `declare` を宣言**

        так 直し方は2つ: 動詞を剥がすことと、綴りを ``latex_to_canonical`` ——
        式の側が使う扉——に通すこと。通したあとの名前は素の識別子（``H_hat``・
        ``G_sup0``）なので、下の走査はそのまま使える。
        """
        from notebook.runtime.shared_parser import UnknownNotationError
        text = _DECL_VERB_RE.sub("", sym_part)
        try:
            converted = self.latex_to_canonical(text)
        except (UnknownNotationError, ValueError, TypeError):
            # 読めない綴りでも宣言そのものは落とさない（従来の走査に任せる）——
            # `_alias_keys` が同じ理由で同じ3つを捕まえている。
            return text
        # **名前の扉は、名前を返さなければならない。** 式の扉は「文全体が数なら
        # それは数」と答えるので、`declare zero as a scalar` の名前が `0` になり、
        # 識別子が消えたまま下の走査に渡って**宣言そのものが落ち**、別名走査が
        # `declare` を宣言していた（実測 2026-09-10、`confirmed`）。訊いた問いは
        # 「名前として何を書いたか」なので、識別子の消えた答えは答えではない。
        if not re.search(_DECL_NAME_RE, converted) and re.search(_DECL_NAME_RE, text):
            return text
        return converted

    def extra_pre_phase1(self, text: str, lang: str) -> "object | None":
        """Shared type-declaration hook, called before Phase 1 of LLMSupervisor.

        Matches any "symbol を <label> とする" pattern.  The operation type is
        inferred from keywords inside *label* (ベクトル → vector, テンソル → tensor,
        行列 → matrix; anything else defaults to scalar).  This means phrases like
        「k を運動量ベクトルとする」or「g を計量テンソルとする」work without exact
        keyword matching; and「xi を１電子エネルギーとする」registers xi as scalar.

        Subclasses with additional backend-specific patterns should run their
        own checks first, then call ``super().extra_pre_phase1(text, lang)``
        as the fallback.
        """
        from notebook.runtime.llm_supervisor import InterpretResult
        # Normalize TeX Greek commands to canonical names so that e.g.
        # "\gamma と B を実数スカラーとして定義する" is processed correctly.
        # Only normalise inside potential symbol positions (before "を").
        # The full display LaTeX is preserved in the original `text`.
        # Normalize TeX Greek commands before は/を boundary (Japanese) or
        # before the as/is/are copula (English) — without this, an English
        # multi-symbol list like "Define k, m, \mu, \hbar as scalars" keeps
        # "\mu"/"\hbar" as backslash-prefixed tokens that the declaration symbol pattern cannot
        # match (it has no leading "\" in its character class), which breaks
        # the comma-repetition chain and makes the regex silently re-anchor
        # on "hbar" alone (search() finds a match starting after the stray
        # backslash), dropping every earlier symbol in the list.
        _pre_boundary = min(
            (text.find(b) for b in ("を", "は", "とは") if text.find(b) > 0),
            default=-1,
        )
        if _pre_boundary < 0:
            _en_cop_m = re.search(_EN_COPULA, text, re.I)
            if _en_cop_m:
                _pre_boundary = _en_cop_m.start()
        if _pre_boundary > 0:
            _before = _TEX_CMD_RE.sub(
                lambda _m: _TEX_GREEK_TO_CANONICAL.get(_m.group(1), _m.group(0)),
                text[:_pre_boundary],
            )
            text = _before + text[_pre_boundary:]
        # Match: (1) を…とする  (2) は/とは <label>  (3) English
        _m_wa = nlp("decl_stmt_wa", "ja").search(text)
        if _m_wa and _looks_like_type_label(_m_wa.groups()[-1].strip()):
            m = _m_wa
        else:
            m = _nl_search("decl_stmt", text)
        if not m:
            return None
        # Extract label from the last capture group.
        groups = [g.strip() for g in m.groups() if g]
        if len(groups) < 2:
            return None
        label = groups[-1]
        # Re-scan the matched text before を/は to collect ALL symbol names.
        # Using groups()[:-1] only captures the *last* symbol in a "A と B と C"
        # list because regex repetition (?:...)* overwrites the same group.
        full_match = m.group(0)
        _wo = full_match.find("を")
        _ha = full_match.find("は")
        _ja_boundary_found = _wo >= 0 or _ha >= 0
        if _ja_boundary_found:
            # Japanese pattern: re-scan text before を/は to collect ALL symbol names.
            # This is needed because regex repetition (?:...と...)* overwrites the same
            # capture group, so groups()[:-1] only retains the last symbol in a list.
            _boundary = min(i for i in [_wo, _ha] if i >= 0)
            sym_part = self._declared_names_source(full_match[:_boundary])
            # **区切りで切ってから**名前を拾う。素の `findall` は区切りの語
            # そのものを名前として拾う——英語側で `Define a and b as scalars`
            # が `and` を宣言していた（実測 2026-09-05、区切りを広げた直後）。
            # 区切りの答えは `operand_boundary.split_name_list` に1つだけある。
            raw_names = [n for piece in split_name_list(sym_part)
                         for n in re.findall(_DECL_NAME_RE, piece)
                         if not re.fullmatch(r"E\d+", n)]
        else:
            # English pattern: re-scan the text before the copula (" as "/" is ")
            # for ALL symbol names, mirroring the Japanese path above — relying on
            # m.groups()[:-1] silently dropped every name but the first and last
            # in a 3+-item comma list (same repeated-capturing-group limitation;
            # J-4, found via the S-s practice-loop scenario's English variant).
            _en_copula_m = re.search(_EN_COPULA, full_match, re.I)
            if _en_copula_m:
                sym_part = self._declared_names_source(
                    full_match[:_en_copula_m.start()])
                raw_names = [n for piece in split_name_list(sym_part, "en")
                             for n in re.findall(_DECL_NAME_RE, piece)]
            else:
                raw_names = [g.strip() for g in m.groups()[:-1] if g and g.strip()]
        if not raw_names:
            return None
        if not _looks_like_type_label(label):
            return None
        _refusal = self._refuse_declaration_this_backend_cannot_make(label, text)
        if _refusal is not None:
            return _refusal
        operation = _infer_type_from_label(label, self._SCHEMA_BACKEND)
        # **A sentence that says "spin operator" must never declare a SCALAR.**
        # The generic label inference defaults to scalar for anything it does not
        # recognise — right for「１電子エネルギー」, wrong for an operator, and
        # measured 2026-08-22 to be what a `magnetic_resonance` session did with
        # every spelling (the reading lived in the Julia handler alone).  The
        # declaration is RECORDED even where the CAS has no operator algebra:
        # spin_algebra is CAS-bypass, so there is nothing to tell the CAS, and
        # refusing at REGISTRATION would stop the researcher writing their own
        # Hamiltonian (CLAUDE.md, "write it down first, compute later").
        _spin_decl = read_spin_operator_declaration(text)
        if operation == "define_scalar_parameter" and _spin_decl:
            operation = "define_spin_operator"
        _extra_time_dep = (
            operation in ("define_scalar_field", "define_vector_field")
            and _label_wants_time_dependent(label)
        )

        # A Greek stem with digits glued to it is that Greek letter SUBSCRIPTED:
        # `omega0` is ω₀, and nobody means an atom spelled o-m-e-g-a-0.  Written
        # that way it can never display as anything but the italic word "omega0"
        # — putting the subscript in the display alone would be worse, because
        # the NAME the researcher must type back would no longer match what they
        # see (the researcher's call, 2026-08-09: correct it at declaration).
        # So the declared name itself becomes `omega_0`, which the digit-only
        # branch below already keeps whole and which displays as \omega_{0}.
        raw_names = [split_greek_digit_name(n) for n in raw_names]

        # Classify each name's subscript:
        #   digit-only  (omega_0)   → name part, keep as-is (unambiguous)
        #   multi-index (a_k_s, a_ks with comma) → component index, strip
        #   2+ alpha chars (H_BCS, k_B) → name part, keep as-is (unambiguous)
        #   single alpha char (xi_k)    → ambiguous, show confirmation dialog
        def _subscript_role(name: str) -> str:
            """Return 'keep', 'strip', or 'ask'."""
            # A DECORATION or a superscript LABEL is part of the name, not an
            # index, so it must not be counted as one — `\hat H_0` arrives here
            # as `H_hat_0`, whose TWO underscores read as "a multi-index" and
            # stripped it to `H`: a declaration of a different quantity, with
            # the message still naming `H_hat_0` (measured 2026-08-21).  The
            # marks are peeled first and put back after, so what the rule below
            # sees is the name's own subscript and nothing else.
            name = _without_decoration_pieces(name)
            if "_" not in name:
                return "keep"
            base, sub = name.split("_", 1)
            if sub.isdigit():
                return "keep"                          # omega_0, Delta_1
            if "," in sub or "_" in sub:
                return "strip"                         # a_k_s, a_{k,s}
            if len(sub) >= 2:
                return "keep"                          # H_BCS, k_B
            return "ask"                               # xi_k, v_i

        # A symbol already registered (e.g. "A_x" auto-created as a component
        # by an earlier define_vector_field("A")) has its subscript role already
        # settled by that registration — re-declaring it (e.g. to also mark it
        # time-dependent) must not re-open the ambiguity dialog.
        _ctx_for_ambiguity = getattr(self, "_context", None)

        def _already_registered(name: str) -> bool:
            return bool(_ctx_for_ambiguity is not None
                        and _ctx_for_ambiguity.get_quantity(name) is not None)

        # Applied to every non-ambiguous name below; defined here (before the
        # ambiguity branch) so both branches can resolve a MIXED list — e.g.
        # "c_u, c_o, xi_0" has c_u/c_o ambiguous and xi_0 unambiguous (digit
        # subscript, always kept).
        def _apply_subscript_role(name: str) -> str:
            role = _subscript_role(name)
            if role == "strip":
                # Strip the INDEX pieces, keep the decoration ones
                # (`c_hat_k_s` → `c_hat`): the marks are part of the name.
                _pieces = name.split("_")
                _kept = [_pieces[0]] + [p for p in _pieces[1:]
                                        if p in _DECORATION_PIECES]
                return "_".join(_kept)
            return name  # keep or ask (ask won't reach here)

        ambiguous = [n for n in raw_names
                     if _subscript_role(n) == "ask" and not _already_registered(n)]
        if ambiguous:
            from notebook.runtime.i18n import t as _t
            name0 = ambiguous[0]
            base0, sub0 = name0.split("_", 1)
            if len(ambiguous) > 1:
                prompt = _t("interpreter.subscript_role_prompt_multi", lang,
                            names=", ".join(ambiguous),
                            subs=", ".join(n.split("_", 1)[1] for n in ambiguous))
            else:
                prompt = _t("interpreter.subscript_role_prompt", lang,
                            name=name0, sub=sub0)

            def _resolve(name: str, as_index: bool) -> str:
                # The chosen interpretation (index vs. name-part) applies
                # uniformly to EVERY ambiguous name in this declaration, not
                # just the first — J-5, found via the S-v practice-loop
                # scenario: "c_u, c_o を実数とする" previously carried only
                # ambiguous[0] into _all_names, silently dropping the rest
                # (they stayed undeclared and later resolved as IndexedSymbol,
                # e.g. c_o → c(o)).
                if name in ambiguous:
                    return name.split("_", 1)[0] if as_index else name
                return _apply_subscript_role(name)

            names_as_index = [_resolve(n, True) for n in raw_names]
            names_as_name  = [_resolve(n, False) for n in raw_names]
            # Candidate A: subscript is an index → use base name
            args_as_index = {"name": names_as_index[0], "_all_names": names_as_index,
                             "_raw_names": raw_names}
            # Candidate B: subscript is part of the name → keep full name
            args_as_name  = {"name": names_as_name[0], "_all_names": names_as_name,
                             "_raw_names": raw_names}
            return InterpretResult(
                ok=True,
                operation=operation,
                args=args_as_index,  # default (shown on single confirm button)
                confidence=1.0,
                source="rule",
                requires_confirmation=True,
                confirmation_prompt=prompt,
                candidates=[
                    {
                        "operation": operation,
                        "args": args_as_index,
                        "display_ja": _t("interpreter.subscript_as_index", lang,
                                         base=base0, sub=sub0, name=name0),
                        "display_en": _t("interpreter.subscript_as_index", "en",
                                         base=base0, sub=sub0, name=name0),
                    },
                    {
                        "operation": operation,
                        "args": args_as_name,
                        "display_ja": _t("interpreter.subscript_as_name", lang,
                                         base=base0, sub=sub0, name=name0),
                        "display_en": _t("interpreter.subscript_as_name", "en",
                                         base=base0, sub=sub0, name=name0),
                    },
                ],
            )

        # No ambiguous subscript: apply keep/strip based on role.
        decl_names = [_apply_subscript_role(n) for n in raw_names]
        _decl_args: dict = {
            "name": decl_names[0],
            "_all_names": decl_names,
            "_raw_names": raw_names,
        }
        if wants_redeclare(text):
            # 既に別の型で宣言されていても通す合図。既定は断る（利用者の決定、
            # 2026-08-22）——宣言は既に書いた式の読み方を決めているので。
            _decl_args["_redeclare"] = True
        if _extra_time_dep:
            _decl_args["_also_time_dependent"] = True
        if operation == "declare_symmetry":
            # `declare_symmetry` は `name` だけでなく `symmetry` を要求する。
            # その値は**研究者が書いたラベルの中にある**（`対称行列` /
            # `hermitian matrix`）ので、そこから読む。書かれていない
            # （`対称性を宣言`）ときは埋めない——検証器が「symmetry が要る」と
            # 言うのが正しい。推測して埋めると、どの対称性かを言っていない文に
            # 黙って symmetric を入れることになる。
            _sym = symmetry_value_for_label(label)
            if _sym:
                _decl_args["symmetry"] = _sym
            # **`対称行列` は型と対称性の両方を言っている。** 片方だけ宣言すれば
            # 他方が落ちるので、基底の型を優先順位表から読んで一緒に渡す
            # （`反対称` のように型を言っていないラベルでは既定のスカラーになり、
            # 何も足さない）。
            _base = _type_from_keywords(label)
            _algebra = TYPE_DECL_ALGEBRA.get(_base)
            if _algebra and _algebra != "scalar":
                _decl_args["algebra"] = _algebra
        if _spin_decl and operation == "define_spin_operator":
            # The NUCLIDE the researcher wrote travels with the declaration: it
            # is the only thing that fixes the spin (`I` is 1/2 for ¹³C and 0 for
            # ¹²C, so "carbon" does not decide it), and it was read and thrown
            # away until now.
            #
            # **The whole sentence is carried, not the captured description**,
            # because the two languages put the nuclide on opposite sides:
            # `^{13}C の核スピン演算子` before, `nuclear spin operator of 13C`
            # after.  Measured 2026-08-22 — with the description alone, English
            # recorded nothing.  Searching the sentence is safe because a nuclide
            # spelling always carries its mass number, so a bare element letter
            # (the operator's own name `I`) can never match.
            _decl_args["_spin_desc"] = text
        return InterpretResult(
            ok=True,
            operation=operation,
            args=_decl_args,
            confidence=1.0,
            source="rule",
            requires_confirmation=False,
        )

    @staticmethod
    def spin_declaration_message(args: dict, lang: str) -> "str | None":
        r"""核種が読めたときだけ、**何をどう読んだか**を返す。

        読めたことが画面に出なければ、研究者は $^{13}$C が伝わったかを確かめられない
        ——「判別できたなら、そう言う」（このプロジェクトの警告の規則の裏返し）。
        """
        from notebook.runtime.i18n import t as _t
        from notebook.runtime.spin_value_set import (find_nuclide, format_numerator,
                                                     ladder_numerators)
        nuc = find_nuclide(str(args.get("_spin_desc", "") or ""))
        if nuc is None:
            return None
        two_s = int(nuc["two_s"])
        return _t("interpreter.spin_op_defined_nuclide", lang,
                  mass=nuc["mass_number"], element=nuc["element"],
                  spin=format_numerator(two_s), count=len(ladder_numerators(two_s)))

    def format_void_op_message(self, operation: str, args: dict, lang: str) -> "str | None":
        """Return a completion message for type-declaration void operations.

        Handles the four shared type-declaration operations; subclasses may
        add further cases before delegating to ``super()``.
        """
        from notebook.runtime.i18n import t as _t
        if operation == "define_spin_operator":
            # Shared, not Julia-only: since 2026-08-22 a Maxima session declares
            # spin operators too (spin_algebra is CAS-bypass, so there is nothing
            # to tell the CAS — but "spin operator" must not become a scalar).
            return (self.spin_declaration_message(args, lang)
                    or _t("interpreter.spin_op_defined", lang))
        _TYPE_DECL_KEYS: dict[str, str] = {
            "define_scalar_parameter": "interpreter.scalar_param_defined",
            "define_vector":           "interpreter.vector_defined",
            "define_matrix":           "interpreter.matrix_defined",
            "define_tensor":           "interpreter.tensor_defined",
        }
        key = _TYPE_DECL_KEYS.get(operation)
        if key:
            # Symbol name is no longer in the message text — it is rendered via
            # KaTeX (latex) and shown in parentheses (symbol_display) by JS.
            if operation == "define_scalar_parameter":
                domain = str(args.get("domain", "") or "").lower().strip()
                sign = str(args.get("sign", "") or "").lower().strip()
                if domain or sign:
                    # "positive" alone (no explicit kind, e.g. "n を正とする") defaults
                    # the displayed kind to "real" for legibility only — the stored
                    # Quantity.domain itself is left None; this is display text, not
                    # a new inferred CAS domain.
                    domain_label = _t(f"domain.{domain}", lang) if domain else (
                        _t("domain.real", lang) if sign else "")
                    if sign:
                        sign_label = _t(f"sign.{sign}", lang)
                        combined = (f"{sign_label}{domain_label}" if lang == "ja"
                                    else f"{sign_label} {domain_label}".strip())
                    else:
                        combined = domain_label
                    return _t("interpreter.scalar_param_defined_domain", lang, domain=combined)
            return _t(key, lang)
        return None

    def clean_void_result(self, value: str) -> str:
        """Map a CAS-specific 'no meaningful output' sentinel to an empty string.

        Backends return backend-specific tokens when a side-effect-only operation
        produces no mathematical output (Julia → ``"nothing"``, Maxima → ``"true"``).
        These are implementation details that must not appear in the UI.

        The base implementation suppresses Julia's ``"nothing"``; subclasses
        extend the list as needed (e.g. Maxima also suppresses ``"true"``).
        """
        return "" if value == "nothing" else value

    def _canonical_name_to_latex(self, name: str) -> str:
        """Convert a bare canonical name (e.g. 'xi', 'Delta') to its LaTeX form.

        Checks the registered quantity's explicit *latex_name* first, then falls
        back to the module-level Greek letter table ``GREEK_CMD``.  Single-letter
        names and unknown names are returned unchanged.
        """
        return canonical_name_to_latex(name, getattr(self, "_context", None))

    _TYPE_DECL_LATEX_OPS: frozenset[str] = frozenset({
        "define_scalar_parameter", "define_vector",
        "define_matrix", "define_tensor",
    })

    def get_void_op_symbol_display(self, operation: str, args: dict) -> str:
        """Return the plain-text symbol name shown in parentheses after the
        rendered LaTeX symbol in the confirmation message.

        Examples:
        - define_scalar_parameter({"name":"xi"}) → "xi"
        - define_fermion_operator({"_op_expr_display":"c'(k)"}) → "c'(k)"

        Subclasses override for backend-specific operator expressions.
        The base class handles type-declaration operations.
        """
        if operation in self._TYPE_DECL_LATEX_OPS:
            raw_names = args.get("_raw_names") or [args.get("name", "")]
            return ", ".join(n for n in raw_names if n)
        return ""

    def get_void_op_display_latex(self, operation: str, args: dict) -> str:
        """Return a LaTeX string for inline display next to the void-op confirmation.

        The shared base handles the four type-declaration operations by mapping
        symbol names (e.g. ``'xi'``, ``'Delta'``) to proper LaTeX commands
        (``'\\xi'``, ``'\\Delta'``).  Subclasses may extend for backend-specific
        operations (e.g. Julia operator definitions) before delegating here.
        """
        if operation in self._TYPE_DECL_LATEX_OPS:
            # Multi-symbol declarations ("R、L、w、C をスカラーとする") must show ALL
            # names.  The interim LLM-supervision step carries `_raw_names`, but the
            # persisted cas_operation step (what TeX export re-renders) carries
            # `registered_names` / `_all_names` with `name` set to only the first —
            # so honour those too, else only the first symbol would be displayed.
            raw_names = (
                args.get("_raw_names")
                or args.get("registered_names")
                or args.get("_all_names")
                or [args.get("name", "")]
            )
            parts = [self._canonical_name_to_latex(n) for n in raw_names if n]
            return ", ".join(parts)
        return ""

    def find_undefined_symbols(self, ir_ast: "Expr | None", context: "SymbolContext") -> list[str]:
        """Return bare names of symbols in *ir_ast* that are not registered in *context*.

        Used after ``register_expression`` to show a UI warning (⚠ banner).
        Does NOT block CAS execution — see ``blocks_on_undefined_symbols`` for that.

        Backend behaviour:
        - Maxima: flags multi-character unknowns; standard Greek letters are in
          ``_KNOWN_SYMBOL_NAMES`` and therefore skipped.
        - Julia: flags all unregistered symbols except those in ``_KNOWN_SYMBOL_NAMES``
          and single-letter free variables handled natively by the CAS.
        - Default (base class): returns ``[]`` — override in each backend subclass.
        """
        return []

    def blocks_on_undefined_symbols(self, ir_ast: "Expr | None", context: "SymbolContext") -> list[str]:
        """Return symbol names that should BLOCK a CAS operation when undefined.

        Unlike ``find_undefined_symbols`` (used for UI warnings), this method
        controls whether ``run_operation`` raises ``UndefinedSymbolsError``.

        Backend behaviour:
        - Maxima: never blocks — free variables are handled natively.
        - Julia: blocks multi-character symbols (base name length > 1).
          Single-letter symbols (x, a, i, j, …) are treated as free algebraic
          variables and are never blocked.
        - Default (base class): returns ``[]``.

        The length > 1 threshold is an intentional heuristic documented in
        CLAUDE.md §Undefined Symbol Detection. Revisit if edge cases arise.
        """
        return []

    supports_operator_adjoint: bool = False
    """Whether ``adjoint_canonical`` is implemented (operator algebra backends)."""

    def simplify_raw_command(self, canonical: str) -> "str | None":
        """Return a CAS command that simplifies an already-CAS-valid *canonical*
        string WITHOUT re-parsing it through the IR pipeline.

        Used by the H-1 formula evaluation ("EOM = E3 - E4"): the expanded RHS is
        a valid CAS expression, and re-parsing it (SharedParser→IR→codegen) can
        drop parentheses around a subtracted multi-term group (turning A−(−b−c)
        into A−b−c).  Default: None (caller falls back to build_command).
        """
        return None

    def adjoint_canonical(self, canonical: str) -> "str | None":
        """Return the canonical of the Hermitian adjoint of a linear operator
        expression, or None when unsupported.

        Operator adjoint is backend-specific (only meaningful where a
        non-commutative operator algebra exists).  Default: None.  Overridden by
        the Julia handler for the L-1 canonical-constraint derivation.
        """
        return None

    def pre_process_canonical(self, canonical: str) -> str:
        """Normalize a canonical expression string before registration.

        Called by ``NotebookEngine.register_expression`` before parsing and storing.
        Default: identity. Backend subclasses override to apply backend-specific
        normalization (e.g. JuliaBackendHandler converts \\uparrow → u).
        """
        return canonical

    # A bracket holding a ';' row separator is a MATRIX ("[2, 1; 1, 2]"); ';'
    # marks rows (no ';' → commutator/list, per the agreed convention).  This
    # neutral matrix(...) form is backend-independent — each CodeGenerator renders
    # it to its own literal (Maxima matrix([..],[..]); Julia [.. ..; .. ..]).
    _MATRIX_LITERAL_RE = re.compile(r"\[([^\[\];]*;[^\[\]]*)\]")

    def normalize_matrix_literal(self, s: str) -> str:
        """Convert a ';'-row bracket literal to the neutral matrix(...) form."""
        if not s or ";" not in s:
            return s
        def _repl(m: "re.Match") -> str:
            rows = [r.strip() for r in m.group(1).split(";")]
            return "matrix(" + ", ".join("[" + r + "]" for r in rows) + ")"
        return self._MATRIX_LITERAL_RE.sub(_repl, s)

    def matrix_literal_to_pmatrix(self, latex: str) -> str:
        """Render a ';'-row matrix literal "[a, b; c, d]" as a LaTeX pmatrix for
        display (no-op when there is no ';')."""
        if not latex or ";" not in latex:
            return latex
        def _repl(m: "re.Match") -> str:
            rows = [r.strip() for r in m.group(1).split(";")]
            body = " \\\\ ".join(" & ".join(c.strip() for c in r.split(","))
                                 for r in rows)
            return r"\begin{pmatrix} " + body + r" \end{pmatrix}"
        return self._MATRIX_LITERAL_RE.sub(_repl, latex)

    def replay_session_declarations(self, adapter: object, commands: list[str]) -> None:
        """Re-execute declaration commands after session load.

        Called by ``NotebookEngine.start()`` when the engine was restored from a
        saved session.  *commands* contains the ``rendered_command`` strings for
        every successful declaration step in the active branch lineage
        (``define_fermion_operator``, ``define_boson_operator``,
        ``define_spin_operator``, ``define_scalar_parameter``).

        Default: no-op.  Julia override executes each command so the fresh
        CAS process has the operator/variable declarations it needs.
        """

    # I18N-1: the session language, for the user-facing messages a handler
    # raises (a bad substitution, an unparsable notation).  Read through a
    # callable rather than snapshotted, because the language is assigned AFTER
    # the engine is built (ui/notebook_bridge.start_engine sets
    # session.language once the engine exists), so any value captured at
    # construction time would be stale.
    #
    # UNBOUND (no engine behind it — a unit test, a script) it reports
    # i18n.DEFAULT_LANG; see docs/developer_manuals/15_i18n_design.md.
    _lang_source: "Callable[[], str] | None" = None

    def bind_language_source(self, source: "Callable[[], str]") -> None:
        """Wire in the engine's live language accessor (``NotebookEngine.lang``)."""
        self._lang_source = source

    @property
    def lang(self) -> str:
        from notebook.runtime.i18n import DEFAULT_LANG
        return (self._lang_source() if self._lang_source is not None
                else DEFAULT_LANG)

    def set_symbol_context(self, context: "SymbolContext") -> None:
        """Wire in the shared SymbolContext after engine initialization or session load.

        Called by NotebookEngine at two points: after __init__ and after from_dict.
        Both backends store the reference identically and propagate it to their
        code generator, so this is a concrete (not per-backend) implementation.
        """
        self._context = context
        if self._builder and getattr(self._builder, "_compiler", None):
            self._builder._compiler.context = context

    # ------------------------------------------------------------------
    # User form → backend code (docs/developer_manuals/32, Phase 0).
    # The single derivation point; the two hooks below are the ONLY
    # backend-specific parts of it.
    # ------------------------------------------------------------------

    def vector_cas_view(self, text: str, value_of) -> "str | None":
        """外積を含む式の CAS 表示（持たないバックエンドは None）。

        研究者の綴り（`~`）は CAS が評価できないことがあり、その場合そのまま渡すと
        **生の CAS エラー**が返る（Core Principle #1）。どう綴り直すか——そもそも
        綴り直せるか——はバックエンドの知識なので、共有層はここで訊く。

        *value_of* は「名前 → その名前を定義した式の右辺」。ベクトルを成分のリストへ
        開くのに要る。
        """
        return None

    def to_backend_code(self, text: str) -> "Lowered":
        """Lower *text* to this backend's code under the bound SymbolContext.

        Never persist the result — the correct lowering depends on what is
        declared *right now* (``k_B`` is ``k(B)`` undeclared, ``k_B`` declared),
        so it must be re-derived per call.  See the module docstring of
        ``notebook.runtime.expr_lowering``.
        """
        from notebook.runtime.expr_lowering import to_backend_code as _lower
        return _lower(self, text)

    def missing_variable_expr_id(self, error: str) -> "str | None":
        """Expression id whose CAS-side variable is missing, per *error*.

        A backend that keeps results in CAS variables can report that one is
        gone (the Python process restarted while the in-memory expressions
        survived); the engine then replays enough history to recreate it and
        retries once.  Both the error wording and the variable naming are
        backend-specific, so the recognition belongs here.  Default: ``None``
        (nothing to recover).  ``JuliaBackendHandler`` overrides.
        """
        return None

    def execute_multi_statement(self, adapter, operation: str, command: str):
        """Run *command* when this backend emits it as several statements.

        Returns the combined adapter result, or ``None`` when the command is a
        single statement for this backend (the caller then executes it
        normally).  Default: ``None`` — no backend behaviour in shared code, and
        no CAS-specific result type constructed there.
        ``MaximaBackendHandler`` overrides for ``define_coordinate_system``.
        """
        return None

    def is_opaque_handle(self, text: str) -> bool:
        """``True`` when *text* is a CAS-side handle, not a user expression.

        Such a string must reach the CAS verbatim.  Default ``False``
        (Maxima canonicals are real expressions); ``JuliaBackendHandler``
        overrides for its ``_nb_<id>`` variable references.
        """
        return False

    def generate_backend_code(self, ir: "Expr", hold_integrals: bool = False) -> "str | None":
        """Render a resolved IR node as this backend's code, or ``None``.

        ``None`` means the backend has no code generator for the IR (the text
        is then passed through unchanged).  ``MaximaBackendHandler`` overrides.
        """
        return None

    # ------------------------------------------------------------------
    # Backend capability hooks (default no-op / structure-preserving).
    # These let shared engine code branch on *capability* rather than on
    # backend identity (no ``isinstance``/``primary_backend ==`` in engine).
    # ------------------------------------------------------------------

    @property
    def binds_symbol_definitions(self) -> bool:
        """``True`` when defining a symbol (``H = RHS``) should be bound into CAS
        state at definition time (Maxima's evaluate-on-assignment semantics).

        Julia preserves structure and does not bind symbol definitions into the
        CAS, so the default is ``False``.  ``MaximaBackendHandler`` overrides to
        ``True``.
        """
        return False

    def init_coordinate_system(self, adapter: object, session: object) -> None:
        """Send coordinate-system setup commands to the CAS at engine start.

        Default: no-op (Julia has no coordinate-system CAS state).
        ``MaximaBackendHandler`` overrides with the actual ``nb_*`` function
        definitions derived from ``session.coordinate_system``.
        """

    def reserved_coord_vars(self, coord_system: str) -> "tuple[str, str, str]":
        """Return the coordinate variable names for *coord_system*.

        Default returns Cartesian ``("x", "y", "z")``; ``MaximaBackendHandler``
        overrides with per-system tables (spherical, cylindrical, …).
        """
        return ("x", "y", "z")

    def coord_component_suffixes(self, coord_system: str) -> "tuple[str, str, str]":
        """Return the field-component suffix triple for *coord_system*.

        Default returns Cartesian ``("x", "y", "z")``; ``MaximaBackendHandler``
        overrides with per-system tables.
        """
        return ("x", "y", "z")

    def prepare_sympy_wick_inputs(
        self, sympy_str: str, ctx: "SymbolContext"
    ) -> "tuple[str, str]":
        """Return ``(statistics_json, sympy_str)`` for the SymPy Wick path.

        Default returns a fermion-default statistics map and the unchanged
        expression.  ``JuliaBackendHandler`` overrides to collect per-operator
        statistics and convert operator notation to ``cd``/``c`` form before the
        ``_nb_sympy_wick`` call.
        """
        return ('{"__default__":"fermion"}', sympy_str)

    def re_evaluate_expr(
        self,
        canonical: str,
        substitutions: "dict[str, str]",
        expr_lookup: "dict[str, str]",
        adapter: "object | None" = None,
    ) -> "str | None":
        """Re-evaluate *canonical* with the given symbol substitutions.

        Used for self-consistent iterations (e.g. mean-field updates) where
        the same expression must be re-evaluated after one or more named
        parameters change value.

        Args:
            canonical:     The canonical string of the expression to re-evaluate.
            substitutions: Mapping {symbol_name: new_canonical_value}.
                           E.g. {"n_k": "0.45", "Delta": "0.1 * xi"}.
            expr_lookup:   Current expression-id → canonical map (for resolving
                           E-refs inside *substitutions* values).

        Returns:
            New canonical string after substitution, or None if unsupported.

        The default implementation returns None (not supported).
        Subclasses override for backend-specific strategies:
          - Maxima: subst([lhs=rhs, ...], canonical)
          - Julia:  multiple substitute() calls on Symbolics coefficients
        """
        return None

    def should_bypass_auto_register_formula(self, formula_str: str, op_text: str) -> bool:
        """Return True to skip inline-formula auto-registration for *op_text*.

        Two cases:
        1. Full sentence still contains the symbol: matched by ``decl_stmt``.
        2. Formula token already extracted; ``op_text`` is a suffix like
           "をベクトルとする" — matched by ``decl_suffix``.

        Subclasses may check additional patterns and fall back to ``super()``.
        """
        if _nl_any("decl_stmt", op_text):
            return True
        m = nlp("decl_suffix", "ja").search(op_text)
        if m and _looks_like_type_label(m.group(1)):
            return True
        m = nlp("decl_suffix", "en").search(op_text)
        if m and _looks_like_type_label(m.group(1).strip()):
            return True
        return False
