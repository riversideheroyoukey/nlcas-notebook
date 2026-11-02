"""LaTeX export HTTP endpoint — extracted from notebook_bridge (R2).

Owns ``GET /export/latex`` and its self-contained TeX-rendering helpers (escape
tables, YOU-card / Adviser-memo → LaTeX conversion).  The path composes under the
bridge's ``/api/notebook`` prefix, so it is byte-for-byte the same as before the
split; only its home moved.

Everything here is pure except the handler's ``get_engine()`` call, which is
imported lazily inside the body (cycle + patch safety — see ui/routes/history.py).
``session_io`` and ``beautify_bare_canonical_latex`` are leaf utilities imported
at module top.
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter

from notebook.runtime.backend_handler import beautify_bare_canonical_latex
import ui.session_io as session_io

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


_TEX_ESCAPE = {
    '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
    '#': r'\#', '^': r'\^{}', '_': r'\_', '{': r'\{', '}': r'\}',
    '~': r'\textasciitilde{}',
    # `<` and `>` are not LaTeX specials, which is why they were missing — but
    # they are not PRINTABLE in text mode either: the preamble deliberately omits
    # fontenc (it breaks uplatex, see docs), so OT1 is in force and `<`/`>` set
    # as `¡`/`¿`.  Measured: the YOU card «D > 0 と仮定する» typeset as «D ¿ 0».
    # The two-character spellings are separate keys: someone who typed
    # `x^2+y^2<=1` wrote ≤, and escaping the `<` alone leaves `$<$=1` — "< =".
    '<=': r'$\le$', '>=': r'$\ge$', '<': r'$<$', '>': r'$>$',
}

# ONE pass, longest key first — never a chain of str.replace.  Several
# replacements CONTAIN characters that are themselves keys (`\^{}`, `$<$`,
# `\textbackslash{}`), so a sequential table re-escapes what an earlier row just
# inserted: measured, `x^2` came out `x\^\{\}2` and `a\b` came out
# `a\textbackslash\{\}b`.  Neither reached a published .tex only because such
# text is usually routed to math mode instead — the ordering was one caller away
# from being wrong.  A single sub() cannot see its own output.
_TEX_ESCAPE_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(_TEX_ESCAPE, key=len, reverse=True)))


def _tex_escape(text: str) -> str:
    """Escape LaTeX special characters in plain (non-math) text."""
    return _TEX_ESCAPE_RE.sub(lambda m: _TEX_ESCAPE[m.group()], text)


# A YOU-card whose text is a bare formula (no Japanese, with LaTeX commands or
# math operators) is rendered as inline math so e.g. "\nabla \cdot E" shows as a
# formula, not an escaped "\textbackslash{}nabla …" mess.  Natural-language cards
# (which contain CJK) keep plain-text escaping.
_CJK_RE_TEX = re.compile(r"[぀-ヿ一-鿿＀-￯]")
_FORMULA_INDICATOR_RE = re.compile(r"\\[A-Za-z]+|[=^]|[A-Za-z0-9_]\s*[+\-*/]\s*[A-Za-z0-9_(\\]")

# A math fragment embedded in an otherwise natural-language YOU card, e.g. the
# "\rho" / "\lambda/\mu" in "E2 の \rho に \lambda/\mu を代入して" or the "S^z_1" in
# "… and S^z_1".  A fragment is either a LaTeX command (extending over its arguments
# / scripts and further commands joined by math operators, so "\lambda/\mu" is one
# span) OR a plain identifier carrying a super/subscript ("S^z_1", "x_i").  These are
# wrapped in inline math rather than backslash-escaped.
_SCRIPT_ID_RE = re.compile(r"[A-Za-z]+(?:\^\{?[A-Za-z0-9]+\}?|_\{?[A-Za-z0-9]+\}?)+")


def _decoration_prefix_one() -> str:
    r"""``(?:\\hat\s*|\\bar\s*|…)*`` —— **引数を取るコマンドは引数と離さない**。

    実測 2026-08-22（利用者報告）: `\hat H_0 をスカラーとする` が公開 TeX に
    `$\hat$ $H_0$ をスカラーとする` と書き出され、**`$\hat$` 単体は不正な TeX**
    なので組版が止まる。断片の規則が「コマンド＋波括弧の引数」しか続かず、
    **空白で区切られた引数**を知らなかった。今日直した導関数の走査と同じ形で、
    直し方も同じ——一覧は再掲せず、変換器の accent 表から作る。
    """
    from notebook.runtime.latex_converter import _DECORATION_SUFFIXES
    names = {c.lstrip("\\") for _, c in _DECORATION_SUFFIXES}
    names |= {"widehat", "widetilde", "overline", "overbar", "vec", "mathbf",
              "boldsymbol"}
    return r"\\(?:" + "|".join(sorted(names, key=len, reverse=True)) + r")\s*"


_MATH_FRAG_RE = re.compile(
    # 装飾コマンド＋その引数＋添字。**引数と離さない**のがこの枝の全部で、
    # 引数は素の名前（`\hat q`）・コマンド（`\hat\psi`）・波括弧（`\hat{H}`）の
    # どれでもよい。
    r"(?:" + _decoration_prefix_one() + r")+"
    r"(?:\{[^{}]*\}|\\[A-Za-z]+|[A-Za-z][A-Za-z0-9]*)"
    r"(?:\^\{[^{}]*\}|_\{[^{}]*\}|[\^_]\\[A-Za-z]+|[\^_][A-Za-z0-9])*"
    # 演算子で続く分も1つの断片（`\dot{\theta} / r`）。この続きを付け忘れると、
    # 装飾の枝が先に当たったぶんだけ断片が短くなり、`/ r` が本文書体に落ちる
    # （実測 2026-08-22、再生成の差分で気づいた）。
    r"(?:\s*[/*+^_-]\s*\\?[A-Za-z0-9]+"
    r"(?:\{[^{}]*\}|[\^_][A-Za-z0-9])*)*"
    r"|"
    r"\\[A-Za-z]+"                                       # leading command
    r"(?:\{[^{}]*\}|\^\{[^{}]*\}|_\{[^{}]*\}|[\^_][A-Za-z0-9])*"   # args / scripts
    r"(?:\s*[/*+^_-]\s*\\?[A-Za-z0-9]+"                  # operator + next token
    r"(?:\{[^{}]*\}|[\^_][A-Za-z0-9])*)*"
    r"|"
    # 素の名前＋スクリプト（`S^z_1`, `x_i`, `c^\dagger_{1\uparrow}`）。
    # **波括弧は釣り合っていること。**  `\{?…\}?` と書いてあったので `c_{1`
    # までで断片が切れ、続きが本文として `$c_{1$$\uparrow$\}` と出力されて
    # uplatex が `Missing } inserted` で止まっていた（実測 2026-08-22）。
    # 波括弧つき → コマンド → 素の実行、の順（先に素を試すと `p_\theta` が
    # `p_t` で止まり `heta` が本文に落ちる）。素の実行は**1文字に狭めないこと**:
    # `normal_order` が `normal_o` + `rder` に割れる（実測 2026-08-22、再生成の差分）。
    r"[A-Za-z]+(?:[\^_]\{[^{}]*\}|[\^_]\\[A-Za-z]+|[\^_][A-Za-z0-9]+)+"
)


# Math function names that may appear in a bare formula and must NOT be counted as
# natural-language words (so "sin(x)+cos(x)" stays a bare formula, not mixed text).
_YOU_MATH_FUNCS = frozenset({
    "sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "sqrt",
    "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "det", "diag",
})


def _has_nl_words(s: str) -> bool:
    """True if *s* contains natural-language words (so it is an instruction, not a
    bare formula).  Counts ASCII words of ≥3 letters that are neither a known math
    function nor a LaTeX command name (\\lambda etc.).  Language-agnostic: an English
    instruction has such words, a bare formula ("\\nabla \\cdot E", "sin(x)+cos(x)")
    does not.  Threshold 2 mirrors interpret_pipeline's pure-formula guard."""
    latex_cmds = {m.group(1).lower() for m in re.finditer(r"\\([a-zA-Z]+)", s)}
    n = sum(1 for w in re.findall(r"\b[a-zA-Z]{3,}\b", s)
            if w.lower() not in _YOU_MATH_FUNCS and w.lower() not in latex_cmds)
    return n >= 2


def _you_card_latex(text: str) -> str:
    """Render a YOU-card: inline math for a bare formula, escaped text otherwise.

    A YOU card may be (1) a bare formula typed as text ("\\nabla \\cdot E") → whole
    inline math; (2) a natural-language instruction in any language with embedded
    LaTeX ("E2 の \\rho に…", "In E2, substitute \\lambda/\\mu for \\rho") → escape
    the prose, render the LaTeX fragments inline; (3) plain prose → escaped text.
    The bare-formula branch is gated on having NO natural-language words so an
    English instruction (no CJK) is not mistaken for a formula."""
    s = text.strip()
    if (s and not _CJK_RE_TEX.search(s) and _FORMULA_INDICATOR_RE.search(s)
            and not _has_nl_words(s)):
        body = beautify_bare_canonical_latex(s)  # '*'→space, sigma→\sigma, sqrt()→\sqrt{}
        return f"${_unicode_to_math_commands(body)}$"
    if "\\" in s or _SCRIPT_ID_RE.search(s):
        # Mixed natural-language + math: render each math fragment (LaTeX command or
        # script-bearing identifier) inline, escape the surrounding prose.  Avoids
        # "\textbackslash{}lambda" and "S\^\{\}z\_1".
        out, last = [], 0
        for m in _MATH_FRAG_RE.finditer(s):
            out.append(_tex_escape(s[last:m.start()]))
            frag = beautify_bare_canonical_latex(m.group(0))
            out.append(f"${_unicode_to_math_commands(frag)}$")
            last = m.end()
        out.append(_tex_escape(s[last:]))
        return "".join(out)
    return _tex_escape(text)


# Unicode Greek + math symbols that appear in Adviser memos as plain text and
# would otherwise fail to compile under (pdf/up)latex.  Mapped to LaTeX commands
# and wrapped in inline math by _text_unicode_to_math.
_NOTE_UNICODE_TO_TEX: dict[str, str] = {
    # Greek lowercase / uppercase
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "Γ": r"\Gamma",
    "δ": r"\delta", "Δ": r"\Delta", "ε": r"\varepsilon", "ζ": r"\zeta",
    "η": r"\eta", "θ": r"\theta", "Θ": r"\Theta", "ι": r"\iota",
    "κ": r"\kappa", "λ": r"\lambda", "Λ": r"\Lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "Ξ": r"\Xi", "π": r"\pi", "Π": r"\Pi",
    "ρ": r"\rho", "σ": r"\sigma", "Σ": r"\Sigma", "τ": r"\tau",
    "υ": r"\upsilon", "φ": r"\varphi", "Φ": r"\Phi", "χ": r"\chi",
    "ψ": r"\psi", "Ψ": r"\Psi", "ω": r"\omega", "Ω": r"\Omega",
    # operators / relations / symbols
    "√": r"\surd", "∞": r"\infty", "×": r"\times", "·": r"\cdot",
    "±": r"\pm", "∓": r"\mp", "≈": r"\approx", "≃": r"\simeq", "≅": r"\cong",
    "≤": r"\leq", "≥": r"\geq", "≪": r"\ll", "≫": r"\gg", "≠": r"\neq",
    "≡": r"\equiv", "∝": r"\propto", "→": r"\to", "←": r"\leftarrow",
    "↔": r"\leftrightarrow", "⇒": r"\Rightarrow", "⇔": r"\Leftrightarrow",
    "↑": r"\uparrow", "↓": r"\downarrow", "⟨": r"\langle", "⟩": r"\rangle",
    "∂": r"\partial", "∇": r"\nabla", "∫": r"\int", "∮": r"\oint",
    "∑": r"\sum", "∏": r"\prod", "∈": r"\in", "∉": r"\notin",
    "⊂": r"\subset", "⊃": r"\supset", "⊆": r"\subseteq", "∪": r"\cup",
    "∩": r"\cap", "⊗": r"\otimes", "⊕": r"\oplus", "†": r"\dagger",
    "‡": r"\ddagger", "ℏ": r"\hbar", "ℓ": r"\ell", "∀": r"\forall",
    "∃": r"\exists", "∅": r"\emptyset", "°": r"^{\circ}", "′": r"\prime",
    "∥": r"\parallel", "⊥": r"\perp",
    # Confirmation-message glyph: several i18n strings (e.g.
    # interpreter.declared_time_dep) are prefixed with "✓ " for UI display and
    # reach the TeX narrative via record_quantity_update()'s manual_note step.
    # Plain "✓" (U+2713) is outside the default upLaTeX font's glyph coverage
    # and breaks compilation if left as a raw Unicode character; \checkmark
    # (amssymb, already in both preambles) renders the same glyph safely.
    "✓": r"\checkmark",
}

_NOTE_UNICODE_RE = re.compile(
    "(?:" + "|".join(re.escape(c) for c in _NOTE_UNICODE_TO_TEX) + ")+"
)


def _unicode_to_math_commands(text: str) -> str:
    r"""Unicode のギリシャ文字・記号を、**数式の中に置ける**コマンドへ。

    `_text_unicode_to_math` は同じ表を使って `$…$` で包む（本文用）。数式の中では
    包んではいけないので、コマンドだけを置く——表は1つ、包み方が2つである。

    実測 2026-08-22: 研究者が `sin(θ)^2` と打った YOU カードが `$sin(θ)^2$` として
    書き出され、**uplatex が `Not two-byte family` で止まっていた**（公開実例
    scenario_integration_variable_notation_en の65行目）。表はこのファイルに
    あったのに、数式にする枝が使っていなかった。
    """
    def _repl(m: "re.Match") -> str:
        # コマンドの直後が英数字だと名前が繋がる（`\theta x`）ので空白で切る。
        return _NOTE_UNICODE_TO_TEX[m.group()] + " "
    return _NOTE_UNICODE_RE.sub(_repl, text)


def _text_unicode_to_math(text: str) -> str:
    """Wrap runs of Unicode Greek/math symbols in a single inline-math group.

    Grouping consecutive symbols into one ``$...$`` avoids producing adjacent
    ``$$`` (which (la)tex would parse as a display-math delimiter).
    """
    def _repl(m: "re.Match") -> str:
        cmds = " ".join(_NOTE_UNICODE_TO_TEX[c] for c in m.group())
        return f"${cmds}$"
    return _NOTE_UNICODE_RE.sub(_repl, text)


#: Markdown bold inside a note.  Non-greedy, no newline, non-empty — a lone `*`
#: is the BULLET marker this same function's caller reads, and must not be
#: touched.  Math is stashed before this runs, so `**` inside `$…$` is safe.
_NOTE_BOLD_RE = re.compile(r'\*\*(?!\s)([^*\n]+?)\*\*')


def _note_inline(segment: str) -> str:
    """Render one non-math text segment: escape, bold, Unicode→math.

    Two bold spellings, because the scenario documents are read BOTH as markdown
    (on GitHub) and as the source of the published TeX: `【…】` is this
    system's own convention, `**…**` is what the scenario authors actually
    write.  Measured 2026-08-22: eleven published `.tex` files carried literal
    asterisks into the PDF — it compiles, and it reads as a typo.
    """
    s = _tex_escape(segment)
    s = re.sub(r'【([^】]*)】', r'\\textbf{\1}', s)
    s = _NOTE_BOLD_RE.sub(r'\\textbf{\1}', s)
    s = _text_unicode_to_math(s)
    return s


def _note_to_latex(text: str) -> str:
    """Convert Adviser-memo text to compilable LaTeX.

    - Inline / display math (``$…$`` / ``$$…$$``, the latter may span lines) is
      protected and preserved verbatim.
    - ``*`` / ``-`` / ``・`` bullet lines become an ``itemize`` environment.
    - ``【…】`` → ``\\textbf{…}``; Unicode Greek/math symbols → inline math.
    - Remaining consecutive text lines are joined with LaTeX line breaks.
    """
    # 1. Protect math (inline + multi-line display) with placeholders so that
    #    line/bullet/Unicode processing never touches math content.
    stash: list[str] = []

    def _protect(m: "re.Match") -> str:
        stash.append(m.group(0))
        return f"\x00M{len(stash) - 1}\x00"

    protected = re.sub(r'\$\$[\s\S]*?\$\$|\$[^$\n]+?\$', _protect, text)

    out: list[str] = []
    text_buf: list[str] = []
    bullets: list[str] = []

    def flush_text():
        if text_buf:
            out.append(" \\\\\n".join(text_buf))
            text_buf.clear()

    def flush_bullets():
        if bullets:
            out.append(r"\begin{itemize}")
            out.extend(rf"  \item {b}" for b in bullets)
            out.append(r"\end{itemize}")
            bullets.clear()

    _disp_only = re.compile(r'^\x00M(\d+)\x00$')
    for line in protected.split("\n"):
        stripped = line.strip()
        if not stripped:
            flush_bullets()
            flush_text()
            out.append("")
            continue
        dm = _disp_only.match(stripped)
        if dm and stash[int(dm.group(1))].startswith("$$"):
            # Standalone display-math line: own paragraph, never suffixed with \\.
            flush_bullets()
            flush_text()
            out.append(stripped)
            continue
        m = re.match(r'^\s*[*\-・]\s+(.*\S)\s*$', line)
        if m:
            flush_text()
            bullets.append(_note_inline(m.group(1)))
        else:
            flush_bullets()
            text_buf.append(_note_inline(line))
    flush_text()
    flush_bullets()

    result = "\n".join(out)
    # 2. Restore protected math.
    for i, blk in enumerate(stash):
        result = result.replace(f"\x00M{i}\x00", blk)
    return result


@router.get("/export/latex")
async def export_latex():
    """Export the active branch timeline as a self-contained LaTeX document."""
    from ui.notebook_bridge import get_engine
    from notebook.runtime.replay import compute_lineage
    from notebook.models.step import StepKind as SK
    from notebook.runtime.lattice_figure import tikz_figure_for_operation

    engine = get_engine()
    lineage = compute_lineage(engine.active_branch_id, engine.steps, engine.branches)

    handler = engine._handler

    lang  = engine.lang
    title = _tex_escape(engine.session.title)

    tex_dir = _PROJECT_ROOT / "i18n" / "tex"
    preamble_path = tex_dir / f"{lang}.tex"
    if not preamble_path.exists():
        preamble_path = tex_dir / "en.tex"
    preamble_lines = preamble_path.read_text(encoding="utf-8").splitlines()

    lineage_ids_tex = {s.step_id for s in lineage}
    companions_tex: dict = {}
    for s in engine.steps.values():
        if s.step_id in lineage_ids_tex:
            continue
        if s.branch_id != engine.active_branch_id:
            continue
        if s.kind != SK.user_input or not s.user_text:
            continue
        parent_id = s.parent_step_ids[0] if s.parent_step_ids else None
        companions_tex.setdefault(parent_id, []).append(s)
    _tex_snum = lambda s: int(s.step_id[1:]) if s.step_id[1:].isdigit() else 0
    for key in companions_tex:
        companions_tex[key].sort(key=_tex_snum)

    # supervision step args lookup (for _raw_names / _op_expr_display recovery)
    _supervision_args: dict = {}
    for s in engine.steps.values():
        if s.kind == SK.llm_supervision and s.operation_args:
            key = tuple(sorted(s.parent_step_ids))
            _supervision_args[key] = s.operation_args

    lines = preamble_lines + [
        '',
        f'\\title{{{title}}}',
        r'\date{\today}',
        '',
        r'\begin{document}',
        r'\maketitle',
        '',
    ]

    for step in lineage:
        # Insert plain-text paragraph for companion user_input steps
        parent_id = step.parent_step_ids[0] if step.parent_step_ids else None
        for comp in companions_tex.get(parent_id, []):
            lines.append(_you_card_latex(comp.user_text))
            lines.append('')

        if step.kind == SK.user_input and step.user_text:
            lines.append(_you_card_latex(step.user_text))
            lines.append('')
        elif step.kind in (SK.expression_registration, SK.cas_operation):
            for ref in step.output_refs:
                expr = engine.expressions.get(ref)
                # skip temp intermediates: hidden ones and scaffolding that
                # never had an E number (INTERNAL-INTERMEDIATE-EID-GAP)
                if expr and not (expr.hidden or expr.internal):
                    tex_latex = handler.clean_latex(expr.latex)
                    # Remove explicit multiplication signs left from canonical input
                    tex_latex = re.sub(r'\s*\*\s*', r' ', tex_latex)
                    lines.append(r'\begin{equation}')
                    lines.append(f'  {tex_latex}')
                    lines.append(f'  \\label{{eq:{expr.display_id}}}')
                    lines.append(r'\end{equation}')
                    lines.append('')
            # Lattice sum: drop the bond-structure figure next to its expansion.
            _fig = tikz_figure_for_operation(
                getattr(step, "operation", None), step.operation_args)
            if _fig:
                lines.append(_fig)
                lines.append('')
            if not step.output_refs and step.operation_args:
                _sup_key_tex = tuple(sorted(step.parent_step_ids))
                _sup_args_tex = _supervision_args.get(_sup_key_tex, {})
                _args_tex = {**_sup_args_tex, **(step.operation_args or {})}
                # Multi-symbol declarations must show ALL names — render every
                # declared symbol, not just the first.  The interim supervision
                # step carries `_raw_names`; the persisted cas_operation step
                # carries `registered_names` / `_all_names` (with `name` = first).
                _raw_names_tex = (
                    _args_tex.get("_raw_names")
                    or _args_tex.get("registered_names")
                    or _args_tex.get("_all_names")
                    or ([_args_tex["name"]] if _args_tex.get("name") else [])
                )
                if _raw_names_tex:
                    names_latex = ", ".join(
                        handler._canonical_name_to_latex(n) for n in _raw_names_tex if n)
                    lines.append(r'\[')
                    lines.append(f'  {names_latex}')
                    lines.append(r'\]')
                    lines.append('')
        elif step.kind == SK.assumption_add:
            # Prefer the TeX-facing display (e.g. "\gamma > 0") over the bare
            # Maxima-safe canonical ("gamma > 0") — same precedence as the
            # /timeline endpoint's assumption entries below.
            display_tex = (step.operation_args.get("display")
                           or step.operation_args.get("canonical", ""))
            lines.append(f'\\noindent\\textbf{{Assumption:}}\\ $\\displaystyle {display_tex}$\\par')
            lines.append('')
        elif step.kind == SK.manual_note and step.user_text:
            lines.append(_note_to_latex(step.user_text))
            lines.append('')

    lines.append(r'\end{document}')

    tex = '\n'.join(lines)
    safe_title = re.sub(r'[^\w\-]', '_', engine.session.title) or 'notebook'

    # Save alongside the current session file if already saved; otherwise sessions/ root.
    current = session_io.get_current_filename()  # e.g. "paper1/experiment.json" or None
    if current:
        tex_path = session_io.sessions_dir() / re.sub(r'\.json$', '.tex', current)
    else:
        tex_path = session_io.sessions_dir() / f'{safe_title}.tex'

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex, encoding='utf-8')

    rel = str(tex_path.relative_to(_PROJECT_ROOT))
    return {"ok": True, "path": rel}
