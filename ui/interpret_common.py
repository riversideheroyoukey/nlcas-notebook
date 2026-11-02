"""Shared low-level constants for the interpret-pipeline feature modules.

Extracted from interpret_pipeline.py (Phase 1 refactor) so feature-specific parser
modules and the main pipeline can share them without a circular import.
"""
from __future__ import annotations

import re

# E-id list and variable separators, shared by the solve and L-1 parsers.
_EIDS_RE = re.compile(r'E\d+', re.U)
_VARS_SEP_RE = re.compile(r'[,、と\s]+', re.U)


from notebook.runtime.nl_patterns import nlp, fragment  # noqa: E402
from notebook.runtime.number_words import (  # noqa: F401
    FULLWIDTH_DIGITS as _FULLWIDTH_DIGITS, number_literal, number_word)
from notebook.runtime.script_label import label_script_at
from notebook.runtime.shared_parser import strip_lhs  # noqa: E402

_EXPR_ID_RE = re.compile(r'^E\d+$')
_HAS_CJK_RE = re.compile(r'[぀-ヿ一-鿿＀-￯]')
def _is_nl_instruction(text: str, lang: str = "ja",
                       names_operation=None) -> bool:
    """True when *text* is a natural-language operation instruction, not a bare formula.

    Historically the pipeline used "contains CJK" as this signal, which silently
    treats every non-CJK input as a formula and so swallows English commands like
    "In E2, substitute …".  The language signal is externalized to
    i18n/nl_patterns.json ("instruction_marker"): CJK for ja, an imperative verb for
    en.  CJK always counts (covers mixed input and not-yet-migrated patterns).

    *names_operation* is the SECOND source, and the two answer different
    questions: the alias table says "this text names an operation", the verb
    list says "this text is imperative".  Measured 2026-09-10, neither alone is
    enough — the list held 35 of the schema's 166 English alias words, so
    ``relabel k to -k in E1`` registered as an expression; and 37 real inputs
    (``Find …``, ``Save the lattice figure``, ``subtract E12 from E11``) are
    named ONLY by the list, because their verbs are not operation aliases.
    Passed in rather than imported: which operations exist is the caller's
    knowledge (backend, tokenizer) — the rule
    ``operand_boundary.text_names_an_operation`` states it, with the numbers.
    A caller with no engine may omit it and gets the verb list alone.
    """
    if _HAS_CJK_RE.search(text):
        return True
    if nlp("instruction_marker", lang).search(text):
        return True
    if names_operation is not None and names_operation(text):
        return True
    # 語ではなく**形**で分かる指示が1つある: `E1 with x = 2`。`with` を
    # `instruction_marker` に足すのは誤りで、式として登録される行に6件出る
    # （`the Ising model with 2 sites` など）——ただしそれらは `=` を**含まない**
    # ので、この形なら安全（実測 2026-09-06）。直す前は canonical
    # `(a*x + b)*with*x = 2` が登録され、`registered` が返っていた。
    return bool(nlp("expr_with_assignment", lang).match(text))

#: Fullwidth digits → ASCII.  A researcher typing Japanese produces them from
#: the IME without noticing, so any field read as a NUMBER has to fold them.
#: (interpret_angular and interpret_spin_species each carry an older private
#: copy with extra characters of their own; this is the one new code uses.)
#: **本体は `notebook/runtime/number_words.py`** —— 数詞の表がそちらへ降りた
#: ときに一緒に降りた（`notebook.runtime` の2箇所が数詞を訊くので、`ui` に
#: 置いたままだと表が2枚になる）。ここは既存の import 元を保つための再輸出。
FULLWIDTH_DIGITS = _FULLWIDTH_DIGITS


def _find_expr_by_canonical(engine: NotebookEngine, canonical: str) -> str | None:
    # Compare against the stored (pre-processed) canonical: a matrix literal
    # "[2,1;1,2]" is stored as matrix([2,1],[1,2]), so normalize the lookup key
    # too — otherwise the same matrix re-registers as a duplicate expression.
    canonical = engine._handler.normalize_matrix_literal(canonical)
    active = engine._active_expr_ids()
    for eid, expr in engine.expressions.items():
        if eid in active and expr.canonical == canonical:
            return eid
    return None
# Matches expression IDs embedded in a formula string, e.g. "J * E1 * E4"
_INLINE_EXPR_ID_RE = re.compile(r'\bE(\d+)\b')
def _expand_expr_ids(text: str, engine: "NotebookEngine") -> str:
    """Replace expression IDs (E1, E4 …) in *text* with their canonical strings.

    E.g. ``"J * E1 * E4"`` → ``"J * (Sx_1 + i * Sy_1) * (Sx_2 - i * Sy_2)"``

    Only substitutes IDs that exist in ``engine.expressions`` AND are reachable
    in the active branch (``_active_expr_ids()``).  Unknown IDs are left as-is
    so they surface as undefined-symbol warnings rather than silent errors.

    The substituted canonical is wrapped in parentheses to preserve operator
    precedence when embedded in a larger expression.
    """
    active_ids = engine._active_expr_ids()

    def _replace(m: re.Match) -> str:
        eid = f"E{m.group(1)}"
        if eid not in active_ids:
            return m.group(0)
        expr = engine.expressions.get(eid)
        if expr is None:
            return m.group(0)
        # Expand to the expression's actual value (canonical).  Only fall back to
        # input_canonical when the canonical is an opaque Julia reference (_nb_Ek)
        # that cannot be substituted into a formula string.  Preferring
        # input_canonical unconditionally was wrong for value-producing results
        # (e.g. a diagonalize result whose input_canonical is the SOURCE matrix).
        canonical = expr.canonical
        if (not canonical or canonical.startswith("_nb_")) and expr.input_canonical:
            canonical = expr.input_canonical
        if not canonical:
            return m.group(0)
        # A sum written WITHOUT a range keeps no range in its canonical — that
        # is the researcher's own writing — and its placeholder pair lives in
        # the IR.  Spliced as raw text it re-parses HERE and is handed a fresh
        # pair, so `E1 - E2` did not cancel even though E2 is E1's own result
        # (EXPR-REF-PROVENANCE cause B).  The operation path already asks this
        # question; asking it through the same door is the whole fix.
        placeholder_view = engine.placeholder_cas_view(eid)
        if placeholder_view:
            canonical = placeholder_view
        # Strip outer equation form "LHS = RHS" → use RHS only.  Bracket-aware:
        # `"=" in canonical` reads a bounded sum's own subscript as the
        # equation and drops the head (shared_parser.strip_lhs).
        canonical = strip_lhs(canonical)
        return f"({canonical})"

    return _INLINE_EXPR_ID_RE.sub(_replace, text)
def _expand_symbol_refs(text: str, engine: "NotebookEngine") -> str:
    """Replace named symbol references in *text* with their defining expression
    canonical.

    This enables decorated symbols like ``\\bar{H}``, ``\\langle ab \\rangle``,
    ``\\frac{dH}{dt}`` — and plain names like ``H`` — to be used as operands in
    calculation formulas, not just as operation targets.

    For example, if ``\\bar{H}`` was registered (via H-1) as E3 with canonical
    ``J * Sz * Sz``, then::

        "2 * \\bar{H} + V"  →  "2 * (J * Sz * Sz) + V"

    Rules
    -----
    - Only expands entries in ``engine.lhs_alias_map`` where
      ``_extract_simple_symbol_name(key)`` returns non-None (atomic symbols).
      Compound LHS entries like ``"H + V"`` are NOT expanded here — they would
      cause ambiguity in formula parsing.
    - Substitutions are applied longest-key-first to avoid partial matches
      (e.g. ``\\bar{H}`` is expanded before ``H``).
    - For plain identifiers (no backslash), word-boundary guards are applied
      so ``H`` in ``\\hat{H}`` is not substituted.
    - The canonical is wrapped in parentheses to preserve operator precedence.
    - The canonical's own LHS label (if any) is stripped: only the RHS is used.
    """
    from notebook.engine import _extract_simple_symbol_name

    if not engine.lhs_alias_map:
        return text

    active_ids = engine._active_expr_ids()

    # Collect expandable entries sorted by key length (longest first)
    candidates: list[tuple[str, str, re.Pattern]] = []
    for lhs_key, expr_id in engine.lhs_alias_map.items():
        if _extract_simple_symbol_name(lhs_key) is None:
            continue  # compound / unmappable — skip
        if expr_id not in active_ids:
            continue
        expr = engine.expressions.get(expr_id)
        if expr is None:
            continue
        canonical = expr.input_canonical or expr.canonical
        if not canonical:
            continue
        # Strip LHS label from canonical if present (bracket-aware; `:=` is
        # excluded inside find_top_level_relation, so no second guard here).
        canonical = strip_lhs(canonical)
        if not canonical:
            continue
        # Build pattern(s) for lhs_key.
        # For expectation-value forms, add alternative notations:
        #   \langle X \rangle  ↔  <X>  (and vice versa)
        patterns: list[str] = []
        if lhs_key.startswith("\\langle") or lhs_key.startswith("<"):
            # Build both \langle ... \rangle and <...> forms
            # Extract base content (between angle brackets)
            _inner_m = re.match(r'^(?:<|\\langle\s*)(.+?)(?:\s*>|\\rangle)(.*)$', lhs_key)
            if _inner_m:
                inner = _inner_m.group(1).strip()
                suffix = _inner_m.group(2).strip()  # e.g. "_0"
                inner_nospace = re.sub(r'\s+', '', inner)
                # Both forms
                for form in (
                    re.escape(f"\\langle {inner} \\rangle{suffix}"),
                    re.escape(f"\\langle {inner}\\rangle{suffix}"),
                    re.escape(f"<{inner_nospace}>{suffix}"),
                    re.escape(f"<{inner}>{suffix}"),
                ):
                    patterns.append(form)
        elif lhs_key.startswith("\\") or "{" in lhs_key:
            patterns.append(re.escape(lhs_key))
        else:
            # Plain identifier: guard against:
            # - being part of a longer name (H in H_eff)
            # - being inside a LaTeX command (\hat{H}, \bar{H})
            # (?<![\\A-Za-z_{]) prevents matching inside \hat{H}, {H} etc.
            patterns.append(
                r'(?<![\\A-Za-z_{}])' + re.escape(lhs_key) + r'(?![A-Za-z_0-9{}\\])'
            )
        for pat_str in patterns:
            try:
                pat = re.compile(pat_str)
                candidates.append((lhs_key, canonical, pat))
            except re.error:
                continue

    if not candidates:
        return text

    candidates.sort(key=lambda c: len(c[0]), reverse=True)

    result = text
    for _key, canonical, pat in candidates:
        # The canonical is DATA, never a replacement template.  A canonical
        # keeps the researcher's own writing (CLAUDE.md), which includes TeX
        # that no CAS spelling replaces — `\sum_i*x_i`, `\int_0^1*x^2*dx`.
        # Passing that as a template makes `re` read `\s`/`\i` as escapes and
        # raise PatternError, so registering "S = \sum_i x_i" and then asking
        # for its right-hand side crashed the request (measured 2026-08-01).
        # A function replacement is substituted literally.
        # A LABEL superscript makes a DIFFERENT name: `G^0` is the free
        # propagator, not the defined `G`.  Expanded blindly it became
        # `(G^0 + G^0\Sigma G)^0` — and anything to the power 0 is 1 — so
        # `\left(1-G^0\Sigma\right)^{-1}` silently turned into
        # `(1-\Sigma)^{-1}` (measured 2026-08-21).  The same question, asked of
        # the same module, as in `supervisor_alias_rewrite`; an EXPONENT
        # (`G^2`) still expands, because that is the defined quantity squared.
        result = pat.sub(
            lambda _m, _c=canonical: (_m.group(0)
                                      if label_script_at(_m.string, _m.end())
                                      else f"({_c})"),
            result)

    return result
# Matches inputs with 2+ whitespace-separated tokens (natural-language phrase, not a single identifier)
_MULTI_WORD_RE = re.compile(r'\S+\s+\S+')


def math_operand(engine, text: str) -> str:
    """Normalise a fragment an NL rule captured as MATHS, once and in one place.

    The researcher writes back what the notebook DISPLAYED, which is TeX, so
    every interceptor that captures an operand — an integrand, a variable, a
    transform rule — has to convert it before handing it on.  Stating that
    conversion at each site is how it ends up applied at some and not others:
    2026-08-01 the definite-integral path got it and the change-of-variables
    path beside it did not, so `\theta` worked in one and silently missed the
    polar-bounds table in the other (open_items S-C-MEASURE-RESIDUALS).

    Idempotent, so an already-canonical operand passes through unchanged; empty
    input and conversion failure both fall back to the original text rather
    than inventing something.  Unicode Greek is NOT handled here — that is
    normalised once for the whole request in `run_interpret_pipeline`, before
    any interceptor sees it.
    """
    raw = (text or "").strip()
    if not raw:
        return raw
    try:
        return (engine._handler.latex_to_canonical(raw) or raw).strip()
    except (ValueError, AttributeError):
        # An operand we cannot convert is left as the researcher wrote it; the
        # caller's own validation decides whether that is usable.
        return raw


def normalize_request_greek(f1_raw: str, tokens) -> str:
    r"""Unicode Greek → TeX, ONCE per request, before any rule-based interceptor.

    CLAUDE.md promises that `ω` in NL input is converted automatically, and
    `normalize_unicode_greek` does exactly that — but it was called at the top of
    ``LLMSupervisor.interpret()``, i.e. AFTER every rule-based interceptor had
    already looked at the raw text.  Measured 2026-08-02: `θ^2 を θ で 0 から 1
    まで定積分して` missed the definite-integral rule entirely and failed, while
    the identical `\theta` spelling worked — and the failure message showed
    `\theta`, proof that the conversion happened just too late to help
    (open_items INTEGRAL-VAR-NOTATION).

    Doing it here is what keeps this from becoming another spelling table: every
    interceptor sees ONE spelling and none needs a Unicode branch of its own —
    the same reason `math_operand` above exists for captured operands.  Both
    backends gained it at once (measured: Julia returns 1/3 for `θ` too).

    Mutates the tokens in place and returns the normalised joined text.
    The YOU card is recorded BEFORE this runs, so the log keeps what the
    researcher actually typed (the rule H1-ALIAS-EID-LEAK established).

    **MATH tokens are normalised too** (2026-08-10).  Until then only ``text``
    tokens were, so the 2026-08-02 repair above stopped one token type short:
    ``ω`` typed into the FORMULA box — or handed over by the handwriting OCR —
    reached Maxima as the symbol ``ω``, which no declared ``\omega`` can ever
    match.  The two spellings of one quantity then lived in the same session as
    different objects, and every operation on them reported success.

    The two fields are not the same kind of text, so they are not treated the
    same.  ``latex`` IS LaTeX, and ``\omega`` is its correct spelling.
    ``canonical`` is CAS code, where ``\omega`` would be wrong — but it is also
    optional (``empty → server converts from latex``), so a canonical carrying
    Unicode is DROPPED and re-derived from the normalised latex rather than
    patched here.  Inventing a second Unicode→canonical map in this layer is
    exactly the duplicate-answer defect this file keeps closing.
    """
    from notebook.runtime.backend_handler import normalize_unicode_greek
    for tok in tokens:
        if getattr(tok, "type", None) == "text" and tok.value:
            tok.value = normalize_unicode_greek(tok.value)
        elif getattr(tok, "type", None) == "math":
            if tok.latex:
                tok.latex = normalize_unicode_greek(tok.latex)
            if tok.canonical and normalize_unicode_greek(tok.canonical) != tok.canonical:
                tok.canonical = ""          # re-derived from the normalised latex
    return normalize_unicode_greek(f1_raw)


def assumption_display_latex(raw: str) -> str:
    r"""Display form for an assumption whose condition was typed as literal text.

    The card sets the condition in MATH mode, so the raw text is not display
    latex: a typed `omega > 0` published as the italic WORD "omega", not ω —
    found by the .tex scan (notebook/runtime/tex_sanity.py) in two committed
    examples.

    This is a NAME-SPELLING question, not a computation, so it must not go
    through the CAS-backed display door: measured, that door EVALUATES an
    H-1-defined symbol, and the damped-oscillator card turned the researcher's
    own `D > 0` into `4 \omega_{0}^2-4 \gamma^2>0` — the name they had just
    defined, gone.  `beautify_bare_canonical_latex` is pure text and leaves `D`
    alone.  The relation spellings it does not carry are mapped here, because
    `#` is a LaTeX parameter character and must not reach the page raw.

    ONE function because the question is asked in two places — the rule-based
    interceptor (`_try_add_assumption`) and the supervisor-routed branch in the
    pipeline — and two copies of an answer is the defect this file keeps closing.
    """
    from notebook.runtime.backend_handler import beautify_bare_canonical_latex
    out = beautify_bare_canonical_latex(raw) or raw
    # The trailing space is required, not cosmetic: `a>=b` would otherwise become
    # `a\geqb`, a command name that does not exist.  Doubled spaces are collapsed
    # because the researcher usually already typed one.
    for src, tex in ((">=", r"\geq "), ("<=", r"\leq "), ("#", r"\neq ")):
        out = out.replace(src, tex)
    return re.sub(r"\s{2,}", " ", out).strip()


def operands_as_written(raw_match, pick, names, current):
    r"""Named operands, re-read from what the RESEARCHER typed.

    2a-sym rewrites a symbol that is some expression's left side into that
    expression's E-id.  The EXPRESSION REFERENCE needs that; a NAMED OPERAND
    must not have it — the two have opposite requirements and one string cannot
    serve both (``InterceptorContext.alias_free_text``).

    ``_parse_solve`` reached this answer twice, a rule at a time: the system
    branch in 2026-08-16 (`solve E1 and E2 and E3 for a, b, c` sent
    ``solve([…], [a, b, E3])`` and answered as a success) and the single-equation
    branch in 2026-08-21.  **Substitution is the third rule with the same
    question**, and measured 2026-08-27 it answered with the E-id: after
    ``solve E1 for x``, ``E1 の x を 2 に置換して`` substituted nothing and
    reported completed — 4 of 10 real spellings, the English ``substitute … for
    x`` forms escaping only because the alias guard's ``for`` happens to cover
    them.  So the answer lives here now, once.

    *raw_match* is the rule's OWN pattern re-run against the researcher's text
    (None when it does not match there, in which case nothing is re-read);
    *pick* reads a group the way that rule reads it.
    """
    if raw_match is None:
        return list(current)
    return [(pick(raw_match, name) or "").strip() or cur
            for name, cur in zip(names, current)]


#: 数詞の読みは `notebook/runtime/number_words.py` に在る。**ここから降りた**——
#: `notebook.runtime` の2箇所（`_looks_like_type_label`・`to_canonical`）が同じ
#: 問いを訊き、`ui` からは import できないので、置いたままにすると表が2枚になる。
#: 既存の `from ui.interpret_common import number_word` を保つための再輸出。
#: 値として書かれた数は `number_literal`（基数と数字だけ）が読む。


# --- テイラー展開の展開中心 ---------------------------------------------------
#: 研究者が書いた中心の綴りから中心の値を読む。`x=0` は `0` である。
#: これは `token_field_mapping.POINT_RE` が既に答えている問いで、そこから
#: 借りている——`taylor_expand_nl[en]` の `(?P<point>[^\s]+)` は `around x=0`
#: を**丸ごと**取り、Maxima に `taylor(f, x, x=0, 3)` が渡って研究者に
#: `Maxima evaluation error` という生の CAS 文言が返っていた（実測 2026-09-05）。
from notebook.runtime.token_field_mapping import POINT_RE  # noqa: E402
from notebook.runtime.i18n import t  # noqa: E402


def taylor_center_note(point_written: "str | None", lang: str) -> "tuple[str, str | None]":
    """展開中心について研究者に何を伝えるか —— **1つの答え**。

    この問いはパイプラインの 3c-pre / 3c（supervisor 経路）にしかなく、
    `interpret_integral` のテイラー・インターセプタは
    ``point = (m.group("point") or "0")`` と**黙って 0 を入れて**いた。
    結果、同じ要求が層によって別の応答になっていた（実測 2026-09-05、
    ``E1 = exp(x)``）:

        E1 を x について 3 次までテイラー展開して   → **黙って** completed
        E1 を x について 3次でテーラー展開して      → 「マクローリン展開とします。
                                                  実行しますか？」と訊く

    args は両方とも同一（``{expr, var, point: '0', order: '3'}``）で、
    「中心が書かれていない」という事実は読んだ層に依らない。訊くほうを残した
    ——文面は既に i18n にあり、**黙って選ぶより研究者に多くを伝える**から。

    返すのは (中心の値, 伝える文 or None)。中心が **書かれていて 0 でない**
    ときも読み取った値を伝える（3c-pre が既にそうしていた）。
    """
    if point_written is None or not str(point_written).strip():
        return "0", t("interpreter.taylor_maclaurin", lang)
    p = str(point_written).strip()
    m = POINT_RE.search(p)
    if m and m.group(0) == p:      # `x=0` と書かれた —— 中心は 0
        p = m.group(1)
    if p != "0":
        return p, t("interpreter.taylor_center_detected", lang, point=p)
    return p, None


#: 名前の並びの区切りは `operand_boundary` の族（CLAUDE.md:
#: 「この族は増える見込みだから次の判別子はそこへ」）。ここは再輸出。
from notebook.runtime.operand_boundary import split_name_list  # noqa: E402,F401


#: alternation のどの枝が一致したかを読む道具は `nl_patterns` の領分
#: （パターンを持つ module がその一致も読む）。ここは再輸出。
from notebook.runtime.nl_patterns import pick_group  # noqa: E402,F401

#: 名前の並びの区切りは `operand_boundary` の族（CLAUDE.md:
#: 「この族は増える見込みだから次の判別子はそこへ」）。ここは再輸出。
from notebook.runtime.operand_boundary import split_name_list  # noqa: E402,F401
