"""Interpretation pipeline: token normalization → LLM supervision → operation execution.

run_interpret_pipeline() is the single entry point called by the /interpret HTTP handler.
All helpers (token normalization, variable inference, inline-formula auto-registration)
live here so that notebook_bridge.py stays a thin HTTP router.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import re

from notebook.runtime.shared_parser import split_relation, strip_lhs
from functools import lru_cache, partial
from pathlib import Path
import sys

from pydantic import BaseModel

_logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from notebook.engine import NotebookEngine
from notebook.runtime.backend_handler import beautify_bare_canonical_latex
from notebook.runtime.i18n import t
from notebook.runtime.backend_handler import _looks_like_type_label
from notebook.runtime.nl_patterns import nlp, nlp_any, nlp_search
from notebook.runtime.operand_boundary import (
    genitive_head_not_an_operation,
    EXPR_ID_IN_TEXT_RE, formula_is_an_operation_operand, is_bracket_operation_text,
    may_fall_back_to_the_recent_expression, name_is_usable_as_an_operand,
    split_leading_compute_verb,
    text_names_an_operation)
from notebook.runtime.llm_supervisor import LLMSupervisor, _try_nested_vec_op, _try_single_vec_op
from notebook.runtime.spin_algebra import ComplexityLimitExceeded as SpinAlgebraComplexityLimitExceeded

from ui.supervisor_alias_rewrite import rewrite_symbols_to_eids
from ui.interpret_common import (normalize_request_greek, _EIDS_RE, _EXPR_ID_RE, _HAS_CJK_RE, taylor_center_note, number_word, number_literal,  # noqa: F401
                                 _VARS_SEP_RE, _is_nl_instruction,
                                 assumption_display_latex)
from ui.interpret_notation import (  # noqa: F401
    try_norm_notation,
    _try_notation_candidates,
)
from ui.interpret_integral import (  # noqa: F401
    _llm_derive_bounds, _latest_integral_form, _parse_new_var_spec,
    _split_top_commas, _try_change_vars, _try_definite_integral,
    _try_integral_in_coordinates,
    _try_taylor_expand,
    _try_series_expand_no_var,
    _try_evaluate_integral, _try_expand_components, _try_integral_input,
)
from ui.interpret_extremum import (  # noqa: F401
    _try_evaluate_extremum, _try_extremum_input,
)
from ui.interpret_free_params import (  # noqa: F401
    _latest_free_param_expr, _parse_conditions,
    _try_add_assumption,
    _try_define_laplace, _try_solve_ode_general,
    _try_specialize_condition, _try_specialize_conditions,
    _try_substitute_free_params,
)
from ui.interpret_common import (  # noqa: F401
    _INLINE_EXPR_ID_RE, _MULTI_WORD_RE, _expand_expr_ids, _expand_symbol_refs,
    _find_expr_by_canonical,
)
from ui.interpret_equation_intercepts import _try_equation_intercepts  # noqa: F401
from ui.interpret_inequality import _try_inequality_intercept  # noqa: F401
from ui.interpret_rules import _try_rule_short_circuits  # noqa: F401
from ui.interpret_early import _try_early_intercepts  # noqa: F401
from ui.interpret_plan import (  # noqa: F401
    _try_plan_auto_select, _try_plan_candidates, _try_plan_freeform, _try_plan_workflow,
)
from ui.interpret_handoff import _try_scalar_handoff  # noqa: F401
from ui.interpret_numeric import _try_numeric_fork_nl  # noqa: F401
from ui.interpret_angular import (  # noqa: F401
    _try_clebsch_gordan,
    _try_couple,
    _try_coupled_state,
    _try_lande_g,
    _try_wigner_eckart,
    _try_wigner_symbol,
)
from ui.interpret_signal import (  # noqa: F401
    _try_z_transform_by_definition,
)
from ui.interpret_registry import _attribute_new_steps
from ui.interpret_registry import (
    Interceptor, InterceptorContext, run_interceptors,
)
from ui.interpret_equation_ops import (
    _match_rhs_op,
    _both_sides_new_lhs,
    _parse_both_sides,
    _parse_rhs_to_operation,
    _parse_solve,
)

from ui.interpret_f1 import (
    _is_save_lattice_figure,
    _ns_extract_body_after,
    _parse_neighbor_sum,
    _parse_symbolic_bond_sum,
    _parse_concretize,
    _spin_model_kind,
    _parse_spin_commutator,
    _parse_spectral_function,
    _parse_spin_spectral,
    _parse_specific_heat,
    _parse_energy_moment,
    _parse_graph_spec,
    _parse_crystal,
    _bond_sum_coeff,
    _parse_dispersion,
    _token_sub,
    _relabel_sites,
    _parse_bond_sum_commutator,
    _parse_neighbor_sum_concretize,
    _parse_bond_sum_substitute,
    # graph/spectral regexes used directly by the dispatcher
)

from ui.interpret_l1 import (
    _parse_inverse_transform,
    _parse_matrix_transform,
    _infer_old_basis,
    _new_basis_symbols,
    _parse_canonical_constraints,
    _is_verify_request,
    _parse_apply_constraints,
    _parse_inner_product,
    _parse_metric_constraints,
    _gram_rows,
    _parse_maxima_matrix,
    _parse_transform_subst,
    _subst_lhs_rhs,
    _op_name_index,
    _is_operator_symbol,
    _operator_statistics,
    _forward_has_operators,
    # internal helpers re-exported so existing import paths (tests) keep working
    _atom_base,
    _basis_atoms_in_order,
    _identifiers_in_order,
    _parse_bracket_matrix,
)


# ---------------------------------------------------------------------------
# Request models (imported by notebook_bridge.py)
# ---------------------------------------------------------------------------

class InputToken(BaseModel):
    type: str                # "text" | "math"
    value: str = ""          # for type=="text"
    latex: str = ""          # for type=="math"
    canonical: str = ""      # for type=="math"; empty → server converts from latex


class InterpretRequest(BaseModel):
    tokens: list[InputToken]
    selected_expr_id: str = ""   # expression selected in the UI when submit was pressed


# ---------------------------------------------------------------------------
# Token normalization
# ---------------------------------------------------------------------------

# U+2212 MINUS SIGN (−) is always a mathematical operator — convert unconditionally.
# U+30FC KATAKANA PROLONGED SOUND MARK (ー), U+2013 EN DASH (–), U+2014 EM DASH (—)
# look like minus only when surrounded by mathematical characters; use context check.
_MATH_DASH_RE = re.compile(
    r'(?<=[0-9A-Za-z.)^*/])'
    r'[–—ー]'
    r'(?=[0-9A-Za-z.(])',
)

#: The same dash, in a MATH token, where the researcher spaced it out.  A pasted
#: formula almost always reads `x – 1`, not `x–1`, and the strict rule above —
#: which requires the operands to touch the dash — left it alone: measured
#: 2026-08-15, `y = x – 1` reached the CAS as `y = x * 1`, a SUBTRACTION SILENTLY
#: TURNED INTO A MULTIPLICATION and registered as a success (the parser drops the
#: character it cannot read, and the two operands then juxtapose).  Spaces are
#: only safe to ignore inside a math token; in prose an em dash between words is
#: punctuation, so the strict rule keeps text tokens.
_MATH_DASH_SPACED_RE = re.compile(
    r'(?<=[0-9A-Za-z.)^*/])'
    r'(\s*)[–—ー](\s*)'
    r'(?=[0-9A-Za-z.(])',
)

#: U+2032 PRIME / U+2033 DOUBLE PRIME — what a word processor makes of `'`.
#: Primes are part of an IDENTIFIER here (CLAUDE.md: `a'(1)` is the creation
#: operator, `f''(x)` a second derivative), so a smart-quoted prime that the
#: parser cannot read does not merely disappear from the display — measured,
#: `f′(x)` reached the CAS as `f(x)`, the UNDIFFERENTIATED function, as a
#: success.  Math tokens only, for the same reason as the dash: `5′` is a
#: minute mark in prose.
_MATH_PRIME_MAP = {0x2032: "'", 0x2033: "''", 0x2035: "'"}


def _fullwidth_to_halfwidth(text: str, math: bool = False) -> str:
    """Convert full-width ASCII variants and math-only Unicode symbols to half-width ASCII.

    Converts:
    - U+FF01–FF5E (full-width ASCII variants) → half-width counterparts
    - U+3000 (ideographic space) → regular space
    - U+2212 (MINUS SIGN −) → hyphen-minus (always mathematical)
    - U+30FC/U+2013/U+2014 (ー/–/—) → hyphen-minus when between math characters
    Japanese kana and kanji are left unchanged.

    *math* relaxes the two rules that are only ambiguous in prose: a spaced dash
    becomes a minus, and a typographic prime becomes an apostrophe.
    """
    out = []
    for ch in text:
        cp = ord(ch)
        if 0xFF01 <= cp <= 0xFF5E:      # full-width ASCII variants → half-width
            out.append(chr(cp - 0xFEE0))
        elif cp == 0x3000:               # ideographic space → regular space
            out.append(' ')
        elif cp == 0x2212:               # MINUS SIGN → hyphen-minus
            out.append('-')
        elif math and cp in _MATH_PRIME_MAP:
            out.append(_MATH_PRIME_MAP[cp])
        else:
            out.append(ch)
    result = ''.join(out)
    if math:
        result = _MATH_DASH_SPACED_RE.sub(r'\1-\2', result)
    return _MATH_DASH_RE.sub('-', result)


def _normalize_tokens(tokens: list[InputToken]) -> list[InputToken]:
    """Normalize full-width ASCII characters in all token fields."""
    result = []
    for tok in tokens:
        if tok.type == "text":
            result.append(InputToken(type="text", value=_fullwidth_to_halfwidth(tok.value)))
        elif tok.type == "math":
            result.append(InputToken(
                type="math",
                latex=_fullwidth_to_halfwidth(tok.latex, math=True),
                canonical=_fullwidth_to_halfwidth(tok.canonical, math=True),
            ))
        else:
            result.append(tok)
    return result


# ---------------------------------------------------------------------------
# Variable inference helpers
# ---------------------------------------------------------------------------

# Operations that require a 'var' argument (differentiation variable etc.)
_VAR_REQUIRED_OPS: frozenset[str] = frozenset({
    "integrate_indefinite",
    "integrate_definite",
    "differentiate",
    "taylor_expand",
})

# Identifiers that appear in canonical expressions but are function names, not variables
_KNOWN_MATH_FUNCTIONS: frozenset[str] = frozenset({
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "atan2", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "asinh", "acosh", "atanh",
    "log", "ln", "exp", "sqrt", "abs",
    "floor", "ceiling", "round", "sign",
    "max", "min", "mod", "gcd", "lcm",
    "gamma", "beta", "erf", "erfc", "zeta",
    "realpart", "imagpart", "conjugate",
})


def _extract_variables(canonical: str) -> set[str]:
    """Return the set of variable names in a CAS canonical expression.

    Handles both Maxima-style (*) and Julia-style implicit multiplication (6x).
    Removes Maxima %-constants (%e, %pi, %i …) then finds identifiers that
    are NOT immediately followed by '(' (which would make them function calls).
    """
    s = re.sub(r'%[a-zA-Z]+', '', canonical)
    # Handle Julia-style implicit multiplication: '6x' → '6 x' so \b matches.
    s = re.sub(r'(\d)([A-Za-z])', r'\1 \2', s)
    tokens = re.findall(r'\b([A-Za-z][A-Za-z0-9_]*)\b(?!\s*\()', s)
    return {t for t in tokens if t not in _KNOWN_MATH_FUNCTIONS}




# ---------------------------------------------------------------------------
# Inline formula auto-registration
# ---------------------------------------------------------------------------

# Detects "ASCII math formula<space>Japanese operation" in a single text token.
# Group 1 = formula part (no hiragana/katakana/kanji),  Group 2 = rest (Japanese).
# ---------------------------------------------------------------------------
# H-1: Equation-form input  "LHS = RHS"
# ---------------------------------------------------------------------------
# Matches inputs like:
#   H = J * Sz_1 * Sz_2       (named quantity / assignment)
#   E_{kin} = p^2 / 2m        (named quantity)
#   -i\hbar \dot{a} = [H, a]  (equation of motion, RHS is bracket)
#
# Guard conditions (ALL must hold to trigger H-1):
#   1. `=` is NOT part of !=, <=, >=, :=, ==
#   2. LHS is non-empty, contains at least one letter/backslash
#   3. LHS is NOT a bare expression ID (E\d+)
#   4. LHS does NOT contain `(` → excludes f(x)=…, solve(…=0,x), integrate(…)=…
#   5. RHS does NOT contain a bare `=` → excludes chained equality a=b=c
#   6. RHS does NOT end with a plain ASCII word (verb/command) → excludes "H=x^2 simplify"
#   7. Whole input has no CJK → natural-language instructions handled by normal path
#   8. RHS is NOT a bare number/zero → excludes Maxima equations like x^2-1=0
#
# The LHS is kept as display label only — it is NOT evaluated.
# The RHS goes through the normal pipeline.
_EQ_SPLIT_RE = re.compile(
    r'^(?P<lhs>[^=\n]+?)'           # LHS (non-greedy, no newline, no =)
    r'\s*(?<![!<>:=])=(?!=)\s*'     # standalone = (not !=, <=, >=, :=, ==)
    r'(?P<rhs>.+)$',                # RHS
    re.DOTALL,
)

# LHS must contain at least one letter or backslash (not a pure number/operator)
_LHS_VALID_RE = re.compile(r'[A-Za-z\\{]')

# RHS that is just a number (possibly with sign/decimal) → Maxima equation, not H-1
_RHS_BARE_NUMBER_RE = re.compile(r'^\s*[+-]?\s*\d+(\.\d+)?\s*$')

# Trailing plain ASCII word at end of RHS → operation instruction leaked in
# e.g. "H = x^2 simplify" → "simplify" is not part of the formula
_RHS_TRAILING_WORD_RE = re.compile(r'\s+[A-Za-z]{3,}\s*$')


class _EqSplit:
    """The two sides of an H-1 equation.  Named `.group()` so the call sites
    read as they did when this was a regex match."""

    __slots__ = ("_lhs", "_rhs")

    def __init__(self, lhs: str, rhs: str):
        self._lhs, self._rhs = lhs, rhs

    def group(self, name: str) -> str:
        return self._lhs if name == "lhs" else self._rhs


def _split_at_top_level_eq(raw: str) -> "_EqSplit | None":
    r"""Split ``LHS = RHS`` at a TOP-LEVEL ``=`` — never one inside brackets.

    `\sum_{i=1}^{3} i^2` was being split at the `=` in its own SUBSCRIPT: the
    LHS came out as `\sum_{i` (which passes every guard below — it has a
    backslash and a brace), and what was registered was the RHS alone, canonical
    `1}^3*i^2`.  The head of the expression was simply gone, with no error
    (measured 2026-08-10).  `\sum_{i=1}^{N} \sum_{j=i+1}^{N} …` escaped only
    because it has TWO `=` and the chained-equality guard rejected it — so the
    plainest bounded sum there is was the broken one.

    `find_top_level_relation` is the bracket-aware scan the parser already uses,
    and `lhs_formatting._lhs_latex_from_expr` was moved onto it for THIS EXACT
    input (CLAUDE.md, `\sum_{i = 14`).  The input path kept its own regex and
    the fix never reached it — the same rule answered in two places, differently.
    """
    from notebook.runtime.shared_parser import find_top_level_relation
    rel = find_top_level_relation(raw)
    if rel is None or rel[1] != "=":
        return None
    at = rel[0]
    # `find_top_level_relation` reports the `=` of `!=` as an `=` (measured:
    # `a != b` → (3, "=")), so the exclusions the old regex carried in its
    # look-around have to be restated on the position it returns.  `==`, `:=`
    # and `<=`/`>=` it already declines or names differently; `!=` it does not.
    if raw[at - 1:at] in ("!", "<", ">", ":", "=") or raw[at + 1:at + 2] == "=":
        return None
    lhs, rhs = raw[:at], raw[at + 1:]
    if not lhs.strip() or not rhs.strip():
        return None
    return _EqSplit(lhs, rhs)


def _names_an_operation(engine):
    """`text_names_an_operation` を、このエンジンの知識で束ねた述語にする。

    走査に要るもの（別名の引き方・語数の上限・仕様・このバックエンドが持つか・
    トークナイザ）はどれもエンジン側の知識なので、規則そのものは
    `operand_boundary` に置き、ここは束ねるだけ。
    """
    def _p(text: str) -> bool:
        return text_names_an_operation(
            text,
            engine._validator.resolve_alias,
            engine._validator.max_alias_words,
            engine._validator.get_spec,
            _tokenize_for_alias_scan)
    return _p


def _is_valid_h1_input(raw: str, names_operation=None) -> "_EqSplit | None":
    """Return the split if *raw* should be handled as H-1, else None.

    *names_operation*: 条件7（これは指示文か）の**2つめの証拠**（別名の表）。
    渡されなければ動詞の一覧だけで判断する——規則と実測は
    `operand_boundary.text_names_an_operation` にある。
    """
    m = _split_at_top_level_eq(raw)
    if m is None:
        return None
    lhs = m.group("lhs").strip()
    rhs = m.group("rhs").strip()
    if (
        # 7. これは**指示文**であって式ではない、という問い。ここは長らく
        #    `_HAS_CJK_RE`（＝日本語側の答え）だけを聞いていて、同じ問いの
        #    英語側の答え（`instruction_marker[en]` の動詞）を持っている
        #    `_is_nl_instruction` を呼んでいなかった。結果、操作を名指した
        #    英文が **式として登録**されていた（実測 2026-09-05）:
        #      `substitute x = 2 into E1` → 2*into(a*x+b)   ← into が関数呼び出し
        #      `set x = 2 in E1`          → 2*in(a*x+b)
        #    `In E1, substitute x = 2` だけが助かっていたのは条件8（右辺が裸の
        #    数）に当たったからで、「指示かどうか」を聞いた結果ではない。
        #    guard は言語で絞らない（`nlp_any`）——ja セッションに英文を打つ
        #    researcher を取りこぼさないため。corpus 523行×2言語で受理の差は0。
        nlp_any("instruction_marker", raw)   # 7. not an NL instruction
        or (names_operation is not None and names_operation(raw))
        or not _LHS_VALID_RE.search(lhs) # 2. valid LHS
        or _EXPR_ID_RE.match(lhs)        # 3. not bare E-ID
        or "(" in lhs                    # 4. no function call in LHS
        or "=" in rhs                    # 5. no chained equality
        or _RHS_BARE_NUMBER_RE.match(rhs)# 8. RHS is not bare number
        or _RHS_TRAILING_WORD_RE.search(rhs)  # 6. no trailing op word
    ):
        return None
    return m

def _tokenize_for_alias_scan(text: str) -> list:
    """The supervisor's OWN tokenizer, so this side and the deciding side count
    the same tokens.  Importing it here rather than writing a second one is the
    point — a different split would make this rule answer "no operation named"
    where the supervisor answers otherwise, which is the shape of the defect it
    exists to close."""
    from notebook.runtime.llm_supervisor import _tokenize
    return _tokenize(text)


_FORMULA_OP_RE = re.compile(
    r'^([^぀-鿿＀-￯]+?)\s*([^\x00-\x7F].*)',
    re.DOTALL,
)
# English inline-formula differentiate: "differentiate <formula> with respect to
# <var>" / "take the partial/total derivative of <formula> with respect to <var>".
# _FORMULA_OP_RE's ASCII/non-ASCII script-transition boundary can never fire on
# all-ASCII English input, so this uses the explicit keyword boundary instead
# ("of ... with respect to") — see the call site below for why this matters.
#: **前置詞は共有の表から取る**（`en_op_var_head_wide`、2026-09-18）。ここは
#: `with respect to|w.r.t.|by` を手書きで持っており、**同じ問いに2つ目の答え**だった
#: ——`def_integral[en]` と積分の枠はその表を読む。割れていた代償は測ってある:
#: `differentiate a x^2 in x` / `for x` / `over x` は枠に当たらず、欄が位置で埋まって
#: `differentiate(expr='x^2', var='a')` → `0` を `completed` で返していた（6前置詞中3）。
def _diff_inline_frame() -> "re.Pattern[str]":
    #: 名前に経路名＋言語の接尾辞の並びを含めない——その規則を**部分文字列**で
    #: 見張っているラチェットが在り、関数名でも鳴る（実測 2026-09-19）。
    from notebook.runtime.nl_patterns import nlp
    return nlp("diff_inline_formula", "en")
def _integrate_inline_formula(raw: str):
    r"""`integrate <式> over <変数>` を切る——微分の枠の積分側。

    英語の微分には枠が在り（`_DIFF_INLINE_FORMULA_EN_RE`）、積分には無かった。
    無いあいだ被積分関数は式として登録されず、**欄が位置で埋まって変数と入れ
    替わって**いた（実測 2026-09-18、英語3綴り × 被積分関数5つの15セル中
    正しいのは3、**6が黙って誤答**——`integrate a x^2 over x` が
    `integrate(expr='x^2', var='a')` を `completed` で返す）。日本語は同じ族で
    正しい: 字種の変わり目が式を切り出すからである。

    **境界つきの綴りには当たらない。** `from … to …`（`def_integral`）と
    `over the region …`（`region_integral`）は専用のインターセプタが持っていて、
    **実測で正しく動く**（E-id の被積分関数も受ける: `integrate E1 over x from 0
    to 1` → `1/3`）。より特定の読みに道を譲るのは、綴りを二重に読まないためで
    ある——ここで式を登録すると、その経路の式番号まで動く。
    """
    from notebook.runtime.nl_patterns import nlp
    if nlp("def_integral", "en").match(raw) or nlp("region_integral", "en").match(raw):
        return None
    return nlp("integrate_inline_formula", "en").match(raw)


def _nl_word_count(text: str) -> int:
    r"""3文字以上の**英単語**の数——数式の一部でないもの。

    ここが「これは式か、それとも英文か」の判定で、切り出したのは同じ問いに
    2箇所で答えないためである（英語の「動詞＋式」の規則も同じ数え方を使う）。

    - LaTeX コマンドとして現れる語（`\alpha`）は記法であって単語ではない。
    - **環境の名前も同じく記法である**（`\begin{pmatrix}` の `pmatrix`）。
      `\begin` / `\end` はコマンドなので除かれていたのに、その引数である
      環境名は**普通の英単語として2回数えられて**いた。だから素の
      `\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}` は——**このシステム自身が
      表示に使う綴り**——「英文だ」と判定されて式の経路を失い、
      「解釈できません」で落ちていた（実測 2026-09-10、TEX-MATRIX-INPUT）。
    - 直後が `(` の識別子は関数**呼び出し**で、既知関数の白名簿に限らない
      （BP-8: `diff(V,t) + diff(V,S,2) = 0` が2語と数えられて式の経路を失った）。
      `(` は隣接必須——`expand (x+1)^2` の空白は NL 動詞を数えさせるためにある。
    """
    cmds = {m.group(1).lower() for m in re.finditer(r'\\([a-zA-Z]+)', text)}
    cmds |= {m.group(1).lower()
             for m in re.finditer(r'\\(?:begin|end)\{([a-zA-Z]+)\*?\}', text)}
    return sum(
        1 for m in re.finditer(r'\b[a-zA-Z]{3,}\b', text)
        if text[m.end():m.end() + 1] != "("
        and m.group(0).lower() not in _KNOWN_MATH_FUNCTIONS
        and m.group(0).lower() not in cmds
    )


#: 仕様では要求フィールドが複数だが、**英語の一文では式が文末まで**の操作。
#: 実測に基づく例外で、推測ではない: `wick contraction of A B` の
#: `contraction_rules` は同じ文からは来ない（Adviser か別の指示で決まる）。
_FORMULA_TO_END_OPS: frozenset = frozenset({"wick_contract"})


#: 後続の節を導く語。これがあれば式は文末まで**行かない**——`_en_verb_formula_re`
#: の前提そのものなので、条件として書いてある。
_TRAILING_CLAUSE_RE = re.compile(
    r"(?i)\b(?:as|for|with|into|over|about|from|to|by|respect|wrt|w\.r\.t\.?)\b")

#: 式参照。`E3, E4 for …` のような**参照を並べた指示**は E-id のインターセプタの
#: 領分で、式として登録するものではない。
_ANY_EXPR_ID_RE = re.compile(r"(?<![A-Za-z0-9_])E\d+(?![A-Za-z0-9_])")


def _formula_runs_to_the_end(formula: str) -> bool:
    r"""残りぜんぶが1つの式か——`_en_verb_formula_re` を通してよいか。

    三つとも、規則の**前提が成り立つか**を聞いている:

    - 英単語が2語以上なら英文であって式ではない（`define c as an electron …`）。
    - 後続の節を導く語があれば、式はそこで終わっている
      （`integrate f over x from 0 to 1`）。
    - **式参照が混じっていれば、それは参照の指示**である。

    最後の条件は実測で足した（2026-08-22、実例の再生成が捕まえた）:
    `Invert E3, E4, E5, E6 for c(k,u), …` の `invert` は行列の逆の別名なので
    この規則が当たり、逆変換の指示ぜんぶを1つの「式」として登録して壊していた。
    利用者経路のテストは全部緑のままで、**実例の再生成だけが捕まえた**。
    """
    # 式番号の除外は参照の並びのため——`2 E3` は積（英語だけ係数が落ちていた、日英の割れ）。
    return (_nl_word_count(formula) < 2
            and not _TRAILING_CLAUSE_RE.search(formula)
            and not (_ANY_EXPR_ID_RE.search(formula) and not _ATOMIC_PRODUCT.fullmatch(formula)))


@lru_cache(maxsize=1)
def _en_verb_formula_re() -> "re.Pattern[str]":
    r"""`<英語の操作名> <式>` を切る。境界は**操作の別名そのもの**。

    日本語は字種の変わり目（`_FORMULA_OP_RE`）で式を切り出すが、その境界は
    全 ASCII の英語では原理的に発火しない。境界を持っていたのは `differentiate`
    と演算子代数の単項族だけで、**操作ごとに足していた**。実測 2026-08-22、
    利用者が通る経路で:

        simplify a x^2 + b x + c   →  expr='x^2'   completed
        factor x^2 - 1             →  expr='x^2'   completed

    式が空白を含むだけで、書いたものと違う式が計算され、**成功として**返っていた
    （同型の英語指示 72 のうち 63）。

    別名は**スキーマのデータ**から作る（列挙し直さない）。対象は**要求フィールドが
    `expr` 1つだけ**の操作に限る——それ以外は式の終わりが文末ではなく
    （`substitute x=2 into E1`）、専用のインターセプタが持っている知識を横取りする
    ことになる。3文字未満と `_` 入りの別名は除く（英文の中で偶然当たる／内部名）。
    """
    from notebook.runtime.operation_validator import OperationValidator
    v = OperationValidator()
    names: set = set()
    for op in v.known_operations:
        spec = v.get_spec(op) or {}
        if (list(spec.get("required_fields") or []) != ["expr"]
                and op not in _FORMULA_TO_END_OPS):
            continue
        for alias in (spec.get("aliases", {}) or {}).get("en") or []:
            alias = alias.strip()
            if len(alias) >= 3 and "_" not in alias:
                names.add(re.escape(alias.lower()))
    # 長い別名から試す（`re evaluate expr` が `re evaluate` に食われないように）。
    alt = "|".join(sorted(names, key=len, reverse=True))
    # 3つ目の空グループは、隣の `differentiate` 規則と同じ (leadin, formula,
    # trailing) の形にするため——呼び出し側が1つで済む。
    #
    # **先頭の冠詞は操作名の一部ではない**（2026-09-17）。`the curl of \vec{A}` は
    # 別名が文頭に立たないので当たらず、書かれた式が**登録されないまま**生の LaTeX で
    # 欄へ落ちて `Maxima evaluation error` が返っていた。冠詞は `the` だけ（`a` は
    # コーパスでは記号名）、そして **leadin（第1群）の中**に入れる——呼び出し側は
    # 飲んだぶんを組み直しで戻す規約なので、外に出すと答える規則が変わる。
    # 測定と却下した案は `tests/test_bare_article_before_an_operation_name.py`。
    return re.compile(
        r"(?i)^\s*((?:the\s+)?(?:" + alt + r"))\b\s*(?:of\s+|the\s+)*(.+?)\s*()$"
    )


async def _try_en_genitive_noun(engine, loop, lang, text, registered) -> "dict | None":
    r"""`<操作> the <名詞> of <式>` の**名詞**——操作でないなら、名指して断る。

    `of` は**2文字**なので `_nl_word_count`（3文字以上を数える）に見えない。だから
    `widget of x^2 - 1` は「英単語1語＋数式」として英語の「動詞＋式」の規則を通り、
    **1つの式として登録**されて `of*widget*(x^2-1)` になっていた——`completed`。
    語数で降りる綴り（`nearest-neighbor of …`）は別名走査に落ち、そちらは
    `expr='nearest-neighbor'` を作っていた。**根は2つ、形は1つ**なので、断りも1つ。

    **判別は操作の別名そのもの**（`operand_boundary.genitive_head_not_an_operation`）
    ——`the curl of the curl of A` の内側や `expand the product of E1 and E2` は
    実在の綴りで、そちらは従来どおり通る。コーパス611行に対してこの述語が発火するのは
    **0 行**（着手条件として先に測った）。

    **案内する道は実測したもの**: `expand x^2 - 1` は計算し、`widget = x^2 - 1` の
    あとの `expand widget` も計算する。

    日本語の `x^2 - 1 の widget を展開して` は**式が正しく切れており**、落ちている
    のは別名走査が置き場のないトークンを捨てる一般則である（別起票
    LEFTOVER-NOUN-DROPPED）。ここでは断らない。
    """
    # **読みはこの層が既に持っている2つ**——英語は操作の別名が境界、日本語は字種の
    # 変わり目。`widget of x^2 - 1 を展開して` は英語の綴りが日本語の文に混じった形で、
    # 後者が切り出す「式」の頭に属格が立つ（実測、直す前は `of*widget*(x^2-1)`）。
    _pfx, _rest = split_leading_compute_verb(text or "")
    m_en = _en_verb_formula_re().match(_rest)
    m_ja = _FORMULA_OP_RE.match(text or "")
    if m_en is not None:
        written = m_en.group(2)
    elif m_ja is not None:
        written = m_ja.group(1)
    else:
        return None
    noun = genitive_head_not_an_operation(written.strip(), _names_an_operation(engine))
    if noun is None:
        return None
    operand = re.sub(r"(?i)^of\s+", "", written.strip()[len(noun):].lstrip()).strip()
    return {"status": "failed",
            "error": t("operand.genitive_head_not_an_operation", lang,
                       noun=noun, operand=operand),
            "registered_expressions": registered}


#: 名前らしい綴り——素の識別子と、**添字を1つ持つ形**（`E_{total}`）。後者を
#: 入れるのは、`E_{total} を簡約して` が `E(total)` を `completed` で返して
#: いたからである（起票 OPERAND-BOUNDARY 残⑤ そのもの）。数学かどうかは
#: `_operand_is_mathematics` が答える——`\vec{A}` も `2 v` もそちらで式として
#: 登録されるので、ここへは来ない。
_OPERAND_NAME_RE = re.compile(r"[A-Za-z_]\w*(?:_\{[A-Za-z0-9_,]+\})?")


def _operand_key(engine, written: "str | None") -> "str | None":
    """研究者が書いた名前を、量の台帳が使う綴りへ。

    `E_{total}` は `E_total` として登録される。正準の扉が識別子を返さないとき
    （`one` → `1`——数値語の表、DECL-NAME-READING が記録している罠）は、書かれた
    ままを鍵にする: 名前の扉は名前を返さなければならない。
    """
    if not written:
        return None
    from notebook.runtime.shared_parser import UnknownNotationError
    try:
        key = engine._handler.latex_to_canonical(written).strip()
        return key if re.fullmatch(r"[A-Za-z_]\w*", key) else written
    except (UnknownNotationError, ValueError, AttributeError, TypeError):
        return written                     # 綴りが読めなければ書いたままを鍵にする


def _operand_name_in_text(text: str) -> "str | None":
    r"""文が**オペランドの位置**に置いた素の名前。無ければ None。

    位置を知っているのは文法であって、名前の見た目ではない——`x で微分して` の
    `x` も未定義の素の名前だが、そこでは**変数**として立っている。だから読み方は
    この層が既に持っている3つ（字種の変わり目・英語の別名境界・英語の微分の枠）を
    そのまま使う——新しい文法は足さない:

    * 日本語は `_FORMULA_OP_RE`（ASCII → CJK の境目）。**続きが `を` か `の`**
      のときだけオペランドで、`で`（変数）は数えない。
    * 英語は `_en_verb_formula_re`（境界は操作の別名そのもの）と、微分・積分の枠
      （境界は変数を導く前置詞）。**後続の節で
      自分で切ることはしない**——微分の枠は既に切っており、測ると足しても
      どのセルも変わらず、しかも `_FORMULA_TO_END_OPS`（`wick contract A B` は
      式が文末まで行く、と書いてある操作）と食い違う。

    数学（`2 v`・`x^2+y^2`）はここに来ない: 手前の gate が式として登録し、文には
    式番号が立つので、guard はその時点で答えが出ている。**それでも数学かどうかを
    訊く**のは、名前の形をしていて数学でもあるもの——**素の既知関数名**
    （`sin`・`log`・`exp`）——のためである（実測、除外が効くのはこれだけ）。
    数えると断り文が「`sin` に値を与えてください」と誘うことになり、素の関数名には
    それ自身の断りが別に在る（CLAUDE.md「A WRITTEN FUNCTION IS APPLIED…」）。
    """
    m = _FORMULA_OP_RE.match(text or "")
    cand = None
    if m and re.match(r"\s*[をの]", m.group(2) or ""):
        cand = m.group(1).strip()
    else:
        m_en = (_en_verb_formula_re().match(text or "")
                or _diff_inline_frame().match(text or "")
                or _integrate_inline_formula(text or ""))
        if m_en:
            cand = m_en.group(2).strip()
    if not cand or _EXPR_ID_RE.match(cand) or _operand_is_mathematics(cand):
        return None
    return cand if _OPERAND_NAME_RE.fullmatch(cand) else None


_HAS_MATH_OP = re.compile(
    r'[+\-*/^=()\[\]]'                                               # operators / parens
    r'|(?:^|\b)(?:sin|cos|tan|cot|sec|csc|log|ln|exp|sqrt'
    r'|arcsin|arccos|arctan|sinh|cosh|tanh)(?:\b)'                   # known function names
    r'|\\[a-zA-Z]+'                                                  # LaTeX commands
)

#: 原子（数・識別子）を並置・`.`・`~` でつないだ形と、原子に見えても機能語であるもの。
_ATOM = r"(?:\d+(?:\.\d+)?|[A-Za-z_]\w*'*)"
_ATOMIC_PRODUCT = re.compile(rf"\s*{_ATOM}(?:\s*[.~]\s*{_ATOM}|\s+{_ATOM})+\s*")
_SHORT_FUNCTION_WORDS = frozenset({"by", "in", "at", "to", "on", "as", "of", "or", "if", "is", "an"})


def _operand_is_mathematics(formula: str, following: "str | None" = None) -> bool:
    r"""オペランドは名前1つより多い数学か（式として登録して渡すか）——名前1つなら 2a-sym が書き換える。

    演算子の記号だけを数えていたので**並置の積（この系の乗算）と `.`・`~` が名前1つ扱い**になり、
    `2 v の回転を求めて` は係数を捨てて `curl(v)` を `completed` で返した。並置は全トークンが原子の
    ときだけ（英単語と2文字の機能語は原子ではない——広い版が `E1 twice` を奪った）、日本語は後ろが
    `の`/`を` のときだけ。測定と経緯: `tests/test_a_coefficient_is_part_of_the_operand.py`。
    """
    if _HAS_MATH_OP.search(formula):
        return True
    if not _ATOMIC_PRODUCT.fullmatch(formula) or (
            following is not None and not re.match(r"\s*[のを]", following)):
        return False
    return (_nl_word_count(formula) == 0
            and not {w.lower() for w in re.findall(r"[A-Za-z]+", formula)} & _SHORT_FUNCTION_WORDS)




def _expand_expr_ids_result(text: str, engine: "NotebookEngine") -> str:
    """Like _expand_expr_ids but always uses expr.canonical (the CAS result).

    Use this when you need the *output* of CAS operations (e.g. E3 = ml²q̈),
    not the user-typed input that was transformed.  _expand_expr_ids uses
    input_canonical which for derivative results holds the SOURCE expression
    (e.g. the Lagrangian), not the differentiated result.
    """
    active_ids = engine._active_expr_ids()

    def _replace(m: re.Match) -> str:
        eid = f"E{m.group(1)}"
        if eid not in active_ids:
            return m.group(0)
        expr = engine.expressions.get(eid)
        if expr is None:
            return m.group(0)
        canonical = expr.canonical
        if not canonical:
            return m.group(0)
        # Strip outer equation form "LHS = RHS" → use RHS only (bracket-aware).
        canonical = strip_lhs(canonical)
        return f"({canonical})"

    return _INLINE_EXPR_ID_RE.sub(_replace, text)








async def _register_named_alias(engine, loop, src_eid: str, name: str,
                                registered: list, canonical_to_id: dict):
    """Bind a NEW named variable *name* to expression *src_eid*'s value.

    Used by "P = En" (naming) and "En を P とする" (assignment).  Registers a new
    expression labelled *name* with En's value and display, so *name* is bound in
    the CAS (P2) and reusable in later formulas.  Returns a math InputToken (to
    consume the input) or None on failure.  Unlike the H-1 dedup path, this never
    drops the label when the value already exists and never re-evaluates the RHS
    (which would mangle a matrix)."""
    src = engine.expressions.get(src_eid)
    if src is None:
        return None
    _val = strip_lhs(src.canonical)
    _disp = strip_lhs(src.latex or _val)
    try:
        expr = await loop.run_in_executor(
            None, partial(engine.register_expression, _val, latex=_disp, lhs_label=name))
    except Exception:
        return None
    canonical_to_id[expr.canonical] = expr.display_id
    registered.append({
        "expr_id": expr.display_id, "canonical": expr.canonical,
        "latex": expr.latex, "step_id": expr.created_by_step,
        "undefined_symbols": expr.undefined_symbols,
        "warnings": expr.warnings,
    })
    return InputToken(type="math", latex=_disp, canonical=_val)


# "En を P とする" / "call En P" / "store En as P" — assign a result to a name.
# The source must be an E-ID and the target a plain identifier (so "a をスカラーと
# する" — a type declaration — does not match).  English equation-form naming
# ("P = En") is handled separately by the language-neutral H-1 auto-registration
# path; this covers only the imperative phrasing, which has no schema "rename"
# operation to fall back to.  Grammar in nl_patterns.json["name_assign"]
# (session-language, docs/28 §3.1; shared groups eid/name, EN store-branch
# eid2/name2).
async def _try_name_assignment(engine, loop, supervisor_text, registered, lang="ja") -> "dict | None":
    """Rule-based: "En を P とする" / "call En P" binds a new named variable P
    to En's value."""
    m = nlp("name_assign", lang).search(supervisor_text)
    if not m:
        return None
    gd = m.groupdict()
    src_eid = gd.get("eid") or gd.get("eid2")
    name = gd.get("name") or gd.get("name2")
    if (src_eid not in engine.expressions
            or src_eid not in engine._active_expr_ids()):
        return None
    _tok = await _register_named_alias(engine, loop, src_eid, name, registered, {})
    if _tok is None:
        return {"status": "failed", "error": f"{src_eid} を {name} とする",
                "registered_expressions": registered}
    return {"status": "registered", "registered_expressions": registered}


# P-1: Poisson bracket support (classical Hamiltonian mechanics, Maxima backend).
# "p_theta を theta の共役運動量とする" / "define p_r as the canonical momentum
# of r" — declare a canonical pair (coordinate, momentum).  "theta と p_theta
# を正準対とする" / "define theta and p_theta as a canonical pair" — same pair,
# opposite trigger phrasing.  "f と g のポアソン括弧を求めて" / "the Poisson
# bracket of f and g".  Externalized to i18n/nl_patterns.json["pb_pair_momentum"
# / "pb_pair_canonical" / "pb_bracket" / "pb_bracket_verb_guard"] (named groups
# momentum/coordinate, f/g — consistent order in both languages for each key).


async def _try_poisson(engine, loop, lang, text, registered) -> "dict | None":
    """Rule-based: canonical-pair declaration and Poisson bracket {f,g}.

    Declarations ("p_r を r の共役運動量とする" / "r と p_r を正準対とする") store
    the pair in SymbolContext.canonical_pairs.  "f と g のポアソン括弧を求めて"
    dispatches the poisson_bracket operation, which builds Σ_i (∂f/∂q_i ∂g/∂p_i −
    ∂f/∂p_i ∂g/∂q_i) over the declared pairs (Maxima backend)."""
    ctx = engine.symbol_context

    # --- Canonical-pair declarations -------------------------------------
    m = nlp("pb_pair_momentum", lang).search(text)
    if not m:
        m = nlp("pb_pair_canonical", lang).search(text)
    if m:
        momentum, coordinate = m.group("momentum"), m.group("coordinate")
        ctx.add_canonical_pair(coordinate, momentum)
        _pb_msg = t("poisson.pair_declared", lang).format(q=coordinate, p=momentum)
        engine.record_quantity_update(t("poisson.pair_declared_label", lang), _pb_msg)
        return {"status": "completed", "operation": "declare_canonical_pair",
                "message": _pb_msg,
                "registered_expressions": registered}

    # --- Poisson bracket {f, g} ------------------------------------------
    # The English phrasing "Poisson bracket of f and g" is unambiguous on its
    # own (unlike the Japanese pattern, which needs a 求め/計算 verb guard
    # since "…のポアソン括弧" alone could appear in other contexts), so the
    # guard (pb_bracket_verb_guard, ja-only) only applies for lang=='ja'.
    if lang != "en" and not nlp("pb_bracket_verb_guard", lang).search(text):
        return None
    mb = nlp("pb_bracket", lang).search(text)
    if not mb:
        return None

    def _resolve(tok: str) -> "str | None":
        # E-ID (active) → itself; H-1 name → its E-ID via lhs_alias_map;
        # otherwise a bare symbol/expression (e.g. r, p_r) → literal operand.
        if tok in engine.expressions and tok in engine._active_expr_ids():
            return tok
        eid = engine.lhs_alias_map.get(tok)
        if eid and eid in engine._active_expr_ids():
            return eid
        if re.fullmatch(r'E\d+', tok):
            return None          # an E-ID that is not active → fail cleanly
        return tok               # bare symbol like q, p_r → pass through literally

    a, b = _resolve(mb.group("f")), _resolve(mb.group("g"))
    if not a or not b:
        return None
    try:
        _step, _val = await loop.run_in_executor(
            None, partial(engine.run_operation, "poisson_bracket", {"a": a, "b": b}))
    except Exception as _e:
        if "__no_canonical_pairs__" in str(_e):
            return {"status": "failed",
                    "error": t("poisson.no_canonical_pairs", lang),
                    "registered_expressions": registered}
        return {"status": "failed", "error": str(_e),
                "registered_expressions": registered}
    _oid = _step.output_refs[0] if _step.output_refs else None
    _oe = engine.expressions.get(_oid) if _oid else None
    return {"status": "completed", "operation": "poisson_bracket", "expr_id": _oid,
            "step_id": _step.step_id,
            "latex": _oe.latex if _oe else "",
            "canonical": _oe.canonical if _oe else _val,
            "operation_label": _step.visible_action,
            "registered_expressions": registered}

# "E_n = X" where E_n is an existing expression-ID (X = 0, a value, or another
# E-ID).  H-1 deliberately skips a bare-E-ID LHS, so without this the input would
# register the literal "E3 = 0" (E3 kept symbolic).  Instead we expand E_n to its
# value and register the EXPLICIT equation  <value> = X  so the value is shown and
# can be solved / used as a constraint (e.g. resonance Im Z = 0).
_EXPR_ID_EQ_RE = re.compile(r'^\s*(E\d+)\s*=\s*(.+?)\s*$', re.DOTALL)

# Multivariable change of integration variables (S-o), e.g.
#   「E1 を x,y から r,theta へ x=r*cos(theta), y=r*sin(theta) で変数変換して」
# Deterministic rule intercept so the structured args (old_vars / new_vars /
# transform) are extracted WITHOUT the LLM — change of variables is routine in
# definite integration and must not depend on a model to parse the substitution.
# The integrand target is an E-ID *or* a literal expression (e.g. `1` for a pure
# volume element, `x^2+y^2`).  A literal is passed straight through to the change
# op, which uses it as the integrand (expr_lookup falls back to the ref verbatim);
# the validator only rejects unknown *E-ID-shaped* refs, so a literal is accepted.
# NL pattern externalized to i18n/nl_patterns.json ("change_vars"; groups
# expr/oldv/newv/trans).












async def _try_vector_op(engine, loop, lang, text, registered) -> "dict | None":
    """Single vector-operator applied to a registered field — both nabla notation
    (`∇·E`, `\\nabla \\times E`, `∇²f`, `∇f`) and the named unary operators
    (`div E`, `div(E)`, `rot E`, `curl E`, `grad f`, `laplacian f`).  Dispatch the
    matching operation so it COMPUTES (via nb_div/nb_curl/…), instead of being
    registered as an unevaluated `nabla . E` / `div(E)`.  Only fires when the
    field is a registered vector/scalar field (so `div t` etc. don't false-match).
    NESTED too — unread here, the LaTeX spelling registered as a formula (NESTED-VEC-OP-LATEX)."""
    res = _try_single_vec_op(text) or _try_nested_vec_op(text)
    _spec = engine._validator.get_spec(res.operation) if res else None
    if res is None or engine.session.primary_backend not in (_spec or {}).get("supported_backends", []):
        return None          # unsupported here: stays WRITABLE (a Julia `\nabla u` registers)
    expr = res.args.get("expr", "")
    _inner = re.fullmatch(r"(\w+)\((\w+)\)", expr)   # the guard is on the INNERMOST field
    inner_op, field = _inner.groups() if _inner else (res.operation, expr)
    # div/curl need a vector, grad/laplacian a scalar — answered by the TYPE module,
    # which reads a declaration or, for an H-1 name, its value.  The strict check
    # keeps `div t` (t a scalar) from false-matching.
    from notebook.runtime.vector_op_types import field_operand
    have, operand = field_operand(field, engine.symbol_context, engine.expressions)
    if have != ("vector" if inner_op in ("divergence", "curl") else "scalar"):
        return None
    expr = f"{inner_op}({operand})" if _inner else operand
    try:
        step, val = await loop.run_in_executor(
            None, partial(engine.run_operation, res.operation, {"expr": expr}))
    except Exception as _e:
        return {"status": "failed", "error": str(_e),
                "registered_expressions": registered}
    oid = step.output_refs[0] if step.output_refs else None
    oe = engine.expressions.get(oid) if oid else None
    return {"status": "completed", "operation": res.operation, "expr_id": oid,
            "step_id": step.step_id,
            "latex": oe.latex if oe else "",
            "canonical": oe.canonical if oe else val,
            "operation_label": step.visible_action,
            # An operation's RESULT can carry a warning too — `pair_substitute`
            # whose rules reached no term says so here.  The registration paths
            # have passed `warnings` through since the bound-index warning was
            # added; this one did not, so a warning attached to a result was
            # rendered nowhere (the same "set and never shown" the bound-index
            # work warned about).
            "warnings": oe.warnings if oe else [],
            "registered_expressions": registered}


# ---------------------------------------------------------------------------
# R-8: notation candidates + confirmation (minimal: |X| and undeclared Δ)
# ---------------------------------------------------------------------------
# Resolution priority (CLAUDE.md "Notation Disambiguation Confirmation UI"):
#   1. explicit SymbolContext declaration   → silent
#   2. active domains narrow it down        → silent (Δ fires only w/o domains)
#   3. notation_usage prior record          → silent
#   4. otherwise                            → candidate list + confirmation
# The chosen role is written to notation_usage by the `resolve_notation`
# pseudo-operation (notebook_bridge._handle_side_effect_operation), so the
# same token is never re-asked in this session — and, because notation_usage
# is persisted, not after reload either.







async def _try_expr_id_equation(engine, loop, lang, text, registered) -> "dict | None":
    m = _EXPR_ID_EQ_RE.match(text)
    if m is None or _HAS_CJK_RE.search(text):
        return None
    eid, rhs_raw = m.group(1), m.group(2).strip()
    if eid not in engine.expressions or eid not in engine._active_expr_ids():
        return None
    # Value of E_n: its RHS if it is itself an equation, else the whole canonical.
    # This becomes the left side of an expression that is REGISTERED below, so a
    # mis-split is persisted, not merely used once.  Measured 2026-08-13: on
    # `\sum_{i=1}^{3} c_i = u a + v b`, typing `E1 = 0` stored
    # `1}^3*c_i = u*a + v*b = 0` — a fragment head and two `=`.
    lhs_val = strip_lhs(engine.expressions[eid].canonical)
    if not lhs_val:
        return None
    rhs_val = _expand_expr_ids(rhs_raw, engine)
    eq_str = f"{lhs_val} = {rhs_val}"
    try:
        expr = await loop.run_in_executor(
            None, partial(engine.register_expression, eq_str))
    except Exception as _e:
        return {"status": "failed", "error": str(_e),
                "registered_expressions": registered}
    registered.append({
        "expr_id": expr.display_id,
        "canonical": expr.canonical,
        "latex": expr.latex,
    })
    return {"status": "registered", "registered_expressions": registered}




































# Region-predicate definite integral, e.g.
#   「x^2+y^2<=1 の領域で x^2+y^2 を定積分」 (unit disk) — a display-only object.
# NL pattern externalized to i18n/nl_patterns.json ("region_integral"; groups
# lhs/op/rhs/integrand).




# NL pattern externalized to i18n/nl_patterns.json ("evaluate_integral"; optional
# group target = E-id or "this integral").

# (The canned rotating-frame step recognizer was removed: the rotating-frame
# scenario is rebuilt on the general single-spin operations — register a custom
# spin H, rotating_frame transform, effective_field, general commutator.)




async def _register_and_report(engine, loop, canonical, latex, registered,
                               route, lhs_label=None) -> "str | None":
    """Register *canonical*, attribute the step to *route*, report it.  Id or None.

    The three registration branches in `_auto_register_inline_formula` compute
    their canonical differently — the leading formula converts the researcher's
    LaTeX, H-1 may CAS-EVALUATE an expanded right-hand side, and the E-id branch
    splices references without evaluating — and those differences are real.
    What is identical is what happens once a canonical exists, and that triple is
    exactly where the copies drifted, twice:

    * the English leading-formula copy dropped ``warnings``, so a bound-name
      collision on a formula typed in English never reached the screen;
    * the route attribution added on 2026-08-23 had to be written into each
      branch separately, and the two that were missed (`h1_equation` covering
      146 corpus lines, `expr_id_formula` covering 4) were invisible to the
      instruction corpus until they were.

    One place, so the next field cannot be added to two branches out of three.
    """
    before = len(getattr(engine, "steps", {}) or {})
    kwargs = {"latex": latex}
    if lhs_label is not None:
        kwargs["lhs_label"] = lhs_label
    try:
        expr = await loop.run_in_executor(
            None, partial(engine.register_expression, canonical, **kwargs))
    except Exception:                          # noqa: BLE001 — caller decides
        return None
    _attribute_new_steps(engine, before, route)
    registered.append({
        "expr_id":           expr.display_id,
        "canonical":         expr.canonical,
        "latex":             expr.latex,
        "step_id":           expr.created_by_step,
        "undefined_symbols": expr.undefined_symbols,
        "warnings":          expr.warnings,
    })
    return expr.display_id


async def _register_leading_formula(engine, loop, formula_latex, registered,
                                    canonical_to_id) -> "tuple[str | None, str]":
    r"""Register the formula at the head of an instruction; return (id, canonical).

    ``id`` is None only when registration FAILED — an id comes back both when
    the expression was newly registered and when an identical one already
    existed, because the caller's next move is the same either way.

    This is the half the Japanese and English branches share.  They differ in
    where the operation's name sits (after the formula in Japanese, before it in
    English), hence in which vetoes apply and what tokens come out — but
    "expand the E-ids, convert, find or register, attribute, report it" was
    written twice, and the two copies had drifted:

    * the English copy dropped ``warnings``, so a bound-name collision on a
      formula typed in English was recorded on the Expression and never reached
      the screen (the `feedback_warn_when_you_cannot_decide` rule is that a
      warning nobody renders is not a warning);
    * the attribution added on 2026-08-23 had to be written into both.

    The route name is ``inline_formula`` for both halves — see the call sites.
    """
    expanded = _expand_expr_ids(formula_latex, engine)
    canonical = engine._handler.latex_to_canonical(expanded) or expanded
    existing = _find_expr_by_canonical(engine, canonical)
    if existing:
        canonical_to_id[canonical] = existing
        return existing, canonical
    eid = await _register_and_report(engine, loop, canonical, formula_latex,
                                     registered, "inline_formula")
    if eid is None:
        return None, canonical
    canonical_to_id[canonical] = eid
    return eid, canonical


async def _auto_register_inline_formula(
    tokens: list[InputToken],
    engine: NotebookEngine,
    loop: asyncio.AbstractEventLoop,
    registered: list[dict],
    canonical_to_id: dict[str, str],
    lang: str = "ja",
) -> list[InputToken]:
    """Split text tokens of the form "formula op-text" into a math token + text token.

    Handles user input like "f(x) = cos(x) をxで微分して。" where the formula
    is typed as plain text rather than as a math chip.
    """
    result: list[InputToken] = []
    for tok in tokens:
        if tok.type != "text":
            result.append(tok)
            continue
        raw = tok.value.strip()

        # Inequality constraint (e.g. "2*x_A + x_B <= 100"): tried before H-1
        # since a constraint doesn't define a named symbol — H-1's own regex
        # already excludes <=/>= so this never overlaps, but trying it first
        # documents the precedence for future readers.
        _ineq = await _try_inequality_intercept(
            raw, engine, loop, registered, canonical_to_id)
        if _ineq is not None:
            result.append(InputToken(type="math", latex=_ineq["latex"],
                                     canonical=_ineq["canonical"]))
            continue

        # 「X を Y とする」/「set X to Y」/「let X be Y」で **Y が値**なら、
        # それは H-1 の `X = Y` である。判別子（`_looks_like_type_label`）も
        # 綴り（`value_definition`）も向こうに在り、ここは畳むだけ——なぜ錨が
        # 要るか・なぜ式番号を含む文には当たらないかは nl_patterns.json の
        # `value_definition` の `_comment`。数は綴りごと畳む（`number_literal`）
        # ので、canonical だけでなく画面と .tex も `x = 0` を言う。
        _vd = nlp_search("value_definition", raw)
        if _vd is not None and not _looks_like_type_label(_vd.group("value")):
            _vd_val = _vd.group("value").strip()
            raw = f'{_vd.group("name")} = {number_literal(_vd_val) or _vd_val}'
            # **トークンも差し替える**——`raw` だけ直すと、どの枝も掴まなかった
            # ときに元の文が追加され、`x を 3 とする` は「解釈できません」の
            # ままだった（H-1 は裸の数を右辺に取らないので、`x = 3` はここを
            # 通り抜けて 2c の素の式登録に届く必要がある）。
            tok = InputToken(type="text", value=raw)

        # H-1: equation-form input  "LHS = RHS"
        # Detect before the E-ID fast path so that "H = J*E1*E4" is handled here.
        # All guard conditions are checked by _is_valid_h1_input().
        _eq_m = _is_valid_h1_input(raw, _names_an_operation(engine))
        if _eq_m is not None:
            _lhs_raw = _eq_m.group("lhs").strip()
            _rhs_raw = _eq_m.group("rhs").strip()
            # Pure naming "P = En": bind a NEW named variable to En's value and
            # bind it in the CAS.  Bypass the dedup/eval path below — that path
            # would drop the label when En's value already exists, or re-evaluate
            # (mangling a matrix RHS).  See _register_named_alias.
            if (re.fullmatch(r'E\d+', _rhs_raw)
                    and _rhs_raw in engine.expressions
                    and _rhs_raw in engine._active_expr_ids()):
                _na = await _register_named_alias(
                    engine, loop, _rhs_raw, _lhs_raw, registered, canonical_to_id)
                if _na is not None:
                    result.append(_na)
                    continue
            # Expand E-IDs in RHS before converting to canonical.
            # NOTE: do NOT call _expand_symbol_refs here — H-1 registration should
            # store the expression as-is.  If the user wrote "V = m g l * (1-cos(q))"
            # and g is already in lhs_alias_map, V must be stored with g kept symbolic.
            # Symbol refs are intentionally NOT expanded during H-1 registration to
            # avoid silent substitutions like g → l*omega^2.
            _rhs_expanded = _expand_expr_ids(_rhs_raw, engine)
            _canonical = engine._handler.latex_to_canonical(_rhs_expanded) or _rhs_expanded
            _lhs_display = _lhs_raw  # keep raw LaTeX for display
            # When expression IDs were substituted, CAS-evaluate the combination
            # so that "EOM = E3 - E4" computes ml²q̈ + mgl sinq rather than
            # storing the literal string "E3 - E4".
            _ids_were_expanded = _rhs_expanded != _rhs_raw
            _h1_eval_latex: str = ""
            if _ids_were_expanded:
                # Re-expand using expr.canonical (CAS result) instead of
                # input_canonical (which _expand_expr_ids uses and which for
                # derivative results holds the source expression, not the output).
                _rhs_result_expanded = _expand_expr_ids_result(_rhs_raw, engine)
                _rhs_result_expanded = _expand_symbol_refs(_rhs_result_expanded, engine)
                _canonical_for_eval = (
                    engine._handler.latex_to_canonical(_rhs_result_expanded)
                    or _rhs_result_expanded
                )
                _canonical, _h1_eval_latex = await loop.run_in_executor(
                    None, partial(engine.evaluate_formula_with_latex, _canonical_for_eval)
                )
            existing_id = _find_expr_by_canonical(engine, _canonical)
            if existing_id:
                canonical_to_id[_canonical] = existing_id
                # Attach LHS label to existing expression if it doesn't have one yet.
                # Use model_copy (immutable update) — do NOT mutate in-place.
                _existing_expr = engine.expressions.get(existing_id)
                if _existing_expr and "=" not in _existing_expr.latex:
                    engine.expressions[existing_id] = _existing_expr.model_copy(
                        update={"latex": f"{_lhs_display} = {_existing_expr.latex}"}
                    )
            else:
                # When IDs were expanded and CAS-evaluated, use the
                # postprocessed display LaTeX (dot notation, q(t)→q, etc.).
                # Fall back to "" (auto-render from canonical) if unavailable.
                _h1_latex = (_h1_eval_latex if _ids_were_expanded
                             else beautify_bare_canonical_latex(_rhs_raw))
                _h1_id = await _register_and_report(
                    engine, loop, _canonical, _h1_latex, registered,
                    "h1_equation", lhs_label=_lhs_display)
                if _h1_id is None:
                    result.append(tok)
                    continue
                canonical_to_id[_canonical] = _h1_id
            result.append(InputToken(type="math", latex=_rhs_raw, canonical=_canonical))
            continue

        # Fast path: pure formula input containing embedded E-IDs (e.g. "E5 + 2 * E7").
        # _FORMULA_OP_RE only matches "formula + operation text" patterns; a bare formula
        # with no Japanese instruction never matches.  Handle it here before that check.
        if (_INLINE_EXPR_ID_RE.search(raw)
                and _HAS_MATH_OP.search(raw)
                and not _is_nl_instruction(raw, lang, _names_an_operation(engine))  # not an NL command (ja: CJK, en: verb)
                and not _EXPR_ID_RE.match(raw)
                and not is_bracket_operation_text(raw)  # bracket → commutator, elsewhere
                ):
            formula_expanded = _expand_expr_ids(raw, engine)
            # Do NOT call _expand_symbol_refs here: inline formula input is treated
            # as a definition (same as H-1) — named symbols like g or V should stay
            # symbolic in the stored canonical.
            if formula_expanded != raw:   # at least one E-ID was resolved
                canonical = engine._handler.latex_to_canonical(formula_expanded) or formula_expanded
                # Register only — do NOT evaluate via CAS.
                # The user can explicitly run simplify/normal_order etc. afterwards.
                existing_id = _find_expr_by_canonical(engine, canonical)
                if existing_id:
                    canonical_to_id[canonical] = existing_id
                else:
                    _eid_id = await _register_and_report(
                        engine, loop, canonical, raw, registered,
                        "expr_id_formula")
                    if _eid_id is None:
                        result.append(tok)
                        continue
                    canonical_to_id[canonical] = _eid_id
                result.append(InputToken(type="math", latex=raw, canonical=canonical))
                continue

        # English differentiate on an inline (non-E-id) formula: register the
        # formula as its own expression FIRST (bypassing symbol-alias
        # substitution, same rationale as the other inline-formula paths in
        # this function — the formula's free variables must stay symbolic),
        # then rewrite the command to reference the new E-id so the existing
        # rule-based differentiate handler (_DIFF_RULE_RE_EN) runs on it.
        # Without this split, the raw formula text falls through to the
        # generic pipeline, where the "known symbol -> E-id" substitution pass
        # (2a-sym in run_interpret_pipeline) can replace a free variable in the
        # formula with its ALREADY-SOLVED value from earlier in the session
        # (e.g. x=0 from a prior solve_system) before differentiation ever
        # runs — silently corrupting the formula with no error
        # (SILENT-DIFF-VAR-ALIAS, found via the lotka_volterra practice-loop
        # scenario's English translation: "differentiate a*x-b*x*y with
        # respect to x" returned 0 instead of a-b*y once x had been solved
        # for elsewhere in the session).
        _di_m = _diff_inline_frame().match(raw) or _integrate_inline_formula(raw)
        _di_prefix = ""
        if _di_m is None and not _HAS_CJK_RE.search(raw):
            # The general rule: the operation's own alias is the boundary.  It
            # runs after the specific one because `differentiate F w.r.t. x` has
            # a TRAILING clause, which the alias alone cannot find.
            #
            # **先頭の計算動詞を飲んでから訊く**（MATRIX-INLINE-OPERAND-EN,
            # 2026-09-11）。この規則は文頭に操作の別名を要求するので、
            # `compute the determinant of [4, 0; 0, 1]` には当たらず、その場に
            # 書いた行列が**登録されないまま**「式を名指していない文」に見えて
            # 直前の式に落ちていた——実測、**別の行列の行列式**が `completed`
            # で返る（trace は直前の**結果**の跡）。日本語が通るのは同じ文が
            # `<式> の行列式を求めて` と**式で始まる**からである。
            # **飲んだぶんは組み直しで戻す。** 落としても答えは同じだった
            # （実測、19綴り）——が、**規則が変わる**: `matrix_trigger` は
            # 計算動詞を要求するので、`determinant E2` になると行列の
            # インターセプタが降り、別名走査が答える。そのインターセプタは
            # 「LLM を通さずに行列の族を回す」ために在り、固有値と特性多項式の
            # 区別のような**そこにしかない知識**を持っている。値が今たまたま
            # 一致することは、族全体で一致することの保証ではない。
            _di_pfx, _di_rest = split_leading_compute_verb(raw)
            _gm = _en_verb_formula_re().match(_di_rest)
            # …and the backend's own "is this a DECLARATION?" answer, which the
            # Japanese inline-formula path has always asked.  Measured
            # 2026-08-22: without it, `Define k, m, \mu, \hbar as scalars`
            # registered the declaration as a formula — fourteen scenarios.
            # One question, one answer.
            if (_gm and _formula_runs_to_the_end(_gm.group(2))
                    and not engine._handler.should_bypass_auto_register_formula(
                        _gm.group(2), raw)):
                _di_m, _di_prefix = _gm, _di_pfx
        if _di_m:
            _di_leadin, _di_formula, _di_trailing = _di_m.groups()
            _di_formula = _di_formula.strip()
            if not _EXPR_ID_RE.match(_di_formula) and _operand_is_mathematics(_di_formula):
                _di_id, _ = await _register_leading_formula(
                    engine, loop, _di_formula, registered, canonical_to_id)
                if _di_id:
                    result.append(InputToken(
                        type="text",
                        value=f"{_di_prefix}{_di_leadin} {_di_id} {_di_trailing}",
                    ))
                    continue

        m = _FORMULA_OP_RE.match(raw)
        if not m:
            result.append(tok)
            continue
        formula_str = m.group(1).strip()
        rest_str    = m.group(2).strip() if len(m.groups()) > 1 else ""
        if _EXPR_ID_RE.match(formula_str) or not _operand_is_mathematics(formula_str, rest_str):
            result.append(tok)
            continue
        # Bracket notation [X,Y] / {X,Y} is commutator/anticommutator input in
        # backends that declare it (Julia).  Do not auto-register; pass the full
        # text through so Phase 0 of the supervisor can handle it.
        if engine._handler.treats_bracket_as_commutator(formula_str) \
                and is_bracket_operation_text(formula_str):
            result.append(tok)
            continue
        # Backend veto: e.g. Julia particle operator definitions where
        # "a'(k) を光子の生成演算子とする" must reach extra_pre_phase1 intact.
        if engine._handler.should_bypass_auto_register_formula(formula_str, rest_str):
            result.append(tok)
            continue
        # The formula may be an OPERAND passed by name rather than the operation's
        # `expr`.  The English rule has always asked this (`_en_verb_formula_re`
        # only fires for operations whose required field is `expr`); this side
        # registered regardless, so the two languages numbered their expressions
        # differently — measured 2026-08-23, 3 of 48 scenarios, and in
        # `bcs_gap_equation` 12 of 14 E-ids meant different things in the two
        # languages.  One question, one answer: `operand_boundary`.
        if formula_is_an_operation_operand(
                rest_str,
                engine._validator.resolve_alias,
                engine._validator.max_alias_words,
                engine._validator.get_spec,
                lambda _op, spec: (engine.session.primary_backend
                                   in spec.get("supported_backends", [])),
                _tokenize_for_alias_scan):
            result.append(tok)
            continue
        # E-ids are expanded before conversion; named symbols (g, V, …) are NOT
        # (`_expand_symbol_refs` is deliberately not called) — inline formula
        # input is a DEFINITION, so its symbols stay symbolic in the canonical.
        _if_id, canonical = await _register_leading_formula(
            engine, loop, formula_str, registered, canonical_to_id)
        if _if_id is None:
            result.append(tok)
            continue
        result.append(InputToken(type="math", latex=formula_str, canonical=canonical))
        if rest_str:
            result.append(InputToken(type="text", value=rest_str))
    return result


# ---------------------------------------------------------------------------
# Main interpretation pipeline
# ---------------------------------------------------------------------------





async def _try_matrix_ops(engine, loop, supervisor_text, registered, lang="ja") -> "dict | None":
    """Rule-based short-circuit for linear-algebra matrix operations on a
    registered matrix: eigenvalues / eigenvectors / inverse / transpose /
    determinant / trace (unary, "Em の<op>を求めて" / "find the <op> of Em") and
    matrix_multiply ("Em と En の行列積を求めて" / "matrix product of Em and En").
    Maps the alias straight to run_operation so matrix workflows (e.g.
    diagonalization) run without the LLM.  NL keywords are in nl_patterns.json."""
    if not nlp("matrix_trigger", lang).search(supervisor_text):
        return None

    async def _dispatch(op: str, args: dict):
        try:
            _step, _val = await loop.run_in_executor(
                None, partial(engine.run_operation, op, args))
        except Exception as _e:
            return {"status": "failed", "error": str(_e),
                    "registered_expressions": registered}
        _oid = _step.output_refs[0] if _step.output_refs else None
        _oe = engine.expressions.get(_oid) if _oid else None
        return {"status": "completed", "operation": op, "expr_id": _oid,
                "step_id": _step.step_id,
                "latex": _oe.latex if _oe else "",
                "canonical": _oe.canonical if _oe else _val,
                "operation_label": _step.visible_action,
                "registered_expressions": registered}

    def _eid(token: str | None) -> "str | None":
        if token and token in engine.expressions and token in engine._active_expr_ids():
            return token
        return None

    # Binary: matrix product "Em と En の行列積を求めて" / "matrix product of Em and En".
    _mm = nlp("matrix_multiply", lang).search(supervisor_text)
    if _mm:
        _a, _b = _eid(_mm.group("a")), _eid(_mm.group("b"))
        if _a and _b:
            return await _dispatch("matrix_multiply", {"a": _a, "b": _b})

    # Unary ops.  Keyword → operation (more specific keywords first).  eigenvalues
    # is guarded against 固有値方程式/特性多項式 (characteristic_polynomial).
    _UNARY = [
        "diagonal_matrix", "diagonalize", "eigenvectors", "eigenvalues",
        "inverse_matrix", "transpose", "determinant", "trace",
    ]
    if nlp("matrix_char_poly_guard", lang).search(supervisor_text):
        return None                       # characteristic_polynomial path, not here
    for _op in _UNARY:
        if nlp(f"matrix_unary_{_op}", lang).search(supervisor_text):
            _m = EXPR_ID_IN_TEXT_RE.search(supervisor_text)
            _id = _eid(_m.group(1)) if _m else None
            if _id is None and may_fall_back_to_the_recent_expression(
                    engine.symbol_context, supervisor_text):
                _id = engine.most_recent_expr_id()
            if _id:
                return await _dispatch(_op, {"expr": _id})
            return None
    return None




# R1a (docs/27): the integral / FREE-PARAM pre-LLM interceptor batch as DATA.
# All share the (engine, loop, lang, text, registered) signature and return
# dict|None; the ordered list here IS the firing order, and each note records
# why the entry sits where it does (rationale that used to be an inline comment
# in the dispatcher).  run_interpret_pipeline appends this to _PRELUDE_INTERCEPTORS
# (below) and fires the whole run via run_interceptors; this batch occupies the
# tail — after change_vars, before inline-formula registration.  Adding a new
# interceptor of this shape is one entry here — no edit to the dispatcher.
_FREEPARAM_INTERCEPTORS: "list[Interceptor]" = [
    Interceptor(
        "integral_in_coordinates",
        lambda c: _try_integral_in_coordinates(c.engine, c.loop, c.lang, c.text,
                                               c.registered),
        note="`in <name> coordinates` is an adverbial phrase, not a variable — "
             "refuse with the spelling that works."),
    Interceptor(
        "definite_integral",
        lambda c: _try_definite_integral(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Definite integral over explicit bounds (closes the "
             "change-of-variables loop)."),
    Interceptor(
        "taylor_expand",
        lambda c: _try_taylor_expand(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Taylor / Maclaurin expansion; rule-based because the positional "
             "mapper mis-assigns its four fields (see the function's docstring)."),
    Interceptor(
        "series_expand_no_var",
        lambda c: _try_series_expand_no_var(c.engine, c.loop, c.lang, c.text,
                                            c.registered),
        note="次数を書いた展開なのに変数が書かれていない綴り。taylor_expand へ回して "
             "「var が要る」と言わせる（黙って代数展開に落ちていた）。"
             "taylor_expand の後——変数が書かれていればそちらが答える。"),
    Interceptor(
        "add_assumption",
        lambda c: _try_add_assumption(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Explicit researcher-stated CAS assumption; before "
             "solve_ode_general so a prior assume has taken effect in the CAS "
             "by the time the ODE is solved."),
    Interceptor(
        "solve_ode_general",
        lambda c: _try_solve_ode_general(c.engine, c.loop, c.lang, c.text, c.registered),
        note="FREE-PARAM: general ODE solution → named arbitrary constants; "
             "must precede the algebraic solve handler so 「一般に解いて」 is "
             "not mis-routed."),
    Interceptor(
        "define_laplace",
        lambda c: _try_define_laplace(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Laplace-transform definition card; anchors Q(s) to ∫₀^∞ q e^{-st} "
             "dt before the transform uses it."),
    Interceptor(
        "substitute_free_params",
        lambda c: _try_substitute_free_params(c.engine, c.loop, c.lang, c.text, c.registered),
        note="FREE-PARAM: fix free parameters by explicit value (initial "
             "conditions of an s-domain expression)."),
    Interceptor(
        "specialize_conditions",
        lambda c: _try_specialize_conditions(c.engine, c.loop, c.lang, c.text, c.registered),
        note="FREE-PARAM: specialize a general solution by SEVERAL conditions; "
             "before the single-condition handler."),
    Interceptor(
        "specialize_condition",
        lambda c: _try_specialize_condition(c.engine, c.loop, c.lang, c.text, c.registered),
        note="FREE-PARAM: specialize a general solution by a single condition "
             "(fix C, k₁, …)."),
    Interceptor(
        "angular_wigner_symbol",
        lambda c: _try_wigner_symbol(c.engine, c.loop, c.lang, c.text, c.registered),
        note="F-9: a bracketed 3j/6j/9j symbol. Heads the F-9 group because its "
             "trigger is the most literal ('3j記号 (…)') and it shares no words "
             "with the others."),
    Interceptor(
        "angular_clebsch_gordan",
        lambda c: _try_clebsch_gordan(c.engine, c.loop, c.lang, c.text, c.registered),
        note="F-9: Clebsch-Gordan coefficient from six named quantum numbers."),
    Interceptor(
        "angular_coupled_state",
        lambda c: _try_coupled_state(c.engine, c.loop, c.lang, c.text, c.registered),
        note="F-9: |J M> in the uncoupled basis. BEFORE angular_couple, whose "
             "'合成/couple' wording can also appear in the same sentence; the "
             "basis words ('非結合基底' / 'uncoupled basis') are what make this "
             "trigger distinct from the generic expand operation."),
    Interceptor(
        "angular_couple",
        lambda c: _try_couple(c.engine, c.loop, c.lang, c.text, c.registered),
        note="F-9: couple j1 ⊗ j2 — allowed totals + the j1·j2 identity."),
    Interceptor(
        "angular_wigner_eckart",
        lambda c: _try_wigner_eckart(c.engine, c.loop, c.lang, c.text, c.registered),
        note="F-9: Wigner-Eckart reduction of a matrix element."),
    Interceptor(
        "angular_lande_g",
        lambda c: _try_lande_g(c.engine, c.loop, c.lang, c.text, c.registered),
        note="F-9: Landé g factor. LAST of the F-9 group: its JP pattern is the "
             "loosest ('<anything> の g 因子'), so the specific triggers above "
             "must have had their chance first."),
    Interceptor(
        "z_transform_by_definition",
        lambda c: _try_z_transform_by_definition(c.engine, c.loop, c.lang, c.text, c.registered),
        note="F-8: definition-based Z-transform (sum, not integral) — same "
             "'by definition' family as the two Fourier entries above."),
    Interceptor(
        "en_genitive_noun",
        lambda c: _try_en_genitive_noun(c.engine, c.loop, c.lang, c.text, c.registered),
        note="`<op> the <noun> of <formula>` where the noun names no operation — "
             "LAST, because every specific reader above must have had its chance "
             "(F-1's `nearest-neighbor sum of …` is claimed before the run even "
             "starts). It is still BEFORE the inline-formula registration, which "
             "is what used to make `of`/`widget` FACTORS of the summand."),
]


# R1b (docs/27): the three NON-UNIFORM top-level interceptors, normalized to the
# registry's async (ctx) -> dict|None shape by threading their extra needs
# through InterceptorContext instead of a bespoke call site.  These are the
# entries R1a could not migrate as plain lambdas.
async def _try_declare_bracket_functional(engine, lang, text, registered) -> "dict | None":
    """R-9: "F を期待値汎函数とする" / "declare F as an expectation functional" →
    register F as a display-only bracket functional (F[X] renders as-is, the CAS
    gets cas(X); no operator algebra).  `期待値/expectation` → cas="expectation",
    otherwise the symbol's own name."""
    m = nlp("declare_bracket_functional", lang).search(text or "")
    if not m:
        return None
    sym = m.group("sym")
    cas = "expectation" if m.groupdict().get("exp") else sym
    engine.symbol_context.bracket_functionals[sym] = cas
    return {"status": "completed", "operation": "declare_bracket_functional",
            "message": t("notation.bracket_functional_declared", lang, sym=sym, cas=cas),
            "registered_expressions": registered}


async def _run_notation_interceptor(ctx: InterceptorContext) -> "dict | None":
    """notation candidates returns (response, rewritten_text).  Write the
    narrowed text back to ctx.text so the followers see the disambiguated form,
    and pass ctx.tokens (the |X| pipe rewrite mutates them in place)."""
    resp, ctx.text = await _try_notation_candidates(
        ctx.engine, ctx.loop, ctx.lang, ctx.text, ctx.tokens, ctx.registered)
    return resp


async def _run_early_interceptor(ctx: InterceptorContext) -> "dict | None":
    """early intercepts may raise SpinAlgebraComplexityLimitExceeded; convert it
    to a short-circuiting failed response (only THIS entry catches it, matching
    the old scoped try/except — a later interceptor raising it still propagates)."""
    try:
        return await _try_early_intercepts(
            ctx.engine, ctx.loop, ctx.lang, ctx.text, ctx.registered)
    except SpinAlgebraComplexityLimitExceeded:
        return {"status": "failed",
                "error": t("interpreter.spin_algebra_too_large", ctx.lang),
                "registered_expressions": ctx.registered}


# R1b (docs/27): the heterogeneous top-level interceptor run (vector-op through
# change-of-variables) as DATA — the same additive form R1a gave the FREE-PARAM
# batch, extended to the interceptors that were NOT uniform.  Order is the list;
# each note carries the ordering rationale that used to be an inline comment in
# run_interpret_pipeline.  The context threads the mutable raw text (the notation
# resolver narrows it for its followers), req_tokens (the |X| pipe rewrite), and
# the supervisor (change_vars bounds).
_PRELUDE_INTERCEPTORS: "list[Interceptor]" = [
    Interceptor(
        "vector_op",
        lambda c: _try_vector_op(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Vector operator (∇·E, div E, grad f, …) → COMPUTE via nb_div/…; "
             "must precede early intercepts, whose inner/cross-product parsers "
             "would otherwise grab \\nabla·E / \\nabla×E.  Strict field guard "
             "keeps it from stealing F-1/spin/L-1 inputs."),
    Interceptor(
        "declare_bracket_functional",
        lambda c: _try_declare_bracket_functional(c.engine, c.lang, c.text, c.registered),
        note="R-9: 「F を期待値汎函数とする」 → register a display-only bracket "
             "functional (F[X] → F(X)/expectation(X), no operator algebra).  "
             "Early, before H-1 equation registration would treat it as text."),
    Interceptor(
        "norm_notation",
        lambda c: try_norm_notation(c.engine, c.loop, c.lang, c.text, c.registered),
        note="二重縦棒 `\\|X\\|` / `||X||` と `\\begin{Vmatrix}` を `norm` 操作へ。"
             "**`notation_candidates` の前**に置く——あちらは同じ綴りを「絶対値か"
             "行列式か」のダイアログに落とさないよう断っており、断りが先に返ると"
             "配線が届かない。関係式（`\\|v\\| = 5`）は claim しないので、"
             "その形は従来どおりあちらの断りへ落ちる。"),
    Interceptor(
        "notation_candidates",
        _run_notation_interceptor,
        note="R-8: ambiguous notation (|X| abs vs det, undeclared Δ).  May "
             "SILENTLY narrow the raw text/tokens (declaration or prior choice) "
             "or return a resolve_notation needs_confirmation.  Early, so the "
             "narrowed text flows to every follower."),
    Interceptor(
        "plan_auto_select",
        lambda c: _try_plan_auto_select(c.engine, c.loop, c.lang, c.text,
                                        c.registered, c.supervisor),
        note="C-Ext-5 Supervisor auto-selection (2026-07-18): explicit "
             "「自動で解法を選んで」 → Adviser proposes candidates (same as "
             "plan_candidates), Supervisor picks ONE, still shown for "
             "confirmation.  Distinct phrase from plan_candidates, tried "
             "first so it isn't shadowed by the broader 「候補」 ask."),
    Interceptor(
        "plan_candidates",
        lambda c: _try_plan_candidates(c.engine, c.loop, c.lang, c.text, c.registered),
        note="C-Ext-5 priority 3: explicit 「解法の候補」 → Adviser candidates "
             "modal; before the fixed workflows so a candidates ask about an "
             "eigen problem is not swallowed by the eigensystem trigger."),
    Interceptor(
        "plan_freeform",
        lambda c: _try_plan_freeform(c.engine, c.loop, c.lang, c.text, c.registered),
        note="C-Ext-5 priority 3': explicit 「計画を立てて」/\"plan this\" → LLM "
             "plan proposal for a request matching no fixed workflow; reuses "
             "the same candidates modal/execute_plan_candidate path."),
    Interceptor(
        "plan_workflow",
        lambda c: _try_plan_workflow(c.engine, c.loop, c.lang, c.text, c.registered),
        note="C-Ext-5: fixed multi-step workflows (narrow composite triggers "
             "only — single-operation asks keep their 1-op route)."),
    Interceptor(
        "scalar_handoff",
        lambda c: _try_scalar_handoff(c.engine, c.loop, c.lang, c.text, c.registered),
        note="SCALAR-HANDOFF (docs/34 §5): explicit `E5 を <backend> セッションに"
             "渡して` → validate, snapshot provenance, and WRITE the target "
             "session file (not opened — same as /fork).  Sits beside "
             "numeric_fork_nl: both are 'this expression leaves the theory "
             "timeline and continues elsewhere', and both name their target "
             "explicitly so neither can swallow an ordinary compute-ask."),
    Interceptor(
        "numeric_fork_nl",
        lambda c: _try_numeric_fork_nl(c.engine, c.loop, c.lang, c.text, c.registered),
        note="N-1 NL trigger (open_items.md): explicit single-variable numeric "
             "sweep ask → deterministic-template fork, confirmed by click; "
             "sits after plan_workflow (same fixed/deterministic composite-ask "
             "family) and before early_intercepts so a clean sweep phrase is "
             "never swallowed by a broader compute-ask handler."),
    Interceptor(
        "early_intercepts",
        _run_early_interceptor,
        note="F-1 / spin-algebra / L-1 rule intercepts; may raise "
             "SpinAlgebraComplexityLimitExceeded → failed response (see the "
             "wrapper)."),
    Interceptor(
        "poisson",
        lambda c: _try_poisson(c.engine, c.loop, c.lang, c.text, c.registered),
        note="P-1: canonical-pair declaration / Poisson bracket; must precede "
             "H-1 inline registration so 「p_r を r の共役運動量とする」 is not "
             "mis-parsed as a formula."),
    Interceptor(
        "expr_id_equation",
        lambda c: _try_expr_id_equation(c.engine, c.loop, c.lang, c.text, c.registered),
        note="「E_n = X」 → explicit equation <value of E_n> = X; must precede "
             "inline-formula registration, which would otherwise store the "
             "literal 'E3 = 0'."),
    Interceptor(
        "extremum_input",
        lambda c: _try_extremum_input(c.engine, c.loop, c.lang, c.text, c.registered),
        note="EXTREMUM-BINDER: display-only `\\max_i x_i`; before inline-formula "
             "handling so the binder is not read as a subscript (which computed "
             "`i*x(i)`).  Needs a SUBSCRIPT, so `\\max(a, b)` still computes."),
    Interceptor(
        "integral_input",
        lambda c: _try_integral_input(c.engine, c.loop, c.lang, c.text, c.registered),
        note="S-o: display-only definite-integral object (LaTeX ∫∫ or region "
             "form); register before inline-formula handling so the ∫ is not "
             "parsed as a plain expression."),
    Interceptor(
        "evaluate_extremum",
        lambda c: _try_evaluate_extremum(c.engine, c.loop, c.lang, c.text,
                                         c.registered),
        note="EXTREMUM-BINDER: evaluate a CONCRETE-range extremum. Fires only "
             "when the target really is one, so the same phrase still reaches "
             "evaluate_integral and the ordinary compute-ask."),
    Interceptor(
        "evaluate_integral",
        lambda c: _try_evaluate_integral(c.engine, c.loop, c.lang, c.text, c.registered),
        note="Evaluate an unevaluated integral object (S-o step 4); before "
             "generic compute."),
    Interceptor(
        "expand_components",
        lambda c: _try_expand_components(c.engine, c.loop, c.lang, c.text,
                                         c.registered),
        note="Open a vector expression into components — the only way a "
             "volume measure (\\int d^3k) reaches the CAS at all."),
    Interceptor(
        "change_vars",
        lambda c: _try_change_vars(c.engine, c.loop, c.lang, c.text, c.registered,
                                   c.supervisor),
        note="S-o: multivariable change of integration variables (Jacobian); "
             "rule-based deterministic parse.  Uses the supervisor only to "
             "PROPOSE new bounds when no table/override applies.  Last prelude "
             "entry, before the FREE-PARAM batch."),
]


async def run_interpret_pipeline(
    req: InterpretRequest,
    engine: NotebookEngine,
    supervisor: LLMSupervisor,
) -> dict:
    """Execute the full interpret pipeline and return the HTTP response dict.

    Steps:
    0a. Normalize full-width ASCII → half-width.
    0.  Auto-detect and register inline "formula + Japanese instruction" patterns.
    1.  Register math tokens.
    2.  Build supervisor text; detect pure-formula input; inject expr ID if absent.
    3.  Interpret via LLMSupervisor; post-process (var inference, taylor point, implicit diff).
    4.  Auto-execute or return needs_confirmation.
    """
    loop = asyncio.get_event_loop()
    lang = engine.lang

    # 0a. Normalize full-width ASCII → half-width
    req_tokens = _normalize_tokens(req.tokens)

    # 0b. Expand expression IDs in math tokens' canonical.
    # E-ID refs ("J * E1 * E4") are resolved here so they can be looked up.
    # Named symbol refs (g, V, …) are NOT expanded: math token input is treated
    # as a definition — symbols should stay symbolic in the stored canonical.
    for tok in req_tokens:
        if tok.type == "math" and tok.canonical:
            expanded = _expand_expr_ids(tok.canonical, engine)
            if expanded != tok.canonical:
                tok.canonical = expanded

    # 0. Auto-detect inline "formula + operation" patterns in plain-text tokens
    registered: list[dict] = []
    canonical_to_id: dict[str, str] = {}

    # 0c. F-1: "<body> の最近接和を N サイト [周期境界]で展開して".
    # Detected on the RAW text BEFORE inline-formula auto-registration, so the
    # operator body (e.g. S^z_i S^z_j) is not first swallowed into a standalone
    # expression (which would hide the site indices from the parser).
    _f1_raw = " ".join(
        (tok.value if tok.type == "text" else (tok.latex or tok.canonical or ""))
        for tok in req_tokens
    ).strip()

    # 2b. Record the raw user input as a YOU-card step BEFORE any rule-based or LLM
    # operation — including the F-1 / spin-algebra / L-1 intercepts below, which all
    # return early.  Without this, those commands (e.g. "格子図を保存して", "…の最近接
    # 和を…展開して") leave no YOU card and never appear in history or the TeX export.
    # Use the original req.text (before alias scan / inject_id) so the card shows what
    # the user actually typed; the user_input step then shares the parent of the
    # immediately following state-affecting step (companion / YOU-card mechanism).
    _raw_user_text = getattr(req, "text", "") or _f1_raw
    _youcard_recorded = False
    if _raw_user_text.strip():
        await loop.run_in_executor(
            None, partial(engine.record_user_input, _raw_user_text)
        )
        _youcard_recorded = True

    # 2b-uni. One spelling for every rule below (see normalize_request_greek).
    _f1_raw = normalize_request_greek(_f1_raw, req_tokens)

    # Top-level pre-LLM interceptor run (R1, docs/27).  The firing ORDER and the
    # rationale for each position are the data in _PRELUDE_INTERCEPTORS +
    # _FREEPARAM_INTERCEPTORS, not code position here; run_interceptors fires them
    # in order and returns the first non-None response.  The context threads the
    # mutable raw text (the notation resolver narrows it for its followers),
    # req_tokens (the |X| pipe rewrite mutates them in place), and the supervisor
    # (change_vars bounds proposal).  Adding a new interceptor is "write a _try_* +
    # add one list entry", never "edit this function".
    _res = await run_interceptors(
        InterceptorContext(engine=engine, loop=loop, lang=lang, text=_f1_raw,
                           registered=registered, tokens=req_tokens,
                           supervisor=supervisor),
        _PRELUDE_INTERCEPTORS + _FREEPARAM_INTERCEPTORS)
    if _res is not None:
        return _res
    tokens = await _auto_register_inline_formula(
        req_tokens, engine, loop, registered, canonical_to_id, lang
    )

    # A DECLARATION's subject is not an expression.  The same question — "does
    # this sentence declare something, so the formula must reach the declaration
    # path intact?" — is already asked of the backend on the inline-formula path
    # (`should_bypass_auto_register_formula`, which is why `a'(k) を光子の生成
    # 演算子とする` works when typed as text).  The MATH-TOKEN path never asked
    # it, so the same declaration written with a math chip — which is what the
    # math input box and the handwriting panel produce — registered `\hat H_0`
    # as an expression and handed the interceptors `E1 をスカラーとする`, which
    # nothing recognises (measured 2026-08-22).  One question, one answer.
    _declaration_text = " ".join(t_.value for t_ in tokens if t_.type == "text")
    _subject_tokens = {
        id(t_) for t_ in tokens
        if t_.type == "math" and _declaration_text
        and engine._handler.should_bypass_auto_register_formula(
            t_.latex or t_.canonical, _declaration_text)
    }

    # 1. Register math tokens (skip if already registered)
    for token in tokens:
        if token.type != "math" or id(token) in _subject_tokens:
            continue
        _canonical_pre = token.canonical.strip() or engine._handler.latex_to_canonical(token.latex)
        # Expand E-ID references here too: the "0b" pass only runs when the client
        # pre-set canonical, but the UI sends LaTeX with an EMPTY canonical (derived
        # just above), so "L*E3 + R*E2 + E1/C = v" would otherwise keep E1,E2,E3 as
        # bare symbols.  Reveal the referenced content at input time (display→eval,
        # matching the integral-form design).
        canonical = _expand_expr_ids(_canonical_pre, engine)
        if not canonical:
            continue
        existing_id = _find_expr_by_canonical(engine, canonical)
        if existing_id:
            canonical_to_id[canonical] = existing_id
            continue
        # When expansion revealed E-ID content, DISPLAY the explicit form (render the
        # expanded canonical with dot notation) instead of the bare-reference input;
        # otherwise keep the input LaTeX.
        _disp = (engine.canonical_display_latex(canonical)
                 if canonical != _canonical_pre else (token.latex or canonical))
        try:
            expr = await loop.run_in_executor(
                None,
                partial(engine.register_expression, canonical, latex=_disp),
            )
        except Exception as e:
            raise RuntimeError(str(e))
        canonical_to_id[canonical] = expr.display_id
        registered.append({
            "expr_id":          expr.display_id,
            "canonical":        expr.canonical,
            "latex":            expr.latex,
            "step_id":          expr.created_by_step,
            "undefined_symbols": expr.undefined_symbols,
            "warnings": expr.warnings,
        })

    # 2. Build supervisor text
    parts: list[str] = []
    for token in tokens:
        if token.type == "text":
            parts.append(token.value)
        elif token.type == "math":
            if id(token) in _subject_tokens:
                # 宣言の主語は**研究者の綴りのまま**渡す（E-ID ではない）。
                parts.append(token.latex or token.canonical)
                continue
            canonical = token.canonical.strip() or engine._handler.latex_to_canonical(token.latex)
            canonical = _expand_expr_ids(canonical, engine)
            parts.append(canonical_to_id.get(canonical, token.latex))
    supervisor_text = " ".join(parts).strip()
    # What the researcher actually wrote, before 2a-sym rewrites their symbols
    # into E-IDs for the LLM's benefit.  Anything RECORDED must come from this:
    # an expression is a record of what the user wrote (CLAUDE.md, "canonical
    # is the user's own writing"), and 2a-sym exists only to disambiguate the
    # supervisor's input.  Registering the rewrite instead put an internal id
    # in the display AND the canonical — after "G = a + b", "G^2 + 1" was
    # recorded as `E1^2 + 1` (H1-ALIAS-EID-LEAK, measured 2026-08-01).  The
    # math-token → E-ID mapping above is deliberately kept — that one IS the
    # user referring to an existing expression.
    alias_free_text = supervisor_text

    supervisor_text = rewrite_symbols_to_eids(supervisor_text, engine)
    # Edge case: only math tokens registered, no operation text
    has_op_text = any(
        token.type == "text" and token.value.strip()
        for token in tokens
    )
    if not has_op_text and registered:
        return {"status": "registered", "registered_expressions": registered}

    if not supervisor_text:
        return {"status": "failed", "error": t("interpreter.empty_input", lang),
                "registered_expressions": registered}

    # Fallback YOU-card recording for inputs whose raw text was empty at the
    # earlier record point (e.g. pure-math-token requests) but produced a
    # supervisor_text here.  Skipped when the early record above already fired.
    if not _youcard_recorded:
        _raw_user_text = getattr(req, "text", "") or supervisor_text
        await loop.run_in_executor(
            None, partial(engine.record_user_input, _raw_user_text)
        )
        _youcard_recorded = True

    _eq = await _try_equation_intercepts(engine, loop, lang, supervisor_text,
                                         registered, alias_free_text)
    if _eq is not None:
        return _eq
    # 2c. Pure-formula detection: ASCII/math-only input with no Japanese instruction
    #     → register as expression directly, skip LLM supervisor.
    #     Guard: skip if the text contains 2+ non-math-function English words (≥3 letters)
    #     — that pattern indicates a natural-language instruction
    #     (e.g. "define a(1) as a bose operator") rather than a formula.
    #     Known math function names (sin, cos, log, …) are excluded from the count so
    #     that "sin(x)^2 + cos(x)^2" is still recognised as a formula.
    # Words that appear as LaTeX commands (\word) are math notation, not NL words.
    # An identifier IMMEDIATELY followed by '(' is a function CALL, not an English
    # word — any callable counts, not just the _KNOWN_MATH_FUNCTIONS whitelist
    # (BP-8: `diff(V,t) + diff(V,S,2) = 0` scored 2 "words" on its two `diff`
    # calls and lost the pure-formula path; the same held for any user-defined
    # f(...)).  The '(' must be adjacent: "expand (x+1)^2" leaves a space, so the
    # NL verb there is still counted.
    _non_func_word_count = _nl_word_count(supervisor_text)
    if (not registered
            and not _is_nl_instruction(supervisor_text, lang,
                                       _names_an_operation(engine))
            and _HAS_MATH_OP.search(supervisor_text)
            and _non_func_word_count < 2
            and not _EXPR_ID_RE.match(supervisor_text.strip())
            and engine._handler.operands_provided(supervisor_text) is not True):
        # Record what the user wrote (see alias_free_text) — the guards above
        # deliberately stay on supervisor_text: what MAY be registered is
        # unchanged, only what is written down.
        display_tex = engine._handler.normalize_display_latex(alias_free_text)
        canonical = engine._handler.latex_to_canonical(display_tex) or display_tex
        existing_id = _find_expr_by_canonical(engine, canonical)
        if existing_id:
            src = engine.expressions[existing_id]
            registered.append({
                "expr_id":          existing_id,
                "canonical":        src.canonical,
                "latex":            src.latex,
                "step_id":          src.created_by_step,
                "undefined_symbols": src.undefined_symbols,
            })
        else:
            try:
                expr = await loop.run_in_executor(
                    None,
                    partial(engine.register_expression, canonical, latex=display_tex),
                )
                registered.append({
                    "expr_id":          expr.display_id,
                    "canonical":        expr.canonical,
                    "latex":            expr.latex,
                    "step_id":          expr.created_by_step,
                    "undefined_symbols": expr.undefined_symbols,
                    "warnings": expr.warnings,
                })
            except Exception:
                # Expression registration failed; fall through to the supervisor.
                # Logged so a genuine registration bug leaves a trace.
                _logger.debug("expression registration failed; deferring to supervisor",
                              exc_info=True)
        if registered:
            return {"status": "registered", "registered_expressions": registered}

    # 2d. Inject expression ID into supervisor_text when none is mentioned.
    #     Priority: (1) UI-selected expression, (2) most recent expression in lineage.
    sel_id = req.selected_expr_id.strip() if req.selected_expr_id else ""

    _backend_vote = engine._handler.operands_provided(supervisor_text)
    _needs_inject = (
        False if _backend_vote is True
        else True if _backend_vote is False
        else supervisor.needs_expression_injection(supervisor_text)
    )
    #: **足場は、名前を持つ対象を黙って別の式に差し替えてはならない。**
    #: 宣言だけされた対象（ベクトル場・行列・演算子…）には `defining_expr_id` が
    #: 無いので 2a-sym が式番号へ書き換えられず、その文は「式を名指していない」と
    #: 見なされて足場が**直前の式**を前置する。結果、研究者が名指した対象について
    #: 訊いたのに**別の式の答えが `completed` で返る**——実測 2026-09-11、
    #: `A のノルムを求めて` が直前に登録した `u` のノルム（5）を返し、
    #: 直前が別のベクトルなら 10 を返した。**答えが「直前に何をしたか」で変わる。**
    #:
    #: 名指しているなら前置しない。そうすると操作は「必要な引数が不足」で断るので、
    #: **誤答が失敗に変わる**——能力にはならないが、沈黙した誤りは消える。
    #: 名前をオペランドとして通す半分は式クラスの欄に触るので別（open_items
    #: OPERAND-BOUNDARY 残①）。通る道は在って、**実測してある**:
    #: `\|A\|` は `sqrt(A_z^2+A_y^2+A_x^2)`、`\nabla \times A` は3成分を返す。
    #:
    #: **実在の指示は1行も変わらない**——コーパス1102行を足場ありとなしで2回測り、
    #: 差分0（「文に書かれていない式番号がオペランドに入る」55行はすべて 2a-sym が
    #: 名前を書き換えたもので、名前は式を持つので通り続ける）。
    #: 渡せない名前も「文が名指したもの」である（③、実測は guard 側の docstring）。
    #: 位置を知っているのは文法なので、ここで読んで渡す。**照合は正準の綴りで**
    #: ——研究者は `E_{total}` と書き、量は `E_total` で登録されている。
    _operand_written = _operand_name_in_text(supervisor_text)
    _operand_name = _operand_key(engine, _operand_written)
    if _needs_inject and may_fall_back_to_the_recent_expression(
            engine.symbol_context, supervisor_text,
            operand_names=(_operand_name,) if _operand_name else ()):
        # 「どれが直前の式か」は族で1つ（`most_recent_expr_id` の docstring に
        # 3通り在った経緯と、畳んだ理由の実測）。
        inject_id = (sel_id if sel_id and sel_id in engine.expressions
                     else engine.most_recent_expr_id())
        if inject_id:
            # 足場であって研究者が書いた文ではない。言語ごとに書くのは、読む
            # 相手がその言語の自然言語パーサと（設定されていれば）LLM だから。
            # 記録に入れてはいけない理由は `record_llm_supervision` の docstring。
            supervisor_text = t("interpreter.expr_ref_prefix", lang,
                                eid=inject_id, text=supervisor_text)

    #: 記録に残す文 —— **研究者が打ったほう**（`supervisor_text` は 2a-sym と
    #: 2d の注入を受けた足場）。契約は `record_llm_supervision` の docstring。
    _typed_text = _raw_user_text or supervisor_text

    # 2e. Rule-based pre-extraction of differentiation order ("2階微分", "三次微分", etc.)
    # Strips the "n階" / "n次" token from supervisor_text before LLM to reduce parsing load,
    # then injects order= into the result args after interpretation.  Externalized to
    # i18n/nl_patterns.json["order_clause"] (named groups digit / word / ntimes).
    # 「回」is accepted as a JA derivative-order counter ("2回微分" = differentiate
    # twice) — the lookahead keeps it scoped to differentiation.  English "twice"/
    # "N times" is only tried when lang!='ja' and the text contains "differentiat"
    # (those bare words aren't otherwise order-specific enough to strip).
    _rule_order: int | None = None
    if lang == "ja":
        _m_ord = nlp("order_clause", "ja").search(supervisor_text)
        if _m_ord:
            _digit_str = _m_ord.group("digit")
            _rule_order = number_word(_digit_str)
            # Strip only the "n階" portion; leave 微分/偏微分 in place for the LLM.
            supervisor_text = (supervisor_text[:_m_ord.start()]
                               + supervisor_text[_m_ord.end():]).strip()
    elif re.search(r'(?i)differentiat', supervisor_text):
        _m_ord_en = nlp("order_clause", lang).search(supervisor_text)
        if _m_ord_en:
            _en_gd = _m_ord_en.groupdict()
            _word = _en_gd.get("word")
            _rule_order = number_word(_word or _en_gd.get("ntimes"))
            supervisor_text = (supervisor_text[:_m_ord_en.start()]
                               + supervisor_text[_m_ord_en.end():])
            supervisor_text = re.sub(r'\s{2,}', ' ', supervisor_text).strip()

    _sc = await _try_rule_short_circuits(
        req, engine, loop, lang, supervisor_text, registered, _rule_order,
        interactive=getattr(supervisor, "_llm", None) is not None,
        # A rule that names an OPERAND reads it here (operands_as_written).
        alias_free_text=alias_free_text)
    if _sc is not None:
        return _sc
    _nm = await _try_name_assignment(engine, loop, supervisor_text, registered, lang)
    if _nm is not None:
        return _nm
    _mx = await _try_matrix_ops(engine, loop, supervisor_text, registered, lang)
    if _mx is not None:
        return _mx
    # 3. Interpret — restrict known_ids to expressions reachable from current branch lineage
    from notebook.runtime.replay import compute_lineage
    lineage_steps = compute_lineage(
        engine.active_branch_id, engine.steps, engine.branches
    )
    lineage_expr_ids: set[str] = set()
    for step in lineage_steps:
        for ref in (step.output_refs or []):
            lineage_expr_ids.add(ref)
        for ref in (step.input_refs or []):
            lineage_expr_ids.add(ref)
    known_ids = lineage_expr_ids & set(engine.expressions.keys()) or set(engine.expressions.keys())
    result = await loop.run_in_executor(
        None, partial(supervisor.interpret, supervisor_text, known_ids, engine.symbol_context, lang,
                      engine._handler)
    )

    # 3-apply. Apply SymbolContext updates (metadata only, no Maxima state).
    # depends() injections are deferred until the user confirms the operation;
    # see pending_dependencies in the needs_confirmation response below.
    if result.ok and result.symbol_updates:
        for upd in result.symbol_updates:
            name = upd.get("name")
            if name:
                kwargs = {k: v for k, v in upd.items() if k not in ("name", "dependencies")}
                if "dependencies" in upd:
                    kwargs["natural_vars"] = upd["dependencies"]
                try:
                    await loop.run_in_executor(
                        None, partial(engine.register_quantity, name, **kwargs)
                    )
                except Exception:
                    pass  # invalid kind from LLM — skip silently

    # 3a. Inject rule-extracted order when LLM omitted it.
    if (result.operation == "differentiate"
            and _rule_order is not None
            and _rule_order > 1
            and "order" not in result.args):
        result = dataclasses.replace(result, args={**result.args, "order": str(_rule_order)})

    # 3a-sym. Resolve symbol names / LHS labels to expression IDs.
    # Covers plain "H", decorated "\bar{H}", compound "H + V",
    # expectation "<H>", derivative "\frac{dH}{dt}", etc.
    #
    # Lookup order (first match wins):
    #   1. lhs_alias_map exact match  (covers ALL LHS forms)
    #   2. lhs_alias_map normalized match (whitespace / thin-space differences)
    #   3. SymbolContext.defining_expr_id exact match
    #   4. SymbolContext via _extract_simple_symbol_name normalization
    if result.ok and result.args:
        from notebook.engine import (
            _extract_simple_symbol_name as _esym,
            _normalize_lhs as _nlhs,
        )
        _resolved_args = dict(result.args)
        # "a" and "b" are commutator / anticommutator operand fields.
        # They must be resolved here so that "H と Sy の交換子" always uses
        # the defining expression for H (E1) rather than the most-recent result.
        for _field in ("expr", "expr_a", "expr_b", "a", "b"):
            _val = _resolved_args.get(_field, "")
            if not (_val and isinstance(_val, str) and not re.match(r'^E\d+$', _val)):
                continue

            # Step 1+2: lhs_alias_map
            _resolved_id = (
                engine.lhs_alias_map.get(_val)
                or engine.lhs_alias_map.get(_nlhs(_val))
            )
            if _resolved_id:
                _resolved_args[_field] = _resolved_id
                continue

            # Step 3: SymbolContext exact match
            _qty = engine.symbol_context.get_quantity(_val)
            if _qty is None:
                # Step 4: normalized decorated form (\bar{H} → H_bar)
                _norm = _esym(_val)
                if _norm:
                    _qty = engine.symbol_context.get_quantity(_norm)
            if _qty is not None and _qty.defining_expr_id:
                _resolved_args[_field] = _qty.defining_expr_id

        if _resolved_args != result.args:
            result = dataclasses.replace(result, args=_resolved_args)

    # 3b-pre. If multiple candidates differ only by `expr` and the user did not
    # mention any expression ID explicitly, default to the most recent expression
    # in the lineage so the user sees a single clear confirmation prompt.
    if (result.requires_confirmation and len(result.candidates) >= 1
            and may_fall_back_to_the_recent_expression(
                engine.symbol_context, supervisor_text)):
        all_cands = [{"operation": result.operation, "args": result.args}] + list(result.candidates)
        ops = {c["operation"] for c in all_cands}

        def _hashable(v):
            # Args may carry list/dict values (e.g. a subscript-confirmation
            # candidate's choice list); make them hashable for set-based dedup.
            if isinstance(v, (list, tuple)):
                return tuple(_hashable(x) for x in v)
            if isinstance(v, dict):
                return tuple(sorted((k, _hashable(val)) for k, val in v.items()))
            return v
        non_expr = [frozenset((k, _hashable(v)) for k, v in c["args"].items() if k != "expr")
                    for c in all_cands]
        if len(ops) == 1 and len(set(non_expr)) == 1:
            latest_expr_id = engine.most_recent_expr_id()
            if latest_expr_id:
                spec = supervisor._v.get_spec(result.operation)
                display = (spec["display"].get(lang) or spec["display"].get("en")
                           if spec else result.operation)
                new_args = {**result.args, "expr": latest_expr_id}
                result = dataclasses.replace(
                    result,
                    args=new_args,
                    confirmation_prompt=t("interpreter.confirm_operation", lang,
                                          expr_id=latest_expr_id, display=display),
                    candidates=[],
                )

    # 3b-pre. When LLM returned a 'var' that does not appear in the expression,
    # strip it so the auto-infer step below can substitute the correct variable.
    # Example: f(t)=sin(t) → LLM returns var=x, but x is not in {t}.
    if (result.operation in _VAR_REQUIRED_OPS
            and 'var' in result.args
            and result.source == 'llm'):
        expr_id = result.args.get('expr')
        _src_expr = engine.expressions.get(expr_id) if expr_id else None
        if _src_expr:
            _llm_var = result.args['var']
            _expr_vars = _extract_variables(_src_expr.display_repr)
            if _expr_vars and _llm_var not in _expr_vars:
                result = dataclasses.replace(
                    result,
                    args={k: v for k, v in result.args.items() if k != 'var'},
                )

    # 3b. Auto-infer 'var' when it is missing from result.args.
    # Only runs when the user/LLM did not specify a variable; never overrides
    # an already-extracted 'var'.
    if result.operation in _VAR_REQUIRED_OPS and 'var' not in result.args:
        expr_id = result.args.get('expr')
        expr = engine.expressions.get(expr_id) if expr_id else None
        if expr:
            inferred = _extract_variables(expr.display_repr)
            # The halves decide the inferred differentiation variable.  No input
            # was found that flips the DECISION (measured 2026-08-13 — the
            # counts are not 1 either way), but the material it is decided from
            # was wrong: for `\sum_{i=1}^3*c_i = y` the naive left side gave the
            # variables of `\sum_{i`.
            _rel = split_relation(expr.canonical)
            if len(inferred) != 1 and _rel is not None and "=" in _rel[1]:
                lhs_vars = _extract_variables(_rel[0])
                rhs_vars = _extract_variables(_rel[2])
                if len(lhs_vars) == 1 and len(rhs_vars) == 1 and lhs_vars != rhs_vars:
                    inferred = rhs_vars
                # Derivative equations like "'diff(y,x,1) = cos(x)": extract
                # the differentiation variable directly from the 'diff notation.
                elif len(inferred) != 1:
                    _dm = re.search(r"'diff\(\w+,\s*([A-Za-z]\w*)\s*,", expr.canonical)
                    if _dm:
                        inferred = {_dm.group(1)}
            if len(inferred) == 1:
                inferred_var = next(iter(inferred))
                new_args = {**result.args, 'var': inferred_var}
                spec = supervisor._v.get_spec(result.operation)
                still_missing = [f for f in (spec['required_fields'] if spec else [])
                                 if f not in new_args]
                display = (spec["display"].get(lang) or spec["display"].get("en")
                           if spec else result.operation)
                prompt = t("interpreter.auto_var_detected", lang,
                           var=inferred_var, display=display)
                if still_missing:
                    if result.operation == 'taylor_expand' and 'order' in still_missing:
                        prompt += "\n" + t("interpreter.taylor_missing_order", lang)
                    else:
                        prompt += t("interpreter.auto_var_missing", lang, missing=still_missing)
                result = dataclasses.replace(
                    result,
                    ok=not bool(still_missing),
                    args=new_args,
                    error=None if not still_missing else f"Missing required fields: {still_missing}",
                    requires_confirmation=True,
                    confirmation_prompt=prompt,
                    candidates=[],  # clear LLM's spurious multi-var candidates
                )

    # 3c-pre. 展開中心を読み、研究者に何を伝えるかを決める。判断そのものは
    # `interpret_common.taylor_center_note` に1つだけ置いてある——ここと
    # `interpret_integral` のインターセプタが同じ問いに別々に答えていて、
    # 後者は黙って 0 を入れていた（実測 2026-09-05）。
    if (result.operation == 'taylor_expand' and 'point' in result.args
            and result.source == 'rule'):
        point_val, note = taylor_center_note(result.args['point'], lang)
        if note:
            result = dataclasses.replace(
                result,
                args={**result.args, 'point': point_val},
                requires_confirmation=True,
                confirmation_prompt=(result.confirmation_prompt + "\n" + note
                                     if result.confirmation_prompt
                                     else note + t("interpreter.confirm_run", lang)),
            )
        elif point_val != result.args['point']:
            result = dataclasses.replace(
                result, args={**result.args, 'point': point_val})

    # 3c. For taylor_expand: default missing 'point' to 0 (Maclaurin expansion)
    if result.operation == 'taylor_expand' and 'point' not in result.args:
        new_args = {**result.args, 'point': '0'}
        spec = supervisor._v.get_spec('taylor_expand')
        still_missing = [f for f in (spec['required_fields'] if spec else [])
                         if f not in new_args]
        note = t("interpreter.taylor_maclaurin", lang)
        if still_missing:
            order_msg = (t("interpreter.taylor_missing_order", lang)
                         if "order" in still_missing else
                         t("interpreter.auto_var_missing", lang, missing=still_missing))
            prompt = note + "\n" + order_msg
        else:
            prompt = note + t("interpreter.confirm_run", lang)
        result = dataclasses.replace(
            result,
            ok=not bool(still_missing),
            args=new_args,
            error=None if not still_missing else f"Missing required fields: {still_missing}",
            requires_confirmation=True,
            confirmation_prompt=(result.confirmation_prompt + "\n" + note
                                 if result.confirmation_prompt and result.ok
                                 else prompt),
            candidates=[],
        )
    # 3d. For differentiate on a "var = rhs" equation: ask whether to treat
    #     the LHS variable as a function of the differentiation variable.
    #     Only triggered when the equation LHS is a plain variable (e.g. y, E)
    #     — not a function-call form like f(x).
    _SIMPLE_VAR_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')
    if (result.operation == 'differentiate'
            and result.ok
            and 'expr' in result.args and 'var' in result.args):
        _eid = result.args['expr']
        _expr = engine.expressions.get(_eid)
        # `_lhs` decides whether the dialog appears at all, is shown inside its
        # prompt, and travels on as the `_implicit_lhs` argument.
        _rel_d = split_relation(_expr.canonical) if _expr else None
        if _expr and _rel_d is not None and "=" in _rel_d[1]:
            _lhs = _rel_d[0]
            if _SIMPLE_VAR_RE.match(_lhs) and '(' not in _lhs:
                _var = result.args['var']
                result = dataclasses.replace(
                    result,
                    ok=True,
                    requires_confirmation=True,
                    confirmation_prompt=t("interpreter.eq_diff_choice", lang,
                                          expr_id=_eid, lhs=_lhs),
                    candidates=[
                        {
                            "operation": "differentiate",
                            "args": {**result.args, "_implicit_lhs": _lhs},
                            "display_ja": t("interpreter.eq_diff_as_func", "ja",
                                            lhs=_lhs, var=_var),
                            "display_en": t("interpreter.eq_diff_as_func", "en",
                                            lhs=_lhs, var=_var),
                        },
                        {
                            "operation": "differentiate",
                            "args": result.args,
                            "display_ja": t("interpreter.eq_diff_both_sides", "ja"),
                            "display_en": t("interpreter.eq_diff_both_sides", "en"),
                        },
                    ],
                )

    # 3e. Generic pending_op: when exactly one required field is still missing,
    # store the partial operation so a bare follow-up answer can complete it.
    _still_gen: list = []
    if not result.ok and result.operation:
        _spec_gen = supervisor._v.get_spec(result.operation)
        _still_gen = [f for f in (_spec_gen['required_fields'] if _spec_gen else [])
                      if f not in result.args]
        if len(_still_gen) == 1:
            _missing_field = _still_gen[0]
            engine.pending_op = {
                "operation": result.operation,
                "args":      result.args,
                "missing":   _still_gen,
            }
            # Replace generic "Missing required fields: [...]" with a helpful prompt.
            _field_msg_key = f"interpreter.missing_field.{_missing_field}"
            _field_msg = t(_field_msg_key, lang)
            if _field_msg == _field_msg_key:  # key not found → fall back to generic
                _field_msg = t("interpreter.missing_field", lang, field=_missing_field)
            _base = result.confirmation_prompt or ""
            if "Missing required fields" in _base or not _base:
                result = dataclasses.replace(result, confirmation_prompt=_field_msg)
            else:
                result = dataclasses.replace(
                    result, confirmation_prompt=_base.rstrip() + "\n" + _field_msg
                )
        else:
            engine.pending_op = None

    if not result.ok:
        await loop.run_in_executor(
            None, partial(engine.record_llm_supervision,
                          _typed_text, None, None, None, False)
        )
        message = result.confirmation_prompt or result.error or t("interpreter.cannot_interpret", lang)
        #: **断り文は、研究者が書いたオペランドについて言う。** 止めたのはこちら
        #: なので（足場を前置しなかった）、返っていたのは「必要な引数が不足して
        #: います: ['expr']」——研究者は `expr` ではなく `zvar` と書いている。
        #: 満たされていない条件は「この notebook がその名前で何も持っていない」
        #: ことで、案内する道は**実測で通るものだけ**（値を与える。スカラー宣言は
        #: 通らない——`g をスカラーとする` のあとの `g を簡約して` は欄がスカラーを
        #: 受けないので、ここへ戻ってくる）。⚠ の未定義記号バナーもここに載せる
        #: ——描画側（`notebook_main.js` のエラー枝）は既に読んでいる。
        _resp = {
            "status": "failed",
            "error":  message,
            "registered_expressions": registered,
        }
        #: **`expr` が欠けているときだけ**——`c を電子の消滅演算子とする` のような
        #: 宣言は `name` を取り、その断り（「統計が書かれていない」など）は設計された
        #: ものである。欄の名前で絞らないと、それを上書きしてしまう（実測、全体
        #: スイートで10件）。
        if (_operand_written and "expr" in _still_gen
                and not name_is_usable_as_an_operand(engine.symbol_context, _operand_name)):
            _resp["error"] = t("operand.name_has_no_value", lang, name=_operand_written)
            _resp["undefined_symbols"] = [_operand_name]
        return _resp

    # Record supervision result (confirmed or pending-confirmation)
    await loop.run_in_executor(
        None, partial(engine.record_llm_supervision,
                      _typed_text, result.operation, result.args,
                      result.candidates or None, True)
    )

    # For differentiate: run auto-select var and dep-conflict check BEFORE
    # requires_confirmation so dep UI is shown even when LLM needs confirmation.
    if result.operation == 'differentiate' and not result.args.get('var'):
        expr_id = result.args.get('expr', '')
        src_expr = engine.expressions.get(expr_id)
        if src_expr:
            # Use display_repr (= cas_repr for Julia _nb_ expressions) so that
            # '6 + 6x' style Julia strings yield {'x'} via _extract_variables.
            expr_vars = _extract_variables(src_expr.display_repr)
            if len(expr_vars) == 1:
                result = dataclasses.replace(
                    result,
                    args={**result.args, 'var': next(iter(expr_vars))},
                    requires_confirmation=False,
                )

    if result.operation == 'differentiate':
        diff_var = result.args.get('var', '')
        src_expr = engine.expressions.get(result.args.get('expr', ''))
        if src_expr and diff_var:
            conflicts = await loop.run_in_executor(
                None, partial(engine.check_diff_context, src_expr.canonical, diff_var)
            )
            if conflicts:
                dep_vars = list({v for v, _ in conflicts})
                return {
                    "status": "needs_dep_confirmation",
                    "dep_vars": dep_vars,
                    "operation": result.operation,
                    "args": result.args,
                    "diff_var": diff_var,
                    "registered_expressions": registered,
                }

    if result.requires_confirmation:
        return {
            "status":               "needs_confirmation",
            "operation":            result.operation,
            "args":                 result.args,
            "confirmation_prompt":  result.confirmation_prompt,
            "candidates":           result.candidates,
            "pending_dependencies": result.dependencies,
            "registered_expressions": registered,
        }

    # 4. Auto-execute — intercept assumption operations before CAS routing
    if result.operation == "add_assumption":
        condition = result.args.get("condition", "").strip()
        if not condition:
            return {"status": "failed", "error": t("interpreter.assume_unknown", lang),
                    "registered_expressions": registered}
        # De-reference expression ID: E3 → canonical='Delta > 0', display='\Delta > 0'
        _cond_expr = engine.expressions.get(condition)
        if _cond_expr:
            condition_canonical = _cond_expr.canonical
            condition_display   = _cond_expr.latex or _cond_expr.canonical
        else:
            condition_canonical = condition
            condition_display = assumption_display_latex(condition)
        try:
            _, asm = await loop.run_in_executor(
                None, partial(engine.add_assumption, condition_canonical, condition_display)
            )
        except Exception as e:
            return {"status": "failed", "error": str(e), "registered_expressions": registered}
        return {
            "status":        "assumption_added",
            "operation":     "add_assumption",
            "assumption_id": asm.assumption_id,
            "condition":     condition_display,
            "message":       t("interpreter.assumed", lang, condition=condition_display),
            "registered_expressions": registered,
        }

    if result.operation == "remove_assumption":
        condition = result.args.get("condition", "").strip()
        if not condition:
            return {"status": "failed", "error": t("interpreter.forget_unknown", lang),
                    "registered_expressions": registered}
        try:
            await loop.run_in_executor(
                None, partial(engine.remove_assumption_by_canonical, condition)
            )
        except Exception as e:
            return {"status": "failed", "error": str(e), "registered_expressions": registered}
        return {
            "status":    "assumption_removed",
            "operation": "remove_assumption",
            "condition": condition,
            "message":   t("interpreter.forgotten", lang, condition=condition),
            "registered_expressions": registered,
        }

    if result.operation == "declare_time_dependent":
        name = result.args.get("name", "").strip()
        if not name:
            return {"status": "failed", "error": t("interpreter.declare_var_unknown", lang),
                    "registered_expressions": registered}
        time_parameter = (result.args.get("time_parameter") or "t").strip() or "t"
        try:
            await loop.run_in_executor(
                None, partial(engine.declare_time_dependent, name, time_parameter)
            )
        except Exception as e:
            return {"status": "failed", "error": str(e), "registered_expressions": registered}
        _msg = t("interpreter.declared_time_dep", lang, name=name, time_var=time_parameter)
        engine.record_quantity_update(t("interpreter.time_dep_label", lang), _msg)
        return {
            "status":    "confirmed",
            "operation": "declare_time_dependent",
            "message":   _msg,
            "registered_expressions": registered,
        }

    if result.operation == "set_notation":
        name = result.args.get("name", "").strip()
        fmt  = result.args.get("format", "").strip().lower()
        if not name:
            return {"status": "failed", "error": t("interpreter.notation_var_unknown", lang),
                    "registered_expressions": registered}
        # Accept natural-language format values from the LLM
        if fmt in ("leibniz", "leibniz_notation", "d/dt", "ライプニッツ"):
            fmt = "leibniz"
        elif fmt in ("dot", "dot_notation", "ドット"):
            fmt = "dot"
        if fmt not in ("dot", "leibniz"):
            return {"status": "failed",
                    "error": t("interpreter.notation_unknown_type", lang, fmt=fmt),
                    "registered_expressions": registered}
        try:
            await loop.run_in_executor(
                None, partial(engine.set_notation, name, fmt)
            )
        except Exception as e:
            return {"status": "failed", "error": str(e), "registered_expressions": registered}
        _updated = await loop.run_in_executor(
            None, partial(engine.rerender_notation_for_quantity, name)
        )
        label = t("interpreter.notation_dot" if fmt == "dot" else "interpreter.notation_leibniz", lang)
        _msg = t("interpreter.notation_changed", lang, name=name, label=label)
        engine.record_quantity_update(t("interpreter.notation_switch", lang), _msg)
        return {
            "status":    "confirmed",
            "operation": "set_notation",
            "message":   _msg,
            "updated_exprs": [{"expr_id": eid, "latex": lat} for eid, lat in _updated],
            "registered_expressions": registered,
        }

    if result.operation == "set_latex_name":
        name  = result.args.get("name",  "").strip()
        latex = result.args.get("latex", "").strip()
        if not name:
            return {"status": "failed", "error": t("interpreter.latex_name_var_unknown", lang),
                    "registered_expressions": registered}
        if not latex:
            return {"status": "failed", "error": t("interpreter.latex_name_missing", lang),
                    "registered_expressions": registered}
        try:
            await loop.run_in_executor(
                None, partial(engine.set_latex_name, name, latex)
            )
        except Exception as e:
            return {"status": "failed", "error": str(e), "registered_expressions": registered}
        _msg = t("interpreter.latex_name_set", lang, name=name, latex=latex)
        engine.record_quantity_update(t("interpreter.latex_name_label", lang), _msg)
        return {
            "status":    "confirmed",
            "operation": "set_latex_name",
            "message":   _msg,
            "registered_expressions": registered,
        }

    try:
        step, value = await loop.run_in_executor(
            None, partial(engine.run_operation, result.operation, result.args)
        )
    except Exception as e:
        import traceback, logging
        from notebook.engine import UndefinedSymbolsError
        logging.error("run_operation failed: %s\n%s", e, traceback.format_exc())
        err_msg = str(e) or repr(e)
        resp: dict = {"status": "failed", "error": err_msg, "registered_expressions": registered}
        if isinstance(e, UndefinedSymbolsError):
            resp["undefined_symbols"] = e.symbols
        return resp

    expr_id = step.output_refs[0] if step.output_refs else None
    expr    = engine.expressions.get(expr_id) if expr_id else None

    # Side-effect operations produce no new expression ID.
    if expr_id is None:
        input_expr_id = result.args.get("expr")
        input_expr    = engine.expressions.get(input_expr_id) if input_expr_id else None

        # to_latex / to_plain_text: show the tex1()/string() output but return the
        # SOURCE expression ID so the tile remains a valid calculation target.
        if result.operation in ("to_latex", "to_plain_text"):
            return {
                "status":    "completed",
                "operation": result.operation,
                "step_id":   step.step_id,
                "expr_id":   input_expr_id or "",
                "canonical": input_expr.canonical if input_expr else value,
                "latex":     value,
                "operation_label": step.visible_action,
                "registered_expressions": registered,
            }

        # For void operations (define_scalar_parameter, define_fermion_operator …),
        # the CAS output is always an implementation detail (Julia: "nothing",
        # Maxima: "true", "_res_", "\mathrm{true}" …).  It must never be shown
        # as the display LaTeX.  Ask the backend directly for the symbol's LaTeX
        # form; only fall back to the CAS value when the backend has no opinion
        # (e.g. define_function where no well-typed symbol is produced).
        _void_latex = engine._handler.get_void_op_display_latex(
            result.operation, result.args
        )
        if _void_latex and not registered:
            # Backend has a well-typed symbol to show — use it directly.
            latex_display = _void_latex
        else:
            # No backend-provided symbol (or formula already shown as CAS tile).
            # Use the CAS output only after stripping backend void sentinels.
            latex_display = (input_expr.latex or input_expr.canonical) if input_expr else value
            latex_display = engine._handler.clean_void_result(latex_display)
        _msg = (engine._handler.format_void_op_message(result.operation, result.args, lang)
                or t("interpreter.defined", lang))
        _sym_display = engine._handler.get_void_op_symbol_display(
            result.operation, result.args
        )
        return {
            "status":         "confirmed",
            "operation":      result.operation,
            "step_id":        step.step_id,
            "message":        _msg,
            "latex":          latex_display,
            "symbol_display": _sym_display,
            "registered_expressions": registered,
        }

    _done = {
        "status":    "completed",
        "operation": result.operation,
        "step_id":   step.step_id,
        "expr_id":   expr_id,
        "canonical": expr.canonical if expr else value,
        "latex":     expr.latex     if expr else "",
        "operation_label": step.visible_action,
        "registered_expressions": registered,
    }
    #: **計算できたときも、名前が空なら言う。** `E_{total} を簡約して` は
    #: `E(total)` を `completed` で返す——未定義の記号を自由変数として通す設計の
    #: とおりで値は正しいが、**研究者にはそれが伝わっていなかった**（応答に
    #: `undefined_symbols` のキーごと不在、日英とも）。系は計算してはいて、
    #: 同じ名前を式の中に書けば（`E_{total} + y`）カードに ⚠ が出る——**式として
    #: 登録されるので掛ける先が在る**からで、名前1つのときは登録されないだけ
    #: だった。描画側は既にこの欄を読んでいる（`renderCasEntryHtml`）。
    if _operand_written and not name_is_usable_as_an_operand(
            engine.symbol_context, _operand_name):
        _done["undefined_symbols"] = [_operand_name]
    return _done
