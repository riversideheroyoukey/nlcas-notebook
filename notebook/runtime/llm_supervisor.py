"""LLM Supervisor — interprets free-form user text as a notebook operation.

Three-phase pipeline:
  Phase 1  Rule-based structured parse (alias scan + particle filtering).
           Fast, deterministic, no LLM call.  High confidence when it works.
  Phase 2  LLM call with a dynamic system prompt built from the operation schema.
           Used only when Phase 1 yields no candidate.
  Phase 3  Caller (CLI) confirms with the user before executing any result.
           Phase 1 hits with a single unambiguous match skip confirmation.
           Everything else (multiple candidates, LLM result) requires confirmation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from notebook.runtime.llm_client import LLMClient

from notebook.runtime.backend_handler import normalize_unicode_greek
from notebook.runtime.i18n import t
from notebook.runtime.nl_patterns import nlp_any, nlp_search, nlp_sub
from notebook.runtime.token_field_mapping import (
    EXPR_FIELDS,
    EXPR_ID_RE as _EXPR_ID_RE,
    HAS_CJK_RE as _HAS_CJK_RE,
    POINT_RE as _POINT_RE,
    PURE_ASCII_RE as _PURE_ASCII_RE,
    map_tokens_to_fields as _map_tokens_to_fields,
    refine_alias_candidates as _refine_candidates,
    scan_aliases,
)
from notebook.runtime.operation_validator import OperationValidator

# ---------------------------------------------------------------------------
# The Japanese tokenizer's two INFRASTRUCTURE strips
# ---------------------------------------------------------------------------
# Classified 2026-08-27 while moving this file's NL vocabulary to
# ``i18n/nl_patterns.json`` (NL-GRAMMAR residual ③).  The line drawn there:
# **a spelling that grows with the OPERATION set is a trigger and belongs in the
# i18n file; one that grows with the LANGUAGE's own function words and
# inflections is infrastructure and stays here.**  A new operation never adds a
# particle and never adds a conjugation of する — but it does add a verb
# ("抽出して"), a field word ("次"), a type word ("実数"), an operator name
# ("発散"), which is why those four left and these two did not.  Same class as
# ``interpret_common._VARS_SEP_RE`` (``[,、と\s]``), exempt for the same reason.
#
# Strip one or more leading/trailing single-char particles
#: **複合助詞は1文字の助詞の前に剥がす。** `について` は `に` ＋ `ついて` ではない。
#: 単文字だけを剥がしていたので `について解いて` は `ついて解いて` になり、別名
#: `解いて` がどのトークンにも現れなかった——実測 2026-09-05、コーパスの22行が
#: この形で、動いていたのは**インターセプタが生テキストで先に取るから**だけ。
#: 同じ問い（どのトークンが文法でオペランドでないか）の英語側の答えは
#: `operand_stopwords[en]` で、そこは複合語 `respect` / `w.r.t.` を知っている。
#: ja が `(?!)` なのは「_tokenize が既に剥がすから」——その前提を本当にする。
#:
#: ここが**基盤**であって引き金でないのは、増え方が操作の集合ではなく
#: **日本語の機能語**に従うからである（`verb_tail_strip` は逆で、i18n にある）。
_JP_COMPOUND_PARTICLES = "について|に関して|に対して|に関する|における"
_JP_PARTICLES_STRIP = re.compile(
    rf"^(?:{_JP_COMPOUND_PARTICLES})+|(?:{_JP_COMPOUND_PARTICLES})+$"
    rf"|^[をのでにがはへとも]+|[をのでにがはへとも]+$")

# Common Japanese command verb suffixes to strip from the end of tokens
# ("して" = "do it", "する" = "do", "せよ/しろ" = imperative)
_JP_COMMAND_SUFFIX = re.compile(r"(?:して|する|しろ|せよ|しなさい)$")

# The two strips that carry VOCABULARY live in i18n/nl_patterns.json:
#   verb_tail_strip      — the computation/extraction verbs, so the noun-form
#                          alias ("実部", "微分") is exposed.  Since 2026-09-05 it
#                          is built from `{{compute_verb_ja}}`, the ONE list of
#                          "compute this" verbs (the question had nine ja answers).
#                          The list is written longest-first for readability; the
#                          claim that used to stand here — that 取り出して must
#                          precede its own 出して suffix — is FALSE when measured:
#                          the `$` anchor plus leftmost-match and backtracking give
#                          the same result either way.
#   ordinal_prefix_strip — "次で/次まで/次の/次に", the `order` field's word
#                          ("1 次までテイラー展開して" → "テイラー展開").

# Strip trailing English sentence punctuation so that "DIFF." matches alias "diff"
_EN_TRAILING_PUNCT = re.compile(r"[.!?:;]+$")

# Script-boundary detection: insert space between ASCII↔CJK characters
_ASCII_TO_CJK = re.compile(r"([A-Za-z0-9])([^\x00-\x7F])")
_CJK_TO_ASCII = re.compile(r"([^\x00-\x7F])([A-Za-z0-9])")

# Japanese "X で (偏)?微分" — X is the differentiation variable
# (``diff_var_particle`` in i18n/nl_patterns.json).
# Used to pre-extract `var` before positional token mapping, which would
# otherwise assign the first ASCII token (often the expression LHS, not the variable).

# Domain-KIND / SIGN keywords for define_scalar_parameter: which set of numbers
# (real / complex / integer / natural), and the sign, extracted INDEPENDENTLY so
# a compound declaration like "正の整数" keeps both qualifiers instead of the
# kind pattern silently winning and dropping the sign (see
# notebook/models/quantity.py Quantity.sign).
#
# The SPELLINGS live in i18n/nl_patterns.json; the ORDER lives here, because the
# order is structure and not language content — "非正" must be tried before
# "正の", or 「非正の実数」 reads as *positive*.
_DOMAIN_KIND_KEYS: list[tuple[str, str]] = [
    ("domain_kind_natural", "natural"),
    ("domain_kind_integer", "integer"),
    ("domain_kind_real", "real"),
    ("domain_kind_complex", "complex"),
]
_DOMAIN_SIGN_KEYS: list[tuple[str, str]] = [
    ("domain_sign_nonneg", "nonneg"),
    ("domain_sign_nonpos", "nonpos"),
    ("domain_sign_positive", "positive"),
    ("domain_sign_negative", "negative"),
]


def _extract_domain_kind(text: str) -> str | None:
    """Return the domain-kind keyword found in *text*, or None if not present.

    Asked in BOTH languages (``nlp_any``), never narrowed to the session's:
    this is "which set of numbers does this sentence name?", and the caller
    reaching it has no reliable language — the same rule, and the same reason,
    as ``backend_handler._nl_any``.
    """
    for key, kind in _DOMAIN_KIND_KEYS:
        if nlp_any(key, text):
            return kind
    return None


def _extract_domain_sign(text: str) -> str | None:
    """Return the sign keyword found in *text*, or None if not present."""
    for key, sign in _DOMAIN_SIGN_KEYS:
        if nlp_any(key, text):
            return sign
    return None


def _norm_kind_written(text: str) -> "str | None":
    """研究者が書いたノルムの**種類**、無ければ None（`norm_kind` に綴りと理由）。

    **群の名前が正準の種類名そのもの**なので、綴り→種類の対応表は要らない。
    """
    from notebook.runtime.nl_patterns import nlp_search
    m = nlp_search("norm_kind", text or "")
    if m is None:
        return None
    return next((k for k, v in m.groupdict().items() if v), None)


def _enrich_args_from_text(result: "InterpretResult", text: str) -> "InterpretResult":
    """別名走査が運ばない**修飾語**を、研究者が書いた文から補う。

    呼び出し口は7つあるので、問いごとに関数を増やすと**7箇所を揃える形**になる
    （SIGN-DIALOG-ASSUME がまさにその形で1つ漏れた）。ここが唯一の入口で、
    中で操作ごとに分ける。いまの中身は3つ——宣言の domain/sign、ノルムの種類、
    微分の偏／全。
    """
    result = _enrich_domain(result, text)
    result = _enrich_norm_kind(result, text)
    return _enrich_derivative_kind(result, text)


def _enrich_derivative_kind(result: "InterpretResult", text: str) -> "InterpretResult":
    r"""`differentiate` の**偏／全**を文から補う（`diff_rule` と同じ marker）。

    別名走査の経路は種別を運んでいなかった——`偏微分` / `全微分` を別名に足して
    素の宣言名でも届くようにしたら見えた（実測 2026-09-12、両方ともフラグ無し）。
    E 番号のある綴りは `diff_rule` が先に答えて種別を立てるので、**経路によって
    別の答え**になっていた（LIMIT-ONE-SIDED の `direction` と同じ形）。
    いま値が割れないのは偶然である——理由と実測は CLAUDE.md。
    """
    import dataclasses
    from notebook.runtime.nl_patterns import nlp_search
    if not (result.ok and result.operation == "differentiate"):
        return result
    args = dict(result.args or {})
    if "_is_total_derivative" in args or "_is_partial_derivative" in args:
        return result
    if nlp_search("total_derivative_marker", text or ""):
        args["_is_total_derivative"] = True
    elif nlp_search("partial_derivative_marker", text or ""):
        args["_is_partial_derivative"] = True
    else:
        return result
    return dataclasses.replace(result, args=args)


def _enrich_norm_kind(result: "InterpretResult", text: str) -> "InterpretResult":
    """`norm` の `kind` を文から補う（候補の args にも同じだけ足す）。"""
    import dataclasses
    if not (result.ok and (result.operation == "norm"
                           or any(c.get("operation") == "norm"
                                  for c in (result.candidates or [])))):
        return result
    kind = _norm_kind_written(text)
    if not kind:
        return result

    def _supplement(a: "dict | None") -> "dict | None":
        out = dict(a or {})
        if "kind" in out:
            return None
        out["kind"] = kind
        return out

    new_args = _supplement(result.args) if result.operation == "norm" else None
    new_candidates = None
    for i, cand in enumerate(result.candidates or []):
        if cand.get("operation") != "norm":
            continue
        enriched = _supplement(cand.get("args"))
        if enriched is None:
            continue
        if new_candidates is None:
            new_candidates = [dict(c) for c in result.candidates]
        new_candidates[i]["args"] = enriched
    if new_args is None and new_candidates is None:
        return result
    return dataclasses.replace(
        result,
        args=new_args if new_args is not None else result.args,
        candidates=(new_candidates if new_candidates is not None
                    else result.candidates))


def _enrich_domain(result: "InterpretResult", text: str) -> "InterpretResult":
    """Supplement domain/sign args for define_scalar_parameter results if not already set.

    **Every alternative the caller may run instead must be supplemented too.**
    When the result carries ``candidates`` (a disambiguation menu), picking one
    calls ``run_operation`` with THAT candidate's args, bypassing ``result.args``
    entirely — so enriching only the default silently drops the qualifiers on
    every dialog path.  Measured 2026-07-31 (open_items SIGN-DIALOG-ASSUME):
    「omega_D を正の実数とする」 fires the subscript-role dialog, and the chosen
    candidate registered omega_D with NEITHER sign NOR domain — the "正の" and
    the "実数" both vanished with no error.  The sign is not decoration: it is
    forwarded to Maxima's ``assume()``, and without it ``integrate`` stops on an
    unanswered sign question, which times out and takes the rest of the session
    with it.
    """
    import dataclasses
    if not (result.ok and result.operation == "define_scalar_parameter"):
        return result

    kind = _extract_domain_kind(text)
    sign = _extract_domain_sign(text)
    if not (kind or sign):
        return result

    def _supplement(a: dict) -> "dict | None":
        """Return an enriched copy of *a*, or None when nothing changed."""
        out = dict(a or {})
        changed = False
        if kind and "domain" not in out:
            out["domain"] = kind
            changed = True
        if sign and "sign" not in out:
            out["sign"] = sign
            changed = True
        return out if changed else None

    new_args = _supplement(result.args)
    new_candidates = None
    for i, cand in enumerate(result.candidates or []):
        if cand.get("operation") != "define_scalar_parameter":
            continue
        enriched = _supplement(cand.get("args"))
        if enriched is None:
            continue
        if new_candidates is None:
            new_candidates = [dict(c) for c in result.candidates]
        new_candidates[i]["args"] = enriched

    if new_args is None and new_candidates is None:
        return result
    return dataclasses.replace(
        result,
        args=new_args if new_args is not None else result.args,
        candidates=(new_candidates if new_candidates is not None
                    else result.candidates),
    )

# 括弧記法・トークン分割・助詞スパンの規則は notebook/runtime/operand_boundary.py。
from notebook.runtime.operand_boundary import (  # noqa: E402
    _bracket_operands, _split_outside_brackets, declared_object_names)




# ---------------------------------------------------------------------------
# Phase 1.5 — known ambiguous terms → forced candidate list
#
# These terms have no single unambiguous mapping in the alias table, so
# Phase 1 passes them to the LLM which tends to pick one answer
# deterministically.  Map them here to an explicit candidate set so the
# user always sees a choice.
# ---------------------------------------------------------------------------
# The SPELLINGS are ``ambiguous_tidy`` / ``ambiguous_transform`` /
# ``ambiguous_clean`` / ``ambiguous_collect`` in i18n/nl_patterns.json; the
# candidate lists stay here, because they are operations, not language.
_AMBIGUOUS_TERM_KEYS: list[tuple[str, list[str]]] = [
    # "整理" — could tidy, simplify, collect, factor, rational-simplify
    ("ambiguous_tidy", ["simplify", "rational_simplify", "expand", "factor", "collect"]),
    # "変形" — could expand, factor, simplify
    ("ambiguous_transform", ["expand", "factor", "simplify"]),
    # "きれい" — could simplify, rational_simplify, trig_simplify, radical_simplify
    ("ambiguous_clean", ["simplify", "rational_simplify", "trig_simplify", "radical_simplify"]),
    # "まとめ" — collect or factor
    ("ambiguous_collect", ["collect", "factor", "simplify"]),
]


@dataclass


class InterpretResult:
    """Outcome of LLMSupervisor.interpret()."""

    ok: bool
    operation: str | None = None
    args: dict = field(default_factory=dict)

    # 0.0–1.0; 1.0 = rule-based single unambiguous match
    confidence: float = 0.0

    # "rule" | "llm"
    source: str = "rule"

    # True  → caller must show confirmation before executing
    # False → caller may execute directly (high-confidence rule hit)
    requires_confirmation: bool = True

    # Human-readable text to show in confirmation prompt
    confirmation_prompt: str = ""

    # Other plausible candidates (for disambiguation menu)
    candidates: list[dict] = field(default_factory=list)

    # Context updates extracted from LLM response
    symbol_updates: list[dict] = field(default_factory=list)
    dependencies: list[dict] = field(default_factory=list)

    error: str | None = None


# ∇ の綴りは1つの表（`notebook/runtime/vector_op_spelling.py`）。単項も入れ子も
# そこを読む——入れ子の鍵が LaTeX を知らず、`\nabla \cdot (\nabla \times \vec{A})`
# が登録だけされて計算されなかった（NESTED-VEC-OP-LATEX）。ここは結果を包むだけ。
from notebook.runtime.vector_op_spelling import (  # noqa: E402
    read_nested as _read_nested_vec_op, read_single as _read_single_vec_op)


def _try_single_vec_op(text: str) -> InterpretResult | None:
    """One ∇ operation on one field (`\\nabla \\times A`, `∇·E`, `div E`, …)."""
    found = _read_single_vec_op(text)
    if found is None:
        return None
    return InterpretResult(ok=True, operation=found[0], args={"expr": found[1]},
                           confidence=1.0, source="rule", requires_confirmation=False)


def _try_nested_vec_op(text: str) -> InterpretResult | None:
    """Phase 0.7: a ∇ operation applied to another (`\\nabla \\cdot (\\nabla \\times A)`,
    `div(curl(A))`, `A の回転の発散`, `the divergence of the curl of A`)."""
    found = _read_nested_vec_op(text)
    if found is None:
        return None
    return InterpretResult(ok=True, operation=found[0], args={"expr": found[1]},
                           confidence=1.0, source="rule", requires_confirmation=False)


# ---------------------------------------------------------------------------

class RuleBasedIntentParser:
    """Rule-based intent parsing (Phases 0–1.5): map natural language / notation to
    a common operation WITHOUT the LLM.  This is the bulk of intent resolution;
    LLMSupervisor consults the LLM only when parse() returns no definitive result.
    To add a rule, extend the phase methods here (or the interpret_pipeline _try_*
    helpers for E-id / async-aware rules).  AUD-5 / C-5: split out of LLMSupervisor
    to fix the name-vs-responsibility mismatch."""

    def __init__(self, validator, backend: str) -> None:
        # `backend` is REQUIRED: it selects which operations are candidates, so a
        # forgotten argument would offer Maxima-only operations in a Julia
        # session (open_items AUDIT-SHARED-CAS-LEAK).
        self._v = validator
        self._backend = backend

    def parse(self, user_text, known_expr_ids, lang: str = "ja", handler=None,
              declared_objects=frozenset()):
        """Run the rule phases on already-normalized *user_text*.  Returns
        (final, phase1): *final* is a definitive InterpretResult to return
        immediately (or None to fall through to the LLM); *phase1* is the
        structured-parse result (possibly partial) the caller uses to decide the
        LLM fallback."""
        bracket = self._try_bracket_parse(user_text)
        if bracket is not None:
            return bracket, None
        if handler is not None:
            pre = handler.extra_pre_phase1(user_text, lang)
            if pre is not None:
                return _enrich_args_from_text(pre, user_text), None
        nested_vec = _try_nested_vec_op(user_text)
        if nested_vec is not None:
            return _enrich_args_from_text(nested_vec, user_text), None
        phrase = self._try_binary_operation_phrase(user_text, lang)
        if phrase is not None:
            return _enrich_args_from_text(phrase, user_text), None
        phase1 = self._try_structured_parse(user_text, known_expr_ids, lang,
                                            declared_objects)
        if phase1 is not None and phase1.ok:
            return _enrich_args_from_text(phase1, user_text), phase1
        phase15 = self._try_ambiguous_parse(user_text, known_expr_ids, lang)
        if phase15 is not None:
            return _enrich_args_from_text(phase15, user_text), phase1
        return None, phase1

    # ------------------------------------------------------------------
    # Phase 0 — bracket notation  [X, Y] / {X, Y}
    # ------------------------------------------------------------------

    def _try_bracket_parse(self, text: str) -> "InterpretResult | None":
        """Detect [X, Y] → commutator (or, classically, Poisson bracket) and
        {X, Y} → anticommutator.

        Runs before Phase 1 so physicists can use standard bracket notation
        without needing natural-language wrappers.  Only activates when the
        operation is supported by the current backend — ``commutator``
        (quantum, julia_full-only) and ``poisson_bracket`` (classical, its
        exact analogue, maxima-only) share the same `[X, Y]` syntax and the
        same ``a``/``b`` argument names, and are mutually exclusive by
        backend, so trying both candidates for the one pattern is unambiguous.
        """
        for opener, operation in (
            ("[", "commutator"),
            ("[", "poisson_bracket"),
            ("{", "anticommutator"),
        ):
            spec = self._v.get_spec(operation)
            if spec is None:
                continue
            if self._backend not in spec.get("supported_backends", []):
                continue
            # `\{A,B\} = δ` は関係式であって操作の要求ではない——取ると
            # `= δ` を黙って捨てて反交換子を計算することになる。
            from notebook.runtime.operand_boundary import is_bracket_operation_text
            m = (_bracket_operands(text, opener)
                 if is_bracket_operation_text(text) else None)
            # A Maxima matrix literal ("[2, 1; 1, 2]", row-separated by ";")
            # also matches _COMMUTATOR_RE's [X, Y] shape (";" falls inside the
            # second, unrestricted capture group) — exclude it explicitly so
            # a bare matrix literal is never misrouted into poisson_bracket
            # (SUBST-... class of bug found via test_matrix_literal.py /
            # test_lotka_volterra_scenario.py regressions, 2026-07-09: a
            # genuine two-argument bracket never contains ";").
            if m and operation == "poisson_bracket" and ";" in m[2]:
                continue
            if m:
                return InterpretResult(
                    ok=True,
                    operation=operation,
                    args={"a": m[0].strip(), "b": m[1].strip()},
                    confidence=1.0,
                    source="rule",
                    requires_confirmation=False,
                )
        return None

    # ------------------------------------------------------------------
    def _try_binary_operation_phrase(self, text: str, lang: str) -> "InterpretResult | None":
        """`A と B の交換子` / `commutator of A and B` を読む。

        規則そのものは `notebook/runtime/operand_boundary.py`（オペランドの境界の
        族）にある。ここが答えるのはバックエンドの可否だけ。
        """
        from notebook.runtime.operand_boundary import read_binary_operation_phrase

        def _supports(operation: str) -> bool:
            spec = self._v.get_spec(operation)
            return (spec is not None
                    and self._backend in spec.get("supported_backends", []))

        found = read_binary_operation_phrase(text, lang, _supports)
        if found is None:
            return None
        operation, a, b = found
        return InterpretResult(ok=True, operation=operation, args={"a": a, "b": b},
                               confidence=1.0, source="rule",
                               requires_confirmation=False)
    # ------------------------------------------------------------------
    # Phase 1.5 — known ambiguous terms
    # ------------------------------------------------------------------

    def _try_ambiguous_parse(
        self,
        text: str,
        known_expr_ids: set[str],
        lang: str = "ja",
    ) -> InterpretResult | None:
        """If the text contains a known ambiguous term, return a candidate list.

        Extracts the first expression ID mentioned in the text as the `expr`
        argument.  Returns None when no ambiguous term is found.
        """
        matched_ops: list[str] | None = None
        for key, ops in _AMBIGUOUS_TERM_KEYS:
            # Both languages, not ``lang`` — these spellings were applied
            # regardless of the session language before the move, and narrowing
            # them now would silently drop a Japanese instruction typed in an
            # English session (backend_handler._nl_any records the measurement).
            if nlp_any(key, text):
                matched_ops = ops
                break
        if matched_ops is None:
            return None

        # Extract the first expression ID from the text
        expr_id = next(
            (tok for tok in re.findall(r"(?<![A-Za-z0-9_])E\d+(?![A-Za-z0-9_])", text) if tok in known_expr_ids),
            None,
        )

        # Keep only operations that are valid for the current backend
        valid_ops = [
            op for op in matched_ops
            if self._v.get_spec(op) is not None
            and self._backend in (self._v.get_spec(op) or {}).get("supported_backends", [])
        ]
        if not valid_ops:
            return None

        candidates = []
        for op in valid_ops:
            spec = self._v.get_spec(op)
            args: dict = {}
            if expr_id and "expr" in spec["required_fields"]:
                args["expr"] = expr_id
            candidates.append({
                "operation": op,
                "args": args,
                "display_ja": spec["display"]["ja"],
                "display_en": spec["display"]["en"],
            })

        prompt_lines = [t("interpreter.ambiguous_ops", lang)]
        for i, c in enumerate(candidates, 1):
            args_str = ", ".join(f"{k}={v}" for k, v in c["args"].items())
            disp = c.get(f"display_{lang}") or c.get("display_en", "")
            prompt_lines.append(f"  {i}. {disp}" + (f": {args_str}" if args_str else ""))

        return InterpretResult(
            ok=True,
            operation=candidates[0]["operation"],
            args=candidates[0]["args"],
            confidence=0.5,
            source="rule",
            requires_confirmation=True,
            confirmation_prompt="\n".join(prompt_lines),
            candidates=candidates,
        )

    # ------------------------------------------------------------------
    # Expression injection guard
    # ------------------------------------------------------------------

    _EXPR_FIELDS = EXPR_FIELDS

    def needs_expression_injection(self, text: str) -> bool:
        """Return True if the supervisor text needs an expression ID injected.

        Returns False (skip injection) when:
        - A Phase 1 alias is found and the operation has no expr-class required fields, OR
        - The remaining tokens already provide enough expr-type tokens to fill all
          expr-class required fields (user explicitly supplied operands like 'f(1)').
        """
        tokens = _tokenize(text)
        for start, end, canonical in scan_aliases(
                tokens, self._v.resolve_alias, self._v.max_alias_words):
            spec = self._v.get_spec(canonical)
            if spec is None:
                continue
            if self._backend not in spec.get("supported_backends", []):
                continue
            # Found a matching operation — check whether the remaining tokens
            # already supply enough operands.
            remaining = [t for j, t in enumerate(tokens)
                         if not (start <= j < end)]
            expr_tokens_count = sum(
                1 for t in remaining
                if _EXPR_ID_RE.match(t) or (not _HAS_CJK_RE.search(t) and not _PURE_ASCII_RE.match(t))
            )
            expr_fields_needed = sum(
                1 for f in spec.get("required_fields", []) if f in self._EXPR_FIELDS
            )
            if expr_fields_needed == 0 or expr_tokens_count >= expr_fields_needed:
                return False
        return True

    # ------------------------------------------------------------------
    # Phase 1 — rule-based structured parse
    # ------------------------------------------------------------------

    def _try_structured_parse(
        self,
        text: str,
        known_expr_ids: set[str],
        lang: str = "ja",
        declared_objects=frozenset(),
    ) -> InterpretResult | None:
        """Scan tokens for alias matches, map remaining tokens to required_fields.

        Returns None when no candidate found (→ caller tries Phase 2).
        Returns InterpretResult with ok=True on success.
        Returns InterpretResult with ok=False when candidates exist but args
        cannot be mapped (caller should ask LLM or show error).
        """
        tokens = _tokenize(text)

        # Find all matching operations (alias → canonical)
        candidates: list[str] = []
        matched_span: dict[str, tuple[int, int]] = {}  # canonical → token span
        for start, end, canonical in scan_aliases(
                tokens, self._v.resolve_alias, self._v.max_alias_words):
            spec = self._v.get_spec(canonical)
            if spec is None:
                continue
            if self._backend not in spec.get("supported_backends", []):
                continue
            if canonical not in candidates:
                candidates.append(canonical)
                matched_span[canonical] = (start, end)

        if not candidates:
            return None  # Phase 2

        # Which member of an alias family the researcher meant — decided by a
        # field they named, or ASKED when only a sibling fits (see the helper;
        # BP-6, ALIAS-COLLISION-UNREACHABLE).
        candidates = _refine_candidates(
            tokens, candidates, matched_span, self._v, self._backend,
            known_expr_ids, lang)

        # Build args for each candidate
        candidate_dicts: list[dict] = []
        for canonical in candidates:
            spec = self._v.get_spec(canonical)
            op_start, op_end = matched_span[canonical]
            remaining = [
                t for i, t in enumerate(tokens)
                if not (op_start <= i < op_end)
            ]
            # Pre-extract 'var' from Japanese "X で (偏)?微分" pattern.
            # Positional mapping assigns the first ASCII token to 'var', but
            # in "L を q で微分して" that first token is the expression LHS ("L"),
            # not the differentiation variable ("q").  The Japanese particle
            # "X で 微分" unambiguously marks X as the variable — use it directly.
            pre_args: dict[str, str] = {}
            if "var" in spec["required_fields"]:
                m = nlp_search("diff_var_particle", text)
                if m:
                    pre_args["var"] = m.group(1)
            remaining_for_map = [t for t in remaining if t not in pre_args.values()]
            reduced_fields = [f for f in spec["required_fields"] if f not in pre_args]
            args = {**pre_args, **_map_tokens_to_fields(
                remaining_for_map, reduced_fields, known_expr_ids, lang,
                optional_fields=spec.get("optional_fields", []), text=text,
                declared_objects=declared_objects)}
            # Extract domain/sign for define_scalar_parameter (optional fields).
            if canonical == "define_scalar_parameter":
                _dom = _extract_domain_kind(text)
                if _dom:
                    args["domain"] = _dom
                _sign = _extract_domain_sign(text)
                if _sign:
                    args["sign"] = _sign
            candidate_dicts.append({
                "operation": canonical,
                "args": args,
                "display_ja": spec["display"]["ja"],
                "display_en": spec["display"]["en"],
                "required_fields": spec["required_fields"],
                "args_complete": all(f in args for f in spec["required_fields"]),
            })

        # ---- Single unambiguous candidate with all fields ----
        complete = [c for c in candidate_dicts if c["args_complete"]]
        if len(complete) == 1 and len(candidates) == 1:
            c = complete[0]
            return InterpretResult(
                ok=True,
                operation=c["operation"],
                args=c["args"],
                confidence=1.0,
                source="rule",
                requires_confirmation=False,
            )

        # ---- Multiple candidates or incomplete args → confirm ----
        if complete:
            prompt_lines = [t("interpreter.interpreted_as", lang)]
            for i, c in enumerate(complete, 1):
                args_str = ", ".join(f"{k}={v}" for k, v in c["args"].items())
                disp = c.get(f"display_{lang}") or c.get("display_en", "")
                prompt_lines.append(f"  {i}. {disp}: {args_str}")
            return InterpretResult(
                ok=True,
                operation=complete[0]["operation"],
                args=complete[0]["args"],
                confidence=0.8,
                source="rule",
                requires_confirmation=True,
                confirmation_prompt="\n".join(prompt_lines),
                candidates=[
                    {"operation": c["operation"], "args": c["args"],
                     "display_ja": c["display_ja"], "display_en": c["display_en"]}
                    for c in complete
                ],
            )

        # Candidates found but args missing — show what we know
        c = candidate_dicts[0]
        missing = [f for f in c["required_fields"] if f not in c["args"]]
        disp = c.get(f"display_{lang}") or c.get("display_en", c["operation"])
        return InterpretResult(
            ok=False,
            operation=c["operation"],
            args=c["args"],
            confidence=0.5,
            source="rule",
            requires_confirmation=True,
            confirmation_prompt=t("interpreter.missing_args", lang,
                                  display=disp, missing=missing),
            error=f"Missing required fields: {missing}",
        )


class LLMSupervisor:
    """Interprets a free-form text command as a notebook operation.

    Args:
        validator: Shared OperationValidator instance.
        llm_client: Optional LLMClient.  When None, Phase 2 is skipped and
                    an error result is returned for Phase-1 failures.
        primary_backend: Backend to restrict candidate operations to.  REQUIRED
                    and keyword-only: it used to default to ``"maxima"``, which
                    meant a caller that forgot it silently got Maxima's operation
                    set — in a Julia session, candidate operations the backend
                    cannot run (open_items AUDIT-SHARED-CAS-LEAK; every one of
                    the 72 existing call sites already passed it, so nothing
                    depended on the default).
    """

    # Maximum number of (user, assistant) pairs kept in the conversation history.
    # Older pairs are dropped so the prompt does not grow unboundedly.
    _HISTORY_MAX: int = 8

    def __init__(
        self,
        validator: OperationValidator | None = None,
        llm_client: "LLMClient | None" = None,
        *,
        primary_backend: str,
        domains: list[str] | None = None,
    ) -> None:
        self._v = validator or OperationValidator()
        self._llm = llm_client
        self._backend = primary_backend
        self._domains: list[str] = domains or []
        # Rule-based intent parser (Phases 0–1.5). The LLM is only a fallback when
        # this returns no definitive result (see interpret()).
        self._rules = RuleBasedIntentParser(self._v, self._backend)
        # Conversation history for multi-turn LLM context.
        # Each entry is (user_text, assistant_summary) where assistant_summary
        # is a compact representation of what the LLM interpreted.
        self._history: list[tuple[str, str]] = []
        # Notation profile loader: resolves domain-specific notation rules and
        # injects hints into the system prompt so the LLM can resolve ambiguous
        # notation without asking for user confirmation.
        try:
            from notebook.runtime.notation_profile_loader import NotationProfileLoader
            self._notation_profile = NotationProfileLoader()
        except Exception:
            self._notation_profile = None

    def clear_history(self) -> None:
        """Reset the conversation history (e.g. when starting a new topic)."""
        self._history.clear()

    def needs_expression_injection(self, text: str) -> bool:
        """Delegate to the rule parser (kept on LLMSupervisor for callers that
        hold a supervisor instance)."""
        return self._rules.needs_expression_injection(text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpret(
        self,
        user_text: str,
        known_expr_ids: set[str] | None = None,
        symbol_context: "SymbolContext | None" = None,
        lang: str = "ja",
        handler: "object | None" = None,
    ) -> InterpretResult:
        """Parse *user_text* into an operation + args.

        Returns an InterpretResult; caller must check .ok and
        .requires_confirmation before executing.

        *handler* is an optional BackendHandler instance.  When provided, its
        ``extra_pre_phase1`` hook is called between Phase 0 and Phase 1 so that
        backend-specific patterns (e.g. Julia particle operators) are resolved
        without touching this shared module.
        """
        # Phase -1 — normalize Unicode Greek letters to TeX commands
        user_text = normalize_unicode_greek(user_text)

        # Phases 0–1.5 — rule-based intent parsing (no LLM). `final` short-circuits
        # immediately; `phase1` is the (possibly partial) structured-parse result
        # used below to decide the LLM fallback.
        #: **宣言された対象の名前は、それ自体が式である**（OPERAND-BOUNDARY 残①）。
        #: 判別子は足場を止めるときと同じ（`declared_object_names`）——スカラーは
        #: 変数として文中に立つので数えない。
        final, phase1 = self._rules.parse(
            user_text, known_expr_ids or set(), lang, handler,
            declared_objects=declared_object_names(symbol_context))
        if final is not None:
            return final

        # Phase 2 — LLM fallback (only when no rule produced a definitive result)
        if self._llm is not None:
            # If Phase 1 already identified the operation (just missing args),
            # return that result directly — LLM can't help with missing args
            # and would cause a hang if the server is slow.
            if phase1 is not None and phase1.operation:
                return _enrich_args_from_text(phase1, user_text)
            llm_result = self._call_llm(user_text, known_expr_ids or set(), symbol_context, lang)
            return _enrich_args_from_text(llm_result, user_text)

        # No LLM — return Phase 1 error if we found partial candidates
        if phase1 is not None:
            return phase1
        return InterpretResult(
            ok=False,
            error=f"Could not interpret: '{user_text}'. No LLM client configured.",
        )

    # ------------------------------------------------------------------
    # Phase 2 — LLM call
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        user_text: str,
        known_expr_ids: set[str],
        symbol_context: "SymbolContext | None" = None,
        lang: str = "ja",
    ) -> InterpretResult:
        system = self._build_system_prompt(known_expr_ids, symbol_context)
        temperature = getattr(self._llm, "default_temperature", 0.0)
        max_tokens = getattr(self._llm, "default_max_tokens", None)
        num_ctx = getattr(self._llm, "default_num_ctx", None)

        # Build prior_messages from conversation history so the LLM can use
        # context to correct typos, resolve pronouns, and express low confidence.
        # Identical prefix → Ollama reuses KV cache (no recomputation).
        prior: list[dict] = []
        for u, a in self._history[-self._HISTORY_MAX:]:
            prior.append({"role": "user",      "content": u})
            prior.append({"role": "assistant", "content": a})

        resp = self._llm.chat(
            system=system,
            user=user_text,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            prior_messages=prior or None,
        )

        if not resp.ok:
            return InterpretResult(ok=False, error=f"LLM error: {resp.error}")

        result = self._parse_llm_response(resp.text, known_expr_ids, lang)

        # Record in history so subsequent calls benefit from context.
        # Store a compact summary of the interpreted operation as the assistant
        # turn so the LLM can reference what it said in prior turns.
        if result.ok and result.operation:
            args_str = ", ".join(f"{k}={v}" for k, v in (result.args or {}).items())
            summary = f"{result.operation}({args_str})" if args_str else result.operation
            self._history.append((user_text, summary))
            # Trim to window size
            if len(self._history) > self._HISTORY_MAX:
                self._history = self._history[-self._HISTORY_MAX:]

        return result

    def _build_system_prompt(self, known_expr_ids: set[str], symbol_context: "SymbolContext | None" = None) -> str:
        import logging
        from notebook.domain_paths import DOMAINS_ROOT
        _domains_root = DOMAINS_ROOT
        prompt_path = Path(__file__).parent.parent.parent / "models" / "default" / "prompts" / "supervisor_system.txt"
        if prompt_path.exists():
            base_prompt = prompt_path.read_text(encoding="utf-8").strip()
        else:
            base_prompt = (
                "You are a command parser for a symbolic mathematics notebook.\n"
                "Parse the user's natural-language command into structured JSON."
            )

        ops = self._v.operations_for_backend(self._backend)
        lines = [
            base_prompt,
            "",
        ]

        # Inject domain-specific hints
        for domain in self._domains:
            hint_path = _domains_root / domain / "supervisor_hint.txt"
            if hint_path.exists():
                lines.append(hint_path.read_text(encoding="utf-8").strip())
                lines.append("")
            else:
                logging.warning("Domain hint not found: %s", hint_path)

        # Inject notation profile: tells the LLM which notation tokens are
        # unambiguous in this domain so it does not set requires_confirmation
        # unnecessarily.
        if self._notation_profile and self._domains:
            hint = self._notation_profile.notation_hint_text(self._domains)
            if hint:
                lines.append(hint)
                lines.append("")

        lines += [
            "Rules:",
            "- Do NOT compute results or generate CAS code.",
            "- Use only the operations listed below.",
            "- Expression arguments must be taken from the available expression IDs.",
            "- Return JSON only, no markdown, no explanation.",
            "",
        ]

        if symbol_context and symbol_context.registry:
            lines.append("Current Symbol Context:")
            for name, prop in symbol_context.registry.items():
                kind = getattr(prop.quantity, "kind", "unknown")
                vars_ = getattr(prop.quantity, "natural_vars", [])
                stat = getattr(prop.statistics, "kind", "unknown")
                lines.append(f"  {name}: {kind} (vars: {vars_}, stat: {stat})")
            lines.append("")

        if known_expr_ids:
            lines.append(f"Available expression IDs: {sorted(known_expr_ids)}")
            lines.append("")

        lines.append("Available operations:")
        for op in ops:
            spec = self._v.get_spec(op)
            en_aliases = spec.get("aliases", {}).get("en", [])
            ja_aliases = spec.get("aliases", {}).get("ja", [])
            req = spec["required_fields"]
            lines.append(
                f'  {op}: required_fields={req}'
                f'  (en: {en_aliases[:3]}, ja: {ja_aliases[:3]})'
            )

        lines += [
            "",
            "Output schema on success:",
            '{"ok": true, "operation": "<name>", "args": {<field>: <value>}, '
            '"confidence": 0.0-1.0, "candidates": [], "reasoning": "...", '
            '"symbol_updates": [{"name": "L", "kind": "function", "dependencies": ["q", "dot_q", "t"]}], '
            '"dependencies": [{"dependent": "L", "independent": ["q", "dot_q", "t"]}]}',
            "",
            "On ambiguity (multiple plausible interpretations):",
            '{"ok": true, "operation": "<best>", "args": {...}, "confidence": 0.5, '
            '"candidates": [{"operation": "...", "args": {...}}, ...], '
            '"reasoning": "...", "symbol_updates": [], "dependencies": []}',
            "",
            "On failure:",
            '{"ok": false, "error": "..."}',
        ]
        prompt = "\n".join(lines)

        num_ctx = getattr(self._llm, "default_num_ctx", None)
        if num_ctx:
            estimated_tokens = len(prompt) // 4
            if estimated_tokens > num_ctx * 0.7:
                logging.warning(
                    "Supervisor system prompt estimated at ~%d tokens, exceeds 70%% of num_ctx=%d. "
                    "Consider shortening domain hints.",
                    estimated_tokens, num_ctx,
                )
        return prompt

    def _parse_llm_response(
        self,
        text: str,
        known_expr_ids: set[str],
        lang: str = "ja",
    ) -> InterpretResult:
        # LLMClient.chat() already strips <think> blocks and code fences.
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            if not text:
                return InterpretResult(ok=False, error=t("interpreter.llm_no_response", lang))
            return InterpretResult(ok=False, error=f"LLM returned invalid JSON: {e}\n{text[:200]}")

        if not data.get("ok"):
            return InterpretResult(ok=False, error=data.get("error", "LLM returned ok=false"))

        raw_op = data.get("operation", "")
        canonical = self._v.resolve_alias(raw_op) or raw_op
        spec = self._v.get_spec(canonical)
        if spec is None:
            return InterpretResult(ok=False, error=f"LLM returned unknown operation: '{raw_op}'")

        args = data.get("args", {})
        confidence = float(data.get("confidence", 0.5))
        raw_candidates = data.get("candidates", [])
        symbol_updates = data.get("symbol_updates", [])
        dependencies = data.get("dependencies", [])

        # Build human-readable confirmation prompt (no CAS syntax)
        args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        prompt = t("interpreter.llm_confirm", lang,
                   display_ja=spec["display"].get("ja", ""),
                   display_en=spec["display"].get("en", ""),
                   args=args_str)
        if raw_candidates:
            prompt += t("interpreter.llm_other_candidates", lang)
            for cand in raw_candidates:
                cname = self._v.resolve_alias(cand.get("operation", "")) or cand.get("operation", "")
                cspec = self._v.get_spec(cname)
                if cspec:
                    cargs_str = ", ".join(f"{k}={v}" for k, v in cand.get("args", {}).items())
                    cdisp = cspec["display"].get(lang) or cspec["display"].get("en", "")
                    prompt += f"\n  - {cdisp}: {cargs_str}"

        return InterpretResult(
            ok=True,
            operation=canonical,
            args=args,
            confidence=confidence,
            source="llm",
            requires_confirmation=True,
            confirmation_prompt=prompt,
            candidates=[
                {
                    "operation": self._v.resolve_alias(c.get("operation", "")) or c.get("operation", ""),
                    "args": c.get("args", {}),
                }
                for c in raw_candidates
            ],
            symbol_updates=symbol_updates,
            dependencies=dependencies,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Split text into tokens, handling Japanese/ASCII mixed input.

    Steps:
    1. Insert space at ASCII↔CJK script boundaries (e.g. "E1を" → "E1 を").
    2. Split on whitespace and standard Japanese punctuation.
    3. Strip leading/trailing single-char Japanese particles from each token.
    4. Discard tokens that become empty after stripping (pure particles like "を", "で").
    """
    # Step 1: script boundary normalization
    text = _ASCII_TO_CJK.sub(r"\1 \2", text)
    text = _CJK_TO_ASCII.sub(r"\1 \2", text)
    # Greek ↔ Japanese boundary (e.g. "Δを" → "Δ を")

    # Step 2: split — **括弧の中は切らない**。
    #
    # 実測 2026-08-22（利用者報告「非可換演算子でガリガリ計算したい」を測っている
    # 最中）: `c(1,u) と c'(1,u) の反交換子` が
    # `anticommutator(a=E1, b="c'(1")` になり、Julia が生の `ParseError:` を返して
    # いた。サイト＋スピンという凝縮系でいちばん基本的な綴りが、**計算に届かない**。
    # QA 自身は `{c(1,u), c'(1,u)} = 1` を正しく返す——壊れていたのは指示の側だけ
    # である。カンマが区切りかどうかは括弧の深さで決まる、というのはこのプロジェクト
    # が `=` の分割で何度も踏んだのと同じ規則。
    raw = _split_outside_brackets(text.strip())

    # Steps 3 & 4
    result: list[str] = []
    for tok in raw:
        if not tok:
            continue
        stripped = _JP_PARTICLES_STRIP.sub("", tok)
        stripped = nlp_sub("verb_tail_strip", stripped)   # "実部を取り出して" → "実部"
        stripped = _JP_COMMAND_SUFFIX.sub("", stripped)
        stripped = _JP_PARTICLES_STRIP.sub("", stripped)  # re-strip: "とする"→"と" exposed by suffix removal
        stripped = nlp_sub("ordinal_prefix_strip", stripped)  # "次でテーラー展開" → "テーラー展開"
        stripped = _EN_TRAILING_PUNCT.sub("", stripped)   # "DIFF." → "DIFF"
        # Split on "として" (compound postposition meaning "as/in the capacity of").
        # e.g. "フェルミオン演算子として定義" → ["フェルミオン演算子", "定義"]
        # This allows the type noun ("フェルミオン演算子") and operation verb ("定義") to
        # each hit their own alias entries independently.
        if "として" in stripped:
            for part in stripped.split("として"):
                part = _JP_PARTICLES_STRIP.sub("", part)
                if part:
                    result.append(part)
            continue
        if stripped:
            result.append(stripped)
        # Discard tokens that become empty after stripping (particles, "して" alone, etc.)
    return result
