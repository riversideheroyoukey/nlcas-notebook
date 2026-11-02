r"""From TOKENS to an operation and its arguments.

Three steps, kept together because they are parts of one question — what did the
researcher ask for, and with what:

* :func:`scan_aliases` — which operation the words name;
* :func:`prefer_operations_holding_named_fields` — which member of an alias
  family, when the researcher named a field that decides it;
* :func:`map_tokens_to_fields` — which leftover token fills which field.

The middle one joined the other two on 2026-08-16 (BP-6).  Its argument is
``map_tokens_to_fields``' own pre-pass 0 — *a researcher who names a field has
removed the ambiguity* — which had been applied to FILLING a field but not to
CHOOSING the operation, so `integrate … lower=0, upper=1` reached
`integrate_indefinite` and the written bounds were dropped in silence.  Keeping
both halves in one file is the point: that is where the split hid.

Split out of ``llm_supervisor`` on 2026-08-15, when the EN-ALIAS-MULTIWORD fix
grew the mapper past that file's hygiene ceiling.  Its 2026-08-06 note had
already named the condition ("do it when the function grows again") and the
obstacle: the mapper shares four regexes with the supervisor, so moving it alone
would import backwards.  Those regexes and ``EXPR_FIELDS`` therefore live HERE
and the supervisor imports them, which is the direction that does not cycle.
"""
from __future__ import annotations

import re

from notebook.runtime.nl_patterns import nlp, nlp_search, pick_group
from notebook.runtime.expr_ref_fields import IDENTIFIER_KEYS
from notebook.runtime.number_words import number_literal

# Expression ID pattern (E1, E2, ...)
EXPR_ID_RE = re.compile(r"^E\d+$")

# CJK codepoint range — tokens containing these are natural-language phrases,
# not math expressions or operator names.
HAS_CJK_RE = re.compile(r"[　-ヿ一-鿿＀-￯]")

# Tokens that consist only of ASCII word characters are valid variable names /
# values.  CJK-containing tokens are natural-language phrases ("この式") and
# should not be mapped to parameter fields like `var`.
PURE_ASCII_RE = re.compile(r"^[A-Za-z0-9_]+$")

# Taylor expansion center: "x=1の周り", "x=0まわり", "x=2で", "x=-1"
POINT_RE = re.compile(r"[A-Za-z]\s*=\s*(-?[\d]+(?:\.[\d]+)?(?:/[\d]+)?)")

#: 極限の点に付いた片側の印 —— `0^+` / `0^{+}` / `0+`（`-` も同様）。
#: `\lim` の**記法**側は変換器の入口（`_limit_direction`）が同じ問いに答えて
#: おり、こちらは**指示文**の側。綴りは同じでも層が違うので両方が要る。
_LIMIT_POINT_SIDE_RE = re.compile(
    r"^(?P<point>[^\s^]+?)\s*(?:\^\s*)?\{?\s*(?P<sign>[+-])\s*\}?$")

#: Fields that expect an EXPRESSION reference rather than a parameter value.
#: ONE definition: this used to exist twice — a class attribute and a
#: module-level copy — and they had already drifted.  The BP-6 fix that added
#: convolve's `f`/`g` (2026-08-06) landed in the module-level copy only, so
#: `needs_expression_injection` kept the old answer.
#: `payoff_a`/`payoff_b` are pure_nash_equilibria's two matrices; without them
#: the positional mapper could fill NEITHER field and the operation was
#: unreachable from natural language entirely — even typing its own alias
#: (measured 2026-08-14, SCENARIO-MACRO case 4).
EXPR_FIELDS: frozenset[str] = frozenset({
    "expr", "expr_ref", "equation", "a", "b", "relation_ref",
    "operator", "hamiltonian", "f", "g", "payoff_a", "payoff_b",
})


def scan_aliases(tokens: list[str], resolve, max_words: int
                 ) -> "list[tuple[int, int, str]]":
    r"""Alias matches over token N-GRAMS, longest first, non-overlapping.

    Returns ``(start, end, canonical)`` spans in text order, ``end`` exclusive.

    Scanning ONE token at a time was the whole of open_items EN-ALIAS-MULTIWORD:
    English is written with spaces, so an alias containing one could never match
    — measured, 126 of the table's aliases, leaving 21 operations with no
    reachable English spelling at all.  Worse than unreachable, 48 of those
    phrases CONTAIN a shorter alias belonging to a DIFFERENT operation, which
    matched instead: `Take the inverse Laplace transform of E1 from s to t` ran
    the FORWARD transform, because `laplace` alone is an alias of it.

    Longest-first with the matched span consumed is what makes the contained
    alias lose.  Japanese is unaffected either way (its aliases are single
    tokens), which is why a JP scenario could never surface this.
    """
    n = len(tokens)
    consumed = [False] * n
    matches: list[tuple[int, int, str]] = []
    for size in range(min(max_words, n), 0, -1):
        for i in range(n - size + 1):
            if any(consumed[i:i + size]):
                continue
            canonical = resolve(" ".join(tokens[i:i + size]))
            if canonical is None:
                continue
            matches.append((i, i + size, canonical))
            for j in range(i, i + size):
                consumed[j] = True
    matches.sort()
    return matches


def field_assignment(token: str) -> tuple[str, str] | None:
    """Read one ``name=value`` token, or None if it is not one.

    The ONE place this file splits on a bare ``=``.  Both readers of that
    spelling go through here — pre-pass 0 below (which fills the field) and
    :func:`named_fields` (which lets it choose the operation) — because they
    are one question, and answering it twice is the shape of defect this whole
    area keeps producing.

    The FIRST ``=`` separates: a value may contain its own (``expr=f(a=1)``),
    a field name never does.
    """
    if "=" not in token:
        return None
    k, _, v = token.partition("=")
    k, v = k.strip(), v.strip().rstrip(",、")
    return (k, v) if k and v else None


def named_fields(tokens: list[str], known_fields: "frozenset[str]") -> set[str]:
    """The field names the researcher wrote as ``name=value``.

    Only names some operation actually accepts count, so an unrecognised
    ``n=0`` is never read as naming a field.  That filter is NOT what keeps a
    substitution from steering the operation — nine field names are one or two
    characters and `solve_ode` really has fields called `x` and `y`; what keeps
    it safe is that a swap needs a discriminating sibling, enumerated in
    `tests/test_named_field_selects_operation.py`.
    """
    out: set[str] = set()
    for tok in tokens:
        kv = field_assignment(tok)
        if kv and kv[0].lower() in known_fields:
            out.add(kv[0].lower())
    return out


def prefer_operations_holding_named_fields(
    tokens: list[str],
    candidates: list[str],
    matched_span: dict[str, tuple[int, int]],
    validator,
    backend: str,
) -> list[str]:
    """Swap a candidate for the sibling that can hold the fields NAMED in *text*.

    Fires only when the researcher removed the ambiguity themselves, and only
    when the answer is unique:

    * they named fields the candidate cannot hold, and
    * exactly one operation sharing the SAME ALIAS WORD can hold them all.

    Anything less decidable is left exactly as it was.  ``laplace``
    (transform vs Laplacian) and ``product`` (multiply vs ∏) are alias
    collisions too, and no field name tells them apart — they keep the index's
    first-come answer rather than acquiring a guess.
    """
    named = named_fields(tokens, validator.all_field_names())
    if not named:
        return candidates
    out: list[str] = []
    for canonical in candidates:
        if named <= validator.field_names(canonical):
            out.append(canonical)
            continue
        start, end = matched_span[canonical]
        alias_word = " ".join(tokens[start:end])
        holders = [
            op for op in validator.operations_for_alias(alias_word)
            if op != canonical
            and backend in (validator.get_spec(op) or {}).get(
                "supported_backends", [])
            and named <= validator.field_names(op)
        ]
        if len(holders) == 1:
            matched_span[holders[0]] = (start, end)
            out.append(holders[0])
        else:
            out.append(canonical)
    return out


def alias_siblings_that_complete(
    tokens: list[str],
    candidates: list[str],
    matched_span: dict[str, tuple[int, int]],
    validator,
    backend: str,
    known_expr_ids: set[str],
    lang: str,
) -> list[str]:
    """Offer the sibling the sentence actually fills, when no candidate is.

    ALIAS-COLLISION-UNREACHABLE.  The first-come winner of a colliding alias
    can be an operation the sentence cannot possibly complete, and the failure
    then asks for fields that make no sense for what was written — measured
    2026-08-17, `laplace of E1` demanded `t` and `s`, which a Laplacian does
    not have, while `scalar_laplacian` needed only the `expr` already given.
    Same for `product of E1 and E2`, which demanded a product's index and
    bounds.

    **The sibling is OFFERED, never substituted.**  The caller adds it as a
    second candidate, so the researcher is asked.  Deciding here would turn a
    safe failure into a silent wrong answer: omitting `t`/`s` is at least as
    likely to be a forgotten argument as it is to mean the Laplacian, and
    nothing in the text says which.  This is the opposite situation from
    `prefer_operations_holding_named_fields`, where the researcher HAD written
    the deciding word.

    Returns a sibling only when exactly one qualifies — anything less
    decidable is left alone.
    """
    out: list[str] = []
    for canonical in candidates:
        spec = validator.get_spec(canonical) or {}
        start, end = matched_span[canonical]
        remaining = [t for i, t in enumerate(tokens) if not (start <= i < end)]
        if _fills(remaining, spec.get("required_fields", []), known_expr_ids, lang):
            return []                      # this one works; nothing to offer
        for op in validator.operations_for_alias(" ".join(tokens[start:end])):
            sib = validator.get_spec(op) or {}
            if op in candidates or op in out:
                continue
            if backend not in sib.get("supported_backends", []):
                continue
            if _fills(remaining, sib.get("required_fields", []),
                      known_expr_ids, lang):
                out.append(op)
                matched_span[op] = (start, end)
    return out if len(out) == 1 else []


def _fills(tokens, required_fields, known_expr_ids, lang) -> bool:
    """Whether the leftover tokens supply every required field."""
    if not required_fields:
        return True
    args = map_tokens_to_fields(list(tokens), list(required_fields),
                                known_expr_ids, lang)
    return all(f in args for f in required_fields)


def refine_alias_candidates(
    tokens: list[str],
    candidates: list[str],
    matched_span: dict[str, tuple[int, int]],
    validator,
    backend: str,
    known_expr_ids: set[str],
    lang: str,
) -> list[str]:
    """Which operations the alias matches REALLY offer, given what was written.

    Two readings of one question, kept in one call because they are the same
    question asked of the same alias family, and the alias index cannot answer
    either (it is single-valued and first-come):

    * the researcher NAMED a deciding field → swap to the sibling that holds it;
    * the researcher named nothing, and the winner cannot be completed while
      exactly one sibling can → OFFER that sibling as a second candidate, so
      the caller asks rather than decides.
    """
    candidates = prefer_operations_holding_named_fields(
        tokens, candidates, matched_span, validator, backend)
    return candidates + alias_siblings_that_complete(
        tokens, candidates, matched_span, validator, backend,
        known_expr_ids, lang)


def map_tokens_to_fields(
    tokens: list[str],
    required_fields: list[str],
    known_expr_ids: set[str],
    lang: str = "ja",
    optional_fields: "list[str] | None" = None,
    text: str = "",
    declared_objects: "frozenset[str] | set[str]" = frozenset(),
) -> dict[str, str]:
    """Map remaining tokens to required_fields positionally.

    Expression IDs are anchored to `expr`-class fields in order; plain
    string tokens fill non-expr fields.
    """
    args: dict[str, str] = {}

    # Pre-pass 0: an EXPLICIT `field=value` wins over everything below.
    #
    # BP-6's original report was about this phrasing and it was still unhandled
    # (measured 2026-08-06): `p=1/6, var=x, lower=1, upper=6 の
    # シャノンエントロピーを計算して` tokenises to
    # ['p=1/6', 'var=x', 'lower=1', 'upper=6', 'シャノンエントロピー'] and the
    # positional pass below filled NOTHING from it — and for kl_divergence it
    # was worse than nothing: the word `KL` from the sentence landed in `p`.
    # A researcher who names the fields has removed the ambiguity; reading the
    # names is strictly better than guessing from position, so it goes first.
    #
    # **The rule applies to OPTIONAL fields too, and used not to.** BP-6's own
    # sentence is "a researcher who names a field has removed the ambiguity",
    # and the CHOOSING half already counts optional fields
    # (`prefer_operations_holding_named_fields` reads required ∪ optional) while
    # this, the FILLING half, read only the required ones — so every optional
    # field written by name was dropped in silence. Measured 2026-09-11:
    # `limit of E1 with var=x, point=0, direction=plus` reached the builder with
    # no `direction`, and a one-sided limit was answered with the two-sided one.
    # The split between filling and choosing is where this file's defects keep
    # hiding; this is that split again.
    _nameable = set(required_fields) | set(optional_fields or ())
    _consumed: set[int] = set()
    for i, tok in enumerate(tokens):
        kv = field_assignment(tok)
        if kv and kv[0] in _nameable and kv[0] not in args:
            args[kv[0]] = kv[1]
            _consumed.add(i)
    if _consumed:
        tokens = [t for i, t in enumerate(tokens) if i not in _consumed]

    # Pre-pass: extract 'order' from numeric tokens before positional mapping.
    # _JP_ORDINAL_PREFIX in _tokenize strips "次で/次まで" so "4次でテーラー展開" →
    # tokens ["4", "テーラー展開"].  After the op token is removed, "4" stands alone
    # as a pure digit.  Map it directly to 'order'.
    # Also handles the unstripped "N次" form as a fallback.
    used_indices: set[int] = set()
    if 'order' in required_fields:
        for i, tok in enumerate(tokens):
            if re.match(r'^\d+$', tok):          # standalone digit: "4"
                args['order'] = tok
                used_indices.add(i)
                break
            m = nlp_search("order_suffix_token", tok)  # unsplit "4次…" fallback
            if m:
                args['order'] = m.group(1)
                used_indices.add(i)
                break

    # Pre-pass: which SIDE a limit is taken from.  Gated on the FIELD, not on
    # the operation name — `direction` is the discriminator's own name, so any
    # operation that grows one is read the same way (the `define_scalar_parameter`
    # special case above this file is what that looks like when it is not).
    #
    # Two spellings, both what a researcher writes: the MARK on the point
    # (`x → 0^+`, `0+`) and the WORD (`右極限` / `from the right`).  Neither
    # reached the operation before — the mark took `point` down with it (the
    # token `0+` is not a number, so nothing filled it) and the word was not an
    # alias at all, so the sentence did not name an operation (LIMIT-ONE-SIDED,
    # measured 2026-09-11).
    if 'direction' in _nameable and 'direction' not in args:
        for i, tok in enumerate(tokens):
            if i in used_indices:
                continue
            m = _LIMIT_POINT_SIDE_RE.match(tok)
            if m:
                args['direction'] = 'plus' if m.group('sign') == '+' else 'minus'
                if 'point' in required_fields and 'point' not in args:
                    args['point'] = m.group('point')
                used_indices.add(i)
                break
        else:
            # Scanned over the WHOLE sentence, not the leftover tokens: the
            # word that names the side (`右極限` / `right-hand limit`) is also
            # the ALIAS that named the operation, and the alias scan has already
            # eaten it by the time this runs.  Measured 2026-09-11 — reading the
            # leftovers found the direction in `… 0 から右から` and not in the
            # spelling everyone writes.
            m = nlp_search("limit_direction", text or " ".join(tokens))
            if m:
                args['direction'] = 'plus' if pick_group(m, 'plus') else 'minus'

    # Pre-pass: extract 'point' from "x=N" / "x=N.M" patterns.
    # Matches "x=1の周り", "x=-1まわり", "x=0で" etc.
    if 'point' in required_fields and 'point' not in args:
        for i, tok in enumerate(tokens):
            if i in used_indices:
                continue
            m = POINT_RE.search(tok)
            if m:
                args['point'] = m.group(1)
                # `x=1` names the VARIABLE as well as the point.  Without this,
                # `E1 の x=1 における留数を求めて` consumed the only token that
                # could fill `var`, and the operation stalled on a missing field
                # (measured 2026-08-06).
                if 'var' in required_fields and 'var' not in args:
                    args['var'] = m.group(0).split('=')[0].strip()
                used_indices.add(i)
                break

    remaining = [t for i, t in enumerate(tokens) if i not in used_indices]

    # Separate expr-ID tokens from parameter tokens.
    # expr_tokens: E-id (E1, E2…) OR math/operator expressions like a(1), a'(1), σ_z
    #   — anything non-CJK that isn't a plain identifier (the latter stays as param_token).
    # param_tokens: plain identifiers (pure A-Za-z0-9_) — fill var/name/lhs/rhs fields.
    # CJK-containing tokens are natural-language phrases ("この式") — excluded from both.
    # Tokens like "a_1", "Sm_1", "a'_1" look like ASCII but are quantum operator
    # expressions (letter + underscore + digit).  Treat them as expr_tokens so they
    # fill operator/hamiltonian fields rather than name/var fields.
    _OP_WITH_SITE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_'†]*_\d+$")
    # **式には名前か数がある。** 3つめの条件は「非 CJK で、素の識別子でない」
    # ——つまり *それ以外ぜんぶ* で、句読点や関係子の単独トークンまで通していた。
    # 実測 2026-09-05: `substitute x = 2 into E1` は `['x','=','2','into','E1']`
    # と切られ、**`=` が最初の式トークン**になって
    #     {'expr': '=', 'lhs': 'x', 'rhs': '2'}
    # ——`E1` が同じ文に立っているのに、である。`In E1, substitute x = 2` だけが
    # 正しかったのは E1 が先に立っていたからで、判定の結果ではない。
    #
    # **「既知の E-id を先に取る」ではない。** それは一度書いて実測で覆った:
    # `heisenberg_equation` は `operator, hamiltonian` の順で、`ハイゼンベルク
    # 方程式 a(1) E5` の E-id は**ハミルトニアン**の側である。順序は書かれた
    # とおりが正しく、落とすべきなのは「どうやっても式ではないもの」だけ。
    _HAS_NAME_OR_NUMBER = re.compile(r"[A-Za-z0-9]")
    #
    # **宣言された対象の名前は、それ自体が式である**（OPERAND-BOUNDARY 残①、
    # 2026-09-11）。宣言には `defining_expr_id` が無いので 2a-sym が式番号へ
    # 書き換えられず、`A` は素の ASCII なので下の `param_tokens` へ行き、
    # `expr` は誰にも埋められないまま「必要な引数が不足」で断られていた
    # ——`A のノルムを求めて` / `the norm of A`、日英とも。
    # **操作の側は最初から名前を扱える**（実測: `norm(expr='A')` は
    # `sqrt(A_z^2+A_y^2+A_x^2)`、`curl(expr='A')` は3成分）ので、欠けていたのは
    # ここ1箇所だった。`formula_is_an_operation_operand` の註が既に
    # 「`operator`/`a`/`b` は**それ自体が名前を持つもの**だから E 番号は要らない」
    # と言っており、その原則を `expr` にも広げたことになる。
    #
    # **スカラーは対象ではない**（`declared_object_names` が除く）——`x` は変数
    # として文中に立つので、数えると `E1 を x で微分して` の `x` が `expr` を
    # 奪う。判別子は足場を止めるときと同じものを使う。
    expr_tokens = [
        t for t in remaining
        if EXPR_ID_RE.match(t)
        or t in declared_objects
        or _OP_WITH_SITE_RE.match(t)
        or (not HAS_CJK_RE.search(t) and not PURE_ASCII_RE.match(t)
            and _HAS_NAME_OR_NUMBER.search(t))
    ]
    # Grammar words are not operands.  Without this the positional pass below
    # filled `s`/`t` from `Take` and `the`, and `delta(to)` was registered as a
    # success (open_items EN-ALIAS-MULTIWORD).  The list is per-language data in
    # i18n/nl_patterns.json, not a literal here.
    _stop = nlp("operand_stopwords", lang)
    # 式トークンになったものはパラメタ欄に**入れない**——両方に入れると、
    # `A を x で微分して` が `expr=A, var=A` になる（同じトークンが2つの欄へ）。
    def _param_value(tok: str) -> "str | None":
        """このトークンがパラメタ欄に入れるなら、その**正準の綴り**。"""
        if EXPR_ID_RE.match(tok) or tok in declared_objects or _stop.match(tok):
            return None
        if PURE_ASCII_RE.match(tok) and not _OP_WITH_SITE_RE.match(tok):
            return tok
        # **名前は、素の識別子だけではない。** 研究者は `\bar{y}` / `y'` / `x_1` と
        # 書く——どれも1つの名前であり、上の `PURE_ASCII_RE` はどれも通さないので
        # 式トークンへ回り、`var` は誰にも埋められないまま「必要な引数が不足して
        # います」で断られていた（実測 2026-09-19、不定積分・部分分数・留数・極限で、
        # 日英とも。`x_1` は普通の変数である）。ここには `\alpha` だけを救う手書きの
        # 規則が在り、`\alpha` が通って `\bar{y}` が通らない理由は**誰も決めて
        # いなかった**。判定と変換は1つの共有ヘルパが答える
        # （`latex_name_to_canonical`。逆向きの `restore_decorations` と同じ表を、
        # 同じファイルで読む）。式（`a+b`・`f(x)`）と包み（`\vec{A}` → `vec(A)`）
        # には `None` を返す。
        from notebook.runtime.latex_converter import latex_name_to_canonical
        return latex_name_to_canonical(tok)

    # **順序は書かれたとおり**——救った名前を末尾へ足すと、`留数` の
    # `E1 の \bar{y} = 1 における留数を求めて` が `var=1, point=y_bar` になり、
    # 生の `Maxima evaluation error` が返る（実測 2026-09-19。素の名前で通るのは
    # トークン順に並ぶからで、判定の結果ではない）。だから1回の走査で決める。
    param_tokens = [v for v in (_param_value(t) for t in remaining) if v is not None]


    expr_fields  = [f for f in required_fields if f in EXPR_FIELDS]
    param_fields = [f for f in required_fields if f not in EXPR_FIELDS and f not in args]

    for f, t in zip(expr_fields, expr_tokens):
        args[f] = t
    for f, t in zip(param_fields, param_tokens):
        args[f] = t

    # **語で書かれた数は数である**——欄の値でも、式のときと同じ規則
    # （`to_canonical` が「文全体が数詞なら数」を答えている。理由と語彙は
    # `notebook/runtime/number_words.number_literal`）。
    #
    # インターセプタが読む綴りは値を**書かれたテキスト**として扱ってからその扉に
    # 渡すので最初から通っていたが、別名走査に落ちる綴りは欄がここで埋まり、
    # そのまま IR を通って `zero` が正当な `Symbol` になる——
    # `substitute x = zero into E1` が `a*zero+b` を **completed** で返していた
    # （起票 FIELD-VALUE-OR-NAME、実測 2026-09-10）。同じ操作の他の綴り
    # （`substitute zero for x in E1` ほか7綴り）は全部通っていたので、これは
    # **この経路にだけ答えが無かった**形である。
    #
    # **どの欄を触ってはいけないかは、既に名指されている。** `IDENTIFIER_KEYS` は
    # 「識別子の名前を運ぶ欄」＝数式として解析しない欄の集合で、`name` はそこに在る。
    # 起票は「スキーマに欄の種別（値／名前／式）が要る」と書いていたが、**要る半分は
    # 既に在った**。この guard は飾りではない: 外すと
    # `define spin operator one` が**名前 `1` の記号**を宣言する（変異試験で実測。
    # 宣言の多くは `extra_pre_phase1` が `name` を自分で埋めるのでここを通らないが、
    # 演算子の宣言は別名走査に落ちて欄が位置で埋まる）。
    for _field, _written in list(args.items()):
        if _field in IDENTIFIER_KEYS or not isinstance(_written, str):
            continue
        _as_number = number_literal(_written)
        if _as_number is not None and _as_number != _written.strip():
            args[_field] = _as_number

    return args
