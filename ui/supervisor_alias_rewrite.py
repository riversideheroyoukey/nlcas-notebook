"""Rewriting a researcher's symbol names into E-IDs for the supervisor.

This is the "2a-sym" pass, lifted out of ``run_interpret_pipeline``: it is a
self-contained text transformation with two hard-won guards, and both of the
defects it has produced were about WHERE its output may be used, not about the
rewrite itself — so it belongs somewhere its rule and its exceptions can be
read together (the same split as ``subscript_policy`` / ``spin_label_canon``).

Its output is for the SUPERVISOR ONLY.  Whatever gets recorded — an
expression's canonical, its display latex — must come from the text as the
researcher typed it; registering the rewrite instead is H1-ALIAS-EID-LEAK
(``E1^2 + 1`` written down for ``G^2 + 1``, measured 2026-08-01).
"""
from __future__ import annotations

import re

from notebook.runtime.nl_patterns import fragment
from notebook.runtime.script_label import label_script_at

#: "Is this symbol the differentiation variable, or H-9's solve-for variable?"
#: — one question, two word orders.  Japanese puts the marker AFTER the symbol
#: (a particle), English BEFORE it (a keyword), so a named group cannot absorb
#: the difference the way it does for a stored pattern: the symbol here is the
#: researcher's own name, escaped and spliced in at runtime.  Hence two
#: fragments and ``fragment()`` rather than ``nlp()``.
_JA_OP_VAR_TAIL = fragment("ja_op_var_tail")
_EN_OP_VAR_HEAD = fragment("en_op_var_head")
#: 英語の「<求める動詞> <記号> from E<n>」。同じく前置詞だけでは足りない形で、
#: 2026-09-06 に solve 一族で測った——`find x from E1` は読めるのに guard が
#: 知らず、`x` が E-id に書き換わっていた。
_EN_SOLVE_VAR_FROM = fragment("en_solve_var_from")
#: 日本語の「E<n> から <記号> を 求め」。**前置き**の形で書く——後ろの尾
#: （`を` ＋ 求める動詞）だけで絞ると `Heisenberg 方程式 H を計算して` の
#: `H` まで守り、コーパス3行が変わる（実測 2026-09-06）。
_JA_SOLVE_FROM_CLAUSE = fragment("ja_solve_from_clause")
#: 助詞も i18n に置く——ここに `を` と直書きして ratchet に捕まった
#: （2026-09-06）。語は i18n、係り方はコード、の線はこの1文字にも及ぶ。
_JA_SOLVE_FROM_TAIL = fragment("ja_solve_from_tail")
#: 英語の「solve … in <記号>」。`in` を共有 `en_op_var_head` に入れるのは
#: 誤りで（二十二問目の実測: 両辺の演算子まで守り日英が割れる）、微分と同じ
#: 前置きの形で書く。
_EN_SOLVE_VAR_CLAUSE = fragment("en_solve_var_clause")
#: 英語の「<操作の動詞の節> … 前置詞 記号」。**前置詞だけでは足りない**——日本語の
#: `ja_op_var_tail` は「助詞＋操作の語」で*操作の語を含む*のに、英語側は前置詞
#: だけだった。だから `differentiate E1 by t` の `t` が E-id に書き換わっていた
#: （実測 2026-09-06。`by` は `diff_rule` が受ける綴りである）。
#: **`by|in|over` を `en_op_var_head` 側に足すのは誤り**: あれは共有なので
#: `multiply both sides of E2 by A` の `A` まで守り、日本語側は守らないので
#: 日英が割れる（実測: コーパス2005対で10件変化、うち2件がこれ）。
#: 2026-09-19 に**微分専用だったこの枠を一族ぜんぶへ広げた**（`en_deriv_var_clause`
#: を削除して `en_op_var_clause` に一般化）——積分・collect が同じ穴に落ちていた。
_EN_OP_VAR_CLAUSE = fragment("en_op_var_clause")
#: `記号 = 値` の左辺は**名前**である（`substitute y = 3 into E1` /
#: `the residue of E1 at y = 1`）。動詞の節に限るのは、素の `y = 3`——H-1 の
#: 定義そのもの——を巻き込まないため。
_EN_NAME_EQ_CLAUSE = fragment("en_name_eq_clause")
_JA_NAME_EQ_CLAUSE = fragment("ja_name_eq_clause")
#: 変数変換は**2つの名前**を名指す（`E1 の y を u に変数変換して`）。助詞が
#: `を`/`に` なので `ja_op_var_particle` では届かない。
_JA_CHANGE_VAR_OLD = fragment("ja_change_var_old")
_JA_CHANGE_VAR_NEW = fragment("ja_change_var_new")
#: 英語側の同じ枠。日本語だけが持っていて、英語は**どちらの名前も守っていなかった**
#: （実測 2026-09-19、`u` を解いたあと `change variable r to u in E1` が落ちる）。
_EN_CHANGE_VAR_CLAUSE = fragment("en_change_var_clause")


#: **「この記号は、その文で*名前として*立っているか」——枠の表。**
#:
#: 各要素は `(記号の前に来るもの, 記号の後に来るもの)`。語は fragment に、
#: 係り方はここに置く。**一覧であることが要点**で、読みが受ける綴りを guard が
#: 知らなければ、その綴りは「名前が式番号に化けて、書いてある欄について
#: 『引数が不足』と言われる」形で壊れる——実測でこれが3度目である
#: （微分・solve・そして 2026-09-19 の積分/極限/留数/部分分数/変数変換/代入）。
#: **読みを1つ足したら、ここに枠を1つ足す。**
_NAME_SLOT_FRAMES: tuple[tuple[str, str], ...] = (
    # 日本語: <記号> + 助詞 + 操作の語（微分・solve・その他の一族）
    ("", r"\s*(?:" + _JA_OP_VAR_TAIL + r")"),
    # 英語: 前置詞 + <記号>
    (_EN_OP_VAR_HEAD + r"\s+", r"\b"),
    # 英語: <操作の動詞の節> … 前置詞 + <記号>
    (_EN_OP_VAR_CLAUSE, r"\b"),
    # 英語: <求める動詞> + <記号> + from E<n>
    (_EN_SOLVE_VAR_FROM, r"\s+from\s+E\d+"),
    # 日本語: E<n> から + <記号> + を
    (_JA_SOLVE_FROM_CLAUSE, _JA_SOLVE_FROM_TAIL),
    # 英語: solve … in + <記号>
    (_EN_SOLVE_VAR_CLAUSE, r"\b"),
    # 英語: <代入/留数の節> … <記号> =
    (_EN_NAME_EQ_CLAUSE, r"\s*="),
    # 日本語: <記号> = 値 … 留数／代入
    ("", _JA_NAME_EQ_CLAUSE),
    # 日本語: <記号> を … に変数変換 ／ を … <記号> に変数変換
    ("", _JA_CHANGE_VAR_OLD),
    (_JA_CHANGE_VAR_NEW, r"\s*に\s*変数変換"),
    # 英語: change variable <記号> to … ／ … to <記号>
    (_EN_CHANGE_VAR_CLAUSE, r"\s+to\b"),
    (_EN_CHANGE_VAR_CLAUSE + r"to\s+", r"\b"),
)


def rewrite_symbols_to_eids(text: str, engine) -> str:
    """Return *text* with every aliased symbol replaced by its E-ID."""
    # 2a-sym. Replace known symbol names (H, H_BCS, …) with their E-IDs in
    # text BEFORE the LLM sees it.  Without this, the LLM can re-resolve
    # a symbol like "H" to a derived expression (e.g. E2 = [H, Sx]) instead of the
    # defining expression (E1 = H = -γBSz).  By pre-substituting "H" → "E1" here
    # the LLM receives unambiguous E-IDs and the correct expression is always used.
    # Only plain identifiers are replaced (not decorated forms like \hat{H}).
    if engine.lhs_alias_map:
        from notebook.engine import _extract_simple_symbol_name as _esym_sv
        # Do NOT replace a symbol when it appears as a differentiation *variable*
        # (e.g. "E1 を k で微分", "E1 を k で 2 回微分") or as an H-9 solve-for
        # *variable* (e.g. "E1 を k について解いて").  Once the user has solved
        # for or aliased a symbol (e.g. E3: k = …), that symbol is registered in
        # lhs_alias_map (k → E3); without this guard "微分 by k" would become
        # "微分 by E3" (d/dE3 instead of d/dk), and "を k について解いて" would
        # become "を E3 について解いて" (solve for the whole expression E3
        # instead of the named quantity k).
        # ONE spelling of "is this symbol the operation's variable?", not two.
        # The two guards below used to carry BYTE-IDENTICAL Japanese 35 lines
        # apart (measured 2026-08-27) — the whole-command check and this
        # lookahead ask the same question of the same words, and only the
        # anchoring differs.  The words are now `ja_op_var_tail` in
        # i18n/nl_patterns.json; the anchoring stays here, because it is
        # structure.
        _diff_var_ahead = r'(?!\s*(?:' + _JA_OP_VAR_TAIL + r'))'
        # English equivalents: "differentiate ... with respect to k",
        # "solve ... for k", "substitute ... for k" — same rationale as the
        # Japanese lookahead above, but English puts the operation keyword
        # BEFORE the variable (SVO order) instead of after (JP postposition),
        # so this needs a lookBEHIND, not a lookahead.  Python's re requires
        # fixed-width lookbehind, hence the literal single-space phrasing.
        #
        # This is the ONE place the English words could not follow
        # ``en_op_var_head`` into the i18n file: the two alternatives are
        # different lengths ("with respect to " is 16 chars, "for " is 4), so
        # they cannot share one fixed-width lookbehind, and a fragment holding
        # an alternation cannot be spliced into one either.  It is a structural
        # constraint of ``re``, not a decision — but it means these two literals
        # must be kept in step with that fragment by hand.
        #
        # It is the SECOND line of defence, and deliberately kept as one.
        # Measured 2026-08-27 over 9,204 (symbol, spelling) pairs from the
        # instruction corpus: with the whole-command check above intact,
        # removing this lookbehind changes NOTHING — every text it would block
        # has already made that check ``continue`` past the symbol, since that
        # check is strictly the more permissive of the two (``\s+`` for the
        # space, case-insensitive, and it also knows ``w.r.t.``).  What made
        # that worth keeping rather than deleting is the other half of the same
        # measurement: breaking ``en_op_var_head`` fails NO test, because on the
        # seven English scenario lines that reach this guard (measured by
        # instrumenting a full replay) THIS lookbehind still catches the
        # trailing occurrence.  The redundancy is what keeps a silent wrong
        # answer — SILENT-DIFF-VAR-ALIAS below — from returning on one edit.
        _diff_var_behind = (
            r'(?<!with respect to )'
            r'(?<!\bfor )'
        )
        # Sort longest-key-first to avoid partial matches (e.g. H_BCS before H).
        _sv_candidates = sorted(engine.lhs_alias_map.items(), key=lambda x: -len(x[0]))
        for _lhs_key, _eid in _sv_candidates:
            _sym = _esym_sv(_lhs_key)
            if not _sym:
                continue
            # Whole-command guard: if this symbol is used ANYWHERE in the
            # command as a differentiation/solve-for variable (not just the
            # occurrence immediately adjacent to the trigger phrase), skip
            # substituting EVERY occurrence of it, including ones earlier in
            # a raw formula argument (e.g. "differentiate a*x-b*x*y with
            # respect to x" — the "x" inside the formula must stay a free
            # variable too, not just the one after "with respect to").
            # Without this, once x is solved elsewhere in the session (x is
            # then in lhs_alias_map), the formula's own x got silently
            # replaced by its solved VALUE before differentiation ever ran,
            # producing a wrong answer with no error (SILENT-DIFF-VAR-ALIAS,
            # found via the lotka_volterra practice-loop scenario's English
            # inline-formula path — re-verified against a live Maxima
            # session: the corrupted command returned 0 instead of a-b*y).
            _quoted = re.escape(_sym)
            _sym_is_op_var_anywhere = bool(re.search(
                "|".join(_pre + _quoted + _post
                         for _pre, _post in _NAME_SLOT_FRAMES),
                text, re.IGNORECASE,
            ))
            if _sym_is_op_var_anywhere:
                continue
            # Only replace if the symbol appears as a standalone word (not inside
            # a longer identifier or preceded by backslash) and is not the
            # differentiation variable.
            # **A PRIME MAKES A DIFFERENT NAME**, by the same sentence the label
            # guard below gives for a superscript and the class above gives for
            # a subscript: `y'` is not `y`.  Measured 2026-09-24 in the
            # regression scenario's own replay — after `y = x^3`,
            # `E2 を y' で微分して` was rewritten to the variable `E1'` and the
            # researcher got a refusal naming a key they never wrote.  English
            # survived only because its frame ends at `\b`, so the variable-slot
            # guard above still fired; the Japanese frame wants the particle
            # immediately after the symbol and a prime stands in between
            # (PRIMED-VAR-NOT-A-VARIABLE).
            _pat = (r'(?<![A-Za-z0-9_\\{])' + _diff_var_behind + r'\b' + re.escape(_sym)
                    + r"\b(?![A-Za-z0-9_'])" + _diff_var_ahead)
            if re.search(_pat, text):
                # A LABEL superscript makes a DIFFERENT name (`G^{(HF)}` is not
                # `G`), exactly as the trailing `_` guard above already says for
                # a subscript.  An EXPONENT does not — `G^2` is the defined
                # quantity squared, and rewriting it is the point (see this
                # module's docstring).  Which of the two a script is belongs to
                # `script_label`, so it is asked rather than re-answered here.
                _src = text
                text = re.sub(
                    _pat,
                    lambda mm: (mm.group(0) if label_script_at(_src, mm.end())
                                else _eid),
                    _src)

    return text
