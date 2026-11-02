"""Which operation arguments may name an expression by id — ONE answer.

EXPR-REF-FIELD-TYPES.  Measured 2026-08-04, the codebase held **three different
answers** to this question:

===============================  ==========================================
Maxima's command builder         a DENYLIST (`_IDENTIFIER_KEYS`): every string
                                 arg is expanded except `name` / `_all_names` /
                                 `_raw_names`, which are passed through unparsed
Julia's command builder          **no notion of identifier fields at all** —
                                 every string arg is expanded
`OperationValidator`             an ALLOWLIST of 13 field names
===============================  ==========================================

The allowlist is what let EXPR-REF through: `rhs` was missing from it, so
`E1 の a(1) に E99 を代入する` (E99 nonexistent) was never refused — the expander
simply failed to substitute, the literal was parsed as a SYMBOL, and
``a^†(1)a(1) = E99²`` was registered as a success.

The invariant, stated precisely:

    **Every field the EXPANDER would substitute must be one the VALIDATOR has
    checked exists.**

Note what this does NOT say.  It is not about whether a field is "semantically a
name" — `var` holds an integration variable, yet the expander substitutes it
(measured: `integrate {expr: E1, var: E1}` builds
``integrate(x^2 + 1, x^2 + 1)``).  Arguing about which fields *ought* to accept a
reference is a separate question, and answering it here would leave the same gap
in a new place.  What matters is that the two sides agree, so behaviour cannot
depend on whether the id being written happens to exist.

That makes the DENYLIST the right shape, and it already exists: identifier
fields are the handful of exceptions, expression fields are everything else.
Both builders and the validator now read this module.
"""
from __future__ import annotations

import re

#: Fields carrying an identifier NAME the researcher chose, never an expression.
#:
#: They are passed through unparsed by the command builders — a subscripted name
#: like `k_B` would otherwise be mis-parsed as ``Subscript(k, B)`` and lowered to
#: ``k(B)`` — and, for the same reason, an `E\d+` here is a NAME, not a
#: reference: ``define_scalar_parameter {"name": "E1"}`` declares a symbol
#: called E1 and must keep working.
#: `pair_constraint` and `boundary` are ENUMERATED VALUES — a lattice name
#: (`square`, `honeycomb`, `custom`) and an edge condition (`open`, `periodic`,
#: `cylinder`).  They are matched against fixed tables in `lattice_ops`, never
#: evaluated, and an E-id in either is meaningless.  They were parsed as
#: mathematics only because nothing said otherwise, which stayed invisible while
#: the parser DISCARDED what it could not read: measured 2026-08-15, an
#: unrecognised Japanese lattice name (`ダイヤモンド`) parsed to nothing, and the
#: operation's own check — which names the lattice and lists the known ones —
#: got to answer.  Once the parser refuses unreadable characters instead of
#: dropping them, that generic refusal arrives FIRST and replaces a specific,
#: actionable message with "the character ダ cannot be read as mathematics".
#: A value that is never mathematics must not be parsed as mathematics; whether
#: the researcher's word is a known lattice is the lattice code's question.
IDENTIFIER_KEYS: frozenset[str] = frozenset({
    "name", "_all_names", "_raw_names", "pair_constraint", "boundary",
    # `kind` は**ノルムの種類**（`frobenius` / `spectral` / `max_col_sum` /
    # `max_row_sum`）。上の格子名と同じで固定表と突き合わせるだけの語であり、
    # 式番号を書いても意味が無い。数式として構文解析すると `max_col_sum` が
    # `\max` の添字付き（＝最大／最小を取る束縛子）として読まれ、
    # **ノルムとは無関係な断り**が研究者に返っていた（実測 2026-09-10）。
    "kind",
    # `side` is "left"/"right" — WHICH SIDE a non-commutative factor goes on,
    # not a quantity.  Parsed as mathematics it became the symbol `left`, and
    # the sided template was then never chosen (measured 2026-08-21).
    "side",
    # `direction` is "plus"/"minus" — WHICH SIDE a limit is taken from.  Same
    # shape as `side`: a word matched against a fixed table, and it is what
    # picks the sided template (LIMIT-ONE-SIDED, 2026-09-11).
    "direction",
})

#: 変数の名前を運ぶ欄。**3つ目の軸**であり、上の2つからは導けない——`var` は
#: `IDENTIFIER_KEYS` に**入っていない**（`integrate {expr: E1, var: E1}` は
#: 実際に展開される、とこのファイルの冒頭が測って書いている）ので、既にある
#: 2つの集合の言葉では「この欄は変数を名指している」と言えない。
#:
#: 読むのは1箇所だけ——`operand_guards.translate_value_bearing_variable_failure`
#: が、CAS が断ったあとに「値を持つ名前が変数の欄に立っていたのか」を訊く
#: （DEFINED-NAME-AS-VARIABLE、実測 2026-09-19）。
#:
#: **これは「断る欄の一覧」ではない。断るかどうかを決めるのは CAS である。**
#: 束縛された名前を変数の欄に置いて15操作を測ると、断るのは `diff` / 定積分 /
#: `taylor` / `limit` / `residue` だけで、`solve` / `sum` / `product` / `ratsimp` /
#: `partfrac` / `charpoly` / `coeff` は**答え、そしてそれが正しい**——`solve` は
#: 「その名前が指すもの」について解くので、`scenario_coulomb_gauge` は
#: `Az_z = \partial A_z/\partial z` と名前をつけてからゲージ条件を `Az_z` に
#: ついて解いている。この集合が決めているのは**断り文が何について語るか**
#: （「{操作}の変数にできません」）であって、断る条件ではない。
#:
#: **`var` だけである。`vars` は測って外した。** その3人の消費者のうち
#: `solve_system` は `block([x, y], …)` で名前を影に隠すので、束縛が在っても
#: 無くても**同じ命令**を出す（実測 2026-09-19、バイト一致）。`build_inverse_
#: transform` も同じ `block` を使い、`define_coordinate_system` は変数を
#: **宣言する**側で、使う側ではない。
#:
#: **`old_var` / `new_var` は 2026-09-19 に足した——出口は足すたびに数え直す。**
#: 起票時（同日の午前）は「2a-sym が名前を式番号へ書き換えるので、CAS へ行く前に
#: 『必要な引数が不足』で止まる」と書いてあり、それは真だった。その 2a-sym の枠を
#: 広げた瞬間（DEFINED-NAME-VAR-REWRITTEN）`E1 の y を u に変数変換して` が CAS
#: まで届くようになり、**生の `Maxima evaluation error`** が出た。断り文の
#: 「{操作}の変数にできません」はこの欄についても真である（変数変換の変数である）。
#: **`new_var` は入れない——測ったら効かなかった。** 束縛された名前を新しい変数に
#: 書いても Maxima は断らず、式をそのまま返す（`changevar` は積分のための関数で、
#: ただの式に当てると何もしない。これは別の欠陥で、起票 CHANGE-VARIABLE-NOOP）。
#: 断らないものに断り文は要らない。複数形の `old_vars` / `new_vars`
#: （`change_integration_variables`）も同じく**測っていないので入れない**
#: ——在る穴だけを塞ぐ。
VARIABLE_FIELDS: frozenset[str] = frozenset({"var", "old_var"})


#: What an expression KEY looks like: the researcher's `E7`, or the scaffolding
#: `_T3` that never got a number (INTERNAL-INTERMEDIATE-EID-GAP).
_EXPR_ID_RE = re.compile(r"^(?:E\d+|_T\d+)$")


def is_expr_ref(value: object) -> bool:
    """Whether *value* is a key into ``engine.expressions``.

    **This is the RESOLUTION-side question, and it is not the same as "did the
    researcher type an E number".**  Nobody types `_T3`; the NL parsers and the
    supervisor keep their own `E\\d+` patterns precisely so that they cannot
    match one.  What this predicate answers is "if I look this string up in
    expr_lookup, is a hit meaningful?" — and there the internal keys must
    count, because a builder that fails to recognise one passes the KEY to the
    CAS as a bare symbol.

    Measured 2026-08-06: this shape was written out by hand at eleven
    resolution sites, so introducing internal keys made `E1 の両辺に 2 をかけて`
    return ``2 E = _{T}_{1}`` — the id itself, rendered as LaTeX, with no error
    anywhere.  The nine sites that pair the check with ``in expr_lookup`` were
    each individually correct and each individually blind to the new key: one
    question, eleven answers.
    """
    return isinstance(value, str) and bool(_EXPR_ID_RE.match(value))


def iter_expr_refs(args: dict) -> "list[tuple[str, str]]":
    """Yield ``(field, expr_id)`` for every argument value that names an expression.

    Walks the same shapes the builders walk: a string, a list of strings, and a
    list of DICTS — `operator_substitute`'s multi-rule form passes
    ``op_rules=[{"op_name": …, "rhs": "E6"}, …]`` and the expander reaches into
    those dicts, so a checker that stops at the top level leaves exactly the
    values the expander will act on unchecked.
    """
    found: list[tuple[str, str]] = []

    def _take(field: str, v) -> None:
        if field in IDENTIFIER_KEYS:
            return
        if isinstance(v, str) and _EXPR_ID_RE.match(v):
            found.append((field, v))

    for field, value in args.items():
        for v in (value if isinstance(value, list) else [value]):
            if isinstance(v, dict):
                for k, dv in v.items():
                    _take(k, dv)
            else:
                _take(field, v)
    return found
