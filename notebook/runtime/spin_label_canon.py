"""Spin-label → QuantumAlgebra index canonicalization.

QuantumAlgebra's ``QuIndex`` accepts ONE character, optionally followed by an
integer subindex (``k``, ``k_1``, ``i_9``, ``ω`` — see ``_QA_INDEX_RE`` for the
measurements); ``QuIndex("sigma")`` raises ``ArgumentError: index names must be
single character``.  Spin labels routinely arrive multi-character — Unicode
normalization turns ``↑``/``↓`` into ``up``/``dn`` and ``\\sigma`` into
``sigma`` — so both routes that emit QA indices (the NL operator-substitute
patterns in ``JuliaBackendHandler`` and the expression path in
``JuliaCodeGenerator``) collapse them through THIS table.

One table, two callers, on purpose: the sigma leak existed precisely because
the NL route had the rule and the generator route did not (open_items
JULIA-INDEXED-SCALAR, "同種の漏れ").  Do not restate the mapping at a call
site.
"""
import re

SPIN_LABEL_CANON: dict[str, str] = {
    # **A spin projection is the ARROW, and QuantumAlgebra takes it directly**
    # (measured 2026-08-22: `QuIndex("↑")` renders `↑`, and QA's rule is ONE
    # unicode character, which an arrow is).  Every spelling of the same
    # projection — glyph, LaTeX command, word — lands on the one canonical.
    #
    # Why not the letters `u`/`d`, which this table used to produce: **an arrow
    # cannot be bound by a sum.**  Nobody writes `\sum_\uparrow`; they write
    # `\sum_\sigma`.  So two DIFFERENT arrows are two different VALUES, and
    # `δ(↑,↓)` is 0 by the same warrant that makes `δ(1,2)` 0 for integer sites
    # — a property of the notation, not a modelling choice.  A letter carries no
    # such warrant: a bare `u` really might be a dummy index, and with `u`/`d`
    # as the canonical every Hubbard commutator came back carrying `δ_{du}`
    # terms that the researcher knows are zero and the system could not.
    #
    # `\sigma` stays a LETTER on purpose: it IS the spin variable, and `δ(s,s')`
    # inside `Σ_σ` must keep the sifting property.  The distinction physics
    # needs — value vs variable — falls straight out of what was written.
    'up': '↑', 'uparrow': '↑',
    'dn': '↓', 'down': '↓', 'downarrow': '↓',
    'sigma': 's',
    'alpha': 'a', 'beta': 'b',
    # **整数スピンの射影**（`⁺ ⁰ ⁻`、2026-09-12）。矢印と同じ理由でここに在る——
    # QA の添字は1文字で、上付き記号はその条件を満たす。名前が ASCII なのも同じ
    # 理由（canonical は再パースされ、識別子規則は `[A-Za-z][A-Za-z0-9]*`）。
    # `m` を冠しているのは磁気量子数の `m` で、`plus`/`zero`/`minus` 単独だと
    # 数詞の表（`number_words` の `zero`）と衝突するため。
    'mplus': '⁺', 'mzero': '⁰', 'mminus': '⁻',
}


#: 入力に書かれた**素の符号** → その canonical NAME。
#:
#: `SPIN_ARROW_INPUT` の整数スピン版である。**矢印には入力の綴りが在り、符号には
#: 無かった**——`PROJECTION_LATEX` は QA のグリフを画面の符号にする片道だけを
#: 持っており、系が印刷した `a_{1,+}` を打ち返すと生の `ParseError:` が返っていた
#: （実測 2026-09-12、OPERAND-BOUNDARY ③ 残）。3つの表が合成して往復が閉じる:
#:
#:     QA `⁺` → 画面 `+`      (`spin_value_set.PROJECTION_LATEX`)
#:     入力 `+` → 名前 `mplus` (この表)
#:     名前 `mplus` → QA `⁺`   (`SPIN_LABEL_CANON`)
#:
#: **位置だけでは判別子にならない。** `\omega_0` は「数字だけの添字は名前の一部」
#: であり、`Y_{1,0}` は球面調和関数の成分である——だから読む側は**宣言**を見る
#: （`julia_latex_mixin`：基底がこのセッションの量子演算子であるときだけ）。
PROJECTION_SIGN_INPUT: dict[str, str] = {
    '+': 'mplus', '0': 'mzero', '-': 'mminus',
    '⁺': 'mplus', '⁰': 'mzero', '⁻': 'mminus',
}


#: Every INPUT spelling of a spin arrow → its CANONICAL NAME.
#:
#: The name is ASCII on purpose.  A canonical is re-parsed by the shared parser,
#: whose identifier rule is `[A-Za-z][A-Za-z0-9]*` — which is exactly why Greek
#: glyphs are turned into `\omega` → `omega` before they ever reach it.  Putting
#: the arrow GLYPH in the canonical was measured 2026-08-22 to make the parser
#: refuse the researcher's own Hamiltonian as an unreadable character.  The
#: arrow appears where it belongs: as the QA index (`SPIN_LABEL_CANON`) and on
#: screen.
SPIN_ARROW_INPUT: dict[str, str] = {
    '↑': 'uparrow', '↓': 'downarrow',
    r'\uparrow': 'uparrow', r'\downarrow': 'downarrow',
    r'\Uparrow': 'uparrow', r'\Downarrow': 'downarrow',
}



#: 演算子の添字らしい形（基底＋随伴の印＋波括弧の下付き）。
_OP_SUBSCRIPT_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9]*)((?:\^\{?\\dagger\}?|')?)_\{([^{}]*)\}")


def projection_signs_to_names(latex: str, integer_spin_ops: "set[str]") -> str:
    r"""素の符号の射影を、その canonical NAME にする（`a_{1,+}` → `a_{1,mplus}`）。

    **判別子は宣言で、しかも整数スピン（＝ボソン）である。** 位置だけで決めると
    実在する記法を壊す——`\omega_0` は「数字だけの添字は名前の一部」（CLAUDE.md の
    添字分類表）、`Y_{1,0}` は球面調和関数の成分、`S_+` は昇降演算子。演算子で
    あるだけでも広すぎて、電子の `c_{1,+}` に当てると**その粒子が持たない射影**を
    黙って添字にしてしまう。呼び出し側（`julia_latex_mixin`）が
    `statistics.kind == "bose"` で絞った名前を渡す。

    2つの形を読む:

    - **コンマ区切り**（`a_{1,+}`）——2026-09-12 以降**系が印刷する綴り**。
    - **貼り付いた形**（`a_{1+}`）——それ以前に印刷していた綴りで、研究者もこう
      書く。末尾の符号1文字だけを切り出すので、`a_{k+q}`（代数的な添字）は
      符号で終わらず当たらない。

    単独の成分（`a_{+}`）は触らない: 区切る相手がいないので、印刷側もコンマを
    入れておらず、往復の対象ではない。
    """
    if not integer_spin_ops:
        return latex

    def _one(m: "re.Match[str]") -> str:
        base, adj, body = m.group(1), m.group(2), m.group(3)
        if base not in integer_spin_ops:
            return m.group(0)
        parts = body.split(",")
        if len(parts) == 1:
            head, sign = body[:-1].strip(), body[-1:]
            if head and sign in PROJECTION_SIGN_INPUT:
                return f"{base}{adj}_{{{head},{PROJECTION_SIGN_INPUT[sign]}}}"
            return m.group(0)
        out = [PROJECTION_SIGN_INPUT.get(p.strip(), p) if i else p
               for i, p in enumerate(parts)]
        return f"{base}{adj}_{{{','.join(out)}}}"

    return _OP_SUBSCRIPT_RE.sub(_one, latex)


def canon_spin_arrows(text: str) -> str:
    r"""Collapse every arrow spelling in *text* to its canonical name.

    Third restatement of this table, removed: the NL subscript normaliser and
    the LaTeX→canonical route each carried their own copy, which is exactly what
    this module's docstring forbids.  Longest spelling first, so `\uparrow` is
    not eaten by a shorter key.
    """
    for spelling, name in sorted(SPIN_ARROW_INPUT.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(spelling, name)
    return text


def canon_spin_label(label: str) -> "str | None":
    """Return the single-character canonical for a known spin label,
    or None when *label* is not a spin label (callers decide the fallback:
    the NL route truncates to the first character, the generator refuses
    multi-character names with a reason)."""
    return SPIN_LABEL_CANON.get(label.lower())


#: What QuantumAlgebra accepts as a symbolic index: ONE unicode character,
#: optionally followed by an integer subindex.  Straight from its README
#: ("symbolic indices must be a single unicode character, with possibly an
#: integer subindex", example ``f'(1,2,:i_9)`` → ``f†(12i₉)``) and confirmed
#: against the running session on 2026-08-01:
#:
#:     QuIndex("k")   → k        QuIndex("k1")    → ArgumentError
#:     QuIndex("k_1") → k₁       QuIndex("omega") → ArgumentError
#:     QuIndex("i_9") → i₉       param(:G_0,"k","ω") → G_0(kω)
#:
#: A plain ``len(name) > 1`` check — which is what this was first written as —
#: refuses ``k_1`` and ``i_9``, both of which QA renders happily.  Note the
#: last line: a GREEK LETTER is a single character, so ``ω`` is a valid index
#: even though the spelled-out ``omega`` is not.
_QA_INDEX_RE = re.compile(r"^.(?:_\d+)?$", re.UNICODE)


def qa_index_name(name: str) -> str:
    """Canonicalise an index name and enforce QuantumAlgebra's index rule.

    The shared half of ``qa_index_literal`` and ``qa_index_symbol``: QA applies
    the same restriction whether the index arrives as ``QuIndex("s")`` or as
    the Julia symbol ``:s`` handed to ``param`` / an operator constructor.
    """
    canon = canon_spin_label(name)
    if canon is not None:
        name = canon
    if not _QA_INDEX_RE.match(name):
        from notebook.runtime.shared_parser import UnknownNotationError
        raise UnknownNotationError(
            "parser.unsupported_notation.julia_multichar_index", index=name)
    return name


def qa_index_symbol(name: str) -> str:
    """Render an index as the Julia symbol literal ``:name``.

    Used where QA takes indices as symbols rather than ``QuIndex`` values —
    ``param(:xi, :k)`` and operator constructors.  Measured 2026-08-01, a
    declared ``G_0(k, omega)`` lowered to ``param(:G_0, :k, :omega)`` and Julia
    answered ``ArgumentError: index names must be single character``: the same
    rule, reached by a different door, so it is enforced with the same check.
    """
    return f":{qa_index_name(name)}"


def qa_index_literal(name: str, neg: bool = False) -> str:
    """Render a QuantumAlgebra index literal, enforcing QA's 1-char rule.

    Lives here rather than in ``JuliaCodeGenerator`` because it IS the index
    spelling rule, not code-generator structure — the same reason
    ``subscript_policy`` holds the subscript rules (SUBSCRIPT-POLICY-SPLIT).
    Further index-spelling discriminators go here, beside the table they read.

    Spin labels that arrive multi-character from Greek/word normalization
    (``sigma``, ``up``, ``down``, …) collapse through the shared table above —
    the same table the NL route's ``_canon_spin`` uses, so the two routes
    cannot drift apart (the sigma leak existed precisely because only the NL
    route had the rule).  Any other multi-character name is refused with a
    reason BEFORE the CAS answers: measured 2026-07-31, ``QuIndex("sigma")``
    raises ``ArgumentError: index names must be single character`` — raw CAS
    text in the researcher's face (Core Principle #1).

    Negation follows the generator's existing convention: QuIndex has no
    minus, so a negated single-letter index is encoded uppercase
    (``-k`` → ``K``).
    """
    name = qa_index_name(name)
    if neg and name[:1].isalpha():
        # Uppercase the LETTER, not the whole string: `k_1`.isalpha() is False,
        # so an `.upper()` on the whole name left `-k_1` as `k_1` — the minus
        # silently gone.  Unreachable while the rule refused sub-indexed names;
        # allowing `k_1` (which QA renders as `k₁`) is what made it reachable.
        name = name[:1].upper() + name[1:]
    return f'_QA.QuIndex("{name}")'


def qa_join_index_symbols(idx_strs: "list[str | None]",
                          bound: "list[str]") -> str:
    """Join `:name` index strings that are ABOUT TO BE HANDED TO QA.

    The check lives at the JOIN, not where the strings are built: the builder
    (`_plain_sym_as_index`) is called speculatively for every function call's
    arguments, including ordinary maths where the argument is not an index at
    all — checking there refused `cos(theta)` (measured 2026-08-01).  By the
    time this runs the caller has decided the callee really is a QA operator
    or a `param`, so a multi-character name here genuinely would become an
    index and QA genuinely would reject it.

    Bound indices are exempt: they never reach QuIndex, because
    `_nb_eval_index_sum` replaces `:name` with sumindex(n) by matching the name.
    """
    for s in idx_strs:
        if s and s.startswith(":") and s[1:] not in bound:
            qa_index_name(s[1:])          # raises with the reason
    return ", ".join(s for s in idx_strs if s is not None)
