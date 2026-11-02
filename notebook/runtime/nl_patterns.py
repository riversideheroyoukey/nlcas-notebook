"""Loader for natural-language *input* grammar patterns.

The notebook accepts natural-language operation commands ("E2 の \\rho に …を代入して",
"In E2, substitute … for \\rho").  These patterns are language-specific *content* and
— exactly like user-visible output strings — must NOT be hardcoded in source.  They
live in ``i18n/nl_patterns.json``, keyed by operation, with one regex per language.

Word-order differences between languages (Japanese SOV with particles vs English SVO)
are absorbed by **named capture groups**: every language's regex for an operation uses
the same group names (``(?P<eid>…)``, ``(?P<lhs>…)``, ``(?P<rhs>…)``, …), so the calling
code reads ``m.group("eid")`` without knowing the language.

Regex flags are embedded inline at the start of the pattern string in the JSON
(``(?i)`` ignore-case, ``(?s)`` dotall); ``re.UNICODE`` is the default for str patterns.

**Shared sub-patterns** live under the ``_fragments`` key and are spliced into a
pattern by writing ``{{name}}``.  A fragment answers ONE question in ONE place:
"how may a researcher spell a symbol name?" was answered three different ways —
``diff_rule`` took ``\\xi_{k}`` but not ``x_{1}``, while ``def_integral`` and
``taylor_expand_nl`` took neither (open_items INTEGRAL-VAR-BRACED, measured
2026-08-06).  An undefined ``{{name}}`` is a hard error, never a literal left in
the regex: that would compile fine and simply never match, which is the silent
failure the fragment exists to remove.

Usage::

    from notebook.runtime.nl_patterns import nlp
    m = nlp("substitute_into", lang).match(text)
    if m:
        eid, lhs, rhs = m.group("eid"), m.group("lhs"), m.group("rhs")
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent.parent / "i18n" / "nl_patterns.json"


@lru_cache(maxsize=1)
def _load() -> dict:
    import json
    return json.loads(_PATH.read_text(encoding="utf-8"))


_FRAGMENT_REF_RE = re.compile(r"\{\{(\w+)\}\}")

#: A fragment may reference a fragment; more nesting than this is a cycle.
_MAX_FRAGMENT_DEPTH = 8


def _expand_fragments(src: str, key: str, lang: str) -> str:
    """Splice ``{{name}}`` references in ``src`` from the ``_fragments`` table.

    The replacement is done with a FUNCTION, not a string: `re.sub` processes
    backslash escapes in a string replacement, and these fragments are regexes
    made mostly of backslashes.

    **Expanded to a fixed point, and any leftover reference is an ERROR.**
    ``re.sub`` does not rescan what it substituted, so with a fragment that
    references a fragment a single pass could leave the inner ``{{name}}``
    standing as a LITERAL — which compiles and never matches, the silent
    failure this module's docstring says cannot happen.

    To be accurate about what this loop did and did not fix: the defect measured
    on 2026-08-27 — ``ja_op_var_tail`` referencing the number table (then
    ``ja_digit``, merged into ``ordinal_token`` on 2026-09-10), the 2a-sym guard
    silently not firing, a differentiation variable rewritten to an E-ID
    (``var: E5`` for ``var: x``), caught by the instruction corpus — was
    ``fragment()`` not calling this function at all.  One pass is enough for the
    one level of nesting that exists today; mutating the loop back to a single
    ``sub`` fails no test, and mutating ``fragment()`` back fails eleven.  The
    loop and the leftover check are here so that the NEXT level cannot fail
    quietly: without them it would compile and never match.
    """
    frags = _load().get("_fragments", {})

    def _one(m: re.Match) -> str:
        name = m.group(1)
        if name not in frags:
            raise KeyError(
                f"nl_patterns.json[{key!r}][{lang!r}] references unknown "
                f"fragment {{{{{name}}}}}; known: {sorted(frags)}")
        return frags[name]

    out = src
    for _ in range(_MAX_FRAGMENT_DEPTH):
        expanded = _FRAGMENT_REF_RE.sub(_one, out)
        if expanded == out:
            break
        out = expanded
    if _FRAGMENT_REF_RE.search(out):
        raise ValueError(
            f"nl_patterns.json[{key!r}][{lang!r}]: fragment references still "
            f"unexpanded after {_MAX_FRAGMENT_DEPTH} passes — a cycle? {out!r}")
    return out


@lru_cache(maxsize=None)
def nlp(key: str, lang: str = "ja") -> re.Pattern:
    """Return the compiled NL pattern for ``key`` in ``lang`` (falls back to ja)."""
    entry = _load()[key]
    src = entry.get(lang) or entry.get("ja")
    if src is None:
        raise KeyError(f"nl_patterns.json[{key!r}] has no {lang!r} or ja pattern")
    return re.compile(_expand_fragments(src, key, lang))


def has(key: str) -> bool:
    """True if an NL pattern is defined for ``key`` (any language)."""
    return key in _load()


def fragment(name: str) -> str:
    """The raw source of the shared sub-pattern ``name``.

    For the one shape ``nlp()`` cannot serve: a pattern that has to be built
    around text known only at RUNTIME — a researcher's own symbol name, escaped
    and spliced in (``supervisor_alias_rewrite``).  A ``{{name}}`` reference
    can only be expanded inside a stored pattern, and there is nowhere in the
    JSON to put "and here goes whatever the researcher called it".

    Use ``nlp()`` for everything else: a caller that reaches for this when the
    pattern is fully known is putting half of it back in Python.
    """
    frags = _load().get("_fragments", {})
    if name not in frags:
        raise KeyError(f"nl_patterns.json has no fragment {name!r}; "
                       f"known: {sorted(frags)}")
    return _expand_fragments(frags[name], "_fragments", name)


#: The languages a pattern may be written in, in the order they are tried.
_LANGS = ("ja", "en")


def patterns_all(key: str) -> list[re.Pattern]:
    """Every DISTINCT compiled pattern for ``key``, ``ja`` first.

    ``nlp(key, "en")`` falls back to the ``ja`` entry when the key has no English
    sibling, so a naive loop over the languages hands back the SAME pattern
    twice.  For a predicate that is harmless; for ``sub`` it is not, because a
    ``$``-anchored strip is not idempotent — measured, ``verb_tail_strip``
    applied twice turns ``計算して求めて`` into ``計算`` where the caller asked for
    ``計算して``.  Dedup here, once, rather than in each caller.
    """
    out: list[re.Pattern] = []
    seen: set[str] = set()
    for lang in _LANGS:
        p = nlp(key, lang)
        if p.pattern not in seen:
            seen.add(p.pattern)
            out.append(p)
    return out


def nlp_any(key: str, text: str) -> bool:
    """Does *text* match ``key`` in ANY language?

    For the questions that are asked BEFORE the session language is known, or by
    code that has no reliable language at all — "is this a declaration?", "which
    type does it name?", "which set of numbers is this?".  Narrowing one of those
    by language is the defect recorded in
    ``backend_handler._nl_any``: a handler with no engine bound answers
    ``DEFAULT_LANG`` and drops the other language's spelling silently.
    """
    return any(p.search(text or "") for p in patterns_all(key))


def nlp_search(key: str, text: str):
    """The first match of ``key`` in any language, or None (same rule as above)."""
    for p in patterns_all(key):
        m = p.search(text or "")
        if m:
            return m
    return None


def nlp_sub(key: str, text: str, repl: str = "") -> str:
    """``text`` with every language's spelling of ``key`` replaced by ``repl``.

    Each DISTINCT pattern is applied once — see ``patterns_all`` for why that
    matters here and not for the predicates.
    """
    out = text
    for p in patterns_all(key):
        out = p.sub(repl, out)
    return out


def pick_group(m, stem: str):
    """alternation のどの枝が一致したかを、**接尾辞を列挙せずに**読む。

    Python の `re` は alternation をまたいだ同名グループを禁じるので、語順の
    違う枝はそれぞれ `expr` / `expr2` … と綴り分けるしかない。読む側でその
    接尾辞を書き並べると**枝を足すたびに2箇所を揃える**形になる——
    `interpret_rules._sr_pick` が `A`〜`D` を並べていたのがそれで、2026-09-05 に
    2つめ（テイラー展開の語順違い）が要ったので1箇所に出した。
    値が入っている枝はどのみち1つなので、接尾辞は問わない。
    """
    gd = m.groupdict()
    if gd.get(stem):
        return gd[stem]
    for name, val in gd.items():
        if val and name.startswith(stem) and name[len(stem):].isalnum():
            return val
    return None


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
