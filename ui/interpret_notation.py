"""R-8: ambiguous-notation candidates + confirmation (|X|, undeclared Δ).

Pure rule-based parsers extracted from interpret_pipeline.py (R-10b).
run_interpret_pipeline imports these back; behaviour is unchanged.
"""
from __future__ import annotations

import re

from notebook.runtime.latex_converter import unreadable_braket
from notebook.runtime.matrix_environment import is_norm_environment_text
from functools import partial

from notebook.runtime.i18n import t

from ui.interpret_common import _is_nl_instruction


_PIPE_SINGLE_RE = re.compile(r'^(?P<pre>[^|]*)\|(?P<inner>[^|]+)\|(?P<post>[^|]*)$')
_PIPE_TOKEN_KEY = "|·|"

#: A DOUBLE bar — the norm.  `\|` is LaTeX's `‖`, and `||x||` spells the same
#: thing with two singles.  Neither is abs and neither is a determinant, so the
#: abs/determinant dialog must not claim them: measured 2026-08-17, `\|v\|`
#: matched `_PIPE_SINGLE_RE` (its backslashes fall into `pre` and `inner`) and
#: offered exactly the two wrong readings, then rewrote to `\abs(v\)`, which the
#: converter reduced to the unbalanced `\abs(v` — silently.
_NORM_BARS_RE = re.compile(r'\\\||\|\|')

#: A bra-ket delimiter.  The bar here separates a bra from a ket and is not a
#: fence around a quantity at all, so neither reading applies.
_BRAKET_RE = re.compile(r'\\langle|\\rangle|⟨|⟩')

#: Two or more SIMPLE bar pairs: `|x| + |y|`, `|A| |B|`.  Ordinary mathematics
#: that `_PIPE_SINGLE_RE` cannot see, because it requires exactly one pair
#: (open_items ABS-PIPE-RESIDUAL).  Pairing is left to right, which is the
#: conventional reading when no bar is doubled and none is nested.
_PIPE_PAIR_RE = re.compile(r'\|(?P<inner>[^|]+)\|')


async def try_norm_notation(engine, loop, lang, raw, registered) -> "dict | None":
    r"""二重縦棒と `\begin{Vmatrix}` を **`norm` 操作**へ配線する。

    `\|v\|` / `||v||` / `\begin{Vmatrix}…\end{Vmatrix}` は「中身のノルムを求めよ」
    という要求である。**記法から操作への配線だけが無かった**（起票 NORM-OPERATION、
    2026-09-10 に操作を追加し、配線をここで足した）。

    **`=` が入っていたら claim しない。** `\|v\| = 5` は**述べた式**であって要求では
    なく、`E2 = \|v\|` は定義である——括弧の族で先に立てた規則と同じ
    （CLAUDE.md「A BRACKET FOLLOWED BY `=` IS A RELATION, NOT A REQUEST」）。
    判定は `find_top_level_relation`——この系で「関係式か」を答える唯一の走査。

    **単一の縦棒には触らない。** `|x|` が絶対値か行列式かは**まだ決まっていない
    問い**で、答える相手は研究者である（下の候補ダイアログ）。`norm_notation` は
    `\|` と `||` だけを綴っているので、ここには当たらない。
    """
    from notebook.runtime.matrix_environment import norm_environment_operand
    from notebook.runtime.nl_patterns import nlp_search, pick_group
    from notebook.runtime.shared_parser import find_top_level_relation
    from notebook.runtime.llm_supervisor import _norm_kind_written

    text = (raw or "").strip()
    if not text:
        return None
    operand = norm_environment_operand(text)
    if operand is None:
        m = nlp_search("norm_notation", text)
        operand = (pick_group(m, "inner") or "").strip() if m is not None else ""
    if not operand:
        return None
    # 述べた式は要求ではない（`\|v\| = 5` / `E2 = \|v\|`）。
    if find_top_level_relation(text) is not None:
        return None

    # **オペランドは、式番号・登録済みの名前・書かれた式のどれでも来る。**
    # 前2つはそのまま渡せる（`norm` が `defining_expr_id` から値を取る）。
    # 書かれた式は研究者の LaTeX なので、式の扉を通してから渡す。
    if operand not in engine.expressions and (
            engine.symbol_context.get_quantity(operand) is None):
        operand = engine._handler.latex_to_canonical(operand) or operand

    args = {"expr": operand}
    kind = _norm_kind_written(text)
    if kind:
        args["kind"] = kind
    try:
        step, val = await loop.run_in_executor(
            None, partial(engine.run_operation, "norm", args))
    except Exception as exc:                       # 断りは断りとして返す
        return {"status": "failed", "error": str(exc),
                "registered_expressions": registered}
    _ex = engine.expressions.get(getattr(step, "result_expr_id", None) or "")
    return {"status": "completed", "operation": "norm",
            "expr_id": getattr(_ex, "display_id", None),
            "latex": getattr(_ex, "latex", None),
            "registered_expressions": registered}


def _pipe_shape(stripped: str) -> str:
    """Which pipe notation is this?  ``norm`` | ``braket`` | ``pairs`` | ``other``.

    Split out because the abs/determinant dialog used to claim everything with a
    single bar pair in it, including two spellings where BOTH of its choices are
    wrong.  Deciding the shape first is what lets each get its own answer.
    """
    if _NORM_BARS_RE.search(stripped):
        return "norm"
    if _BRAKET_RE.search(stripped):
        return "braket"
    bars = stripped.count("|")
    if bars >= 4 and bars % 2 == 0:
        return "pairs"
    return "other"
_BARE_DELTA_RE = re.compile(r'(?<![A-Za-z_])\\?Delta(?![A-Za-z_0-9])')
def _rewrite_pipe_in_tokens(req_tokens, fn: str) -> None:
    """In-place: replace the single |inner| pair with fn(inner) in each token's
    computable field.  Math tokens keep their original latex (display stays
    |A|); only the canonical becomes the explicit function call."""
    for tok in req_tokens:
        for field in ("canonical", "value"):
            s = getattr(tok, field, "") or ""
            if "|" not in s:
                continue
            m = _PIPE_SINGLE_RE.match(s.strip())
            if m:
                setattr(tok, field,
                        f"{m.group('pre')}{fn}({m.group('inner').strip()}){m.group('post')}")
async def _try_notation_candidates(engine, loop, lang, text, req_tokens,
                                   registered) -> "tuple[dict | None, str]":
    """Returns (response, updated_raw_text).  response is None when nothing
    fired (possibly after a SILENT rewrite of the tokens/raw text)."""
    raw = text or ""
    stripped = raw.strip()
    # **ここには別名の表の証拠を渡さない**（2026-09-10、実測して決めた）。
    # 他の gate（H-1・`expr_id_formula`・素の式）は「操作を名指しているか」を
    # 足すと silent-wrong が消えるが、ここは逆だった:
    #
    #   `the determinant of |E1|`
    #      渡さない → 縦棒の意味を訊くダイアログ（研究者が答えられる）
    #      渡す     → ノルム／絶対値の断り（**答えられる問いを断りに変える**）
    #
    # 縦棒が絶対値か行列式かは**この文が操作を名指していても未決**で、
    # 訊く相手は研究者である。
    if not stripped or _is_nl_instruction(stripped, lang):
        return None, raw
    ctx = engine.symbol_context

    # ノルムは**綴りが2つ**ある: 二重縦棒（下で読む）と `\begin{Vmatrix}`。
    # 後者は縦棒の**文字**を持たないので下の分岐に届かず、一般の
    # 「解釈できません」——理由も代わりの書き方も出ない断り——に落ちていた
    # （実測 2026-09-10、TEX-MATRIX-INPUT）。綴りの表は環境を知っている側
    # （`matrix_environment`）に置き、**断りの判断と文言はここ1箇所**のまま。
    if is_norm_environment_text(stripped):
        return ({"status": "failed",
                 "error": t("notation.pipe_norm", lang),
                 "registered_expressions": registered}, raw)

    # ── |X| : absolute value vs determinant ────────────────────────────
    # `\langle \Phi_0 | A | \Phi_0 \rangle` —— 両側が同じ状態なら
    # `\langle A \rangle_{\Phi_0}` と同じ量で、変換器が読む。**分岐の入り口で
    # 外す**こと: 中で shape だけ変えると、縦棒の対として絶対値／行列式ダイアログに
    # 落ちる（実測 2026-08-20、`\langle\Phi_0| T[…]|\Phi_0\rangle` が
    # `abs(T[…])` を提案してきた——2026-08-17 にノルムで塞いだのと同じ穴）。
    if "|" in stripped and unreadable_braket(stripped):
        shape = _pipe_shape(stripped)
        if shape in ("norm", "braket"):
            # BOTH of this dialog's choices are wrong for these, so it must not
            # claim them — measured 2026-08-17, `\|v\|` WAS claimed, offered
            # abs/determinant, and rewrote to the unbalanced `\abs(v`, silently.
            #
            # Refused HERE rather than left to fall through, because the generic
            # `unsupported_notation.pipe` (raised character-by-character in the
            # parser) cannot know which spelling this is, and the two want
            # different things said.  A refusal whose reason does not fit the
            # input is what this project has spent the day removing.
            return ({"status": "failed",
                     "error": t(f"notation.pipe_{shape}", lang),
                     "registered_expressions": registered}, raw)
        if shape == "pairs":
            # `|x| + |y|` — ordinary mathematics the single-pair regex cannot
            # see.  ONE decision covers every pair: the bars in one expression
            # are the same notation, and asking per pair would ask the same
            # question twice.  A remembered answer applies here too.
            inners = [mm.group("inner").strip()
                      for mm in _PIPE_PAIR_RE.finditer(stripped)]
            role = ctx.notation_usage.get(_PIPE_TOKEN_KEY)
            if role not in ("abs", "determinant"):
                role = None
                props = [ctx.get(i) for i in inners]
                algs = {p.quantity.math_type.algebra for p in props if p is not None}
                if len(algs) == 1 and len(props) == len([p for p in props if p]):
                    role = ("determinant" if algs.pop() in ("matrix", "tensor")
                            else "abs")
            if role:
                fn = "abs" if role == "abs" else "determinant"
                _rewrite_pipe_in_tokens(req_tokens, fn)
                return None, _PIPE_PAIR_RE.sub(
                    lambda mm: f"{fn}({mm.group('inner').strip()})", stripped)

            def _pairs_cand(role_: str, fn_: str) -> dict:
                rewritten = _PIPE_PAIR_RE.sub(
                    lambda mm: f"{fn_}({mm.group('inner').strip()})", stripped)
                return {
                    "operation": "resolve_notation",
                    "args": {"token": _PIPE_TOKEN_KEY, "role": role_,
                             "rewritten": rewritten, "latex": stripped},
                    "display_ja": t(f"notation.pipe_{role_}_choice", "ja",
                                    inner=", ".join(inners)),
                    "display_en": t(f"notation.pipe_{role_}_choice", "en",
                                    inner=", ".join(inners)),
                }
            cands = [_pairs_cand("abs", "abs"),
                     _pairs_cand("determinant", "determinant")]
            return ({
                "status": "needs_confirmation",
                "operation": cands[0]["operation"],
                "args": cands[0]["args"],
                "confirmation_prompt": t("notation.pipe_prompt", lang,
                                         token=" , ".join(f"|{i}|" for i in inners)),
                "candidates": cands,
                "registered_expressions": registered,
                "_user_text": stripped,
            }, raw)
        m = _PIPE_SINGLE_RE.match(stripped)
        if not m:
            return None, raw  # nested/odd bars → R-3 explicit error path
        inner = m.group("inner").strip()
        role = ctx.notation_usage.get(_PIPE_TOKEN_KEY)
        if role not in ("abs", "determinant"):
            role = None
            prop = ctx.get(inner)  # declaration narrows it down (priority 1)
            if prop is not None:
                alg = prop.quantity.math_type.algebra
                role = "determinant" if alg in ("matrix", "tensor") else "abs"
        if role:
            fn = "abs" if role == "abs" else "determinant"
            _rewrite_pipe_in_tokens(req_tokens, fn)
            new_raw = f"{m.group('pre')}{fn}({inner}){m.group('post')}"
            return None, new_raw

        def _pipe_cand(role_: str, fn_: str) -> dict:
            rewritten = f"{m.group('pre')}{fn_}({inner}){m.group('post')}"
            return {
                "operation": "resolve_notation",
                "args": {"token": _PIPE_TOKEN_KEY, "role": role_,
                         "rewritten": rewritten, "latex": stripped},
                "display_ja": t(f"notation.pipe_{role_}_choice", "ja", inner=inner),
                "display_en": t(f"notation.pipe_{role_}_choice", "en", inner=inner),
            }
        cands = [_pipe_cand("abs", "abs"), _pipe_cand("determinant", "determinant")]
        return ({
            "status": "needs_confirmation",
            "operation": cands[0]["operation"],
            "args": cands[0]["args"],
            "confirmation_prompt": t("notation.pipe_prompt", lang,
                                     token=f"|{inner}|"),
            "candidates": cands,
            "registered_expressions": registered,
            "_user_text": stripped,
        }, raw)

    # ── undeclared Δ in a formula: Laplacian vs symbol ──────────────────
    # Fires ONLY when the active domains cannot narrow it down (condition 4);
    # with any domain set, the existing behavior (resolver defaults) stands.
    if (_BARE_DELTA_RE.search(stripped)
            and ctx.get("Delta") is None
            and not getattr(engine.session, "domains", [])):
        role = ctx.notation_usage.get("Delta")
        if role == "laplacian":
            return None, raw  # remembered: resolver's default IS the Laplacian
        if role == "symbol":
            # remembered as a symbol but the declaration is gone — restore it
            await loop.run_in_executor(
                None, partial(engine.register_quantity, "Delta", kind="scalar"))
            return None, raw
        if role is None:
            def _delta_cand(role_: str, extra: "dict | None" = None) -> dict:
                args = {"token": "Delta", "role": role_,
                        "rewritten": stripped, "latex": ""}
                if extra:
                    args.update(extra)
                return {
                    "operation": "resolve_notation",
                    "args": args,
                    "display_ja": t(f"notation.delta_{role_}_choice", "ja"),
                    "display_en": t(f"notation.delta_{role_}_choice", "en"),
                }
            cands = [_delta_cand("laplacian"),
                     _delta_cand("symbol", {"declare": "Delta"})]
            return ({
                "status": "needs_confirmation",
                "operation": cands[0]["operation"],
                "args": cands[0]["args"],
                "confirmation_prompt": t("notation.delta_prompt", lang),
                "candidates": cands,
                "registered_expressions": registered,
                "_user_text": stripped,
            }, raw)

    # ── C_i(t) : the subscript is an INDEX vs part of the NAME ──────────
    cand = _applied_subscript_candidates(engine, lang, stripped, registered)
    if cand is not None:
        return cand, raw

    return None, raw


#: ``X_i(...)`` — a base, a SINGLE ALPHABETIC subscript, applied to arguments.
#: Digits are excluded on purpose: the project's own subscript-classification
#: rule already says "digit only → always part of the name" (``omega_0`` is
#: omega-naught), so ``G_0(k,omega)`` is settled and asking would be noise.
_APPLIED_SUB_RE = re.compile(
    r"(?<![A-Za-z0-9_\\])(?P<base>[A-Za-z][A-Za-z0-9]*)"
    r"_(?:(?P<bare>[A-Za-z])|\{(?P<braced>[A-Za-z])\})\s*\(")


def _applied_subscript_candidates(engine, lang, stripped, registered):
    r"""``C_i(t)`` — ask whether ``i`` is an index or part of the name.

    This is the one case AUDIT-INDEXED-CALL leaves genuinely undecidable:
    ``C_i(t)`` wants an index and ``p_x(x)`` wants a label, and they are the
    same shape.  Everything the researcher may already have said is consulted
    FIRST (subscript_policy's order), so the dialog is the last resort:

      builtin family → sum binding → base declaration → prior answer → ASK

    **Both answers work**, which is what makes this askable at all.  docs/32
    Phase 4 ③ withdrew an earlier type dialog because one branch silently broke
    summation; since Maxima's subscripted function landed, "index" gives
    ``C[i](t)`` and "name" gives ``C_i(t)`` and neither loses anything.
    """
    from notebook.runtime.subscript_policy import APPLIED_SUBSCRIPT_KEY
    ctx = engine.symbol_context
    m = _APPLIED_SUB_RE.search(stripped)
    if not m:
        return None
    base = m.group("base")
    sub = m.group("bare") or m.group("braced")
    full = f"{base}_{sub}"

    # already settled — never ask twice, and never ask about a CAS function
    if ctx.notation_usage.get(f"{APPLIED_SUBSCRIPT_KEY}{base}"):
        return None
    if ctx.get(full) is not None or ctx.get(base) is not None:
        return None
    resolver = getattr(engine._handler, "_resolver_for_notation", None)
    known = getattr(resolver, "known_function_names", None)
    if known is None:
        from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
        known = MaximaSemanticResolver.known_function_names
    if full in known:
        return None
    # bound by an enclosing sum → subscript_policy already answers "index"
    if re.search(r"\\sum_\{?" + re.escape(sub) + r"\}?(?![A-Za-z0-9])", stripped):
        return None

    def _cand(role: str) -> dict:
        return {
            "operation": "resolve_notation",
            "args": {"token": f"{APPLIED_SUBSCRIPT_KEY}{base}", "role": role,
                     "rewritten": stripped, "latex": ""},
            "display_ja": t(f"notation.applied_sub_{role}_choice", "ja",
                            base=base, sub=sub),
            "display_en": t(f"notation.applied_sub_{role}_choice", "en",
                            base=base, sub=sub),
        }

    cands = [_cand("index"), _cand("name")]
    return {
        "status": "needs_confirmation",
        "operation": cands[0]["operation"],
        "args": cands[0]["args"],
        "confirmation_prompt": t("notation.applied_sub_prompt", lang,
                                 token=full, base=base, sub=sub),
        "candidates": cands,
        "registered_expressions": registered,
        "_user_text": stripped,
    }
