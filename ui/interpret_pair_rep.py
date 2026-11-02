"""PAIR-REP の NL 入口 —— 「A を En のもとで時間発展させて」。

ルールベース、LLM の手前。`register_integral_form` / 格子和と同じで、記述子を作る
操作はエンジンのメソッドを直接呼ぶ（`run_operation` の一般経路には乗せない——
返るのが CAS の値ではなく**構造**だからである）。

対象と時間変数以外は何も推測しない: 既定の時間変数は `t`。
"""
from __future__ import annotations

from functools import partial

from notebook.runtime.nl_patterns import nlp, pick_group


async def _try_heisenberg_operator(engine, loop, lang, text, registered) -> "dict | None":
    """`c(m) を E1 のもとで時間発展させて` / `evolve c(m) under E1`."""
    stripped = (text or "").strip()
    if not stripped:
        return None
    m = nlp("heisenberg_operator_nl", lang).search(stripped)
    if m is None:
        return None
    # 語順の枝は `pick_group` で読む——同名グループが alternation をまたげないので
    # `eid` / `eid2` … と綴り分けるしかなく、接尾辞を呼ぶ側に書き並べると
    # **枝を足すたびに2箇所を揃える**形になる（二十八問目）。
    eid = (pick_group(m, "eid") or "").strip()
    # **研究者が名前を付けたハミルトニアンは、名前で呼べること。**  交換子は
    # `E1 と S_p の交換子` を既に受けるのに、こちらは E-id だけを受けていた——
    # そして式番号は言語で揃わない（日本語の経路はオペランドも登録する）ので、
    # 番号で書いたシナリオは黙って**別の式のもとで**時間発展した（実測 2026-08-22:
    # `H_U` のつもりが `n_{1↑}` になり、`e^{-it}c_{1↑}` を成功として返した）。
    if eid not in engine.expressions:
        eid = engine.lhs_alias_map.get(eid, eid)
    if eid not in engine.expressions or eid not in engine._active_expr_ids():
        return None
    op = (pick_group(m, "op") or "").strip()
    time_var = (pick_group(m, "time") or "t").strip()
    try:
        expr = await loop.run_in_executor(
            None, partial(engine.heisenberg_operator, op, eid, time_var))
    except Exception as exc:      # CAS の失敗も「閉じない」も、理由つきの失敗として返す
        return {"status": "failed", "error": str(exc),
                "registered_expressions": registered}
    registered.append({"expr_id": expr.display_id, "canonical": expr.canonical,
                       "latex": expr.latex})
    return {
        "status": "completed",
        "operation": "heisenberg_operator",
        "expr_id": expr.display_id,
        "latex": expr.latex,
        "canonical": expr.canonical,
        "registered_expressions": registered,
    }


#: 綴り → 操作。日本語の「反交換関係／反交換子」はどちらも同じ量である。
_BRACKET_KIND = {
    "反交換関係": "anticommutator", "反交換子": "anticommutator",
    "交換関係": "commutator", "交換子": "commutator",
    "anticommutator": "anticommutator", "commutator": "commutator",
}


async def _try_pair_bracket(engine, loop, lang, text, registered) -> "dict | None":
    """`E5 と E6 の反交換関係` —— **対が絡むときだけ**claim する。

    対が絡まない普通の括弧は既存の一般経路の仕事なので、そちらへ落とす。ここが
    決めているのは「被演算子が対か」だけで、括弧の意味を二度目に定義してはいない
    （演算子部の代数は `pair_bracket` が既存の操作へそのまま渡す）。
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    m = nlp("pair_bracket_nl", lang).search(stripped)
    if m is None:
        return None
    a_id, b_id = m.group("a").strip(), m.group("b").strip()
    active = engine._active_expr_ids()
    if a_id not in active or b_id not in active:
        return None
    exprs = engine.expressions
    if not (getattr(exprs.get(a_id), "pair_representation", None)
            or getattr(exprs.get(b_id), "pair_representation", None)):
        return None                      # 対でないなら一般経路のもの
    kind = _BRACKET_KIND.get((m.group("kind") or "").lower())
    if kind is None:
        return None
    try:
        expr = await loop.run_in_executor(
            None, partial(engine.pair_bracket, kind, a_id, b_id))
    except Exception as exc:
        return {"status": "failed", "error": str(exc),
                "registered_expressions": registered}
    registered.append({"expr_id": expr.display_id, "canonical": expr.canonical,
                       "latex": expr.latex})
    return {
        "status": "completed",
        "operation": kind,
        "expr_id": expr.display_id,
        "latex": expr.latex,
        "canonical": expr.canonical,
        "registered_expressions": registered,
    }
