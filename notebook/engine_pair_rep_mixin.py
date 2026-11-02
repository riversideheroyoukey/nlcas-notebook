r"""ハイゼンベルク描像の演算子を、係数⊗演算子の対として作る（PAIR-REP 第1スライス）。

設計と根拠は :mod:`notebook.runtime.pair_representation` の docstring と
docs/33 §5–§6・docs/34 §2 にある。ここに置いてあるのは**エンジン側の配線**だけ:
CAS に判定を投げ、返ってきた対を記述子にして登録する。

`register_integral_form` / `register_bond_sum` と同じ「遅延評価の記述子」経路を
使う——実体は構造で、表示は利用者の記法である。
"""
from __future__ import annotations

from notebook.models.expression import Expression
from notebook.runtime.i18n import t
from notebook.runtime.pair_representation import (
    CLOSURE_CAP, REASON_NOT_CLOSED, REASON_NOT_CONCRETE, closure_is_concrete, build_pair_representation,
    coefficient_product, coefficient_sum, is_unity, is_vanishing,
    matrix_entries, name_to_subscript_latex, pair_display_latex, parse_closure,
    parse_exponential_row, pair_terms_of, qa_to_input, terms_from_row,
)


class EnginePairRepMixin:
    """`NotebookEngine` の PAIR-REP 部分。"""

    def _display(self, latex: str) -> str:
        r"""係数・演算子の表示を、**このセッションの綴り**に揃える。

        係数はスカラー側（SymPy）から来るので、研究者が `\varepsilon` と書いたか
        `\epsilon` と書いたかを知らない——`\epsilon_{m}` と返してくる。表示ブリッジは
        その記録を持っている（PHI-GLYPH-SPLIT）ので、そこを通す。
        """
        return self._handler.normalize_display_latex(
            self._handler._apply_display_bridge(latex, self.symbol_context))

    def heisenberg_operator(self, operator: str, hamiltonian_ref: str,
                            time_var: str = "t") -> Expression:
        r"""``A(t) = e^{iHt}Ae^{-iHt}`` を (係数, 演算子) の対として登録する。

        二段構えだが、問いは1つずつである。**QA には閉包を訊く**——`ad_H` の連鎖が
        どの演算子で閉じ、その行列は何か。**スカラー側には指数を訊く**——
        `exp(itM)` の第1行。QA に係数を組み立てさせないのは、c 数を分解して
        しまうからで（docs/33 §6.1）、そこが対表現の要点そのものである。

        対角なら 1×1 で `e^{-iε_m t}`、非対角な二次形式なら 2×2 で
        `cos(wt)c_1 - i sin(wt)c_2`、原子極限の相互作用も閉じる（実測）。
        上限（`CLOSURE_CAP`）まで閉じなければ理由付きで断る——打ち切った級数は
        短時間展開であって摂動論ではない（docs/33 §6.2）。
        """
        lookup = {k: v.canonical for k, v in self.expressions.items()}
        build = self._handler.build_command(
            "heisenberg_operator",
            {"operator": operator, "hamiltonian": hamiltonian_ref,
             "time": time_var}, lookup)
        if not build.ok:
            raise ValueError(build.error)
        walked = self._adapter.execute(build.command)
        if not (walked.ok and walked.value):
            raise ValueError(walked.value or "CAS error")
        closure = parse_closure(walked.value)
        if closure is None:
            raise ValueError(t(REASON_NOT_CLOSED, self.lang, operator=operator,
                               hamiltonian=hamiltonian_ref, cap=CLOSURE_CAP))
        keys, rows = closure
        entries = matrix_entries(keys, rows)
        if not closure_is_concrete(keys, entries):
            raise ValueError(t(REASON_NOT_CONCRETE, self.lang,
                               operator=operator, hamiltonian=hamiltonian_ref))
        command = self._handler.pair_exponential_command(
            entries, len(keys), time_var)
        exponential = self._adapter.execute(command)
        if not (exponential.ok and exponential.value):
            raise ValueError(exponential.value or "CAS error")
        # 第1基底は研究者が書いたもの——その綴りのまま持つ。残りは QA の印字なので
        # 入力として読み直せる形に戻す（`†` → `'`）。
        operators = [operator] + [qa_to_input(k) for k in keys[1:]]
        terms = terms_from_row(parse_exponential_row(exponential.value),
                               keys, operators)
        for _term in terms:
            _term["coefficient_latex"] = self._display(_term["coefficient_latex"])
        if not terms:
            raise ValueError(t(REASON_NOT_CLOSED, self.lang, operator=operator,
                               hamiltonian=hamiltonian_ref, cap=CLOSURE_CAP))
        descriptor = build_pair_representation(
            terms=terms, time_var=time_var,
            source_latex=self._handler.normalize_display_latex(
                name_to_subscript_latex(operator)),
            hamiltonian_ref=hamiltonian_ref,
        )
        return self.register_direct(pair_display_latex(terms),
                                    pair_representation=descriptor)

    def pair_bracket(self, kind: str, a_id: str, b_id: str) -> Expression:
        r"""``{A(t), B(t')}`` —— **係数の積 × 等時刻代数**。

        対表現が構造的に正しいことの見どころ（docs/34 §2）。演算子部の代数は既存の
        ``commutator`` / ``anticommutator`` がそのまま答え、この層がするのは
        係数を掛けて足すことだけである。多項どうしなら項の組み合わせすべての和に
        なり、**同じ演算子に落ちた係数はスカラー側で簡約される**——2サイト模型の
        ``{c_1(t), c_1^\dagger(u)}`` が ``cos(w(t-u))`` になるのはそこで、
        異時刻の反交換関係が**時間差だけの関数**になるという性質が出る。

        等時刻代数が c 数を返すと結果は**演算子を含まないスカラー**になり、
        SCALAR-HANDOFF がその先の境界を運ぶ。
        """
        a, b = self.expressions[a_id], self.expressions[b_id]
        terms_a = pair_terms_of(a.pair_representation, latex=a.latex,
                                canonical=a.canonical)
        terms_b = pair_terms_of(b.pair_representation, latex=b.latex,
                                canonical=b.canonical)
        # 組み合わせが1つだけなら、等時刻代数を**タイムラインに残す**——それ自体が
        # 読みたい中間結果である（`{c,c^\dagger}=1`）。何組もあるときは残さない:
        # 見たいのは和のほうで、組ごとの行は雑音になる。
        show_steps = len(terms_a) == 1 and len(terms_b) == 1
        groups: "dict[str, list]" = {}
        for ta in terms_a:
            for tb in terms_b:
                op_canonical, op_latex = self._operator_bracket(
                    kind, ta["operator"], tb["operator"], show_steps)
                if is_vanishing(op_latex):
                    continue
                key = op_canonical
                slot = groups.setdefault(key, [op_latex, []])
                slot[1].append(coefficient_product(ta.get("coefficient", "1"),
                                                   tb.get("coefficient", "1")))
        if not groups:
            return self.register_direct("0")
        terms = []
        for op_latex, coeffs in groups.values():
            value, latex = self._simplify_scalar(coefficient_sum(coeffs))
            terms.append({
                "coefficient_latex": self._display(latex),
                "coefficient": value,
                "operator_latex": "" if is_unity(op_latex) else op_latex,
                "operator": "" if is_unity(op_latex) else op_latex,
            })
        descriptor = None
        if any(t_["operator_latex"] for t_ in terms):
            descriptor = build_pair_representation(
                terms=terms,
                time_var=(a.pair_representation or {}).get("time_var", "t"),
                source_latex=f"{a.latex} , {b.latex}",
                hamiltonian_ref=(a.pair_representation or {}).get(
                    "hamiltonian_ref", ""),
            )
        return self.register_direct(pair_display_latex(terms),
                                    pair_representation=descriptor)

    def _operator_bracket(self, kind: str, op_a: str, op_b: str,
                          record: bool) -> "tuple[str, str]":
        """演算子部だけの括弧。``record`` ならタイムラインにも残す。"""
        if record:
            step, _raw = self.run_operation(kind, {"a": op_a, "b": op_b})
            expr = self.expressions[(step.output_refs or [""])[0]]
            return expr.canonical, expr.latex
        build = self._handler.build_command(
            kind, {"a": op_a, "b": op_b},
            {k: v.canonical for k, v in self.expressions.items()})
        if not build.ok:
            raise ValueError(build.error)
        value, latex = self._cas_value_and_latex(build.command)
        return value, latex

    def _simplify_scalar(self, expr: str) -> "tuple[str, str]":
        """係数をスカラー側で簡約して ``(値, latex)``。"""
        result = self._adapter.execute(
            self._handler.scalar_simplify_command(expr))
        if not (result.ok and result.value):
            raise ValueError(result.value or "CAS error")
        value, _, latex = result.value.strip().strip('"').partition("|")
        return value.strip(), latex.strip()
