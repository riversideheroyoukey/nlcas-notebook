r"""係数⊗演算子の対表現（PAIR-REP、着手 2026-08-21）。

**なぜ「新しい種類のオブジェクト」を作らないのか。** ハイゼンベルク描像の演算子は

.. math:: A(t) = e^{iHt} A e^{-iHt}

であって、これは*定義に忠実に読めば* QuantumAlgebra が既に表せる演算子の
**入れ子交換子の和**である（docs/33 §5）::

    A(t) = Σ_n (it)^n/n! · ad_H^n(A)

QA には `exp(::QuExpr)` が無いのでサンドイッチは直書きできないが、`ad_H^n(A)` は
`comm` の連鎖として厳密に計算できる——**実測済み**。そして `H` が二次形式なら
その連鎖は**有限個の演算子で閉じる**ので、係数は指数関数に再総和できる。

**しかし係数は QA に置けない**（docs/33 §6.1、実測）::

    exp(-im*eps*t) * c(1)  →  (exp(0)*cos(-eps*t) + abs(exp(0)*sin(-eps*t))i) c(1)

QA は c 数を分解し、`e^{-i\varepsilon t}` のまま保持しない。だから実体は
**(スカラー係数, QA 演算子) の対の和**であって、利用者記法 `\hat c_{n}(t)` は
その表示にすぎない——`integral_descriptor` / `lattice_descriptor` と同じ
「実体は遅延評価される構造、表示は利用者の記法」パターンである。

**この対表現が構造的に正しいことの根拠**（docs/34 §2）: 異時刻の反交換関係が

.. math:: \{c(t), c^\dagger(t')\} = f(t)\overline{f(t')}\,\{c,c^\dagger\}

——**係数の積 × 等時刻代数**——として出る。時間を添字に畳む案（却下済み）が
踏んでいた `δ(t,t')` の罠を、この形は構造的に踏めない。

**この段（第1スライス）が扱う範囲。** `ad_H(A) = λA` すなわち **A が ad_H の
固有演算子**である場合を**厳密に**（打ち切りなしで）扱う::

    [H, A] = λ A   ⟹   A(t) = e^{iλt} A

対角な `H₀ = Σ_n ε_n c†_n c_n` に対する `c_m`（λ = -ε_m）と `c†_m`（λ = +ε_m）が
これで、グリーン関数の運動方程式が要求するのはこの形である。

**閉じない場合は断る。** 打ち切った Hadamard 級数は**短時間展開**であって摂動論では
なく（docs/33 §6.2）、両者を混ぜると周波数空間へ行けない。黙って級数を返すより、
理由を述べて止まるほうが正しい。非対角な二次形式（連鎖が複数の演算子で閉じる場合）は
係数が行列指数になる——第2スライス。
"""
from __future__ import annotations

import re

#: 判定できなかったときに Julia 側が返す印。
NOT_CLOSED = "NOTCLOSED"

#: ad の連鎖を追う基底の上限。超えたら「閉じない」と答える——**測っていない
#: ことを断言しない**ために、断り文はこの打ち切りを述べる。二次形式は演算子1つ
#: あたり高々モード数で閉じ、相互作用は（原子極限を除いて）閉じない。
CLOSURE_CAP = 8

#: 断りの理由（i18n キー）。
REASON_NOT_CLOSED = "pair_rep.chain_does_not_close"
REASON_NOT_CONCRETE = "pair_rep.chain_is_not_concrete"

#: QA が「まだ束縛が残っている」ことを印字で示す綴り。基底に出たら、それは
#: 1つの演算子ではなく**自由添字を持つ族**である。
_BOUND_MARKERS = ("∑", "#", "δ(")


def parse_eigenoperator_result(raw: str) -> "tuple[str, str, str] | None":
    """Julia の判定結果 ``"-1|eps(m)|c(m)"`` を ``(比, 余分な係数, 演算子)`` に。

    閉じなかった (``NOTCLOSED``) ときは None。
    """
    text = (raw or "").strip().strip('"')
    if not text or text.startswith(NOT_CLOSED):
        return None
    parts = text.split("|")
    if len(parts) != 3:
        return None
    ratio, params, operator = (p.strip() for p in parts)
    if not ratio or not operator:
        return None
    return ratio, params, operator


def eigenvalue_text(ratio: str, params: str) -> str:
    """``("-1", "eps(m)")`` → ``"-eps(m)"``、``("2", "")`` → ``"2"``。

    数値比は QA が Float で返す（`-1.0`）ことがあるので整数に見えるものは整数で
    書く——CLAUDE.md の「厳密な量に float を使わない」に合わせる。
    """
    r = _tidy_number(ratio)
    if not params:
        return r
    if r == "1":
        return params
    if r == "-1":
        return f"-{params}"
    return f"{r}*{params}"


def _tidy_number(text: str) -> str:
    """``-1.0`` → ``-1``、``2.0`` → ``2``。整数でないものはそのまま。"""
    try:
        value = float(text)
    except ValueError:
        return text
    return str(int(value)) if value.is_integer() else text


def time_factor_latex(eigenvalue: str, time_var: str) -> str:
    r"""``("-eps(m)", "t")`` → ``"e^{-i eps_{m} t}"``。

    `A(t) = e^{iλt}A` の係数。λ の符号は**指数の中へ畳む**——`e^{i(-x)t}` ではなく
    `e^{-i x t}` と書くのが物理の綴りであり、往復もこちらのほうが素直である。
    """
    lam = name_to_subscript_latex(eigenvalue.strip())
    if lam.startswith("-"):
        return f"e^{{-i {lam[1:].strip()} {time_var}}}"
    return f"e^{{i {lam} {time_var}}}"


#: ``eps(m)`` / ``c†(m)`` / ``V(p, q)`` —— QA が印字する「名前＋印＋添字」の綴り。
#: 随伴の印は QA が `†` で書く（このシステムの表示は `^{\\dagger}`）。
_CALL_RE = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_]*)([†']*)\(([^()]*)\)")


def name_to_subscript_latex(text: str) -> str:
    r"""``eps(m)`` → ``eps_{m}``、``c†(m)`` → ``c^{\dagger}_{m}``。

    QA の印字を利用者の添字記法へ戻す。研究者が書いた名前はそのまま使う——
    このセッションで `eps` と書いたなら `\varepsilon` に「直す」ことはしない
    （PHI-GLYPH-SPLIT と同じ立場）。
    """
    def _sub(m: "re.Match") -> str:
        marks = "".join("^{\\dagger}" for _ in m.group(2))
        return f"{m.group(1)}{marks}_{{{m.group(3).replace(' ', '')}}}"
    return _CALL_RE.sub(_sub, text)


def pair_display_latex(terms: "list[dict]") -> str:
    """対の和を表示 LaTeX に。``[{coefficient_latex, operator_latex}, …]``。

    **和になった係数は括弧で囲う。** 原子極限の相互作用では係数が
    ``1 - e^{-iUt}`` になり、囲まないと ``c_1 + 1 - e^{-iUt} c^†_2c_1c_2`` ——
    項が1つ増えた別の式に読める（実測 2026-08-22）。``bracket_latex`` が演算子側で
    しているのと同じことを、係数側でもする。
    """
    pieces = []
    for term in terms:
        coeff = (term.get("coefficient_latex") or "").strip()
        op = (term.get("operator_latex") or "").strip()
        if coeff and coeff != "1":
            # **両側**を囲う。係数が和なら（原子極限の `1 - e^{-iUt}`）項が増えて
            # 読め、演算子が和なら（フェルミオンの交換子 `1 - 2c^†c`）係数が
            # 第1項にしか掛からないように読める。同じ規則なので同じ場所で当てる。
            if op and _is_sum(coeff):
                coeff = _fenced(coeff)
            if coeff and _is_sum(op):
                op = _fenced(op)
            pieces.append(f"{coeff} {op}".strip())
        else:
            pieces.append(op or coeff)
    out = " + ".join(p for p in pieces if p)
    return out.replace("+ -", "- ")


def build_pair_representation(*, terms: "list[dict]", time_var: str,
                              source_latex: str, hamiltonian_ref: str) -> dict:
    """保存する記述子。``Expression.pair_representation`` に入る形。

    `NumericTask` / `ScalarHandoff` と同じ規律で、**由来をスナップショットする**——
    どの演算子を、どのハミルトニアンのもとで、どの時間変数について時間発展させた
    のかは、結果の意味の一部である。
    """
    return {
        "terms": list(terms),
        "time_var": time_var,
        "source_latex": source_latex,
        "hamiltonian_ref": hamiltonian_ref,
    }



def bracket_latex(term_a: dict, term_b: dict, operator_latex: str) -> str:
    r"""``f(t) g(t') × {A,B}`` の表示。

    等時刻代数の答えが **c 数** なら（``{c,c^\dagger}=1``）、残るのは係数の積だけで、
    それは**演算子を含まないスカラー**である——そのまま SCALAR-HANDOFF で
    スカラー CAS へ渡せる形になる。これがこの表現の見どころで、時間を添字に畳む案が
    踏んでいた ``δ(t,t')`` の罠は、ここでは踏みようがない。
    """
    coeff = " ".join(x for x in (term_a.get("coefficient_latex", ""),
                                 term_b.get("coefficient_latex", "")) if x).strip()
    op = (operator_latex or "").strip()
    if op in ("", "1"):
        return coeff or "1"
    if not coeff:
        return op
    return f"{coeff} {op if not _is_sum(op) else _fenced(op)}"


def _fenced(latex: str) -> str:
    return f"\\left({latex}\\right)"


def _is_sum(latex: str) -> bool:
    """最上位に ``+`` / ``-`` があるか（括弧・波括弧の中は数えない）。

    等時刻代数が**和**を返すことは普通にある（フェルミオンの交換子は
    ``1 - 2c^†c``）。係数を前に置くだけだと ``e^{…}e^{…} 1 - 2c^†c`` となり、
    これは別の式として読める——公開物に出る前に括弧で囲う。
    """
    depth = 0
    for i, ch in enumerate(latex):
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif depth == 0 and ch in "+-" and i > 0:
            return True
    return False


def parse_closure(raw: str) -> "tuple[list[str], list[dict[int, str]]] | None":
    r"""``"c(1);c(2)|2:-1*w;1:-1*w"`` → 基底と ad の行列。

    Julia 側の歩きが返す形。``NOTCLOSED`` （上限まで閉じなかった）は None。
    行 *i* は ``ad_H(基底_i) = Σ_j 係数 × 基底_j`` の係数を列番号つきで並べたもの。
    """
    text = (raw or "").strip().strip('"')
    if not text or text.startswith(NOT_CLOSED) or "|" not in text:
        return None
    keys_part, rows_part = text.split("|", 1)
    keys = [k for k in keys_part.split(";") if k]
    if not keys:
        return None
    rows: "list[dict[int, str]]" = []
    for row in rows_part.split(";"):
        cells: "dict[int, str]" = {}
        for cell in row.split(","):
            if not cell.strip():
                continue
            idx, _, entry = cell.partition(":")
            try:
                j = int(idx)
            except ValueError:
                return None
            # 同じ基底に落ちる項が複数あることはある——足す。
            cells[j] = f"({cells[j]}) + ({entry})" if j in cells else entry
        rows.append(cells)
    if len(rows) != len(keys):
        return None
    return keys, rows


def sympy_name(param: str) -> str:
    """``eps(m)`` → ``eps_m``: QA の印字を SymPy の識別子へ。

    表示は SymPy 自身の latex に任せる（``eps_m`` は ``eps_{m}`` と出る）ので、
    ここが決めるのは**綴りが一意に戻せること**だけである。
    """
    def _sub(m: "re.Match") -> str:
        args = "_".join(a.strip() for a in m.group(3).split(",") if a.strip())
        marks = "_dag" * len(m.group(2))
        return f"{m.group(1)}{marks}" + (f"_{args}" if args else "")
    return _CALL_RE.sub(_sub, param).strip()


def matrix_entries(keys: "list[str]", rows: "list[dict[int, str]]") -> "list[str]":
    """行優先の n×n 成分（SymPy が読める綴り）。空きは ``0``。"""
    n = len(keys)
    out: "list[str]" = []
    for i in range(n):
        for j in range(1, n + 1):
            entry = rows[i].get(j)
            out.append(sympy_name(_tidy_products(entry)) if entry else "0")
    return out


def _tidy_products(entry: str) -> str:
    """``-1.0*w`` → ``-1*w``: QA の Float 係数を整数で書き戻す。"""
    head, sep, tail = entry.partition("*")
    return _tidy_number(head) + sep + tail


def parse_exponential_row(raw: str) -> "list[tuple[str, str]]":
    """``"cos(t*w)|\\cos{…}@@-I*sin(t*w)|- i \\sin{…}"`` → [(値, latex), …]。"""
    text = (raw or "").strip().strip('"')
    out: "list[tuple[str, str]]" = []
    for record in text.split("@@"):
        value, _, latex = record.partition("|")
        out.append((value.strip(), latex.strip()))
    return out


def terms_from_row(row: "list[tuple[str, str]]", keys: "list[str]",
                   operators: "list[str]") -> "list[dict]":
    """係数の行と基底から、対の項を組む。**係数 0 の項は落とす**。"""
    terms: "list[dict]" = []
    for (value, latex), key, operator in zip(row, keys, operators):
        if value in ("0", "0.0", ""):
            continue
        terms.append({
            "coefficient_latex": latex,
            "coefficient": value,
            "operator_latex": name_to_subscript_latex(key),
            "operator": operator,
            "operator_qa": key,
        })
    return terms


def qa_to_input(text: str) -> str:
    r"""QA の印字を**入力として読み直せる綴り**へ: ``c†(2) c(1)`` → ``c'(2) c(1)``。

    対は次の括弧の被演算子になるので、保存する演算子は入力の綴りでなければ
    ならない——QA の随伴の印 ``†`` を戻さずに保存していたとき
    ``{c(t), c†(t')}`` は **0** になった（実測 2026-08-22）。
    """
    return (text or "").replace("†", "'")


#: 演算子部が c 数 1 になった（＝演算子が消えた）ことを表す綴り。
_UNITY = ("1", "1.0", "")


def is_unity(latex: str) -> bool:
    r"""等時刻代数の答えが c 数 1 か。``{c,c^\dagger}`` はここに落ちる。"""
    return (latex or "").strip() in _UNITY


def is_vanishing(latex: str) -> bool:
    """答えが 0 か。0 の組み合わせは和に入れない。"""
    return (latex or "").strip() in ("0", "0.0")


def coefficient_product(a: str, b: str) -> str:
    """2つの係数の積（スカラー CAS が読める綴り）。片方が空なら他方。"""
    a, b = (a or "").strip(), (b or "").strip()
    if not a or a == "1":
        return b or "1"
    if not b or b == "1":
        return a
    return f"({a})*({b})"


def coefficient_sum(parts: "list[str]") -> str:
    """同じ演算子に落ちた係数の和。"""
    kept = [p for p in parts if p]
    if not kept:
        return "0"
    return kept[0] if len(kept) == 1 else " + ".join(f"({p})" for p in kept)


def pair_terms_of(descriptor: "dict | None", *, latex: str,
                  canonical: str) -> "list[dict]":
    """対の項（対でない式は「係数 1 の演算子」1項）。"""
    if descriptor and descriptor.get("terms"):
        return list(descriptor["terms"])
    return [{"coefficient_latex": "", "coefficient": "1",
             "operator": canonical, "operator_latex": latex}]


def closure_is_concrete(keys: "list[str]", entries: "list[str]") -> bool:
    r"""閉包が**具体的な演算子**で閉じたか（族ではないか）。

    実測 2026-08-22、一般の2体項 `\sum_{pqsn} V_{pqsn} c^†_pc^†_qc_sc_n` に対して
    歩きは「閉じた」と答えた——が、その基底は
    ``∑₁ δ(1q) c†(#₁) c(n) c(s)`` のように**和の印と自由添字を抱えたまま**で、
    文字列一致で1つの演算子と数えていただけだった。係数も `V(#₁1sn)` となり、
    SymPy の識別子に落とすと別々の行列要素が同じ記号に潰れる。

    答えとして出てしまえば、`c_1 + i V_{1qsn} t c^†_q c_n c_s - i V_{} t ∑₁ …`
    ——誰も書いていない量である。**推測しないで断る**ための判別材料がこれ。
    """
    return not any(m in text for text in list(keys) + list(entries)
                   for m in _BOUND_MARKERS)
