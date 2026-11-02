r"""時間順序演算子の読み取り（GF-NOTATION、2026-08-20）。

``\langle T[\psi(x,t)\psi^\dagger(y,u)]\rangle`` の ``T`` は演算子だが、実測では
こうなっていた:

    \langle T \psi(x,t) \psi^\dagger(y,u) \rangle  →  T が**ただの因子**
    T[\psi(x,t) \psi^\dagger(y,u)]                 →  生の pydantic ValidationError
    \mathcal{T}                                    →  `mathcalT` という名前に化ける
    T_\tau（松原）                                  →  `T(tau)`＝τ で添字付けした T

**素の ``T`` は読まない。** このコーパス自身が同じ文字を2つの意味で使っている——
``\int d^3x\,\psi^*(x) T(x) \psi(x)`` の ``T`` は1体の運動エネルギー演算子であり、
``H = T + V`` の ``T`` も運動エネルギーである。文面だけでは決まらないので、推測すれば
ハミルトニアンのほうを壊す。素の ``T`` は因子のまま残し、⟨…⟩ の中に立っている場合だけ
**警告**する（断らず・黙らず——`bound_name_collision` と同じ立場）。

**曖昧でない綴りだけ読む:**

- ``\mathcal{T}`` / ``\mathscr{T}`` —— 花文字は時間順序専用（だからそう書き分ける）
- ``T_\tau`` —— 松原の虚時間順序。添字 τ がそう言っている
- ``T[…]`` / ``T\{…\}`` —— ただし ``T[i,j]``（行列要素、既に正しく通っている）と
  競合する。**判別材料は中身**で、添字の並びには関数適用が現れない。中身に適用形が
  あれば順序、なければ従来どおり添字

**落とし先は ``time_order(X)``**（松原は ``time_order(X, tau)``）。⟨…⟩ を
``expectation(...)`` に落としたのと同じ形で、CAS で使える名前付き関数として合成できる。
θ 関数への展開は、この対象に対する**操作**として後から載る——読みの側では決めない。
"""
from __future__ import annotations

import re

from notebook.runtime.latex_typography import braced_group

#: 順序演算子の綴り。花文字は単独で曖昧でなく、`T` は括弧か τ 添字を伴うときだけ。
_SCRIPT_T = re.compile(r"\\(?:mathcal|mathscr)\s*\{\s*T\s*\}")
#: 松原の虚時間順序 ``T_\tau`` の添字。**τ だけ**である。任意の識別子を許すと、
#: 添字は「順序である」と言わなくなる —— 実測 2026-08-21、1体の運動エネルギー
#: ``\hat T(\mathbf r)`` は装飾を畳んだ段階で ``T_hat(...)`` になっており、この
#: 規則が ``_hat`` を τ 添字として読んで
#: ``time_order((vec(r))*psi_hat_sigma(vec(r)), hat)`` を返していた。ハミルトニアン
#: の1体項が丸ごと時間順序演算子になり、**公開済みの実例にそのまま載っていた**
#: （scenario_greens_function_from_hamiltonian の E4）。素の ``T`` を読まないと
#: 決めた理由（この文字は運動エネルギーでもある）は、``T_x`` にもそのまま効く。
_TAU_SUB = re.compile(r"_\s*(?:\{\s*(\\?tau)\s*\}|(\\?tau)(?![A-Za-z0-9]))")

#: 演算対象がここで終わる（項の切れ目）。
_TERM_END = re.compile(r"\\rangle|[)\]}+,=]|(?<![A-Za-z0-9_^])-")


def _read_group(s: str, i: int) -> "tuple[str, int] | None":
    """``[...]`` / ``\\{...\\}`` / ``(...)`` を1つ読む。"""
    if i < len(s) and s[i] == "[":
        depth, k = 0, i
        while k < len(s):
            if s[k] == "[":
                depth += 1
            elif s[k] == "]":
                depth -= 1
                if depth == 0:
                    return s[i + 1:k], k + 1
            k += 1
        return None
    m = re.compile(r"\\\{").match(s, i)
    if m:
        depth, k = 1, m.end()
        while k < len(s):
            if re.compile(r"\\\{").match(s, k):
                depth += 1
            elif re.compile(r"\\\}").match(s, k):
                depth -= 1
                if depth == 0:
                    return s[m.end():k], k + 2
            k += 1
        return None
    g = braced_group(s, i)
    return (g[0], g[1]) if g is not None else None


def _read_term(s: str, i: int) -> "tuple[str, int]":
    """前置形の演算対象——項の終わりまで（``\\rangle``・閉じ括弧・``+``/``-`` の手前）。"""
    depth, k = 0, i
    while k < len(s):
        ch = s[k]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and _TERM_END.match(s, k):
            break
        k += 1
    return s[i:k].strip(), k


def _has_application(text: str) -> bool:
    """中身に関数適用があるか。添字の並び（``i,j``）には現れない。"""
    return re.search(r"[A-Za-z0-9_}\]]\s*(?:\\left\s*)?\(", text) is not None


def rewrite(s: str) -> str:
    r"""順序演算子を ``time_order(...)`` に畳む。読めない綴りはそのまま返す。"""
    out: list[str] = []
    i = 0
    while i < len(s):
        script = _SCRIPT_T.match(s, i)
        plain_t = None
        if not script:
            pm = re.compile(r"(?<![A-Za-z0-9_])T").match(s, i)
            plain_t = pm if pm else None
        if not script and not plain_t:
            out.append(s[i])
            i += 1
            continue
        j = (script or plain_t).end()
        var = None
        tm = _TAU_SUB.match(s, j)
        if tm:
            var = tm.group(1) or tm.group(2)
            j = tm.end()
        grouped = _read_group(s, j)
        # 素の `T` は、括弧を伴うか τ 添字が付くときだけ順序として読む。
        if plain_t and var is None and grouped is None:
            out.append(s[i:plain_t.end()])
            i = plain_t.end()
            continue
        if grouped is not None:
            body, nxt = grouped
            if plain_t and var is None and not _has_application(body):
                # `T[i,j]` —— 行列要素。従来の読みを変えない。
                out.append(s[i:plain_t.end()])
                i = plain_t.end()
                continue
        else:
            body, nxt = _read_term(s, j)
            if not body:
                out.append(s[i:j])
                i = j
                continue
        var_name = var.lstrip("\\") if var else None
        out.append(f"time_order({body}" + (f", {var_name})" if var_name else ")"))
        i = nxt
    return "".join(out)


#: 素の ``T`` が ⟨…⟩ の中で因子として読まれたときの警告キー。
BARE_ORDERING_WARNING = "warning.bare_time_ordering"


def find_bare_ordering(ir) -> "list[tuple[str, str]]":
    r"""``\langle T \psi \psi^\dagger \rangle`` の素の ``T`` を見つける。

    断らない——``T`` は運動エネルギーでもあり、このコーパス自身が両方の意味で
    使っている。黙りもしない——因子として読んだ事実は、書いた本人にしか正誤が
    判断できない。事実だけ伝えて研究者に残す（`bound_name_collision` と同じ立場）。
    """
    from notebook.models.ir import FunctionCall, Mul, NonCommutativeMul, Symbol
    found: list[tuple[str, str]] = []

    def walk(node):
        if isinstance(node, FunctionCall) and isinstance(node.func, Symbol) \
                and node.func.name == "expectation" and node.args:
            body = node.args[0]
            if isinstance(body, (Mul, NonCommutativeMul)):
                for a in body.args:
                    if isinstance(a, Symbol) and a.name == "T":
                        if not any(n == "T" for n, _ in found):
                            found.append(("T", BARE_ORDERING_WARNING))
        for f in getattr(type(node), "model_fields", {}):
            v = getattr(node, f, None)
            if getattr(type(v), "model_fields", None) is not None:
                walk(v)
            elif isinstance(v, list):
                for it in v:
                    if getattr(type(it), "model_fields", None) is not None:
                        walk(it)

    if ir is not None:
        walk(ir)
    return found
