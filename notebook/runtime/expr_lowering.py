"""利用者書式 → CAS コード の唯一の導出口。

docs/developer_manuals/32_user_cas_boundary_repair_plan.md の Phase 0。

システムには本来3つの別物がある:

===================  =========================================  ==================
表現                 例                                          誰のものか
===================  =========================================  ==================
利用者表現            ``V_i``, ``omega_0``, ``k_1``               利用者
CAS コード           ``V(i)``, ``omega_0``, ``k_1``              バックエンド
CAS ハンドル          ``_nb_E3``, CAS が返した文字列               バックエンド
===================  =========================================  ==================

``Expression.canonical`` は歴史的に「利用者表現」と「CAS ハンドル」を兼ねてきた
一方で、**CAS コードだけが一級の存在になっていなかった** — 各所で場当たり的に
parse → resolve → generate を再実行するか、あるいは丸ごと素通しされていた。
素通しされた側と lowering された側が同じコマンドで衝突すると、CAS は
「一致しなかった」とは言わないので **黙って誤答が返る**
(open_items AUDIT-SUBST-INDEXED)。

このモジュールはその導出を1箇所に集める。

**CAS 交換可能性 (Core Principle #10 / 計画の不変条件 R7)**:
このモジュールは CAS 固有の文字列・トークン・分岐を**1つも持たない**。
不透明ハンドルの判定もコード生成も、``BackendHandler`` サブクラス側の
フック (:meth:`~BackendHandler.is_opaque_handle` /
:meth:`~BackendHandler.generate_backend_code`) が決める。第3の CAS は
ハンドラのサブクラスを1つ足すだけで載る。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from notebook.runtime.backend_handler import BackendHandler

#: ``lowered``     — IR を経由して CAS コードを生成した。
#: ``passthrough`` — 変換せずそのまま使う (CAS ハンドル・空文字・式でないもの)。
#: ``refused``     — 安全に変換できない。**呼び出し側は操作を失敗させること**
#:                   (計画の不変条件 R4「沈黙禁止」)。決して成功として通さない。
LoweringStatus = Literal["lowered", "passthrough", "refused"]


@dataclass(frozen=True)
class Lowered:
    """lowering の結果。``reason`` は開発者向けで i18n 対象外。"""

    code: str
    status: LoweringStatus
    reason: str = ""

    @property
    def ok(self) -> bool:
        """``refused`` でなければ *code* をそのまま CAS に渡してよい。"""
        return self.status != "refused"


class LoweringRefused(ValueError):
    """An expression could not be safely lowered to backend code.

    Carries the expression id and the developer-facing reason.  Raised at the
    ``expr_lookup`` boundary so the caller turns it into a failed operation:
    handing the raw text to the CAS instead is exactly the silent path this
    module exists to close (invariant R4).
    """

    def __init__(self, expr_id: str, reason: str) -> None:
        super().__init__(f"{expr_id}: {reason}")
        self.expr_id = expr_id
        self.reason = reason


#: Keys in ``expr_lookup`` that are NOT an expression's canonical text and must
#: never be lowered: ``__input__`` is the user-facing form Julia reconstructs IR
#: from, ``__sympy__`` is a SymPy-evaluatable string, and ``__user__`` is the
#: pre-lowering copy this module itself writes.
_RESERVED_PREFIXES = ("__input__", "__sympy__", "__user__")

_E_ID_RE = re.compile(r"\bE\d+\b")


def referenced_expr_ids(args: dict) -> "set[str]":
    """E-ids mentioned anywhere in *args* (strings, lists, dict values).

    A superset is fine and cheap: lowering an id the operation turns out not to
    use costs one parse and changes nothing.  Missing one would put a raw string
    back on the wire, which is the bug.
    """
    found: set[str] = set()

    def _scan(v) -> None:
        if isinstance(v, str):
            found.update(_E_ID_RE.findall(v))
        elif isinstance(v, dict):
            for item in v.values():
                _scan(item)
        elif isinstance(v, (list, tuple, set)):
            for item in v:
                _scan(item)

    _scan(args)
    return found


def lower_expr_lookup(handler: "BackendHandler", args: dict,
                      expr_lookup: dict) -> dict:
    """Return *expr_lookup* with the expressions *args* refers to lowered to
    backend code, keeping the pre-lowering text under ``__user__<eid>``.

    This is the boundary that makes an operation's SOURCE and its ARGUMENTS
    speak the same language.  Before it existed, ``build_command``'s per-argument
    loop lowered the arguments while several builders took the expression body
    verbatim from here — and where the two met in one command (``subst``) the
    mismatch was silent (open_items AUDIT-SUBST-INDEXED).

    **Call this only where a lowered argument meets the body inside ONE
    command.**  Today that is ``MaximaBackendHandler._build_substitute``.
    Applying it to every operation was tried and measured wrong (2026-07-30):
    the builders dispatched before the per-argument loop pair the source with
    text that is NOT lowered — canonical pair names, variable names, their own
    ``subst`` calls — so lowering one side breaks them (a Poisson bracket
    returned 0; ``convolve`` hung the CAS).  They are self-consistent on raw
    text; making them uniform means lowering their other side too, per builder.
    See docs/developer_manuals/32 §Phase 3.

    Raises :class:`LoweringRefused` rather than passing raw text through.
    """
    out = dict(expr_lookup)
    for eid in referenced_expr_ids(args):
        text = expr_lookup.get(eid)
        if not isinstance(text, str) or eid.startswith(_RESERVED_PREFIXES):
            continue
        r = to_backend_code(handler, text)
        if r.status == "refused":
            raise LoweringRefused(eid, r.reason)
        if r.code != text:
            out[f"__user__{eid}"] = text
            out[eid] = r.code
    return out


def to_backend_code(handler: "BackendHandler", text: str) -> Lowered:
    """*text* をハンドラのバックエンドが解釈できるコードに落とす。

    記号の解決規則は宣言状態に依存する (``k_B`` は未宣言なら ``k(B)``、
    宣言済みなら ``k_B``) ため、結果は**保存してはならない** — 呼び出しの
    たびにハンドラの現在の ``SymbolContext`` のもとで導出し直すこと
    (計画の不変条件 R0)。
    """
    from notebook.runtime.shared_parser import UnknownNotationError

    if not text or not text.strip():
        return Lowered(text, "passthrough", "empty")

    if handler.is_opaque_handle(text):
        return Lowered(text, "passthrough", "backend handle")

    try:
        ir = handler.parse_ir(text)
    except UnknownNotationError:
        # 未知記法はエンジン境界が言語を知って翻訳する既存の契約。
        # ここで飲み込むと利用者向けメッセージが失われるので素通しする。
        raise
    except Exception as exc:
        # 安全弁 (R4): parse / resolve のどの失敗も、黙って生文字列を
        # CAS に流す理由にはならない。理由付きで refused にし、呼び出し側に
        # 操作を失敗させる。ここを狭い except にすると、想定外の例外だけが
        # 「黙って素通し」に戻ってしまう。
        return Lowered(text, "refused", f"{type(exc).__name__}: {exc}")

    if ir is None:
        return Lowered(text, "passthrough", "not a parseable expression")

    code = handler.generate_backend_code(ir)
    if code is None:
        return Lowered(text, "passthrough", "backend has no code generator")
    return Lowered(code, "lowered")
