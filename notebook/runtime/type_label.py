r"""「この型の説明は何を宣言するか」を **1つの表**で答える。

宣言文の型ラベル（`A を <型の説明> とする` の `<型の説明>`）から
`define_*` / `declare_*` の操作を決める。

**なぜこのモジュールがあるか（実測 2026-09-05）。** 同じ問いに2つの表が
別々に答えていた:

| 表 | どこ | 何を知っていたか |
|---|---|---|
| A | 操作スキーマ自身の `aliases` | 宣言系13操作の ja 116語（研究者に見せている語彙） |
| B | `i18n/type_kw_*` ＋ `_infer_type_from_label` の優先順位 | 5語だけ |

静的に突き合わせると **64/116 が不一致**、実パイプラインでは **11綴りが
別の操作に届いて `confirmed` を返していた**——限定語が黙って落ちる形で:

| 打った文 | 届いていた | 正しくは | 落ちていたもの |
|---|---|---|---|
| `A を対称行列とする` | `define_matrix` | `declare_symmetry` | 対称性 |
| `A を反対称行列とする` | `define_matrix` | `declare_symmetry` | 反対称性 |
| `A をエルミート行列とする` | `define_matrix` | `declare_symmetry` | エルミート性 |
| `A をスカラー関数とする` | `define_scalar_parameter` | `define_scalar_field` | 場であること |
| `A をベクトル関数とする` | `define_vector` | `define_vector_field` | 場であること |
| `q を時間の関数として宣言` | `define_scalar_parameter` | `declare_time_dependent` | 時間依存 |

原因は表Bの中に2つあった。**(a) 部分文字列照合なので限定語が無視される**
（`対称`行列 → 行列、ベクトル`関数` → ベクトル）。**(b) 語彙が足りない**
（`スカラー関数`・`時間の関数` は表Aには在るのに表Bに無い）。2026-08-27 の
∇族の註「fragment に語を足しても dict が知らなければ読みは黙って降りる」と
同型で、あれは「①の一問」と明記されていた。

**照合はラベル**全体**の完全一致**である。部分文字列にすると `define` が
`define matrix` の中に出て何もかも `define_function` になる（試作で実測）。
完全一致なら `時間依存のスカラー場` はどの別名とも一致せず、既存の優先順位表
（場 > 時間依存 > テンソル > 行列 > ベクトル > スカラー）にそのまま落ちる
——**組み合わせラベルのための優先順位は構造なので残す**。

**バックエンドで絞る。** `define_fermion_operator` は julia_full だけなので、
Maxima セッションで `フェルミオン演算子` をそこへ送ってはいけない。
"""
from __future__ import annotations

import re
from functools import lru_cache

from notebook.runtime.nl_patterns import fragment, nlp

#: 宣言系の操作だけを索引に入れる（`simplify` の別名は型の説明ではない）。
_DECL_PREFIXES = ("define_", "declare_")

# 別名から**宣言の動詞句**を落として型の説明だけにする。語は i18n にあり
# （`ja_alias_decl_tail` / `en_alias_decl_head`）、**係り方だけがここに残る**
# ——日本語は後ろ（語尾）、英語は前（動詞）で、名前付きグループでは吸収できない。
# `ja_decl_suffix` の活用一覧（`ja_decl_infl`）はそのまま共有している。
# `〜として定義する` は2段（`として` + `定義する`）なので繰り返し当てる。
_JA_DECL_TAIL = re.compile(fragment("ja_alias_decl_tail"))
_EN_DECL_HEAD = re.compile(fragment("en_alias_decl_head"), re.I)

#: 動詞だけ・機能語だけになった語幹は索引に入れない。`define` は
#: `define_function` の別名だが、型の説明としては何も言っていない——入れると
#: 「A を定義する」が関数宣言になり、スカラー既定（`１電子エネルギー` などの
#: 物理ラベルのため）を壊す。
_BARE_STEM_RE = re.compile(fragment("bare_decl_verb"), re.I)


def _is_bare_declaration_verb(stem: str) -> bool:
    return bool(_BARE_STEM_RE.fullmatch(stem or ""))


def label_stem(text: str, lang: str) -> str:
    """別名／ラベルから宣言の動詞句を落とした「型の説明」を返す。"""
    s = (text or "").strip()
    if lang == "ja":
        prev = None
        while prev != s:
            prev = s
            s = _JA_DECL_TAIL.sub("", s).strip()
        return s
    s = _EN_DECL_HEAD.sub("", s).strip()
    s = re.sub(r"[_\s]+", " ", s).strip().lower()
    # 冠詞は型の名前の一部ではない。`decl_stmt[en]` は `a` だけを落とすので
    # `an antisymmetric function` の `an` が残り、別名と一致しなかった
    # （実測 2026-09-05）。
    return re.sub(r"^(?:an?|the)\s+", "", s)


@lru_cache(maxsize=1)
def _validator():
    """スキーマは静的なので1度だけ読む。**呼ぶたびに `OperationValidator()` を
    作ると JSON を読み直す**——この関数はラベル1件ごとに呼ばれる。"""
    from notebook.runtime.operation_validator import OperationValidator
    return OperationValidator()


@lru_cache(maxsize=1)
def _stem_index() -> "dict[str, dict[str, str]]":
    """`{"ja": {語幹: 操作}, "en": {...}}`。スキーマは静的なので1度だけ作る。"""
    validator = _validator()
    index: "dict[str, dict[str, str]]" = {"ja": {}, "en": {}}
    for op in validator.known_operations:
        if not op.startswith(_DECL_PREFIXES):
            continue
        spec = validator.get_spec(op) or {}
        for lang in ("ja", "en"):
            for alias in (spec.get("aliases") or {}).get(lang, []):
                stem = label_stem(alias, lang)
                if not stem or _is_bare_declaration_verb(stem):
                    continue
                index[lang].setdefault(stem, op)
    return index


def _named_declaration(label: str) -> "str | None":
    """型ラベルが名指している宣言操作（バックエンドを問わない）。"""
    index = _stem_index()
    for lang in ("ja", "en"):
        op = index[lang].get(label_stem(label, lang))
        if op is not None:
            return op
    return None


def declaration_operation_for_label(
    label: str, backend: "str | None" = None
) -> "str | None":
    """型ラベルが**名指している**宣言操作。無ければ None（呼び出し側が既存表へ）。

    *backend* を与えると、そのバックエンドが対応していない操作は返さない。
    そのとき**何を名指していたか**は `declaration_unsupported_here` が返す
    ——「そんな宣言は無い」と「このバックエンドでは動かせない」は別の答えで、
    後者を既定のスカラーに落とすと**「演算子」と書いた文がスカラーを登録して
    成功を返す**（実測 2026-09-06、Maxima セッションで7綴り）。
    """
    op = _named_declaration(label)
    if op is None:
        return None
    if backend is not None:
        spec = _validator().get_spec(op) or {}
        if backend not in (spec.get("supported_backends") or []):
            return None
    return op


def declaration_unsupported_here(
    label: str, backend: "str | None"
) -> "str | None":
    """このバックエンドが**動かせない**宣言を名指しているなら、その操作名。

    `define_fermion_operator` / `define_boson_operator` は julia_full だけで、
    Maxima セッションには演算子代数が無い。受け入れると `c'(k) c(k)` がただの積に
    なり、その後の計算が全部おかしくなるので、**登録の時点で理由を言う**
    （利用者の判断、2026-09-06）。スピンは `spin_algebra` が CAS-bypass で
    実装されているため Maxima でも受ける——その違いがそのまま出る。
    """
    if backend is None:
        return None
    op = _named_declaration(label)
    if op is None:
        return None
    spec = _validator().get_spec(op) or {}
    if backend in (spec.get("supported_backends") or []):
        return None
    return op


def symmetry_value_for_label(label: str) -> "str | None":
    """`declare_symmetry` の `symmetry` 欄を、研究者が書いた語から読む。

    値は CAS 側の語彙（`symmetric` / `antisymmetric` / `hermitian`）で、
    綴りは `i18n/nl_patterns.json` の `symmetry_kw`。

    **「`反対称` は `対称` を含むので順序が効く」と最初に書いたが、測ると
    偽だった**（2026-09-05）。並びを入れ替える変異はテストを1件も落とさず、
    日英7綴りで読みが完全に一致する——`反`/`anti` が識別接頭辞なので、`re` は
    走査位置ごとに選択肢を試し、短い側がその位置で当たることがない。長い順に
    並べてあるのは可読性のためで、規則ではない。
    """
    for lang in ("ja", "en"):
        m = nlp("symmetry_kw", lang).search(label or "")
        if not m:
            continue
        for value, hit in (m.groupdict() or {}).items():
            if hit:
                return value
    return None
