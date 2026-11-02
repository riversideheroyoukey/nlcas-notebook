r"""指示コーパスが**見ていない綴り**を数える —— CORPUS-WORKING-ONLY の測定器。

    .venv/bin/python scripts/nl_spelling_coverage.py                 # 全体の報告
    .venv/bin/python scripts/nl_spelling_coverage.py --keys rhs_op   # 変更した鍵だけ
    .venv/bin/python scripts/nl_spelling_coverage.py --asymmetry     # 日英の片側参照
    .venv/bin/python scripts/nl_spelling_coverage.py --pairs         # 片方だけ畳まれた問い

**なぜ在るか。** `tests/corpus/instruction_corpus.json` の523行は
`docs/scenarios/*.md` から収穫されている。シナリオは**通る指示**で書かれているので、
このコーパスは「動かない組み合わせ」を構造的に含まない。つまり
`scripts/measure_instruction_corpus.py` の差分が 0 でも、それは

  「安全である」ではなく「**その綴りはこのコーパスに1行も無い**」

かもしれない。実際そう読み違えたことが2問連続であった（四十一・四十二問目、
2026-09-07）。`右辺を` は18行・`簡約` は5行あるのに交差は **0行**で、
壊れていたのはまさにその交差だった。

**測り方。** 綴りは発明しない——**系が自分で「受ける」と言っている表**、
`i18n/nl_patterns.json` を読む。パターンを literal run に割り（`nl_worklist.py`
と同じ規則）、それが**入力として**どこかに書かれているかを見る:

  corpus     `tests/corpus/instruction_corpus.json` の `jp` / `en`
  scenarios  `docs/scenarios/*.md`
  tests      `tests/**/*.py` の**ソース**（＝リテラルとして打たれた文字列）

どこにも無い綴りは、**誰も一度も打っていない**。そこは差分検証が原理的に
無力な領域である。

**この道具が見ていないもの**（「安全」と言う道具は、何を見ていないか1つ挙げられ
なければまだ測っていない——[[feedback-verify-the-guard-catches-its-own-defect]] ⑤）:

1. **一致は部分文字列で、経路ではない。** ある綴りがコーパス行に現れても、
   その綴りを**その鍵が読んだ**とは限らない（別の規則が先に claim しうる）。
   つまりこれは上限であって、実際の測定量はもっと少ない。
2. **fragment から parametrize したテストは、リテラルを持たない**ので
   「打たれていない」と数える。これは意図的で、そういうテストは語を消すと
   行ごと消えるため、その語が測られている証拠にならない（同 memo ⑥）。
3. **逆に、テストに文字列として在るだけでは「測られた」とは言えない。**
   主張のないところに綴りを書けばこの数は下がる。数を下げるときは
   「打って、何になるべきか主張する」こと。
4. literal run は**パターンの alternation の枝そのものではない**。
   `クレブシュ・ゴルダン` は `・` で2つに割れる。族の単位で読むこと。
5. **1文字の綴りは数えない**（日本語は2文字から、英字は4文字から——短いものは
   正規表現の断片と区別が付かないため）。四十八問目の `十` はこれで漏れており、
   見つけたのは下の `neutral_fragment_asymmetry` のほうだった。

`nl_worklist.py` は import した時点で走ってしまう（実測 2026-09-10、
`sys.argv` を取り違えて落ちた）。ここは関数だけを置き、
`tests/test_spelling_space_coverage.py` から import される。
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
PATTERNS = ROOT / "i18n" / "nl_patterns.json"
CORPUS = ROOT / "tests" / "corpus" / "instruction_corpus.json"
SCENARIOS = ROOT / "docs" / "scenarios"
TESTS = ROOT / "tests"

#: 正規表現の**構文**。literal を取り出す前に消す。`nl_worklist.py` と同じ表で、
#: そこに書かれた3つの実測（グループの中身まで食べた／`_comment` を読んだ／
#: コーパスの日本語欄は `jp`）はここでも同じく効く。
_SYNTAX_RE = re.compile("|".join([
    r"\{\{[A-Za-z0-9_]+\}\}",
    r"\(\?P<[A-Za-z0-9_]+>",
    r"\(\?P=[A-Za-z0-9_]+\)",
    r"\(\?[:=!#i]",
    r"\(\?<[=!]",
    r"\\\\[A-Za-z]+",   # LaTeX コマンド。先に消さないと `\t` が食われ 'heta' が残る
    r"\\[A-Za-z]",      # \s \b \d \w ...
    r"\[[^\]]*\]",
    r"\{\d+(?:,\d*)?\}",
]))
_PUNCT_RE = re.compile(r"[()|?*+^$.\\/\-\s,:;=<>!&#%@~\"']")
_JA_RE = re.compile(r"[ぁ-んァ-ン一-龥]")
_TOK_RE = re.compile(r"[ぁ-んァ-ン一-龥ー]+|[A-Za-z]+")

#: 英字の literal をどの長さから数えるか。短い綴りは正規表現の断片と区別が
#: つかないので落とす（`nl_worklist.py` と同じ 4）。日本語は 2。
_MIN_EN, _MIN_JA = 4, 2


def literals(pattern: str) -> list:
    """パターンから「研究者が打ちうる語」を取り出す。"""
    out = []
    stripped = _PUNCT_RE.sub("\x00", _SYNTAX_RE.sub("\x00", pattern))
    for run in stripped.split("\x00"):
        for tok in _TOK_RE.findall(run):
            if _JA_RE.search(tok):
                if len(tok) >= _MIN_JA:
                    out.append(tok)
            elif len(tok) >= _MIN_EN:
                out.append(tok.lower())
    return out


def _pattern_strings(value) -> list:
    """正規表現の欄だけ集める（`_` で始まる欄は散文なので除く）。"""
    acc = []

    def walk(node):
        if isinstance(node, str):
            acc.append(node)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for key, item in node.items():
                if not key.startswith("_"):
                    walk(item)

    walk(value)
    return acc


def _load() -> dict:
    return json.loads(PATTERNS.read_text(encoding="utf-8"))


def spelling_space(data: "dict | None" = None) -> dict:
    """綴り → それを綴っている場所の集合（``key:名前`` / ``frag:名前``）。"""
    data = data if data is not None else _load()
    frags = data.get("_fragments", {})
    out = collections.defaultdict(set)
    for key, value in data.items():
        if key.startswith("_"):
            continue
        for pattern in _pattern_strings(value):
            for lit in literals(pattern):
                out[lit].add("key:" + key)
    for name, value in frags.items():
        if name.startswith("_"):
            continue
        for pattern in _pattern_strings(value):
            for lit in literals(pattern):
                out[lit].add("frag:" + name)
    return dict(out)


def _corpus_texts() -> list:
    rows = json.loads(CORPUS.read_text(encoding="utf-8"))["rows"]
    return [" \x01 ".join(t for t in (r.get("jp"), r.get("en")) if isinstance(t, str))
            for r in rows]


def _joined(paths) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in paths)


def sources() -> dict:
    """綴りを探す先。値は「そこに書かれた入力の文字列」。"""
    return {
        "corpus": "\n".join(_corpus_texts()),
        "scenarios": _joined(sorted(SCENARIOS.glob("*.md"))),
        "tests": _joined(sorted(TESTS.rglob("*.py"))),
    }


def _contains(haystack: str, haystack_low: str, lit: str) -> bool:
    return lit in (haystack_low if lit.isascii() else haystack)


def coverage(data: "dict | None" = None) -> dict:
    """綴り → {"corpus": bool, "scenarios": bool, "tests": bool}。"""
    space = spelling_space(data)
    src = sources()
    low = {k: v.lower() for k, v in src.items()}
    return {lit: {name: _contains(src[name], low[name], lit) for name in src}
            for lit in space}


def never_typed(data: "dict | None" = None) -> list:
    """どこにも入力として書かれていない綴り（ソート済み）。"""
    return sorted(lit for lit, where in coverage(data).items()
                  if not any(where.values()))


def corpus_rows_for_key(key: str, data: "dict | None" = None) -> dict:
    """その鍵の綴りが、指示コーパスの**何行**に現れるか。

    `measure_instruction_corpus.py` の差分が 0 だったとき、これを読む。
    0 行なら差分 0 は「安全」ではなく「測っていない」である。
    """
    data = data if data is not None else _load()
    frags = data.get("_fragments", {})
    if key not in data and key not in frags:
        raise KeyError(f"nl_patterns.json に {key!r} は無い")
    value = data.get(key, frags.get(key))
    # **ここでは fragment を展開する。** `spelling_space()` は展開しない——
    # そちらは「その綴りはどこに書かれているか」を1箇所に帰属させたいので、
    # fragment の語は fragment のものとして数える。ここが答えるのは別の問い
    # （「この鍵を変えたとき、コーパスは何行それを見ているか」）で、鍵が
    # 引いている語彙は**その鍵の綴りである**。展開しないと `order_clause` の
    # 綴りは `times` 1語しか出ず、道具のほうが嘘をつく（実測 2026-09-10）。
    lits = sorted({lit for p in _pattern_strings(value)
                   for lit in literals(_expand(p, frags))})
    texts = _corpus_texts()
    low = [t.lower() for t in texts]
    return {lit: sum(1 for t, tl in zip(texts, low)
                     if _contains(t, tl, lit)) for lit in lits}


# --- 日英の非対称 ------------------------------------------------------------
_REF_RE = re.compile(r"\{\{(\w+)\}\}")
_MAX_DEPTH = 8


def _expand(src: str, frags: dict) -> str:
    """``{{name}}`` を不動点まで展開する（読み取り専用。未知名はそのまま残す）。"""
    out = src
    for _ in range(_MAX_DEPTH):
        nxt = _REF_RE.sub(lambda m: frags.get(m.group(1), m.group(0)), out)
        if nxt == out:
            break
        out = nxt
    return out


def _refs(src: str, frags: dict, depth: int = 0) -> set:
    out = set()
    for name in _REF_RE.findall(src or ""):
        out.add(name)
        if depth < _MAX_DEPTH and name in frags:
            out |= _refs(frags[name], frags, depth + 1)
    return out


def is_language_neutral(name: str) -> bool:
    """その fragment は「言語によらない問い」に答えているか。

    名前が `_ja` / `_en` で終わる（または `ja_` / `en_` で始まる）ものは
    **その言語の綴り**で、片側にしか無くて当然。それ以外——`ordinal_token`
    （数はどう書かれるか）、`symbol`（記号名はどう綴られるか）——は
    **両方の言語が同じ表に訊くべき**である。
    """
    return not (name.endswith(("_ja", "_en")) or name.startswith(("ja_", "en_")))


def neutral_fragment_asymmetry(data: "dict | None" = None) -> list:
    """日英そろった鍵のうち、**言語中立な fragment を片側しか引いていない**もの。

    ここが 2026-09-10 の四十八問目を見つけた測定である。`order_clause` は
    日本語だけが `{{ordinal_token}}` を引き、英語は `twice|once` と `\\d+` を
    自前で綴っていたので、同じ表に載っている `thrice` と「語の数＋times」が
    **どこにも届かず 1階を completed で返して**いた。

    非対称そのものは欠陥ではない（†は生成側だけ、`neighbour` は英語だけ）。
    残っているものには理由が要る、というのがこの表の契約である。
    """
    data = data if data is not None else _load()
    frags = data.get("_fragments", {})
    rows = []
    for key, value in data.items():
        if key.startswith("_") or not isinstance(value, dict):
            continue
        ja, en = value.get("ja"), value.get("en")
        if not (isinstance(ja, str) and isinstance(en, str)):
            continue
        r_ja = {n for n in _refs(ja, frags) if is_language_neutral(n)}
        r_en = {n for n in _refs(en, frags) if is_language_neutral(n)}
        if r_ja - r_en or r_en - r_ja:
            rows.append((key, sorted(r_ja - r_en), sorted(r_en - r_ja)))
    return rows


def _report_space() -> None:
    cov = coverage()
    total = len(cov)
    per = {name: sum(1 for w in cov.values() if w[name])
           for name in ("corpus", "scenarios", "tests")}
    gaps = sorted(lit for lit, w in cov.items() if not any(w.values()))
    print(f"綴り空間（nl_patterns.json の literal run）: {total}")
    if not total:
        print("  ← 0 件。表が読めていない（道具の側の故障を先に疑う）")
        return
    for name, n in per.items():
        print(f"  {name:10s} に在る: {n:4d}  ({n * 100 // total}%)")
    print(f"  どこにも無い        : {len(gaps):4d}  ({len(gaps) * 100 // total}%)")
    if gaps:
        space = spelling_space()
        print("\n一度も打たれていない綴り:")
        for lit in gaps:
            print(f"  {lit:22s} {sorted(space[lit])}")



# ---------------------------------------------------------------------------
# 三つ目の読み: **片方の言語にだけ在る fragment**（2026-09-12 に scratch で使い、
# 宣言の繋辞・代入の動詞・解く文の語順の3問を選んだ。2026-09-13 に production へ）
# ---------------------------------------------------------------------------
def _language_of(name: str) -> "tuple[str | None, str | None]":
    """fragment 名から (語幹, 言語)。印は**接頭辞にも接尾辞にも**付く。"""
    for pre in ("ja_", "en_"):
        if name.startswith(pre):
            return name[3:], pre[:2]
    for suf in ("_ja", "_en"):
        if name.endswith(suf):
            return name[:-3], suf[1:]
    return None, None


def _key_readers(name: str, data: dict) -> frozenset:
    """その fragment を読んでいる**鍵**（fragment 経由の読者は別の層なので除く）。"""
    token = "{{" + name + "}}"
    return frozenset(
        key for key, value in data.items()
        if isinstance(value, dict)
        for lang in ("ja", "en")
        if isinstance(value.get(lang), str) and token in value[lang])


def language_pair_report(data: "dict | None" = None) -> "tuple[list, list]":
    r"""(名前の違う鏡, 片側にしか無いもの) を返す。

    **片方の言語にだけ fragment が在るということは、その問いが片方では畳まれ、
    もう片方では鍵ごとに答えられている印である**——2026-09-12 に宣言の繋辞
    （英語10鍵が自前で綴り、読めない綴りは総称に拾われて `define_scalar_parameter`
    を confirmed で返していた）を見つけたのがこの読みである。

    **盲点があった: 語幹の綴りが日英で違う鏡を、片側扱いしていた。**
    `ja_op_var_particle` と `en_op_var_head_wide` は同じ問い（操作の変数はどの語で
    導かれるか）に答え、**同じ3鍵が読んでいる**。判別は名前ではなく
    **読む鍵の集合が完全に一致するか**で、校正すると既知の鏡4組を拾い、
    2↔1 の対応（`ja_decl_suffix` ＋ `ja_decl_suffix_named` ↔ `en_decl_copula`）は
    正しく鏡と主張しない——そこは理由を書く側の仕事である。
    """
    data = data if data is not None else _load()
    frags = data.get("_fragments", {})
    by_stem: dict = {}
    for name in frags:
        stem, lang = _language_of(name)
        if stem:
            by_stem.setdefault(stem, {})[lang] = name
    lone = {"ja": [], "en": []}
    for stem, per_lang in by_stem.items():
        if len(per_lang) == 1:
            lang, name = next(iter(per_lang.items()))
            lone[lang].append(name)

    mirrors, paired = [], set()
    for ja_name in lone["ja"]:
        ja_readers = _key_readers(ja_name, data)
        if not ja_readers:
            continue
        for en_name in lone["en"]:
            if ja_readers == _key_readers(en_name, data):
                mirrors.append((ja_name, en_name, sorted(ja_readers)))
                paired.update((ja_name, en_name))
    rest = [(n, lg, len(_key_readers(n, data)))
            for lg in ("ja", "en") for n in lone[lg] if n not in paired]
    rest.sort(key=lambda r: -r[2])
    return mirrors, rest


def _report_pairs() -> None:
    mirrors, rest = language_pair_report()
    print(f"名前の違う鏡（読む鍵が完全一致）: {len(mirrors)} 組")
    for ja_name, en_name, keys in mirrors:
        print(f"  {ja_name:26} ↔ {en_name:26} {keys}")
    print(f"\n片側にしか無い fragment: {len(rest)}（読む鍵の多い順）")
    for name, lang, n in rest[:15]:
        print(f"  {n:3}  {name:26} ({lang})")


def _report_keys(keys) -> None:
    for key in keys:
        counts = corpus_rows_for_key(key)
        total = sum(1 for v in counts.values() if v)
        print(f"\n{key}: 綴り {len(counts)} 件のうち、コーパスに在るのは {total} 件")
        for lit, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            mark = "  " if n else "← 0行（差分は何も言っていない）"
            print(f"  {n:4d}  {lit:22s} {mark}")


def _report_asymmetry() -> None:
    rows = neutral_fragment_asymmetry()
    print(f"言語中立な fragment を片側しか引いていない鍵: {len(rows)}")
    for key, only_ja, only_en in rows:
        print(f"  {key}")
        if only_ja:
            print(f"      ja のみ: {only_ja}")
        if only_en:
            print(f"      en のみ: {only_en}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keys", nargs="+", help="この鍵の綴りがコーパスに何行あるか")
    ap.add_argument("--asymmetry", action="store_true",
                    help="日英で片側しか引いていない言語中立 fragment")
    ap.add_argument("--pairs", action="store_true",
                    help="片方の言語にだけ在る fragment（＝片方だけ畳まれた問い）")
    args = ap.parse_args(argv)
    if args.keys:
        _report_keys(args.keys)
    elif args.asymmetry:
        _report_asymmetry()
    elif args.pairs:
        _report_pairs()
    else:
        _report_space()
        print()
        _report_asymmetry()
        print()
        _report_pairs()


if __name__ == "__main__":
    main()
